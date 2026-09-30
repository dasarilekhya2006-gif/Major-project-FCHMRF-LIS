"""Command-line runner for Phase 5: fcHMRF-LIS spatial statistical modeling."""

import argparse
from pathlib import Path
import sys
import yaml
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.logging import setup_logging, get_logger
from src.core.seed import set_seed
from src.data.loaders import load_nifti
from src.data.sample_data import create_sample_3d_statistical_nifti
from src.fchmrf_lis.parameters import FCHMRFParameters, MethodMode
from src.fchmrf_lis.fchmrf_model import FCHMRFModel
from src.fchmrf_lis.visualization import plot_fchmrf_summary

logger = get_logger("run_fchmrf_lis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fcHMRF-LIS spatial False Discovery Rate control on neuroimaging statistical maps."
    )
    parser.add_argument(
        "--z-scores",
        type=str,
        default=None,
        help="Path to 3D NIfTI file containing voxel-level z-statistics.",
    )
    parser.add_argument(
        "--roi-mask",
        type=str,
        default=None,
        help="Path to 3D binary ROI mask (restricting analysis domain).",
    )
    parser.add_argument(
        "--dl-prior",
        type=str,
        default=None,
        help="Path to 3D continuous deep-learning prior probability map in [0, 1].",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/fchmrf_config.yaml",
        help="Path to fcHMRF YAML configuration.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["original_fchmrf", "adaptive_dl_fchmrf"],
        help="Methodological mode distinction.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Nominal spatial FDR target level (e.g. 0.05).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Spatial coupling parameter (Ising inverse temperature).",
    )
    parser.add_argument(
        "--lambda-dl",
        type=float,
        default=None,
        help="Weighting hyperparameter for DL prior in adaptive mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Destination directory for output NIfTI files.",
    )
    parser.add_argument(
        "--fig-dir",
        type=str,
        default=None,
        help="Destination directory for summary figures.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Filename prefix for saved outputs.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging(log_level="INFO")
    args = parse_args()

    # Load configuration
    cfg_p = Path(args.config)
    if cfg_p.exists():
        with open(cfg_p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        logger.warning(f"Config file {cfg_p} not found. Using defaults.")
        cfg = {}

    fchmrf_cfg = cfg.get("fchmrf_lis", {})
    input_cfg = cfg.get("inputs", {})
    out_cfg = cfg.get("outputs", {})

    # Determine parameters
    mode_str = args.mode or fchmrf_cfg.get("method_mode", "original_fchmrf")
    method_mode = MethodMode(mode_str)
    beta = float(args.beta if args.beta is not None else fchmrf_cfg.get("beta", 0.8))
    sigma_spatial = float(fchmrf_cfg.get("sigma_spatial", 2.5))
    sigma_intensity = float(fchmrf_cfg.get("sigma_intensity", 1.5))
    w_spatial = float(fchmrf_cfg.get("w_spatial", 1.0))
    w_pure_spatial = float(fchmrf_cfg.get("w_pure_spatial", 0.2))
    sigma_pure_spatial = float(fchmrf_cfg.get("sigma_pure_spatial", 4.0))
    lambda_dl = float(args.lambda_dl if args.lambda_dl is not None else fchmrf_cfg.get("lambda_dl", 0.5))
    alpha = float(args.alpha if args.alpha is not None else fchmrf_cfg.get("alpha", 0.05))
    max_iterations = int(fchmrf_cfg.get("max_iterations", 40))
    tolerance = float(fchmrf_cfg.get("tolerance", 1e-4))
    damping = float(fchmrf_cfg.get("damping", 0.5))
    fix_null = bool(fchmrf_cfg.get("fix_null", True))

    params = FCHMRFParameters(
        method_mode=method_mode,
        beta=beta,
        sigma_spatial=sigma_spatial,
        sigma_intensity=sigma_intensity,
        w_spatial=w_spatial,
        w_pure_spatial=w_pure_spatial,
        sigma_spatial_pure=sigma_pure_spatial,
        lambda_dl=lambda_dl,
        alpha=alpha,
        max_iterations=max_iterations,
        tolerance=tolerance,
        damping=damping,
        fix_null=fix_null,
    )

    output_dir = Path(args.output_dir or out_cfg.get("nifti_dir", "outputs/fchmrf_results"))
    fig_dir = Path(args.fig_dir or out_cfg.get("figures_dir", "outputs/figures"))
    prefix = args.prefix or out_cfg.get("prefix", "fchmrf_lis")

    # Resolve input paths
    z_path_str = args.z_scores or input_cfg.get("z_score_path")
    roi_path_str = args.roi_mask or input_cfg.get("roi_mask_path")
    dl_path_str = args.dl_prior or input_cfg.get("dl_prior_path")

    z_path = Path(z_path_str) if z_path_str and Path(z_path_str).exists() else None
    roi_path = Path(roi_path_str) if roi_path_str and Path(roi_path_str).exists() else None
    dl_path = Path(dl_path_str) if dl_path_str and Path(dl_path_str).exists() else None

    # Synthesize data if none exists
    if z_path is None:
        logger.info("No valid input z-score NIfTI found. Generating synthetic statistical volume...")
        set_seed(42)
        demo_z_path = Path("data/raw/synthetic_statistical_demo.nii.gz")
        create_sample_3d_statistical_nifti(
            demo_z_path,
            shape=(32, 32, 16),
            snr=2.8,
            seed=42,
        )
        z_path = demo_z_path

    # Load volumes
    z_vol = load_nifti(z_path)
    z_data = z_vol.data
    affine = z_vol.affine

    roi_mask_data = None
    if roi_path and roi_path.exists():
        roi_vol = load_nifti(roi_path)
        roi_mask_data = roi_vol.data > 0.5
        logger.info(f"Loaded ROI mask: {np.sum(roi_mask_data):,} active voxels.")

    dl_prior_data = None
    if dl_path and dl_path.exists():
        dl_vol = load_nifti(dl_path)
        dl_prior_data = np.clip(dl_vol.data, 0.0, 1.0)
        logger.info("Loaded 3D U-Net prior probability map.")

    # Initialize model
    model = FCHMRFModel(params=params)

    # Execute inference
    logger.info(
        f"Executing Phase 5 fcHMRF-LIS: mode={params.method_mode.value}, "
        f"beta={params.beta}, alpha={params.alpha}"
    )
    result = model.fit_predict(
        z_scores=z_data,
        dl_prior=dl_prior_data,
        spatial_mask=roi_mask_data,
        affine=affine,
    )

    # Save results
    saved_files = result.save_nifti_results(output_dir=output_dir, prefix=prefix)

    # Plot summary visualization
    fig_path = fig_dir / f"{prefix}_summary.png"
    plot_fchmrf_summary(
        result=result,
        z_scores=z_data,
        output_path=fig_path,
        dpi=out_cfg.get("dpi", 300),
    )

    logger.info("=" * 60)
    logger.info("Phase 5 fcHMRF-LIS Spatial Statistical Modeling Summary:")
    logger.info(f"  Method Mode:         {result.method_mode}")
    logger.info(f"  Domain Analyzed:     {result.num_analyzed_voxels:,} voxels")
    logger.info(f"  Discoveries:         {result.num_discoveries:,} voxels ({100.0 * result.num_discoveries / max(1, result.num_analyzed_voxels):.2f}%)")
    logger.info(f"  LIS Cutoff Thresh:   {result.fdr_results['lis_threshold']:.4f}")
    logger.info(f"  Estimated FDR:       {result.fdr_results['estimated_fdr']:.4f} (target alpha = {result.fdr_results['nominal_alpha']})")
    logger.info(f"  Mean-Field Steps:    {result.model_parameters['mean_field_iterations']} (final delta = {result.model_parameters['mean_field_final_delta']:.2e})")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
