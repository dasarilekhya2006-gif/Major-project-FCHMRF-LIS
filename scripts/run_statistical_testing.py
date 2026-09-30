"""Command-line interface to execute voxel-level statistical testing for neuroimaging data."""

import argparse
from pathlib import Path
import sys
import yaml
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.logging import setup_logging, get_logger
from src.core.seed import set_seed
from src.data.sample_data import create_sample_3d_statistical_nifti
from src.statistical_analysis.voxel_testing import compute_statistics
from src.statistical_analysis.visualization import plot_statistical_maps
from src.statistical_analysis.design import (
    create_one_sample_design,
    create_two_sample_design,
    create_paired_design,
    create_glm_design,
)

logger = get_logger("run_statistical_testing")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run voxel-level statistical testing on preprocessed neuroimaging volumes."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to 4D NIfTI volume (samples along axis 3) or multiple subjects.",
    )
    parser.add_argument(
        "--roi-mask",
        type=str,
        default=None,
        help="Path to 3D binary ROI mask (e.g. from Phase 3 3D U-Net).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/statistical_config.yaml",
        help="Path to statistical configuration YAML.",
    )
    parser.add_argument(
        "--test-type",
        type=str,
        default=None,
        choices=["one_sample_t", "two_sample_t", "paired_t", "glm", "direct_z"],
        help="Statistical test procedure override.",
    )
    parser.add_argument(
        "--alternative",
        type=str,
        default=None,
        choices=["greater", "two_sided", "less"],
        help="Alternative hypothesis directionality override.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Destination directory for output NIfTI maps.",
    )
    parser.add_argument(
        "--fig-dir",
        type=str,
        default=None,
        help="Destination directory for diagnostic figures.",
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

    stat_cfg = cfg.get("statistical_analysis", {})
    roi_cfg = cfg.get("roi_restriction", {})
    out_cfg = cfg.get("outputs", {})

    test_type = args.test_type or stat_cfg.get("test_type", "one_sample_t")
    alternative = args.alternative or stat_cfg.get("alternative", "greater")
    popmean = stat_cfg.get("popmean", 0.0)
    equal_var = stat_cfg.get("equal_var", False)
    nan_policy = stat_cfg.get("nan_policy", "mask_out")

    output_dir = Path(args.output_dir or out_cfg.get("nifti_dir", "outputs/statistical_maps"))
    fig_dir = Path(args.fig_dir or out_cfg.get("figures_dir", "outputs/figures"))
    prefix = args.prefix or out_cfg.get("prefix", "voxel_stat")

    # Determine ROI mask
    roi_path_str = args.roi_mask or (roi_cfg.get("mask_path") if roi_cfg.get("enabled", True) else None)
    roi_path = Path(roi_path_str) if roi_path_str and Path(roi_path_str).exists() else None

    # Handle input data
    input_path = Path(args.input) if args.input else None
    if input_path is None or not input_path.exists():
        logger.info("No input volume provided or found. Synthesizing multi-subject 4D group data for demonstration...")
        set_seed(42)
        shape_3d = (32, 32, 16)
        n_subjects = 12
        raw_vols = []
        for s in range(n_subjects):
            s_path = Path(f"data/raw/synthetic_group_sub_{s:02d}.nii.gz")
            create_sample_3d_statistical_nifti(
                s_path,
                shape=shape_3d,
                snr=2.5,
                seed=42 + s,
            )
            raw_vols.append(s_path)
        input_data = raw_vols
    else:
        input_data = input_path

    logger.info(f"Executing Phase 4 voxel-level testing: test={test_type}, alternative={alternative}")
    if roi_path:
        logger.info(f"Restricting statistical analysis to Phase 3 ROI mask: {roi_path}")
    else:
        logger.info("No valid ROI mask detected or specified. Analyzing full volume.")

    # Execute statistical testing
    stat_maps = compute_statistics(
        data=input_data,
        roi_mask=roi_path,
        test_type=test_type,
        alternative=alternative,
        popmean=popmean,
        equal_var=equal_var,
        nan_policy=nan_policy,
    )

    # Save spatially aligned NIfTI maps
    saved_maps = stat_maps.save_nifti_maps(output_dir=output_dir, prefix=prefix)

    # Plot multi-panel visualization
    fig_dest = fig_dir / f"{prefix}_summary_slices.png"
    plot_statistical_maps(
        stat_maps=stat_maps,
        output_path=fig_dest,
        p_threshold=out_cfg.get("p_threshold", 0.05),
        dpi=out_cfg.get("figure_dpi", 300),
    )

    # Verify fcHMRF-LIS compatibility
    fchmrf_inputs = stat_maps.to_fchmrf_inputs(restrict_to_mask=True)
    logger.info(
        f"Verified fcHMRF-LIS compatibility: "
        f"In-mask z-scores shape: {fchmrf_inputs['z_scores'].shape}, "
        f"Mean z: {np.mean(fchmrf_inputs['z_scores']):.3f}, "
        f"Std z: {np.std(fchmrf_inputs['z_scores']):.3f}, "
        f"Min p: {np.min(fchmrf_inputs['p_values']):.4e}"
    )

    logger.info("Phase 4 Voxel-Level Statistical Testing completed successfully.")


if __name__ == "__main__":
    main()
