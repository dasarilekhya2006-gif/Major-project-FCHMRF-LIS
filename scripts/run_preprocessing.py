"""Command-line interface to execute the neuroimaging preprocessing pipeline with QC generation."""

import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.logging import setup_logging, get_logger
from src.data.sample_data import create_sample_3d_statistical_nifti
from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.schemas import PreprocessingConfig
from src.preprocessing.qc import generate_preprocessing_qc_plot

logger = get_logger("run_preprocessing")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 2 neuroimaging preprocessing pipeline."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input NIfTI file (.nii or .nii.gz). If omitted, a sample synthetic volume is generated.",
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Optional path to reference brain mask.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/preprocessing_config.yaml",
        help="Path to preprocessing configuration YAML file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory where preprocessed outputs and metadata will be saved.",
    )
    parser.add_argument(
        "--qc-dir",
        type=str,
        default="outputs/qc",
        help="Directory where visual QC figures will be saved.",
    )
    parser.add_argument(
        "--generate-qc",
        action="store_true",
        default=True,
        help="Generate visual quality control plot.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging(log_level="INFO")
    args = parse_args()

    # Load configuration
    cfg_path = Path(args.config)
    if cfg_path.exists():
        logger.info(f"Loading preprocessing configuration from {cfg_path}")
        config = PreprocessingConfig.from_yaml(cfg_path)
    else:
        logger.warning(f"Config file {cfg_path} not found. Using default PreprocessingConfig.")
        config = PreprocessingConfig()

    # If no input provided, generate a sample synthetic volume for testing
    if args.input is None or not Path(args.input).exists():
        temp_input = Path("data/raw/synthetic_sample_input.nii.gz")
        temp_input.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Generating synthetic 3D test volume at {temp_input}")
        create_sample_3d_statistical_nifti(
            temp_input, shape=(60, 60, 30), voxel_size=(2.0, 2.0, 2.0), snr=2.5, seed=42
        )
        input_path = temp_input
    else:
        input_path = Path(args.input)

    # Initialize and execute pipeline
    pipeline = PreprocessingPipeline(config=config)
    logger.info(f"Executing preprocessing pipeline on {input_path}")
    preprocessed = pipeline.run(input_path, mask_path=args.mask)

    # Save outputs
    out_dir = Path(args.output_dir)
    saved_files = preprocessed.save_preprocessed(out_dir, prefix=input_path.stem.replace(".nii", ""))
    logger.info(f"Saved preprocessed volume: {saved_files['volume']}")
    logger.info(f"Saved preprocessing metadata: {saved_files['metadata']}")

    # Generate visual QC plot if requested
    if args.generate_qc:
        from src.data.loaders import load_nifti
        raw_vol = load_nifti(input_path)
        qc_dir = Path(args.qc_dir)
        qc_dir.mkdir(parents=True, exist_ok=True)
        qc_path = qc_dir / f"{input_path.stem.replace('.nii', '')}_qc.png"
        generate_preprocessing_qc_plot(raw_vol, preprocessed, output_path=qc_path)
        logger.info(f"Visual QC report created: {qc_path}")

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
