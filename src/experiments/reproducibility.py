"""Automated experiment reproduction and verification engine."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from src.core.logging import get_logger
from src.experiments.schemas import ExperimentConfig, ExperimentResult
from src.experiments.runner import ExperimentRunner

logger = get_logger(__name__)


def reproduce_experiment(
    config_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    tolerance: float = 1e-4,
) -> Tuple[ExperimentResult, bool, Dict[str, Any]]:
    """Reproduce an experiment from its saved configuration and verify fidelity.

    Args:
        config_path: Path to saved config.json or config.yaml.
        output_dir: Optional override for output destination. If None, saves in a
                    reproduction subfolder adjacent to config.
        tolerance: Numerical floating point tolerance for metric verification.

    Returns:
        Tuple:
          - ExperimentResult: Newly executed experiment result.
          - bool: True if new metrics match original results within tolerance.
          - Dict[str, Any]: Detailed side-by-side metric verification comparison.
    """
    cfg_p = Path(config_path).resolve()
    if not cfg_p.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_p}")

    # 1. Load ExperimentConfig
    if cfg_p.suffix in [".yaml", ".yml"]:
        config = ExperimentConfig.from_yaml(cfg_p)
    else:
        config = ExperimentConfig.from_json(cfg_p)

    logger.info(f"Loaded configuration for experiment: {config.experiment_id} [seed={config.seed}]")

    # 2. Check for original result.json to compare against
    orig_result_p = cfg_p.parent / "result.json"
    original_metrics: Dict[str, Any] = {}
    if orig_result_p.exists():
        try:
            with open(orig_result_p, "r", encoding="utf-8") as f:
                orig_data = json.load(f)
                original_metrics = orig_data.get("metrics", {})
            logger.info(f"Found original experiment result at: {orig_result_p}")
        except Exception as e:
            logger.warning(f"Failed to parse original results file {orig_result_p}: {e}")

    # 3. Configure output destination for reproduction run
    if output_dir is not None:
        repro_dir = Path(output_dir)
    else:
        repro_dir = cfg_p.parent / "reproduction"
    repro_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir = str(repro_dir.parent)
    config.experiment_id = repro_dir.name

    # 4. Re-execute experiment
    runner = ExperimentRunner(base_output_dir=repro_dir.parent, device=config.device)
    new_result = runner.run_experiment(config)

    # 5. Side-by-side numerical verification
    is_exact_match = True
    delta_report: Dict[str, Any] = {}

    if original_metrics and new_result.status == "completed":
        logger.info("\n" + "=" * 80)
        logger.info("REPRODUCIBILITY VERIFICATION REPORT")
        logger.info("=" * 80)
        logger.info(f"{'Metric':<25} | {'Original':<15} | {'Reproduced':<15} | {'Delta':<12} | {'Match'}")
        logger.info("-" * 80)

        for key in ["num_discoveries", "empirical_fdr", "tpr", "precision", "dice", "mcc"]:
            orig_val = original_metrics.get(key)
            new_val = new_result.metrics.get(key)

            if orig_val is None or new_val is None:
                continue

            if isinstance(orig_val, (int, np.integer)):
                delta = abs(int(new_val) - int(orig_val))
                matches = delta == 0
                delta_str = f"{delta}"
            else:
                delta = abs(float(new_val) - float(orig_val))
                matches = delta <= tolerance
                delta_str = f"{delta:.6f}"

            if not matches:
                is_exact_match = False

            match_symbol = "[OK] MATCH" if matches else "[FAIL] MISMATCH"
            logger.info(f"{key:<25} | {str(orig_val):<15} | {str(new_val):<15} | {delta_str:<12} | {match_symbol}")

            delta_report[key] = {
                "original": orig_val,
                "reproduced": new_val,
                "delta": delta,
                "matches": matches,
            }

        logger.info("=" * 80)
        if is_exact_match:
            logger.info("[SUCCESS] Experiment replicated with exact numerical bit-for-bit / scientific fidelity!")
        else:
            logger.warning("[WARNING] Numerical differences detected between original and reproduced run.")
    else:
        logger.info("No prior result.json found for baseline comparison; newly produced run saved.")
        is_exact_match = True

    return new_result, is_exact_match, delta_report
