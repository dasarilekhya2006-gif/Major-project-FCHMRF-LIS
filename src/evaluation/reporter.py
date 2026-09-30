"""Formatting and reporting tools for neuroimaging benchmark results."""

from pathlib import Path
from typing import Dict, List
import numpy as np

from src.core.logging import get_logger
from src.evaluation.metrics import EvaluationMetrics

logger = get_logger(__name__)


def generate_metrics_markdown_table(results: Dict[str, EvaluationMetrics]) -> str:
    """Format benchmark metrics into a GitHub-flavored Markdown table.

    Args:
        results: Mapping from method name to EvaluationMetrics.

    Returns:
        str: Formatted Markdown table.
    """
    headers = [
        "Method",
        "Empirical FDR",
        "Power (TPR)",
        "FNDR",
        "F1-Score",
        "Dice",
        "MCC",
        "Discoveries",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for method, m in results.items():
        row = [
            f"**{method}**",
            f"{m.empirical_fdr:.4f}",
            f"{m.tpr:.4f}",
            f"{m.fndr:.4f}",
            f"{m.f1_score:.4f}",
            f"{m.dice:.4f}",
            f"{m.mcc:.4f}",
            f"{m.num_discoveries} (TP={m.true_positives}, FP={m.false_positives})",
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def export_metrics_to_csv(
    results: Dict[str, EvaluationMetrics],
    output_path: Path,
) -> None:
    """Save benchmark metrics to CSV file.

    Args:
        results: Mapping of method names to metrics.
        output_path: Output CSV destination.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("method,empirical_fdr,tpr,fndr,precision,f1_score,mcc,dice,num_discoveries,tp,fp\n")
        for method, m in results.items():
            f.write(
                f"{method},{m.empirical_fdr},{m.tpr},{m.fndr},{m.precision},"
                f"{m.f1_score},{m.mcc},{m.dice},{m.num_discoveries},{m.true_positives},{m.false_positives}\n"
            )
    logger.info(f"Exported benchmark metrics to {output_path}")
