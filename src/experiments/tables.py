"""Automated comparison table generation (CSV, Markdown, LaTeX) for research experiments."""

from pathlib import Path
from typing import Dict, List, Union
import csv

from src.experiments.schemas import ExperimentResult


def export_suite_summary_csv(
    results: List[ExperimentResult],
    output_path: Union[str, Path],
) -> Path:
    """Export comprehensive suite metrics and configurations to a single CSV spreadsheet."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "experiment_id",
        "name",
        "status",
        "dataset_type",
        "grid_size",
        "snr",
        "seed",
        "norm_method",
        "unet_enabled",
        "test_type",
        "fchmrf_mode",
        "beta",
        "sigma_spatial",
        "lambda_dl",
        "nominal_alpha",
        "adaptive_enabled",
        "adaptive_algorithm",
        "total_runtime_seconds",
        "empirical_fdr",
        "tpr_power",
        "fpr",
        "specificity",
        "precision",
        "f1_score",
        "dice",
        "mcc",
        "num_discoveries",
        "true_positives",
        "false_positives",
        "true_negatives",
        "false_negatives",
        "centroid_distance_voxels",
    ]

    with open(out_p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            cfg = r.config
            m = r.metrics or {}
            row = {
                "experiment_id": r.experiment_id,
                "name": cfg.name,
                "status": r.status,
                "dataset_type": cfg.dataset.dataset_type,
                "grid_size": str(cfg.dataset.grid_size),
                "snr": cfg.dataset.snr,
                "seed": cfg.seed,
                "norm_method": cfg.preprocessing.norm_method,
                "unet_enabled": cfg.unet.enabled,
                "test_type": cfg.statistical_test.test_type,
                "fchmrf_mode": cfg.fchmrf.method_mode,
                "beta": cfg.fchmrf.beta,
                "sigma_spatial": cfg.fchmrf.sigma_spatial,
                "lambda_dl": cfg.fchmrf.lambda_dl,
                "nominal_alpha": cfg.fchmrf.alpha,
                "adaptive_enabled": cfg.adaptive.enabled,
                "adaptive_algorithm": cfg.adaptive.algorithm if cfg.adaptive.enabled else "none",
                "total_runtime_seconds": round(r.total_runtime, 3),
                "empirical_fdr": m.get("empirical_fdr", "N/A"),
                "tpr_power": m.get("tpr", "N/A"),
                "fpr": m.get("fpr", "N/A"),
                "specificity": m.get("specificity", "N/A"),
                "precision": m.get("precision", "N/A"),
                "f1_score": m.get("f1_score", "N/A"),
                "dice": m.get("dice", "N/A"),
                "mcc": m.get("mcc", "N/A"),
                "num_discoveries": m.get("num_discoveries", 0),
                "true_positives": m.get("true_positives", 0),
                "false_positives": m.get("false_positives", 0),
                "true_negatives": m.get("true_negatives", 0),
                "false_negatives": m.get("false_negatives", 0),
                "centroid_distance_voxels": m.get("centroid_distance_voxels", "N/A"),
            }
            writer.writerow(row)

    return out_p


def generate_comparison_markdown_table(results: List[ExperimentResult]) -> str:
    """Format suite experiment results into a GitHub Flavored Markdown table."""
    headers = [
        "Experiment ID",
        "Method Mode",
        "SNR",
        "Seed",
        "Emp FDR",
        "Power (TPR)",
        "Precision",
        "Dice",
        "MCC",
        "Discoveries (TP / FP)",
        "Runtime (s)",
    ]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for r in results:
        cfg = r.config
        m = r.metrics or {}

        fdr_val = f"{m.get('empirical_fdr', 0.0):.4f}" if "empirical_fdr" in m else "N/A"
        tpr_val = f"{m.get('tpr', 0.0):.4f}" if "tpr" in m else "N/A"
        prec_val = f"{m.get('precision', 0.0):.4f}" if "precision" in m else "N/A"
        dice_val = f"{m.get('dice', 0.0):.4f}" if "dice" in m else "N/A"
        mcc_val = f"{m.get('mcc', 0.0):.4f}" if "mcc" in m else "N/A"
        disc_val = f"{m.get('num_discoveries', 0)} ({m.get('true_positives', 0)}/{m.get('false_positives', 0)})"

        row = [
            f"**{r.experiment_id}**",
            f"`{cfg.fchmrf.method_mode}`",
            f"{cfg.dataset.snr:.1f}",
            f"{cfg.seed}",
            fdr_val,
            tpr_val,
            prec_val,
            dice_val,
            mcc_val,
            disc_val,
            f"{r.total_runtime:.2f}",
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def generate_comparison_latex_table(results: List[ExperimentResult]) -> str:
    """Format suite experiment results into a publication-ready LaTeX table."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Empirical evaluation of spatial false discovery rate methods across experimental conditions.}",
        r"\label{tab:spatial_fdr_comparison}",
        r"\small",
        r"\begin{tabular}{llcccccccc}",
        r"\toprule",
        r"\textbf{Experiment ID} & \textbf{Method} & \textbf{SNR} & \textbf{FDR ($\alpha=0.05$)} & \textbf{Power (TPR)} & \textbf{Precision} & \textbf{Dice} & \textbf{MCC} & \textbf{TP / FP} & \textbf{Time (s)} \\",
        r"\midrule",
    ]

    for r in results:
        cfg = r.config
        m = r.metrics or {}

        clean_id = r.experiment_id.replace("_", r"\_")
        clean_mode = cfg.fchmrf.method_mode.replace("_", r"\_")

        fdr_val = f"{m.get('empirical_fdr', 0.0):.3f}" if "empirical_fdr" in m else "--"
        tpr_val = f"{m.get('tpr', 0.0):.3f}" if "tpr" in m else "--"
        prec_val = f"{m.get('precision', 0.0):.3f}" if "precision" in m else "--"
        dice_val = f"{m.get('dice', 0.0):.3f}" if "dice" in m else "--"
        mcc_val = f"{m.get('mcc', 0.0):.3f}" if "mcc" in m else "--"
        tp_fp = f"{m.get('true_positives', 0)} / {m.get('false_positives', 0)}"
        time_val = f"{r.total_runtime:.2f}"

        line = f"{clean_id} & {clean_mode} & {cfg.dataset.snr:.1f} & {fdr_val} & {tpr_val} & {prec_val} & {dice_val} & {mcc_val} & {tp_fp} & {time_val} \\\\"
        lines.append(line)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    return "\n".join(lines)
