"""Rigorous statistical hypothesis testing and effect size estimation for benchmark comparisons."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PairedTestResult:
    """Statistical summary of paired hypothesis test between two methods."""

    metric_name: str
    baseline_name: str
    candidate_name: str
    n_samples: int
    mean_baseline: float
    mean_candidate: float
    mean_difference: float  # candidate - baseline
    ci_95_lower: float
    ci_95_upper: float
    t_statistic: float
    t_pvalue: float
    wilcoxon_statistic: float
    wilcoxon_pvalue: float
    cohens_d: float
    cliffs_delta: float
    is_statistically_significant: bool  # at alpha = 0.05
    holm_bonferroni_pvalue: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "baseline_name": self.baseline_name,
            "candidate_name": self.candidate_name,
            "n_samples": self.n_samples,
            "mean_baseline": self.mean_baseline,
            "mean_candidate": self.mean_candidate,
            "mean_difference": self.mean_difference,
            "ci_95": [self.ci_95_lower, self.ci_95_upper],
            "t_statistic": self.t_statistic,
            "t_pvalue": self.t_pvalue,
            "wilcoxon_statistic": self.wilcoxon_statistic,
            "wilcoxon_pvalue": self.wilcoxon_pvalue,
            "cohens_d": self.cohens_d,
            "cliffs_delta": self.cliffs_delta,
            "is_significant": self.is_statistically_significant,
            "adjusted_pvalue": self.holm_bonferroni_pvalue,
        }


def calculate_cohens_d(x: np.ndarray, y: np.ndarray, paired: bool = True) -> float:
    """Calculate Cohen's d effect size for paired or independent samples.

    For paired samples, Cohen's d_z = mean(diff) / std(diff).
    Interpretation (Cohen, 1988):
      |d| < 0.2: Negligible
      0.2 <= |d| < 0.5: Small
      0.5 <= |d| < 0.8: Medium
      |d| >= 0.8: Large
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if paired:
        diff = y - x
        sd = float(np.std(diff, ddof=1))
        if sd < 1e-12:
            return 0.0
        return float(np.mean(diff) / sd)
    else:
        n1, n2 = len(x), len(y)
        s1, s2 = np.var(x, ddof=1), np.var(y, ddof=1)
        s_pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / max(1, n1 + n2 - 2))
        if s_pooled < 1e-12:
            return 0.0
        return float((np.mean(y) - np.mean(x)) / s_pooled)


def calculate_cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate Cliff's delta non-parametric effect size: delta = 2*W/(n1*n2) - 1.

    Interpretation (Romano et al., 2006):
      |delta| < 0.147: Negligible
      0.147 <= |delta| < 0.33: Small
      0.33 <= |delta| < 0.474: Medium
      |delta| >= 0.474: Large
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0

    # Pairwise comparisons
    greater = np.sum(y[:, None] > x[None, :])
    lesser = np.sum(y[:, None] < x[None, :])
    return float((greater - lesser) / (n_x * n_y))


def paired_statistical_comparison(
    baseline_values: Union[List[float], np.ndarray],
    candidate_values: Union[List[float], np.ndarray],
    metric_name: str,
    baseline_name: str = "Baseline",
    candidate_name: str = "Proposed",
    alpha: float = 0.05,
) -> PairedTestResult:
    """Perform formal parametric and non-parametric paired hypothesis tests between two methods.

    Executes:
      1. Paired Student's t-test (two-sided)
      2. Wilcoxon signed-rank test (two-sided)
      3. 95% Confidence Interval for mean difference
      4. Cohen's d_z effect size
      5. Cliff's delta non-parametric effect size
    """
    x = np.asarray(baseline_values, dtype=np.float64)
    y = np.asarray(candidate_values, dtype=np.float64)

    if len(x) != len(y):
        raise ValueError(f"Sample size mismatch: baseline ({len(x)}) vs candidate ({len(y)})")

    n = len(x)
    diff = y - x
    mean_diff = float(np.mean(diff))

    # Handle degenerate case of zero variance (e.g. constant values)
    std_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0

    if std_diff < 1e-12:
        ci_lower = mean_diff
        ci_upper = mean_diff
        t_stat = 0.0
        t_pval = 1.0
        w_stat = 0.0
        w_pval = 1.0
    else:
        se_diff = std_diff / np.sqrt(n)
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
        ci_lower = mean_diff - t_crit * se_diff
        ci_upper = mean_diff + t_crit * se_diff

        t_res = stats.ttest_rel(y, x)
        t_stat = float(t_res.statistic)
        t_pval = float(t_res.pvalue)

        # Wilcoxon signed-rank test
        try:
            w_res = stats.wilcoxon(y, x, zero_method="pratt")
            w_stat = float(w_res.statistic)
            w_pval = float(w_res.pvalue)
        except Exception:
            w_stat = 0.0
            w_pval = 1.0

    d = calculate_cohens_d(x, y, paired=True)
    c_delta = calculate_cliffs_delta(x, y)
    is_sig = bool(t_pval < alpha or w_pval < alpha)

    return PairedTestResult(
        metric_name=metric_name,
        baseline_name=baseline_name,
        candidate_name=candidate_name,
        n_samples=n,
        mean_baseline=float(np.mean(x)),
        mean_candidate=float(np.mean(y)),
        mean_difference=mean_diff,
        ci_95_lower=ci_lower,
        ci_95_upper=ci_upper,
        t_statistic=t_stat,
        t_pvalue=t_pval,
        wilcoxon_statistic=w_stat,
        wilcoxon_pvalue=w_pval,
        cohens_d=d,
        cliffs_delta=c_delta,
        is_statistically_significant=is_sig,
    )


def apply_holm_bonferroni(
    tests: List[PairedTestResult],
    alpha: float = 0.05,
) -> List[PairedTestResult]:
    """Apply Holm-Bonferroni step-down correction across a family of paired hypothesis tests."""
    if not tests:
        return []

    # Sort tests by ascending unadjusted p-value (using t-test p-value)
    sorted_indices = np.argsort([t.t_pvalue for t in tests])
    m = len(tests)

    for rank, idx in enumerate(sorted_indices):
        orig_p = tests[idx].t_pvalue
        # Holm-Bonferroni adjusted p-value = min(1.0, orig_p * (m - rank))
        adj_p = min(1.0, orig_p * (m - rank))
        tests[idx].holm_bonferroni_pvalue = float(adj_p)
        tests[idx].is_statistically_significant = bool(adj_p < alpha)

    return tests
