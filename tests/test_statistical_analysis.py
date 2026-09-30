"""Unit tests for statistical analysis and classical FDR procedures."""

import numpy as np
import pytest

from src.statistical_analysis.classical_fdr import benjamini_hochberg, benjamini_yekutieli
from src.statistical_analysis.storey_qvalue import storey_qvalues, estimate_pi0
from src.statistical_analysis.local_fdr import efron_locfdr
from src.statistical_analysis.mixture_models import fit_two_group_mixture_em


def test_benjamini_hochberg_nominal_control() -> None:
    """Test BH procedure rejects low p-values and controls thresholding."""
    p_vals = np.array([0.001, 0.002, 0.01, 0.04, 0.2, 0.5, 0.8, 0.9])
    rejected, q_vals, thresh = benjamini_hochberg(p_vals, alpha=0.05)
    assert rejected[0]  # First two should be rejected
    assert rejected[1]
    assert not rejected[-1]  # Highest p-value should not be rejected
    assert len(q_vals) == len(p_vals)


def test_storey_qvalues() -> None:
    """Test Storey q-value calculation."""
    p_vals = np.linspace(0.001, 0.999, 100)
    rejected, q_vals, pi0 = storey_qvalues(p_vals, alpha=0.05)
    assert 0.0 < pi0 <= 1.0
    assert len(rejected) == 100


def test_two_group_mixture_em() -> None:
    """Test fitting two-group Gaussian mixture model."""
    null_z = np.random.randn(500)
    active_z = np.random.randn(50) + 3.0
    z_all = np.concatenate([null_z, active_z])

    params = fit_two_group_mixture_em(z_all, fix_null=True)
    assert params.p0 > 0.80
    assert params.mu1 > 1.5
