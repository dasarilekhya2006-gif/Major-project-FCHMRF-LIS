"""Experimental design specifications and contrast definitions for voxel-wise neuroimaging statistics."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from src.core.exceptions import StatisticalInferenceError
from src.core.logging import get_logger

logger = get_logger(__name__)


class TestType(str, Enum):
    """Supported voxel-level statistical hypothesis tests."""

    __test__ = False

    ONE_SAMPLE_T = "one_sample_t"
    TWO_SAMPLE_T = "two_sample_t"
    PAIRED_T = "paired_t"
    GLM = "glm"
    DIRECT_Z = "direct_z"


class AlternativeHypothesis(str, Enum):
    """Directionality of alternative hypothesis H_1."""

    TWO_SIDED = "two_sided"
    GREATER = "greater"  # Positive activation (typical neuroimaging contrast)
    LESS = "less"        # Deactivation / negative contrast


@dataclass
class ExperimentalDesign:
    """Encapsulates the experimental design matrix, contrasts, and statistical hypothesis configuration.

    Attributes:
        test_type: TestType enum specifying the analysis procedure.
        design_matrix: 2D array of shape (N_samples, P_regressors).
        contrast: 1D or 2D array of contrast weights (length P).
        regressor_names: Optional human-readable labels for each column in design_matrix.
        subject_ids: Optional list of subject identifiers for each row in design_matrix.
        alternative: Alternative hypothesis directionality ('greater', 'two_sided', 'less').
        equal_var: Whether to assume equal variance across groups (for two-sample t-test).
        popmean: Hypothesized population mean (for one-sample t-test, default 0.0).
    """

    test_type: Union[TestType, str] = TestType.ONE_SAMPLE_T
    design_matrix: Optional[np.ndarray] = None
    contrast: Optional[np.ndarray] = None
    regressor_names: List[str] = field(default_factory=list)
    subject_ids: List[str] = field(default_factory=list)
    alternative: Union[AlternativeHypothesis, str] = AlternativeHypothesis.GREATER
    equal_var: bool = False  # Default to Welch's t-test for robustness
    popmean: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.test_type, str):
            self.test_type = TestType(self.test_type.lower())
        if isinstance(self.alternative, str):
            self.alternative = AlternativeHypothesis(self.alternative.lower())

        if self.design_matrix is not None:
            self.design_matrix = np.asarray(self.design_matrix, dtype=np.float64)
            if self.design_matrix.ndim == 1:
                self.design_matrix = self.design_matrix[:, np.newaxis]
            if self.design_matrix.ndim != 2:
                raise StatisticalInferenceError(
                    f"Design matrix must be 2D, got shape {self.design_matrix.shape}"
                )

        if self.contrast is not None:
            self.contrast = np.asarray(self.contrast, dtype=np.float64)
            if self.contrast.ndim == 1 and self.design_matrix is not None:
                if len(self.contrast) != self.design_matrix.shape[1]:
                    raise StatisticalInferenceError(
                        f"Contrast length ({len(self.contrast)}) does not match "
                        f"design matrix regressors ({self.design_matrix.shape[1]})."
                    )


def create_one_sample_design(
    num_samples: int,
    popmean: float = 0.0,
    alternative: Union[AlternativeHypothesis, str] = AlternativeHypothesis.GREATER,
    subject_ids: Optional[List[str]] = None,
) -> ExperimentalDesign:
    """Create a standard one-sample design testing group mean activation > popmean."""
    design_mat = np.ones((num_samples, 1), dtype=np.float64)
    contrast = np.array([1.0], dtype=np.float64)
    return ExperimentalDesign(
        test_type=TestType.ONE_SAMPLE_T,
        design_matrix=design_mat,
        contrast=contrast,
        regressor_names=["Intercept"],
        subject_ids=subject_ids or [f"sub_{i:02d}" for i in range(num_samples)],
        alternative=alternative,
        popmean=popmean,
    )


def create_two_sample_design(
    num_group1: int,
    num_group2: int,
    equal_var: bool = False,
    alternative: Union[AlternativeHypothesis, str] = AlternativeHypothesis.GREATER,
    group1_label: str = "Group1",
    group2_label: str = "Group2",
) -> ExperimentalDesign:
    """Create a two-sample independent group design (e.g. Patients vs Controls).

    Design matrix:
      Column 0: Group 1 indicator (1 if Group 1, 0 otherwise)
      Column 1: Group 2 indicator (1 if Group 2, 0 otherwise)
    Contrast:
      [1.0, -1.0] testing Group 1 > Group 2.
    """
    n_total = num_group1 + num_group2
    design_mat = np.zeros((n_total, 2), dtype=np.float64)
    design_mat[:num_group1, 0] = 1.0
    design_mat[num_group1:, 1] = 1.0

    contrast = np.array([1.0, -1.0], dtype=np.float64)
    subjs = [f"{group1_label}_{i:02d}" for i in range(num_group1)] + [
        f"{group2_label}_{i:02d}" for i in range(num_group2)
    ]

    return ExperimentalDesign(
        test_type=TestType.TWO_SAMPLE_T,
        design_matrix=design_mat,
        contrast=contrast,
        regressor_names=[group1_label, group2_label],
        subject_ids=subjs,
        alternative=alternative,
        equal_var=equal_var,
    )


def create_paired_design(
    num_pairs: int,
    alternative: Union[AlternativeHypothesis, str] = AlternativeHypothesis.GREATER,
    condition1_label: str = "ConditionA",
    condition2_label: str = "ConditionB",
) -> ExperimentalDesign:
    """Create a paired within-subject design (e.g. Pre-treatment vs Post-treatment)."""
    design_mat = np.ones((num_pairs, 1), dtype=np.float64)
    contrast = np.array([1.0], dtype=np.float64)
    return ExperimentalDesign(
        test_type=TestType.PAIRED_T,
        design_matrix=design_mat,
        contrast=contrast,
        regressor_names=[f"{condition1_label}_minus_{condition2_label}"],
        subject_ids=[f"pair_{i:02d}" for i in range(num_pairs)],
        alternative=alternative,
    )


def create_glm_design(
    design_matrix: np.ndarray,
    contrast: np.ndarray,
    regressor_names: Optional[List[str]] = None,
    alternative: Union[AlternativeHypothesis, str] = AlternativeHypothesis.GREATER,
) -> ExperimentalDesign:
    """Create a general linear model (GLM) design with custom regressors and contrast."""
    return ExperimentalDesign(
        test_type=TestType.GLM,
        design_matrix=design_matrix,
        contrast=contrast,
        regressor_names=regressor_names or [f"regressor_{i}" for i in range(design_matrix.shape[1])],
        alternative=alternative,
    )
