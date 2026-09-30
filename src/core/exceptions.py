"""Custom exception hierarchy for the Adaptive DL fcHMRF-LIS project."""

class NeuroimagingError(Exception):
    """Base exception for all errors within the project."""
    pass


class ConfigurationError(NeuroimagingError):
    """Raised when configuration parameters are invalid or missing."""
    pass


class PipelineError(NeuroimagingError):
    """Raised when an end-to-end pipeline execution stage fails."""
    pass


class DataValidationError(NeuroimagingError):
    """Raised when input neuroimaging volumes, masks, or affines are malformed."""
    pass


class EmptyVolumeError(DataValidationError):
    """Raised when a neuroimaging volume contains only zeros or uninformative empty voxels."""
    pass


class CorruptNiftiError(DataValidationError):
    """Raised when a NIfTI file is truncated, unreadable, or possesses a corrupted header."""
    pass


class InvalidAffineError(DataValidationError):
    """Raised when an affine transformation matrix is singular or contains invalid values."""
    pass


class DimensionMismatchError(DataValidationError):
    """Raised when volume or mask spatial dimensions do not match expected shapes."""
    pass


class PreprocessingError(NeuroimagingError):
    """Raised when spatial masking, standardization, or graph construction fails."""
    pass


class DeepLearningModelError(NeuroimagingError):
    """Raised when deep learning model construction, forward pass, or checkpointing fails."""
    pass


class StatisticalInferenceError(NeuroimagingError):
    """Raised when FDR, p-value transformation, or two-group mixture estimation fails."""
    pass


class ModelConvergenceError(NeuroimagingError):
    """Raised when fcHMRF mean-field iteration or EM fails to converge within tolerances."""
    pass


class RobustnessAnalysisError(NeuroimagingError):
    """Raised when perturbation or stress-testing pipelines encounter an unrecoverable state."""
    pass


class MemoryLimitExceededError(NeuroimagingError):
    """Raised when the constructed spatial graph would exceed configured memory safety limits."""
    def __init__(self, message: str, details: dict):
        super().__init__(message)
        self.details = details

