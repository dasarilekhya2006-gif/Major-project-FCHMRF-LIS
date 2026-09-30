"""Hyperparameter search space definitions for adaptive fcHMRF-LIS tuning."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.core.logging import get_logger

logger = get_logger(__name__)


class ParameterType(str, Enum):
    """Supported hyperparameter domain types."""

    CONTINUOUS = "continuous"
    INTEGER = "integer"
    CATEGORICAL = "categorical"


@dataclass
class HyperparameterRange:
    """Definition of a single hyperparameter's bounds and defaults.

    Attributes:
        name: Name matching FCHMRFParameters attribute.
        param_type: ParameterType.
        bounds: (min_val, max_val) for numerical, or list of choices for categorical.
        default_val: Standard fixed baseline value.
        log_scale: If True, samples and scales on logarithmic scale.
    """

    name: str
    param_type: ParameterType
    bounds: Union[Tuple[float, float], List[Any]]
    default_val: Any
    log_scale: bool = False

    def clip_value(self, val: Any) -> Any:
        """Clip a numerical value strictly within defined bounds."""
        if self.param_type in (ParameterType.CONTINUOUS, ParameterType.INTEGER):
            low, high = self.bounds[0], self.bounds[1]
            clipped = max(low, min(high, float(val)))
            return int(round(clipped)) if self.param_type == ParameterType.INTEGER else clipped
        return val

    def normalize(self, val: Any) -> float:
        """Map value to normalized unit interval [0, 1] for surrogate GP modeling."""
        if self.param_type in (ParameterType.CONTINUOUS, ParameterType.INTEGER):
            low, high = self.bounds[0], self.bounds[1]
            if self.log_scale:
                val_log = np.log(max(val, 1e-8))
                low_log = np.log(max(low, 1e-8))
                high_log = np.log(max(high, 1e-8))
                return float((val_log - low_log) / (high_log - low_log + 1e-12))
            return float((val - low) / (high - low + 1e-12))
        elif self.param_type == ParameterType.CATEGORICAL:
            idx = self.bounds.index(val)
            return float(idx / max(1, len(self.bounds) - 1))
        return 0.5

    def denormalize(self, norm_val: float) -> Any:
        """Map normalized value in [0, 1] back to original hyperparameter domain."""
        clamped = max(0.0, min(1.0, float(norm_val)))
        if self.param_type in (ParameterType.CONTINUOUS, ParameterType.INTEGER):
            low, high = self.bounds[0], self.bounds[1]
            if self.log_scale:
                low_log = np.log(max(low, 1e-8))
                high_log = np.log(max(high, 1e-8))
                val = np.exp(low_log + clamped * (high_log - low_log))
            else:
                val = low + clamped * (high - low)
            return int(round(val)) if self.param_type == ParameterType.INTEGER else float(val)
        elif self.param_type == ParameterType.CATEGORICAL:
            idx = int(round(clamped * (len(self.bounds) - 1)))
            return self.bounds[min(idx, len(self.bounds) - 1)]
        return self.default_val


@dataclass
class SearchSpace:
    """Manages multi-dimensional parameter search domains for fcHMRF-LIS."""

    parameters: List[HyperparameterRange] = field(default_factory=list)

    def add_parameter(self, param: HyperparameterRange) -> None:
        self.parameters.append(param)

    @property
    def dimension(self) -> int:
        return len(self.parameters)

    @property
    def names(self) -> List[str]:
        return [p.name for p in self.parameters]

    def get_default_parameters(self) -> Dict[str, Any]:
        """Extract baseline predefined default parameter dictionary."""
        return {p.name: p.default_val for p in self.parameters}

    def sample_random(self, rng: Optional[np.random.RandomState] = None) -> Dict[str, Any]:
        """Draw a random uniform sample across the search space."""
        if rng is None:
            rng = np.random.RandomState()
        sample = {}
        for p in self.parameters:
            u = rng.uniform(0.0, 1.0)
            sample[p.name] = p.denormalize(u)
        return sample

    def to_normalized_vector(self, param_dict: Dict[str, Any]) -> np.ndarray:
        """Vectorize a parameter dictionary into a normalized array in [0, 1]^D."""
        vec = np.zeros(self.dimension, dtype=np.float64)
        for i, p in enumerate(self.parameters):
            val = param_dict.get(p.name, p.default_val)
            vec[i] = p.normalize(val)
        return vec

    def from_normalized_vector(self, vec: np.ndarray) -> Dict[str, Any]:
        """Convert a normalized array in [0, 1]^D back to parameter dictionary."""
        params = {}
        for i, p in enumerate(self.parameters):
            params[p.name] = p.denormalize(vec[i])
        return params


def create_fchmrf_search_space(
    include_lambda_dl: bool = True,
    include_spatial_bandwidths: bool = True,
) -> SearchSpace:
    """Construct standard mathematically valid search space for fcHMRF-LIS tuning.

    Parameters Searched:
      1. beta: Spatial coupling parameter in [0.1, 2.5] (baseline: 0.8)
      2. sigma_spatial: Spatial Euclidean kernel bandwidth in [1.0, 5.0] (baseline: 2.5)
      3. sigma_intensity: Contrast difference kernel bandwidth in [0.5, 3.5] (baseline: 1.5)
      4. lambda_dl (optional): Weight of 3D U-Net prior in [0.0, 1.0] (baseline: 0.5)
      5. w_pure_spatial: Weight of pure spatial proximity in [0.0, 0.5] (baseline: 0.2)
    """
    space = SearchSpace()

    # 1. Spatial coupling strength beta
    space.add_parameter(
        HyperparameterRange(
            name="beta",
            param_type=ParameterType.CONTINUOUS,
            bounds=(0.1, 2.5),
            default_val=0.8,
            log_scale=False,
        )
    )

    if include_spatial_bandwidths:
        # 2. Spatial distance bandwidth sigma_spatial
        space.add_parameter(
            HyperparameterRange(
                name="sigma_spatial",
                param_type=ParameterType.CONTINUOUS,
                bounds=(1.0, 5.0),
                default_val=2.5,
                log_scale=False,
            )
        )
        # 3. Radiometric contrast difference bandwidth sigma_intensity
        space.add_parameter(
            HyperparameterRange(
                name="sigma_intensity",
                param_type=ParameterType.CONTINUOUS,
                bounds=(0.5, 3.5),
                default_val=1.5,
                log_scale=False,
            )
        )
        # 4. Pure spatial kernel weight
        space.add_parameter(
            HyperparameterRange(
                name="w_pure_spatial",
                param_type=ParameterType.CONTINUOUS,
                bounds=(0.0, 0.5),
                default_val=0.2,
                log_scale=False,
            )
        )

    if include_lambda_dl:
        # 5. Deep learning prior weight lambda_dl
        space.add_parameter(
            HyperparameterRange(
                name="lambda_dl",
                param_type=ParameterType.CONTINUOUS,
                bounds=(0.0, 1.0),
                default_val=0.5,
                log_scale=False,
            )
        )

    return space
