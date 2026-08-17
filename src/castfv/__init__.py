"""CAST-FV: compact, solution-label-free finite-volume state optimization."""

from .config import CaseConfig, OptimizationConfig
from .features import coordinate_boundary_features
from .models import build_state_map, parameter_count
from .optimize import OptimizationResult, TimeSeriesResult, advance_physical_time, optimize_state
from .residuals import finite_volume_objective, finite_volume_residual, zero_state

__all__ = [
    "CaseConfig",
    "OptimizationConfig",
    "OptimizationResult",
    "TimeSeriesResult",
    "advance_physical_time",
    "build_state_map",
    "coordinate_boundary_features",
    "finite_volume_objective",
    "finite_volume_residual",
    "optimize_state",
    "parameter_count",
    "zero_state",
]

__version__ = "0.1.0"
