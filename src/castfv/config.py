"""Configuration objects for the compact CAST-FV reference implementation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CaseConfig:
    """Nondimensional Cartesian cavity problem.

    The state ordering is ``[p, u, v, T]`` in two dimensions and
    ``[p, u, v, w, T]`` in three dimensions.
    """

    dimension: int = 2
    cells: int = 48
    reynolds: float = 100.0
    peclet: float = 30.0
    density: float = 1.0
    lid_velocity: float = 1.0
    top_scalar: float = 1.0
    wall_scalar: float = 0.0
    pressure_gauge_weight: float = 1.0e-2

    def __post_init__(self) -> None:
        if self.dimension not in (2, 3):
            raise ValueError("dimension must be 2 or 3")
        if self.cells < 4:
            raise ValueError("cells must be at least 4")
        if self.reynolds <= 0.0 or self.peclet <= 0.0:
            raise ValueError("reynolds and peclet must be positive")
        if self.density <= 0.0:
            raise ValueError("density must be positive")
        if self.pressure_gauge_weight < 0.0:
            raise ValueError("pressure_gauge_weight must be non-negative")

    @property
    def spacing(self) -> float:
        return 1.0 / self.cells

    @property
    def cell_volume(self) -> float:
        return self.spacing**self.dimension

    @property
    def face_area(self) -> float:
        return self.spacing ** (self.dimension - 1)

    @property
    def viscosity(self) -> float:
        return 1.0 / self.reynolds

    @property
    def state_channels(self) -> int:
        return self.dimension + 2

    @property
    def feature_channels(self) -> int:
        return 8 if self.dimension == 2 else 12

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        return (self.cells,) * self.dimension


@dataclass(frozen=True)
class OptimizationConfig:
    """Finite-budget, case-specific neural state optimization settings."""

    architecture: str = "cnn"
    budget: int = 200
    learning_rate: float = 1.0e-3
    seed: int = 7
    width: int | None = None
    corrector_steps: int = 4
    weight_decay: float = 1.0e-5
    gradient_clip: float = 1.0
    record_every: int = 10

    def __post_init__(self) -> None:
        if self.budget < 1:
            raise ValueError("budget must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.width is not None and self.width < 4:
            raise ValueError("width must be at least 4")
        if self.corrector_steps < 1:
            raise ValueError("corrector_steps must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.gradient_clip <= 0.0:
            raise ValueError("gradient_clip must be positive")
        if self.record_every < 1:
            raise ValueError("record_every must be positive")
