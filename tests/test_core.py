from __future__ import annotations

import math

import pytest
import torch

from castfv import (
    CaseConfig,
    OptimizationConfig,
    advance_physical_time,
    build_state_map,
    coordinate_boundary_features,
    finite_volume_residual,
    optimize_state,
    zero_state,
)


def test_default_grid_matches_reported_protocol() -> None:
    assert CaseConfig().cells == 48


@pytest.mark.parametrize("dimension", [2, 3])
def test_zero_state_satisfies_zero_boundary_problem(dimension: int) -> None:
    case = CaseConfig(
        dimension=dimension,
        cells=4,
        lid_velocity=0.0,
        top_scalar=0.0,
        wall_scalar=0.0,
    )
    state = zero_state(case)
    residual = finite_volume_residual(state, case)
    torch.testing.assert_close(residual, torch.zeros_like(residual))


@pytest.mark.parametrize(
    ("dimension", "expected_channels"),
    [(2, 8), (3, 12)],
)
def test_feature_shape(dimension: int, expected_channels: int) -> None:
    case = CaseConfig(dimension=dimension, cells=5)
    features = coordinate_boundary_features(case, "cpu")
    assert features.shape == (1, expected_channels, *((5,) * dimension))


@pytest.mark.parametrize("architecture", ["cnn", "unet", "transformer", "grid_gnn", "residual_corrector"])
def test_2d_architecture_output_shape(architecture: str) -> None:
    case = CaseConfig(dimension=2, cells=6)
    optimization = OptimizationConfig(
        architecture=architecture,
        budget=1,
        width=8 if architecture != "transformer" else 16,
    )
    model = build_state_map(case, optimization)
    features = coordinate_boundary_features(case, "cpu")
    if architecture == "residual_corrector":
        state = model(features, lambda candidate: finite_volume_residual(candidate, case))
    else:
        state = model(features)
    assert state.shape == (1, 4, 6, 6)


def test_fixed_budget_retains_a_finite_state() -> None:
    case = CaseConfig(dimension=2, cells=6)
    optimization = OptimizationConfig(
        architecture="cnn",
        budget=3,
        learning_rate=1.0e-3,
        width=8,
        record_every=1,
    )
    result = optimize_state(case, optimization, device="cpu")
    assert result.state.shape == (1, 4, 6, 6)
    assert len(result.history) == 3
    assert 1 <= result.best_iteration <= optimization.budget
    assert math.isfinite(result.best_objective)
    assert result.parameters > 0


def test_physical_time_advancement_returns_one_retained_state_per_level() -> None:
    case = CaseConfig(dimension=2, cells=6)
    optimization = OptimizationConfig(
        architecture="cnn",
        budget=2,
        learning_rate=1.0e-3,
        width=8,
        record_every=1,
    )
    series = advance_physical_time(
        case,
        optimization,
        time_step=0.5,
        levels=2,
        device="cpu",
    )
    assert series.times == [0.0, 0.5, 1.0]
    assert len(series.states) == 3
    assert len(series.levels) == 2
    assert all(state.shape == (1, 4, 6, 6) for state in series.states)
