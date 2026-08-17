"""Finite-budget CAST-FV state construction and physical-time advancement."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn as nn

from .config import CaseConfig, OptimizationConfig
from .features import coordinate_boundary_features
from .models import ResidualCorrector2D, build_state_map, parameter_count
from .residuals import finite_volume_objective, finite_volume_residual, zero_state


@dataclass
class OptimizationResult:
    """Lowest-objective state retained from one fixed-time optimization."""

    state: torch.Tensor
    history: list[dict[str, float | int]]
    best_objective: float
    best_iteration: int
    parameters: int
    architecture: str


@dataclass
class TimeSeriesResult:
    """Sequence of independently parameterized states at physical time levels."""

    times: list[float]
    states: list[torch.Tensor]
    levels: list[OptimizationResult]


def resolve_device(requested: str | torch.device | None = None) -> torch.device:
    if requested is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def _candidate_state(
    model: nn.Module,
    features: torch.Tensor,
    case: CaseConfig,
    previous_state: torch.Tensor | None,
    time_step: float | None,
) -> torch.Tensor:
    if isinstance(model, ResidualCorrector2D):
        return model(
            features,
            lambda candidate: finite_volume_residual(
                candidate,
                case,
                previous_state=previous_state,
                time_step=time_step,
            ),
        )
    return model(features)


def optimize_state(
    case: CaseConfig,
    optimization: OptimizationConfig,
    *,
    previous_state: torch.Tensor | None = None,
    time_step: float | None = None,
    device: str | torch.device | None = None,
    verbose: bool = False,
) -> OptimizationResult:
    """Construct one state using only finite-volume residual gradients.

    The optimizer remains at one physical time for exactly ``budget`` updates.
    Every finite candidate is eligible for the retained checkpoint. No target
    field, CFD trajectory or pretrained state-map weight is accepted.
    """

    run_device = resolve_device(device)
    torch.manual_seed(optimization.seed)
    np.random.seed(optimization.seed)
    if run_device.type == "cuda":
        torch.cuda.manual_seed_all(optimization.seed)

    features = coordinate_boundary_features(case, run_device)
    history_state = None
    if previous_state is not None:
        history_state = previous_state.detach().to(device=run_device, dtype=features.dtype)
        if time_step is None or time_step <= 0.0:
            raise ValueError("a positive time_step is required for an unsteady state")

    model = build_state_map(case, optimization).to(run_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimization.learning_rate,
        weight_decay=optimization.weight_decay,
    )
    best_objective = float("inf")
    best_iteration = 0
    best_state: torch.Tensor | None = None
    history: list[dict[str, float | int]] = []

    for iteration in range(1, optimization.budget + 1):
        optimizer.zero_grad(set_to_none=True)
        state = _candidate_state(model, features, case, history_state, time_step)
        objective, metrics = finite_volume_objective(
            state,
            case,
            previous_state=history_state,
            time_step=time_step,
        )
        objective_value = float(objective.detach().cpu())
        if not np.isfinite(objective_value):
            raise FloatingPointError(f"non-finite objective at iteration {iteration}")
        if objective_value < best_objective:
            best_objective = objective_value
            best_iteration = iteration
            best_state = state.detach().cpu().clone()

        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), optimization.gradient_clip)
        optimizer.step()

        should_record = (
            iteration == 1
            or iteration == optimization.budget
            or iteration % optimization.record_every == 0
        )
        if should_record:
            row: dict[str, float | int] = {"iteration": iteration, **metrics}
            history.append(row)
            if verbose:
                print(
                    f"iteration={iteration:05d} objective={objective_value:.6e} "
                    f"continuity={metrics['continuity']:.3e}",
                    flush=True,
                )

    if best_state is None:
        raise RuntimeError("the optimization budget produced no finite candidate")
    return OptimizationResult(
        state=best_state,
        history=history,
        best_objective=best_objective,
        best_iteration=best_iteration,
        parameters=parameter_count(model),
        architecture=optimization.architecture,
    )


def advance_physical_time(
    case: CaseConfig,
    optimization: OptimizationConfig,
    *,
    time_step: float,
    levels: int,
    device: str | torch.device | None = None,
    verbose: bool = False,
) -> TimeSeriesResult:
    """Advance by repeating fresh, fixed-time residual minimizations.

    Only the retained state from the previous level enters the backward-Euler
    accumulation term. A new neural state map and AdamW optimizer are created at
    every level, so no pretrained or cross-case operator is formed.
    """

    if time_step <= 0.0:
        raise ValueError("time_step must be positive")
    if levels < 1:
        raise ValueError("levels must be positive")

    previous = zero_state(case)
    times = [0.0]
    states = [previous.clone()]
    level_results: list[OptimizationResult] = []

    for level in range(1, levels + 1):
        if verbose:
            print(f"\nphysical level {level}/{levels}: t={level * time_step:g}", flush=True)
        level_optimization = replace(optimization, seed=optimization.seed + level - 1)
        result = optimize_state(
            case,
            level_optimization,
            previous_state=previous,
            time_step=time_step,
            device=device,
            verbose=verbose,
        )
        previous = result.state.detach().cpu()
        level_results.append(result)
        times.append(level * time_step)
        states.append(previous.clone())

    return TimeSeriesResult(times=times, states=states, levels=level_results)
