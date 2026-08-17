"""Small, transparent output helpers for CAST-FV examples."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import CaseConfig, OptimizationConfig
from .optimize import OptimizationResult, TimeSeriesResult


def _state_dictionary(state: torch.Tensor, case: CaseConfig) -> dict[str, np.ndarray]:
    array = state.detach().cpu().numpy()[0]
    names = ("p", "u", "v", "T") if case.dimension == 2 else ("p", "u", "v", "w", "T")
    return {name: array[index] for index, name in enumerate(names)}


def _write_history(path: Path, history: list[dict[str, float | int]]) -> None:
    if not history:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def save_optimization_result(
    output_directory: str | Path,
    result: OptimizationResult,
    case: CaseConfig,
    optimization: OptimizationConfig,
) -> Path:
    """Save one retained state, sparse history and readable metadata."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "retained_state.npz", **_state_dictionary(result.state, case))
    _write_history(output / "optimization_history.csv", result.history)
    metadata = {
        "case": asdict(case),
        "optimization": asdict(optimization),
        "best_objective": result.best_objective,
        "best_iteration": result.best_iteration,
        "parameters": result.parameters,
        "information_condition": "equations, mesh coordinates, boundary values, and optional previous physical state",
        "solution_labels": 0,
        "offline_pretraining": False,
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def save_time_series(
    output_directory: str | Path,
    series: TimeSeriesResult,
    case: CaseConfig,
    optimization: OptimizationConfig,
    time_step: float,
) -> Path:
    """Save the initial state and every retained physical-time state."""

    output = Path(output_directory)
    states_directory = output / "states"
    histories_directory = output / "histories"
    states_directory.mkdir(parents=True, exist_ok=True)
    histories_directory.mkdir(parents=True, exist_ok=True)

    for index, (time, state) in enumerate(zip(series.times, series.states, strict=True)):
        np.savez_compressed(
            states_directory / f"state_{index:04d}_t_{time:.6f}.npz",
            **_state_dictionary(state, case),
        )
    for level, result in enumerate(series.levels, start=1):
        _write_history(histories_directory / f"level_{level:04d}.csv", result.history)

    metadata = {
        "case": asdict(case),
        "optimization": asdict(optimization),
        "time_step": time_step,
        "times": series.times,
        "best_objectives": [result.best_objective for result in series.levels],
        "best_iterations": [result.best_iteration for result in series.levels],
        "fresh_state_map_per_level": True,
        "time_scheme": "backward Euler",
        "solution_labels": 0,
        "offline_pretraining": False,
    }
    (output / "series_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def plot_state(path: str | Path, state: torch.Tensor, case: CaseConfig, title: str) -> Path:
    """Plot speed and passive scalar for a 2D field or a 3D mid-plane."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    values = _state_dictionary(state, case)
    if case.dimension == 2:
        speed = np.sqrt(values["u"] ** 2 + values["v"] ** 2)
        scalar = values["T"]
        vertical_label = "$y$"
    else:
        midpoint = case.cells // 2
        speed = np.sqrt(values["u"] ** 2 + values["v"] ** 2 + values["w"] ** 2)[:, midpoint, :]
        scalar = values["T"][:, midpoint, :]
        vertical_label = "$z$"

    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
            "axes.linewidth": 0.8,
        }
    ):
        figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.45), constrained_layout=True)
        for axis, quantity, label, colormap in (
            (axes[0], speed, "Velocity magnitude", "viridis"),
            (axes[1], scalar, "Passive scalar", "inferno"),
        ):
            image = axis.imshow(
                quantity,
                origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0),
                aspect="equal",
                cmap=colormap,
            )
            axis.set_xlabel("$x$")
            axis.set_ylabel(vertical_label)
            axis.set_title(label)
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure.suptitle(title)
        figure.savefig(output, dpi=180, bbox_inches="tight")
        plt.close(figure)
    return output
