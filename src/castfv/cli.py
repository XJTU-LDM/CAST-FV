"""Command-line interface for the minimal CAST-FV release."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .config import CaseConfig, OptimizationConfig
from .optimize import advance_physical_time, optimize_state
from .output import plot_state, save_optimization_result, save_time_series


ARCHITECTURES = ("cnn", "unet", "transformer", "grid_gnn", "residual_corrector")


def _add_case_arguments(parser: argparse.ArgumentParser, *, dimension: bool = True) -> None:
    if dimension:
        parser.add_argument("--dimension", type=int, choices=(2, 3), default=2)
    parser.add_argument("--cells", type=int, default=48)
    parser.add_argument("--reynolds", type=float, default=100.0)
    parser.add_argument("--peclet", type=float, default=30.0)


def _add_optimization_arguments(parser: argparse.ArgumentParser, *, architecture: bool = True) -> None:
    if architecture:
        parser.add_argument("--architecture", choices=ARCHITECTURES, default="cnn")
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--corrector-steps", type=int, default=4)
    parser.add_argument("--record-every", type=int, default=10)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--verbose", action="store_true")


def _case_from(arguments: argparse.Namespace, dimension: int | None = None) -> CaseConfig:
    return CaseConfig(
        dimension=dimension or arguments.dimension,
        cells=arguments.cells,
        reynolds=arguments.reynolds,
        peclet=arguments.peclet,
    )


def _optimization_from(
    arguments: argparse.Namespace,
    architecture: str | None = None,
    seed: int | None = None,
) -> OptimizationConfig:
    return OptimizationConfig(
        architecture=architecture or arguments.architecture,
        budget=arguments.budget,
        learning_rate=arguments.learning_rate,
        seed=arguments.seed if seed is None else seed,
        width=arguments.width,
        corrector_steps=arguments.corrector_steps,
        record_every=arguments.record_every,
    )


def _device_from(arguments: argparse.Namespace) -> str | None:
    return None if arguments.device == "auto" else arguments.device


def _print_summary(summary: dict[str, object]) -> None:
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def run_steady(arguments: argparse.Namespace) -> None:
    case = _case_from(arguments)
    optimization = _optimization_from(arguments)
    output = Path(arguments.output)
    result = optimize_state(
        case,
        optimization,
        device=_device_from(arguments),
        verbose=arguments.verbose,
    )
    save_optimization_result(output, result, case, optimization)
    plot_state(output / "retained_state.png", result.state, case, "CAST-FV retained steady state")
    _print_summary(
        {
            "architecture": result.architecture,
            "best_iteration": result.best_iteration,
            "best_objective": result.best_objective,
            "parameters": result.parameters,
            "output": str(output.resolve()),
        }
    )


def run_unsteady(arguments: argparse.Namespace) -> None:
    case = _case_from(arguments)
    optimization = _optimization_from(arguments)
    output = Path(arguments.output)
    series = advance_physical_time(
        case,
        optimization,
        time_step=arguments.time_step,
        levels=arguments.levels,
        device=_device_from(arguments),
        verbose=arguments.verbose,
    )
    save_time_series(output, series, case, optimization, arguments.time_step)
    plot_state(
        output / "last_retained_state.png",
        series.states[-1],
        case,
        f"CAST-FV retained state at t={series.times[-1]:g}",
    )
    _print_summary(
        {
            "architecture": optimization.architecture,
            "levels": arguments.levels,
            "times": series.times,
            "best_objectives": [level.best_objective for level in series.levels],
            "fresh_state_map_per_level": True,
            "output": str(output.resolve()),
        }
    )


def run_compare(arguments: argparse.Namespace) -> None:
    case = _case_from(arguments, dimension=2)
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for index, architecture in enumerate(arguments.architectures):
        optimization = _optimization_from(
            arguments,
            architecture=architecture,
            seed=arguments.seed + index,
        )
        result = optimize_state(
            case,
            optimization,
            device=_device_from(arguments),
            verbose=arguments.verbose,
        )
        architecture_output = output / architecture
        save_optimization_result(architecture_output, result, case, optimization)
        plot_state(
            architecture_output / "retained_state.png",
            result.state,
            case,
            f"CAST-FV {architecture}",
        )
        rows.append(
            {
                "architecture": architecture,
                "best_objective": result.best_objective,
                "best_iteration": result.best_iteration,
                "parameters": result.parameters,
            }
        )

    with (output / "architecture_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _print_summary({"comparison": rows, "output": str(output.resolve())})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="castfv",
        description="Solution-label-free finite-volume neural state optimization.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    steady = subparsers.add_parser("steady", help="construct one steady state")
    _add_case_arguments(steady)
    _add_optimization_arguments(steady)
    steady.add_argument("--output", default="outputs/steady")
    steady.set_defaults(function=run_steady)

    unsteady = subparsers.add_parser("unsteady", help="advance through physical time levels")
    _add_case_arguments(unsteady)
    _add_optimization_arguments(unsteady)
    unsteady.add_argument("--time-step", type=float, default=1.0)
    unsteady.add_argument("--levels", type=int, default=3)
    unsteady.add_argument("--output", default="outputs/unsteady")
    unsteady.set_defaults(function=run_unsteady)

    compare = subparsers.add_parser("compare", help="compare 2D state parameterizations")
    _add_case_arguments(compare, dimension=False)
    _add_optimization_arguments(compare, architecture=False)
    compare.add_argument("--architectures", nargs="+", choices=ARCHITECTURES, default=list(ARCHITECTURES))
    compare.add_argument("--output", default="outputs/architecture_comparison")
    compare.set_defaults(function=run_compare)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    arguments.function(arguments)
