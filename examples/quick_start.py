"""Minimal public API example; no CFD labels or reference trajectories are used."""

from castfv import CaseConfig, OptimizationConfig, optimize_state
from castfv.output import plot_state, save_optimization_result


case = CaseConfig(dimension=2, cells=16, reynolds=100.0, peclet=30.0)
optimization = OptimizationConfig(
    architecture="cnn",
    budget=25,
    learning_rate=1.0e-3,
    width=16,
    record_every=5,
)

result = optimize_state(case, optimization, device="cpu", verbose=True)
output = save_optimization_result("outputs/quick_start", result, case, optimization)
plot_state(output / "retained_state.png", result.state, case, "CAST-FV quick start")

print(f"best objective: {result.best_objective:.6e}")
print(f"retained iteration: {result.best_iteration}")
print(f"trainable parameters: {result.parameters}")
