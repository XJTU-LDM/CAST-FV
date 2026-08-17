"""Two-level physical-time example with a fresh state map at every level."""

from castfv import CaseConfig, OptimizationConfig, advance_physical_time
from castfv.output import plot_state, save_time_series


case = CaseConfig(dimension=2, cells=16, reynolds=100.0, peclet=30.0)
optimization = OptimizationConfig(
    architecture="cnn",
    budget=20,
    learning_rate=1.0e-3,
    width=16,
    record_every=5,
)

series = advance_physical_time(
    case,
    optimization,
    time_step=1.0,
    levels=2,
    device="cpu",
    verbose=True,
)
output = save_time_series("outputs/unsteady", series, case, optimization, time_step=1.0)
plot_state(output / "last_retained_state.png", series.states[-1], case, "CAST-FV at t=2")

print("times:", series.times)
print("best objectives:", [level.best_objective for level in series.levels])
