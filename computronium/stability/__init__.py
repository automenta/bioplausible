"""computronium-stability — Calibrated stability guard for dynamical neural systems.

Calibrated on:
- Settling/energy-based dynamics (energy minimization, predictive settling)
- Non-normal linear dynamics (Ginibre ensemble)
- 16 real substrate × settling-dynamics coordinates
  (windowed growth = 1.000, FKR 0% at τ=1.029)

Scope statement (mandatory v1):
This guard is calibrated for energy-minimization
coordinates and non-normal linear dynamics.
General-transformer collapse detection is future calibration
work, not a v1 claim.

Quick start:
    import torch
    from computronium_stability import attach, StabilityVerdict

    model = torch.nn.Linear(10, 10)
    guard = attach(model)

    for step in range(100):
        x = torch.randn(32, 10)
        y = model(x)
        verdict = guard.check({"x": x, "y": y, "loss": y.pow(2).mean()})
        if verdict.kill:
            print(f"Killed at step {step}: {verdict}")
            break
"""

from computronium.stability.basin import (
    BasinStabilityEstimator,
    estimate_basin_stability,
    estimate_basin_stability_multistart,
)
from computronium.stability.calibration import (
    DEMO_GOOD_COORDINATES,
    DISAGREEMENT_COORDINATES,
    OVERHEAD_BUDGET,
    PR5Calibration,
    calibrate_demo_harvest,
    ginibre_run,
    harvest_bad_statistics,
    harvest_good_statistics,
    probe_interval_for_overhead,
    unrolled_divergence,
)
from computronium.stability.config import (
    BasinConfig,
    GuardConfig,
    LyapunovConfig,
    SettlingConfig,
    SpectralRadiusConfig,
    create_basin_estimator,
    create_guard,
    create_lyapunov_estimator,
    create_settling_monitor,
    create_spectral_radius_estimator,
)
from computronium.stability.frontier import (
    FrontierAggregator,
    FrontierRecord,
)
from computronium.stability.guard import (
    DEFAULT_TAU,
    ExternalTransitionFn,
    GuardDecision,
    GuardHandle,
    StabilityGuard,
    StabilityVerdict,
    StatisticKind,
    StepState,
    attach,
    calibrate_threshold,
    measure_guard_overhead,
    quantify_proxy_disagreement,
)
from computronium.stability.lyapunov import (
    LyapunovEstimator,
    estimate_lyapunov_exponent,
    estimate_lyapunov_spectrum,
)
from computronium.stability.resources import ResourceUsage
from computronium.stability.settling import (
    SettlingMonitor,
    measure_settling_time,
    measure_settling_time_full_state,
)
from computronium.stability.spectral_radius import (
    SpectralRadiusEstimator,
    estimate_spectral_radius,
    estimate_spectral_radius_full_jacobian,
)

__version__ = "0.1.0"

__all__ = [  # ruff: ignore[unsorted-dunder-all]
    # Guard API (primary)
    "attach",
    "StabilityGuard",
    "StabilityVerdict",
    "GuardHandle",
    "GuardDecision",
    "DEFAULT_TAU",
    "calibrate_threshold",
    "quantify_proxy_disagreement",
    "measure_guard_overhead",
    # Spectral radius
    "SpectralRadiusEstimator",
    "estimate_spectral_radius",
    "estimate_spectral_radius_full_jacobian",
    # Lyapunov
    "LyapunovEstimator",
    "estimate_lyapunov_exponent",
    "estimate_lyapunov_spectrum",
    # Settling
    "SettlingMonitor",
    "measure_settling_time",
    "measure_settling_time_full_state",
    # Basin stability
    "BasinStabilityEstimator",
    "estimate_basin_stability",
    "estimate_basin_stability_multistart",
    # Frontier
    "FrontierRecord",
    "FrontierAggregator",
    # Resources
    "ResourceUsage",
    # Config + Factories
    "SpectralRadiusConfig",
    "LyapunovConfig",
    "SettlingConfig",
    "BasinConfig",
    "GuardConfig",
    "create_spectral_radius_estimator",
    "create_lyapunov_estimator",
    "create_settling_monitor",
    "create_basin_estimator",
    "create_guard",
    # Type aliases
    "StepState",
    "ExternalTransitionFn",
    "StatisticKind",
    # PR-5 demo-harvest calibration
    "PR5Calibration",
    "calibrate_demo_harvest",
    "harvest_good_statistics",
    "harvest_bad_statistics",
    "probe_interval_for_overhead",
    "unrolled_divergence",
    "ginibre_run",
    "DEMO_GOOD_COORDINATES",
    "DISAGREEMENT_COORDINATES",
    "OVERHEAD_BUDGET",
]
