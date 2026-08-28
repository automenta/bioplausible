"""computronium-stability v0.1.0

Calibrated stability guard for dynamical neural systems.

Calibrated on:
- Settling/energy-based dynamics (energy minimization,
  predictive settling)
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

from computronium_stability.basin import (
    BasinStabilityEstimator,
    estimate_basin_stability,
    estimate_basin_stability_multistart,
)
from computronium_stability.guard import (
    DEFAULT_TAU,
    GuardDecision,
    GuardHandle,
    StabilityGuard,
    StabilityVerdict,
    attach,
)
from computronium_stability.lyapunov import (
    LyapunovEstimator,
    estimate_lyapunov_exponent,
    estimate_lyapunov_spectrum,
)
from computronium_stability.settling import (
    SettlingMonitor,
    measure_settling_time,
    measure_settling_time_full_state,
)
from computronium_stability.spectral_radius import (
    SpectralRadiusEstimator,
    estimate_spectral_radius,
)

__version__ = "0.1.0"

__all__ = [
    # Guard API (primary)
    "attach",
    "StabilityGuard",
    "StabilityVerdict",
    "GuardHandle",
    "GuardDecision",
    "DEFAULT_TAU",
    # Spectral radius
    "SpectralRadiusEstimator",
    "estimate_spectral_radius",
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
]
