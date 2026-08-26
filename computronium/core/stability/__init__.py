"""Stability-Plasticity Frontier Metrics.

Provides stability monitors, resource accounting, and frontier record
aggregation for the joint architecture.
"""

from computronium.core.profiling import ResourceUsage
from computronium.core.stability.basin import (
    BasinStabilityEstimator,
    estimate_basin_stability,
    estimate_basin_stability_multistart,
)
from computronium.core.stability.guard import (
    CalibrationReport,
    DisagreementReport,
    GuardDecision,
    StabilityGuard,
    calibrate_threshold,
    measure_guard_overhead,
    quantify_proxy_disagreement,
)
from computronium.core.stability.frontier import (
    FrontierAggregator,
    FrontierRecord,
)
from computronium.core.stability.lyapunov import (
    LyapunovEstimator,
    estimate_lyapunov_exponent,
    estimate_lyapunov_spectrum,
)
from computronium.core.stability.settling import (
    SettlingMonitor,
    measure_settling_time,
    measure_settling_time_full_state,
)
from computronium.core.stability.spectral_radius import (
    SpectralRadiusEstimator,
    estimate_spectral_radius,
    estimate_spectral_radius_full_jacobian,
)

__all__ = [
    # Core types
    "ResourceUsage",
    "FrontierRecord",
    "FrontierAggregator",
    # Spectral radius
    "estimate_spectral_radius",
    "SpectralRadiusEstimator",
    "estimate_spectral_radius_full_jacobian",
    # Guard (PR-5)
    "StabilityGuard",
    "GuardDecision",
    "CalibrationReport",
    "DisagreementReport",
    "calibrate_threshold",
    "quantify_proxy_disagreement",
    "measure_guard_overhead",
    # Lyapunov
    "estimate_lyapunov_exponent",
    "LyapunovEstimator",
    "estimate_lyapunov_spectrum",
    # Settling
    "measure_settling_time",
    "SettlingMonitor",
    "measure_settling_time_full_state",
    # Basin stability
    "estimate_basin_stability",
    "BasinStabilityEstimator",
    "estimate_basin_stability_multistart",
]
