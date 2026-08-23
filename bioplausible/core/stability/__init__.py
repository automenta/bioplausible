"""Stability-Plasticity Frontier Metrics.

Provides stability monitors, resource accounting, and frontier record
aggregation for the joint architecture.
"""

from bioplausible.core.stability.frontier import (
    ResourceUsage,
    FrontierRecord,
    FrontierAggregator,
)
from bioplausible.core.stability.spectral_radius import (
    estimate_spectral_radius,
    SpectralRadiusEstimator,
    estimate_spectral_radius_full_jacobian,
)
from bioplausible.core.stability.lyapunov import (
    estimate_lyapunov_exponent,
    LyapunovEstimator,
    estimate_lyapunov_spectrum,
)
from bioplausible.core.stability.settling import (
    measure_settling_time,
    SettlingMonitor,
    measure_settling_time_full_state,
)
from bioplausible.core.stability.basin import (
    estimate_basin_stability,
    BasinStabilityEstimator,
    estimate_basin_stability_multistart,
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