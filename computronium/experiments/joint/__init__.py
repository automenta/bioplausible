"""Joint Architecture Benchmark Experiments.

This package contains the 5 benchmark levels for the joint architecture:
- Level 1: Adaptation Efficiency
- Level 2: Compute Efficiency
- Level 3: Structural Robustness
- Level 3.5: Algorithm Migration
- Level 4: Z3 Fixed Weights

Each suite result carries a ``claims_scope`` audit status (see
``_claims.py``): L3.5 and L3 are ``plumbing_only`` (no ψ mediation in the
forward loop today); L1 and L2 are ``psi_wired_uncontrolled`` — ψ is stepped
inside forward and modulates computation (plasticity types empirically
differentiate), but θ trains concurrently and no frozen-θ control exists, so
they are suggestive rather than clean ψ evidence. Z3 is ``psi_engaged``
(R8 gate landed 2026-09-01): every ``evaluate_z3`` run embeds its own
engagement gate + planted-ψ control and downgrades itself to
``plumbing_only`` when the gate fails. Rewiring L1/L2 with frozen-θ phases +
``ThetaInvarianceAudit`` remains open and would upgrade their status.
"""

from computronium.experiments.joint._claims import (
    CLAIMS_SCOPE_PLUMBING_ONLY,
    CLAIMS_SCOPE_PSI_ENGAGED,
    CLAIMS_SCOPE_PSI_WIRED_UNCONTROLLED,
)

__all__ = [
    "CLAIMS_SCOPE_PLUMBING_ONLY",
    "CLAIMS_SCOPE_PSI_ENGAGED",
    "CLAIMS_SCOPE_PSI_WIRED_UNCONTROLLED",
    "adaptation_efficiency",  # ruff: ignore[undefined-export]
    "algorithm_migration",  # ruff: ignore[undefined-export]
    "compute_efficiency",  # ruff: ignore[undefined-export]
    "structural_robustness",  # ruff: ignore[undefined-export]
    "z3_fixed_weights",  # ruff: ignore[undefined-export]
]
