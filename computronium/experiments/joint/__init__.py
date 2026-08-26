"""Joint Architecture Benchmark Experiments.

This package contains the 5 benchmark levels for the joint architecture:
- Level 1: Adaptation Efficiency
- Level 2: Compute Efficiency
- Level 3: Structural Robustness
- Level 3.5: Algorithm Migration
- Level 4: Z3 Fixed Weights

The L1/L2/L3/L3.5 shakedown suites are INSTRUMENTATION SHELLS: their toy
models bypass the ψ/θ split (``forward()`` ignores plasticity; training
updates θ freely), so they validate plumbing and metrics wiring only. They
are not citable as ψ-mediated evidence; real-data claims route through the
Z3 path (TODO4 session-log finding, upheld after review).
"""

from computronium.experiments.joint._claims import SUITE_CLAIMS_SCOPE

__all__ = [
    "SUITE_CLAIMS_SCOPE",
    "adaptation_efficiency",
    "algorithm_migration",
    "compute_efficiency",
    "structural_robustness",
    "z3_fixed_weights",
]
