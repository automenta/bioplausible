"""Study progress snapshots shared by the runner, dashboard, and CLI.

``StudyStats`` aggregates the state of one Optuna study at a point in time. It is
defined here (not in a root script) so the TUI dashboard can type-check against
it without a root-import dependency, and the runner/CLI can reuse the enrichment
helpers when polling studies.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["StudyStats", "estimate_param_count", "infer_arch_group"]

_CONV_INDICATORS = ("conv", "graph", "spatial")

_PARAM_ESTIMATES = {
    "modern_conv_eqprop": 242_250,
    "conv_eqprop": 37_674,
    "graph_eqprop": 3_466,
    "backprop_mlp": 2_410,
    "eqprop_mlp": 3_466,
    "lazy_eqprop": 4_522,
    "neural_cube": 6_538,
    "holomorphic_ep": 3_466,
    "forward_forward": 3_786,
    "pepita": 4_106,
    "hebbian_chain": 4_458,
    "deep_hebbian": 4_458,
    "diff_target_prop": 6_634,
}


def infer_arch_group(model_name: str) -> str:
    """Return the architecture group ('conv' or 'mlp') for a model name."""
    return (
        "conv" if any(ind in model_name.lower() for ind in _CONV_INDICATORS) else "mlp"
    )


def estimate_param_count(model_name: str) -> int:
    """Return a rough static parameter-count estimate for a known model.

    These are approximations for dashboard display; exact values come from
    ``trial.user_attrs`` when a run writes them.
    """
    return _PARAM_ESTIMATES.get(model_name, 0)


@dataclass
class StudyStats:
    """Snapshot of a study's progress at a moment in time."""

    study_name: str
    family: str
    model: str
    task: str
    budget: int
    complete: int
    total: int
    best_acc: float | None
    last_trial_time_s: float | None
    model_type: str = ""
    locality: str = ""
    arch_group: str = ""  # "conv" or "mlp"
    param_count: int = 0
