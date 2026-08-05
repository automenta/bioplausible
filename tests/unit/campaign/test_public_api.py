"""Public API surface and end-to-end estimator wiring (FIX2a §4.2 §4.3)."""

from __future__ import annotations

import bioplausible.zoo  # ruff: ignore[unused-import]  # triggers model registration
from bioplausible import campaign


def test_package_exports_framework_names():
    for name in (
        "Arm",
        "ArmPlan",
        "Campaign",
        "CampaignResult",
        "CampaignRunner",
        "Choice",
        "ExperimentLogger",
        "GateSettings",
        "ParamDistribution",
        "SearchSpace",
        "bound_estimator",
        "estimate_param_count",
        "parse_distribution",
        "run_gates",
        "run_tier0",
        "run_tier05",
        "validate_yaml",
    ):
        assert hasattr(campaign, name), name


def test_bound_estimator_is_exported_and_callable():
    est = campaign.bound_estimator("backprop_mlp", input_dim=64, output_dim=10)
    # forward-only FFLayers are weightless, so num_layers=1 is just Linear(64,10).
    assert est({"hidden_dim": 64, "num_layers": 1}) == 650


def test_gate_settings_is_exported():
    assert campaign.GateSettings(input_dim=2, output_dim=2).input_dim == 2
