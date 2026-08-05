"""Unit tests for campaign orchestration and gating (FIX2a §2, §8)."""

from __future__ import annotations

from bioplausible.campaign import tiers
from bioplausible.campaign.runner import CampaignRunner, run_gates
from bioplausible.campaign.schema import validate_yaml

MINIMAL = validate_yaml(
    "meta: {name: gate, created: '2026-08-05'}\n"
    "arms:\n"
    "  mlp:\n"
    "    input_dim: 64\n"
    "    num_classes: 10\n"
    "    max_params: 210000\n"
    "    models: [backprop_mlp, eqprop_mlp]\n"
    "tasks:\n"
    "  - name: digits\n"
    "    epochs: 5\n"
    "    input_dim: 64\n"
    "    num_classes: 10\n"
    "reproducibility:\n"
    "  seed: 1\n"
)


def test_plan_resolves_geometry():
    plans = CampaignRunner(MINIMAL).plan()
    assert len(plans) == 1
    (arm,) = plans
    assert arm.name == "mlp"
    assert arm.models == ("backprop_mlp", "eqprop_mlp")
    assert arm.input_dim == 64
    assert arm.output_dim == 10


def test_dry_run_mentions_models():
    text = CampaignRunner(MINIMAL).dry_run()
    assert "backprop_mlp" in text
    assert "eqprop_mlp" in text


def test_run_gates_excludes_failed_models(monkeypatch):
    def fake_tier0(models, _settings, **_kwargs):
        return [
            tiers.TierOutcome(
                tier="tier0",
                model=m,
                task="xor,spiral,circles",
                passed=m == "backprop_mlp",
                reason="r",
            )
            for m in models
        ]

    def fake_tier05(models, _settings, **_kwargs):
        return [
            tiers.TierOutcome(
                tier="tier0.5",
                model=m,
                task="digits",
                passed=True,
                reason="pass",
                metrics={"mean_acc": 0.99},
            )
            for m in models
        ]

    monkeypatch.setattr("bioplausible.campaign.tiers.run_tier0", fake_tier0)
    monkeypatch.setattr("bioplausible.campaign.tiers.run_tier05", fake_tier05)

    result = run_gates(MINIMAL, n_seeds=3)
    tier0_models = {o.model for o in result.tiers["tier0"]}
    tier05_models = {o.model for o in result.tiers["tier0.5"]}
    assert tier0_models == {"backprop_mlp", "eqprop_mlp"}
    assert tier05_models == {"backprop_mlp"}
