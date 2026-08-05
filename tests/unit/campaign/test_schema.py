"""Unit tests for the Pydantic v2 campaign schema (FIX2a §3)."""

from __future__ import annotations

import pytest

from bioplausible.campaign.schema import load_campaign, validate_yaml

MINIMAL_CAMPAIGN = """
meta:
  name: "minimal"
  created: "2026-08-05"
arms:
  mlp:
    input_dim: 64
    num_classes: 10
    max_params: 210000
    models:
      - backprop_mlp
tasks:
  - name: digits
    epochs: 5
    input_dim: 64
    num_classes: 10
"""

FULL_CAMPAIGN = """
meta:
  name: "gate"
  description: "triage"
  created: "2026-08-05"
compute:
  device: "auto"
  max_parallel: 2
search_space:
  base:
    hidden_dim: [16, 32, 64, 128]
    num_layers: [1, 2, 4]
    lr: [1e-4, 1e-2, "log"]
    optimizer: adam
model_overrides:
  eqprop_mlp:
    beta: [0.01, 0.5]
    gradient_method: equilibrium
  neural_cube:
    cube_size: [3, 4, 5]
constraints:
  - "estimate(config) <= 210000"
protocols:
  default: end2end
  overrides:
    standard_fa: layerwise
arms:
  mlp:
    input_dim: 64
    num_classes: 10
    max_params: 210000
    models: [backprop_mlp, eqprop_mlp, neural_cube, standard_fa]
hpo:
  n_trials: 20
  n_seeds: 3
reproducibility:
  seed: 7
"""


def test_minimal_campaign_validates():
    campaign = validate_yaml(MINIMAL_CAMPAIGN)
    assert campaign.meta.name == "minimal"
    assert campaign.arms["mlp"].models == ["backprop_mlp"]
    assert campaign.protocols.resolve("backprop_mlp") == "end2end"


def test_full_campaign_build_search_space():
    campaign = validate_yaml(FULL_CAMPAIGN)
    space = campaign.build_search_space()
    assert set(space.base) == {"hidden_dim", "num_layers", "lr"}
    assert space.defaults == {"optimizer": "adam"}
    assert "beta" in space.for_model("eqprop_mlp")
    assert space.constants["eqprop_mlp"] == {"gradient_method": "equilibrium"}
    assert "beta" not in space.for_model("backprop_mlp")


def test_protocol_resolution():
    campaign = validate_yaml(FULL_CAMPAIGN)
    assert campaign.protocols.resolve("standard_fa") == "layerwise"
    assert campaign.protocols.resolve("backprop_mlp") == "end2end"


def test_arm_dimension_resolution():
    campaign = validate_yaml(MINIMAL_CAMPAIGN)
    assert campaign.arm_input_dim("mlp") == 64
    assert campaign.arm_output_dim("mlp") == 10


def test_arm_input_shape_product():
    campaign = validate_yaml(
        MINIMAL_CAMPAIGN.replace(
            "  mlp:\n    input_dim: 64\n", "  mlp:\n    input_shape: [3, 32, 32]\n"
        )
    )
    assert campaign.arm_input_dim("mlp") == 3 * 32 * 32


def test_missing_arms_raises():
    with pytest.raises(Exception):
        validate_yaml("meta: {name: 'x'}\n")


def test_empty_yaml_raises():
    with pytest.raises(ValueError):
        validate_yaml("")


def test_extra_keys_forbidden():
    bad = MINIMAL_CAMPAIGN + "unknown_section:\n  a: 1\n"
    with pytest.raises(Exception):
        validate_yaml(bad)


def test_load_campaign_file(tmp_path):
    path = tmp_path / "campaign.yaml"
    path.write_text(MINIMAL_CAMPAIGN, encoding="utf-8")
    campaign = load_campaign(path)
    assert campaign.meta.name == "minimal"
