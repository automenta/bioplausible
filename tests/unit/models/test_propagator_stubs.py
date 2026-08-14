"""
Tests for the propagator-to-model compatibility alias map in the Registry.

When a propagator name that requires model-level control is queried,
``Registry.get()`` resolves it to the model-side implementation via the
``_ALIASES`` compatibility map (a lookup, not an error). Genuine propagators
still resolve to their registered classes.
"""

import pytest
import torch

from bioplausible.core.registry import ComponentCategory, Registry
from bioplausible.zoo.models.forward_only import PEPITA, ForwardForwardNet
from bioplausible.zoo.models.target_prop import DifferenceTargetProp


class TestPropagatorCrossReference:
    """Model-side-only algorithms resolve to the model category via aliases."""

    def test_ff_cross_reference(self):
        """``ff`` resolves to the ForwardForwardNet model class."""
        result = Registry.get(ComponentCategory.PROPAGATOR, "ff")
        assert result is ForwardForwardNet

    def test_pepita_cross_reference(self):
        """``pepita`` resolves to the PEPITA model class."""
        result = Registry.get(ComponentCategory.PROPAGATOR, "pepita")
        assert result is PEPITA

    def test_target_prop_cross_reference(self):
        """``target_prop`` resolves to the DifferenceTargetProp model class."""
        result = Registry.get(ComponentCategory.PROPAGATOR, "target_prop")
        assert result is DifferenceTargetProp

    def test_difference_target_prop_cross_reference(self):
        """``difference_target_prop`` resolves to the same model class."""
        result = Registry.get(ComponentCategory.PROPAGATOR, "difference_target_prop")
        assert result is DifferenceTargetProp

    def test_predictive_coding_cross_reference(self):
        """``predictive_coding`` resolves to the PredictiveCodingHybrid model class."""
        from bioplausible.zoo.models.predictive_coding import PredictiveCodingHybrid

        result = Registry.get(ComponentCategory.PROPAGATOR, "predictive_coding")
        assert result is PredictiveCodingHybrid

    def test_unknown_propagator_still_raises(self):
        """An unknown propagator (no alias, no registration) still raises."""
        with pytest.raises(ValueError) as exc:
            Registry.get(ComponentCategory.PROPAGATOR, "nonexistent_propagator")
        assert "Available" in str(exc.value)

    def test_working_propagators_still_resolve(self):
        """Working propagators that ARE registered still resolve normally."""
        from bioplausible.core.local_learning.rules.eqprop import EqProp

        result = Registry.get(ComponentCategory.PROPAGATOR, "eq_prop")
        assert result is EqProp

    def test_aliases_returns_compatibility_map(self):
        """``Registry.aliases()`` returns the full alias map."""
        aliases = Registry.aliases()
        assert aliases["ff"] == (ComponentCategory.MODEL, "forward_forward")
        assert aliases["pepita"] == (ComponentCategory.MODEL, "pepita")

    def test_resolve_alias_returns_canonical(self):
        """``resolve_alias`` returns the canonical (category, name)."""
        cat, name = Registry.resolve_alias(ComponentCategory.PROPAGATOR, "ff")
        assert cat == ComponentCategory.MODEL
        assert name == "forward_forward"

    def test_resolve_alias_passthrough_for_real_name(self):
        """``resolve_alias`` returns the name unchanged when it is not an alias."""
        cat, name = Registry.resolve_alias(ComponentCategory.MODEL, "eqprop")
        assert cat == ComponentCategory.MODEL
        assert name == "eqprop"

    def test_get_metadata_resolves_alias(self):
        """``get_metadata`` transparently resolves alias names to the model."""
        meta = Registry.get_metadata(ComponentCategory.PROPAGATOR, "ff")
        assert meta.category == ComponentCategory.MODEL
        assert meta.name == "forward_forward"


def test_pepita_spatial_input_flatten():
    """PEPITA must accept [B, C, H, W] input (demo/CoreTrainer path).

    Regression for the demo failure recorded in TODO: PEPITA's train_step did
    not flatten image inputs before the feedback_matrix projection.
    """
    model = PEPITA(input_dim=64, hidden_dim=16, output_dim=10, num_layers=2)
    x = torch.randn(4, 1, 8, 8)
    y = torch.randint(0, 10, (4,))
    out = model.forward(x)
    assert out.shape == (4, 10)
    result = model.train_step(x, y)
    assert "loss" in result
    assert "accuracy" in result


def test_forward_forward_spatial_input_flatten():
    """ForwardForwardNet must accept [B, C, H, W] input (demo/CoreTrainer path)."""
    model = ForwardForwardNet(input_dim=64, hidden_dim=16, output_dim=10, num_layers=2)
    x = torch.randn(4, 1, 8, 8)
    y = torch.randint(0, 10, (4,))
    out = model.forward(x)
    assert out.shape == (4, 10)
    result = model.train_step(x, y)
    assert "loss" in result
    assert "accuracy" in result
