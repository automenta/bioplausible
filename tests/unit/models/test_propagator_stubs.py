"""
Tests for the propagator-to-model cross-reference in the Registry.

When a propagator name that requires model-level control is queried,
Registry.get() raises ValueError with a cross-reference to the model-side
implementation.
"""

import pytest
import torch

from bioplausible.core.registry import ComponentCategory, Registry
from bioplausible.zoo.models.forward_only import PEPITA, ForwardForwardNet


class TestPropagatorCrossReference:
    """Model-side-only algorithms redirect to model category with helpful message."""

    def test_ff_cross_reference(self):
        with pytest.raises(ValueError) as exc:
            Registry.get(ComponentCategory.PROPAGATOR, "ff")
        msg = str(exc.value)
        assert "forward_forward" in msg
        assert "bioplausible.zoo.models.forward_only.ForwardForwardNet" in msg

    def test_pepita_cross_reference(self):
        with pytest.raises(ValueError) as exc:
            Registry.get(ComponentCategory.PROPAGATOR, "pepita")
        msg = str(exc.value)
        assert "pepita" in msg
        assert "bioplausible.zoo.models.forward_only.PEPITA" in msg
        assert "requires model-level control" in msg

    def test_target_prop_cross_reference(self):
        with pytest.raises(ValueError) as exc:
            Registry.get(ComponentCategory.PROPAGATOR, "target_prop")
        msg = str(exc.value)
        assert "diff_target_prop" in msg
        assert "bioplausible.zoo.models.target_prop.DifferenceTargetProp" in msg

    def test_difference_target_prop_cross_reference(self):
        with pytest.raises(ValueError) as exc:
            Registry.get(ComponentCategory.PROPAGATOR, "difference_target_prop")
        msg = str(exc.value)
        assert "diff_target_prop" in msg
        assert "requires model-level control" in msg

    def test_predictive_coding_cross_reference(self):
        with pytest.raises(ValueError) as exc:
            Registry.get(ComponentCategory.PROPAGATOR, "predictive_coding")
        msg = str(exc.value)
        assert "predictive_coding_hybrid" in msg
        assert "bioplausible.zoo.models.predictive_coding" in msg

    def test_actual_model_registration_is_reachable(self):
        """The model-side classes are actually registered and accessible via the model category."""
        model = Registry.get(ComponentCategory.MODEL, "pepita")
        assert model is not None
        # Can be instantiated with a simple forward pass
        instance = model(input_dim=10, hidden_dim=16, output_dim=2, num_layers=1)
        import torch

        x = torch.randn(4, 10)
        y = instance(x)
        assert y.shape == (4, 2)

    def test_unknown_propagator_still_raises(self):
        """An unknown propagator (no cross-ref) still gets the generic error."""
        with pytest.raises(ValueError) as exc:
            Registry.get(ComponentCategory.PROPAGATOR, "nonexistent_propagator")
        assert "Available" in str(exc.value)

    def test_working_propagators_still_resolve(self):
        """Working propagators that ARE registered still resolve normally."""
        from bioplausible.zoo.propagators.eqprop import EqProp

        result = Registry.get(ComponentCategory.PROPAGATOR, "eq_prop")
        assert result is EqProp


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
    model = ForwardForwardNet(
        input_dim=64, hidden_dim=16, output_dim=10, num_layers=2
    )
    x = torch.randn(4, 1, 8, 8)
    y = torch.randint(0, 10, (4,))
    out = model.forward(x)
    assert out.shape == (4, 10)
    result = model.train_step(x, y)
    assert "loss" in result
    assert "accuracy" in result
