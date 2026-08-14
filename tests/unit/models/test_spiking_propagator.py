"""Tests for zoo/propagators/spiking.py (STDPLearningRule).

Verifies the STDP propagator registers, instantiates with a model, executes a
one-step update without error, and exhibits the defining Hebbian behavior:
correlated pre/post activity strengthens the corresponding weight.
"""

import torch

from bioplausible.zoo.models.eqprop import BackpropMLP
from bioplausible.core.local_learning.rules.spiking import STDPLearningRule


def _make_model(input_dim=8, hidden_dim=8, output_dim=4):
    model = BackpropMLP(
        input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=2
    )
    return model


def _make_rule(model, **kwargs):
    return STDPLearningRule(list(model.parameters()), model, **kwargs)


class TestSTDPLearningRule:
    def test_registered(self):
        from bioplausible.core.registry import ComponentCategory, Registry

        cls = Registry.get(ComponentCategory.PROPAGATOR, "stdp")
        assert cls is STDPLearningRule

    def test_instantiates_with_model(self):
        model = _make_model()
        rule = _make_rule(model)
        assert rule.model is model

    def test_one_step_updates_weights(self):
        model = _make_model()
        rule = _make_rule(model, num_steps=3)
        w_before = [p.data.clone() for p in model.parameters()]
        x = torch.randn(8, 8)
        y = torch.randint(0, 4, (8,))
        rule.step(x, y)
        changed = any(
            not torch.allclose(b, p.data) for b, p in zip(w_before, model.parameters())
        )
        assert changed, "STDP step should update at least one weight"

    def test_weights_remain_finite(self):
        model = _make_model()
        rule = _make_rule(model, num_steps=5)
        x = torch.randn(8, 8)
        y = torch.randint(0, 4, (8,))
        rule.step(x, y)
        for p in model.parameters():
            assert torch.isfinite(p.data).all()

    def test_hebbian_strengthens_correlated_weight(self):
        """Correlated pre/post activity should increase the co-active weight."""
        model = BackpropMLP(input_dim=4, hidden_dim=4, output_dim=1, num_layers=1)
        with torch.no_grad():
            model.net[0].weight.zero_()
            model.net[0].bias.zero_()
        # Single layer: input(4) -> output(1). Allocate a small positive bias so
        # the output neuron spikes, then correlate pre-neuron 0 with that spike.
        with torch.no_grad():
            model.net[0].weight[0, 0] = 1.0  # only pre-neuron 0 connects strongly
            model.net[0].bias[0] = 0.0
        rule = _make_rule(
            model, num_steps=5, lr=0.1, threshold=0.5, a_plus=1.0, a_minus=0.1
        )
        w = model.net[0].weight
        w00_before = w[0, 0].item()
        w01_before = w[0, 1].item()

        # All-inputs-on, single output neuron that will spike (bias 0, but with
        # a leading pre spike on neuron 0 driving weight[0,0]).
        x = torch.full((16, 4), 3.0)  # strongly positive -> sigmoid ~1 -> spikes
        y = torch.ones(16, dtype=torch.long)
        rule.step(x, y)

        w00_after = w[0, 0].item()
        assert w00_after > w00_before, (
            f"STDP should strengthen the correlated weight: {w00_before} -> {w00_after}"
        )

    def test_incompatible_model_raises(self):
        from torch import nn

        class NoGraph(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 4)

            def forward(self, x):
                return x

        model = NoGraph()
        rule = _make_rule(model)
        x = torch.randn(4, 4)
        y = torch.zeros(4, dtype=torch.long)
        try:
            rule.step(x, y)
        except TypeError:
            return
        raise AssertionError(
            "STDP should raise TypeError on a model without transition_modules"
        )
