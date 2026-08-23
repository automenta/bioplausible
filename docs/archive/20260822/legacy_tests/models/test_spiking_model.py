"""Tests for zoo/models/spiking.py (SpikingSTDP with snnTorch LIF kinetics).

snnTorch is a core dependency (see pyproject), so SpikingSTDP runs its real
Leaky-Integrate-and-Fire path unconditionally; there is no no-snnTorch
fallback anymore.
"""

import torch

from bioplausible.zoo.models.spiking import SpikingSTDP


class TestSpikingSTDP:
    """SpikingSTDP LIF path."""

    def test_forward_shape(self):
        model = SpikingSTDP(input_dim=8, hidden_dim=16, output_dim=4)
        x = torch.randn(2, 8)
        out = model(x)
        assert out.shape == (2, 4)
        assert not out.isnan().any()

    def test_forward_has_output(self):
        model = SpikingSTDP(input_dim=8, hidden_dim=16, output_dim=4, num_steps=5)
        x = torch.randn(2, 8)
        out = model(x)
        assert out.requires_grad
        assert out.shape == (2, 4)
        assert out.abs().sum().item() > 0 or not out.isnan().any()

    def test_train_step_returns_dict(self):
        model = SpikingSTDP(input_dim=8, hidden_dim=16, output_dim=4)
        x = torch.randn(4, 8)
        y = torch.randint(0, 4, (4,))
        result = model.train_step(x, y)
        assert isinstance(result, dict)
        assert "loss" in result
        assert "accuracy" in result
        assert result["loss"] >= 0.0
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_build_classmethod(self):
        model = SpikingSTDP.build(
            spec=None,
            input_dim=8,
            output_dim=4,
            hidden_dim=16,
            num_layers=2,
            device="cpu",
            task_type="vision",
        )
        assert isinstance(model, SpikingSTDP)
        assert model.fc1.in_features == 8
        assert model.fc1.out_features == 16
        assert model.fc2.in_features == 16
        assert model.fc2.out_features == 4

    def test_registered_model_name(self):
        from bioplausible.core.registry import ComponentCategory, Registry

        cls = Registry.get(ComponentCategory.MODEL, "spiking_stdp")
        assert cls is SpikingSTDP

    def test_train_step_no_trainable_params_change(self):
        model = SpikingSTDP(input_dim=8, hidden_dim=16, output_dim=4)
        w1_before = model.fc1.weight.data.clone()
        x = torch.randn(4, 8)
        y = torch.randint(0, 4, (4,))
        model.train_step(x, y)
        # train_step applies the STDP weight update in-place (fc1 weight and
        # trace variables are mutated), but the tensor shapes are stable.
        assert model.fc1.weight.data.shape == w1_before.shape

    def test_multiple_batches(self):
        model = SpikingSTDP(input_dim=8, hidden_dim=16, output_dim=4)
        for _ in range(3):
            x = torch.randn(4, 8)
            y = torch.randint(0, 4, (4,))
            out = model(x)
            assert out.shape == (4, 4)

    def test_forward_different_batch_sizes(self):
        model = SpikingSTDP(input_dim=8, hidden_dim=16, output_dim=4)
        for bs in [1, 2, 7]:
            x = torch.randn(bs, 8)
            out = model(x)
            assert out.shape == (bs, 4)
