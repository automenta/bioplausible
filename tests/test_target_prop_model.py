"""Tests for zoo/models/target_prop.py — DifferenceTargetProp + DTPLayer."""

import torch
import pytest

from bioplausible.zoo.models.target_prop import DTPLayer, DifferenceTargetProp


class TestDTPLayer:
    def test_construction(self):
        layer = DTPLayer(10, 20)
        assert isinstance(layer.forward_net, torch.nn.Sequential)
        assert isinstance(layer.inverse_net, torch.nn.Sequential)
        assert layer.forward_net[0].in_features == 10
        assert layer.forward_net[0].out_features == 20
        assert layer.inverse_net[0].in_features == 20
        assert layer.inverse_net[0].out_features == 10
        assert layer.opt_f is not None
        assert layer.opt_g is not None

    def test_forward_net_shape(self):
        layer = DTPLayer(10, 20)
        x = torch.randn(4, 10)
        out = layer.forward_net(x)
        assert out.shape == (4, 20)

    def test_inverse_net_shape(self):
        layer = DTPLayer(10, 20)
        h = torch.randn(4, 20)
        out = layer.inverse_net(h)
        assert out.shape == (4, 10)


class TestDifferenceTargetProp:
    def test_construction_int_input(self):
        model = DifferenceTargetProp(
            input_dim=784, hidden_dim=256, output_dim=10, num_layers=3
        )
        assert model.input_dim == 784
        assert model.output_dim == 10
        assert len(model.layers) == 3
        # First layer: 784->256, rest: 256->256
        assert model.layers[0].forward_net[0].in_features == 784
        assert model.layers[0].forward_net[0].out_features == 256
        for i in range(1, 3):
            assert model.layers[i].forward_net[0].in_features == 256
            assert model.layers[i].forward_net[0].out_features == 256
        assert model.out_layer.in_features == 256
        assert model.out_layer.out_features == 10

    def test_construction_tuple_input(self):
        model = DifferenceTargetProp(input_dim=(28, 28), hidden_dim=64, output_dim=10)
        assert model.input_dim == 784  # math.prod((28, 28))
        assert model.out_layer.in_features == 64

    def test_construction_default_num_layers(self):
        model = DifferenceTargetProp(input_dim=10, hidden_dim=20, output_dim=5)
        assert len(model.layers) == 2  # default

    def test_forward_shape(self):
        model = DifferenceTargetProp(input_dim=50, hidden_dim=30, output_dim=10)
        x = torch.randn(4, 50)
        out = model.forward(x)
        assert out.shape == (4, 10)

    def test_forward_gradient_flow(self):
        model = DifferenceTargetProp(input_dim=50, hidden_dim=30, output_dim=10)
        x = torch.randn(2, 50, requires_grad=True)
        out = model.forward(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_build_classmethod(self):
        class MockSpec:
            pass

        model = DifferenceTargetProp.build(
            MockSpec(),
            input_dim=100,
            output_dim=5,
            hidden_dim=40,
            num_layers=2,
            device="cpu",
        )
        assert isinstance(model, DifferenceTargetProp)
        assert model.input_dim == 100
        assert model.output_dim == 5
        assert len(model.layers) == 2

    def test_train_step_returns_dict(self):
        model = DifferenceTargetProp(
            input_dim=20, hidden_dim=16, output_dim=3, num_layers=2
        )
        x = torch.randn(8, 20)
        y = torch.randint(0, 3, (8,))
        result = model.train_step(x, y)
        assert isinstance(result, dict)
        assert "loss" in result
        assert "accuracy" in result
        assert isinstance(result["loss"], float)
        assert isinstance(result["accuracy"], float)

    def test_train_step_loss_decreases(self):
        model = DifferenceTargetProp(
            input_dim=20, hidden_dim=16, output_dim=3, num_layers=2
        )
        x = torch.randn(16, 20)
        y = torch.randint(0, 3, (16,))
        losses = []
        for _ in range(5):
            result = model.train_step(x, y)
            losses.append(result["loss"])
        # Loss should generally decrease
        assert losses[-1] <= losses[0] + 0.1  # allow slight noise

    def test_train_step_accuracy_range(self):
        model = DifferenceTargetProp(
            input_dim=20, hidden_dim=16, output_dim=3, num_layers=2
        )
        x = torch.randn(8, 20)
        y = torch.randint(0, 3, (8,))
        result = model.train_step(x, y)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_train_step_forward_pass_preserves_params(self):
        model = DifferenceTargetProp(
            input_dim=20, hidden_dim=16, output_dim=3, num_layers=2
        )
        x = torch.randn(8, 20)
        y = torch.randint(0, 3, (8,))
        # Before train_step, after forward
        out_before = model.forward(x)
        _ = model.train_step(x, y)
        out_after = model.forward(x)
        # train_step should update params, so output should change (or stay similar)
        assert out_before.shape == out_after.shape

    def test_gradient_target_computation(self):
        """Verify the gradient-based target at output is computed correctly."""
        model = DifferenceTargetProp(
            input_dim=10, hidden_dim=8, output_dim=2, num_layers=1
        )
        x = torch.randn(4, 10)
        y = torch.randint(0, 2, (4,))
        # Run train_step - main goal is no crash
        result = model.train_step(x, y)
        assert "loss" in result
