"""Tests for zoo/models/predictive_coding.py — FabricPCGraphPCN + PredictiveCodingHybrid."""

import pytest
import torch
from bioplausible.config.unified import ModelConfig
from bioplausible.zoo.models.predictive_coding import (
    FabricPCGraphPCN,
    PredictiveCodingHybrid,
)


class TestFabricPCGraphPCN:
    def test_construction_defaults(self):
        model = FabricPCGraphPCN()
        assert model.config is not None
        assert model.config.input_dim == 784
        assert model.config.output_dim == 10
        assert model.config.hidden_dims == [256]
        assert hasattr(model, "_params")
        assert model._mode == "pcn"

    def test_construction_explicit_dims(self):
        model = FabricPCGraphPCN(input_dim=50, hidden_dim=30, output_dim=5)
        assert model.config.input_dim == 50
        assert model.config.output_dim == 5
        assert model.config.hidden_dims == [30]

    def test_construction_extra_kwargs(self):
        model = FabricPCGraphPCN(
            input_dim=10,
            hidden_dim=8,
            output_dim=3,
            infer_steps=10,
            eta_infer=0.1,
            mode="backprop",
        )
        assert model._mode == "backprop"
        assert model.config.extra.get("infer_steps") == 10
        assert model.config.extra.get("eta_infer") == pytest.approx(0.1)

    def test_forward_shape(self):
        model = FabricPCGraphPCN(input_dim=10, hidden_dim=8, output_dim=3)
        x = torch.randn(4, 10)
        out = model.forward(x)
        assert out.shape == (4, 3)

    def test_to_device(self):
        model = FabricPCGraphPCN(input_dim=10, hidden_dim=8, output_dim=3)
        model.to(torch.device("cpu"))
        assert model._device == torch.device("cpu")
        # Verify params are on cpu
        for node_name in model._params:
            for param_name in model._params[node_name]:
                assert model._params[node_name][param_name].device == torch.device(
                    "cpu"
                )

    def test_train_step_pcn_mode(self):
        model = FabricPCGraphPCN(
            input_dim=10, hidden_dim=8, output_dim=3, infer_steps=5, eta_infer=0.01
        )
        x = torch.randn(4, 10)
        y = torch.randint(0, 3, (4,))
        result = model.train_step(x, y)
        assert isinstance(result, dict)
        assert "loss" in result
        assert "accuracy" in result
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_train_step_backprop_mode(self):
        model = FabricPCGraphPCN(
            input_dim=10, hidden_dim=8, output_dim=3, mode="backprop"
        )
        x = torch.randn(4, 10)
        y = torch.randint(0, 3, (4,))
        result = model.train_step(x, y)
        assert isinstance(result, dict)
        assert "loss" in result
        assert "accuracy" in result

    def test_build_classmethod(self):
        class MockSpec:
            name = "test_model"
            default_lr = 0.01

        model = FabricPCGraphPCN.build(
            MockSpec(),
            input_dim=50,
            output_dim=5,
            hidden_dim=30,
            num_layers=2,
            device="cpu",
            task_type="vision",
        )
        assert isinstance(model, FabricPCGraphPCN)
        assert model.config.input_dim == 50

    def test_build_num_layers_cap(self):
        class MockSpec:
            name = "test"
            default_lr = 0.001

        model = FabricPCGraphPCN.build(
            MockSpec(),
            input_dim=10,
            output_dim=3,
            hidden_dim=8,
            num_layers=10,
            device="cpu",
            task_type="vision",
        )
        assert len(model.config.hidden_dims) == 5  # capped at 5


class TestPredictiveCodingHybrid:
    def test_construction(self):
        model = PredictiveCodingHybrid(input_dim=10, hidden_dim=8, output_dim=3)
        assert len(model.layers) == 2  # 10->8, 8->3
        assert len(model.top_down) == 2  # 8->10, 3->8
        assert model.layers[0].in_features == 10
        assert model.layers[0].out_features == 8
        assert model.layers[1].in_features == 8
        assert model.layers[1].out_features == 3

    def test_construction_from_config(self):
        config = ModelConfig(name="test", input_dim=20, output_dim=4, hidden_dims=[16])
        model = PredictiveCodingHybrid(config=config)
        assert len(model.layers) == 2
        assert model.layers[0].in_features == 20

    def test_construction_empty_hidden_dims(self):
        config = ModelConfig(name="test", input_dim=10, output_dim=3, hidden_dims=[])
        model = PredictiveCodingHybrid(config=config)
        # BioModel default hidden_dim=256 falls back when hidden_dims=[]
        assert model.config.hidden_dims == []
        assert len(model.layers) == 2  # 10->256, 256->3

    def test_forward_shape(self):
        model = PredictiveCodingHybrid(input_dim=10, hidden_dim=8, output_dim=3)
        x = torch.randn(4, 10)
        out = model.forward(x)
        assert out.shape == (4, 3)

    def test_train_step_returns_dict(self):
        model = PredictiveCodingHybrid(input_dim=20, hidden_dim=16, output_dim=4)
        x = torch.randn(8, 20)
        y = torch.randint(0, 4, (8,))
        result = model.train_step(x, y)
        assert isinstance(result, dict)
        assert "loss" in result
        assert "accuracy" in result
        assert isinstance(result["loss"], float)
        assert isinstance(result["accuracy"], float)

    def test_train_step_loss_decreases(self):
        model = PredictiveCodingHybrid(input_dim=10, hidden_dim=8, output_dim=3)
        x = torch.randn(8, 10)
        y = torch.randint(0, 3, (8,))
        losses = []
        for _ in range(5):
            result = model.train_step(x, y)
            losses.append(result["loss"])
        assert losses[-1] <= losses[0] + 0.1

    def test_train_step_accuracy_range(self):
        model = PredictiveCodingHybrid(input_dim=20, hidden_dim=16, output_dim=4)
        x = torch.randn(8, 20)
        y = torch.randint(0, 4, (8,))
        result = model.train_step(x, y)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_build_classmethod(self):
        class MockSpec:
            name = "test"

        model = PredictiveCodingHybrid.build(
            MockSpec(),
            input_dim=50,
            output_dim=5,
            hidden_dim=30,
            num_layers=2,
            device="cpu",
            task_type="vision",
        )
        assert isinstance(model, PredictiveCodingHybrid)
        assert model.config.input_dim == 50
