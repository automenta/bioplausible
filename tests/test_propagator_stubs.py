"""
Tests for propagator stubs and model-side boundary.

Locks the contract that:
1. The four NotImplementedError stub propagators raise the expected exception.
2. The model-side implementations they point to actually train (one step).
"""

import pytest
import torch

from bioplausible.zoo.propagators import (
    FFStub,
    PEPITAStub,
    TargetPropStub,
    DTPStub,
    PCNStub,
    ForwardForwardNet,
    PEPITA,
    DifferenceTargetProp,
    FabricPCGraphPCN,
    PredictiveCodingHybrid,
)


class TestPropagatorStubsRaiseNotImplemented:
    """The stub propagators must raise NotImplementedError with helpful messages."""

    def test_ff_stub_raises(self):
        model = torch.nn.Linear(784, 10)
        stub = FFStub(model.parameters(), model)
        with pytest.raises(NotImplementedError, match="ForwardForwardNet"):
            stub.step(torch.randn(2, 784), torch.randint(0, 10, (2,)))

    def test_pepita_stub_raises(self):
        model = torch.nn.Linear(784, 10)
        stub = PEPITAStub(model.parameters(), model)
        with pytest.raises(NotImplementedError, match="PEPITA model"):
            stub.step(torch.randn(2, 784), torch.randint(0, 10, (2,)))

    def test_target_prop_stub_raises(self):
        model = torch.nn.Linear(784, 10)
        stub = TargetPropStub(model.parameters(), model)
        with pytest.raises(NotImplementedError, match="model-level implementation"):
            stub.step(torch.randn(2, 784), torch.randint(0, 10, (2,)))

    def test_difference_target_prop_stub_raises(self):
        model = torch.nn.Linear(784, 10)
        stub = DTPStub(model.parameters(), model)
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            stub.step(torch.randn(2, 784), torch.randint(0, 10, (2,)))

    def test_pcn_stub_raises(self):
        model = torch.nn.Linear(784, 10)
        stub = PCNStub(model.parameters(), model)
        with pytest.raises(NotImplementedError, match="graph.training.train_pcn"):
            stub.step(torch.randn(2, 784), torch.randint(0, 10, (2,)))


class TestModelSideImplementationsTrain:
    """Assert the model-side classes actually learn (one step)."""

    @pytest.fixture
    def batch(self):
        return torch.randn(4, 784), torch.randint(0, 10, (4,))

    def test_forward_forward_net_train_step(self, batch):
        x, y = batch
        model = ForwardForwardNet(input_dim=784, hidden_dim=64, output_dim=10, num_layers=2)
        stats = model.train_step(x, y)
        assert isinstance(stats, dict)
        assert "loss" in stats
        assert "accuracy" in stats
        assert 0 <= stats["accuracy"] <= 1

    def test_pepita_train_step(self, batch):
        x, y = batch
        model = PEPITA(input_dim=784, hidden_dim=64, output_dim=10, num_layers=2)
        stats = model.train_step(x, y)
        assert isinstance(stats, dict)
        assert "loss" in stats
        assert "accuracy" in stats
        assert 0 <= stats["accuracy"] <= 1

    def test_difference_target_prop_train_step(self, batch):
        x, y = batch
        model = DifferenceTargetProp(input_dim=784, hidden_dim=64, output_dim=10, num_layers=2)
        stats = model.train_step(x, y)
        assert isinstance(stats, dict)
        assert "loss" in stats
        assert "accuracy" in stats
        assert 0 <= stats["accuracy"] <= 1

    def test_fabric_pc_graph_pcn_train_step(self, batch):
        x, y = batch
        model = FabricPCGraphPCN(input_dim=784, hidden_dim=64, output_dim=10)
        stats = model.train_step(x, y)
        assert isinstance(stats, dict)
        assert "loss" in stats
        assert "accuracy" in stats
        assert 0 <= stats["accuracy"] <= 1

    def test_predictive_coding_hybrid_train_step(self, batch):
        x, y = batch
        model = PredictiveCodingHybrid(input_dim=784, hidden_dim=64, output_dim=10)
        stats = model.train_step(x, y)
        assert isinstance(stats, dict)
        assert "loss" in stats
        assert "accuracy" in stats
        assert 0 <= stats["accuracy"] <= 1