"""Reproducibility Unit Tests.

Validates fixed-seed reproducibility: identical model weights, identical loss
trajectory (5 steps), and identical outputs after training. (Environment-capture
coverage lives in ``test_repro_check.py`` against the real ``bioplausible.utils``
helpers.)

Target: <10s total.
"""

import pytest
import torch

from bioplausible.core.registry import ComponentCategory, Registry
from bioplausible.zoo import get_model_spec

# Constants
SEED_WEIGHTS = 12345
SEED_DATA = 42
SEED_TRAIN = 999
SEED_FULL = 12345
LOSS_TOL = 1e-6
OUTPUT_RTOL = 1e-5
OUTPUT_ATOL = 1e-7

# =============================================================================
# Helpers
# =============================================================================


def _instantiate_model(model_name: str, input_dim: int = 64, output_dim: int = 10):
    """Instantiate a model via its build() method."""
    spec = get_model_spec(model_name)
    model_cls = Registry.get(ComponentCategory.MODEL, model_name)

    if not hasattr(model_cls, "build"):
        raise NotImplementedError(f"{model_name} has no build() method")

    return model_cls.build(
        spec=spec,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=64,
        num_layers=2,
        device="cpu",
        task_type="vision",
    )


# =============================================================================
# Reproducibility Tests
# =============================================================================


@pytest.fixture(scope="module")
def test_models():
    """Subset of models that reliably instantiate and train."""
    return ["eqprop_mlp", "forward_forward", "pepita", "directed_ep"]


class TestReproducibility:
    """Tests for reproducibility guarantees."""

    @pytest.mark.parametrize(
        "model_name", ["eqprop_mlp", "forward_forward", "pepita", "directed_ep"]
    )
    def test_fixed_seed_identical_weights(self, model_name):
        """Fixed seed should produce identical initial weights."""
        torch.manual_seed(SEED_WEIGHTS)
        model1 = _instantiate_model(model_name)

        torch.manual_seed(SEED_WEIGHTS)
        model2 = _instantiate_model(model_name)

        # Compare all parameters
        for (name1, p1), (name2, p2) in zip(
            model1.named_parameters(), model2.named_parameters()
        ):
            assert name1 == name2, f"Parameter name mismatch: {name1} vs {name2}"
            assert torch.equal(p1, p2), f"Parameter {name1} differs with same seed"

    @pytest.mark.parametrize(
        "model_name", ["eqprop_mlp", "forward_forward", "pepita", "directed_ep"]
    )
    def test_fixed_seed_identical_loss_trajectory(self, model_name):
        """Fixed seed should produce identical loss trajectory for 5 steps."""
        # Create deterministic synthetic data
        torch.manual_seed(SEED_DATA)
        x = torch.randn(32, 64)
        y = torch.randint(0, 10, (32,))

        # Train model 1
        torch.manual_seed(SEED_TRAIN)
        model1 = _instantiate_model(model_name)
        model1.train()
        losses1 = []
        for _ in range(5):
            result = model1.train_step(x, y)
            if isinstance(result, dict) and "loss" in result:
                losses1.append(result["loss"])
            else:
                # For models that return None from train_step, do standard training
                break

        # Train model 2 with same seed
        torch.manual_seed(SEED_TRAIN)
        model2 = _instantiate_model(model_name)
        model2.train()
        losses2 = []
        for _ in range(5):
            result = model2.train_step(x, y)
            if isinstance(result, dict) and "loss" in result:
                losses2.append(result["loss"])
            else:
                break

        # Compare loss trajectories
        assert len(losses1) == len(losses2), (
            f"Loss trajectory length mismatch: {len(losses1)} vs {len(losses2)}"
        )
        for i, (l1, l2) in enumerate(zip(losses1, losses2)):
            assert abs(l1 - l2) < LOSS_TOL, f"Loss at step {i} differs: {l1} vs {l2}"

    @pytest.mark.parametrize(
        "model_name", ["eqprop_mlp", "forward_forward", "pepita", "directed_ep"]
    )
    def test_fixed_seed_identical_output_after_training(self, model_name):
        """Fixed seed should produce identical outputs after training."""
        # Create deterministic synthetic data
        torch.manual_seed(SEED_DATA)
        x = torch.randn(32, 64)
        y = torch.randint(0, 10, (32,))

        # Train model 1
        torch.manual_seed(SEED_TRAIN)
        model1 = _instantiate_model(model_name)
        model1.train()
        for _ in range(3):
            model1.train_step(x, y)

        model1.eval()
        with torch.no_grad():
            out1 = model1(x)

        # Train model 2 with same seed
        torch.manual_seed(SEED_TRAIN)
        model2 = _instantiate_model(model_name)
        model2.train()
        for _ in range(3):
            model2.train_step(x, y)

        model2.eval()
        with torch.no_grad():
            out2 = model2(x)

        assert torch.allclose(out1, out2, rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL), (
            f"{model_name}: non-deterministic output after training with fixed seed"
        )


class TestModelStateSerialization:
    """Tests for model state serialization (weights, config)."""

    @pytest.mark.parametrize("model_name", ["eqprop_mlp", "forward_forward"])
    def test_model_state_dict_serialization(self, model_name):
        """Model state dict should be serializable and loadable."""
        torch.manual_seed(SEED_DATA)
        model = _instantiate_model(model_name)

        # Get state dict
        state_dict = model.state_dict()

        # Verify it's a valid dict of tensors
        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0
        for key, tensor in state_dict.items():
            assert isinstance(key, str)
            assert isinstance(tensor, torch.Tensor)

        # Create new model and load
        torch.manual_seed(SEED_DATA)
        model2 = _instantiate_model(model_name)
        model2.load_state_dict(state_dict)

        # Verify loaded weights match
        for (name1, p1), (name2, p2) in zip(
            model.named_parameters(), model2.named_parameters()
        ):
            assert torch.equal(p1, p2), f"Parameter {name1} differs after load"

    @pytest.mark.parametrize("model_name", ["eqprop_mlp", "forward_forward"])
    def test_model_config_serialization(self, model_name):
        """Model state should round-trip through a serializable container."""
        import io

        torch.manual_seed(SEED_DATA)
        model = _instantiate_model(model_name)

        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)
        restored = torch.load(buf, weights_only=True)

        assert isinstance(restored, dict)
        assert set(restored.keys()) == set(model.state_dict().keys())


# =============================================================================
# Integration Test: Full Training Reproducibility
# =============================================================================


def test_full_training_reproducibility():
    """Full training run should be reproducible with fixed seed."""
    model_name = "eqprop_mlp"

    # Create deterministic data
    torch.manual_seed(SEED_DATA)
    x = torch.randn(64, 64)
    y = torch.randint(0, 10, (64,))

    # Run 1
    torch.manual_seed(SEED_FULL)
    model1 = _instantiate_model(model_name)
    model1.train()
    losses1 = []
    for epoch in range(2):
        perm = torch.randperm(len(x))
        for i in range(0, len(x), 16):
            idx = perm[i : i + 16]
            xb, yb = x[idx], y[idx]
            result = model1.train_step(xb, yb)
            if isinstance(result, dict) and "loss" in result:
                losses1.append(result["loss"])

    model1.eval()
    with torch.no_grad():
        out1 = model1(x[:16])

    # Run 2 with same seed
    torch.manual_seed(SEED_FULL)
    model2 = _instantiate_model(model_name)
    model2.train()
    losses2 = []
    for epoch in range(2):
        perm = torch.randperm(len(x))
        for i in range(0, len(x), 16):
            idx = perm[i : i + 16]
            xb, yb = x[idx], y[idx]
            result = model2.train_step(xb, yb)
            if isinstance(result, dict) and "loss" in result:
                losses2.append(result["loss"])

    model2.eval()
    with torch.no_grad():
        out2 = model2(x[:16])

    # Compare
    assert len(losses1) == len(losses2), "Loss trajectory length differs"
    for i, (l1, l2) in enumerate(zip(losses1, losses2)):
        assert abs(l1 - l2) < OUTPUT_RTOL, f"Loss at step {i} differs: {l1} vs {l2}"

    assert torch.allclose(out1, out2, rtol=OUTPUT_RTOL, atol=OUTPUT_ATOL), (
        "Output differs after training"
    )
