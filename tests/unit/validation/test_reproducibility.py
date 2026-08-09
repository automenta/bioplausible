"""Reproducibility Unit Tests.

Validates:
1. Fixed seed → identical model weights, identical loss trajectory (5 steps)
2. Environment capture (git commit, torch version, deps hash) serializes correctly

Target: <10s total.
"""

import hashlib
import json
import subprocess
import sys
from typing import Any

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
COMMIT_MIN_LEN = 8
HASH_LEN = 16
TIMEOUT_DEPS = 30
TIMEOUT_GIT = 10

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


def _get_deps_hash() -> str:
    """Get a hash of installed dependencies for reproducibility."""
    try:
        # Use uv to get dependencies
        result = subprocess.run(
            ["uv", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_DEPS,
            check=False,
        )
        if result.returncode == 0:
            return hashlib.sha256(result.stdout.encode()).hexdigest()[:HASH_LEN]
    except Exception:
        pass

    # Fallback: hash of key package versions
    key_packages = ["torch", "numpy", "pytest"]
    versions = {}
    for pkg in key_packages:
        try:
            versions[pkg] = __import__(pkg).__version__
        except Exception:
            versions[pkg] = "unknown"
    return hashlib.sha256(str(versions).encode()).hexdigest()[:HASH_LEN]


def _get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd="/home/me/bioplausible",
            timeout=TIMEOUT_GIT,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return "unknown"


def _get_environment_info() -> dict[str, Any]:
    """Capture environment information for reproducibility."""
    return {
        "git_commit": _get_git_commit(),
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "deps_hash": _get_deps_hash(),
        "cuda_available": torch.cuda.is_available(),
        "device": "cpu",
    }


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


class TestEnvironmentCapture:
    """Tests for environment capture serialization."""

    def test_environment_capture_contains_required_fields(self):
        """Environment info should contain all required fields."""
        env = _get_environment_info()

        required_fields = [
            "git_commit",
            "torch_version",
            "python_version",
            "deps_hash",
            "cuda_available",
            "device",
        ]

        for field in required_fields:
            assert field in env, f"Missing field: {field}"
            assert env[field] is not None, f"Field {field} is None"
            assert env[field] != "", f"Field {field} is empty"

    def test_environment_capture_serializes_to_json(self):
        """Environment info should be JSON serializable."""
        env = _get_environment_info()

        # Should not raise
        json_str = json.dumps(env)
        assert isinstance(json_str, str)

        # Should be able to deserialize
        env2 = json.loads(json_str)
        assert env2 == env

    def test_git_commit_is_valid(self):
        """Git commit should be a valid hash."""
        commit = _get_git_commit()
        assert commit != "unknown", "Should be able to get git commit"
        assert len(commit) >= COMMIT_MIN_LEN, f"Commit hash too short: {commit}"
        # Should be hex
        int(commit, 16)  # Will raise if not hex

    def test_deps_hash_is_consistent(self):
        """Deps hash should be consistent across calls."""
        hash1 = _get_deps_hash()
        hash2 = _get_deps_hash()
        assert hash1 == hash2, "Deps hash should be deterministic"

    def test_environment_capture_deterministic(self):
        """Multiple calls should produce identical environment info."""
        env1 = _get_environment_info()
        env2 = _get_environment_info()
        assert env1 == env2, "Environment capture should be deterministic"


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
