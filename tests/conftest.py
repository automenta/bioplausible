"""Shared test fixtures and configuration."""

import shutil
import sys
import tempfile
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

# Hard dependencies — no mock stubs needed
# bioplausible.acceleration checks for cupy
from unittest.mock import MagicMock  # ruff: ignore[module-import-not-at-top-of-file]

sys.modules["cupy"] = MagicMock()

import pytest
import torch
from torch import nn

from bioplausible.zoo.models.transitions import TransitionGraphMixin

# --- Shared Model Fixtures ---


class SimpleMLP(TransitionGraphMixin, nn.Module):
    """Minimal 2-layer MLP for eqprop tests."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 20, output_dim: int = 5):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h)
        return h


@pytest.fixture
def simple_mlp() -> SimpleMLP:
    return SimpleMLP()


@pytest.fixture
def sample_batch() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(4, 10)
    y = torch.randint(0, 5, (4,))
    return x, y


def pytest_unconfigure(config: object) -> None:
    """Clean up test artifacts after session ends."""
    kb_tmp = Path(tempfile.gettempdir()) / "bioplausible-knowledgebase.json"
    if kb_tmp.exists():
        kb_tmp.unlink()
    kb_tmp_dir = Path(tempfile.gettempdir()) / "bioplausible_kb"
    if kb_tmp_dir.exists():
        shutil.rmtree(kb_tmp_dir, ignore_errors=True)
    cwd_kb = Path.cwd() / "knowledgebase.json"
    if cwd_kb.exists():
        cwd_kb.unlink()


# --- E.2 Shared Fixtures (test reorg) ---


@pytest.fixture(scope="session")
def synthetic_classification() -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic synthetic classification data for all fast tests."""
    torch.manual_seed(42)
    X = torch.randn(200, 64)
    y = (X.sum(dim=1) > 0).long() % 10
    return X, y


@pytest.fixture
def mnist_quick_task():
    """MNIST task in quick_mode (small subset, no download).

    Returns a VisionTask configured for quick test runs.
    """
    from bioplausible.tasks.vision import VisionTask

    return VisionTask("mnist", quick_mode=True)


@pytest.fixture
def eqprop_model():
    """Minimal StandardEqProp for settling/contrastive tests."""
    from bioplausible.core.config import ModelConfig
    from bioplausible.zoo.models.eqprop import StandardEqProp

    config = ModelConfig(name="test", input_dim=64, output_dim=10, max_steps=5)
    return StandardEqProp(config=config)


# --- Sprint 4.3.4 Synthetic Fixtures (zero I/O, zero download) ---


@pytest.fixture(scope="session")
def synthetic_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """A small deterministic batch (x, y) for fast feedforward tests."""
    torch.manual_seed(0)
    x = torch.randn(8, 64)
    y = torch.randint(0, 10, (8,))
    return x, y


@pytest.fixture(scope="session")
def synthetic_vision_task() -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic image-shaped classification tensors (no MNIST download).

    Returns (images, labels) where images are (N, 1, 16, 16). Tests that need a
    real VisionTask loader should use tests/slow/ instead.
    """
    torch.manual_seed(1)
    n = 64
    images = torch.randn(n, 1, 16, 16)
    # Inject a weak spatial signal so the task is learnable.
    images += (images.mean(dim=(2, 3), keepdim=True) > 0).float() * 0.5
    labels = (images.mean(dim=(2, 3)).squeeze(1) > 0).long() % 10
    return images, labels


@pytest.fixture(scope="session")
def synthetic_lm_task() -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic token-sequence batch for LM tests (no download).

    Returns (input_ids, target_ids) of shape (N, seq_len) over a small vocab.
    """
    torch.manual_seed(2)
    seq_len = 24
    vocab_size = 256
    n = 8
    ids = torch.randint(1, vocab_size, (n, seq_len))
    input_ids = ids[:, :-1]
    target_ids = ids[:, 1:]
    return input_ids, target_ids
