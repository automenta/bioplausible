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
from unittest.mock import MagicMock  # noqa: E402

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
