"""Shared test fixtures and configuration."""

import logging
import shutil
import sys
import tempfile
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

# Hard dependencies — no mock stubs needed
# computronium.acceleration checks for cupy
from unittest.mock import MagicMock  # ruff: ignore[module-import-not-at-top-of-file]

sys.modules["cupy"] = MagicMock()

from typing import cast

import pytest
import torch
from torch import nn


def _transition_modules_autodiscover(model: nn.Module) -> list[nn.Module]:
    """Auto-discover transition modules for models with standard structure.

    Mirrors the deprecated TransitionGraphMixin logic for test fixtures.
    """
    # 1. Explicit ModuleList (most common)
    layers = getattr(model, "layers", None)
    if isinstance(layers, nn.ModuleList):
        return list(layers)
    # 2. Forward layers
    forward_layers = getattr(model, "forward_layers", None)
    if isinstance(forward_layers, nn.ModuleList):
        return list(forward_layers)

    raise NotImplementedError(
        f"{type(model).__name__} has no transition_modules(). "
        "Define `self.layers: nn.ModuleList[nn.Module]` or implement "
        "transition_modules()."
    )


def lm_train_step(
    model: nn.Module, input_ids: torch.Tensor, target_ids: torch.Tensor | None = None
) -> dict[str, float]:
    """Route a TileLM-style token-id training step through ``dispatch_train_step``.

    TileLM exposes no self-owned learning rule (``train_step`` raises
    ``NotImplementedError``), so the dispatcher's BPTT fallback owns the step.
    This mirrors the historical standalone ``train_step`` contract.
    """
    from computronium.core.trainer import dispatch_train_step

    if target_ids is None:
        target_ids = input_ids.clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def bptt_step(x: torch.Tensor, y: torch.Tensor) -> dict[str, object]:
        logits = model(x)
        loss = model.compute_loss(logits, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        with torch.no_grad():
            perplexity = torch.exp(torch.clamp(loss, max=80)).item()
        return {"loss": loss.item(), "perplexity": perplexity}

    return dispatch_train_step(
        model=cast("nn.Module", model),
        x=input_ids,
        y=target_ids,
        adapt_input=lambda x: x,
        bptt_step=bptt_step,
    )


# --- Shared Model Fixtures ---


class SimpleMLP(nn.Module):
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

    def transition_modules(self) -> list[nn.Module]:
        """Return linear layers as transition modules for settling."""
        return [m for m in self.layers if isinstance(m, nn.Linear)]


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
    kb_tmp = Path(tempfile.gettempdir()) / "computronium-knowledgebase.json"
    if kb_tmp.exists():
        kb_tmp.unlink()
    kb_tmp_dir = Path(tempfile.gettempdir()) / "computronium_kb"
    if kb_tmp_dir.exists():
        shutil.rmtree(kb_tmp_dir, ignore_errors=True)
    cwd_kb = Path.cwd() / "knowledgebase.json"
    if cwd_kb.exists():
        cwd_kb.unlink()


def pytest_collection_modifyitems(config: object, items: list[object]) -> None:
    """Apply GPU-marked skips when CUDA is unavailable.

    Any test carrying ``gpu_only`` is skipped on CPU-only machines; ``gpu``
    tests run on whatever device is present (they should be device-agnostic).
    """
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "gpu_only" in item.keywords:
            item.add_marker(skip_gpu)


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
    from computronium.tasks.vision import VisionTask

    return VisionTask("mnist", quick_mode=True)


@pytest.fixture
def eqprop_model():
    """Minimal native_eqprop_mlp for settling/contrastive tests."""
    from computronium.models.native.eqprop_native import native_eqprop_mlp

    return native_eqprop_mlp(input_dim=64, hidden_dim=32, output_dim=10, num_layers=1, beta=0.5, settle_steps=5)


# --- Sprint 4.3.4 Synthetic Fixtures (zero I/O, zero download) ---


@pytest.fixture(scope="session")
def device() -> str:
    """Return 'cuda' if available, else 'cpu'.

    Persistent CUDA is avoided; tests that need a live GPU should use the
    ``gpu`` / ``gpu_only`` markers and place tensors on the returned device.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def cuda_available() -> bool:
    """Whether CUDA is available to tests."""
    return torch.cuda.is_available()


@pytest.fixture
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


# --- Sprint 1.1 GPU Fixtures (session-scoped, placed on CUDA) ---


@pytest.fixture(scope="session")
def gpu_device() -> str:
    """CUDA device for GPU tests (raises if CUDA unavailable)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available", allow_module_level=False)
    return "cuda"


@pytest.fixture(scope="session")
def synthetic_batch_gpu(gpu_device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic (x, y) batch on CUDA for GPU-accelerated tests."""
    torch.manual_seed(0)
    x = torch.randn(128, 64, device=gpu_device)
    y = torch.randint(0, 10, (128,), device=gpu_device)
    return x, y


@pytest.fixture(scope="session")
def synthetic_vision_task_gpu(gpu_device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic image-shaped classification tensors on CUDA."""
    torch.manual_seed(1)
    n = 128
    images = torch.randn(n, 1, 16, 16, device=gpu_device)
    images += (images.mean(dim=(2, 3), keepdim=True) > 0).float() * 0.5
    labels = (images.mean(dim=(2, 3)).squeeze(1) > 0).long() % 10
    return images, labels


@pytest.fixture(scope="session")
def synthetic_lm_task_gpu(gpu_device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic token-sequence batch on CUDA for LM tests."""
    torch.manual_seed(2)
    seq_len = 24
    vocab_size = 256
    n = 128
    ids = torch.randint(1, vocab_size, (n, seq_len), device=gpu_device)
    return ids[:, :-1], ids[:, 1:]
