"""Tests for the unified Checkpoint module (core/checkpoint.py)."""

import pathlib
import tempfile

import pytest
import torch
from torch import nn

from bioplausible.core.checkpoint import (
    Checkpoint,
    find_trial_artifact,
    load_checkpoint,
    load_checkpoint_into_model,
    save_checkpoint,
)


class _SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


@pytest.fixture
def model_and_state():
    m = _SimpleModel()
    state = m.state_dict()
    return m, state


@pytest.fixture
def tmp_path_obj():
    with tempfile.TemporaryDirectory() as d:
        yield pathlib.Path(d)


def test_save_and_load_checkpoint(model_and_state, tmp_path_obj):
    """Round-trip save/load preserves all fields."""
    _, state = model_and_state
    ckpt: Checkpoint = {
        "model_state_dict": state,
        "optimizer_state_dict": {"lr": 0.001},
        "epoch": 5,
        "global_step": 1000,
        "metrics": {"val_loss": 0.123},
        "config": {"model": "TestMLP", "epochs": 10},
    }
    path = tmp_path_obj / "test.pt"
    save_checkpoint(path, ckpt)

    loaded = load_checkpoint(path)
    assert loaded["epoch"] == 5
    assert loaded["global_step"] == 1000
    assert loaded["metrics"]["val_loss"] == 0.123  # type: ignore[index]
    assert len(loaded["model_state_dict"]) == len(state)


def test_load_checkpoint_into_model(model_and_state, tmp_path_obj):
    """load_checkpoint_into_model restores model state."""
    model, state = model_and_state
    ckpt: Checkpoint = {"model_state_dict": state}
    path = tmp_path_obj / "test_model.pt"
    save_checkpoint(path, ckpt)

    # Create a fresh model and load
    fresh = _SimpleModel()
    # Verify weights differ initially (random init)
    with torch.no_grad():
        fresh.fc.weight.copy_(torch.randn_like(fresh.fc.weight))
    old_weight = fresh.fc.weight.clone()

    loaded = load_checkpoint_into_model(path, fresh)
    assert "epoch" not in loaded
    # After loading, weights should match
    assert torch.allclose(fresh.fc.weight, model.fc.weight)
    assert not torch.allclose(fresh.fc.weight, old_weight)


def test_save_checkpoint_creates_parent_dir(tmp_path_obj):
    """save_checkpoint creates parent directories when mkdir=True."""
    nested = tmp_path_obj / "nested" / "dir" / "ckpt.pt"
    ckpt: Checkpoint = {"model_state_dict": {}}
    save_checkpoint(nested, ckpt, mkdir=True)
    assert nested.exists()


def test_load_nonexistent_checkpoint(tmp_path_obj):
    """load_checkpoint raises FileNotFoundError for missing path."""
    missing = tmp_path_obj / "does_not_exist.pt"
    with pytest.raises(FileNotFoundError, match="does_not_exist"):
        load_checkpoint(missing)


def test_checkpoint_minimal_fields(tmp_path_obj):
    """A checkpoint with only model_state_dict is valid."""
    ckpt: Checkpoint = {"model_state_dict": {"w": torch.tensor(1.0)}}
    path = tmp_path_obj / "minimal.pt"
    save_checkpoint(path, ckpt)
    loaded = load_checkpoint(path)
    assert "model_state_dict" in loaded
    assert torch.equal(loaded["model_state_dict"]["w"], torch.tensor(1.0))  # type: ignore[index]


def test_find_trial_artifact_directory(tmp_path_obj):
    """A directory artifact resolves to its model.pt and survives the context."""
    art = tmp_path_obj / "artifacts"
    art.mkdir()
    d = art / "trial_7_mlp"
    d.mkdir()
    target = d / "model.pt"
    target.write_text("weights")
    with find_trial_artifact(7, art) as p:
        assert p == str(target)
        assert pathlib.Path(p).read_text(encoding="utf-8") == "weights"


def test_find_trial_artifact_zip_cleans_temp(tmp_path_obj):
    """A zipped artifact extracts to a temp dir that is removed on exit."""
    import zipfile

    art = tmp_path_obj / "artifacts"
    art.mkdir()
    with zipfile.ZipFile(art / "trial_9_mlp.zip", "w") as zf:
        zf.writestr("model.pt", "zipped")
    with find_trial_artifact(9, art) as p:
        assert p is not None and pathlib.Path(p).read_text(encoding="utf-8") == "zipped"
        assert list(art.glob("tmp*")) == []
    assert not pathlib.Path(p).exists()


def test_find_trial_artifact_prefers_dir(tmp_path_obj):
    """A directory artifact wins over a same-suffix zip when both exist."""
    import zipfile

    art = tmp_path_obj / "artifacts"
    art.mkdir()
    d = art / "trial_8_mlp"
    d.mkdir()
    (d / "model.pt").write_text("dir")
    with zipfile.ZipFile(art / "trial_8_mlp.zip", "w") as zf:
        zf.writestr("model.pt", "zip")
    with find_trial_artifact(8, art) as p:
        assert p == str(d / "model.pt")


def test_find_trial_artifact_missing(tmp_path_obj):
    """Returns None (yielded, not raised) when no artifact or no dir exists."""
    art = tmp_path_obj / "artifacts"
    art.mkdir()
    with find_trial_artifact(99, art) as p:
        assert p is None
    with find_trial_artifact(1, tmp_path_obj / "nope") as p:
        assert p is None
