"""Tests for load_weights zoo helper."""

from pathlib import Path

import torch

from computronium.zoo import load_weights


def test_load_weights_noop_on_empty_path():
    """load_weights should be a no-op when path is None or empty."""
    model = torch.nn.Linear(10, 5)
    load_weights(model, "")
    # Should not raise
    assert True


def test_load_weights_freeze_layers(tmp_path: Path):
    """freeze_layers=True should freeze loaded parameters."""
    model = torch.nn.Linear(10, 5)
    state = model.state_dict()
    save_path = str(tmp_path / "test.pt")
    torch.save(state, save_path)

    model2 = torch.nn.Linear(10, 5)
    load_weights(model2, save_path, freeze_layers=True)

    for name, param in model2.named_parameters():
        if name in state:
            assert not param.requires_grad, f"{name} should be frozen"
