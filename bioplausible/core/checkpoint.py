"""Unified Checkpoint TypedDict and save/load utilities (Phase C.3).

Provides a single ``Checkpoint`` TypedDict and ``save_checkpoint`` /
``load_checkpoint`` helpers that all trainer paths can use, replacing
the three ad-hoc formats in ``CoreTrainer``, ``EquiTile``, and
``ExecutionEngine``.
"""

import logging
import pathlib
from typing import TypedDict

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class Checkpoint(TypedDict, total=False):
    """Unified checkpoint format for all models and trainers.

    All fields except ``model_state_dict`` are optional — the only
    invariant is that ``model_state_dict`` is present.  Additional
    model-specific keys can be stored in ``extra``.
    """

    model_state_dict: dict[str, Tensor]
    optimizer_state_dict: dict[str, object] | None
    scheduler_state_dict: dict[str, object] | None
    config: dict[str, object]
    epoch: int
    global_step: int
    metrics: dict[str, object]
    metadata: dict[str, object]
    extra: dict[str, object]


def save_checkpoint(
    path: str | pathlib.Path,
    checkpoint: Checkpoint,
    *,
    mkdir: bool = True,
) -> None:
    """Save a checkpoint to disk.

    Parameters
    ----------
    path : str | Path
        Destination path (typically ``.pt``).
    checkpoint : Checkpoint
        The checkpoint data to persist.
    mkdir : bool
        If True, create parent directories automatically.
    """
    path_obj = pathlib.Path(path)
    if mkdir:
        path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(checkpoint), path_obj)
    logger.info("Checkpoint saved: %s  (epoch %s)", path_obj, checkpoint.get("epoch"))


def load_checkpoint(
    path: str | pathlib.Path,
    *,
    map_location: str | torch.device | None = None,
) -> Checkpoint:
    """Load a checkpoint from disk.

    Parameters
    ----------
    path : str | Path
        Path to the saved checkpoint.
    map_location : str | torch.device | None
        Device mapping (passed through to ``torch.load``).

    Returns
    -------
    Checkpoint
        The loaded checkpoint dict.
    """
    path_obj = pathlib.Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path_obj}")
    data = torch.load(path_obj, map_location=map_location, weights_only=False)
    return Checkpoint(data)


def load_checkpoint_into_model(
    path: str | pathlib.Path,
    model: torch.nn.Module,
    *,
    strict: bool = True,
    map_location: str | torch.device | None = None,
) -> Checkpoint:
    """Load a checkpoint and restore model state in one call.

    Parameters
    ----------
    path : str | Path
        Checkpoint path.
    model : nn.Module
        The model to load state into (in-place).
    strict : bool
        Whether to enforce strict key matching (passed to
        ``load_state_dict``).
    map_location : str | torch.device | None
        Device mapping.

    Returns
    -------
    Checkpoint
        The full checkpoint (callers can access ``optimizer_state_dict``,
        ``epoch``, etc.).
    """
    checkpoint = load_checkpoint(path, map_location=map_location)
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=strict)  # type: ignore[arg-type]
    return checkpoint


__all__ = [
    "Checkpoint",
    "load_checkpoint",
    "load_checkpoint_into_model",
    "save_checkpoint",
]
