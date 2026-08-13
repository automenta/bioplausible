"""Unified Checkpoint TypedDict and save/load utilities (Phase C.3).

Provides a single ``Checkpoint`` TypedDict and ``save_checkpoint`` /
``load_checkpoint`` helpers that all trainer paths can use, replacing
the three ad-hoc formats in ``CoreTrainer``, ``EquiTile``, and
``ExecutionEngine``.
"""

from __future__ import annotations

import pathlib
import shutil
import tempfile
import zipfile
from contextlib import contextmanager
from typing import TYPE_CHECKING, NotRequired, Required, TypedDict

import torch
from torch import Tensor

from bioplausible.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = get_logger()


class Checkpoint(TypedDict):
    """Unified checkpoint format for all models and trainers.

    ``model_state_dict`` is the only required key — the invariant every
    checkpoint satisfies.  All other fields are optional; additional
    model-specific keys can be stored in ``extra``.
    """

    model_state_dict: Required[dict[str, Tensor]]
    optimizer_state_dict: NotRequired[dict[str, object] | None]
    scheduler_state_dict: NotRequired[dict[str, object] | None]
    config: NotRequired[dict[str, object]]
    epoch: NotRequired[int]
    global_step: NotRequired[int]
    metrics: NotRequired[dict[str, object]]
    metadata: NotRequired[dict[str, object]]
    extra: NotRequired[dict[str, object]]


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


def _dir_artifact(root: pathlib.Path, prefix: str) -> str | None:
    """Return the ``model.pt`` path inside a matching artifact directory."""
    for item in root.iterdir():
        if item.is_dir() and item.name.startswith(prefix):
            p = item / "model.pt"
            return str(p) if p.exists() else None
    return None


def _zip_artifact(
    root: pathlib.Path, prefix: str
) -> tuple[str | None, pathlib.Path | None]:
    """Extract a matching zipped artifact to a temp dir; return (path, temp_dir)."""
    for item in root.iterdir():
        if item.is_dir() or item.suffix != ".zip" or not item.name.startswith(prefix):
            continue
        temp_dir = pathlib.Path(tempfile.mkdtemp())
        try:
            with zipfile.ZipFile(item, "r") as zf:
                zf.extract("model.pt", temp_dir)
            return str(temp_dir / "model.pt"), temp_dir
        except (OSError, RuntimeError, KeyError) as e:
            logger.warning("Failed to extract artifact %s: %s", item, e)
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None, None
    return None, None


@contextmanager
def find_trial_artifact(
    trial_id: int, artifact_dir: str | pathlib.Path = "artifacts"
) -> Iterator[str | None]:
    """Yield the path to a trial's saved ``model.pt`` (dir or zipped artifact).

    Scans ``artifact_dir`` for a ``trial_{trial_id}_*`` entry, preferring an
    on-disk directory over a zipped archive; a zip is extracted to a temporary
    directory removed when the context exits. Yields ``None`` when no artifact
    is found.

    Args:
        trial_id: Trial id whose artifact is sought.
        artifact_dir: Directory holding per-trial artifact subdirs/zips.

    Yields:
        The path to ``model.pt`` (``str``) or ``None`` if not found.
    """
    root = pathlib.Path(artifact_dir)
    if not root.exists():
        yield None
        return

    prefix = f"trial_{trial_id}_"
    found = _dir_artifact(root, prefix)
    temp_dir = None
    if found is not None:
        yield found
        return

    found, temp_dir = _zip_artifact(root, prefix)
    try:
        yield found
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)


__all__ = [
    "Checkpoint",
    "find_trial_artifact",
    "load_checkpoint",
    "load_checkpoint_into_model",
    "save_checkpoint",
]
