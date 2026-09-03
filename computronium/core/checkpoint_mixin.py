"""Checkpoint mixin — unified save/load using core.checkpoint.Checkpoint.

Provides ``save_checkpoint`` and ``load_checkpoint`` methods that wrap the
canonical ``core.checkpoint`` helpers, eliminating the 6+ ad-hoc
implementations scattered across the codebase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from computronium.core.checkpoint import (
    Checkpoint,
    load_checkpoint_into_model,
    save_checkpoint,
)
from computronium.core.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    import torch

logger = get_logger()


class CheckpointMixin:
    """Mixin providing standardized checkpoint save/load.

    Expected attributes on the host class:
        - ``config`` with optional ``to_dict()`` method
        - ``_step_count: int`` (training step counter)
        - ``_epoch: int`` (current epoch, optional)
    """

    def save_checkpoint(
        self: Self,
        path: str | Path,
        metadata: dict | None = None,
        *,
        optimizer_state: dict | None = None,
        scheduler_state: dict | None = None,
    ) -> None:
        """Save model checkpoint to disk.

        Args:
            path: Destination path (typically ``.pt``).
            metadata: Optional extra metadata to store.
            optimizer_state: Optional optimizer state dict.
            scheduler_state: Optional scheduler state dict.
        """
        config_dict = {}
        if hasattr(self, "config") and self.config is not None:
            if hasattr(self.config, "to_dict"):
                config_dict = self.config.to_dict()
            elif hasattr(self.config, "__dataclass_fields__"):
                # Frozen dataclass
                import dataclasses

                config_dict = dataclasses.asdict(self.config)

        epoch = getattr(self, "_epoch", 0)
        global_step = getattr(self, "_step_count", 0)

        ckpt: Checkpoint = {
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer_state,
            "scheduler_state_dict": scheduler_state,
            "config": config_dict,
            "epoch": epoch,
            "global_step": global_step,
            "metadata": metadata or {},
        }

        save_checkpoint(path, ckpt)
        logger.info(
            "Checkpoint saved: %s (epoch %d, step %d)", path, epoch, global_step
        )

    def load_checkpoint(
        self: Self,
        path: str | Path,
        *,
        device: torch.device | str | None = None,
        strict: bool = True,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
    ) -> Checkpoint:
        """Load model checkpoint from disk.

        Args:
            path: Checkpoint path.
            device: Target device (default: current model device).
            strict: Whether to enforce strict key matching.
            load_optimizer: Whether to return optimizer state in checkpoint.
            load_scheduler: Whether to return scheduler state in checkpoint.

        Returns:
            The full checkpoint dict (contains optimizer/scheduler states if present).
        """
        if device is None:
            device = next(self.parameters()).device

        checkpoint = load_checkpoint_into_model(
            path, self, strict=strict, map_location=device
        )

        if hasattr(self, "_epoch"):
            self._epoch = checkpoint.get("epoch", 0)
        if hasattr(self, "_step_count"):
            self._step_count = checkpoint.get("global_step", 0)

        logger.info(
            "Checkpoint loaded: %s (epoch %d, step %d)",
            path,
            self._epoch,
            self._step_count,
        )
        return checkpoint
