"""SystemTrainer class for training 5-D composable systems."""

from __future__ import annotations

import dataclasses
from dataclasses import field
from typing import TYPE_CHECKING

import torch

from computronium.core.logging import get_logger
from computronium.ontology import System

if TYPE_CHECKING:
    from types import TracebackType

logger = get_logger()


@dataclasses.dataclass
class SystemTrainer:
    """Trainer for 5-D composable systems.

    Orchestrates the pipeline:
        Substrate.forward_op → Geometry.route → StateDynamics.settle
        → CreditAssignment.compute_pseudo_gradient → ParameterUpdate.step

    The trainer accepts a pre-composed System and data providers,
    enabling clean separation of model architecture from training loop.
    """

    system: System
    config: SystemTrainerConfig
    train_data: _DataProvider
    val_data: _DataProvider | None = None

    # Training state
    current_epoch: int = field(default=0, init=False)
    global_step: int = field(default=0, init=False)
    history: list[dict[str, float]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._setup_device()
        self._set_seed()
        # Move system components to device
        if hasattr(self.system.geometry, "to"):
            self.system.geometry.to(self.device)  # type: ignore[attr-defined]

    def _setup_device(self) -> None:
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        logger.info("SystemTrainer using device: %s", self.device)

    def _set_seed(self) -> None:
        torch.manual_seed(self.config.seed)
        if self.config.deterministic:
            torch.use_deterministic_algorithms(True)

    def train_epoch(self) -> dict[str, float]:
        """Run one training epoch."""
        self.system.geometry.train()

        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_energy = 0.0
        num_batches = 0

        for _, (x, y) in enumerate(self.train_data):
            x = x.to(self.device)
            y = y.to(self.device)

            metrics = self.system.train_step(x, y)

            epoch_loss += metrics.get("loss", 0.0)
            epoch_acc += metrics.get("accuracy", 0.0)
            epoch_energy += metrics.get("energy", 0.0)
            num_batches += 1
            self.global_step += 1

            if self.global_step % self.config.log_every_n_steps == 0:
                logger.info(
                    "Step %d: loss=%.4f, acc=%.4f, energy=%.4f",
                    self.global_step,
                    metrics.get("loss", 0.0),
                    metrics.get("accuracy", 0.0),
                    metrics.get("energy", 0.0),
                )

        avg_loss = epoch_loss / max(num_batches, 1)
        avg_acc = epoch_acc / max(num_batches, 1)
        avg_energy = epoch_energy / max(num_batches, 1)

        epoch_metrics = {
            "epoch": self.current_epoch,
            "train_loss": avg_loss,
            "train_acc": avg_acc,
            "train_energy": avg_energy,
            "global_step": self.global_step,
        }

        if self.val_data is not None:
            val_metrics = self.validate()
            epoch_metrics.update(val_metrics)

        self.history.append(epoch_metrics)
        self.current_epoch += 1

        logger.info(
            "Epoch %d: train_loss=%.4f, train_acc=%.4f, train_energy=%.4f",
            self.current_epoch,
            avg_loss,
            avg_acc,
            avg_energy,
        )

        return epoch_metrics

    def validate(self) -> dict[str, float]:
        """Run validation epoch."""
        if self.val_data is None:
            return {}

        self.system.geometry.eval()
        val_loss = 0.0
        val_acc = 0.0
        num_batches = 0

        with torch.no_grad():
            for x, y in self.val_data:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.system.forward(x)
                loss = torch.nn.functional.cross_entropy(logits, y)
                acc = (logits.argmax(-1) == y).float().mean().item()

                val_loss += loss.item()
                val_acc += acc
                num_batches += 1

        return {
            "val_loss": val_loss / max(num_batches, 1),
            "val_acc": val_acc / max(num_batches, 1),
        }

    def fit(self) -> list[dict[str, float]]:
        """Run full training loop."""
        logger.info("Starting training for %d epochs", self.config.max_epochs)

        for _ in range(self.config.max_epochs):
            self.train_epoch()

        logger.info("Training complete")
        return self.history

    def close(self) -> None:
        """Clean up resources (e.g., move model to CPU, clear CUDA cache)."""
        if hasattr(self, "system") and self.system is not None:
            if hasattr(self.system.geometry, "cpu"):
                self.system.geometry.cpu()
        if hasattr(self, "device") and self.device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("SystemTrainer resources cleaned up")

    def __enter__(self) -> SystemTrainer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        self.close()
        return False


__all__ = ["SystemTrainer"]
