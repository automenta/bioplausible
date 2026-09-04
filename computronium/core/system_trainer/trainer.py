"""SystemTrainer class for training 5-D composable systems."""

from __future__ import annotations

import dataclasses
from dataclasses import field
from typing import TYPE_CHECKING

import torch

from computronium.core.logging import get_logger
from computronium.core.system_trainer._resume import (
    DOMAIN_EPOCH,
    TrainerSnapshot,
    fold_in,
)

if TYPE_CHECKING:
    from types import TracebackType

    from torch import Tensor

    from computronium.core.system_trainer.config import (
        SystemTrainerConfig,
        _DataProvider,
    )
    from computronium.ontology import System

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

    def _begin_epoch(self) -> None:
        """Seed the stream the epoch's shuffle permutation draws from."""
        if not self.config.resumable:
            return
        torch.manual_seed(
            fold_in(self.config.seed, self.current_epoch, 0, domain=DOMAIN_EPOCH)
        )

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
        self._begin_epoch()

        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_energy = 0.0
        num_samples = 0
        num_samples = 0

        for batch_idx, (x, y) in enumerate(self.train_data):
            if self.config.resumable:
                torch.manual_seed(
                    fold_in(self.config.seed, self.current_epoch, batch_idx)
                )
            x = x.to(self.device)  # ruff: ignore[redefined-loop-name]
            y = y.to(self.device)  # ruff: ignore[redefined-loop-name]

            metrics = self.system.train_step(x, y)
            batch = x.size(0)

            epoch_loss += metrics.get("loss", 0.0) * batch
            epoch_acc += (
                metrics.get("free_accuracy", metrics.get("nudged_fit_accuracy", 0.0))
                * batch
            )
            epoch_energy += metrics.get("energy", 0.0) * batch
            num_samples += batch
            self.global_step += 1

            if self.global_step % self.config.log_every_n_steps == 0:
                logger.info(
                    "Step %d: loss=%.4f, free_acc=%.4f, energy=%.4f",
                    self.global_step,
                    metrics.get("loss", 0.0),
                    metrics.get(
                        "free_accuracy", metrics.get("nudged_fit_accuracy", 0.0)
                    ),
                    metrics.get("energy", 0.0),
                )

        denom = max(num_samples, 1)
        avg_loss = epoch_loss / denom
        avg_acc = epoch_acc / denom
        avg_energy = epoch_energy / denom

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
        val_ce_sum = 0.0
        val_correct = 0
        num_samples = 0

        with torch.no_grad():
            for x, y in self.val_data:
                x = x.to(self.device)  # ruff: ignore[redefined-loop-name]
                y = y.to(self.device)  # ruff: ignore[redefined-loop-name]

                logits = self.system.forward(x)
                ce = torch.nn.functional.cross_entropy(logits, y, reduction="sum")
                val_ce_sum += ce.item()
                val_correct += (logits.argmax(-1) == y).sum().item()
                num_samples += y.size(0)

        denom = max(num_samples, 1)
        val_loss = val_ce_sum / denom
        return {
            "val_loss": val_loss,
            "val_acc": val_correct / denom,
            "val_ppl": torch.exp(torch.tensor(val_loss)).item(),
        }

    def snapshot(self) -> TrainerSnapshot:
        """Capture the full resume state (theta, optimizer state, counters)."""
        theta = {
            name: t.detach().clone() for name, t in self.system.geometry.params.items()
        }
        buffers: dict[str, Tensor] = getattr(
            self.system.update, "_momentum_buffers", {}
        )
        opt_state = {name: t.detach().clone() for name, t in buffers.items()}
        return TrainerSnapshot(
            epoch=self.current_epoch,
            global_step=self.global_step,
            history=tuple(self.history),
            theta=theta,
            opt_state=opt_state,
        )

    @classmethod
    def from_snapshot(
        cls,
        *,
        system: System,
        config: SystemTrainerConfig,
        train_data: _DataProvider,
        snapshot: TrainerSnapshot,
        val_data: _DataProvider | None = None,
    ) -> SystemTrainer:
        """Build a trainer that resumes ``snapshot`` at ``snapshot.epoch``.

        With ``config.resumable`` enabled and the same ``seed`` / data
        stream, continuing training is bitwise identical to never having
        stopped (R11.2.24). ``max_epochs`` counts total epochs.
        """
        trainer = cls(
            system=system, config=config, train_data=train_data, val_data=val_data
        )
        trainer._restore(snapshot)
        return trainer

    def _restore(self, snap: TrainerSnapshot) -> None:
        self.system.geometry.update_params({
            name: t.to(self.device) for name, t in snap.theta.items()
        })
        if snap.opt_state:
            buffers = getattr(self.system.update, "_momentum_buffers", None)
            if buffers is None:
                raise TypeError(
                    f"{type(self.system.update).__name__} carries no "
                    "optimizer state to restore into"
                )
            for name, t in snap.opt_state.items():
                buffers[name] = t.to(self.device).clone()
        self.current_epoch = snap.epoch
        self.global_step = snap.global_step
        self.history = list(snap.history)
        logger.info(
            "Resumed from snapshot: epoch %d, global_step %d",
            snap.epoch,
            snap.global_step,
        )

    def fit(self) -> list[dict[str, float]]:
        """Run the training loop to ``config.max_epochs`` total epochs."""
        logger.info(
            "Starting training for %d epochs (from epoch %d)",
            self.config.max_epochs,
            self.current_epoch,
        )

        while self.current_epoch < self.config.max_epochs:
            self.train_epoch()

        logger.info("Training complete")
        return self.history

    def close(self) -> None:
        """Clean up resources (e.g., move model to CPU, clear CUDA cache)."""
        if hasattr(self, "system") and self.system is not None:  # ruff: ignore[collapsible-if]
            if hasattr(self.system.geometry, "cpu"):
                self.system.geometry.cpu()
        if hasattr(self, "device") and self.device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("SystemTrainer resources cleaned up")

    def __enter__(self) -> SystemTrainer:  # ruff: ignore[non-self-return-type]
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
