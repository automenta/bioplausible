"""
PyTorch Lightning Callbacks for Bioplausible MLOps

Standard PL callback implementations replacing ad-hoc visualisation and
analysis paths with idiomatic Lightning hooks.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from pytorch_lightning.callbacks import Callback

from computronium.core.logging import get_logger

logger = get_logger()


if TYPE_CHECKING:
    import pytorch_lightning as pl

__all__ = [
    "BioPrecisionCallback",
    "BioPredictionWriter",
    "EnergyConvergenceCallback",
    "logger",
]


class EnergyConvergenceCallback(Callback):
    """
    Stops training when the equilibrium energy of EqProp/Hebbian
    models converges below a threshold.
    """

    def __init__(self, monitor: str = "train_energy", patience: int = 3):
        """
        Args:
            monitor: Metric name to watch.
            patience: Epochs with no improvement before stopping.
        """
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self._best = float("inf")
        self._counter = 0

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        val = trainer.callback_metrics.get(self.monitor)
        if val is None:
            return

        current = float(val)
        if current < self._best:
            self._best = current
            self._counter = 0
        else:
            self._counter += 1

        if self._counter >= self.patience:
            trainer.should_stop = True
            logger.info("[BioPrecisionCallback] Down-casting to FP32")


class BioPrecisionCallback(Callback):
    """
    Safety callback that down-casts model weights to FP32 before
    each training step for sensitive bio-plausible optimizers.
    """

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if trainer.precision == "32-true":
            return

        bio_keywords = ("eqprop", "hebbian", "chl", "ep_", "feedback", "smep")
        opt_name = getattr(pl_module, "optimizer_name", "").lower()

        if any(kw in opt_name for kw in bio_keywords):
            logger.info(
                "[BioPrecisionCallback] Down-casting %s to FP32",
                pl_module.__class__.__name__,
            )
            pl_module.to(torch.float32)


class BioPredictionWriter(Callback):
    """
    Writes predictions to disk asynchronously for downstream analysis.
    """

    def __init__(self, output_dir: str = "./predictions"):
        super().__init__()
        self.output_dir = output_dir
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        self._buffer: list[dict[str, object]] = []

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: object,
        batch: object,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        x, y = batch
        with torch.no_grad():
            logits = pl_module(x)
            preds = logits.argmax(dim=1)

        for i in range(y.size(0)):
            self._buffer.append({
                "batch_idx": batch_idx,
                "true": y[i].item(),
                "pred": preds[i].item(),
            })

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        path = Path(self.output_dir) / f"epoch_{trainer.current_epoch}.jsonl"
        with path.open("w") as f:
            for rec in self._buffer:
                f.write(json.dumps(rec) + "\n")
        self._buffer.clear()
