"""Verify the reported ``peak_memory_mb`` against torch's CUDA allocator.

Experiment plan §6: we must know what ``peak_memory_mb`` actually measures
before building scaling-law analysis on it. This probe trains one short epoch
of a ``CoreTrainer`` backprop run with ``track_memory=True`` and compares the
reported per-batch peak against ``torch.cuda.max_memory_allocated()``.

Usage:
    uv run python scripts/verify_memory_measurement.py
"""

from __future__ import annotations

import logging
import sys

import torch

import computronium.zoo  # ruff: ignore[unused-import]  (registration side effect; mirrors scripts/preliminary_run.py)
from computronium.core.trainer import CoreTrainer, TrainerConfig

_ALLOWED_RATIO_LO: float = 0.9
_ALLOWED_RATIO_HI: float = 1.1

logger = logging.getLogger("verify_memory")


def _main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    if not torch.cuda.is_available():
        logger.error("Memory verification requires CUDA.")
        raise SystemExit(1)

    config = TrainerConfig(
        model="backprop_mlp",
        task="mnist",
        epochs=1,
        batch_size=128,
        model_kwargs={"input_dim": 784, "hidden_dim": 256, "output_dim": 10},
        device="cuda",
        run_validation=False,
        track_energy=True,
        save_checkpoints=False,
        seed=0,
    )
    trainer = CoreTrainer(config)
    metrics = trainer.fit()
    if not metrics:
        logger.error("No metrics produced.")
        raise SystemExit(1)

    reported = metrics[-1].peak_memory_mb
    allocator = torch.cuda.max_memory_allocated() / (1024 * 1024)
    reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)

    logger.info(
        "reported peak_memory_mb : %s", f"{reported:.1f}" if reported else "None"
    )
    logger.info("max_memory_allocated() : %.1f", allocator)
    logger.info("max_memory_reserved()  : %.1f", reserved)

    if not reported:
        logger.info("RESULT: UNVERIFIED (reported value missing)")
        return

    ratio = reported / max(allocator, 1e-9)
    within = _ALLOWED_RATIO_LO <= ratio <= _ALLOWED_RATIO_HI
    logger.info("ratio                   : %.3f", ratio)
    verdict = "PASS" if within else "FAIL"
    logger.info(
        "RESULT: %s (within 10%% of max_memory_allocated)",
        verdict,
    )


if __name__ == "__main__":
    _main()
