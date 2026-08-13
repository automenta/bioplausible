"""Reproducibility utilities — single source of truth for RNG seeding.

Consolidates the seven previously-duplicated ``_set_seed`` / ``set_all_seeds``
implementations scattered across ``cli/run.py``, ``core/trainer.py``,
``equitile/benchmarks/rigorous.py`` and ``equitile/utils/reproducibility.py``.

Callers that need cudnn + deterministic-algorithms behaviour pass
``deterministic=True``; the simpler CLI/quick-trial callers use the default.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from bioplausible.core.logging import get_logger
from bioplausible.utils import seed_everything as _seed_everything

logger = get_logger()

__all__ = ["set_all_seeds"]


def set_all_seeds(seed: int = 42, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA) RNGs.

    Thin adapter over :func:`bioplausible.utils.seed_everything` (the single
    seeding API). The deterministic path delegates verbatim; the
    non-deterministic path applies the minimal RNG seed subset when a caller
    does not need cuDNN/algorithm determinism.

    Args:
        seed: Integer seed applied to every RNG.
        deterministic: When ``True``, also enables
            ``torch.use_deterministic_algorithms`` and the cuDNN deterministic
            mode (and disables cuDNN benchmarking). Use this only when
            bit-exact reproducibility is required — it can slow training.
    """
    if deterministic:
        _seed_everything(seed, device="cpu", deterministic=True)
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
