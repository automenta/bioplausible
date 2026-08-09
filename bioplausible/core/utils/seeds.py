"""Reproducibility utilities — single source of truth for RNG seeding.

Consolidates the seven previously-duplicated ``_set_seed`` / ``set_all_seeds``
implementations scattered across ``cli/run.py``, ``core/trainer.py``,
``equitile/benchmarks/rigorous.py`` and ``equitile/utils/reproducibility.py``.

Callers that need cudnn + deterministic-algorithms behaviour pass
``deterministic=True``; the simpler CLI/quick-trial callers use the default.
"""

from __future__ import annotations

import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)

__all__ = ["set_all_seeds"]


def set_all_seeds(seed: int = 42, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA) RNGs.

    Args:
        seed: Integer seed applied to every RNG.
        deterministic: When ``True``, also enables
            ``torch.use_deterministic_algorithms`` and the cudnn deterministic
            mode (and disables cudnn benchmarking). Use this only when
            bit-exact reproducibility is required — it can slow training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    if deterministic:
        torch.use_deterministic_algorithms(True)
