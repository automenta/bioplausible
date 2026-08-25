"""Execution dashboard package with pluggable backends."""

# ruff: noqa: RUF067
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Final

PURE: Final = os.environ.get("COMPUTRONIUM_PURE_DASHBOARD", "true").lower() == "true"

if TYPE_CHECKING or PURE:
    from computronium.execution.dashboard._pure import BRAILLE_FRAMES, Dashboard
else:
    from computronium.execution.dashboard._rich import Dashboard

    BRAILLE_FRAMES: tuple[str, ...] = ()

__all__ = [
    "BRAILLE_FRAMES",
    "DASHBOARD",
    "PURE",
    "Dashboard",
    "logger",
]

logger = logging.getLogger(__name__)
DASHBOARD = Dashboard()
