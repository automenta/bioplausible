"""Conditional execution dashboard with a dependency-free default renderer."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

PURE: Final = True

if TYPE_CHECKING or PURE:
    from computronium.execution._pure_dashboard import BRAILLE_FRAMES, Dashboard
else:
    from computronium.execution._rich_dashboard import Dashboard

    BRAILLE_FRAMES: tuple[str, ...] = ()

__all__ = ["BRAILLE_FRAMES", "DASHBOARD", "PURE", "Dashboard", "logger"]

logger = logging.getLogger(__name__)
DASHBOARD = Dashboard()
