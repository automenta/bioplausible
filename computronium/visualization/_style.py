"""Shared gallery style: one place for colors, layout, and saving."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

COLOR_ARM = "#2f6f4f"
COLOR_CONTRAST = "#7a4f9e"
COLOR_WALLED = "#c44e52"
COLOR_FEASIBLE = "#55a868"
COLOR_CHANCE = "#888888"

CHANCE_LINESTYLE = "--"
DPI = 150


def apply_style(fig: Figure) -> None:
    fig.tight_layout()


def save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)


def chance_line(ax: Axes, chance: float, label: str) -> None:
    ax.axhline(chance, color=COLOR_CHANCE, linestyle=CHANCE_LINESTYLE, linewidth=1)
    ax.text(0.01, chance, label, fontsize=8, color=COLOR_CHANCE, va="bottom")
