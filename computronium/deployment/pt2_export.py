"""
PT2 (torch.export) export utilities.

Replaces the deprecated torch.jit path, which is unsupported on Python 3.14+.
"""

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from pathlib import Path


def export_to_pt2(
    model: nn.Module,
    input_sample: torch.Tensor | tuple,
    path: str | Path,
) -> str:
    """
    Export model to a serialized ``torch.export`` (PT2) program.

    Replaces the deprecated ``torch.jit`` path, which is unsupported on
    Python 3.14+.

    Args:
        model: Model to export.
        input_sample: Example input for tracing.
        path: Output path (should end in ``.pt2``).

    Returns:
        Path to exported model.
    """
    model.eval()

    if isinstance(input_sample, torch.Tensor):  # ruff: ignore[if-else-block-instead-of-if-exp]
        args = (input_sample,)
    else:
        args = input_sample

    program = torch.export.export(model, args)
    torch.export.save(program, str(path))

    return str(path)


__all__ = [
    "export_to_pt2",
]
