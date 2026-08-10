"""NiceGUI compatibility shims for Python 3.14.

``nicegui`` depends on ``vbuild`` which calls ``pkgutil.find_loader`` at import
time. That attribute was removed in Python 3.12+, so importing nicegui on the
project's required Python (>=3.14) raises ``AttributeError``. This shim injects
an importlib-based equivalent before nicegui is imported.

Call :func:`apply_compat_shims` at the very top of ``demo/main.py``.
"""

from __future__ import annotations

import importlib.util
import pkgutil


def apply_compat_shims() -> None:
    """Restore ``pkgutil.find_loader`` if missing (Python 3.12+)."""
    if hasattr(pkgutil, "find_loader"):
        return

    def find_loader(name: str) -> object | None:
        try:
            spec = importlib.util.find_spec(name)
        except ImportError, ValueError:
            return None
        return spec.loader if spec is not None else None

    pkgutil.find_loader = find_loader  # type: ignore[attr-defined]
