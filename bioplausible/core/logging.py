"""Logging helpers.

A thin wrapper over the stdlib ``logging`` module. Existing call-sites that
already use ``logging.getLogger(__name__)`` are idiomatic and remain
unchanged — this helper simply removes the ``__name__`` boilerplate for new
modules via automatic caller-frame introspection.

Example:
    from bioplausible.core.logging import get_logger
    logger = get_logger()  # picks up the caller module's __name__
"""

from __future__ import annotations

import inspect
import logging

__all__ = ["get_logger"]


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger, defaulting to the caller module's ``__name__``.

    Args:
        name: Optional explicit logger name. When ``None`` (the default),
            the logger is named after the calling module's ``__name__`` via
            stack inspection — equivalent to ``logging.getLogger(__name__)``
            without forcing every module to spell it out.

    Returns:
        A ``logging.Logger`` instance.
    """
    if name is None:
        frame = inspect.currentframe()
        if frame is not None and frame.f_back is not None:
            name = frame.f_back.f_globals.get("__name__", "bioplausible")
        else:
            name = "bioplausible"
    return logging.getLogger(name)
