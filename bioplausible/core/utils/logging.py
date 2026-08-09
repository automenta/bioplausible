"""Centralized logging helper.

Provides a single ``get_logger`` function that infers the caller's module name,
removing the need for 100+ inline ``logging.getLogger(__name__)`` calls.

Usage:
    from bioplausible.core.utils.logging import get_logger

    logger = get_logger()
    logger.info("message", extra={"key": "value"})
"""

from __future__ import annotations

import inspect
import logging

__all__ = ["get_logger"]


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger for the calling module.

    If ``name`` is not provided, the caller's ``__name__`` is inferred from
    the stack frame, matching the standard ``logging.getLogger(__name__)``
    pattern but without boilerplate.

    Args:
        name: Optional explicit logger name. If ``None``, inferred from caller.

    Returns:
        A ``logging.Logger`` instance.
    """
    if name is None:
        frame = inspect.currentframe()
        try:
            if frame and frame.f_back:
                name = frame.f_back.f_globals.get("__name__", "bioplausible")
            else:
                name = "bioplausible"
        finally:
            del frame
    return logging.getLogger(name)