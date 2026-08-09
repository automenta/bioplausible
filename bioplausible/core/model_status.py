"""Model status metadata (Plan 8 Track D1).

A small, deliberate vocabulary for the experimental state of every registered
model. Used by sweeps to skip known-broken probes by default (``status:broken``)
and by documentation/parity runners to flag experiments appropriately.

Statuses follow Plan 8 §D1:

- ``stable``: works at the parity target architecture/depth, multi-seed.
- ``experimental``: brand-new or partially-working; results preliminary.
- ``broken``: known to not learn or to crash on canonical tasks; excluded
  from default sweeps unless the user explicitly opts in (``--include-broken``).
- ``deprecated``: kept for historical reference only; do not run.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["STATUS_TAG_PREFIX", "ModelStatus", "status_tag"]


class ModelStatus(StrEnum):
    """Closed vocabulary of model lifecycle statuses (Plan 8 §D1)."""

    STABLE = "stable"
    EXPERIMENTAL = "experimental"
    BROKEN = "broken"
    DEPRECATED = "deprecated"


#: Prefix used for status tags stored on registry metadata (``status:stable``).
STATUS_TAG_PREFIX = "status:"


def status_tag(status: ModelStatus | str) -> str:
    """Render a status as the registry tag form (``status:<value>``)."""
    value = status.value if isinstance(status, ModelStatus) else str(status)
    return f"{STATUS_TAG_PREFIX}{value}"
