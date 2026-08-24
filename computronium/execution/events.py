"""Event-sink protocol decoupling decision modules from the UI singleton.

Decision modules (:class:`~computronium.execution.strategy.ExecutionStrategy`
and :class:`~computronium.execution.engine.ExecutionEngine`) emit lifecycle and
insight events through an :class:`EventSink` instead of importing the global
``DASHBOARD`` object, so headless runs (CI, distributed sweeps, standard tests)
can inject a :class:`NullEventSink` and never pull in the UI stack.
"""

from __future__ import annotations

from typing import Protocol

__all__ = ["EventSink", "NullEventSink", "dashboard_sink"]


class EventSink(Protocol):
    """Lifecycle and insight events a decision engine may emit."""

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def update(self) -> None: ...

    def log(self, message: str, style: str = "") -> None: ...

    def set_trial(
        self,
        trial_id: str,
        model: str,
        task: str,
        tier: str,
        params: dict[str, object],
    ) -> None: ...

    def update_progress(
        self, epoch: int, total_epochs: int, metrics: dict[str, float]
    ) -> None: ...

    def complete_trial(self, status: str, metrics: dict[str, object]) -> None: ...

    def set_insight(self, text: str) -> None: ...

    def set_system_status(self, status: str, style: str = "white") -> None: ...


class NullEventSink:
    """No-op sink used for headless execution (CI, sweeps, tests)."""

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def update(self) -> None: ...

    def log(self, message: str, style: str = "") -> None: ...

    def set_trial(
        self,
        trial_id: str,
        model: str,
        task: str,
        tier: str,
        params: dict[str, object],
    ) -> None: ...

    def update_progress(
        self, epoch: int, total_epochs: int, metrics: dict[str, float]
    ) -> None: ...

    def complete_trial(self, status: str, metrics: dict[str, object]) -> None: ...

    def set_insight(self, text: str) -> None: ...

    def set_system_status(self, status: str, style: str = "white") -> None: ...


def dashboard_sink() -> EventSink:
    """Return the ``DASHBOARD`` singleton as an :class:`EventSink`.

    Imported lazily so library code never drags in the UI stack; use at the app
    boundary (e.g. the ``biopl-scientist`` entry point) to re-enable the live UI.
    """
    from computronium.execution.dashboard import DASHBOARD

    return DASHBOARD
