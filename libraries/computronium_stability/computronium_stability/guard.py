"""Calibrated stability guard for unattended training runs.

Primary API:
    from computronium_stability import attach, StabilityVerdict

    guard = attach(model)
    verdict = guard.check(step_state)
    if verdict.kill:
        # Handle instability
        ...

Calibration:
- Default threshold τ=1.029 calibrated on:
  - 16 real settling coordinates (8 substrates × 2 settling dynamics)
  - Windowed growth = 1.000 on all 16 → false-kill rate 0%
  - Non-normal linear dynamics (Ginibre ensemble) ROC calibration
- Scope: energy-minimization coordinates and non-normal linear dynamics only
- General-transformer collapse detection: future calibration work, not a v1 claim
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor

from computronium_stability.spectral_radius import (
    SpectralRadiusEstimator,
)

# Type aliases for framework-agnostic state
StepState = dict[str, Any]
TransitionFn = Callable[[StepState], StepState]

StatisticKind = Literal["fast_proxy", "windowed_growth"]
DEFAULT_TAU = 1.029


@dataclass(frozen=True, slots=True)
class GuardDecision:
    """Outcome of one guard probe."""

    statistic: float
    threshold: float
    kill: bool
    statistic_kind: StatisticKind


@dataclass(frozen=True, slots=True)
class StabilityVerdict:
    """Result of a stability check on a training step."""

    kill: bool
    decisions: tuple[GuardDecision, ...]
    max_statistic: float
    threshold: float
    step: int

    def __bool__(self) -> bool:
        """True if any decision triggers a kill."""
        return self.kill


class StabilityGuard:
    """Threshold guard on stability statistics of a dynamical system.

    Two statistic modes:
    - ``fast_proxy``: one-step Jacobian-vector gain (cheap, ~10× step cost)
    - ``windowed_growth``: peak activity growth over a settling window
      (tracks asymptotic divergence directly; separates good/unstable runs
      that the one-step proxy conflates)

    Kills a run when any statistic exceeds the calibrated threshold.

    Example:
        guard = StabilityGuard(threshold=1.029, statistic="windowed_growth", window=10)
        verdict = guard.check(step_state, transition_fn)
    """

    def __init__(
        self,
        threshold: float = DEFAULT_TAU,
        estimator: SpectralRadiusEstimator | None = None,
        statistic: StatisticKind = "windowed_growth",
        window: int = 10,
    ):
        self.threshold = threshold
        self.estimator = estimator or SpectralRadiusEstimator(fast_mode=True)
        self.statistic = statistic
        self.window = window

    def probe(
        self,
        transition_fn: TransitionFn,
        state: StepState,
    ) -> float:
        """Compute the guard statistic at the given state."""
        match self.statistic:
            case "fast_proxy":
                return self._fast_proxy(transition_fn, state)
            case "windowed_growth":
                return self._windowed_growth(transition_fn, state)

    def _fast_proxy(
        self,
        transition_fn: TransitionFn,
        state: StepState,
    ) -> float:
        """Fast proxy: single perturbation step."""
        activity = self._extract_activity(state)
        if activity is None:
            return 0.0

        eps = self.estimator.perturbation_scale
        v = torch.randn_like(activity)
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)

        x_perturbed = activity + eps * v
        state_perturbed = {**state, "x": x_perturbed}

        with torch.no_grad():
            next_state = transition_fn(state)
            next_perturbed = transition_fn(state_perturbed)

        next_act = self._extract_activity(next_perturbed)
        base_act = self._extract_activity(next_state)
        if next_act is None or base_act is None:
            return 0.0
        delta = next_act - base_act
        Jv = delta / eps
        return Jv.norm(dim=-1).mean().item()

    def _windowed_growth(
        self,
        transition_fn: TransitionFn,
        state: StepState,
    ) -> float:
        """Windowed growth: peak activity growth over settling window."""
        activity = self._extract_activity(state)
        if activity is None:
            return 1.0

        base_norm = torch.linalg.vector_norm(activity) + 1e-12
        peak = 1.0

        with torch.no_grad():
            current = state
            for _ in range(self.window):
                nxt = transition_fn(current)
                next_activity = self._extract_activity(nxt)
                if next_activity is None:
                    break
                growth = float(torch.linalg.vector_norm(next_activity)) / float(base_norm)
                peak = max(peak, growth)
                current = nxt
        return peak

    def _extract_activity(self, state: StepState) -> Tensor | None:
        """Extract the primary activity tensor from step state.

        Tries common keys in order. Override by subclassing or wrapping.
        """
        for key in ("x", "activity", "hidden", "output", "y"):
            if key in state and isinstance(state[key], Tensor):
                return state[key]
        # Fallback: first tensor value
        for v in state.values():
            if isinstance(v, Tensor):
                return v
        return None

    def decide(self, statistic: float, statistic_kind: StatisticKind) -> GuardDecision:
        """Classify a statistic against the calibrated threshold."""
        return GuardDecision(
            statistic=statistic,
            threshold=self.threshold,
            kill=statistic > self.threshold,
            statistic_kind=statistic_kind,
        )

    def check(
        self,
        state: StepState,
        transition_fn: TransitionFn,
        step: int = 0,
    ) -> StabilityVerdict:
        """Run the guard on a step state.

        Args:
            state: Current step state (dict with tensors).
            transition_fn: Function that advances state by one step.
            step: Current step number (for reporting).

        Returns:
            StabilityVerdict with kill decision and details.
        """
        decisions = []
        max_stat = 0.0

        # Primary statistic
        stat = self.probe(transition_fn, state)
        decisions.append(self.decide(stat, self.statistic))
        max_stat = max(max_stat, stat)

        # Always also compute the other statistic for visibility
        other_kind = "fast_proxy" if self.statistic == "windowed_growth" else "windowed_growth"
        other_guard = StabilityGuard(
            threshold=self.threshold,
            estimator=self.estimator,
            statistic=other_kind,
            window=self.window,
        )
        other_stat = other_guard.probe(transition_fn, state)
        decisions.append(other_guard.decide(other_stat, other_kind))
        max_stat = max(max_stat, other_stat)

        return StabilityVerdict(
            kill=any(d.kill for d in decisions),
            decisions=tuple(decisions),
            max_statistic=max_stat,
            threshold=self.threshold,
            step=step,
        )

    def __call__(
        self,
        state: StepState,
        transition_fn: TransitionFn,
        step: int = 0,
    ) -> StabilityVerdict:
        return self.check(state, transition_fn, step)


@dataclass(frozen=True, slots=True)
class GuardHandle:
    """Handle for an attached stability guard.

    Use ``check()`` at each training step. When done, call ``detach()``.
    """

    guard: StabilityGuard
    model: torch.nn.Module
    transition_fn: TransitionFn

    def check(self, state: StepState, step: int = 0) -> StabilityVerdict:
        """Check stability at the current step."""
        return self.guard.check(state, self.transition_fn, step)

    def detach(self) -> None:
        """Detach the guard (no-op, for API symmetry)."""


def attach(
    model: torch.nn.Module,
    threshold: float = DEFAULT_TAU,
    statistic: StatisticKind = "windowed_growth",
    window: int = 10,
    transition_fn: TransitionFn | None = None,
) -> GuardHandle:
    """Attach a stability guard to a PyTorch model.

    Args:
        model: PyTorch module to monitor.
        threshold: Kill threshold (default τ=1.029, calibrated on settling dynamics).
        statistic: "windowed_growth" (recommended) or "fast_proxy".
        window: Window size for windowed_growth statistic.
        transition_fn: Optional custom transition function. If not provided,
            uses a default that runs ``model(x)`` and returns the output.

    Returns:
        GuardHandle with ``check(state, step)`` method.

    Example:
        model = torch.nn.Linear(10, 10)
        guard = attach(model)

        for step in range(100):
            x = torch.randn(32, 10)
            verdict = guard.check({"x": x})
            if verdict.kill:
                break
    """
    if transition_fn is None:

        def default_transition(state: StepState) -> StepState:
            x = state.get("x")
            if x is None:
                return state
            with torch.no_grad():
                y = model(x)
            return {**state, "y": y, "x": y}  # Recurrent: output becomes next input

        transition_fn = default_transition

    guard = StabilityGuard(
        threshold=threshold,
        statistic=statistic,
        window=window,
    )
    return GuardHandle(guard=guard, model=model, transition_fn=transition_fn)
