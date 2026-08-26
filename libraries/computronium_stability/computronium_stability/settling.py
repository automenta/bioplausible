"""Settling Time Measurement: Dynamical latency metrics.

Framework-agnostic: works with any callable transition function that maps
state dicts to state dicts, where state dicts contain PyTorch tensors.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

# Type aliases for framework-agnostic state
StepState = dict[str, Tensor]
TransitionFn = Callable[[StepState], StepState]


def measure_settling_time(
    transition_fn: TransitionFn,
    state: StepState,
    tolerance: float = 1e-4,
    max_steps: int = 1000,
    activity_key: str = "x",
    norm_type: str = "relative",
) -> tuple[int, list[float]]:
    """Measure settling time of the transition.

    Runs the transition until the activity change falls below tolerance
    or max_steps is reached.

    Args:
        transition_fn: State transition function.
        state: Initial state.
        tolerance: Convergence tolerance.
        max_steps: Maximum steps to simulate.
        activity_key: Activity key to monitor.
        norm_type: "relative" (||Δx||/||x||) or "absolute" (||Δx||).

    Returns:
        Tuple of (settling_steps, step_norms_history).
    """
    step_norms = []
    state_current = state

    for step in range(max_steps):
        x_before = state_current.get(activity_key)
        if x_before is None or not isinstance(x_before, Tensor):
            return step, step_norms

        with torch.no_grad():
            state_next = transition_fn(state_current)

        x_after = state_next.get(activity_key)
        if x_after is None or not isinstance(x_after, Tensor):
            return step, step_norms

        delta = x_after - x_before

        if norm_type == "relative":
            norm = delta.norm(dim=-1).mean() / (x_before.norm(dim=-1).mean() + 1e-8)
        else:
            norm = delta.norm(dim=-1).mean()

        step_norms.append(norm.item())

        if norm < tolerance:
            return step + 1, step_norms

        state_current = state_next

    return max_steps, step_norms


@dataclass(slots=True)
class SettlingMonitor:
    """Configurable settling time monitor with trajectory recording."""

    tolerance: float = 1e-4
    max_steps: int = 1000
    activity_key: str = "x"
    norm_type: str = "relative"
    record_trajectory: bool = False

    def __call__(
        self,
        transition_fn: TransitionFn,
        state: StepState,
    ) -> tuple[int, list[float], list[StepState] | None]:
        """Measure settling time, optionally recording trajectory."""
        step_norms = []
        trajectory: list[StepState] | None = [] if self.record_trajectory else None
        state_current = state

        for step in range(self.max_steps):
            if trajectory is not None:
                trajectory.append({k: v.clone() if isinstance(v, Tensor) else v for k, v in state_current.items()})

            x_before = state_current.get(self.activity_key)
            if x_before is None or not isinstance(x_before, Tensor):
                if trajectory is not None:
                    trajectory.append({k: v.clone() if isinstance(v, Tensor) else v for k, v in state_current.items()})
                return step, step_norms, trajectory

            with torch.no_grad():
                state_next = transition_fn(state_current)

            x_after = state_next.get(self.activity_key)
            if x_after is None or not isinstance(x_after, Tensor):
                if trajectory is not None:
                    trajectory.append({k: v.clone() if isinstance(v, Tensor) else v for k, v in state_next.items()})
                return step, step_norms, trajectory

            delta = x_after - x_before

            if self.norm_type == "relative":
                norm = delta.norm(dim=-1).mean() / (x_before.norm(dim=-1).mean() + 1e-8)
            else:
                norm = delta.norm(dim=-1).mean()

            step_norms.append(norm.item())

            if norm < self.tolerance:
                if trajectory is not None:
                    trajectory.append({k: v.clone() if isinstance(v, Tensor) else v for k, v in state_next.items()})
                return step + 1, step_norms, trajectory

            state_current = state_next

        if trajectory is not None:
            trajectory.append({k: v.clone() if isinstance(v, Tensor) else v for k, v in state_current.items()})
        return self.max_steps, step_norms, trajectory

    def fast_proxy(
        self,
        transition_fn: TransitionFn,
        state: StepState,
    ) -> int:
        """Fast proxy: estimate settling from first few steps.

        Uses exponential fit to first 5 steps to predict settling time.
        """
        state_current = state
        norms = []

        for _ in range(min(5, self.max_steps)):
            x_before = state_current.get(self.activity_key)
            if x_before is None or not isinstance(x_before, Tensor):
                return self.max_steps

            with torch.no_grad():
                state_next = transition_fn(state_current)

            x_after = state_next.get(self.activity_key)
            if x_after is None or not isinstance(x_after, Tensor):
                return self.max_steps

            delta = x_after - x_before

            if self.norm_type == "relative":
                norm = delta.norm(dim=-1).mean() / (x_before.norm(dim=-1).mean() + 1e-8)
            else:
                norm = delta.norm(dim=-1).mean()

            norms.append(norm.item())
            state_current = state_next

        if len(norms) < 2:
            return self.max_steps

        # Fit exponential decay: norm ≈ a * exp(-b * t)
        # log(norm) ≈ log(a) - b * t
        log_norms = torch.log(torch.tensor(norms, dtype=torch.float32) + 1e-12)
        t = torch.arange(len(norms), dtype=torch.float32)

        # Linear regression for log(norm) = c - b*t
        A = torch.stack([torch.ones_like(t), -t], dim=1)
        result = torch.linalg.lstsq(A, log_norms)
        coeffs = result.solution
        b = coeffs[1].item()

        if b > 0:
            # Time to reach tolerance: tolerance = a * exp(-b * t)
            # t = (log(a) - log(tolerance)) / b
            a = torch.exp(coeffs[0]).item()
            if a > self.tolerance:
                estimated = int(
                    (
                        torch.log(torch.tensor(a))
                        - torch.log(torch.tensor(self.tolerance))
                    )
                    / b
                )
                return min(estimated, self.max_steps)

        return self.max_steps


def measure_settling_time_full_state(
    transition_fn: TransitionFn,
    state: StepState,
    tolerance: float = 1e-4,
    max_steps: int = 1000,
) -> tuple[int, dict[str, list[float]]]:
    """Measure settling time across all state components.

    Args:
        transition_fn: State transition function.
        state: Initial state.
        tolerance: Convergence tolerance.
        max_steps: Maximum steps.

    Returns:
        Tuple of (settling_steps, dict of step_norms per component).
    """
    step_norms: dict[str, list[float]] = {
        "activity": [],
        "plastic": [],
        "substrate": [],
    }
    state_current = state
    state_next = state_current

    for step in range(max_steps):
        max_norm = 0.0

        with torch.no_grad():
            state_next = transition_fn(state_current)

        # Check all tensor components
        for key, val in state_current.items():
            if isinstance(val, Tensor):
                next_val = state_next.get(key)
                if next_val is not None and isinstance(next_val, Tensor):
                    delta = next_val - val
                    norm = delta.norm().item() / (val.norm().item() + 1e-8)
                    # Categorize by key prefix
                    if key.startswith(("x", "activity", "hidden", "y")):
                        step_norms["activity"].append(norm)
                    elif key.startswith(("psi", "plastic", "controller", "logits")):
                        step_norms["plastic"].append(norm)
                    elif key.startswith(("sigma", "substrate", "memristive", "optical")):
                        step_norms["substrate"].append(norm)
                    else:
                        step_norms["activity"].append(norm)  # default
                    max_norm = max(max_norm, norm)

        if max_norm < tolerance:
            return step + 1, step_norms

        state_current = state_next

    return max_steps, step_norms