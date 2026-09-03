"""Settling Time Measurement: Dynamical latency metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    from computronium.state import CompositeState, SystemContext


def measure_settling_time(
    transition_fn: Callable[[CompositeState, SystemContext], CompositeState],
    z: CompositeState,
    context: SystemContext,
    tolerance: float = 1e-4,
    max_steps: int = 1000,
    activity_key: str = "x",
    norm_type: str = "relative",
) -> tuple[int, list[float]]:
    """Measure settling time of the joint transition.

    Runs the transition until the activity change falls below tolerance
    or max_steps is reached.

    Args:
        transition_fn: Joint transition function.
        z: Initial joint state.
        context: System context.
        tolerance: Convergence tolerance.
        max_steps: Maximum steps to simulate.
        activity_key: Activity key to monitor.
        norm_type: "relative" (||Δx||/||x||) or "absolute" (||Δx||).

    Returns:
        Tuple of (settling_steps, step_norms_history).
    """
    step_norms = []
    z_current = z

    for step in range(max_steps):
        x_before = z_current.activity[activity_key]

        with torch.no_grad():
            z_next = transition_fn(z_current, context)

        x_after = z_next.activity[activity_key]
        delta = x_after - x_before

        if norm_type == "relative":
            norm = delta.norm(dim=-1).mean() / (x_before.norm(dim=-1).mean() + 1e-8)
        else:
            norm = delta.norm(dim=-1).mean()

        step_norms.append(norm.item())

        if norm < tolerance:
            return step + 1, step_norms

        z_current = z_next

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
        transition_fn: Callable[[CompositeState, SystemContext], CompositeState],
        z: CompositeState,
        context: SystemContext,
    ) -> tuple[int, list[float], list[CompositeState] | None]:
        """Measure settling time, optionally recording trajectory."""
        step_norms = []
        trajectory = [] if self.record_trajectory else None
        z_current = z

        for step in range(self.max_steps):
            if trajectory is not None:
                trajectory.append(z_current.clone())

            x_before = z_current.activity[self.activity_key]

            with torch.no_grad():
                z_next = transition_fn(z_current, context)

            x_after = z_next.activity[self.activity_key]
            delta = x_after - x_before

            if self.norm_type == "relative":
                norm = delta.norm(dim=-1).mean() / (x_before.norm(dim=-1).mean() + 1e-8)
            else:
                norm = delta.norm(dim=-1).mean()

            step_norms.append(norm.item())

            if norm < self.tolerance:
                if trajectory is not None:
                    trajectory.append(z_next)
                return step + 1, step_norms, trajectory

            z_current = z_next

        if trajectory is not None:
            trajectory.append(z_current)
        return self.max_steps, step_norms, trajectory

    def fast_proxy(
        self,
        transition_fn: Callable[[CompositeState, SystemContext], CompositeState],
        z: CompositeState,
        context: SystemContext,
    ) -> int:
        """Fast proxy: estimate settling from first few steps.

        Uses exponential fit to first 5 steps to predict settling time.
        """
        z_current = z
        norms = []

        for _ in range(min(5, self.max_steps)):
            x_before = z_current.activity[self.activity_key]

            with torch.no_grad():
                z_next = transition_fn(z_current, context)

            x_after = z_next.activity[self.activity_key]
            delta = x_after - x_before

            if self.norm_type == "relative":
                norm = delta.norm(dim=-1).mean() / (x_before.norm(dim=-1).mean() + 1e-8)
            else:
                norm = delta.norm(dim=-1).mean()

            norms.append(norm.item())
            z_current = z_next

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
            # t = (log(a) - log(tolerance)) / b  # ruff: ignore[commented-out-code]
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


def measure_settling_time_full_state(  # ruff: ignore[complex-structure]
    transition_fn: Callable[[CompositeState, SystemContext], CompositeState],
    z: CompositeState,
    context: SystemContext,
    tolerance: float = 1e-4,
    max_steps: int = 1000,
) -> tuple[int, dict[str, list[float]]]:
    """Measure settling time across all state components (activity, plastic, substrate).

    Args:
        transition_fn: Joint transition function.
        z: Initial joint state.
        context: System context.
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
    z_current = z
    z_next = z_current  # Initialize

    for step in range(max_steps):
        max_norm = 0.0

        with torch.no_grad():
            z_next = transition_fn(z_current, context)

        # Activity
        if z_current.activity:
            for key, x_before in z_current.activity.items():
                x_after = z_next.activity.get(key)
                if x_after is not None:
                    delta = x_after - x_before
                    norm = delta.norm(dim=-1).mean() / (
                        x_before.norm(dim=-1).mean() + 1e-8
                    )
                    step_norms["activity"].append(norm.item())
                    max_norm = max(max_norm, norm.item())
                break  # Just check first activity key

        # Plastic
        if z_current.plastic:
            for key, psi_before in z_current.plastic.items():
                psi_after = z_next.plastic.get(key)
                if psi_after is not None:
                    delta = psi_after - psi_before
                    norm = delta.norm().item() / (psi_before.norm().item() + 1e-8)
                    step_norms["plastic"].append(norm)
                    max_norm = max(max_norm, norm)
                break

        # Substrate
        if z_current.substrate:
            for key, sigma_before in z_current.substrate.items():
                sigma_after = z_next.substrate.get(key)
                if sigma_after is not None:
                    delta = sigma_after - sigma_before
                    norm = delta.norm().item() / (sigma_before.norm().item() + 1e-8)
                    step_norms["substrate"].append(norm)
                    max_norm = max(max_norm, norm)
                break

        if max_norm < tolerance:
            return step + 1, step_norms

        z_current = z_next

    return max_steps, step_norms
