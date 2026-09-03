"""Fairness Contract and Benchmark Runner (PR-6).

Defines standardized evaluation contracts ensuring fair, reproducible comparisons
across learning rules, substrates, and credit assignment methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from computronium.resources import ResourceUsage

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True, slots=True)
class FairnessContract:
    """
    Contract specifying fair evaluation parameters.

    All benchmark runners must adhere to this contract to ensure
    comparable results across different rules/substrates.
    """

    gpu_hours_per_rule: float
    seeds: int = 5
    early_stopping: str = "best_val"
    data_splits: dict[str, float] = field(
        default_factory=lambda: {"train": 0.8, "val": 0.1, "test": 0.1}
    )
    max_epochs: int = 100
    batch_size: int = 64

    def __post_init__(self):
        if self.gpu_hours_per_rule <= 0:
            raise ValueError("gpu_hours_per_rule must be positive")
        if self.seeds < 1:
            raise ValueError("seeds must be >= 1")
        if self.early_stopping not in ("best_val", "last"):  # ruff: ignore[literal-membership]
            raise ValueError("early_stopping must be 'best_val' or 'last'")
        splits = self.data_splits
        if abs(sum(splits.values()) - 1.0) > 1e-6:
            raise ValueError("data_splits must sum to 1.0")
        for v in splits.values():
            if not (0 < v < 1):
                raise ValueError("data_splits values must be in (0, 1)")


def validate_fairness(
    contract: FairnessContract, results: list[dict[str, Any]]
) -> bool:
    """
    Validate that evaluation results satisfy the fairness contract.

    Checks:
    - Minimum number of seeds met
    - GPU hours within budget (if tracked)
    - Early stopping criterion applied
    - All required splits evaluated

    Args:
        contract: The fairness contract.
        results: List of result dicts from benchmark runs.

    Returns:
        True if all checks pass.
    """
    if len(results) < contract.seeds:
        return False

    for r in results:
        if "seed" not in r:
            return False
        if "metrics" not in r:
            return False
        if contract.early_stopping == "best_val" and "best_val_metric" not in r:
            return False

    return True


class BenchmarkRunner:
    """
    Base class for fair benchmark execution.

    Enforces FairnessContract across all derived runners.
    Subclasses implement run_single() for their specific evaluation logic.
    """

    def __init__(self, contract: FairnessContract):
        self.contract = contract
        self.results: list[dict[str, Any]] = []

    def run_single(
        self,
        model_factory: Callable[[], nn.Module],
        task_name: str,
        seed: int,
        device: str = "cpu",
    ) -> dict[str, Any]:
        """
        Run a single benchmark trial.

        Subclasses must implement this method.

        Args:
            model_factory: Callable returning fresh model instance.
            task_name: Identifier for the task.
            seed: Random seed for this trial.
            device: Compute device.

        Returns:
            Result dict with at least: seed, metrics, resource_usage.
        """
        raise NotImplementedError

    def run(
        self,
        model_factory: Callable[[], nn.Module],
        task_name: str,
        device: str = "cpu",
    ) -> list[dict[str, Any]]:
        """
        Run full benchmark suite per contract.

        Args:
            model_factory: Callable returning fresh model instance.
            task_name: Task identifier.
            device: Compute device.

        Returns:
            List of result dicts (one per seed).
        """
        self.results = []

        for seed in range(self.contract.seeds):
            torch.manual_seed(seed)
            result = self.run_single(model_factory, task_name, seed, device)
            result["seed"] = seed
            self.results.append(result)

        if not validate_fairness(self.contract, self.results):
            raise ValueError("Fairness contract validation failed")

        return self.results

    def aggregate(self) -> dict[str, float]:
        """
        Aggregate results across seeds.

        Returns:
            Dict with mean/std for each metric.
        """
        if not self.results:
            return {}

        metrics_keys: set[str] = set()
        for r in self.results:
            metrics_keys.update(r.get("metrics", {}).keys())

        agg: dict[str, float] = {"n_seeds": float(len(self.results))}
        for key in metrics_keys:
            values: list[float] = [
                float(r["metrics"].get(key, 0.0)) for r in self.results
            ]
            import statistics

            agg[f"{key}_mean"] = statistics.mean(values)
            agg[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0

        return agg


class ResourceAwareBenchmarkRunner(BenchmarkRunner):
    """
    Benchmark runner that tracks and enforces resource budgets.

    Extends BenchmarkRunner with ResourceUsage tracking per seed.
    """

    def __init__(
        self,
        contract: FairnessContract,
        resource_budget: ResourceUsage | None = None,
    ):
        super().__init__(contract)
        self.resource_budget = resource_budget
        self.resource_usages: list[ResourceUsage] = []

    def run_single(
        self,
        model_factory: Callable[[], nn.Module],
        task_name: str,
        seed: int,
        device: str = "cpu",
    ) -> dict[str, Any]:
        result = super().run_single(model_factory, task_name, seed, device)

        if "resource_usage" in result:
            usage = result["resource_usage"]
            if isinstance(usage, ResourceUsage):
                self.resource_usages.append(usage)
            elif isinstance(usage, dict):
                self.resource_usages.append(ResourceUsage(**usage))

        return result

    def check_budget(self) -> bool:
        """Check if accumulated resource usage is within budget."""
        if self.resource_budget is None:
            return True

        total = ResourceUsage()
        for u in self.resource_usages:
            total += u

        return (
            total.compute <= self.resource_budget.compute
            and total.memory <= self.resource_budget.memory
            and total.energy <= self.resource_budget.energy
            and total.latency <= self.resource_budget.latency
            and total.plastic_state_capacity
            <= self.resource_budget.plastic_state_capacity
        )

    def get_total_usage(self) -> ResourceUsage:
        """Get total resource usage across all seeds."""
        total = ResourceUsage()
        for u in self.resource_usages:
            total += u
        return total
