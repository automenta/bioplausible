"""
Execution strategy lifecycle: planning next experiments and batching.
"""

import random
from dataclasses import dataclass

from computronium.execution.candidate_gen import CandidateGenerator, ExecutionStrategyConfig
from computronium.execution.task import ExperimentTask


@dataclass
class ExecutionStrategy:
    """
    The Brains. Decides what to run next.
    """

    generator: CandidateGenerator

    @classmethod
    def from_config(cls, config: ExecutionStrategyConfig) -> "ExecutionStrategy":
        return cls(generator=CandidateGenerator(config))

    def plan_next(self) -> ExperimentTask | None:
        """
        Scans all possibilities and returns the highest priority experiment.
        """
        candidates = self.generator.generate_candidates()

        if not candidates:
            return None

        # Standard Tier Calibration
        progress = self.generator.state.get_progress()
        total_standard_trials = 0
        for model in progress.values():
            for task in model.values():
                if "standard" in task:
                    total_standard_trials += task["standard"].get("count", 0)

        if total_standard_trials < 50:
            boost_applied = False
            for c in candidates:
                if c.tier.value == "standard":
                    c.priority += 500.0  # Massive boost
                    boost_applied = True

            if boost_applied:
                from computronium.core.logging import get_logger
                logger = get_logger("AutoScientist")
                logger.info(
                    "Calibration Mode Active: Boosted Standard Tier"
                    " candidates (Count: %d/50)",
                    total_standard_trials,
                )

        candidates.sort(key=lambda x: x.priority + random.uniform(0, 5), reverse=True)
        return candidates[0]

    def plan_batch(self, batch_size: int) -> list[ExperimentTask]:
        """
        Generate a batch of unique, high-priority experiments.
        """
        candidates = self.generator.generate_candidates()
        if not candidates:
            return []

        # Add noise to priority for diversity
        for c in candidates:
            c.priority += random.uniform(0, 5)

        candidates.sort(key=lambda x: x.priority, reverse=True)

        batch = []
        seen = set()
        for c in candidates:
            key = f"{c.model_name}_{c.task_name}_{c.tier.value}"
            if key not in seen:
                batch.append(c)
                seen.add(key)
            if len(batch) >= batch_size:
                break

        return batch


__all__ = [
    "ExecutionStrategy",
]