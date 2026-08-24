from __future__ import annotations

import multiprocessing
from typing import TYPE_CHECKING

from computronium.hyperopt.experiment import run_single_trial_task

if TYPE_CHECKING:
    from computronium.execution.task import ExperimentTask

__all__ = [
    "ParallelTrialRunner",
]


class ParallelTrialRunner:
    """
    Executes a batch of experiments in parallel using multiprocessing.
    """

    def __init__(self, num_workers: int, db_path: str):
        self.num_workers = num_workers
        self.db_path = db_path

    def run_batch(
        self, tasks: list[ExperimentTask], configs: list[dict[str, object]]
    ) -> list[dict[str, float] | None]:
        """
        Run a batch of tasks.

        Args:
            tasks: List of ExperimentTask objects.
            configs: List of resolved configuration dictionaries corresponding to
                tasks. (Must be resolved in main process to avoid DB write
                contention on 'ask')

        Returns:
            List of result metrics (or None for failures).
        """
        if not tasks:
            return []

        # Prepare arguments for workers
        worker_args = []
        for task, config in zip(tasks, configs):
            args = {
                "task_obj": task,  # Pass for metadata
                "config": config,
                "db_path": self.db_path,
            }
            worker_args.append(args)

        with multiprocessing.Pool(processes=self.num_workers) as pool:
            results = pool.map(self._wrapped_worker, worker_args)

        return results

    @staticmethod
    def _wrapped_worker(args):
        """
        Static wrapper to unpack args and call the logic.
        Redefined here to ensure visibility or call the global one.
        """
        task = args["task_obj"]
        config = args["config"]
        db_path = args["db_path"]

        return run_single_trial_task(
            task=task.task_name,
            model_name=task.model_name,
            config=config,
            storage_path=db_path,
            quick_mode=(task.tier.name == "SMOKE"),
            verbose=False,
        )
