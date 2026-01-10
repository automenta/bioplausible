"""
Experiment scheduler module for hyperparameter optimization.

This module manages experiment scheduling with intelligent epoch allocation.
"""

import time
from eqprop_trainer.config import GLOBAL_CONFIG
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np


class ExperimentScheduler:
    """Manages experiment scheduling with intelligent epoch allocation."""

    def __init__(self, baseline_results: Dict[str, Dict[str, float]], max_walltime_multiplier: float = 3.0):
        self.baseline_results = baseline_results
        self.max_walltime_multiplier = max_walltime_multiplier
        self.experiment_queue = []
        self.running_experiments = []
        self.completed_experiments = []
        self.generation = 0  # Track current generation for adaptive scheduling

    def add_experiment(self, model_name: str, config: Dict[str, Any], priority: float = 0.0):
        """Add an experiment to the queue with priority-based scheduling."""
        # Calculate initial walltime estimate based on baseline
        baseline_time = self.baseline_results.get(model_name, {}).get('walltime', 120.0)
        max_allowed_time = baseline_time * self.max_walltime_multiplier

        experiment = {
            'model_name': model_name,
            'config': config,
            'priority': priority,
            'baseline_time': baseline_time,
            'max_time': max_allowed_time,
            'status': 'queued',
            'start_time': None,
            'end_time': None,
            'allocated_epochs': 0,
            'actual_epochs': 0,
            'estimated_performance': 0.0,
            'performance_history': [],  # Track performance over time
            'promise_score': 0.0,  # How promising this experiment looks
            'generation': self.generation
        }

        self.experiment_queue.append(experiment)
        self.experiment_queue.sort(key=lambda x: x['priority'], reverse=True)

    def get_next_experiment(self) -> Optional[Dict[str, Any]]:
        """Get the next experiment to run based on scheduling policy."""
        if not self.experiment_queue:
            return None

        # For shallow-first search, start with fewer epochs and expand based on promise
        experiment = self.experiment_queue.pop(0)

        # Allocate epochs intelligently based on generation and promise
        if self.generation == 0:
            # First generation: shallow search
            baseline_epochs = GLOBAL_CONFIG.epochs
        elif self.generation == 1:
            # Second generation: slightly deeper but kept consistent for fairness
            baseline_epochs = GLOBAL_CONFIG.epochs
        else:
            # Later generations: adapt based on promise, but start at 3
            baseline_epochs = GLOBAL_CONFIG.epochs

        experiment['allocated_epochs'] = baseline_epochs

        # Adjust based on promise score if available
        if experiment['promise_score'] > 0.8:
            experiment['allocated_epochs'] = min(10, baseline_epochs * 2)
        elif experiment['promise_score'] > 0.5:
            experiment['allocated_epochs'] = min(8, baseline_epochs * 1.5)

        experiment['status'] = 'running'
        experiment['start_time'] = time.time()
        self.running_experiments.append(experiment)

        return experiment

    def _find_experiment_by_id(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        """Find an experiment by its ID."""
        for exp in self.running_experiments:
            if exp.get('trial_id') == experiment_id:
                return exp
        return None

    def _calculate_performance_score(self, metrics: Dict[str, float]) -> float:
        """Calculate composite performance score."""
        acc = metrics.get('accuracy', 0.0)
        ppl = metrics.get('perplexity', 10.0)
        iter_time = metrics.get('iteration_time', 1.0)
        params = metrics.get('param_count', 1.0)

        # Composite performance score (higher is better)
        # Accuracy contributes positively, others contribute negatively
        performance_score = acc - (ppl / 100.0) - (iter_time / 10.0) - (params / 100.0)
        return performance_score

    def _should_prune_by_time_budget(self, experiment: Dict[str, Any], actual_epochs: int, elapsed_time: float) -> bool:
        """Check if experiment should be pruned based on time budget."""
        # HARD PRUNING: Derived limit from total budget
        # e.g. 60s / 3 epochs = 20s/epoch
        # e.g. 60s / 5 epochs = 12s/epoch
        per_epoch_limit = GLOBAL_CONFIG.max_trial_time / max(GLOBAL_CONFIG.epochs, 1)

        current_rate = elapsed_time / actual_epochs
        if current_rate > per_epoch_limit:
            print(f"PRUNED: Experiment {experiment['trial_id']} exceeded {per_epoch_limit:.1f}s/epoch limit (Budget: {GLOBAL_CONFIG.max_trial_time}s / {GLOBAL_CONFIG.epochs} epochs)")
            return True

        projected_total_time = current_rate * experiment['allocated_epochs']
        baseline_expected_time = experiment['baseline_time'] * self.max_walltime_multiplier

        if projected_total_time > baseline_expected_time * 3:  # 3x safety margin beyond max multiplier
            print(f"Pruned experiment {experiment['trial_id']} - Projected time {projected_total_time:.1f}s exceeds 3x baseline threshold {baseline_expected_time * 3:.1f}s")
            return True

        # Additional check: if current elapsed time already exceeds max allowed time
        if elapsed_time > experiment['max_time']:
            print(f"Pruned experiment {experiment['trial_id']} - Elapsed time {elapsed_time:.1f}s exceeds max allowed time {experiment['max_time']:.1f}s")
            return True

        # EARLY PRUNING: Check if current iteration is taking way too long compared to baseline
        # If we're at epoch 1 and already taking 50x longer than baseline per epoch, prune immediately
        if actual_epochs == 1 and elapsed_time > experiment['baseline_time'] * 20:
            print(f"EARLY PRUNING: Experiment {experiment['trial_id']} took {elapsed_time:.1f}s for epoch 1, which is >20x baseline {experiment['baseline_time']:.1f}s")
            return True

        return False

    def _update_trial_status(self, experiment_id: int, status: str):
        """Update trial status in storage."""
        try:
            if hasattr(self, 'storage') and self.storage:
                trial = self.storage.get_trial(experiment_id)
                if trial:
                    self.storage.update_trial(experiment_id, status=status)
        except:
            pass  # Ignore errors when updating trial status

    def _adjust_allocated_epochs(self, experiment: Dict[str, Any], promise_score: float, actual_epochs: int):
        """Adjust allocated epochs based on promise score and time budget."""
        if experiment['start_time'] is None:
            return

        elapsed_time = time.time() - experiment['start_time']
        time_budget_remaining = experiment['max_time'] - elapsed_time

        # Only extend if we have sufficient time budget and the experiment is promising
        if (promise_score > 0.6 and
            actual_epochs < experiment['allocated_epochs'] and
            time_budget_remaining > 30):  # At least 30 seconds remaining

            # Extend epochs conservatively
            extension = min(3, max(1, experiment['allocated_epochs'] // 2))
            experiment['allocated_epochs'] = min(
                experiment['allocated_epochs'] + extension,
                int(experiment['max_time'] / max(elapsed_time / max(actual_epochs, 1), 0.001) * 0.8)  # Don't exceed 80% of projected time
            )

        # Check if we should terminate early due to poor performance
        if (promise_score < 0.1 and actual_epochs >= 3 and
            elapsed_time > experiment['baseline_time'] * 0.3):  # Used 30% of baseline time
            experiment['allocated_epochs'] = actual_epochs  # Stop early

    def update_experiment_progress(self, experiment_id: int, metrics: Dict[str, float], actual_epochs: int = 0):
        """Update experiment with latest metrics and adjust scheduling."""
        experiment = self._find_experiment_by_id(experiment_id)
        if not experiment:
            return

        try:
            # Update actual epochs
            experiment['actual_epochs'] = actual_epochs

            # Calculate performance indicator
            performance_score = self._calculate_performance_score(metrics)
            experiment['estimated_performance'] = performance_score

            # Track performance history
            experiment['performance_history'].append({
                'epoch': actual_epochs,
                'performance': performance_score,
                'timestamp': time.time()
            })

            # Calculate promise score based on current performance and trend
            promise_score = self.calculate_promise_score(experiment)
            experiment['promise_score'] = promise_score

            # Check if experiment is taking significantly longer than baseline (pruning condition)
            if experiment['start_time'] is not None:
                elapsed_time = time.time() - experiment['start_time']

                # Prune if taking more than 5x longer than baseline for the epochs completed
                if actual_epochs > 0 and self._should_prune_by_time_budget(experiment, actual_epochs, elapsed_time):
                    experiment['allocated_epochs'] = actual_epochs  # Stop immediately
                    self._update_trial_status(experiment_id, 'pruned')
                    return

            # If experiment is showing promise, consider extending it
            self._adjust_allocated_epochs(experiment, promise_score, actual_epochs)
        except Exception as e:
            print(f"Error updating experiment progress: {e}")
            # Continue with original values if calculation fails

    def calculate_promise_score(self, experiment: Dict[str, Any]) -> float:
        """Calculate how promising an experiment is based on performance and trends."""
        if not experiment.get('performance_history', []):
            return 0.0

        try:
            # Get the most recent performance
            recent_perf = experiment['performance_history'][-1]['performance']

            # Simple promise calculation based on current performance
            promise = max(0.0, min(1.0, recent_perf + 0.5))  # Shift range to [0, 1]

            # Boost if we're seeing improvement trend
            if len(experiment['performance_history']) >= 2:
                prev_perf = experiment['performance_history'][-2]['performance']
                if recent_perf > prev_perf:
                    promise = min(1.0, promise * 1.2)  # Boost for improving trend, cap at 1.0

            # Penalize if taking too long relative to baseline
            if experiment.get('start_time'):
                elapsed = time.time() - experiment['start_time']
                baseline_ratio = elapsed / max(experiment['baseline_time'], 1)
                if baseline_ratio > 2.0:
                    promise *= 0.8  # Penalize for taking too long
                elif baseline_ratio > 3.0:
                    promise *= 0.5  # Heavy penalty for excessive time

            return min(1.0, max(0.0, promise))
        except Exception:
            # Return a conservative promise score if calculation fails
            return 0.3

    def advance_generation(self):
        """Advance to the next generation, affecting scheduling strategy."""
        self.generation += 1
        # Re-sort experiments based on updated generation info
        self.experiment_queue.sort(key=lambda x: x['priority'], reverse=True)