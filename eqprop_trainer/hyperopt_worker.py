"""
Hyperopt worker module for hyperparameter optimization.

This module contains the background worker that runs hyperparameter optimization trials.
"""

import sys
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np

from PyQt6.QtCore import QThread, pyqtSignal

from hyperopt.engine import EvolutionaryOptimizer, OptimizationConfig
from hyperopt.experiment import TrialRunner
from hyperopt.storage import HyperoptStorage
from hyperopt.metrics import get_pareto_frontier, rank_trials
from gui.algorithms import MODEL_REGISTRY
from hyperopt.metrics import get_pareto_frontier, rank_trials
from gui.algorithms import MODEL_REGISTRY
from eqprop_trainer.config import GLOBAL_CONFIG


class HyperoptSearchWorker(QThread):
    """Background worker that runs hyperparameter optimization trials."""

    trial_completed = pyqtSignal(int, bool)  # trial_id, success
    status_update = pyqtSignal(str)
    baseline_completed = pyqtSignal(str, dict)  # model_name, results
    experiment_progress = pyqtSignal(int, dict)  # trial_id, metrics
    progress_update = pyqtSignal(int, str, str)  # percentage, phase, message
    insight_update = pyqtSignal(str)  # insight message

    def __init__(self, optimizer, runner, scheduler):
        super().__init__()
        self.optimizer = optimizer
        self.runner = runner
        self.scheduler = scheduler
        self.running = True
        self.paused = False # Auto-start by default
        self.total_models = len(optimizer.model_names) if optimizer.model_names else 1
        self.completed_baselines = 0

    def _get_transformer_config(self, model_name: str) -> Dict[str, Any]:
        """Generate default configuration for transformer models."""
        return {
            'lr': 3e-4,
            'hidden_dim': 128,
            'num_layers': 2,
            'steps': 12,
            'epochs': GLOBAL_CONFIG.epochs,
            'model_name': model_name
        }

    def _get_deep_hebbian_config(self, model_name: str) -> Dict[str, Any]:
        """Generate default configuration for deep hebbian models."""
        return {
            'lr': 5e-4,
            'hidden_dim': 128,
            'num_layers': 500, # Explicitly configured for depth
            'epochs': GLOBAL_CONFIG.epochs,
            'model_name': model_name
        }

    def _get_eqprop_config(self, model_name: str) -> Dict[str, Any]:
        """Generate default configuration for EqProp models."""
        config = {
            'lr': 1e-3,
            'hidden_dim': 64,
            'steps': 12,
            'epochs': GLOBAL_CONFIG.epochs,
            'model_name': model_name
        }
        if 'MLP' in model_name:
            config['beta'] = 0.22 # Required for EqProp MLP
            config['num_layers'] = 10 # Required for EqProp MLP
        return config

    def _get_chl_config(self, model_name: str) -> Dict[str, Any]:
        """Generate default configuration for CHL models."""
        return {
            'lr': 1e-3,
            'hidden_dim': 128,
            'num_layers': 10,
            'model_name': model_name,
            'beta': 0.1, # Required for CHL
            'steps': 20,
            'epochs': GLOBAL_CONFIG.epochs
        }

    def _get_general_config(self, model_name: str) -> Dict[str, Any]:
        """Generate general default configuration."""
        return {
            'lr': 3e-4,
            'hidden_dim': 128,
            'num_layers': 2,
            'steps': 12,
            'epochs': GLOBAL_CONFIG.epochs,
            'model_name': model_name
        }

    def get_default_config_for_model(self, model_name):
        """Generate a default configuration for a given model."""
        # Default configuration based on model type
        if 'Transformer' in model_name:
            return self._get_transformer_config(model_name)
        elif 'Deep Hebbian' in model_name:
            return self._get_deep_hebbian_config(model_name)
        elif 'MLP' in model_name or 'EqProp' in model_name:
            return self._get_eqprop_config(model_name)
        elif 'CHL' in model_name:
            return self._get_chl_config(model_name)
        else:
            # General default config
            return self._get_general_config(model_name)

    def run(self):
        try:
            # First run a single baseline
            self.status_update.emit("Running single baseline experiment...")
            self.progress_update.emit(0, "Baselines", "Starting single baseline experiment...")
            self.insight_update.emit("Starting single baseline experiment with conventional state-of-the-art solution.")

            # Baseline execution handled via queue_baseline from Dashboard
            pass

            self.status_update.emit("Baseline experiments completed. Starting optimization...")
            self.progress_update.emit(30, "Optimization", "Starting hyperparameter optimization...")
            self.insight_update.emit("Baselines complete. Starting hyperparameter optimization with shallow-first search strategy.")

            # The optimizer should have already initialized its population during construction
            # Now we need to add initial experiments to the scheduler based on the optimizer's population
            for model_name in self.optimizer.model_names:
                # Add initial experiments for each model from the optimizer's population
                try:
                    # Try to get initial configurations from the optimizer
                    initial_configs = self.optimizer.get_initial_configs(model_name)
                    if initial_configs:
                        for config in initial_configs:
                            self.scheduler.add_experiment(model_name, config, priority=0.7)  # Higher priority for initial configs
                    else:
                        # Fallback: create default configurations
                        for i in range(2):  # Add 2 initial experiments per model
                            config = self.get_default_config_for_model(model_name)
                            self.scheduler.add_experiment(model_name, config, priority=0.5)
                except AttributeError:
                    # If get_initial_configs doesn't exist, create default configs
                    for i in range(2):  # Add 2 initial experiments per model
                        config = self.get_default_config_for_model(model_name)
                        self.scheduler.add_experiment(model_name, config, priority=0.5)

            # Now run optimization
            optimization_start_time = time.time()
            total_optimization_time = 0
            trial_count = 0

            while self.running:
                if self.paused:
                    time.sleep(0.1)
                    continue

                # Get next trial from scheduler
                experiment = self.scheduler.get_next_experiment()

                if experiment is None:
                    # No queued experiments, evolve population
                    self.status_update.emit("Evolving to next generation...")
                    self.progress_update.emit(70, "Evolution", "Evolving population to next generation...")
                    self.insight_update.emit("Evolving population. Using Pareto-based selection to guide search.")

                    for model_name in self.optimizer.model_names:
                        try:
                            self.optimizer.evolve_generation(model_name)
                        except Exception as e:
                            print(f"Error evolving generation for {model_name}: {e}")

                    self.scheduler.advance_generation()
                    time.sleep(1)
                    continue

                # Create trial in storage
                trial_id = self.runner.storage.create_trial(
                    experiment['model_name'],
                    experiment['config']
                )
                experiment['trial_id'] = trial_id

                # Run trial with allocated epochs
                trial = self.runner.storage.get_trial(trial_id)
                if trial:
                    trial_count += 1
                    self.progress_update.emit(30 + int((trial_count / max(1, self.total_models * 5)) * 60),
                                            "Trials",
                                            f"Running trial {trial_id} ({trial_count}) - {trial.model_name}")

                    # Temporarily override runner epochs for this specific trial
                    # NO! AGGRESSIVE REFACTOR: Enforce GLOBAL_CONFIG.epochs always.
                    # The TrialRunner deals with this.
                    # self.runner.epochs = experiment['allocated_epochs'] 
                    
                    # Use quick mode for efficiency during search
                    original_quick_mode = self.runner.quick_mode
                    self.runner.quick_mode = GLOBAL_CONFIG.quick_mode # Explicitly set from global

                    success = False
                    start_time = time.time()
                    try:
                        # Define pruning callback
                        def pruning_callback(trial_id, epoch, metrics):
                            # Update scheduler with intermediate progress
                            # Note: scheduler.update_experiment_progress expects full Trial object usually,
                            # but here we are just checking for pruning conditions.
                            # We can manually trigger the check logic.
                            
                            # Update experiment state in scheduler
                            self.scheduler.update_experiment_progress(
                                trial_id, 
                                metrics, 
                                actual_epochs=epoch
                            )
                            
                            # Check if scheduler allocated_epochs has been reduced below current epoch
                            # Only prune if we are stopping EARLY (before the natural end of the trial)
                            if experiment.get('allocated_epochs', 0) <= epoch and epoch < self.runner.epochs:
                                return True # Prune!
                            return False

                        success = self.runner.run_trial(trial_id, pruning_callback=pruning_callback)
                    except Exception as e:
                        print(f"Error running trial {trial_id}: {e}")
                        success = False

                    trial_duration = time.time() - start_time
                    total_optimization_time += trial_duration

                    # Restore original settings
                    self.runner.quick_mode = original_quick_mode

                    # Update scheduler with results
                    if success:
                        updated_trial = self.runner.storage.get_trial(trial_id)
                        if updated_trial:
                            metrics = {
                                'accuracy': updated_trial.accuracy or 0.0,
                                'perplexity': updated_trial.perplexity or 100.0,
                                'iteration_time': updated_trial.iteration_time or 1.0,
                                'param_count': updated_trial.param_count or 1.0
                            }

                            self.scheduler.update_experiment_progress(
                                trial_id,
                                metrics,
                                actual_epochs=updated_trial.epochs_completed
                            )

                            # Update insight with trial results
                            self.insight_update.emit(f"Trial {trial_id} completed: {updated_trial.accuracy:.3f} accuracy, {updated_trial.perplexity:.2f} perplexity. Promise: {experiment.get('promise_score', 0):.2f}")

                    self.trial_completed.emit(trial_id, success)

                    # Update Pareto frontiers
                    try:
                        self.optimizer.update_pareto_frontiers()
                    except Exception as e:
                        print(f"Error updating Pareto frontiers: {e}")
                else:
                    print(f"Failed to create trial for experiment: {experiment}")
                    time.sleep(1)  # Brief pause before continuing
        except Exception as e:
            print(f"Critical error in worker thread: {e}")
            import traceback
            traceback.print_exc()
            self.insight_update.emit(f"Critical error occurred: {str(e)}. Check logs for details.")

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def stop(self):
        self.running = False
        self.wait()

    def queue_baseline(self):
        """Queue the baseline experiment."""
        # Use valid args from config
        self.status_update.emit("Queueing baseline experiment...")
        # Create a single baseline config
        baseline_config = {
             'lr': 3e-4,
             'hidden_dim': 256,
             'num_layers': 4,
             'steps': 15,
             'epochs': GLOBAL_CONFIG.epochs,
             'model_name': 'Backprop (Transformer)'
        }
        
        # Add to scheduler with high priority
        self.scheduler.add_experiment(baseline_config['model_name'], baseline_config, priority=100.0)