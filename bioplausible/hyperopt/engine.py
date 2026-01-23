"""
Evolutionary Optimization Engine (Optuna-based)

Thin wrapper around Optuna for backward compatibility.
Replaces ~260 lines of custom NSGA-II with Optuna's implementation.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import optuna

    from .optuna_bridge import create_optuna_space, create_study, get_pareto_trials

    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False



@dataclass
class OptimizationConfig:
    """Configuration for the optimization process."""

    population_size: int = 20
    n_generations: int = 10
    mutation_rate: float = 0.3  # Unused with Optuna
    crossover_rate: float = 0.7  # Unused with Optuna
    elite_fraction: float = 0.2  # Unused with Optuna
    random_seed: int = 42
    use_p2p: bool = False
    task: str = "shakespeare"


class EvolutionaryOptimizer:
    """
    Multi-objective evolutionary optimizer using Optuna.

    This is a backward-compatible wrapper around Optuna's built-in algorithms.
    The custom NSGA-II implementation has been replaced with optuna.samplers.NSGAIISampler.
    """

    def __init__(
        self,
        model_names: List[str],
        config: OptimizationConfig = None,
        storage: Any = None,  # Optuna handles storage
        p2p_controller: Any = None,
    ):
        if not HAS_OPTUNA:
            raise ImportError(
                "Optuna is required for EvolutionaryOptimizer. "
                "Install with: pip install optuna\n"
                "Or use legacy evolution code from bioplausible.evolution (deprecated)."
            )

        self.model_names = model_names
        self.config = config or OptimizationConfig()

        # Create Optuna studies (one per model)
        self.studies = {}
        for model_name in model_names:
            self.studies[model_name] = create_study(
                model_names=[model_name],
                n_objectives=2,  # accuracy, loss (or other metrics)
                storage=None if storage is None else storage.get_storage_url(),
                study_name=f"{model_name}_{self.config.task}",
                use_pruning=True,
                sampler_name="nsga2",  # Use NSGA-II for multi-objective
            )

        self.p2p_controller = p2p_controller

    def initialize_population(self, model_name: str):
        """
        Initialize random population for a model.

        With Optuna, this is handled automatically by the sampler.
        """
        # Optuna handles population initialization
        pass

    def select_parents(self, model_name: str):
        """
        Select parents using tournament selection based on Pareto ranking.

        With Optuna, parent selection is handled by the sampler.
        """
        # Optuna's NSGA-II sampler handles parent selection
        pass

    def generate_offspring(
        self, model_name: str, parents: List[Any], n_offspring: int
    ):
        """
        Generate offspring through crossover and mutation.

        With Optuna, crossover/mutation is handled by the sampler.
        """
        # Optuna's NSGA-II sampler handles offspring generation
        pass

    def evolve_generation(self, model_name: str):
        """
        Evolve to the next generation.

        With Optuna, call study.optimize() for one generation.
        """
        # Optuna handles evolution internally
        pass

    def get_next_trial(self, model_name: str = None):
        """
        Get the next pending trial to run.

        Args:
            model_name: Model name or None for any model

        Returns:
            Trial configuration or None
        """
        if model_name is None:
            model_name = self.model_names[0]

        study = self.studies.get(model_name)
        if study is None:
            return None

        # Ask Optuna for next trial suggestion
        trial = study.ask()
        config = create_optuna_space(trial, model_name)

        return {
            "trial": trial,
            "config": config,
            "model_name": model_name,
        }

    def update_pareto_frontiers(self):
        """
        Update Pareto frontier markings for all models.

        With Optuna, Pareto frontiers are computed on-the-fly.
        """
        # Optuna computes Pareto frontiers automatically
        pass

    def get_best_configs(self, model_name: str, top_k: int = 5):
        """
        Get best configurations for a model.

        Args:
            model_name: Model name
            top_k: Number of top configs to return

        Returns:
            List of best configurations
        """
        study = self.studies.get(model_name)
        if study is None:
            return []

        # Get Pareto frontier
        pareto_trials = get_pareto_trials(study)

        # Convert to config format
        configs = []
        for trial in pareto_trials[:top_k]:
            configs.append(
                {
                    "config": trial.params,
                    "accuracy": trial.values[0] if trial.values else 0.0,
                    "loss": trial.values[1] if len(trial.values) > 1 else float("inf"),
                }
            )

        return configs

    def get_statistics(self):
        """
        Get optimization statistics.

        Returns:
            Dictionary of statistics
        """
        stats = {}
        for model_name, study in self.studies.items():
            trials = study.get_trials()
            completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]

            stats[model_name] = {
                "total_trials": len(trials),
                "completed_trials": len(completed),
                "best_trials": len(study.best_trials) if completed else 0,
                "best_value": study.best_value if len(study.directions) == 1 and completed else None,
            }

        return stats
