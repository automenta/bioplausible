"""
Optuna Bridge for Bioplausible

Maps ModelSpec and SearchSpace definitions to Optuna suggest_* calls.
Replaces custom evolution code with Optuna's proven algorithms.
"""

from typing import Any, Callable, Dict, List, Optional

import optuna
from optuna.pruners import HyperbandPruner, MedianPruner
from optuna.samplers import NSGAIISampler, TPESampler

from bioplausible.models.registry import ModelSpec, get_model_spec


def create_optuna_space(
    trial: optuna.Trial, model_name: str, constraints: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create Optuna hyperparameter space from ModelSpec.

    Args:
        trial: Optuna trial object
        model_name: Name of model from ModelRegistry
        constraints: Optional constraints (max_layers, max_hidden, etc.)

    Returns:
        Config dictionary with sampled hyperparameters
    """
    spec = get_model_spec(model_name)
    config = {}

    # Universal hyperparameters
    config["lr"] = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

    # Architecture
    max_hidden = constraints.get("max_hidden", 512) if constraints else 512
    config["hidden_dim"] = trial.suggest_categorical(
        "hidden_dim", [64, 128, 256, min(512, max_hidden)]
    )

    max_layers = constraints.get("max_layers", 30) if constraints else 30
    if spec.family in ["eqprop", "hybrid", "hebbian"]:
        # Deeper networks for bio-plausible algorithms
        config["num_layers"] = trial.suggest_int("num_layers", 2, max_layers)
    else:
        config["num_layers"] = trial.suggest_int("num_layers", 2, min(6, max_layers))

    # Model-specific parameters
    if spec.has_beta:
        # Different ranges for different families
        if "Finite-Nudge" in spec.name:
            config["beta"] = trial.suggest_float("beta", 0.5, 3.0)
        elif "Holomorphic" in spec.name:
            config["beta"] = trial.suggest_float("beta", 0.01, 0.3)
        else:
            config["beta"] = trial.suggest_float("beta", 0.05, 0.5)

    if spec.has_steps:
        max_steps = constraints.get("max_steps", 40) if constraints else 40
        if "Transformer" in spec.name:
            config["steps"] = trial.suggest_int("steps", 5, min(20, max_steps))
        else:
            config["steps"] = trial.suggest_int("steps", 5, max_steps)

    # Custom hyperparams from ModelSpec
    for param, default in spec.custom_hyperparams.items():
        if isinstance(default, float):
            if param.endswith("_scale") or param.endswith("_rate"):
                # Scale/rate parameters - log space
                config[param] = trial.suggest_float(param, default * 0.1, default * 10.0, log=True)
            else:
                # Other floats - linear
                config[param] = trial.suggest_float(param, default * 0.5, default * 2.0)
        elif isinstance(default, int):
            config[param] = trial.suggest_int(param, max(1, default // 2), default * 2)
        elif isinstance(default, str):
            # Categorical - just use default for now
            config[param] = default

    return config


def create_study(
    model_names: List[str],
    n_objectives: int = 2,
    storage: Optional[str] = None,
    study_name: Optional[str] = None,
    use_pruning: bool = True,
    sampler_name: str = "tpe",
) -> optuna.Study:
    """
    Create an Optuna study for hyperparameter optimization.

    Args:
        model_names: List of model names to optimize
        n_objectives: Number of objectives (1=single, 2=multi like accuracy+loss)
        storage: Storage URL (e.g., "sqlite:///optuna.db"). None for in-memory.
        study_name: Name for the study
        use_pruning: Whether to use automatic pruning
        sampler_name: "tpe", "nsga2", or "random"

    Returns:
        Optuna study object
    """
    # Direction: maximize accuracy, minimize loss
    directions = ["maximize", "minimize"] if n_objectives == 2 else ["maximize"]

    # Sampler selection
    if sampler_name == "nsga2":
        sampler = NSGAIISampler()
    elif sampler_name == "random":
        sampler = optuna.samplers.RandomSampler()
    else:  # TPE
        sampler = TPESampler(multivariate=True, n_startup_trials=10)

    # Pruner selection
    pruner = HyperbandPruner() if use_pruning else MedianPruner()

    study = optuna.create_study(
        directions=directions,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )

    return study


def get_pareto_trials(study: optuna.Study) -> List[optuna.trial.FrozenTrial]:
    """
    Get Pareto frontier trials from a multi-objective study.

    Args:
        study: Optuna study

    Returns:
        List of trials on the Pareto frontier
    """
    if len(study.directions) == 1:
        # Single objective - just return best trial
        return [study.best_trial]

    # Multi-objective - get Pareto front
    return study.best_trials


def trial_to_metrics(trial: optuna.trial.FrozenTrial) -> Dict[str, Any]:
    """
    Convert Optuna trial to metrics format compatible with existing code.

    Args:
        trial: Optuna trial

    Returns:
        Metrics dictionary
    """
    metrics = {
        "config": trial.params,
        "trial_id": trial.number,
        "state": trial.state.name,
    }

    if trial.values:
        if len(trial.values) == 2:
            metrics["accuracy"] = trial.values[0]
            metrics["loss"] = trial.values[1]
        else:
            metrics["score"] = trial.values[0]

    return metrics


def optimize_with_callback(
    study: optuna.Study,
    objective: Callable,
    n_trials: int,
    callbacks: Optional[List[Callable]] = None,
) -> None:
    """
    Run optimization with custom callbacks (for UI updates).

    Args:
        study: Optuna study
        objective: Objective function
        n_trials: Number of trials to run
        callbacks: List of callback functions
    """
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=callbacks,
        show_progress_bar=True,
    )
