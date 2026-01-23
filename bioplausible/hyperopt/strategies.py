"""
Search Strategy Implementations

DEPRECATED: Use Optuna samplers instead.
- GridSampler for grid search
- RandomSampler for random search  
- TPESampler for smarter Bayesian optimization

These functions are kept only for backward compatibility.
"""

import itertools
from typing import Any, Dict, Iterator, List, Union

import numpy as np


def GridSearch(space: Dict[str, Union[List, tuple]]) -> Iterator[Dict[str, Any]]:
    """
    Generator for Grid Search over a parameter space.
    
    DEPRECATED: Use optuna.samplers.GridSampler instead.

    Args:
        space: Dict where keys are param names and values are lists of discrete choices.

    Yields:
        Config dictionary
    """
    keys = []
    values = []

    for k, v in space.items():
        if isinstance(v, list):
            keys.append(k)
            values.append(v)
        elif isinstance(v, tuple):
            # Simple discretization for grid search if tuple provided
            # (min, max, type)
            if len(v) == 3 and v[2] == "int":
                keys.append(k)
                values.append(list(range(int(v[0]), int(v[1]) + 1)))

    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))


def RandomSearch(
    space: Dict[str, Union[List, tuple]], n_iter: int = 10
) -> Iterator[Dict[str, Any]]:
    """
    Generator for Random Search.
    
    DEPRECATED: Use optuna.samplers.RandomSampler instead.

    Args:
        space: Dict of parameter spaces.
        n_iter: Number of configurations to sample.

    Yields:
        Config dictionary
    """
    rng = np.random.default_rng()

    for _ in range(n_iter):
        config = {}
        for param_name, param_spec in space.items():
            if isinstance(param_spec, tuple):
                # Continuous or integer range
                min_val, max_val, scale = param_spec
                if scale == "log":
                    val = float(np.exp(rng.uniform(np.log(min_val), np.log(max_val))))
                elif scale == "int":
                    val = int(rng.integers(min_val, max_val + 1))
                else:  # linear
                    val = float(rng.uniform(min_val, max_val))
                config[param_name] = val
            elif isinstance(param_spec, list):
                # Discrete choice
                choice = rng.choice(param_spec)
                config[param_name] = (
                    int(choice)
                    if isinstance(choice, (np.integer, np.int64))
                    else choice
                )
        yield config
