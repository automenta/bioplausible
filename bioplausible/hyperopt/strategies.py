"""
Search Strategy Implementations

Implements GridSearch and RandomSearch generators.
"""

import itertools
from typing import Any, Dict, Iterator, List, Union

import numpy as np


def GridSearch(space: Dict[str, Union[List, tuple]]) -> Iterator[Dict[str, Any]]:
    """
    Generator for Grid Search over a parameter space.

    Args:
        space: Dict where keys are param names and values are lists of discrete choices.
               Continuous ranges (tuples) are NOT supported for Grid Search directly,
               they should be discretized first or handled by caller.

    Yields:
        Config dictionary
    """
    # Filter out non-list values (assume fixed or range requiring discretization)
    # For now, we only grid search over explicit lists.
    # If a tuple is found, we might need to error or sample discrete points?
    # Let's assume standard usage is providing discrete lists.

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
            # Ignore continuous for now or sample 3 points?
            # Let's ignore to avoid explosion
            pass
        else:
            # Fixed value
            pass

    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))


def RandomSearch(
    space: Dict[str, Union[List, tuple]], n_iter: int = 10
) -> Iterator[Dict[str, Any]]:
    """
    Generator for Random Search.

    Args:
        space: Dict of parameter spaces.
        n_iter: Number of configurations to sample.

    Yields:
        Config dictionary
    """
    from bioplausible.hyperopt.search_space import SearchSpace

    # Wrap in SearchSpace object
    search_space = SearchSpace("temp", space)
    rng = np.random.default_rng()

    for _ in range(n_iter):
        yield search_space.sample(rng)
