"""
Search Space Definitions

Defines the hyperparameter search spaces for each model type in the registry.
"""

from typing import Dict, Any, Tuple, List, Union
import numpy as np
from bioplausible.models.registry import MODEL_REGISTRY

# Type aliases
NumberRange = Tuple[
    float, float, str
]  # (min, max, scale) where scale in ['log', 'linear', 'int']
DiscreteChoice = List[Union[int, float, str]]


class SearchSpace:
    """Hyperparameter search space for a model."""

    def __init__(
        self, name: str, params: Dict[str, Union[NumberRange, DiscreteChoice]]
    ):
        self.name = name
        self.params = params

    def sample(self, rng: np.random.Generator = None) -> Dict[str, Any]:
        """Sample a random configuration from the search space."""
        if rng is None:
            rng = np.random.default_rng()

        config = {}
        for param_name, param_spec in self.params.items():
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
            else:
                # Discrete choice - ensure native Python type
                choice = rng.choice(param_spec)
                config[param_name] = (
                    int(choice)
                    if isinstance(choice, (np.integer, np.int64))
                    else choice
                )

        return config

    def mutate(
        self,
        config: Dict[str, Any],
        mutation_rate: float = 0.3,
        rng: np.random.Generator = None,
    ) -> Dict[str, Any]:
        """Mutate a configuration."""
        if rng is None:
            rng = np.random.default_rng()

        mutated = config.copy()
        for param_name, param_spec in self.params.items():
            if rng.random() < mutation_rate:
                if isinstance(param_spec, tuple):
                    min_val, max_val, scale = param_spec
                    if scale == "log":
                        # Gaussian perturbation in log space
                        current = mutated.get(param_name)
                        if current is None:
                            # Initialize if missing
                            current = rng.uniform(min_val, max_val)

                        log_val = np.log(current) + rng.normal(0, 0.5)
                        mutated[param_name] = float(
                            np.clip(np.exp(log_val), min_val, max_val)
                        )
                    elif scale == "int":
                        # Random walk
                        current = mutated.get(param_name)
                        if current is None:
                            current = int((min_val + max_val) / 2)

                        delta = rng.integers(-2, 3)
                        mutated[param_name] = int(
                            np.clip(current + delta, min_val, max_val)
                        )
                    else:  # linear
                        # Gaussian perturbation
                        current = mutated.get(param_name)
                        if current is None:
                            current = (min_val + max_val) / 2.0

                        span = max_val - min_val
                        mutated[param_name] = float(
                            np.clip(
                                current + rng.normal(0, span * 0.1), min_val, max_val
                            )
                        )
                else:
                    # Random new choice
                    mutated[param_name] = rng.choice(param_spec)

        return mutated

    def crossover(
        self,
        config1: Dict[str, Any],
        config2: Dict[str, Any],
        rng: np.random.Generator = None,
    ) -> Dict[str, Any]:
        """Crossover two configurations."""
        if rng is None:
            rng = np.random.default_rng()

        child = {}
        for param_name in self.params.keys():
            # Uniform crossover
            child[param_name] = (
                config1[param_name] if rng.random() < 0.5 else config2[param_name]
            )

        return child

    def apply_constraints(self, constraints: Dict[str, Any]) -> "SearchSpace":
        """
        Return a new SearchSpace with constraints applied.

        Supported constraints:
        - max_layers (int): Cap the maximum number of layers
        - max_hidden (int): Cap the maximum hidden dimension
        - max_steps (int): Cap the maximum equilibrium steps
        """
        if not constraints:
            return self

        new_params = self.params.copy()

        for param, spec in new_params.items():
            # Handle discrete choices (list)
            if isinstance(spec, list):
                if param == "num_layers" and "max_layers" in constraints:
                    new_params[param] = [x for x in spec if x <= constraints["max_layers"]]
                    if not new_params[param]: # Fallback if empty
                        new_params[param] = [min(spec)]

                elif param == "hidden_dim" and "max_hidden" in constraints:
                    new_params[param] = [x for x in spec if x <= constraints["max_hidden"]]
                    if not new_params[param]:
                        new_params[param] = [min(spec)]

                elif param == "steps" and "max_steps" in constraints:
                    new_params[param] = [x for x in spec if x <= constraints["max_steps"]]
                    if not new_params[param]:
                        new_params[param] = [min(spec)]

            # Handle continuous ranges (tuple)
            elif isinstance(spec, tuple):
                min_val, max_val, scale = spec

                if param == "num_layers" and "max_layers" in constraints:
                    max_val = min(max_val, constraints["max_layers"])
                    if min_val > max_val: min_val = max_val
                    new_params[param] = (min_val, max_val, scale)

                elif param == "hidden_dim" and "max_hidden" in constraints:
                    max_val = min(max_val, constraints["max_hidden"])
                    if min_val > max_val: min_val = max_val
                    new_params[param] = (min_val, max_val, scale)

                elif param == "steps" and "max_steps" in constraints:
                    max_val = min(max_val, constraints["max_steps"])
                    if min_val > max_val: min_val = max_val
                    new_params[param] = (min_val, max_val, scale)

        return SearchSpace(self.name, new_params)


# Define search spaces for all models
SEARCH_SPACES = {
    "Backprop Baseline": SearchSpace(
        "Backprop Baseline",
        {
            "lr": (1e-5, 1e-2, "log"),
            "hidden_dim": [64, 128, 256, 512],
            "num_layers": [2, 4, 6],
        },
    ),
    "EqProp MLP": SearchSpace(
        "EqProp MLP",
        {
            "lr": (1e-5, 1e-2, "log"),
            "beta": (0.05, 0.5, "linear"),
            "steps": (5, 20, "int"),
            "hidden_dim": [64, 128],
            "num_layers": [5, 10, 15],
        },
    ),
    # Research Models
    "Holomorphic EqProp": SearchSpace(
        "Holomorphic EqProp",
        {
            "lr": (1e-4, 1e-2, "log"),
            "beta": (0.01, 0.3, "linear"),
            "steps": (10, 40, "int"),
            "hidden_dim": [64, 128],
        },
    ),
    "Directed EqProp (Deep EP)": SearchSpace(
        "Directed EqProp (Deep EP)",
        {
            "lr": (1e-4, 1e-2, "log"),
            "beta": (0.1, 0.5, "linear"),
            "steps": (10, 40, "int"),
            "hidden_dim": [64, 128],
        },
    ),
    "Finite-Nudge EqProp": SearchSpace(
        "Finite-Nudge EqProp",
        {
            "lr": (1e-4, 1e-2, "log"),
            "beta": (0.5, 3.0, "linear"), # Large beta
            "steps": (10, 40, "int"),
            "hidden_dim": [64, 128],
        },
    ),
    "Conv EqProp (CIFAR-10)": SearchSpace(
        "Conv EqProp (CIFAR-10)",
        {
            "lr": (1e-4, 1e-2, "log"),
            "steps": (10, 25, "int"),
            "hidden_dim": [128, 256],
        },
    ),
    # Hybrid & Experimental
    "Adaptive Feedback Alignment": SearchSpace(
        "Adaptive Feedback Alignment",
        {
            "lr": (1e-4, 1e-2, "log"),
            "fa_scale": (0.5, 1.5, "linear"),
            "adapt_rate": (0.001, 0.1, "log"),
            "hidden_dim": [64, 128, 256],
        },
    ),
    "Equilibrium Alignment": SearchSpace(
        "Equilibrium Alignment",
        {
            "lr": (1e-4, 1e-2, "log"),
            "beta": (0.1, 0.5, "linear"),
            "steps": (10, 30, "int"),
            "align_weight": (0.1, 1.0, "linear"),
        },
    ),
    # Add Missing Spaces
    "Layerwise Equilibrium FA": SearchSpace(
        "Layerwise Equilibrium FA",
        {
            "lr": (1e-4, 1e-2, "log"),
            "hidden_dim": [64, 128],
            "num_layers": [2, 4, 6]
        }
    ),
    "Energy Guided FA": SearchSpace(
        "Energy Guided FA",
        {
            "lr": (1e-4, 1e-2, "log"),
            "energy_scale": (0.1, 1.0, "linear"),
            "hidden_dim": [64, 128]
        }
    ),
    "Predictive Coding Hybrid": SearchSpace(
        "Predictive Coding Hybrid",
        {
            "lr": (1e-4, 1e-2, "log"),
            "steps": (10, 30, "int"),
            "hidden_dim": [64, 128]
        }
    ),
    "Sparse Equilibrium": SearchSpace(
        "Sparse Equilibrium",
        {
            "lr": (1e-4, 1e-2, "log"),
            "beta": (0.05, 0.3, "linear"),
            "sparsity": (0.1, 0.9, "linear"),
            "hidden_dim": [128, 256]
        }
    ),
    "Momentum Equilibrium": SearchSpace(
        "Momentum Equilibrium",
        {
            "lr": (1e-4, 1e-2, "log"),
            "momentum": (0.5, 0.95, "linear"),
            "steps": (10, 30, "int")
        }
    ),
    "Stochastic FA": SearchSpace(
        "Stochastic FA",
        {
            "lr": (1e-4, 1e-2, "log"),
            "noise_scale": (0.01, 0.2, "log"),
            "hidden_dim": [64, 128]
        }
    ),
    "Energy Minimizing FA": SearchSpace(
        "Energy Minimizing FA",
        {
            "lr": (1e-4, 1e-2, "log"),
            "hidden_dim": [64, 128]
        }
    ),

    # Transformers
    "EqProp Transformer (Attention Only)": SearchSpace(
        "EqProp Transformer (Attention Only)",
        {
            "lr": (1e-5, 1e-2, "log"),
            "steps": (5, 12, "int"),
            "hidden_dim": [64, 128, 256],
            "num_layers": [2, 3],
        },
    ),
    "EqProp Transformer (Full)": SearchSpace(
        "EqProp Transformer (Full)",
        {
            "lr": (1e-5, 1e-2, "log"),
            "steps": (5, 20, "int"),
            "hidden_dim": [64, 128],
            "num_layers": [2, 3],
        },
    ),
    "EqProp Transformer (Hybrid)": SearchSpace(
        "EqProp Transformer (Hybrid)",
        {
            "lr": (1e-5, 1e-2, "log"),
            "steps": (5, 15, "int"),
            "hidden_dim": [128, 256],
            "num_layers": [2, 3],
        },
    ),
    "EqProp Transformer (Recurrent)": SearchSpace(
        "EqProp Transformer (Recurrent)",
        {
            "lr": (1e-5, 1e-2, "log"),
            "steps": (10, 30, "int"),
            "hidden_dim": [128, 256],
            "num_layers": [1],  # Recurrent uses single block
        },
    ),
    "DFA (Direct Feedback Alignment)": SearchSpace(
        "DFA (Direct Feedback Alignment)",
        {
            "lr": (1e-5, 1e-2, "log"),
            "hidden_dim": [64, 128, 256],
            "num_layers": [10, 20, 30],
        },
    ),
    "CHL (Contrastive Hebbian)": SearchSpace(
        "CHL (Contrastive Hebbian)",
        {
            "lr": (1e-5, 1e-2, "log"),
            "beta": (0.05, 0.3, "linear"),
            "steps": (10, 30, "int"),
            "hidden_dim": [64, 128, 256],
            "num_layers": [10, 20, 30],
        },
    ),
    "Deep Hebbian (Hundred-Layer)": SearchSpace(
        "Deep Hebbian (Hundred-Layer)",
        {
            "lr": (1e-5, 5e-3, "log"),
            "hidden_dim": [64, 128],
            "num_layers": [50, 100, 150],  # Test deep scaling
        },
    ),
}

# Import helpers here to avoid circular imports if necessary,
# but usually strategies depends on SearchSpace, not vice versa.
# However, for consistency, we can export GridSearch/RandomSearch from here if we want them accessible via search_space.py
from bioplausible.hyperopt.strategies import GridSearch, RandomSearch

def get_search_space(model_name: str) -> SearchSpace:
    """Get the search space for a model."""
    # 1. Try hardcoded spaces first (for customized ranges)
    if model_name in SEARCH_SPACES:
        return SEARCH_SPACES[model_name]

    # 2. Try to generate from registry
    # Check if exact name in registry
    spec = next((s for s in MODEL_REGISTRY if s.name == model_name), None)

    if spec:
        params = {
            "lr": (1e-5, 1e-2, "log"),
            "hidden_dim": [64, 128, 256],
            "num_layers": [2, 4, 6],
        }

        if spec.has_beta:
             params["beta"] = (0.05, 0.5, "linear")

        if spec.has_steps:
             params["steps"] = (5, 30, "int")

        return SearchSpace(model_name, params)

    # 3. Fallback for completely unknown models
    # Try to infer based on name/flags if possible, or raise error
    if "EqProp" in model_name:
         params = {
            "lr": (1e-5, 1e-2, "log"),
            "beta": (0.05, 0.5, "linear"),
            "steps": (5, 20, "int"),
            "hidden_dim": [64, 128],
         }
         return SearchSpace(model_name, params)

    raise ValueError(f"No search space defined for model: {model_name}")
