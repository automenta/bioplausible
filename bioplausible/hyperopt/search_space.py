"""
Search Space Definitions

Defines the hyperparameter search spaces for each model type in the registry.
"""

from typing import Dict, Any, Tuple, List, Union
import numpy as np

# Type aliases
NumberRange = Tuple[float, float, str]  # (min, max, scale) where scale in ['log', 'linear', 'int']
DiscreteChoice = List[Union[int, float, str]]

class SearchSpace:
    """Hyperparameter search space for a model."""
    
    def __init__(self, name: str, params: Dict[str, Union[NumberRange, DiscreteChoice]]):
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
                if scale == 'log':
                    val = float(np.exp(rng.uniform(np.log(min_val), np.log(max_val))))
                elif scale == 'int':
                    val = int(rng.integers(min_val, max_val + 1))
                else:  # linear
                    val = float(rng.uniform(min_val, max_val))
                config[param_name] = val
            else:
                # Discrete choice - ensure native Python type
                choice = rng.choice(param_spec)
                config[param_name] = int(choice) if isinstance(choice, (np.integer, np.int64)) else choice
        
        return config
    
    def mutate(self, config: Dict[str, Any], mutation_rate: float = 0.3, rng: np.random.Generator = None) -> Dict[str, Any]:
        """Mutate a configuration."""
        if rng is None:
            rng = np.random.default_rng()
        
        mutated = config.copy()
        for param_name, param_spec in self.params.items():
            if rng.random() < mutation_rate:
                if isinstance(param_spec, tuple):
                    min_val, max_val, scale = param_spec
                    if scale == 'log':
                        # Gaussian perturbation in log space
                        current = mutated.get(param_name)
                        if current is None:
                            # Initialize if missing
                            current = rng.uniform(min_val, max_val)
                        
                        log_val = np.log(current) + rng.normal(0, 0.5)
                        mutated[param_name] = float(np.clip(np.exp(log_val), min_val, max_val))
                    elif scale == 'int':
                        # Random walk
                        current = mutated.get(param_name)
                        if current is None:
                            current = int((min_val + max_val) / 2)
                            
                        delta = rng.integers(-2, 3)
                        mutated[param_name] = int(np.clip(current + delta, min_val, max_val))
                    else:  # linear
                        # Gaussian perturbation
                        current = mutated.get(param_name)
                        if current is None:
                            current = (min_val + max_val) / 2.0
                            
                        span = max_val - min_val
                        mutated[param_name] = float(np.clip(
                            current + rng.normal(0, span * 0.1),
                            min_val, max_val
                        ))
                else:
                    # Random new choice
                    mutated[param_name] = rng.choice(param_spec)
        
        return mutated
    
    def crossover(self, config1: Dict[str, Any], config2: Dict[str, Any], rng: np.random.Generator = None) -> Dict[str, Any]:
        """Crossover two configurations."""
        if rng is None:
            rng = np.random.default_rng()
        
        child = {}
        for param_name in self.params.keys():
            # Uniform crossover
            child[param_name] = config1[param_name] if rng.random() < 0.5 else config2[param_name]
        
        return child


# Define search spaces for all models
SEARCH_SPACES = {
    "Backprop (Transformer)": SearchSpace(
        "Backprop (Transformer)",
        {
            'lr': (1e-5, 1e-2, 'log'),
            'hidden_dim': [64, 128, 256, 512],
            'num_layers': [2, 4, 6],
        }
    ),
    
    "EqProp MLP": SearchSpace(
        "EqProp MLP",
        {
            'lr': (1e-5, 1e-2, 'log'),
            'beta': (0.05, 0.5, 'linear'),
            'steps': (5, 20, 'int'),
            'hidden_dim': [64, 128],
            'num_layers': [5, 10, 15],
        }
    ),
    
    "EqProp Transformer (Attention Only)": SearchSpace(
        "EqProp Transformer (Attention Only)",
        {
            'lr': (1e-5, 1e-2, 'log'),
            'steps': (5, 12, 'int'),
            'hidden_dim': [64, 128, 256],
            'num_layers': [2, 3],
        }
    ),
    
    "EqProp Transformer (Full)": SearchSpace(
        "EqProp Transformer (Full)",
        {
            'lr': (1e-5, 1e-2, 'log'),
            'steps': (5, 20, 'int'),
            'hidden_dim': [64, 128],
            'num_layers': [2, 3],
        }
    ),
    
    "EqProp Transformer (Hybrid)": SearchSpace(
        "EqProp Transformer (Hybrid)",
        {
            'lr': (1e-5, 1e-2, 'log'),
            'steps': (5, 15, 'int'),
            'hidden_dim': [128, 256],
            'num_layers': [2, 3],
        }
    ),
    
    "EqProp Transformer (Recurrent)": SearchSpace(
        "EqProp Transformer (Recurrent)",
        {
            'lr': (1e-5, 1e-2, 'log'),
            'steps': (10, 30, 'int'),
            'hidden_dim': [128, 256],
            'num_layers': [1],  # Recurrent uses single block
        }
    ),
    
    "DFA (Direct Feedback Alignment)": SearchSpace(
        "DFA (Direct Feedback Alignment)",
        {
            'lr': (1e-5, 1e-2, 'log'),
            'hidden_dim': [64, 128, 256],
            'num_layers': [10, 20, 30],
        }
    ),
    
    "CHL (Contrastive Hebbian)": SearchSpace(
        "CHL (Contrastive Hebbian)",
        {
            'lr': (1e-5, 1e-2, 'log'),
            'beta': (0.05, 0.3, 'linear'),
            'steps': (10, 30, 'int'),
            'hidden_dim': [64, 128, 256],
            'num_layers': [10, 20, 30],
        }
    ),
    
    "Deep Hebbian (Hundred-Layer)": SearchSpace(
        "Deep Hebbian (Hundred-Layer)",
        {
            'lr': (1e-5, 5e-3, 'log'),
            'hidden_dim': [64, 128],
            'num_layers': [50, 100, 150],  # Test deep scaling
        }
    ),
}


def get_search_space(model_name: str) -> SearchSpace:
    """Get the search space for a model."""
    if model_name not in SEARCH_SPACES:
        raise ValueError(f"No search space defined for model: {model_name}")
    return SEARCH_SPACES[model_name]
