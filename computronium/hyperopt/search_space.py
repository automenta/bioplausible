"""
Search Space Definitions

Defines the hyperparameter search spaces for each learning rule.
"""

import numpy as np

# Type aliases

__all__ = [
    "RULE_SPACES",
    "DiscreteChoice",
    "NumberRange",
    "SearchSpace",
    "get_rule_space",
    "get_search_space",
]
NumberRange = tuple[
    float, float, str
]  # (min, max, scale) where scale in ['log', 'linear', 'int']
DiscreteChoice = list[int | float | str]

#: Length of a :data:`NumberRange` spec ``(min, max, scale)``.
_RANGE_LEN = 3

#: Probability that a crossover picks from the first parent vs the second.
_CROSSOVER_BIAS = 0.5


class SearchSpace:
    """
    Hyperparameter search space for a model.

    Note: This class now only stores parameter definitions.
    All sampling/mutation/crossover is handled by Optuna.
    Use optuna_bridge.create_optuna_space() for optimization.
    """

    def __init__(self, name: str, params: dict[str, NumberRange | DiscreteChoice]):
        self.name = name
        self.params = params

    def _sample_param(self, name: str, space: NumberRange | DiscreteChoice) -> object:
        """Draw a single value for a parameter from its spec."""
        if isinstance(space, list):
            return self._sample_discrete(space)
        if isinstance(space, tuple) and len(space) == _RANGE_LEN:
            # Number range
            min_val, max_val, scale = space
            if scale == "int":
                return int(np.random.randint(min_val, max_val + 1))
            if scale == "log":
                # Log uniform
                return float(
                    np.exp(np.random.uniform(np.log(min_val), np.log(max_val)))
                )
            # Linear
            return float(np.random.uniform(min_val, max_val))
        raise ValueError(f"Invalid spec for '{name}': {space!r}")

    @staticmethod
    def _sample_discrete(space: DiscreteChoice) -> object:
        """Pick a discrete choice, normalizing numpy scalar scalars."""
        value = np.random.choice(space)
        return value.item() if isinstance(value, np.generic) else value

    def sample(self) -> dict[str, object]:
        """Sample a random configuration from the search space."""
        return {
            name: self._sample_param(name, space) for name, space in self.params.items()
        }

    def crossover(
        self, config_a: dict[str, object], config_b: dict[str, object]
    ) -> dict[str, object]:
        """Uniform crossover: per-parameter pick from either parent, or resample if absent."""
        child: dict[str, object] = {}
        for name, space in self.params.items():
            if name in config_a and name in config_b:
                child[name] = (
                    config_a[name]
                    if np.random.rand() < _CROSSOVER_BIAS
                    else config_b[name]
                )
            elif name in config_a:
                child[name] = config_a[name]
            elif name in config_b:
                child[name] = config_b[name]
            else:
                child[name] = self._sample_param(name, space)
        return child

    def _mutate_param(
        self,
        name: str,
        space: NumberRange | DiscreteChoice,
        value: object,
        perturb: bool,
    ) -> object:
        """Return a mutated (or clamped) value for one parameter."""
        if isinstance(space, list):
            # Discrete choice: snap to nearest allowed value, then (optionally)
            # jump to a different choice.
            nearest = min(space, key=lambda choice: abs(float(choice) - float(value)))
            if not perturb:
                return nearest
            others = [choice for choice in space if choice != nearest]
            return np.random.choice(others) if others else nearest
        if not (isinstance(space, tuple) and len(space) == _RANGE_LEN):
            raise ValueError(f"Invalid spec for '{name}': {space!r}")
        return self._mutate_range(space, value, perturb)

    @staticmethod
    def _mutate_range(
        space: NumberRange,
        value: object,
        perturb: bool,
    ) -> object:
        """Clamp a numeric value to its range; optionally perturb it."""
        min_val, max_val, scale = space
        if scale == "int":
            current = min(max(round(value), min_val), max_val)
            if not perturb:
                return current
            delta = np.random.choice([-1, 0, 1])
            return min(max(current + delta, min_val), max_val)
        current = min(max(float(value), min_val), max_val)
        if not perturb:
            return current
        if scale == "log":
            factor = float(np.exp(np.random.uniform(-1.0, 1.0)))
            return min(max(current * factor, min_val), max_val)
        width = max_val - min_val
        return min(
            max(current + float(np.random.uniform(-width, width)), min_val),
            max_val,
        )

    def mutate(
        self,
        config: dict[str, object],
        mutation_rate: float = 0.1,
        rng: object = None,
    ) -> dict[str, object]:
        """Mutate a config respecting bounds; clamp out-of-range values.

        Args:
            config: The configuration to mutate (not modified; a copy is returned).
            mutation_rate: Fraction of parameters to perturb (0.0 = clamp-only).
            rng: Optional random number generator (defaults to ``np.random``).
        """
        rand = rng if rng is not None else np.random
        mutated = dict(config)
        for name, space in self.params.items():
            if name not in mutated:
                continue
            value = mutated[name]
            mutated[name] = self._mutate_param(
                name, space, value, perturb=rand.rand() < mutation_rate
            )
        return mutated

    def apply_constraints(self, constraints: dict[str, object]) -> SearchSpace:
        """
        Return a new constrained search space based on constraints dictionary.
        Supports max_hidden, max_layers, max_steps.
        """
        import copy

        new_params = copy.deepcopy(self.params)

        # Constraint name → candidate param keys (RULE_SPACES names the settle
        # budget ``max_steps``; the legacy evolution spaces used ``steps``).
        mapping = [
            ("max_hidden", "hidden_dim"),
            ("max_layers", "num_layers"),
            ("max_steps", "steps"),
            ("max_steps", "max_steps"),
        ]

        for const_key, param_key in mapping:
            limit = constraints.get(const_key)
            if limit is None or param_key not in new_params:
                continue
            space = new_params[param_key]
            if isinstance(space, list):
                new_params[param_key] = [v for v in space if v <= limit]
            elif isinstance(space, tuple) and len(space) == _RANGE_LEN:
                min_val, max_val, scale = space
                new_max = min(max_val, limit)
                new_max = max(new_max, min_val)  # Safe fallback
                new_params[param_key] = (min_val, new_max, scale)

        return SearchSpace(self.name + "_constrained", new_params)


def get_available_models() -> list[str]:
    """Rule families with a sampled search space (the native-model surface)."""
    return sorted(RULE_SPACES)


def get_search_space(model_name: str) -> SearchSpace:
    """Resolve the search space for a learning-rule key.

    Raises:
        ValueError: If ``model_name`` is not a ``RULE_SPACES`` rule key.
    """
    if model_name not in RULE_SPACES:
        raise ValueError(f"No search space defined for model: {model_name}")
    return SearchSpace(model_name, RULE_SPACES[model_name])


# Continuous, log-sampled search spaces per learning rule (plan §4A, §10).
# These replace the coarse discrete grids with true Bayesian ranges so that
# (a) TPE explores the posterior rather than a handful of points, and (b) each
# rule is compared at its own optimum — including rule-specific equilibrium
# hyperparameters (damping, step size, max iterations, convergence threshold).
RULE_SPACES: dict[str, dict[str, NumberRange | DiscreteChoice]] = {
    "backprop": {
        "learning_rate": (1e-5, 1e-1, "log"),
        "weight_decay": (1e-6, 1e-2, "log"),
        "hidden_dim": (32, 1024, "log"),
        "num_layers": (1, 6, "int"),
    },
    "eqprop": {
        # Energy-contrastive EqProp update scale is `lr * (gn - gf) / beta`.
        # Plan-6 §8.6 / §10.2 found the previous range (lr 1e-5..1e-2,
        # beta 0.05..0.5) starved the rule: the optimal operating point is
        # lr ~ 0.05-0.1 and beta ~ 0.01-0.1 (hand-tuned probe reached 34% at
        # lr=0.05, beta=0.1 over 100 steps, vs 10-14% with the old space).
        # Diverged probes [lr too high for a given beta] are quarantined by the
        # sweep's nan_divergence defect gate — the space is not responsible for
        # preventing divergence; it must include the working region.
        "learning_rate": (1e-2, 5e-1, "log"),
        "weight_decay": (1e-6, 1e-2, "log"),
        "hidden_dim": (32, 1024, "log"),
        "num_layers": (1, 6, "int"),
        "beta": (1e-3, 1e-1, "log"),
        "max_steps": (5, 100, "int"),
        "damping": (0.0, 0.9, "linear"),
        "tol": (1e-6, 1e-2, "log"),
        "convergence_threshold": (1e-4, 1e-2, "log"),
        "convergence_start": (2, 10, "int"),
        "sparse_ratio": (0.5, 1.0, "linear"),
        "momentum": (0.0, 0.9, "linear"),
        # Plan 8: separate true β from per-layer update scaling
        "update_scale": (1e-2, 1e1, "log"),
        "update_scale_by_depth": (1e-1, 1e1, "log"),
        # Plan 8: recurrent weight initialization knob
        "w_rec_init": ["zero", "xavier"],
        "w_rec_gain": (1e-3, 1e0, "log"),
        # Plan 8 B3: explicit feedback-pathway knobs (DirectedEP).
        # ``feedback_gain`` scales the output→hidden feedback drive in the
        # nudged phase; ``feedback_init_gain`` sets the xavier gain of the
        # feedback weight matrices. Both are optimization/magnitude knobs for
        # the feedback pathway, mathematically distinct from ``beta``.
        "feedback_gain": (1e-2, 1e1, "log"),
        "feedback_init_gain": (1e-3, 1e0, "log"),
    },
    "neural_cube": {
        # Honest space (P0a): every knob is real — accepted by ``NeuralCube.__init__``
        # or routed by the training loop. ``hidden_dim``/``damping``/``tol`` were
        # silently dropped by ``build_model_kwargs`` (phantom drift, §0.1); re-add
        # each dimension in the same change that implements it on the model.
        "learning_rate": (1e-5, 1e-1, "log"),
        "weight_decay": (1e-6, 1e-2, "log"),
        "cube_size": (3, 10, "int"),
        "max_steps": (5, 100, "int"),
    },
    "pepita": {
        "learning_rate": (1e-4, 1e-1, "log"),
        "hidden_dim": (32, 512, "log"),
        "num_layers": (1, 4, "int"),
    },
    "forward_forward": {
        "learning_rate": (1e-3, 1e-1, "log"),
        "hidden_dim": (32, 512, "log"),
        "num_layers": (1, 4, "int"),
        "threshold": (0.5, 5.0, "linear"),
        "layer_lr": (1e-3, 1e-1, "log"),
        "classifier_lr": (1e-3, 1e-1, "log"),
    },
    "feedback_alignment": {
        "learning_rate": (1e-4, 1e-1, "log"),
        "hidden_dim": (32, 512, "log"),
        "num_layers": (1, 4, "int"),
        "alpha": (0.1, 1.0, "linear"),
        "feedback_mode": ["random", "symmetric", "transpose"],
        "use_spectral_norm": [True, False],
    },
    "target_prop": {
        "learning_rate": (1e-3, 1e-1, "log"),
        "target_lr": (1e-2, 1e0, "log"),
        "hidden_dim": (32, 512, "log"),
        "num_layers": (1, 4, "int"),
    },
}


def get_rule_space(rule: str) -> dict[str, NumberRange | DiscreteChoice]:
    """Return the continuous search space for a learning rule.

    Args:
        rule: Rule key from ``RULE_SPACES`` (e.g. ``"backprop"``).

    Returns:
        Parameter name → continuous range or discrete choice.

    Raises:
        ValueError: If the rule has no defined space.
    """
    try:
        return RULE_SPACES[rule]
    except KeyError:
        raise ValueError(
            f"No rule space defined for '{rule}'. Available: {sorted(RULE_SPACES)}"
        ) from None
