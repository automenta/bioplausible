"""
Search Space Definitions

Defines the hyperparameter search spaces for each model type in the registry.
"""

import hashlib
import inspect
from dataclasses import dataclass

import numpy as np

from bioplausible.core.exceptions import SpaceSignatureMismatchError
from bioplausible.core.registry import ComponentCategory, Registry
from bioplausible.zoo import get_model_spec

# Type aliases

__all__ = [
    "RULE_SPACES",
    "SEARCH_SPACES",
    "ConstructorSurface",
    "DiscreteChoice",
    "NumberRange",
    "SearchSpace",
    "emit_rule_space_surfaces",
    "get_rule_space",
    "get_search_space",
    "surface_for_rule",
    "validate_all_rule_spaces",
    "validate_rule_space",
]
NumberRange = tuple[
    float, float, str
]  # (min, max, scale) where scale in ['log', 'linear', 'int']
DiscreteChoice = list[int | float | str]


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

    def sample(self) -> dict[str, object]:
        """Sample a random configuration from the search space."""
        config = {}
        for name, space in self.params.items():
            if isinstance(space, list):
                # Discrete choice
                config[name] = np.random.choice(space)
                # Convert numpy types to python native
                if isinstance(config[name], (np.generic)):
                    config[name] = config[name].item()
            elif isinstance(space, tuple) and len(space) == 3:
                # Number range
                min_val, max_val, scale = space
                if scale == "int":
                    config[name] = int(np.random.randint(min_val, max_val + 1))
                elif scale == "log":
                    # Log uniform
                    log_min = np.log(min_val)
                    log_max = np.log(max_val)
                    config[name] = float(np.exp(np.random.uniform(log_min, log_max)))
                else:
                    # Linear
                    config[name] = float(np.random.uniform(min_val, max_val))
        return config

    def apply_constraints(self, constraints: dict[str, object]) -> SearchSpace:
        """
        Return a new constrained search space based on constraints dictionary.
        Supports max_hidden, max_layers, max_steps.
        """
        import copy

        new_params = copy.deepcopy(self.params)

        mapping = {
            "max_hidden": "hidden_dim",
            "max_layers": "num_layers",
            "max_steps": "steps",
        }

        for const_key, limit in constraints.items():
            if const_key in mapping:
                param_key = mapping[const_key]
                if param_key in new_params:
                    space = new_params[param_key]
                    if isinstance(space, list):
                        new_params[param_key] = [v for v in space if v <= limit]
                    elif isinstance(space, tuple) and len(space) == 3:
                        min_val, max_val, scale = space
                        new_max = min(max_val, limit)
                        new_max = max(new_max, min_val)  # Safe fallback
                        new_params[param_key] = (min_val, new_max, scale)

        return SearchSpace(self.name + "_constrained", new_params)


# Define search spaces for all models
SEARCH_SPACES = {
    "backprop_mlp": SearchSpace(
        "backprop_mlp",
        {
            "lr": (1e-5, 1e-2, "log"),
            "hidden_dim": [32, 64, 128, 256],
            "num_layers": [1, 2, 4],
        },
    ),
    "eqprop_mlp": SearchSpace(
        "eqprop_mlp",
        {
            "lr": (1e-5, 1e-2, "log"),
            "beta": (0.05, 0.5, "linear"),
            "steps": (5, 20, "int"),
            "hidden_dim": [32, 64, 128],
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
            "beta": (0.5, 3.0, "linear"),  # Large beta
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
        {"lr": (1e-4, 1e-2, "log"), "hidden_dim": [64, 128], "num_layers": [2, 4, 6]},
    ),
    "Energy Guided FA": SearchSpace(
        "Energy Guided FA",
        {
            "lr": (1e-4, 1e-2, "log"),
            "energy_scale": (0.1, 1.0, "linear"),
            "hidden_dim": [64, 128],
        },
    ),
    "Predictive Coding Hybrid": SearchSpace(
        "Predictive Coding Hybrid",
        {"lr": (1e-4, 1e-2, "log"), "steps": (10, 30, "int"), "hidden_dim": [64, 128]},
    ),
    "Sparse Equilibrium": SearchSpace(
        "Sparse Equilibrium",
        {
            "lr": (1e-4, 1e-2, "log"),
            "beta": (0.05, 0.3, "linear"),
            "sparsity": (0.1, 0.9, "linear"),
            "hidden_dim": [128, 256],
        },
    ),
    "Momentum Equilibrium": SearchSpace(
        "Momentum Equilibrium",
        {
            "lr": (1e-4, 1e-2, "log"),
            "momentum": (0.5, 0.95, "linear"),
            "steps": (10, 30, "int"),
        },
    ),
    "Stochastic FA": SearchSpace(
        "Stochastic FA",
        {
            "lr": (1e-4, 1e-2, "log"),
            "noise_scale": (0.01, 0.2, "log"),
            "hidden_dim": [64, 128],
        },
    ),
    "Energy Minimizing FA": SearchSpace(
        "Energy Minimizing FA", {"lr": (1e-4, 1e-2, "log"), "hidden_dim": [64, 128]}
    ),
    # Transformers
    "eqprop_transformer": SearchSpace(
        "eqprop_transformer",
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
    "equitile": SearchSpace(
        "equitile",
        {
            "lr": (1e-4, 1e-1, "log"),
            "inference_steps": (5, 30, "int"),
            "neurons_per_tile": [32, 64, 128],
            "tiles_per_layer": [4, 8, 16],
            "num_layers": [3, 5, 8],
            "sparsity_threshold": (0.01, 0.2, "linear"),
        },
    ),
    "EquiTile EP": SearchSpace(
        "EquiTile EP",
        {
            "lr": (1e-4, 1e-1, "log"),
            "beta": (0.05, 0.5, "linear"),
            "inference_steps": (10, 50, "int"),
            "neurons_per_tile": [32, 64, 128],
            "tiles_per_layer": [4, 8, 16],
            "num_layers": [3, 5, 8],
        },
    ),
    "LM EquiTile": SearchSpace(
        "LM EquiTile",
        {
            "lr": (1e-5, 1e-3, "log"),
            "neurons_per_tile": [64, 128],
            "tiles_per_layer": [4, 8],
            "num_layers": [4, 6],
            "embed_dim": [128, 256],
            "num_heads": [2, 4],
        },
    ),
    "RL EquiTile": SearchSpace(
        "RL EquiTile",
        {
            "lr": (1e-4, 1e-2, "log"),
            "neurons_per_tile": [32, 64],
            "tiles_per_layer": [2, 4, 8],
            "num_layers": [2, 3],
            "entropy_coef": (0.001, 0.05, "log"),
            "value_coef": (0.1, 1.0, "linear"),
        },
    ),
    "Conv EquiTile": SearchSpace(
        "Conv EquiTile",
        {
            "lr": (1e-4, 1e-2, "log"),
            "neurons_per_tile": [32, 64, 128],
            "tiles_per_layer": [2, 4, 8],
            "num_fc_layers": [1, 2, 3],
            "dropout": (0.0, 0.5, "linear"),
        },
    ),
}


def get_search_space(model_name: str) -> SearchSpace:
    """Get the search space for a model."""
    # 1. Try hardcoded spaces first (for customized ranges)
    if model_name in SEARCH_SPACES:
        return SEARCH_SPACES[model_name]

    # 2. Try to generate from registry
    # Check if exact name in registry
    try:
        spec = get_model_spec(model_name)
    except ValueError:
        spec = None

    if spec:
        params = {
            "lr": (1e-5, 1e-2, "log"),
            "hidden_dim": [64, 128, 256],
            "num_layers": [2, 4, 6],
        }

        return SearchSpace(model_name, params)

    # 3. Heuristic fallback: assume EqProp-ish defaults for unregistered models
    if "EqProp" in model_name:
        params = {
            "lr": (1e-5, 1e-2, "log"),
            "beta": (0.05, 0.5, "linear"),
            "steps": (5, 20, "int"),
            "hidden_dim": [64, 128],
        }
        return SearchSpace(model_name, params)

    if "Backprop" in model_name:
        return SEARCH_SPACES["backprop_mlp"]

    raise ValueError(f"No search space defined for model: {model_name}")


# Continuous, log-sampled search spaces per learning rule (plan §4A, §10).
# These replace the coarse discrete grids with true Bayesian ranges so that
# (a) TPE explores the posterior rather than a handful of points, and (b) each
# rule is compared at its own optimum — including rule-specific equilibrium
# hyperparameters (damping, step size, max iterations, convergence threshold).
RULE_SPACES: dict[str, dict[str, NumberRange | DiscreteChoice]] = {
    "backprop": {
        "lr": (1e-5, 1e-1, "log"),
        "weight_decay": (1e-6, 1e-2, "log"),
        "hidden_dim": (32, 1024, "log"),
        "num_layers": (1, 6, "int"),
    },
    "eqprop": {
        "lr": (1e-5, 1e-1, "log"),
        "weight_decay": (1e-6, 1e-2, "log"),
        "hidden_dim": (32, 1024, "log"),
        "num_layers": (1, 6, "int"),
        "beta": (0.01, 3.0, "log"),
        "max_steps": (5, 100, "int"),
        "damping": (0.0, 0.9, "linear"),
        "tol": (1e-6, 1e-2, "log"),
        "convergence_threshold": (1e-4, 1e-2, "log"),
        "convergence_start": (2, 10, "int"),
    },
    "neural_cube": {
        # Honest space (P0a): every knob is real — accepted by ``NeuralCube.__init__``
        # or routed by the training loop. ``hidden_dim``/``damping``/``tol`` were
        # silently dropped by ``build_model_kwargs`` (phantom drift, §0.1); re-add
        # each dimension in the same change that implements it on the model.
        "lr": (1e-5, 1e-1, "log"),
        "weight_decay": (1e-6, 1e-2, "log"),
        "cube_size": (3, 10, "int"),
        "max_steps": (5, 100, "int"),
    },
    "pepita": {
        "lr": (1e-4, 1e-1, "log"),
        "hidden_dim": (32, 512, "log"),
        "num_layers": (1, 4, "int"),
    },
    "forward_forward": {
        "lr": (1e-3, 1e-1, "log"),
        "hidden_dim": (32, 512, "log"),
        "num_layers": (1, 4, "int"),
        "threshold": (0.5, 5.0, "linear"),
        "layer_lr": (1e-3, 1e-1, "log"),
        "classifier_lr": (1e-3, 1e-1, "log"),
    },
    "feedback_alignment": {
        "lr": (1e-4, 1e-1, "log"),
        "hidden_dim": (32, 512, "log"),
        "num_layers": (1, 4, "int"),
        "alpha": (0.1, 1.0, "linear"),
        "feedback_mode": ["random", "symmetric", "transpose"],
        "use_spectral_norm": [True, False],
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


# ---------------------------------------------------------------------------
# P0a — RULE_SPACES ↔ constructor integrity gate (plan §P0a)
# ---------------------------------------------------------------------------

# Rule keys whose registered model name differs from the rule key; all others
# share the key with their registered model (``eqprop`` → ``StandardEqProp``, …).
_RULE_TO_MODEL: dict[str, str] = {
    "backprop": "backprop_mlp",
}


def _model_name_for_rule(rule: str) -> str:
    """Resolve the registered model name that ``build_model_kwargs`` constructs."""
    return _RULE_TO_MODEL.get(rule, rule)


# Keys the training/optimization pipeline consumes from a sampled config outside
# the model constructor. These are *not* phantoms: ``lr``/``weight_decay`` etc.
# are consumed by the optimizer even when the model's ``__init__`` never sees
# them. Architecture & equilibrium dimensions (``hidden_dim``, ``damping``,
# ``tol``, ``convergence_*``) are deliberately NOT listed — those are exactly the
# knobs P0a is meant to catch when silently dropped.
_TRAINING_HYPERPARAMS: frozenset[str] = frozenset({
    "lr",
    "weight_decay",
    "dropout",
    "momentum",
    "batch_size",
    "num_epochs",
    "optimizer",
    "scheduler",
    "gradient_clip",
    "warmup_epochs",
    "reg_lambda",
    "weight_init",
    "betas",
    "lr_scheduler",
})


@dataclass(frozen=True, slots=True)
class ConstructorSurface:
    """Machine-readable constructor-surface record for one rule (P0a).

    Captures, as of the current commit: which advertised search-space keys are
    real constructor parameters, which are absorbed via ``**kwargs``, which are
    training-loop hyperparameters, and which are *phantoms* (silently dropped by
    ``build_model_kwargs``). A non-empty :attr:`phantoms` fails the gate.
    """

    rule: str
    model: str
    signature: str
    space_hash: str
    accepted: tuple[str, ...]
    absorbs_kwargs: bool
    sinks: tuple[tuple[str, str], ...]
    phantoms: frozenset[str]

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible dict for KB storage."""
        return {
            "rule": self.rule,
            "model": self.model,
            "signature": self.signature,
            "space_hash": self.space_hash,
            "accepted": list(self.accepted),
            "absorbs_kwargs": self.absorbs_kwargs,
            "sinks": [list(kv) for kv in self.sinks],
            "phantoms": sorted(self.phantoms),
        }


def _constructor_surface(model_cls: object) -> tuple[frozenset[str], bool, str]:
    """Inspect ``model_cls.__init__`` → (accepted params, has **kwargs, signature)."""
    try:
        sig = inspect.signature(model_cls.__init__)
    except TypeError, ValueError:  # un-inspectable C callable etc.; treat as opaque
        return frozenset(), False, repr(model_cls)
    accepted: set[str] = set()
    for name, param in sig.parameters.items():
        if name in {"self", "args", "kwargs"}:
            continue
        accepted.add(name)
    has_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    return frozenset(accepted), has_kwargs, str(sig)


def _space_hash(space: dict[str, object]) -> str:
    """SHA-256 of the canonical (sorted) space definition — the honest surface pin."""
    canonical = repr(sorted(space.items(), key=lambda kv: kv[0]))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def surface_for_rule(rule: str) -> ConstructorSurface:
    """Compute the constructor-surface record for one rule without raising.

    Args:
        rule: Rule key into ``RULE_SPACES`` (e.g. ``"neural_cube"``).

    Returns:
        The :class:`ConstructorSurface` for the rule (phantoms may be non-empty).

    Raises:
        ValueError: If the rule has no defined space or no registered model.
    """
    space = get_rule_space(rule)
    model_name = _model_name_for_rule(rule)
    model_cls = Registry.get(ComponentCategory.MODEL, model_name)
    accepted, has_kwargs, signature = _constructor_surface(model_cls)

    sinks: list[tuple[str, str]] = []
    phantoms: set[str] = set()
    for key, _spec in space.items():
        if key in accepted:
            sink = "constructor"
        elif has_kwargs:
            sink = "kwargs"
        elif key in _TRAINING_HYPERPARAMS:
            sink = "training"
        else:
            sink = "phantom"
            phantoms.add(key)
        sinks.append((key, sink))
    sinks.sort()

    return ConstructorSurface(
        rule=rule,
        model=model_name,
        signature=signature,
        space_hash=_space_hash(space),
        accepted=tuple(sorted(accepted)),
        absorbs_kwargs=has_kwargs,
        sinks=tuple(sinks),
        phantoms=frozenset(phantoms),
    )


def validate_rule_space(rule: str) -> ConstructorSurface:
    """Assert one rule's space matches its model constructor (P0a gate).

    Raises:
        SpaceSignatureMismatchError: If any advertised key is a phantom.
        ValueError: If the rule has no defined space or no registered model.
    """
    surface = surface_for_rule(rule)
    if surface.phantoms:
        raise SpaceSignatureMismatchError(rule, surface.phantoms)
    return surface


def validate_all_rule_spaces() -> dict[str, ConstructorSurface]:
    """Validate every ``RULE_SPACES`` entry against its constructor.

    Returns:
        Mapping of rule → :class:`ConstructorSurface` (all phantoms empty).

    Raises:
        SpaceSignatureMismatchError: On the first rule with phantom knobs.
    """
    surfaces: dict[str, ConstructorSurface] = {}
    for rule in sorted(RULE_SPACES):
        surfaces[rule] = validate_rule_space(rule)
    return surfaces


def emit_rule_space_surfaces(kb: object) -> dict[str, str]:
    """Write each rule's constructor surface into a KnowledgeBase (P0a emitter).

    Idempotent per-rule: uses a deterministic entry id (``SURFACE-{rule}``) so
    re-runs replace rather than duplicate. The records become queryable
    audit-trail artifacts for the P2 flywheel and for pricing historical numbers.

    Args:
        kb: A :class:`~bioplausible.knowledge.kb.KnowledgeBase` instance.

    Returns:
        Mapping of rule → the persisted KnowledgeBase entry id.
    """
    from bioplausible.knowledge.kb import KnowledgeEntry

    written: dict[str, str] = {}
    for rule in sorted(RULE_SPACES):
        surface = surface_for_rule(rule)
        entry = KnowledgeEntry(
            id=f"SURFACE-{rule}",
            topic="rule_space_surface",
            model_family=rule,
            finding="honest" if not surface.phantoms else "phantom",
            details=f"{surface.model} :: {surface.signature}",
            confidence=1.0 if not surface.phantoms else 0.0,
            tags=["rule-space", "surface", rule],
            source="validator",
            hyperparameters=dict(RULE_SPACES[rule]),
            extra={"surface": surface.to_dict()},
        )
        written[rule] = kb.add_entry(entry)
    return written
