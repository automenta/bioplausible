"""
Search Space Definitions

Defines the hyperparameter search spaces for each model type in the registry.
"""

import hashlib
import inspect
from dataclasses import dataclass

import numpy as np

from computronium.core.exceptions import SpaceSignatureMismatchError
from computronium.core.model_spec import get_model_spec
from computronium.core.registry import ComponentCategory, Registry

# Type aliases

__all__ = [
    "RULE_SPACES",
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


# ---------------------------------------------------------------------------
# Model → rule resolution (single source of truth is RULE_SPACES).
# ---------------------------------------------------------------------------

# Registered model family → RULE_SPACES rule key. A model whose family appears
# here inherits the rule's continuous space verbatim, so sampling (evolution),
# P0a auditing, and Optuna constraint injection all see the same ranges.
_FAMILY_TO_RULE: dict[str, str] = {
    "backprop": "backprop",
    "baseline": "backprop",
    "backpropagation": "backprop",
    "eqprop": "eqprop",
    "fa": "feedback_alignment",
    "feedback_alignment": "feedback_alignment",
    "target_prop": "target_prop",
    "target-prop": "target_prop",
    "forward_only": "forward_forward",
    "forward-only": "forward_forward",
    "mep": "forward_forward",
}

# Families that own a real learning rule but no RULE_SPACES entry yet get a
# minimal, honest fallback rather than a hand-divergent curated grid.
_FALLBACK_SPACE: dict[str, NumberRange | DiscreteChoice] = {
    "learning_rate": (1e-5, 1e-1, "log"),
    "hidden_dim": (32, 512, "log"),
    "num_layers": (1, 6, "int"),
}


def _registered_families() -> dict[str, str]:
    """Map registered model name → family, for pool/curation queries."""
    out: dict[str, str] = {}
    for name in Registry.list(ComponentCategory.MODEL)[ComponentCategory.MODEL.value]:
        meta = Registry.get_metadata(ComponentCategory.MODEL, name)
        out[name] = meta.family or "experimental"
    return out


def get_model_spec_for_space(model_name: str) -> tuple[str, str] | None:
    """Resolve ``(family, rule)`` for a model name, or ``None`` if unregistered."""
    try:
        spec = get_model_spec(model_name)
    except ValueError:
        return None
    family = spec.family.lower()
    return family, _FAMILY_TO_RULE.get(family, "")


def get_available_models() -> list[str]:
    """Registered model names resolvable by :func:`get_search_space`.

    The pool for evolutionary "new architecture" discovery: every registered
    model, so a sampled config always carries a constructible name.
    """
    return sorted(_registered_families())


def get_search_space(model_name: str) -> SearchSpace:
    """Resolve the search space for a model from its registered family.

    When the model's family maps to a ``RULE_SPACES`` rule, the rule's space is
    used verbatim (the single canonical range set). Registered families without
    a rule fall back to :data:`_FALLBACK_SPACE`.

    Raises:
        ValueError: If ``model_name`` is neither a ``RULE_SPACES`` rule key nor
        a registered model.
    """
    # A model name that is itself a ``RULE_SPACES`` key uses that rule verbatim,
    # keeping sampling identical to the P0a constructor gate (which resolves
    # rule→model via the same key).
    if model_name in RULE_SPACES:
        return SearchSpace(model_name, RULE_SPACES[model_name])
    resolved = get_model_spec_for_space(model_name)
    if resolved is None:
        raise ValueError(f"No search space defined for model: {model_name}")
    _, rule = resolved
    params = RULE_SPACES[rule] if rule else _FALLBACK_SPACE
    return SearchSpace(model_name, params)


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


# ---------------------------------------------------------------------------
# P0a — RULE_SPACES ↔ constructor integrity gate (plan §P0a)
# ---------------------------------------------------------------------------

# Rule keys whose registered model name differs from the rule key; all others
# share the key with their registered model (``eqprop`` → ``StandardEqProp``, …).
_RULE_TO_MODEL: dict[str, str] = {
    "backprop": "backprop_mlp",
    "target_prop": "diff_target_prop",
}


def _model_name_for_rule(rule: str) -> str:
    """Resolve the registered model name that ``build_model_kwargs`` constructs."""
    return _RULE_TO_MODEL.get(rule, rule)


# Keys the training/optimization pipeline consumes from a sampled config outside
# the model constructor. These are *not* phantoms: ``learning_rate``/``weight_decay`` etc.
# are consumed by the optimizer even when the model's ``__init__`` never sees
# them. Architecture & equilibrium dimensions (``hidden_dim``, ``damping``,
# ``tol``, ``convergence_*``) are deliberately NOT listed — those are exactly the
# knobs P0a is meant to catch when silently dropped.
_TRAINING_HYPERPARAMS: frozenset[str] = frozenset({
    "learning_rate",
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
        kb: A :class:`~computronium.knowledge.kb.KnowledgeBase` instance.

    Returns:
        Mapping of rule → the persisted KnowledgeBase entry id.
    """
    from computronium.knowledge.kb import KnowledgeEntry

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
