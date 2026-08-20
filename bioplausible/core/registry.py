"""Unified Registry System for Bioplausible.

Decorator-based registry for all components (models, propagators,
optimizers, sparsity) enabling AutoScientist to query and compose
intelligently."""

import pathlib
from collections.abc import Callable
from dataclasses import MISSING, dataclass, field, fields
from enum import Enum, StrEnum
from typing import Literal, Protocol, TypeVar, cast

from bioplausible.core.exceptions import IncompatibilityError as _IncompatibilityError
from bioplausible.core.logging import get_logger

logger = get_logger()

Component = TypeVar("Component")  # registered class or factory callable

# Credit-assignment strategy a component uses. Closed set so the
# AutoScientist can rely on membership when composing capability queries.
CreditAssignmentType = Literal[
    "gradient",
    "equilibrium",
    "hebbian",
    "target",
    "forward-only",
    "spiking",
    "backpropagation",
    "local",
]

IncompatibilityError = _IncompatibilityError


class ComponentCategory(str, Enum):
    """Categories of components in the registry."""

    MODEL = "model"
    PROPAGATOR = "propagator"
    OPTIMIZER = "optimizer"
    # Update/constraint strategies that transform gradients or project weights
    # AFTER an optimizer steps (e.g. Muon Newton-Schulz, spectral norm clipping).
    # Distinct from OPTIMIZER because they are not torch.optim estimators: they
    # expose ``transform_gradient`` (strategies) or ``step`` (constraints) with
    # no parameter/costate ownership.
    UPDATE_STRATEGY = "update_strategy"
    CONSTRAINT = "constraint"
    # Training-side controllers (e.g. DynamicEquiTile topology controller) that
    # are not nn.Modules and never run a forward pass.
    CONTROLLER = "controller"
    SPARSITY = "sparsity"
    METRIC = "metric"
    TASK = "task"
    TRACK = "track"
    # Kernel acceleration backends for bio-plausible algorithms
    KERNEL_BACKEND = "kernel_backend"


class Domain(str, Enum):
    """Supported domains."""

    VISION = "vision"
    LM = "lm"
    RL = "rl"
    GRAPH = "graph"
    TIMESERIES = "timeseries"
    TABULAR = "tabular"
    SCIENTIFIC = "scientific"
    CONTINUAL = "continual"
    MULTITASK = "multitask"


class LocalityLevel(str, Enum):
    """Credit assignment locality level."""

    GLOBAL = "global"  # Full backprop
    LAYERWISE = "layerwise"  # Layer-local (Forward-Forward, Target Prop)
    LOCAL = "local"  # Neuron/synapse local (Hebbian, STDP, EquiTile)
    EQUILIBRIUM = "equilibrium"  # Energy-based (EqProp, CHL)
    FORWARD_ONLY = "forward-only"  # No backward pass (PEPITA, FF)


class ComputeProfile(str, Enum):
    """Compute profile for hardware affinity."""

    GPU = "gpu"
    CPU = "cpu"
    NEUROMORPHIC = "neuromorphic"
    ANALOG = "analog"
    OPTICAL = "optical"
    MEMRISTOR = "memristor"
    DISTRIBUTED = "distributed"


class Capability(StrEnum):
    """Capabilities a model can provide or a propagator can require."""

    TRANSITION_GRAPH = "transition_graph"  # Model implements transition_modules()
    STANDARD_AUTOGRAD = "standard_autograd"  # Standard forward + loss.backward()
    CONTRASTIVE = "contrastive"  # Implements get_hebbian_pairs()


@dataclass(frozen=True, slots=True)
class ComponentMetadata:
    """Metadata for registered components enabling intelligent composition."""

    name: str
    category: ComponentCategory
    domains: list[Domain] = field(default_factory=lambda: [Domain.VISION])
    locality_level: LocalityLevel = LocalityLevel.GLOBAL
    compute_profile: ComputeProfile = ComputeProfile.GPU
    bio_plausibility_score: float = 0.5  # 0.0 = backprop, 1.0 = fully bio-plausible
    credit_assignment_type: CreditAssignmentType = "gradient"
    requires_backward: bool = True
    memory_complexity: str = "O(N)"  # O(1) for MEP, O(N) standard
    min_params: int | None = None
    max_params: int | None = None
    typical_lr_range: tuple[float, float] = (1e-5, 1e-1)
    typical_batch_size_range: tuple[float, float] = (16, 512)
    supports_mixed_precision: bool = True
    supports_gradient_accumulation: bool = True
    supports_distributed: bool = False
    tags: list[str] = field(default_factory=list)
    citation: str | None = None
    description: str = ""
    version: str = "1.0.0"
    # Algorithm family tag (per REFACTOR2 §3.2): "eqprop", "fa", "hebbian",
    # "forward_only", "target_prop", "spiking", "predictive_coding", "backprop",
    # "mep", "equitile", etc. Directory layout mirrors this but `family` is the
    # canonical searchable attribute for grouping in the README/Registry queries.
    family: str = ""
    # Required and provided capabilities (per REFACTOR3 §4)
    requires: list[str] = field(default_factory=list)
    provides: list[str] = field(default_factory=list)
    extra: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _QueryFilter:
    """Immutable capability query predicate spec.

    Held frozen to make intent explicit and to allow future caching
    over the predicate. Each field is a filter; ``None`` means
    ``no constraint on this axis``.
    """

    domain: Domain | None = None
    locality: LocalityLevel | None = None
    compute: ComputeProfile | None = None
    requires_backward: bool | None = None
    min_bio_score: float | None = None
    max_bio_score: float | None = None
    tags: list[str] | None = None
    credit_type: CreditAssignmentType | None = None
    family: str | None = None
    _predicates: tuple[_Predicate, ...] = field(init=False, repr=False, default=())

    def __post_init__(self) -> None:
        """Build the predicate dispatch table once at construction."""
        predicates: list[_Predicate] = []
        if self.domain is not None:
            predicates.append(_DomainIn(self.domain))
        if self.locality is not None:
            predicates.append(_LocalityIs(self.locality))
        if self.compute is not None:
            predicates.append(_ComputeIs(self.compute))
        if self.requires_backward is not None:
            predicates.append(_RequiresBackwardIs(self.requires_backward))
        if self.min_bio_score is not None:
            predicates.append(_MinBioScore(self.min_bio_score))
        if self.max_bio_score is not None:
            predicates.append(_MaxBioScore(self.max_bio_score))
        if self.credit_type is not None:
            predicates.append(_CreditTypeIs(self.credit_type))
        if self.tags is not None:
            predicates.append(_TagsAll(frozenset(self.tags)))
        if self.family is not None:
            predicates.append(_FamilyIs(self.family))
        object.__setattr__(self, "_predicates", tuple(predicates))

    def matches(self, meta: ComponentMetadata) -> bool:
        """Return True iff ``meta`` satisfies every constraint in this filter."""
        return all(predicate(meta) for predicate in self._predicates)


class _Predicate(Protocol):
    """Single-axis capability predicate."""

    def __call__(self, meta: ComponentMetadata) -> bool: ...


@dataclass(frozen=True, slots=True)
class _DomainIn:
    """True iff ``meta`` declares the target domain."""

    domain: Domain

    def __call__(self, meta: ComponentMetadata) -> bool:
        return self.domain in meta.domains


@dataclass(frozen=True, slots=True)
class _LocalityIs:
    """True iff ``meta`` locality matches exactly."""

    locality: LocalityLevel

    def __call__(self, meta: ComponentMetadata) -> bool:
        return meta.locality_level == self.locality


@dataclass(frozen=True, slots=True)
class _ComputeIs:
    """True iff ``meta`` compute profile matches exactly."""

    compute: ComputeProfile

    def __call__(self, meta: ComponentMetadata) -> bool:
        return meta.compute_profile == self.compute


@dataclass(frozen=True, slots=True)
class _RequiresBackwardIs:
    """True iff ``meta`` backward requirement matches exactly."""

    requires_backward: bool

    def __call__(self, meta: ComponentMetadata) -> bool:
        return meta.requires_backward == self.requires_backward


@dataclass(frozen=True, slots=True)
class _MinBioScore:
    """True iff ``meta`` bio-plausibility is at least the bound."""

    min_bio_score: float

    def __call__(self, meta: ComponentMetadata) -> bool:
        return meta.bio_plausibility_score >= self.min_bio_score


@dataclass(frozen=True, slots=True)
class _MaxBioScore:
    """True iff ``meta`` bio-plausibility is at most the bound."""

    max_bio_score: float

    def __call__(self, meta: ComponentMetadata) -> bool:
        return meta.bio_plausibility_score <= self.max_bio_score


@dataclass(frozen=True, slots=True)
class _CreditTypeIs:
    """True iff ``meta`` credit-assignment type matches exactly."""

    credit_type: CreditAssignmentType

    def __call__(self, meta: ComponentMetadata) -> bool:
        return meta.credit_assignment_type == self.credit_type


@dataclass(frozen=True, slots=True)
class _TagsAll:
    """True iff ``meta`` carries every required tag."""

    tags: frozenset[str]

    def __call__(self, meta: ComponentMetadata) -> bool:
        return all(tag in meta.tags for tag in self.tags)


@dataclass(frozen=True, slots=True)
class _FamilyIs:
    """True iff ``meta`` family matches exactly."""

    family: str

    def __call__(self, meta: ComponentMetadata) -> bool:
        return meta.family == self.family


class Registry:
    """Central registry for all components.

    Supports decorator-based registration, querying by capability
    metadata, and constraint satisfaction for AutoScientist composition.

    Some learning rules (FF, PEPITA, TargetProp, PCN) require model-level
    control and are registered as models, not propagators. When queried as
    propagators, ``get()`` resolves them to the model-side implementation
    via the :attr:`_ALIASES` compatibility map (a lookup, not an error).
    """

    _components: dict[
        str, dict[str, dict[str, object]]
    ] = {}  # category -> {name: {cls, metadata}}

    # Compatibility map: names that alias to a registered component in a
    # *different* category. Currently propagator names that map to
    # model-side implementations (the learning rule lives in the model's
    # ``train_step``, not in a separate propagator object).
    _ALIASES: dict[str, tuple[ComponentCategory, str]] = {
        "ff": (ComponentCategory.MODEL, "forward_forward"),
        "target_prop": (ComponentCategory.MODEL, "diff_target_prop"),
        "difference_target_prop": (ComponentCategory.MODEL, "diff_target_prop"),
        "predictive_coding": (
            ComponentCategory.MODEL,
            "predictive_coding_hybrid",
        ),
    }

    @staticmethod
    def _resolve_category(category: ComponentCategory | str) -> ComponentCategory:
        """Resolve category from string or enum."""
        if isinstance(category, str):
            return ComponentCategory(category)
        return category

    @classmethod
    def register(
        cls, category: ComponentCategory, name: str | None = None, **metadata_kwargs
    ) -> Callable[[Component], Component]:
        """Decorator factory to register a model/propagator/optimizer/etc.

        Accepts either a class or any callable (e.g. a preset factory
        function like ``smep``). If ``name`` is None the registered
        name defaults to ``component.__name__``.
        """
        if category not in cls._components:
            cls._components[category] = {}

        def decorator(component: Component, _name: str = name) -> Component:
            if _name is None:
                _name = getattr(component, "__name__", repr(component))
            if _name in cls._components[category]:
                logger.warning("Overwriting component %s/%s", category.value, _name)
            metadata = ComponentMetadata(
                name=_name, category=category, **metadata_kwargs
            )
            cls._infer_metadata(component, metadata)
            cls._components[category][_name] = {
                "class": component,
                "metadata": metadata,
            }
            # Attach metadata to the component for introspection. We only
            # set attributes on classes that accept them (factory functions
            # without ``__dict__`` would raise); use try/except defensively.
            try:
                component._registry_metadata = metadata  # type: ignore[attr-defined]
                component._registry_name = name  # type: ignore[attr-defined]
                component._registry_category = category  # type: ignore[attr-defined]
            except AttributeError, TypeError:
                pass
            logger.info("Registered %s: %s", category.value, name)
            return component

        return decorator

    @classmethod
    def _infer_metadata(cls, component: object, metadata: ComponentMetadata) -> None:
        """Infer metadata from component attributes if not explicitly provided.

        Uses ``object.__setattr__`` to bypass the frozen dataclass restriction
        — this is an internal initialisation helper, not user-facing mutation.
        """
        overrides = getattr(component, "_registry_metadata_overrides", {})
        for fd in fields(ComponentMetadata):
            if fd.name in overrides:
                continue
            if not hasattr(component, fd.name):
                continue
            current = getattr(metadata, fd.name)
            default = fd.default
            if default is not MISSING and current == default:
                object.__setattr__(metadata, fd.name, getattr(component, fd.name))
            elif default is MISSING and fd.default_factory is not MISSING:
                # Fields with default_factory (e.g. provides=[], requires=[])
                # start empty. Only infer if the component attribute differs
                # from the factory-produced default.
                factory_default = fd.default_factory()
                if current == factory_default:
                    object.__setattr__(metadata, fd.name, getattr(component, fd.name))

    @classmethod
    def get(cls, category: ComponentCategory | str, name: str) -> object:
        """Get a registered component (class or factory callable) by name.

        If ``name`` is a known alias (see :attr:`_ALIASES`), the lookup
        transparently follows it to the canonical component.
        """
        cat = cls._resolve_category(category)
        if cat not in cls._components:
            raise ValueError(f"Unknown category: {cat}")
        if name not in cls._components[cat]:
            # Follow compatibility aliases (e.g. propagator "ff" → model "forward_forward").
            alias = cls._ALIASES.get(name)
            if alias is not None and alias[0] == cat:
                target_cat, target_name = alias
                logger.info(
                    "Component %r not registered directly under %s; "
                    "resolving alias to %s/%s.",
                    name,
                    cat.value,
                    target_cat.value,
                    target_name,
                )
                return cls.get(target_cat, target_name)
            if alias is not None and cat == ComponentCategory.PROPAGATOR:
                target_cat, target_name = alias
                logger.info(
                    "Propagator %r is a model-side learner (registered as "
                    "%s/%r); the model's train_step handles training — "
                    "no propagator object is created.",
                    name,
                    target_cat.value,
                    target_name,
                )
                return cls.get(target_cat, target_name)
            available = list(cls._components[cat].keys())
            raise ValueError(f"Unknown {cat.value}: {name}. Available: {available}")
        return cls._components[cat][name]["class"]

    @classmethod
    def aliases(cls) -> dict[str, tuple[ComponentCategory, str]]:
        """Return the compatibility alias map (read-only view).

        Maps alias names to ``(canonical_category, canonical_name)`` tuples.
        Used by discovery code (e.g. AutoScientist) to enumerate every
        addressable learning rule regardless of whether it lives in the
        MODEL or PROPAGATOR namespace.
        """
        return dict(cls._ALIASES)

    @classmethod
    def resolve_alias(
        cls, category: ComponentCategory | str, name: str
    ) -> tuple[ComponentCategory, str]:
        """Resolve ``name`` in ``category`` through any alias chain.

        Returns the canonical ``(category, name)`` without instantiating.
        If ``name`` is not an alias, it is returned unchanged.
        """
        cat = cls._resolve_category(category)
        seen: set[str] = set()
        current: tuple[ComponentCategory, str] = (cat, name)
        while current[1] in cls._ALIASES and (alias := cls._ALIASES.get(current[1])):
            if current[1] in seen:
                logger.warning("Alias cycle detected at %r", current[1])
                break
            seen.add(current[1])
            current = alias
        return current

    @classmethod
    def get_metadata(
        cls, category: ComponentCategory | str, name: str
    ) -> ComponentMetadata:
        """Get metadata for a registered component.

        Resolves alias names transparently by delegating to
        :meth:`resolve_alias`.
        """
        cat = cls._resolve_category(category)
        if cat not in cls._components:
            raise ValueError(f"Unknown category: {cat}")
        # Resolve aliases (e.g. propagator "ff" → model "forward_forward").
        resolved_cat, resolved_name = cls.resolve_alias(cat, name)
        cat, name = resolved_cat, resolved_name
        if name not in cls._components[cat]:
            available = list(cls._components[cat].keys())
            raise ValueError(f"Unknown {cat.value}: {name}. Available: {available}")
        return cast("ComponentMetadata", cls._components[cat][name]["metadata"])

    @classmethod
    def list(
        cls, category: ComponentCategory | str | None = None
    ) -> dict[str, list[str]]:
        """List all registered components, optionally filtered by category."""
        if category is not None:
            cat = cls._resolve_category(category)
            if cat not in cls._components:
                return {cat.value: []}
            return {cat.value: list(cls._components[cat].keys())}
        return {cat.value: list(comps.keys()) for cat, comps in cls._components.items()}

    @classmethod
    def query(
        cls,
        category: ComponentCategory | str | None = None,
        domain: Domain | None = None,
        locality: LocalityLevel | None = None,
        compute: ComputeProfile | None = None,
        requires_backward: bool | None = None,
        min_bio_score: float | None = None,
        max_bio_score: float | None = None,
        tags: list[str] | None = None,
        credit_type: CreditAssignmentType | None = None,
        family: str | None = None,
    ) -> list[dict[str, object]]:
        """Query registry with capability constraints.

        Returns list of ``{name, category, class, metadata}`` dict
        entries matching ALL criteria. Designed for AutoScientist
        composition.
        """
        flt = _QueryFilter(
            domain=domain,
            locality=locality,
            compute=compute,
            requires_backward=requires_backward,
            min_bio_score=min_bio_score,
            max_bio_score=max_bio_score,
            tags=tags,
            credit_type=credit_type,
            family=family,
        )
        cats = [category] if category else list(cls._components.keys())
        categories = [cls._resolve_category(c) for c in cats]

        results: list[dict[str, object]] = []
        for cat in categories:
            if cat not in cls._components:
                continue
            for name, info in cls._components[cat].items():
                meta = cast("ComponentMetadata", info["metadata"])
                if flt.matches(meta):
                    results.append({
                        "name": name,
                        "category": cat,
                        "class": info["class"],
                        "metadata": meta,
                    })
        return results

    @classmethod
    def get_compatible(
        cls,
        model_name: str,
        model_category: ComponentCategory = ComponentCategory.MODEL,
    ) -> dict[str, list[dict[str, object]]]:
        """Get components compatible with a given model."""
        model_meta = cls.get_metadata(model_category, model_name)
        primary_domain = model_meta.domains[0] if model_meta.domains else None

        compat: dict[str, list[dict[str, object]]] = {}
        for cat in ComponentCategory:
            if cat == model_category:
                continue
            compat[cat.value] = cls.query(category=cat, domain=primary_domain)
        return compat

    @classmethod
    def check_compatibility(cls, propagator_name: str, model_name: str) -> bool:
        """Return True if the model provides all capabilities the propagator requires.

        Relies on declarative ``requires`` and ``provides`` metadata only.
        """
        prop_meta = cls.get_metadata(ComponentCategory.PROPAGATOR, propagator_name)
        model_meta = cls.get_metadata(ComponentCategory.MODEL, model_name)
        required = set(prop_meta.requires)
        provided = set(model_meta.provides)
        return required.issubset(provided)

    @classmethod
    def to_system(cls, model_name: str, **model_kwargs) -> "System":
        """Project a registered model into the 5-D ontology as a System.

        Creates a ModelAdapter for the registered model and returns its
        5-D System projection. This enables the AutoScientist to query
        models along the five orthogonal axes instead of a flat list.

        Args:
            model_name: Name of the registered model (resolves aliases).
            **model_kwargs: Arguments passed to the model constructor.
                If not provided, uses minimal defaults (input_dim=10, hidden_dim=20, output_dim=3).

        Returns:
            A System instance composed of the inferred 5 layers.
        """
        from bioplausible.core.ontology import ModelAdapter

        # Resolve the canonical model name (handles aliases like "ff" -> "forward_forward")
        cat, name = cls.resolve_alias(ComponentCategory.MODEL, model_name)

        # Get the model class/factory and metadata
        component = cls.get(cat, name)
        metadata = cls.get_metadata(cat, name)

        # Instantiate the model with defaults if no kwargs provided
        if not model_kwargs:
            model_kwargs = {"input_dim": 10, "hidden_dim": 20, "output_dim": 3}

        if isinstance(component, type):
            model = component(**model_kwargs)
        else:
            model = component(**model_kwargs)

        # Adapt to 5-D ontology
        adapter = ModelAdapter(model, metadata)
        return adapter.to_system()

    @classmethod
    def to_system_by_category(cls, category: ComponentCategory | str, name: str) -> "System":
        """Project any registered component into the 5-D ontology.

        For models, this is the same as ``to_system``. For other categories
        (propagators, optimizers, etc.), it creates a minimal System with
        the component mapped to its corresponding layer.
        """
        from bioplausible.core.ontology import (
            ModelAdapter, System, compose_system,
            DigitalSubstrate, FeedforwardGeometry, InstantaneousDynamics,
            ThermodynamicContrast, EuclideanUpdate,
            GeometryConfig, StateDynamicsConfig,
            CreditAssignmentConfig, ParameterUpdateConfig,
        )

        cat = cls._resolve_category(category)
        component = cls.get(cat, name)
        metadata = cls.get_metadata(cat, name)

        if isinstance(component, type):
            instance = component()
        else:
            instance = component()

        if cat == ComponentCategory.MODEL:
            adapter = ModelAdapter(instance, metadata)
            return adapter.to_system()

        # For non-model components, create a minimal system with defaults
        # and the component mapped to its layer
        substrate = DigitalSubstrate()
        geometry = FeedforwardGeometry(GeometryConfig(input_dim=10, output_dim=3, hidden_dims=(20,)))
        dynamics = InstantaneousDynamics()
        credit = ThermodynamicContrast()
        update = EuclideanUpdate()

        # This is a placeholder - in practice, propagators map to CreditAssignment,
        # optimizers map to ParameterUpdate, etc.
        return compose_system(substrate, geometry, dynamics, credit, update)

    @classmethod
    def clear(cls) -> None:
        """Clear the registry (mainly for testing)."""
        cls._components.clear()

    @classmethod
    def export_yaml(cls, path: str) -> None:
        """Export all registered component metadata to a YAML file."""
        import yaml  # local import: keep module import cheap (AGENTS.md)

        export_data: dict[str, dict[str, dict[str, object]]] = {}
        for category, comps in cls._components.items():
            cat_name = category.value
            export_data[cat_name] = {}
            for name, info in comps.items():
                meta = cast("ComponentMetadata", info["metadata"])
                export_data[cat_name][name] = {
                    "name": meta.name,
                    "category": meta.category.value,
                    "domains": [d.value for d in meta.domains],
                    "locality_level": meta.locality_level.value,
                    "compute_profile": meta.compute_profile.value,
                    "bio_plausibility_score": meta.bio_plausibility_score,
                    "credit_assignment_type": meta.credit_assignment_type,
                    "requires_backward": meta.requires_backward,
                    "memory_complexity": meta.memory_complexity,
                    "tags": meta.tags,
                    "description": meta.description,
                    "citation": meta.citation,
                    "version": meta.version,
                    "family": meta.family,
                }

        with pathlib.Path(path).open("w", encoding="utf-8") as f:
            yaml.dump(export_data, f, default_flow_style=False, sort_keys=False)

        n_components = sum(len(v) for v in export_data.values())
        logger.info("Registry exported to %s: %d components", path, n_components)


# Convenience decorators
def register_model(name: str | None = None, **kwargs) -> Callable:
    """Register a model component."""
    return Registry.register(ComponentCategory.MODEL, name, **kwargs)


def register_propagator(name: str | None = None, **kwargs) -> Callable:
    """Register a propagator/learning-rule component."""
    return Registry.register(ComponentCategory.PROPAGATOR, name, **kwargs)


def register_optimizer(name: str | None = None, **kwargs) -> Callable:
    """Register an optimizer component."""
    return Registry.register(ComponentCategory.OPTIMIZER, name, **kwargs)


def register_update_strategy(name: str | None = None, **kwargs) -> Callable:
    """Register an update-strategy component (gradient transformation)."""
    return Registry.register(ComponentCategory.UPDATE_STRATEGY, name, **kwargs)


def register_constraint(name: str | None = None, **kwargs) -> Callable:
    """Register a constraint component (post-step weight projection)."""
    return Registry.register(ComponentCategory.CONSTRAINT, name, **kwargs)


def register_controller(name: str | None = None, **kwargs) -> Callable:
    """Register a training-side controller component (not an ``nn.Module``)."""
    return Registry.register(ComponentCategory.CONTROLLER, name, **kwargs)


def register_sparsity(name: str | None = None, **kwargs) -> Callable:
    """Register a sparsity component."""
    return Registry.register(ComponentCategory.SPARSITY, name, **kwargs)


def register_metric(name: str | None = None, **kwargs) -> Callable:
    """Register a metric component."""
    return Registry.register(ComponentCategory.METRIC, name, **kwargs)


def register_task(name: str | None = None, **kwargs) -> Callable:
    """Register a task component."""
    return Registry.register(ComponentCategory.TASK, name, **kwargs)


def list_models() -> list[str]:
    """Convenience: list all registered model names."""
    return list(Registry._components.get(ComponentCategory.MODEL, {}).keys())


__all__ = [
    "Capability",
    "ComponentCategory",
    "ComponentMetadata",
    "ComputeProfile",
    "CreditAssignmentType",
    "Domain",
    "IncompatibilityError",
    "LocalityLevel",
    "Registry",
    "list_models",
    "register_constraint",
    "register_controller",
    "register_metric",
    "register_model",
    "register_optimizer",
    "register_propagator",
    "register_sparsity",
    "register_task",
    "register_update_strategy",
]
