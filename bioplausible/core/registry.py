"""Unified Registry System for Bioplausible.

Decorator-based registry for all components (models, propagators,
optimizers, sparsity) enabling AutoScientist to query and compose
intelligently."""

import builtins
import logging
import pathlib
from collections.abc import Callable
from dataclasses import MISSING, dataclass, field, fields
from enum import Enum, StrEnum
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

Component = TypeVar("Component")  # registered class or factory callable


class IncompatibilityError(TypeError):
    """Raised when a propagator requires capabilities the model does not provide.

    This is a hard error — no fallback, no silent discovery.
    """


class ComponentCategory(str, Enum):
    """Categories of components in the registry."""

    MODEL = "model"
    PROPAGATOR = "propagator"
    OPTIMIZER = "optimizer"
    SPARSITY = "sparsity"
    METRIC = "metric"


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
    credit_assignment_type: str = (
        "gradient"  # gradient, equilibrium, hebbian, target, forward-only, spiking
    )
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
    extra: dict[str, Any] = field(default_factory=dict)


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
    tags: builtins.list[str] | None = None
    credit_type: str | None = None
    family: str | None = None

    def matches(self, meta: ComponentMetadata) -> bool:
        """Return True iff ``meta`` satisfies every constraint in this filter."""
        return (
            (self.domain is None or self.domain in meta.domains)
            and (self.locality is None or meta.locality_level == self.locality)
            and (self.compute is None or meta.compute_profile == self.compute)
            and (
                self.requires_backward is None
                or meta.requires_backward == self.requires_backward
            )
            and (
                self.min_bio_score is None
                or meta.bio_plausibility_score >= self.min_bio_score
            )
            and (
                self.max_bio_score is None
                or meta.bio_plausibility_score <= self.max_bio_score
            )
            and (
                self.credit_type is None
                or meta.credit_assignment_type == self.credit_type
            )
            and (
                self.tags is None or builtins.all(tag in meta.tags for tag in self.tags)
            )
            and (self.family is None or meta.family == self.family)
        )


class Registry:
    """Central registry for all components.

    Supports decorator-based registration, querying by capability
    metadata, and constraint satisfaction for AutoScientist composition.

    Some learning rules (FF, PEPITA, TargetProp, PCN) require model-level
    control and are registered as models, not propagators. When queried as
    propagators, ``get()`` raises with a cross-reference to the model-side
    implementation.
    """

    _components: dict[
        str, dict[str, dict[str, Any]]
    ] = {}  # category -> {name: {cls, metadata}}

    # Cross-reference: propagator names that map to model-side implementations.
    # These are not registered directly as propagators because they require
    # model-level control of the forward/training loop.
    _PROPAGATOR_TO_MODEL: dict[str, tuple[str, str]] = {
        "ff": (
            "forward_forward",
            "bioplausible.zoo.models.forward_only.ForwardForwardNet",
        ),
        "pepita": ("pepita", "bioplausible.zoo.models.forward_only.PEPITA"),
        "target_prop": (
            "diff_target_prop",
            "bioplausible.zoo.models.target_prop.DifferenceTargetProp",
        ),
        "difference_target_prop": (
            "diff_target_prop",
            "bioplausible.zoo.models.target_prop.DifferenceTargetProp",
        ),
        "predictive_coding": (
            "predictive_coding_hybrid",
            "bioplausible.zoo.models.predictive_coding.PredictiveCodingHybrid",
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

        def decorator(component: Component) -> Component:
            nonlocal name
            if name is None:
                name = getattr(component, "__name__", repr(component))
            if name in cls._components[category]:
                logger.warning("Overwriting component %s/%s", category.value, name)
            metadata = ComponentMetadata(
                name=name, category=category, **metadata_kwargs
            )
            cls._infer_metadata(component, metadata)
            cls._components[category][name] = {
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
    def _infer_metadata(cls, component: Any, metadata: ComponentMetadata) -> None:
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
    def get(cls, category: ComponentCategory | str, name: str) -> Any:
        """Get a registered component (class or factory callable) by name."""
        cat = cls._resolve_category(category)
        if cat not in cls._components:
            raise ValueError(f"Unknown category: {cat}")
        if name not in cls._components[cat]:
            # Check cross-reference for propagator→model mappings.
            cross_ref = cls._PROPAGATOR_TO_MODEL.get(name)
            if cross_ref is not None and cat == ComponentCategory.PROPAGATOR:
                model_name, model_path = cross_ref
                raise ValueError(
                    f"Propagator {name!r} is not registered as a propagator "
                    f"because it requires model-level control of the "
                    f"forward/training loop.\n"
                    f"Use Registry.get(ComponentCategory.MODEL, {model_name!r}) instead.\n"
                    f"Model-side class: {model_path}"
                )
            available = list(cls._components[cat].keys())
            raise ValueError(f"Unknown {cat.value}: {name}. Available: {available}")
        return cls._components[cat][name]["class"]

    @classmethod
    def get_metadata(
        cls, category: ComponentCategory | str, name: str
    ) -> ComponentMetadata:
        """Get metadata for a registered component."""
        cat = cls._resolve_category(category)
        if cat not in cls._components:
            raise ValueError(f"Unknown category: {cat}")
        if name not in cls._components[cat]:
            available = list(cls._components[cat].keys())
            raise ValueError(f"Unknown {cat.value}: {name}. Available: {available}")
        return cls._components[cat][name]["metadata"]

    @classmethod
    def list(
        cls, category: ComponentCategory | str | None = None
    ) -> dict[str, builtins.list[str]]:
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
        tags: builtins.list[str] | None = None,
        credit_type: str | None = None,
        family: str | None = None,
    ) -> builtins.list[dict[str, Any]]:
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

        results: builtins.list[dict[str, Any]] = []
        for cat in categories:
            if cat not in cls._components:
                continue
            for name, info in cls._components[cat].items():
                meta: ComponentMetadata = info["metadata"]
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
    ) -> dict[str, builtins.list[dict[str, Any]]]:
        """Get components compatible with a given model."""
        model_meta = cls.get_metadata(model_category, model_name)
        primary_domain = model_meta.domains[0] if model_meta.domains else None

        compat: dict[str, builtins.list[dict[str, Any]]] = {}
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
    def clear(cls) -> None:
        """Clear the registry (mainly for testing)."""
        cls._components.clear()

    @classmethod
    def export_yaml(cls, path: str) -> None:
        """Export all registered component metadata to a YAML file."""
        import yaml  # local import: keep module import cheap (AGENTS.md)

        export_data: dict[str, dict[str, dict[str, Any]]] = {}
        for category, comps in cls._components.items():
            cat_name = category.value
            export_data[cat_name] = {}
            for name, info in comps.items():
                meta = info["metadata"]
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


def register_sparsity(name: str | None = None, **kwargs) -> Callable:
    """Register a sparsity component."""
    return Registry.register(ComponentCategory.SPARSITY, name, **kwargs)


def register_metric(name: str | None = None, **kwargs) -> Callable:
    """Register a metric component."""
    return Registry.register(ComponentCategory.METRIC, name, **kwargs)


def list_models() -> list[str]:
    """Convenience: list all registered model names."""
    return list(Registry._components.get(ComponentCategory.MODEL, {}).keys())


__all__ = [
    "Capability",
    "ComponentCategory",
    "ComponentMetadata",
    "ComputeProfile",
    "Domain",
    "IncompatibilityError",
    "LocalityLevel",
    "Registry",
    "list_models",
    "register_metric",
    "register_model",
    "register_optimizer",
    "register_propagator",
    "register_sparsity",
]
