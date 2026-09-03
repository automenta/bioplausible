"""Unified Registry System for Bioplausible.

Decorator-based registry for all components (models, propagators,
optimizers, sparsity) enabling AutoScientist to query and compose
intelligently."""

import itertools
import pathlib
from dataclasses import MISSING, dataclass, field, fields
from enum import Enum, StrEnum
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar, cast

if TYPE_CHECKING:
    from collections.abc import Callable

    from computronium.ontology import System

from computronium.core.exceptions import IncompatibilityError as _IncompatibilityError
from computronium.core.logging import get_logger

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


class ComponentCategory(str, Enum):  # ruff: ignore[replace-str-enum]
    """Categories of components in the registry.

    Core categories (for AutoScientist composition):
    1. MODEL - Model architectures (including model-side learners: FF, TP, PCN, Hebbian)
    2. CREDIT_ASSIGNMENT - Learning rules/propagators (Backprop, FA, EP, TP, etc.)
    3. PARAM_UPDATE - Parameter updates (optimizers + update strategies + constraints)
    4. HARDWARE - Hardware substrates, kernel backends, sparsity

    Auxiliary categories (for infrastructure):
    - METRIC - Evaluation metrics
    - TASK - Benchmark tasks
    - TRACK - Validation tracks
    """

    # Core categories
    MODEL = "model"
    CREDIT_ASSIGNMENT = "credit_assignment"
    PARAM_UPDATE = "param_update"
    HARDWARE = "hardware"

    # Auxiliary categories
    METRIC = "metric"
    TASK = "task"
    TRACK = "track"


class LocalityLevel(str, Enum):  # ruff: ignore[replace-str-enum]
    """Credit assignment locality level."""

    GLOBAL = "global"  # Full backprop
    LAYERWISE = "layerwise"  # Layer-local (Forward-Forward, Target Prop)
    LOCAL = "local"  # Neuron/synapse local (Hebbian, STDP, EquiTile)
    EQUILIBRIUM = "equilibrium"  # Energy-based (EqProp, CHL)
    FORWARD_ONLY = "forward-only"  # No backward pass (PEPITA, FF)


class ComputeProfile(str, Enum):  # ruff: ignore[replace-str-enum]
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
    # "mep", "tile", etc. Directory layout mirrors this but `family` is the
    # canonical searchable attribute for grouping in the README/Registry queries.
    family: str = ""
    # Domain support: "vision", "lm", "rl", "graph", "tabular", "timeseries", "scientific"
    domain: str = ""
    # Required and provided capabilities (per REFACTOR3 §4)
    requires: list[str] = field(default_factory=list)
    provides: list[str] = field(default_factory=list)
    extra: dict[str, object] = field(default_factory=dict)
    # Ontology axes (Phase 0): explicit 5-D layer assignments for native models.
    # When populated, ModelAdapter.to_system() becomes deterministic instead of heuristic.
    # Values correspond to ontology config class names (e.g., "DigitalSubstrate",
    # "RecurrentGeometry", "EnergyMinimizationDynamics", "ThermodynamicContrast",
    # "EuclideanUpdate") or axis enum values.
    ontology_substrate: str = ""
    ontology_geometry: str = ""
    ontology_dynamics: str = ""
    ontology_credit: str = ""
    ontology_update: str = ""


@dataclass(frozen=True, slots=True)
class _QueryFilter:
    """Immutable capability query predicate spec.

    Held frozen to make intent explicit and to allow future caching
    over the predicate. Each field is a filter; ``None`` means
    ``no constraint on this axis``.
    """

    locality: LocalityLevel | None = None
    compute: ComputeProfile | None = None
    requires_backward: bool | None = None
    min_bio_score: float | None = None
    max_bio_score: float | None = None
    tags: list[str] | None = None
    credit_type: CreditAssignmentType | None = None
    family: str | None = None
    domain: str | None = None
    _predicates: tuple[_Predicate, ...] = field(init=False, repr=False, default=())

    def __post_init__(self) -> None:
        """Build the predicate dispatch table once at construction."""
        predicates: list[_Predicate] = []
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
        if self.domain is not None:
            predicates.append(_DomainIs(self.domain))
        object.__setattr__(self, "_predicates", tuple(predicates))

    def matches(self, meta: ComponentMetadata) -> bool:
        """Return True iff ``meta`` satisfies every constraint in this filter."""
        return all(predicate(meta) for predicate in self._predicates)


class _Predicate(Protocol):
    """Single-axis capability predicate."""

    def __call__(self, meta: ComponentMetadata) -> bool: ...


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


@dataclass(frozen=True, slots=True)
class _DomainIs:
    """True iff ``meta`` domain matches exactly."""

    domain: str

    def __call__(self, meta: ComponentMetadata) -> bool:
        return meta.domain == self.domain


_registration_state = {"native": False, "blocked": False}


def _ensure_native_registered() -> None:
    """Lazily populate the Registry with native models + optimizers.

    Deferred from import time so light imports of this module stay
    torch-free (module-boundary invariant). Idempotent; permanently
    disabled once :meth:`Registry.clear` takes ownership of contents.
    """
    state = _registration_state
    if state["blocked"] or state["native"]:
        return
    state["native"] = True


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
    ] = {}  # category -> {name: {cls, metadata}}  # ruff: ignore[mutable-class-default]

    # Compatibility map: names that alias to a registered component in a
    # *different* category. Currently propagator names that map to
    # model-side implementations (the learning rule lives in the model's
    # ``train_step``, not in a separate propagator object).
    _ALIASES: dict[str, tuple[ComponentCategory, str]] = {  # ruff: ignore[mutable-class-default]
        "ff": (ComponentCategory.MODEL, "forward_forward"),
        "target_prop": (ComponentCategory.MODEL, "diff_target_prop"),
        "difference_target_prop": (ComponentCategory.MODEL, "diff_target_prop"),
        "predictive_coding": (
            ComponentCategory.MODEL,
            "predictive_coding_hybrid",
        ),
        # Legacy model name aliases to native models (Phase 0)
        "backprop_mlp": (ComponentCategory.MODEL, "native_backprop_mlp"),
        "eqprop_mlp": (ComponentCategory.MODEL, "native_eqprop_mlp"),
        "fa_mlp": (ComponentCategory.MODEL, "native_fa_mlp"),
        "pepita_mlp": (ComponentCategory.MODEL, "native_pepita_mlp"),
        "tile_ep": (ComponentCategory.MODEL, "native_tile_ep"),
        "tile_fa": (ComponentCategory.MODEL, "native_tile_fa"),
        "tile_tp": (ComponentCategory.MODEL, "native_tile_tp"),
        "tile_snn": (ComponentCategory.MODEL, "native_tile_snn"),
        "tile_hebbian": (ComponentCategory.MODEL, "native_tile_hebbian"),
        "tile_pc": (ComponentCategory.MODEL, "native_tile_pc"),
        "tile_gnn": (ComponentCategory.MODEL, "native_tile_gnn"),
        "diffusion_eqprop": (ComponentCategory.MODEL, "native_diffusion_eqprop"),
        "momentum_eqprop": (ComponentCategory.MODEL, "native_momentum_eqprop"),
        "sparse_eqprop": (ComponentCategory.MODEL, "native_sparse_eqprop"),
        "ternary_eqprop": (ComponentCategory.MODEL, "native_ternary_eqprop"),
        "holomorphic_ep": (ComponentCategory.MODEL, "native_holomorphic_ep"),
        "directed_ep": (ComponentCategory.MODEL, "native_directed_ep"),
        "finite_nudge_ep": (ComponentCategory.MODEL, "native_finite_nudge_ep"),
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

        def decorator(component: Component, _name: str | None = name) -> Component:
            if _name is None:
                _name = getattr(component, "__name__", repr(component))
            # _name is guaranteed to be str at this point
            assert _name is not None  # ruff: ignore[assert]
            if _name in cls._components[category]:
                logger.info("Overwriting component %s/%s", category.value, _name)
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
        _ensure_native_registered()
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
        _ensure_native_registered()
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
    def _names_with_aliases(cls, cat: ComponentCategory) -> list[str]:
        """Registered names for ``cat`` plus any aliases resolving into it."""
        registered = cls._components.get(cat, {})
        names = list(registered.keys())
        names.extend(
            alias
            for alias, (target_cat, target_name) in cls._ALIASES.items()
            if target_cat == cat and target_name in registered and alias not in names
        )
        return names

    @classmethod
    def list(
        cls, category: ComponentCategory | str | None = None
    ) -> dict[str, list[str]]:
        """List all registered components (including aliases), optionally
        filtered by category."""
        _ensure_native_registered()
        if category is not None:
            cat: ComponentCategory = cls._resolve_category(category)
            return {cat.value: cls._names_with_aliases(cat)}
        return {
            cat.value: cls._names_with_aliases(cat)
            for cat in cls._components
            if isinstance(cat, ComponentCategory)
        }

    @classmethod
    def query(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
        cls,
        category: ComponentCategory | str | None = None,
        locality: LocalityLevel | None = None,
        compute: ComputeProfile | None = None,
        requires_backward: bool | None = None,
        min_bio_score: float | None = None,
        max_bio_score: float | None = None,
        tags: list[str] | None = None,
        credit_type: CreditAssignmentType | None = None,
        family: str | None = None,
        domain: str | None = None,
    ) -> list[dict[str, object]]:
        """Query registry with capability constraints.

        Returns list of ``{name, category, class, metadata}`` dict
        entries matching ALL criteria. Designed for AutoScientist
        composition.
        """
        _ensure_native_registered()
        flt = _QueryFilter(
            locality=locality,
            compute=compute,
            requires_backward=requires_backward,
            min_bio_score=min_bio_score,
            max_bio_score=max_bio_score,
            tags=tags,
            credit_type=credit_type,
            family=family,
            domain=domain,
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
        model_meta = cls.get_metadata(model_category, model_name)  # ruff: ignore[unused-variable]

        compat: dict[str, list[dict[str, object]]] = {}
        for cat in ComponentCategory:
            if cat == model_category:
                continue
            compat[cat.value] = cls.query(category=cat)
        return compat

    @classmethod
    def query_axis(
        cls,
        substrate: str | list[str] | None = None,
        geometry: str | list[str] | None = None,
        dynamics: str | list[str] | None = None,
        credit: str | list[str] | None = None,
        update: str | list[str] | None = None,
        category: ComponentCategory | str | None = None,
        min_bio_score: float | None = None,
        max_bio_score: float | None = None,  # ruff: ignore[unused-class-method-argument]
    ) -> list[dict[str, object]]:
        """Query the registry by explicit 5-D ontology axis values.

        This is the primary interface for the AutoScientist to discover
        models along specific ontology axes. Uses the explicit `ontology_*`
        fields on ComponentMetadata for native models, with heuristic
        fallback for legacy models.

        Args:
            substrate: Substrate axis value(s) (e.g., "DigitalSubstrate", "SparseSubstrate")
            geometry: Geometry axis value(s) (e.g., "RecurrentGeometry", "TileGeometry")
            dynamics: Dynamics axis value(s) (e.g., "EnergyMinimizationDynamics", "InstantaneousDynamics")
            credit: Credit axis value(s) (e.g., "ThermodynamicContrast", "RandomProjectionsCredit")
            update: Update axis value(s) (e.g., "EuclideanUpdate", "RiemannianOrthogonalUpdate")
            category: Optional category filter
            min_bio_score: Minimum bio-plausibility score
            max_bio_score: Maximum bio-plausibility score

        Returns:
            List of matching components with their 5-D layer assignments.
        """
        fixed: dict[str, str | list[str]] = {}
        if substrate is not None:
            fixed["substrate"] = substrate
        if geometry is not None:
            fixed["geometry"] = geometry
        if dynamics is not None:
            fixed["dynamics"] = dynamics
        if credit is not None:
            fixed["credit"] = credit
        if update is not None:
            fixed["update"] = update

        return cls.query_ontology(
            fixed=fixed if fixed else None,
            category=category,
            min_bio_score=min_bio_score,
        )

    @classmethod
    def to_system(cls, model_name: str, **model_kwargs) -> System:
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
        from computronium.ontology import ModelAdapter

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
            # component is a callable factory function
            model = component(**model_kwargs)  # type: ignore[operator]

        # Check if the factory returned a native System directly
        # (bypasses ModelAdapter for native 5-D compositions)
        from computronium.ontology import System

        if isinstance(model, System):
            return model

        # Adapt to 5-D ontology via ModelAdapter (legacy path)
        adapter = ModelAdapter(model, metadata)
        return adapter.to_system()

    @classmethod
    def to_system_by_category(
        cls, category: ComponentCategory | str, name: str
    ) -> System:
        """Project any registered component into the 5-D ontology.

        For models, this is the same as ``to_system``. For other categories
        (propagators, optimizers, etc.), it creates a minimal System with
        the component mapped to its corresponding layer.
        """
        from computronium.core.system_trainer import compose_system
        from computronium.ontology import (
            DigitalSubstrate,
            EuclideanUpdate,
            FeedforwardGeometry,
            GeometryConfig,
            InstantaneousDynamics,
            ModelAdapter,
            ThermodynamicContrast,
        )

        cat = cls._resolve_category(category)
        component = cls.get(cat, name)
        metadata = cls.get_metadata(cat, name)

        if isinstance(component, type):  # ruff: ignore[if-else-block-instead-of-if-exp]
            instance = component()
        else:
            instance = component()  # type: ignore[operator]

        if cat == ComponentCategory.MODEL:
            adapter = ModelAdapter(instance, metadata)
            return adapter.to_system()

        # For non-model components, create a minimal system with defaults
        # and the component mapped to its layer
        substrate = DigitalSubstrate()
        geometry = FeedforwardGeometry(
            GeometryConfig.feedforward(input_dim=10, output_dim=3, hidden_dims=(20,))
        )
        dynamics = InstantaneousDynamics()
        credit = ThermodynamicContrast()
        update = EuclideanUpdate()

        # This is a placeholder - in practice, propagators map to CreditAssignment,
        # optimizers map to ParameterUpdate, etc.
        return compose_system(substrate, geometry, dynamics, credit, update)

    @classmethod
    def query_ontology(  # ruff: ignore[complex-structure, too-many-branches, too-many-locals]
        cls,
        fixed: dict[str, str | list[str]] | None = None,
        sweep: str | None = None,
        sweep_values: list[str] | None = None,
        category: ComponentCategory | str | None = None,
        min_bio_score: float | None = None,
    ) -> list[dict[str, object]]:
        """Query the registry along the 5-D ontology axes.

        Enables structured ablation studies by holding some layers constant
        and sweeping others. This is the primary interface for the AutoScientist
        to explore the hypercube of computronium systems.

        Args:
            fixed: Dictionary of layer -> value(s) to hold constant.
                Keys: "substrate", "geometry", "dynamics", "credit", "update"
                Values: single value or list of values
            sweep: Layer to sweep over ("substrate", "geometry", "dynamics", "credit", "update")
            sweep_values: Values to sweep for the sweep layer
            category: Optional category filter
            domain: Optional domain filter
            min_bio_score: Minimum bio-plausibility score

        Returns:
            List of matching components with their 5-D layer assignments.
        """
        # Layer to metadata field mapping
        layer_fields = {
            "substrate": "compute_profile",
            "geometry": "family",  # topology_type inferred from family
            "dynamics": "locality_level",
            "credit": "credit_assignment_type",
            "update": "tags",  # optimizer tags like "muon", "spectral"
        }

        # Map ontology layer values to registry metadata values (list of possible matches)
        layer_value_map = {
            "substrate": {
                "Digital": [ComputeProfile.GPU, ComputeProfile.CPU],
                "DigitalSubstrate": [ComputeProfile.GPU, ComputeProfile.CPU],
                "Memristive": [ComputeProfile.MEMRISTOR],
                "MemristiveSubstrate": [ComputeProfile.MEMRISTOR],
                "Neuromorphic": [ComputeProfile.NEUROMORPHIC],
                "NeuromorphicSubstrate": [ComputeProfile.NEUROMORPHIC],
                "Optical": [ComputeProfile.OPTICAL],
                "OpticalSubstrate": [ComputeProfile.OPTICAL],
                "Quantum": [ComputeProfile.ANALOG],
                "QuantumSubstrate": [ComputeProfile.ANALOG],
                "Sparse": [ComputeProfile.GPU],
                "SparseSubstrate": [ComputeProfile.GPU],
                "Ternary": [ComputeProfile.GPU],
                "TernarySubstrate": [ComputeProfile.GPU],
            },
            "geometry": {
                "Feedforward": [
                    "backprop",
                    "fa",
                    "forward_only",
                    "hebbian",
                    "target_prop",
                    "mep",
                ],
                "FeedforwardGeometry": [
                    "backprop",
                    "fa",
                    "forward_only",
                    "hebbian",
                    "target_prop",
                    "mep",
                ],
                "Recurrent": ["eqprop", "recurrent", "equilibrium", "ep", "chl"],
                "RecurrentGeometry": [
                    "eqprop",
                    "recurrent",
                    "equilibrium",
                    "ep",
                    "chl",
                ],
                "TileMesh": ["tile"],
                "TileGeometry": ["tile"],
                "Neuromorphic": ["neuromorphic", "fabric"],
                "SpatialLattice": ["spatial_lattice", "neural_cube", "3d"],
            },
            "dynamics": {
                "Instantaneous": [LocalityLevel.FORWARD_ONLY, LocalityLevel.GLOBAL],
                "InstantaneousDynamics": [
                    LocalityLevel.FORWARD_ONLY,
                    LocalityLevel.GLOBAL,
                ],
                "EnergyMinimization": [LocalityLevel.EQUILIBRIUM],
                "EnergyMinimizationDynamics": [LocalityLevel.EQUILIBRIUM],
                "PredictiveSettling": [LocalityLevel.EQUILIBRIUM],
                "PredictiveSettlingDynamics": [LocalityLevel.EQUILIBRIUM],
                "SpikeIntegration": [LocalityLevel.LOCAL],
                "SpikeIntegrationDynamics": [LocalityLevel.LOCAL],
                "DiffusionDynamics": [LocalityLevel.EQUILIBRIUM],
            },
            "credit": {
                "ThermodynamicContrast": ["equilibrium", "hebbian", "gradient"],
                "RandomProjections": [
                    "feedback_alignment",
                    "random_projections",
                    "target",
                    "gradient",
                ],
                "RandomProjectionsCredit": [
                    "feedback_alignment",
                    "random_projections",
                    "target",
                    "gradient",
                ],
                "LocalGoodness": ["forward_only", "hebbian", "gradient"],
                "LocalGoodnessCredit": ["forward_only", "hebbian", "gradient"],
                "TemporalTrace": ["spiking", "temporal_trace", "gradient"],
                "TemporalTraceCredit": ["spiking", "temporal_trace", "gradient"],
                "TargetInversion": ["target", "target_inversion", "gradient"],
                "TargetInversionCredit": ["target", "target_inversion", "gradient"],
                "HomeostaticCredit": ["hebbian", "homeostatic"],
                "BackpropCredit": ["gradient", "backprop"],
            },
            "update": {
                "Euclidean": ["sgd", "adam", "adamw", "plain", "gradient"],
                "EuclideanUpdate": ["sgd", "adam", "adamw", "plain", "gradient"],
                "RiemannianOrthogonal": ["muon", "riemannian"],
                "RiemannianOrthogonalUpdate": ["muon", "riemannian"],
                "SpectralConstrained": ["spectral"],
                "SpectralConstrainedUpdate": ["spectral"],
                "NaturalGradient": ["fisher", "natural"],
                "NaturalGradientUpdate": ["fisher", "natural"],
                "ElasticConsolidation": ["ewc", "elastic"],
                "ElasticConsolidationUpdate": ["ewc", "elastic"],
            },
        }

        # Map field name to query parameter name
        field_to_param = {
            "compute_profile": "compute",
            "locality_level": "locality",
            "credit_assignment_type": "credit_type",
            "family": "family",
            "tags": "tags",
        }

        # Build list of query_kwargs for each combination of fixed values
        if fixed:
            # For each layer, get the list of possible values
            fixed_params: dict[str, list] = {}
            for layer, values in fixed.items():
                if layer not in layer_fields:
                    continue
                field = layer_fields[layer]
                val_list = values if isinstance(values, list) else [values]
                mapped = []
                for v in val_list:
                    mapped.extend(layer_value_map[layer].get(v, [v]))
                param_name = field_to_param.get(field, field)
                fixed_params[param_name] = mapped

            # Generate all combinations of fixed parameters
            param_names = list(fixed_params.keys())
            param_values = list(fixed_params.values())
            all_combinations = list(itertools.product(*param_values))

            # Run query for each combination and merge results
            all_results: list[dict[str, object]] = []
            for combo in all_combinations:
                query_kwargs = dict(zip(param_names, combo))
                query_kwargs.update({
                    "category": category,
                    "min_bio_score": min_bio_score,
                })
                # Remove None values
                query_kwargs = {k: v for k, v in query_kwargs.items() if v is not None}
                results = cls.query(**query_kwargs)
                all_results.extend(results)

            # Deduplicate results by name+category
            seen = set()
            results = []
            for r in all_results:
                key = (r["name"], r["category"])
                if key not in seen:
                    seen.add(key)
                    results.append(r)
        else:
            # No fixed constraints, just run single query
            query_kwargs = {
                "category": category,
                "min_bio_score": min_bio_score,
            }
            results = cls.query(**query_kwargs)

        # If sweep is specified, filter and augment with sweep values
        if sweep and sweep_values:
            sweep_field = layer_fields.get(sweep)
            if sweep_field:
                sweep_mapped = []
                for v in sweep_values:
                    sweep_mapped.extend(layer_value_map[sweep].get(v, [v]))
                filtered = []
                for r in results:
                    meta = r["metadata"]
                    meta_val = getattr(meta, sweep_field, None)
                    if isinstance(meta_val, list):
                        if any(v in meta_val for v in sweep_mapped):
                            filtered.append(r)
                    elif meta_val in sweep_mapped:
                        filtered.append(r)
                results = filtered

        # Augment results with 5-D layer assignments
        for r in results:
            r["ontology_layers"] = cls._infer_ontology_layers(r["metadata"])  # type: ignore[arg-type]

        return results

    @classmethod
    def _infer_ontology_layers(cls, meta: ComponentMetadata) -> dict[str, str]:
        """Infer the 5-D ontology layer assignments for a component.

        Uses explicit ontology_axes fields when available (for native models),
        otherwise falls back to heuristic inference from metadata.
        """
        # If explicit ontology axes are provided, use them directly
        if (
            meta.ontology_substrate
            or meta.ontology_geometry
            or meta.ontology_dynamics
            or meta.ontology_credit
            or meta.ontology_update
        ):
            return {
                "substrate": meta.ontology_substrate or meta.compute_profile.value,
                "geometry": meta.ontology_geometry or (meta.family or "feedforward"),
                "dynamics": meta.ontology_dynamics or meta.locality_level.value,
                "credit": meta.ontology_credit or meta.credit_assignment_type,
                "update": meta.ontology_update
                or (",".join(meta.tags) if meta.tags else "euclidean"),
            }

        # Heuristic fallback for legacy models
        return {
            "substrate": meta.compute_profile.value,
            "geometry": meta.family or "feedforward",
            "dynamics": meta.locality_level.value,
            "credit": meta.credit_assignment_type,
            "update": ",".join(meta.tags) if meta.tags else "euclidean",
        }

    @classmethod
    def clear(cls) -> None:
        """Clear the registry (mainly for testing).

        Disables lazy native registration so subsequent reads reflect the
        explicitly-managed (empty) state instead of repopulating it.
        """
        cls._components.clear()
        _registration_state["blocked"] = True

    @classmethod
    def export_yaml(cls, path: str) -> None:
        """Export all registered component metadata to a YAML file."""
        import yaml  # local import: keep module import cheap (AGENTS.md)

        export_data: dict[str, dict[str, dict[str, object]]] = {}
        for cat_name, comps in cls._components.items():
            export_data[cat_name] = {}
            for name, info in comps.items():
                meta = cast("ComponentMetadata", info["metadata"])
                export_data[cat_name][name] = {
                    "name": meta.name,
                    "category": meta.category.value,
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
                    "domain": meta.domain,
                    "ontology_substrate": meta.ontology_substrate,
                    "ontology_geometry": meta.ontology_geometry,
                    "ontology_dynamics": meta.ontology_dynamics,
                    "ontology_credit": meta.ontology_credit,
                    "ontology_update": meta.ontology_update,
                }

        with pathlib.Path(path).open("w", encoding="utf-8") as f:
            yaml.dump(export_data, f, default_flow_style=False, sort_keys=False)

        n_components = sum(len(v) for v in export_data.values())
        logger.info("Registry exported to %s: %d components", path, n_components)


# Convenience decorators (core categories)
def register_model(name: str | None = None, **kwargs) -> Callable:
    """Register a model component."""
    return Registry.register(ComponentCategory.MODEL, name, **kwargs)


def register_credit_assignment(name: str | None = None, **kwargs) -> Callable:
    """Register a credit assignment / learning rule component."""
    return Registry.register(ComponentCategory.CREDIT_ASSIGNMENT, name, **kwargs)


def register_param_update(name: str | None = None, **kwargs) -> Callable:
    """Register a parameter update component (optimizer, update strategy, or constraint)."""
    return Registry.register(ComponentCategory.PARAM_UPDATE, name, **kwargs)


def register_hardware(name: str | None = None, **kwargs) -> Callable:
    """Register a hardware component (substrate, kernel backend, sparsity)."""
    return Registry.register(ComponentCategory.HARDWARE, name, **kwargs)


# Convenience decorators (auxiliary categories)
def register_metric(name: str | None = None, **kwargs) -> Callable:
    """Register a metric component."""
    return Registry.register(ComponentCategory.METRIC, name, **kwargs)


def register_task(name: str | None = None, **kwargs) -> Callable:
    """Register a task component."""
    return Registry.register(ComponentCategory.TASK, name, **kwargs)


def register_track(name: str | None = None, **kwargs) -> Callable:
    """Register a validation track component."""
    return Registry.register(ComponentCategory.TRACK, name, **kwargs)


# Deprecated aliases (for backward compatibility)
def register_propagator(name: str | None = None, **kwargs) -> Callable:
    """Register a credit assignment component (deprecated: use register_credit_assignment).

    Registers under CREDIT_ASSIGNMENT (new).
    """
    import warnings

    warnings.warn(
        "register_propagator is deprecated, use register_credit_assignment",
        DeprecationWarning,
        stacklevel=2,
    )
    return Registry.register(ComponentCategory.CREDIT_ASSIGNMENT, name, **kwargs)


def register_optimizer(name: str | None = None, **kwargs) -> Callable:
    """Register a parameter update component (deprecated: use register_param_update).

    Registers under PARAM_UPDATE (new).
    """
    import warnings

    warnings.warn(
        "register_optimizer is deprecated, use register_param_update",
        DeprecationWarning,
        stacklevel=2,
    )
    return Registry.register(ComponentCategory.PARAM_UPDATE, name, **kwargs)


def register_update_strategy(name: str | None = None, **kwargs) -> Callable:
    """Register an update-strategy component (deprecated: use register_param_update).

    Registers under PARAM_UPDATE (new).
    """
    import warnings

    warnings.warn(
        "register_update_strategy is deprecated, use register_param_update",
        DeprecationWarning,
        stacklevel=2,
    )
    return Registry.register(ComponentCategory.PARAM_UPDATE, name, **kwargs)


def register_constraint(name: str | None = None, **kwargs) -> Callable:
    """Register a constraint component (deprecated: use register_param_update).

    Registers under PARAM_UPDATE (new).
    """
    import warnings

    warnings.warn(
        "register_constraint is deprecated, use register_param_update",
        DeprecationWarning,
        stacklevel=2,
    )
    return Registry.register(ComponentCategory.PARAM_UPDATE, name, **kwargs)


def register_sparsity(name: str | None = None, **kwargs) -> Callable:
    """Register a sparsity component (deprecated: use register_hardware).

    Registers under HARDWARE (new).
    """
    import warnings

    warnings.warn(
        "register_sparsity is deprecated, use register_hardware",
        DeprecationWarning,
        stacklevel=2,
    )
    return Registry.register(ComponentCategory.HARDWARE, name, **kwargs)


def register_controller(name: str | None = None, **kwargs) -> Callable:
    """Register a training-side controller component (deprecated: use register_hardware).

    Registers under HARDWARE (new).
    """
    import warnings

    warnings.warn(
        "register_controller is deprecated, use register_hardware",
        DeprecationWarning,
        stacklevel=2,
    )
    return Registry.register(ComponentCategory.HARDWARE, name, **kwargs)


def list_models() -> list[str]:
    """Convenience: list all registered model names (including aliases).

    Delegates to :meth:`Registry.list`, so first access triggers the lazy
    native registration exactly like every other Registry read.
    """
    return Registry.list(ComponentCategory.MODEL)[ComponentCategory.MODEL.value]


__all__ = [
    "Capability",
    "ComponentCategory",
    "ComponentMetadata",
    "ComputeProfile",
    "CreditAssignmentType",
    "IncompatibilityError",
    "LocalityLevel",
    "Registry",
    "list_models",
    "register_constraint",
    "register_controller",
    "register_credit_assignment",
    "register_hardware",
    "register_metric",
    "register_model",
    "register_optimizer",
    "register_param_update",
    "register_propagator",
    "register_sparsity",
    "register_task",
    "register_track",
    "register_update_strategy",
]
