"""5-Dimensional Physico-Computational Ontology for Bioplausible Systems.

This module defines the five orthogonal axes (S x G x D x C x U) that compose
any computronium neural network. The tensor product of these primitives
replaces the flat registry of 111+ hardcoded model permutations with a
generative, mathematically rigorous composition engine.

Ontology Layers:
    1. Substrate (S) - Physical state space constraints (precision, noise, sparsity)
    2. Geometry (G) - Topology & routing (spatial arrangement of nodes)
    3. StateDynamics (D) - Forward evolution & settling (how activations evolve)
    4. CreditAssignment (C) - Error routing & pseudo-gradients (learning signal)
    5. ParameterUpdate (U) - Optimization rule (how pseudo-gradients become ΔW)

Each layer is a Protocol enabling structural typing and zero-cost abstraction.
The composing System[TS, TG, TD, TC, TU] uses PEP 695 generics for full
type safety: invalid compositions are caught at type-check time.
"""

from computronium.ontology.credit import (
    BackpropCredit,
    CreditAssignment,
    CreditAssignmentConfig,
    GradientCredit,
    HomeostaticCredit,
    LocalGoodnessCredit,
    Phase,
    RandomProjectionsCredit,
    TargetInversionCredit,
    TemporalTraceCredit,
    ThermodynamicContrast,
)
from computronium.ontology.dynamics import (
    DiffusionDynamics,
    EnergyMinimizationDynamics,
    InstantaneousDynamics,
    LazyStateDynamics,
    PredictiveSettlingDynamics,
    SpikeIntegrationDynamics,
    StateDynamics,
    StateDynamicsConfig,
)

# Utility functions
from computronium.ontology.geometry import (
    FeedforwardGeometry,
    Geometry,
    GeometryConfig,
    RecurrentGeometry,
    TileGeometry,
    _set_param_name,
    layer_stack,
)
from computronium.ontology.plasticity import (
    FastWeightPlasticity,
    NullPlasticity,
    PlasticityConfig,
    PlasticityPrimitive,
    RoutingPlasticity,
    RuleStatePlasticity,
    SubstrateCoupledPlasticity,
    TransitionFn,
)
from computronium.ontology.substrate import (
    AnalogSubstrate,
    ComplexSubstrate,
    DigitalSubstrate,
    MemristiveSubstrate,
    NeuromorphicSubstrate,
    NoisySubstrate,
    OpticalSubstrate,
    QuantizedSubstrate,
    QuantumSubstrate,
    SparseSubstrate,
    Substrate,
    SubstrateConfig,
    SubstrateType,
    TernarySubstrate,
    substrate_from_config,
)
from computronium.ontology.system import (
    FAMILY_TOLERANCES,
    ModelAdapter,
    System,
    SystemConfig,
    SystemState,
)
from computronium.ontology.system import (
    Phase as SystemPhase,
)
from computronium.ontology.update import (
    ElasticConsolidationUpdate,
    EuclideanUpdate,
    NaturalGradientUpdate,
    ParameterUpdate,
    ParameterUpdateConfig,
    RiemannianOrthogonalUpdate,
    SpectralConstrainedUpdate,
)
from computronium.ontology.utils import (
    ConfigFactory,
    _learnable_weight_names,
    apply_pseudo_gradients,
)

# Re-export transition types from state module for convenience
from computronium.state import (
    CompositeState,
    CoupledTransition,
    StateRegistry,
    StateVariable,
    SystemContext,
)

__all__ = [
    # Substrate
    "SubstrateType",
    "SubstrateConfig",
    "Substrate",
    "DigitalSubstrate",
    "AnalogSubstrate",
    "MemristiveSubstrate",
    "NeuromorphicSubstrate",
    "OpticalSubstrate",
    "QuantumSubstrate",
    "SparseSubstrate",
    "TernarySubstrate",
    "ComplexSubstrate",
    "NoisySubstrate",
    "QuantizedSubstrate",
    "substrate_from_config",
    # Geometry
    "GeometryConfig",
    "Geometry",
    "FeedforwardGeometry",
    "RecurrentGeometry",
    "TileGeometry",
    # StateDynamics
    "StateDynamicsConfig",
    "StateDynamics",
    "EnergyMinimizationDynamics",
    "PredictiveSettlingDynamics",
    "SpikeIntegrationDynamics",
    "InstantaneousDynamics",
    "DiffusionDynamics",
    "LazyStateDynamics",
    # CreditAssignment
    "CreditAssignmentConfig",
    "CreditAssignment",
    "Phase",
    "ThermodynamicContrast",
    "RandomProjectionsCredit",
    "LocalGoodnessCredit",
    "TemporalTraceCredit",
    "TargetInversionCredit",
    "HomeostaticCredit",
    "GradientCredit",
    "BackpropCredit",
    # ParameterUpdate
    "ParameterUpdateConfig",
    "ParameterUpdate",
    "EuclideanUpdate",
    "RiemannianOrthogonalUpdate",
    "SpectralConstrainedUpdate",
    "NaturalGradientUpdate",
    "ElasticConsolidationUpdate",
    # Plasticity (M-axis)
    "PlasticityPrimitive",
    "PlasticityConfig",
    "TransitionFn",
    "NullPlasticity",
    "FastWeightPlasticity",
    "RoutingPlasticity",
    "RuleStatePlasticity",
    "SubstrateCoupledPlasticity",
    # System
    "SystemConfig",
    "System",
    "SystemState",
    "FAMILY_TOLERANCES",
    "ModelAdapter",
    # Utility functions
    "_learnable_weight_names",
    "_set_param_name",
    "apply_pseudo_gradients",
    "layer_stack",
    "ConfigFactory",
    # State types (from computronium.state)
    "CompositeState",
    "SystemContext",
    "StateRegistry",
    "StateVariable",
    "CoupledTransition",
    "SystemPhase",
]
