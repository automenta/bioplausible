"""
Bioplausible: Unified Platform for Bio-Plausible Learning Research

Minimal, clean API for training and experimentation.

Quick Start:
    from bioplausible import SystemTrainer, ExperimentConfig

    config = ExperimentConfig(...)
    trainer = SystemTrainer.from_configs(config, train_data, val_data)
    history = trainer.fit()

Or using native models:
    from bioplausible import native_eqprop_mlp, native_backprop_mlp, native_fa_mlp

    system = native_eqprop_mlp(input_dim=784, hidden_dim=256, output_dim=10)
    # Use with SystemTrainer

One-Line System Construction:
    from bioplausible import create_backprop_mlp, create_eqprop_mlp, create_fa_mlp
    from bioplausible import create_routing_mlp, create_fast_weight_mlp

    system = create_backprop_mlp(input_dim=784, hidden_dims=(256, 128), output_dim=10)
    system = create_eqprop_mlp(input_dim=784, hidden_dims=(256, 128), output_dim=10, beta=0.5, n_iters=20)
    system = create_fa_mlp(input_dim=784, hidden_dims=(256, 128), output_dim=10)
    system = create_routing_mlp(input_dim=784, hidden_dims=(256, 128), output_dim=10)
    system = create_fast_weight_mlp(input_dim=784, hidden_dims=(256, 128), output_dim=10)

Two-Tier Propagator / Model Architecture:
----------------------------------------
The zoo provides two complementary interfaces for bio-plausible learning:

1. Learning rules (``bioplausible.core.local_learning.rules``): Learning rules
   implemented as drop-in ``torch.optim.Optimizer`` subclasses
   (`BioOptimizer`, `LearningRuleOptimizer`). These mutate parameters of any
   model: Backprop, FeedbackAlignment, EqProp, ContrastiveHebbianLearning,
   MEP presets (smep, sdmep, ...). Use via the Registry API:
   ``Registry.get(ComponentCategory.PARAM_UPDATE, "eq_prop")``.

2. Model side (`bioplausible.zoo.models`): Learning rules that require
   model-side control of the forward/training loop (custom dual-phase passes,
   learned inverse maps, settling dynamics with internal state). These expose
   ``train_step(x, y) -> dict[str, float]`` instead of ``optimizer.step()``.

Some algorithms (FF, PEPITA, TargetProp, PCN) inherently require model-level
control and are registered as models, not propagators. Querying them via
``Registry.get(ComponentCategory.CREDIT_ASSIGNMENT, "pepita")`` resolves through the
compatibility alias map to the model-side registration
``(Registry.get(ComponentCategory.MODEL, "pepita"))`` — no ``ValueError`` raised.
"""

__version__ = "1.0.0"

# Eager imports for core API (always available)
from bioplausible.core.ontology import (
    # Core 5-D ontology
    System,
    DigitalSubstrate,
    AnalogSubstrate,
    MemristiveSubstrate,
    NeuromorphicSubstrate,
    OpticalSubstrate,
    QuantumSubstrate,
    FeedforwardGeometry,
    RecurrentGeometry,
    TileGeometry,
    GeometryConfig,
    InstantaneousDynamics,
    EnergyMinimizationDynamics,
    PredictiveSettlingDynamics,
    SpikeIntegrationDynamics,
    StateDynamicsConfig,
    BackpropCredit,
    ThermodynamicContrast,
    ThermodynamicContrastCredit,  # alias
    RandomProjectionsCredit,
    LocalGoodnessCredit,
    TemporalTraceCredit,
    TargetInversionCredit,
    CreditAssignmentConfig,
    EuclideanUpdate,
    RiemannianOrthogonalUpdate,
    SpectralConstrainedUpdate,
    NaturalGradientUpdate,
    ElasticConsolidationUpdate,
    ParameterUpdateConfig,
    SubstrateConfig,
    SystemState,
    SystemConfig,
)

from bioplausible.core.plasticity import (
    # Plasticity primitives
    NullPlasticity,
    PlasticityConfig,
    RoutingPlasticity,
    FastWeightPlasticity,
    SubstrateCoupledPlasticity,
    RuleStatePlasticity,
)

from bioplausible.core.joint import (
    # Joint 6-D architecture
    CompositeState,
    StateRegistry,
    SystemContext,
    CoupledTransition,
)

from bioplausible.core.presets import (
    # Preset factories (5-D)
    create_backprop_mlp,
    create_eqprop_mlp,
    create_fa_mlp,
    # Preset factories (6-D)
    create_routing_mlp,
    create_fast_weight_mlp,
)

from bioplausible.core.system_trainer import (
    # System composition (5-D and 6-D)
    compose_system,
    compose_system_from_configs,
    extract_config,
    compose_joint_system,
    compose_joint_system_from_configs,
    # Core system factories
    create_backprop_system,
    create_eqprop_system,
    create_fa_system,
    # Trainers
    SystemTrainer,
    SystemTrainerConfig,
)

# Lazy imports for heavy dependencies (zoo, experiment, config)
# Name -> (submodule_path, attr_or_None). attr None returns the submodule itself.
_LAZY: dict[str, tuple[str, str | None]] = {
    "ExperimentConfig": ("bioplausible.config.experiment", "ExperimentConfig"),
    "ModelConfig": ("bioplausible.config.experiment", "ModelConfig"),
    "TrainingConfig": ("bioplausible.config.experiment", "TrainingConfig"),
    "DataConfig": ("bioplausible.config.experiment", "DataConfig"),
    "HardwareConfig": ("bioplausible.config.experiment", "HardwareConfig"),
    "OntologyConfig": ("bioplausible.config.experiment", "OntologyConfig"),
    "make_vision_preset": ("bioplausible.config.experiment", "make_vision_preset"),
    "make_lm_preset": ("bioplausible.config.experiment", "make_lm_preset"),
    "make_graph_preset": (
        "bioplausible.config.experiment",
        "make_graph_preset",
    ),
    "make_rl_preset": ("bioplausible.config.experiment", "make_rl_preset"),
    "make_timeseries_preset": (
        "bioplausible.config.experiment",
        "make_timeseries_preset",
    ),
    "native_eqprop_mlp": (
        "bioplausible.models.native.eqprop_native",
        "native_eqprop_mlp",
    ),
    "native_backprop_mlp": (
        "bioplausible.models.native.backprop_native",
        "native_backprop_mlp",
    ),
    "native_fa_mlp": (
        "bioplausible.models.native.fa_native",
        "native_fa_mlp",
    ),
    "native_pepita_mlp": (
        "bioplausible.models.native.pepita_native",
        "native_pepita_mlp",
    ),
    "native_tile_ep": (
        "bioplausible.models.native.tile_native",
        "native_tile_ep",
    ),
    "native_tile_fa": (
        "bioplausible.models.native.tile_native",
        "native_tile_fa",
    ),
    "native_tile_tp": (
        "bioplausible.models.native.tile_native",
        "native_tile_tp",
    ),
    "native_tile_snn": (
        "bioplausible.models.native.tile_native",
        "native_tile_snn",
    ),
    "muon_backprop": ("bioplausible.zoo.mep.presets", "muon_backprop"),
    "smep": ("bioplausible.zoo.mep.presets", "smep"),
    "smep_fast": ("bioplausible.zoo.mep.presets", "smep_fast"),
}

__all__ = [
    "__version__",
    # Core 5-D ontology
    "System",
    "DigitalSubstrate",
    "AnalogSubstrate",
    "MemristiveSubstrate",
    "NeuromorphicSubstrate",
    "OpticalSubstrate",
    "QuantumSubstrate",
    "FeedforwardGeometry",
    "RecurrentGeometry",
    "TileGeometry",
    "GeometryConfig",
    "InstantaneousDynamics",
    "EnergyMinimizationDynamics",
    "PredictiveSettlingDynamics",
    "SpikeIntegrationDynamics",
    "StateDynamicsConfig",
    "BackpropCredit",
    "ThermodynamicContrast",
    "ThermodynamicContrastCredit",
    "RandomProjectionsCredit",
    "LocalGoodnessCredit",
    "TemporalTraceCredit",
    "TargetInversionCredit",
    "CreditAssignmentConfig",
    "EuclideanUpdate",
    "RiemannianOrthogonalUpdate",
    "SpectralConstrainedUpdate",
    "NaturalGradientUpdate",
    "ElasticConsolidationUpdate",
    "ParameterUpdateConfig",
    "SubstrateConfig",
    "SystemState",
    "SystemConfig",
    # 5-D factories
    "compose_system",
    "compose_system_from_configs",
    "extract_config",
    "create_backprop_system",
    "create_eqprop_system",
    "create_fa_system",
    # Joint 6-D architecture
    "CompositeState",
    "StateRegistry",
    "SystemContext",
    "CoupledTransition",
    "NullPlasticity",
    "PlasticityConfig",
    "RoutingPlasticity",
    "FastWeightPlasticity",
    "SubstrateCoupledPlasticity",
    "RuleStatePlasticity",
    # 6-D factories
    "compose_joint_system",
    "compose_joint_system_from_configs",
    # Preset factories (5-D)
    "create_backprop_mlp",
    "create_eqprop_mlp",
    "create_fa_mlp",
    # Preset factories (6-D)
    "create_routing_mlp",
    "create_fast_weight_mlp",
    # Trainers
    "SystemTrainer",
    "SystemTrainerConfig",
]


# ruff: file-ignore[raise-vanilla-args]
def __getattr__(name: str) -> object:
    """Lazily import a top-level symbol on first access."""
    if name not in _LAZY:
        raise AttributeError("cannot find")
    module_name, attr = _LAZY[name]
    module = __import__(module_name, fromlist=[attr] if attr else ["*"])
    value: object = module if attr is None else getattr(module, attr)
    setattr(__import__(__name__), name, value)  # cache on the module
    return value


def __dir__() -> list[str]:
    return sorted(__all__)