"""
Bioplausible: Unified Platform for Bio-Plausible Learning Research

Minimal, clean API for training and experimentation.

Quick Start:
    from computronium import SystemTrainer, ExperimentConfig

    config = ExperimentConfig(...)
    trainer = SystemTrainer.from_configs(config, train_data, val_data)
    history = trainer.fit()

Or using native models:
    from computronium import native_eqprop_mlp, native_backprop_mlp, native_fa_mlp

    system = native_eqprop_mlp(input_dim=784, hidden_dim=256, output_dim=10)
    # Use with SystemTrainer

One-Line System Construction:
    from computronium import (
        create_backprop_mlp,
        create_eqprop_mlp,
        create_fa_mlp,
        create_ff_mlp,
        create_pepita_mlp,
        create_tp_mlp,
        create_pc_mlp,
        create_hebbian_mlp,
        create_snn_mlp,
        create_spiking_snn_mlp,
        create_routing_mlp,
        create_fast_weight_mlp,
    )

    system = create_backprop_mlp(
        input_dim=784, hidden_dims=(256, 128), output_dim=10
    )
    system = create_eqprop_mlp(
        input_dim=784, hidden_dims=(512, 512, 512), output_dim=10, beta=0.1, inference_steps=20
    )
    system = create_fa_mlp(
        input_dim=784, hidden_dims=(256, 128), output_dim=10
    )
    system = create_ff_mlp(
        input_dim=784, hidden_dims=(256, 128), output_dim=10
    )
    system = create_pepita_mlp(
        input_dim=784, hidden_dims=(256, 128), output_dim=10
    )
    system = create_tp_mlp(
        input_dim=784, hidden_dims=(256, 128), output_dim=10
    )
    system = create_pc_mlp(
        input_dim=784, hidden_dims=(256, 128), output_dim=10
    )
    system = create_hebbian_mlp(
        input_dim=784, hidden_dims=(256, 128), output_dim=10
    )
    system = create_snn_mlp(
        input_dim=784, hidden_dims=(256, 128), output_dim=10
    )
    system = create_spiking_snn_mlp(
        input_dim=784, hidden_dims=(256, 128), output_dim=10
    )
    system = create_routing_mlp(
        input_dim=784, hidden_dims=(256, 128), output_dim=10
    )
    system = create_fast_weight_mlp(
        input_dim=784, hidden_dims=(256, 128), output_dim=10
    )

Two-Tier Propagator / Model Architecture:
----------------------------------------
The zoo provides two complementary interfaces for bio-plausible learning:

1. Learning rules (``computronium.core.local_learning.rules``): Learning rules
   implemented as drop-in ``torch.optim.Optimizer`` subclasses
   (`BioOptimizer`, `LearningRuleOptimizer`). These mutate parameters of any
   model: Backprop, FeedbackAlignment, EqProp, ContrastiveHebbianLearning,
   MEP presets (smep, sdmep, ...). Use via the Registry API:
   ``Registry.get(ComponentCategory.PARAM_UPDATE, "eq_prop")``.

2. Model side (`computronium.zoo.models`): Learning rules that require
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

# Lazy imports for heavy dependencies (zoo, experiment, config, core components)
# Name -> (submodule_path, attr_or_None). attr None returns the submodule itself.
_LAZY: dict[str, tuple[str, str | None]] = {
    # 6-D Joint Architecture
    "CompositeState": ("computronium.core.joint.state", "CompositeState"),
    "CoupledTransition": ("computronium.core.joint.transition", "CoupledTransition"),
    "StateRegistry": ("computronium.core.joint.state", "StateRegistry"),
    "SystemContext": ("computronium.core.joint.context", "SystemContext"),
    # Core 5-D Ontology
    "AnalogSubstrate": ("computronium.core.ontology", "AnalogSubstrate"),
    "BackpropCredit": ("computronium.core.ontology", "BackpropCredit"),
    "CreditAssignmentConfig": ("computronium.core.ontology", "CreditAssignmentConfig"),
    "DigitalSubstrate": ("computronium.core.ontology", "DigitalSubstrate"),
    "ElasticConsolidationUpdate": (
        "computronium.core.ontology",
        "ElasticConsolidationUpdate",
    ),
    "EnergyMinimizationDynamics": (
        "computronium.core.ontology",
        "EnergyMinimizationDynamics",
    ),
    "EuclideanUpdate": ("computronium.core.ontology", "EuclideanUpdate"),
    "FeedforwardGeometry": ("computronium.core.ontology", "FeedforwardGeometry"),
    "GeometryConfig": ("computronium.core.ontology", "GeometryConfig"),
    "InstantaneousDynamics": ("computronium.core.ontology", "InstantaneousDynamics"),
    "LocalGoodnessCredit": ("computronium.core.ontology", "LocalGoodnessCredit"),
    "MemristiveSubstrate": ("computronium.core.ontology", "MemristiveSubstrate"),
    "NaturalGradientUpdate": ("computronium.core.ontology", "NaturalGradientUpdate"),
    "NeuromorphicSubstrate": ("computronium.core.ontology", "NeuromorphicSubstrate"),
    "OpticalSubstrate": ("computronium.core.ontology", "OpticalSubstrate"),
    "ParameterUpdateConfig": ("computronium.core.ontology", "ParameterUpdateConfig"),
    "PredictiveSettlingDynamics": (
        "computronium.core.ontology",
        "PredictiveSettlingDynamics",
    ),
    "QuantumSubstrate": ("computronium.core.ontology", "QuantumSubstrate"),
    "RandomProjectionsCredit": (
        "computronium.core.ontology",
        "RandomProjectionsCredit",
    ),
    "RecurrentGeometry": ("computronium.core.ontology", "RecurrentGeometry"),
    "RiemannianOrthogonalUpdate": (
        "computronium.core.ontology",
        "RiemannianOrthogonalUpdate",
    ),
    "SpectralConstrainedUpdate": (
        "computronium.core.ontology",
        "SpectralConstrainedUpdate",
    ),
    "SpikeIntegrationDynamics": (
        "computronium.core.ontology",
        "SpikeIntegrationDynamics",
    ),
    "StateDynamicsConfig": ("computronium.core.ontology", "StateDynamicsConfig"),
    "SubstrateConfig": ("computronium.core.ontology", "SubstrateConfig"),
    "System": ("computronium.core.ontology", "System"),
    "SystemConfig": ("computronium.core.ontology", "SystemConfig"),
    "SystemState": ("computronium.core.ontology", "SystemState"),
    "TargetInversionCredit": ("computronium.core.ontology", "TargetInversionCredit"),
    "TemporalTraceCredit": ("computronium.core.ontology", "TemporalTraceCredit"),
    "ThermodynamicContrast": ("computronium.core.ontology", "ThermodynamicContrast"),
    "ThermodynamicContrastCredit": (
        "computronium.core.ontology",
        "ThermodynamicContrastCredit",
    ),
    "TileGeometry": ("computronium.core.ontology", "TileGeometry"),
    # Plasticity Primitives
    "FastWeightPlasticity": ("computronium.core.plasticity", "FastWeightPlasticity"),
    "NullPlasticity": ("computronium.core.plasticity", "NullPlasticity"),
    "PlasticityConfig": ("computronium.core.plasticity", "PlasticityConfig"),
    "RoutingPlasticity": ("computronium.core.plasticity", "RoutingPlasticity"),
    "RuleStatePlasticity": ("computronium.core.plasticity", "RuleStatePlasticity"),
    "SubstrateCoupledPlasticity": (
        "computronium.core.plasticity",
        "SubstrateCoupledPlasticity",
    ),
    # Preset Factories (5-D)
    "create_backprop_mlp": ("computronium.core.presets", "create_backprop_mlp"),
    "create_eqprop_mlp": ("computronium.core.presets", "create_eqprop_mlp"),
    "create_fa_mlp": ("computronium.core.presets", "create_fa_mlp"),
    "create_ff_mlp": ("computronium.core.presets", "create_ff_mlp"),
    "create_pepita_mlp": ("computronium.core.presets", "create_pepita_mlp"),
    "create_tp_mlp": ("computronium.core.presets", "create_tp_mlp"),
    "create_pc_mlp": ("computronium.core.presets", "create_pc_mlp"),
    "create_hebbian_mlp": ("computronium.core.presets", "create_hebbian_mlp"),
    "create_snn_mlp": ("computronium.core.presets", "create_snn_mlp"),
    "create_spiking_snn_mlp": ("computronium.core.presets", "create_spiking_snn_mlp"),
    "create_tile_mlp": ("computronium.core.presets", "create_tile_mlp"),
    # Preset Factories (6-D)
    "create_routing_mlp": ("computronium.core.presets", "create_routing_mlp"),
    "create_fast_weight_mlp": ("computronium.core.presets", "create_fast_weight_mlp"),
    # System Trainers
    "SystemTrainer": ("computronium.core.system_trainer", "SystemTrainer"),
    "SystemTrainerConfig": ("computronium.core.system_trainer", "SystemTrainerConfig"),
    "compose_joint_system": (
        "computronium.core.system_trainer",
        "compose_joint_system",
    ),
    "compose_joint_system_from_configs": (
        "computronium.core.system_trainer",
        "compose_joint_system_from_configs",
    ),
    "compose_system": ("computronium.core.system_trainer", "compose_system"),
    "compose_system_from_configs": (
        "computronium.core.system_trainer",
        "compose_system_from_configs",
    ),
    "create_backprop_system": (
        "computronium.core.system_trainer",
        "create_backprop_system",
    ),
    "create_eqprop_system": (
        "computronium.core.system_trainer",
        "create_eqprop_system",
    ),
    "create_fa_system": ("computronium.core.system_trainer", "create_fa_system"),
    "extract_config": ("computronium.core.system_trainer", "extract_config"),
    # Config/Experiment
    "ExperimentConfig": ("computronium.config.experiment", "ExperimentConfig"),
    "ModelConfig": ("computronium.config.experiment", "ModelConfig"),
    "TrainingConfig": ("computronium.config.experiment", "TrainingConfig"),
    "DataConfig": ("computronium.config.experiment", "DataConfig"),
    "HardwareConfig": ("computronium.config.experiment", "HardwareConfig"),
    "OntologyConfig": ("computronium.config.experiment", "OntologyConfig"),
    "make_vision_preset": ("computronium.config.experiment", "make_vision_preset"),
    "make_lm_preset": ("computronium.config.experiment", "make_lm_preset"),
    "make_graph_preset": (
        "computronium.config.experiment",
        "make_graph_preset",
    ),
    "make_rl_preset": ("computronium.config.experiment", "make_rl_preset"),
    "make_timeseries_preset": (
        "computronium.config.experiment",
        "make_timeseries_preset",
    ),
    # Native Models
    "native_eqprop_mlp": (
        "computronium.models.native.eqprop_native",
        "native_eqprop_mlp",
    ),
    "native_backprop_mlp": (
        "computronium.models.native.backprop_native",
        "native_backprop_mlp",
    ),
    "native_fa_mlp": (
        "computronium.models.native.fa_native",
        "native_fa_mlp",
    ),
    "native_pepita_mlp": (
        "computronium.models.native.pepita_native",
        "native_pepita_mlp",
    ),
    "native_tile_ep": (
        "computronium.models.native.tile_native",
        "native_tile_ep",
    ),
    "native_tile_fa": (
        "computronium.models.native.tile_native",
        "native_tile_fa",
    ),
    "native_tile_tp": (
        "computronium.models.native.tile_native",
        "native_tile_tp",
    ),
    "native_tile_snn": (
        "computronium.models.native.tile_native",
        "native_tile_snn",
    ),
    # MEP Presets
    "muon_backprop": ("computronium.zoo.mep.presets", "muon_backprop"),
    "smep": ("computronium.zoo.mep.presets", "smep"),
    "smep_fast": ("computronium.zoo.mep.presets", "smep_fast"),
}

__all__ = [
    "AnalogSubstrate",
    "BackpropCredit",
    "CompositeState",
    "CoupledTransition",
    "CreditAssignmentConfig",
    "DigitalSubstrate",
    "ElasticConsolidationUpdate",
    "EnergyMinimizationDynamics",
    "EuclideanUpdate",
    "FastWeightPlasticity",
    "FeedforwardGeometry",
    "GeometryConfig",
    "InstantaneousDynamics",
    "LocalGoodnessCredit",
    "MemristiveSubstrate",
    "NaturalGradientUpdate",
    "NeuromorphicSubstrate",
    "NullPlasticity",
    "OpticalSubstrate",
    "ParameterUpdateConfig",
    "PlasticityConfig",
    "PredictiveSettlingDynamics",
    "QuantumSubstrate",
    "RandomProjectionsCredit",
    "RecurrentGeometry",
    "RiemannianOrthogonalUpdate",
    "RoutingPlasticity",
    "RuleStatePlasticity",
    "SpectralConstrainedUpdate",
    "SpikeIntegrationDynamics",
    "StateDynamicsConfig",
    "StateRegistry",
    "SubstrateConfig",
    "SubstrateCoupledPlasticity",
    "System",
    "SystemConfig",
    "SystemContext",
    "SystemState",
    "SystemTrainer",
    "SystemTrainerConfig",
    "TargetInversionCredit",
    "TemporalTraceCredit",
    "ThermodynamicContrast",
    "ThermodynamicContrastCredit",
    "TileGeometry",
    "__version__",
    "compose_joint_system",
    "compose_joint_system_from_configs",
    "compose_system",
    "compose_system_from_configs",
    "create_backprop_mlp",
    "create_backprop_system",
    "create_eqprop_mlp",
    "create_eqprop_system",
    "create_fa_mlp",
    "create_fa_system",
    "create_fast_weight_mlp",
    "create_ff_mlp",
    "create_hebbian_mlp",
    "create_pepita_mlp",
    "create_pc_mlp",
    "create_routing_mlp",
    "create_snn_mlp",
    "create_tp_mlp",
    "extract_config",
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
