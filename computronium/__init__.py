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

2. Model side (`computronium.models`): Learning rules that require
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
    # 6-D Joint Architecture (facade exports only)
    "CompositeState": ("computronium.state", "CompositeState"),
    "CoupledTransition": ("computronium.core.joint.transition", "CoupledTransition"),
    "StateRegistry": ("computronium.state", "StateRegistry"),
    "Registry": ("computronium.core.registry", "Registry"),
    "SystemContext": ("computronium.state", "SystemContext"),
    # Core 5-D Ontology (new decomposed modules)
    "AnalogSubstrate": ("computronium.ontology.substrate", "AnalogSubstrate"),
    "BackpropCredit": ("computronium.ontology.credit", "BackpropCredit"),
    "CreditAssignmentConfig": (
        "computronium.ontology.credit",
        "CreditAssignmentConfig",
    ),
    "DigitalSubstrate": ("computronium.ontology.substrate", "DigitalSubstrate"),
    "ElasticConsolidationUpdate": (
        "computronium.ontology.update",
        "ElasticConsolidationUpdate",
    ),
    "EnergyMinimizationDynamics": (
        "computronium.ontology.dynamics",
        "EnergyMinimizationDynamics",
    ),
    "EuclideanUpdate": ("computronium.ontology.update", "EuclideanUpdate"),
    "FeedforwardGeometry": ("computronium.ontology.geometry", "FeedforwardGeometry"),
    "GeometryConfig": ("computronium.ontology.geometry", "GeometryConfig"),
    "InstantaneousDynamics": (
        "computronium.ontology.dynamics",
        "InstantaneousDynamics",
    ),
    "LocalGoodnessCredit": ("computronium.ontology.credit", "LocalGoodnessCredit"),
    "MemristiveSubstrate": ("computronium.ontology.substrate", "MemristiveSubstrate"),
    "NaturalGradientUpdate": ("computronium.ontology.update", "NaturalGradientUpdate"),
    "NeuromorphicSubstrate": (
        "computronium.ontology.substrate",
        "NeuromorphicSubstrate",
    ),
    "OpticalSubstrate": ("computronium.ontology.substrate", "OpticalSubstrate"),
    "ParameterUpdateConfig": ("computronium.ontology.update", "ParameterUpdateConfig"),
    "PredictiveSettlingDynamics": (
        "computronium.ontology.dynamics",
        "PredictiveSettlingDynamics",
    ),
    "QuantumSubstrate": ("computronium.ontology.substrate", "QuantumSubstrate"),
    "QuantizedSubstrate": ("computronium.ontology.substrate", "QuantizedSubstrate"),
    "RandomProjectionsCredit": (
        "computronium.ontology.credit",
        "RandomProjectionsCredit",
    ),
    "RecurrentGeometry": ("computronium.ontology.geometry", "RecurrentGeometry"),
    "RiemannianOrthogonalUpdate": (
        "computronium.ontology.update",
        "RiemannianOrthogonalUpdate",
    ),
    "SpectralConstrainedUpdate": (
        "computronium.ontology.update",
        "SpectralConstrainedUpdate",
    ),
    "SpikeIntegrationDynamics": (
        "computronium.ontology.dynamics",
        "SpikeIntegrationDynamics",
    ),
    "StateDynamicsConfig": ("computronium.ontology.dynamics", "StateDynamicsConfig"),
    "SubstrateConfig": ("computronium.ontology.substrate", "SubstrateConfig"),
    "System": ("computronium.ontology.system", "System"),
    "SystemConfig": ("computronium.ontology.system", "SystemConfig"),
    "SystemState": ("computronium.ontology.system", "SystemState"),
    "TargetInversionCredit": ("computronium.ontology.credit", "TargetInversionCredit"),
    "TemporalTraceCredit": ("computronium.ontology.credit", "TemporalTraceCredit"),
    "ThermodynamicContrast": ("computronium.ontology.credit", "ThermodynamicContrast"),
    "ThermodynamicContrastCredit": (
        "computronium.ontology.credit",
        "ThermodynamicContrastCredit",
    ),
    "TileGeometry": ("computronium.ontology.geometry", "TileGeometry"),
    # Plasticity Primitives (from computronium.state)
    "FastWeightPlasticity": (
        "computronium.ontology.plasticity",
        "FastWeightPlasticity",
    ),
    # core.plasticity.NullPlasticity ≡ core.joint.transition.NullPlasticity —
    # the class compose_joint_system special-cases into the 5-D delegation path.
    "NullPlasticity": ("computronium.core.plasticity", "NullPlasticity"),
    "PlasticityConfig": ("computronium.state", "PlasticityConfig"),
    "RoutingPlasticity": ("computronium.ontology.plasticity", "RoutingPlasticity"),
    "RuleStatePlasticity": ("computronium.ontology.plasticity", "RuleStatePlasticity"),
    "SubstrateCoupledPlasticity": (
        "computronium.ontology.plasticity",
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
    # Domains
    "create_task": ("computronium.domains.factory", "create_task"),
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
        "computronium.models.native",
        "native_eqprop_mlp",
    ),
    "native_backprop_mlp": (
        "computronium.models.native",
        "native_backprop_mlp",
    ),
    "native_fa_mlp": (
        "computronium.models.native",
        "native_fa_mlp",
    ),
    "native_pepita_mlp": (
        "computronium.models.native",
        "native_pepita_mlp",
    ),
    "native_tile_ep": (
        "computronium.models.native",
        "native_tile_ep",
    ),
    "native_tile_fa": (
        "computronium.models.native",
        "native_tile_fa",
    ),
    "native_tile_tp": (
        "computronium.models.native",
        "native_tile_tp",
    ),
    "native_tile_snn": (
        "computronium.models.native",
        "native_tile_snn",
    ),
    "native_diffusion_eqprop": (
        "computronium.models.native",
        "native_diffusion_eqprop",
    ),
    "native_momentum_eqprop": (
        "computronium.models.native",
        "native_momentum_eqprop",
    ),
    "native_sparse_eqprop": (
        "computronium.models.native",
        "native_sparse_eqprop",
    ),
    "native_ternary_eqprop": (
        "computronium.models.native",
        "native_ternary_eqprop",
    ),
    "native_holomorphic_ep": (
        "computronium.models.native",
        "native_holomorphic_ep",
    ),
    "native_directed_ep": (
        "computronium.models.native",
        "native_directed_ep",
    ),
    "native_finite_nudge_ep": (
        "computronium.models.native",
        "native_finite_nudge_ep",
    ),
    # MEP Presets
    "muon_backprop": ("computronium.mep.presets", "muon_backprop"),
    "smep": ("computronium.mep.presets", "smep"),
    "smep_fast": ("computronium.mep.presets", "smep_fast"),
    # NN Layers (CP-C)
    "ComputroniumLinear": ("computronium.nn", "ComputroniumLinear"),
    "replace_linear_with_computronium": (
        "computronium.nn",
        "replace_linear_with_computronium",
    ),
    "CreditRule": ("computronium.nn", "CreditRule"),
    "CreditRuleConfig": ("computronium.nn", "CreditRuleConfig"),
    "PlasticityType": ("computronium.nn", "PlasticityType"),
}

__all__ = [
    "AnalogSubstrate",
    "BackpropCredit",
    "CompositeState",
    "ComputroniumLinear",
    "CoupledTransition",
    "CreditAssignmentConfig",
    "CreditRule",
    "CreditRuleConfig",
    "DataConfig",
    "DigitalSubstrate",
    "ElasticConsolidationUpdate",
    "EnergyMinimizationDynamics",
    "EuclideanUpdate",
    "ExperimentConfig",
    "FastWeightPlasticity",
    "FeedforwardGeometry",
    "GeometryConfig",
    "HardwareConfig",
    "InstantaneousDynamics",
    "LocalGoodnessCredit",
    "MemristiveSubstrate",
    "ModelConfig",
    "NaturalGradientUpdate",
    "NeuromorphicSubstrate",
    "NullPlasticity",
    "OpticalSubstrate",
    "ParameterUpdateConfig",
    "PlasticityConfig",
    "PlasticityType",
    "PredictiveSettlingDynamics",
    "QuantizedSubstrate",
    "QuantumSubstrate",
    "RandomProjectionsCredit",
    "RecurrentGeometry",
    "Registry",
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
    "TrainingConfig",
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
    "create_pc_mlp",
    "create_pepita_mlp",
    "create_routing_mlp",
    "create_snn_mlp",
    "create_spiking_snn_mlp",
    "create_task",
    "create_tile_mlp",
    "create_tp_mlp",
    "extract_config",
    "make_graph_preset",
    "make_lm_preset",
    "make_rl_preset",
    "make_timeseries_preset",
    "make_vision_preset",
    "muon_backprop",
    "native_backprop_mlp",
    "native_diffusion_eqprop",
    "native_directed_ep",
    "native_eqprop_mlp",
    "native_fa_mlp",
    "native_finite_nudge_ep",
    "native_holomorphic_ep",
    "native_momentum_eqprop",
    "native_pepita_mlp",
    "native_sparse_eqprop",
    "native_ternary_eqprop",
    "native_tile_ep",
    "native_tile_fa",
    "native_tile_snn",
    "native_tile_tp",
    "replace_linear_with_computronium",
    "smep",
    "smep_fast",
]


# ruff: file-ignore[raise-vanilla-args, RUF022]
def __getattr__(name: str) -> object:
    """Lazily import a top-level symbol on first access.

    Attribute access also triggers deferred Registry population so light
    submodule imports (e.g. ``computronium.core.registry``) stay torch-free
    while top-level consumers see a fully populated Registry.
    """
    from computronium.core.registry import _ensure_native_registered

    _ensure_native_registered()
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = _LAZY[name]
    module = __import__(module_name, fromlist=[attr] if attr else ["*"])
    value: object = module if attr is None else getattr(module, attr)
    setattr(__import__(__name__), name, value)  # cache on the module
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
