"""Core package: Registry, CoreTrainer, Config, Model."""

# Lazy package init (Sprint 0.5): `import computronium.core.registry` must NOT
# pull the zoo. `core.registry` depends only on stdlib + exceptions, but the
# eager `core.trainer`/`core.model` imports drag in `zoo` via
# `is_learning_rule_optimizer`. Expose symbols on demand so light consumers stay
# fast; heavy symbols (CoreTrainer) import the zoo on first access.

_LAZY: dict[str, tuple[str, str | None]] = {
    "BioModel": ("computronium.core.model", "BioModel"),
    "LayerRole": ("computronium.config.unified", "LayerRole"),
    "ModelConfig": ("computronium.config.unified", "ModelConfig"),
    "compute_hidden_dims": ("computronium.config.unified", "compute_hidden_dims"),
    "resolve_hidden_dims": ("computronium.config.unified", "resolve_hidden_dims"),
    "ComponentCategory": ("computronium.core.registry", "ComponentCategory"),
    "ComponentMetadata": ("computronium.core.registry", "ComponentMetadata"),
    "ComputeProfile": ("computronium.core.registry", "ComputeProfile"),
    "LocalityLevel": ("computronium.core.registry", "LocalityLevel"),
    "Registry": ("computronium.core.registry", "Registry"),
    "register_constraint": ("computronium.core.registry", "register_constraint"),
    "register_controller": ("computronium.core.registry", "register_controller"),
    "register_credit_assignment": (
        "computronium.core.registry",
        "register_credit_assignment",
    ),
    "register_hardware": ("computronium.core.registry", "register_hardware"),
    "register_metric": ("computronium.core.registry", "register_metric"),
    "register_model": ("computronium.core.registry", "register_model"),
    "register_optimizer": ("computronium.core.registry", "register_optimizer"),
    "register_param_update": ("computronium.core.registry", "register_param_update"),
    "register_propagator": ("computronium.core.registry", "register_propagator"),
    "register_sparsity": ("computronium.core.registry", "register_sparsity"),
    "register_task": ("computronium.core.registry", "register_task"),
    "register_track": ("computronium.core.registry", "register_track"),
    "register_update_strategy": (
        "computronium.core.registry",
        "register_update_strategy",
    ),
    "CoreTrainer": ("computronium.core.trainer", "CoreTrainer"),
    "TrainerConfig": ("computronium.core.trainer", "TrainerConfig"),
    "TrainingMetrics": ("computronium.core.trainer", "TrainingMetrics"),
    "TrainingMixin": ("computronium.core.training_mixin", "TrainingMixin"),
    "SpectralMixin": ("computronium.core.spectral_mixin", "SpectralMixin"),
    "CheckpointMixin": ("computronium.core.checkpoint_mixin", "CheckpointMixin"),
    "BaseMetrics": ("computronium.core.metrics", "BaseMetrics"),
    "EpochMetrics": ("computronium.core.metrics", "EpochMetrics"),
    # Ontology (5-D tensor product) - NEW decomposed modules
    "Substrate": ("computronium.ontology.substrate", "Substrate"),
    "Geometry": ("computronium.ontology.geometry", "Geometry"),
    "StateDynamics": ("computronium.ontology.dynamics", "StateDynamics"),
    "CreditAssignment": ("computronium.ontology.credit", "CreditAssignment"),
    "ParameterUpdate": ("computronium.ontology.update", "ParameterUpdate"),
    "System": ("computronium.ontology.system", "System"),
    "SystemConfig": ("computronium.ontology.system", "SystemConfig"),
    "SystemState": ("computronium.ontology.system", "SystemState"),
    "SubstrateConfig": ("computronium.ontology.substrate", "SubstrateConfig"),
    "GeometryConfig": ("computronium.ontology.geometry", "GeometryConfig"),
    "StateDynamicsConfig": ("computronium.ontology.dynamics", "StateDynamicsConfig"),
    "CreditAssignmentConfig": ("computronium.ontology.credit", "CreditAssignmentConfig"),
    "ParameterUpdateConfig": ("computronium.ontology.update", "ParameterUpdateConfig"),
    # Reference implementations
    "DigitalSubstrate": ("computronium.ontology.substrate", "DigitalSubstrate"),
    "FeedforwardGeometry": ("computronium.ontology.geometry", "FeedforwardGeometry"),
    "RecurrentGeometry": ("computronium.ontology.geometry", "RecurrentGeometry"),
    "TileGeometry": ("computronium.ontology.geometry", "TileGeometry"),
    "InstantaneousDynamics": ("computronium.ontology.dynamics", "InstantaneousDynamics"),
    "ThermodynamicContrast": ("computronium.ontology.credit", "ThermodynamicContrast"),
    "EuclideanUpdate": ("computronium.ontology.update", "EuclideanUpdate"),
    "ModelAdapter": ("computronium.ontology.system", "ModelAdapter"),
    # Hardware substrates
    "AnalogSubstrate": ("computronium.ontology.substrate", "AnalogSubstrate"),
    "MemristiveSubstrate": ("computronium.ontology.substrate", "MemristiveSubstrate"),
    "NeuromorphicSubstrate": ("computronium.ontology.substrate", "NeuromorphicSubstrate"),
    "OpticalSubstrate": ("computronium.ontology.substrate", "OpticalSubstrate"),
    "QuantumSubstrate": ("computronium.ontology.substrate", "QuantumSubstrate"),
    "QuantizedSubstrate": ("computronium.ontology.substrate", "QuantizedSubstrate"),
    "NoisySubstrate": ("computronium.ontology.substrate", "NoisySubstrate"),
    # SystemTrainer
    "SystemTrainerConfig": ("computronium.core.system_trainer", "SystemTrainerConfig"),
    "SystemTrainer": ("computronium.core.system_trainer", "SystemTrainer"),
    "compose_system": ("computronium.core.system_trainer", "compose_system"),
    "create_eqprop_system": (
        "computronium.core.system_trainer",
        "create_eqprop_system",
    ),
    "create_backprop_system": (
        "computronium.core.system_trainer",
        "create_backprop_system",
    ),
    "create_fa_system": ("computronium.core.system_trainer", "create_fa_system"),
    # Distributed Trainer
    "DistributedConfig": ("computronium.core.distributed_trainer", "DistributedConfig"),
    "DistributedSystemTrainer": (
        "computronium.core.distributed_trainer",
        "DistributedSystemTrainer",
    ),
    "NodeRegistry": ("computronium.core.distributed_trainer", "NodeRegistry"),
    "DHTRouter": ("computronium.core.distributed_trainer", "DHTRouter"),
    "FederatedAggregator": (
        "computronium.core.distributed_trainer",
        "FederatedAggregator",
    ),
    # Joint Architecture (6-D tensor product: S ⊗ G ⊗ D ⊗ M ⊗ C ⊗ U)
    "StateVariable": ("computronium.state", "StateVariable"),
    "StateRegistry": ("computronium.state", "StateRegistry"),
    "CompositeState": ("computronium.state", "CompositeState"),
    "JointTrajectoryRecorder": ("computronium.core.joint", "JointTrajectoryRecorder"),
    "SystemContext": ("computronium.state", "SystemContext"),
    "CoupledTransition": ("computronium.core.joint.transition", "CoupledTransition"),
    "PlasticityPrimitive": ("computronium.state", "PlasticityPrimitive"),
    "PlasticityConfig": ("computronium.state", "PlasticityConfig"),
    "NullPlasticity": ("computronium.state", "NullPlasticity"),
    "LegacyDynamicsAsCoupledTransition": (
        "computronium.core.joint",
        "LegacyDynamicsAsCoupledTransition",
    ),
    "JointTrajectory": ("computronium.core.joint", "JointTrajectory"),
    "ConsolidationConfig": ("computronium.core.joint", "ConsolidationConfig"),
    "consolidate": ("computronium.core.joint", "consolidate"),
}

__all__ = sorted(_LAZY)


def __getattr__(name: str) -> object:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = _LAZY[name]
    module = __import__(module_name, fromlist=[attr] if attr else ["*"])
    value: object = module if attr is None else getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)