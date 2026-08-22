"""Core package: Registry, CoreTrainer, Config, Model."""

# Lazy package init (Sprint 0.5): `import bioplausible.core.registry` must NOT
# pull the zoo. `core.registry` depends only on stdlib + exceptions, but the
# eager `core.trainer`/`core.model` imports drag in `zoo` via
# `is_learning_rule_optimizer`. Expose symbols on demand so light consumers stay
# fast; heavy symbols (CoreTrainer) import the zoo on first access.

_LAZY: dict[str, tuple[str, str | None]] = {
    "BioModel": ("bioplausible.core.model", "BioModel"),
    "LayerRole": ("bioplausible.config.unified", "LayerRole"),
    "ModelConfig": ("bioplausible.config.unified", "ModelConfig"),
    "compute_hidden_dims": ("bioplausible.config.unified", "compute_hidden_dims"),
    "resolve_hidden_dims": ("bioplausible.config.unified", "resolve_hidden_dims"),
    "ComponentCategory": ("bioplausible.core.registry", "ComponentCategory"),
    "ComponentMetadata": ("bioplausible.core.registry", "ComponentMetadata"),
    "ComputeProfile": ("bioplausible.core.registry", "ComputeProfile"),
    "LocalityLevel": ("bioplausible.core.registry", "LocalityLevel"),
    "Registry": ("bioplausible.core.registry", "Registry"),
    "register_constraint": ("bioplausible.core.registry", "register_constraint"),
    "register_controller": ("bioplausible.core.registry", "register_controller"),
    "register_credit_assignment": (
        "bioplausible.core.registry",
        "register_credit_assignment",
    ),
    "register_hardware": ("bioplausible.core.registry", "register_hardware"),
    "register_metric": ("bioplausible.core.registry", "register_metric"),
    "register_model": ("bioplausible.core.registry", "register_model"),
    "register_optimizer": ("bioplausible.core.registry", "register_optimizer"),
    "register_param_update": ("bioplausible.core.registry", "register_param_update"),
    "register_propagator": ("bioplausible.core.registry", "register_propagator"),
    "register_sparsity": ("bioplausible.core.registry", "register_sparsity"),
    "register_task": ("bioplausible.core.registry", "register_task"),
    "register_track": ("bioplausible.core.registry", "register_track"),
    "register_update_strategy": (
        "bioplausible.core.registry",
        "register_update_strategy",
    ),
    "CoreTrainer": ("bioplausible.core.trainer", "CoreTrainer"),
    "TrainerConfig": ("bioplausible.core.trainer", "TrainerConfig"),
    "TrainingMetrics": ("bioplausible.core.trainer", "TrainingMetrics"),
    "TrainingMixin": ("bioplausible.core.training_mixin", "TrainingMixin"),
    "SpectralMixin": ("bioplausible.core.spectral_mixin", "SpectralMixin"),
    "CheckpointMixin": ("bioplausible.core.checkpoint_mixin", "CheckpointMixin"),
    "BaseMetrics": ("bioplausible.core.metrics", "BaseMetrics"),
    "EpochMetrics": ("bioplausible.core.metrics", "EpochMetrics"),
    # Ontology (5-D tensor product)
    "Substrate": ("bioplausible.core.ontology", "Substrate"),
    "Geometry": ("bioplausible.core.ontology", "Geometry"),
    "StateDynamics": ("bioplausible.core.ontology", "StateDynamics"),
    "CreditAssignment": ("bioplausible.core.ontology", "CreditAssignment"),
    "ParameterUpdate": ("bioplausible.core.ontology", "ParameterUpdate"),
    "System": ("bioplausible.core.ontology", "System"),
    "SystemState": ("bioplausible.core.ontology", "SystemState"),
    "SubstrateConfig": ("bioplausible.core.ontology", "SubstrateConfig"),
    "GeometryConfig": ("bioplausible.core.ontology", "GeometryConfig"),
    "StateDynamicsConfig": ("bioplausible.core.ontology", "StateDynamicsConfig"),
    "CreditAssignmentConfig": ("bioplausible.core.ontology", "CreditAssignmentConfig"),
    "ParameterUpdateConfig": ("bioplausible.core.ontology", "ParameterUpdateConfig"),
    # Reference implementations
    "DigitalSubstrate": ("bioplausible.core.ontology", "DigitalSubstrate"),
    "FeedforwardGeometry": ("bioplausible.core.ontology", "FeedforwardGeometry"),
    "RecurrentGeometry": ("bioplausible.core.ontology", "RecurrentGeometry"),
    "TileGeometry": ("bioplausible.core.ontology", "TileGeometry"),
    "InstantaneousDynamics": ("bioplausible.core.ontology", "InstantaneousDynamics"),
    "ThermodynamicContrast": ("bioplausible.core.ontology", "ThermodynamicContrast"),
    "EuclideanUpdate": ("bioplausible.core.ontology", "EuclideanUpdate"),
    "ModelAdapter": ("bioplausible.core.ontology", "ModelAdapter"),
    # Hardware substrates
    "AnalogSubstrate": ("bioplausible.core.ontology", "AnalogSubstrate"),
    "MemristiveSubstrate": ("bioplausible.core.ontology", "MemristiveSubstrate"),
    "NeuromorphicSubstrate": ("bioplausible.core.ontology", "NeuromorphicSubstrate"),
    "OpticalSubstrate": ("bioplausible.core.ontology", "OpticalSubstrate"),
    "QuantumSubstrate": ("bioplausible.core.ontology", "QuantumSubstrate"),
    "QuantizedSubstrate": ("bioplausible.core.ontology", "QuantizedSubstrate"),
    "NoisySubstrate": ("bioplausible.core.ontology", "NoisySubstrate"),
    # SystemTrainer
    "SystemTrainerConfig": ("bioplausible.core.system_trainer", "SystemTrainerConfig"),
    "SystemTrainer": ("bioplausible.core.system_trainer", "SystemTrainer"),
    "compose_system": ("bioplausible.core.system_trainer", "compose_system"),
    "create_eqprop_system": (
        "bioplausible.core.system_trainer",
        "create_eqprop_system",
    ),
    "create_backprop_system": (
        "bioplausible.core.system_trainer",
        "create_backprop_system",
    ),
    "create_fa_system": ("bioplausible.core.system_trainer", "create_fa_system"),
    # Distributed Trainer
    "DistributedConfig": ("bioplausible.core.distributed_trainer", "DistributedConfig"),
    "DistributedSystemTrainer": (
        "bioplausible.core.distributed_trainer",
        "DistributedSystemTrainer",
    ),
    "NodeRegistry": ("bioplausible.core.distributed_trainer", "NodeRegistry"),
    "DHTRouter": ("bioplausible.core.distributed_trainer", "DHTRouter"),
    "FederatedAggregator": (
        "bioplausible.core.distributed_trainer",
        "FederatedAggregator",
    ),
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
