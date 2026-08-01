"""Core package: Registry, CoreTrainer, Config, Model."""

from bioplausible.core.config import (
    LayerRole,
    ModelConfig,
    compute_hidden_dims,
    resolve_hidden_dims,
)
from bioplausible.core.model import BioModel
from bioplausible.core.registry import (
    ComponentCategory,
    ComponentMetadata,
    ComputeProfile,
    Domain,
    LocalityLevel,
    Registry,
    register_constraint,
    register_controller,
    register_metric,
    register_model,
    register_optimizer,
    register_propagator,
    register_sparsity,
    register_update_strategy,
)
from bioplausible.core.trainer import (
    CoreTrainer,
    TrainerConfig,
    TrainingMetrics,
)

__all__ = [
    # Config & Model
    "BioModel",
    "LayerRole",
    "ModelConfig",
    "compute_hidden_dims",
    "resolve_hidden_dims",
    # Registry
    "Registry",
    "ComponentCategory",
    "Domain",
    "LocalityLevel",
    "ComputeProfile",
    "ComponentMetadata",
    "register_model",
    "register_propagator",
    "register_optimizer",
    "register_update_strategy",
    "register_constraint",
    "register_controller",
    "register_sparsity",
    "register_metric",
    # Trainer
    "CoreTrainer",
    "TrainerConfig",
    "TrainingMetrics",
]
