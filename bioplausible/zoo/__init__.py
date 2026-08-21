"""Zoo Package - Unified Component Registry.

All models, propagators, optimizers, sparsity methods are registered here
with rich metadata enabling AutoScientist composition.
"""

from torch import nn

from bioplausible.core.logging import get_logger
from bioplausible.core.registry import (
    ComponentCategory,
    ComponentMetadata,
    ComputeProfile,
    LocalityLevel,
    Registry,
    register_metric,
    register_model,
    register_optimizer,
    register_propagator,
    register_sparsity,
)

# Import submodules to trigger registration
from bioplausible.zoo import models, optimizers, propagators, sparsity

logger = get_logger("bioplausible.zoo")


class ModelSpec:
    """Adapter providing attribute-style access to a model's Registry metadata.

    Subset of fields consumed by the hyperparameter metamodel and reporting
    tools — these are flat strings/booleans/lists/dicts rather than the
    structured ``ComponentMetadata``.
    """

    __slots__ = (
        "citation",
        "credit_assignment_type",
        "credit_locality",
        "custom_hyperparams",
        "default_lr",
        "description",
        "family",
        "model_type",
        "name",
        "requires_backward",
        "tags",
        "variant",
        "version",
    )

    def __init__(self, meta: ComponentMetadata) -> None:
        self.name = meta.name
        self.family = meta.family if meta.family else "experimental"
        self.model_type = meta.credit_assignment_type
        self.credit_assignment_type = meta.credit_assignment_type
        self.variant = meta.extra.get("variant")
        self.custom_hyperparams = meta.extra.get("custom_hyperparams", {}) or {}
        self.default_lr = meta.typical_lr_range[0] if meta.typical_lr_range else 1e-3
        self.credit_locality = meta.locality_level.value
        self.requires_backward = meta.requires_backward
        self.citation = meta.citation
        self.description = meta.description
        self.tags = list(meta.tags)
        self.version = meta.version


def get_model_spec(name: str) -> ModelSpec:
    """Get a :class:`ModelSpec` for the registered model ``name``.

    Raises:
        ValueError: if no model is registered under ``name``.
    """
    meta = Registry.get_metadata(ComponentCategory.MODEL, name)
    return ModelSpec(meta)


def load_weights(
    model: nn.Module,
    path: str,
    device: str = "cpu",
    strict: bool = False,
    freeze_layers: bool = False,
) -> None:
    """Load weights from a checkpoint path into ``model``.

    Args:
        model: Target model whose state dict is updated in place.
        path: Path to a ``.pt``/``.pth`` checkpoint file.
        device: Device to map the loaded tensors onto.
        strict: If True, require an exact match of keys.
        freeze_layers: If True, freeze every parameter whose name appears
            in the loaded state dict (useful for transfer-learning probes).
    """
    if not path:
        return
    try:
        logger.info("Loading weights from %s", path)
        from bioplausible.core.checkpoint import load_checkpoint

        state_dict = load_checkpoint(path, map_location=device)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        if missing:
            logger.info("Missing keys: %d", len(missing))
        if unexpected:
            logger.info("Unexpected keys: %d", len(unexpected))
        if freeze_layers:
            logger.info("Freezing loaded layers for transfer learning")
            for name, param in model.named_parameters():
                if name in state_dict:
                    param.requires_grad = False
                else:
                    logger.info("  -> %s remains trainable", name)
    except OSError, RuntimeError, ValueError, KeyError:
        logger.exception("Failed to load weights from %s", path)


def get_propagators_for_model(
    model_name: str,
) -> list[dict[str, object]]:
    """Return propagators compatible with a model's locality + backward flag."""
    model_meta = Registry.get_metadata(ComponentCategory.MODEL, model_name)
    return Registry.query(
        category=ComponentCategory.PROPAGATOR,
        locality=model_meta.locality_level,
        requires_backward=model_meta.requires_backward,
    )


def get_optimizers_for_propagator(
    propagator_name: str,
) -> list[dict[str, object]]:
    """Return optimizers compatible with a propagator's backward flag."""
    prop_meta = Registry.get_metadata(ComponentCategory.PROPAGATOR, propagator_name)
    return Registry.query(
        category=ComponentCategory.OPTIMIZER,
        requires_backward=prop_meta.requires_backward,
    )


# ============================================================================
# Legacy adapters removed (REFACTOR3). Callers now use core.registry.Registry
# and core.registry.ComponentCategory directly. See:
#   - bioplausible.deployment
#   - examples/tutorials.py
# ============================================================================

__all__ = [
    "ComponentCategory",
    "ComponentMetadata",
    "ComputeProfile",
    "LocalityLevel",
    "Registry",
    "get_model_spec",
    "get_optimizers_for_propagator",
    "get_propagators_for_model",
    "load_weights",
    "models",
    "optimizers",
    "propagators",
    "register_metric",
    "register_model",
    "register_optimizer",
    "register_propagator",
    "register_sparsity",
    "sparsity",
]
