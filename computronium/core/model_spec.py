"""Model-spec and checkpoint-loading helpers over the component Registry.

Extracted from the retired ``computronium.zoo`` package; these are the only
zoo surfaces with live consumers (execution guards, hyperopt, CLI tooling).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from computronium.core.logging import get_logger
from computronium.core.registry import ComponentCategory, ComponentMetadata, Registry

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

logger = get_logger("computronium.model_spec")


class ModelSpec:
    """Attribute-style view of a registered model's Registry metadata.

    Flat strings/booleans/lists/dicts consumed by the hyperparameter
    metamodel and reporting tools, rather than the structured
    :class:`ComponentMetadata`.
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


def _resolve_state_dict(loaded: Mapping[str, object]) -> dict[str, Tensor]:
    """Accept both the unified ``Checkpoint`` format and raw state dicts."""
    nested = loaded.get("model_state_dict")
    if isinstance(nested, Mapping):
        return cast("dict[str, Tensor]", dict(nested))
    return cast("dict[str, Tensor]", dict(loaded))


def _apply_state_dict(
    model: Module,
    state_dict: dict[str, Tensor],
    strict: bool,
    freeze_layers: bool,
) -> None:
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing:
        logger.info("Missing keys: %d", len(missing))
    if unexpected:
        logger.info("Unexpected keys: %d", len(unexpected))
    if not freeze_layers:
        return
    logger.info("Freezing loaded layers for transfer learning")
    for name, param in model.named_parameters():
        if name in state_dict:
            param.requires_grad = False
        else:
            logger.info("  -> %s remains trainable", name)


def load_weights(
    model: Module,
    path: str,
    device: str = "cpu",
    strict: bool = False,
    freeze_layers: bool = False,
) -> None:
    """Load weights from a checkpoint path into ``model``.

    Accepts both the unified :class:`~computronium.core.checkpoint.Checkpoint`
    format and raw ``state_dict`` files.

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
        from computronium.core.checkpoint import load_checkpoint

        state_dict = _resolve_state_dict(load_checkpoint(path, map_location=device))
        _apply_state_dict(model, state_dict, strict, freeze_layers)
    except OSError, RuntimeError, ValueError, KeyError:
        logger.exception("Failed to load weights from %s", path)
