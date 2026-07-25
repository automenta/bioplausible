"""Zoo Package - Unified Component Registry.

All models, propagators, optimizers, sparsity methods are registered here
with rich metadata enabling AutoScientist composition.
"""

import logging
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from bioplausible.core.registry import (
    ComponentCategory,
    ComponentMetadata,
    ComputeProfile,
    Domain,
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

logger = logging.getLogger("bioplausible.zoo")


class _LegacyModelSpec:
    """Adapter providing legacy ModelSpec interface from Registry metadata."""

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
        "task_compat",
        "variant",
        "version",
    )

    _FAMILY_TAGS = frozenset((
        "eqprop",
        "fa",
        "forward-only",
        "forward_only",
        "hebbian",
        "predictive_coding",
        "spiking",
        "target_prop",
        "backprop",
    ))

    def __init__(self, meta: ComponentMetadata) -> None:
        self.name = meta.name
        # Prefer the explicit `family` metadata field; fall back to a tag.
        # Normalize hyphenated tags (e.g. "forward-only") to underscore form
        # (e.g. "forward_only") so downstream metamodel `_FAMILY_TAGS` and
        # `family == "forward_only"` comparisons match consistently.
        self.family = meta.family or next(
            (
                t.replace("-", "_")
                for t in meta.tags
                if t.replace("-", "_") in self._FAMILY_TAGS
            ),
            "experimental",
        )
        # task_compat from domains
        self.task_compat = [d.value for d in meta.domains]
        # model_type from credit_assignment_type
        self.model_type = meta.credit_assignment_type
        # For backward compat with metamodel expecting credit_assignment_type
        self.credit_assignment_type = meta.credit_assignment_type
        # variant/custom_hyperparams are not directly stored; could be in extra
        self.variant = meta.extra.get("variant")
        self.custom_hyperparams = meta.extra.get("custom_hyperparams", {}) or {}
        self.default_lr = meta.typical_lr_range[0] if meta.typical_lr_range else 1e-3
        self.credit_locality = meta.locality_level.value
        self.requires_backward = meta.requires_backward
        self.citation = meta.citation
        self.description = meta.description
        self.tags = list(meta.tags)
        self.version = meta.version


def get_model_spec(name: str) -> _LegacyModelSpec:
    """Get a legacy-compatible ModelSpec from the Registry by model name.

    Raises:
        ValueError: if no model is registered under ``name``.
    """
    meta = Registry.get_metadata(ComponentCategory.MODEL, name)
    return _LegacyModelSpec(meta)


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
        state_dict = torch.load(path, map_location=device)
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
    except Exception:
        logger.exception("Failed to load weights from %s", path)


def get_models_for_task(
    domain: Domain,
    locality: LocalityLevel | None = None,
    requires_backward: bool | None = None,
) -> list[dict[str, Any]]:
    """Return all models compatible with a task (domain + locality + backward)."""
    return Registry.query(
        category=ComponentCategory.MODEL,
        domain=domain,
        locality=locality,
        requires_backward=requires_backward,
    )


def get_propagators_for_model(
    model_name: str,
) -> list[dict[str, Any]]:
    """Return propagators compatible with a model's locality + backward flag."""
    model_meta = Registry.get_metadata(ComponentCategory.MODEL, model_name)
    return Registry.query(
        category=ComponentCategory.PROPAGATOR,
        locality=model_meta.locality_level,
        requires_backward=model_meta.requires_backward,
    )


def get_optimizers_for_propagator(
    propagator_name: str,
) -> list[dict[str, Any]]:
    """Return optimizers compatible with a propagator's backward flag."""
    prop_meta = Registry.get_metadata(ComponentCategory.PROPAGATOR, propagator_name)
    return Registry.query(
        category=ComponentCategory.OPTIMIZER,
        requires_backward=prop_meta.requires_backward,
    )


# ============================================================================
# Legacy adapter: ModelZoo / OptimizerZoo
# ============================================================================


def _resolve_component_class(
    name: str, categories: tuple[ComponentCategory, ...]
) -> tuple[Any, ComponentCategory]:
    """Look up a registered component across multiple categories.

    Iterates ``categories`` in order, returning the first match. Raises a
    ValueError listing available names across all categories if absent.
    """
    available: list[str] = []
    for cat in categories:
        comps = Registry._components.get(cat, {})
        if name in comps:
            return comps[name]["class"], cat
        available.extend(f"{cat.value}/{n}" for n in comps)
    raise ValueError(
        f"Unknown component '{name}' in categories {[c.value for c in categories]}. "
        f"Available: {available}"
    )


class ModelZoo:
    """Legacy adapter providing ``cls.get(name, **params)`` → model instance.

    Used by ``experiments.utils.ExperimentRunner`` and ``deployment.py``.
    """

    @staticmethod
    def get(name: str, **params: Any) -> nn.Module:
        cls, _ = _resolve_component_class(name, (ComponentCategory.MODEL,))
        return cls(**params)


class OptimizerZoo:
    """Legacy adapter providing ``cls.get(name, params, model=model, **kwargs)``.

    Used by ``experiments.utils.ExperimentRunner``. Looks the name up in
    OPTIMIZER first, then PROPAGATOR (since preset factories like ``smep``
    are registered as propagators). ``params`` is forwarded as the first
    positional argument — for torch.optim-style optimizers it should be
    an iterable of parameters; for MEP preset factories it is also an
    iterable of parameters and ``model`` is supplied as a keyword.
    """

    @staticmethod
    def get(
        name: str,
        params: Iterable[Any],
        model: nn.Module | None = None,
        **kwargs: Any,
    ) -> Any:
        cls, _ = _resolve_component_class(
            name, (ComponentCategory.OPTIMIZER, ComponentCategory.PROPAGATOR)
        )
        if model is not None:
            try:
                return cls(params, model=model, **kwargs)
            except TypeError:
                # Plain torch.optim optimizers don't accept ``model=``.
                return cls(params, **kwargs)
        return cls(params, **kwargs)


__all__ = [
    "ComponentCategory",
    "ComponentMetadata",
    "ComputeProfile",
    "Domain",
    "LocalityLevel",
    "ModelZoo",
    "OptimizerZoo",
    "Registry",
    "get_model_spec",
    "get_models_for_task",
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
