"""Core package: Registry, CoreTrainer, Config, Model."""

# Lazy package init (Sprint 0.5): `import bioplausible.core.registry` must NOT
# pull the zoo. `core.registry` depends only on stdlib + exceptions, but the
# eager `core.trainer`/`core.model` imports drag in `zoo` via
# `is_learning_rule_optimizer`. Expose symbols on demand so light consumers stay
# fast; heavy symbols (CoreTrainer) import the zoo on first access.

_LAZY: dict[str, tuple[str, str | None]] = {
    "BioModel": ("bioplausible.core.model", "BioModel"),
    "LayerRole": ("bioplausible.core.config", "LayerRole"),
    "ModelConfig": ("bioplausible.core.config", "ModelConfig"),
    "compute_hidden_dims": ("bioplausible.core.config", "compute_hidden_dims"),
    "resolve_hidden_dims": ("bioplausible.core.config", "resolve_hidden_dims"),
    "ComponentCategory": ("bioplausible.core.registry", "ComponentCategory"),
    "ComponentMetadata": ("bioplausible.core.registry", "ComponentMetadata"),
    "ComputeProfile": ("bioplausible.core.registry", "ComputeProfile"),
    "Domain": ("bioplausible.core.registry", "Domain"),
    "LocalityLevel": ("bioplausible.core.registry", "LocalityLevel"),
    "Registry": ("bioplausible.core.registry", "Registry"),
    "register_constraint": ("bioplausible.core.registry", "register_constraint"),
    "register_controller": ("bioplausible.core.registry", "register_controller"),
    "register_metric": ("bioplausible.core.registry", "register_metric"),
    "register_model": ("bioplausible.core.registry", "register_model"),
    "register_optimizer": ("bioplausible.core.registry", "register_optimizer"),
    "register_propagator": ("bioplausible.core.registry", "register_propagator"),
    "register_sparsity": ("bioplausible.core.registry", "register_sparsity"),
    "register_update_strategy": (
        "bioplausible.core.registry", "register_update_strategy"
    ),
    "CoreTrainer": ("bioplausible.core.trainer", "CoreTrainer"),
    "TrainerConfig": ("bioplausible.core.trainer", "TrainerConfig"),
    "TrainingMetrics": ("bioplausible.core.trainer", "TrainingMetrics"),
    "EqPropTrainer": ("bioplausible.core.trainer", "EqPropTrainer"),
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
