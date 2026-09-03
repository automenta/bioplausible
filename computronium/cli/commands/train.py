"""Training commands for the CLI."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping
from dataclasses import replace
from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any, cast

from computronium.cli.shared import logger

if TYPE_CHECKING:
    from dataclasses import DataclassInstance

    from torch import Tensor
    from torch.utils.data import DataLoader

    from computronium.core.system_trainer.protocol import JointSystem
    from computronium.ontology import System

__all__ = ["add_train_subparsers", "run_core_train", "run_from_yaml", "run_training"]

_CREDIT_TYPE_ALIASES = {"backprop": "gradient"}


class _FlattenLoader:
    """Wrapper that flattens input tensors from a DataLoader."""

    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        for x, y in self.loader:
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            yield x, y

    def __len__(self) -> int:
        return len(self.loader)


def _section_config[T: DataclassInstance](
    section: Mapping[str, object], cls: type[T]
) -> T:
    """Flat-preset YAML I/O boundary (the isolated ``Any`` seam).

    A preset section names its axis primitive ``type``; the classmethod of
    that name on the config class carries the primitive's defaults. Section
    keys the classmethod accepts are passed through; the rest overlay the
    constructed config via :func:`dataclasses.replace`.
    """
    tag = section.get("type")
    if not isinstance(tag, str):
        msg = f"preset section missing string 'type' tag: {dict(section)!r}"
        raise ValueError(msg)
    factory = getattr(cls, tag)
    params = signature(factory).parameters
    var_kw = any(p.kind is Parameter.VAR_KEYWORD for p in params.values())
    passed = {
        k: v for k, v in section.items() if k != "type" and (var_kw or k in params)
    }
    base: T = cast("T", factory(**passed))
    rest = {k: v for k, v in section.items() if k != "type" and k not in passed}
    return replace(base, **rest) if rest else base


def _build_system_from_flat_config(
    substrate_cfg: Mapping[str, object],
    geometry_cfg: Mapping[str, object],
    dynamics_cfg: Mapping[str, object],
    plasticity_cfg: Mapping[str, object],
    credit_cfg: Mapping[str, object],
    update_cfg: Mapping[str, object],
) -> System | JointSystem:
    """Build a System from a preset YAML's flat config sections.

    Null or missing plasticity composes the 5-D path; any other M-tag
    composes the joint system.
    """
    from computronium.core.joint.transition import PlasticityConfig
    from computronium.core.system_trainer import (
        compose_joint_system_from_configs,
        compose_system_from_configs,
    )
    from computronium.ontology import (
        CreditAssignmentConfig,
        GeometryConfig,
        ParameterUpdateConfig,
        StateDynamicsConfig,
        SubstrateConfig,
    )

    credit_section = {
        **credit_cfg,
        "type": _CREDIT_TYPE_ALIASES.get(
            str(credit_cfg.get("type")), credit_cfg.get("type")
        ),
    }

    substrate_config = _section_config(substrate_cfg, SubstrateConfig)
    geometry_config = _section_config(geometry_cfg, GeometryConfig)
    dynamics_config = _section_config(dynamics_cfg, StateDynamicsConfig)
    credit_config = _section_config(credit_section, CreditAssignmentConfig)
    update_config = _section_config(update_cfg, ParameterUpdateConfig)

    if plasticity_cfg.get("type") is None:
        return compose_system_from_configs(
            substrate=substrate_config,
            geometry=geometry_config,
            dynamics=dynamics_config,
            credit=credit_config,
            update=update_config,
        )

    return compose_joint_system_from_configs(
        substrate=substrate_config,
        geometry=geometry_config,
        dynamics=dynamics_config,
        plasticity=_section_config(plasticity_cfg, PlasticityConfig),
        credit=credit_config,
        update=update_config,
    )


def add_train_subparsers(subparsers: argparse._SubParsersAction) -> None:
    """Add training-related subparsers."""
    # ---- train ----
    train_parser = subparsers.add_parser(
        "train", help="Run training session or from YAML config"
    )
    train_parser.add_argument("--config", help="Path to YAML config file")
    train_parser.add_argument(
        "--model", help="Model name (required if not using --config)"
    )
    train_parser.add_argument(
        "--task", default="vision", choices=["vision", "lm", "rl"], help="Task type"
    )
    train_parser.add_argument("--dataset", help="Dataset name")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    # ---- core-train ----
    core_parser = subparsers.add_parser(
        "core-train", help="Train using CoreTrainer (new)"
    )
    core_parser.add_argument(
        "--model", default="backprop_mlp", help="Model name from Zoo registry"
    )
    core_parser.add_argument("--task", default="mnist", help="Task/dataset name")
    core_parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    core_parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    core_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    core_parser.add_argument("--optimizer", default="adam", help="Optimizer name")
    core_parser.add_argument(
        "--hidden-dim", type=int, default=256, help="Hidden dimension"
    )
    core_parser.add_argument(
        "--device", default="auto", help="Device (auto, cpu, cuda)"
    )
    core_parser.add_argument(
        "--no-track-energy", action="store_true", help="Disable energy tracking"
    )

    # ---- from-config ----
    config_parser = subparsers.add_parser(
        "from-config", help="Train from YAML config file"
    )
    config_parser.add_argument(
        "--config", required=True, help="Path to YAML config file"
    )
    config_parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Override device from config (auto, cpu, cuda)",
    )


def run_training(args: argparse.Namespace) -> None:
    """Run a single training session (``--config`` for YAML)."""
    if args.config:
        run_from_yaml(args)
        return

    if not args.model:
        logger.error("--model required when not using --config")
        return

    # TODO: Implement simple training loop
    logger.info("Training %s on %s for %d epochs", args.model, args.task, args.epochs)


def run_core_train(args: argparse.Namespace) -> None:
    """Train via the new CoreTrainer API."""
    logger.info("Core training %s on %s", args.model, args.task)
    # TODO: Implement CoreTrainer usage


def run_from_yaml(args: argparse.Namespace) -> None:
    """Run training from a YAML config file (flat preset format)."""
    import torch
    from omegaconf import OmegaConf

    from computronium.core.system_trainer import SystemTrainer, SystemTrainerConfig
    from computronium.domains.factory import create_task

    cfg = OmegaConf.load(args.config)
    config = cast(
        "dict[str, Any]", OmegaConf.to_container(cfg, resolve=True)
    )  # isolated YAML I/O boundary

    training_cfg = config.get("training", {})

    # CLI --device overrides config
    device = getattr(args, "device", "auto")
    if device == "auto":
        device = training_cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = training_cfg.get("max_epochs", 10)
    batch_size = training_cfg.get("batch_size", 64)
    task_name = training_cfg.get("task", "mnist")

    task = create_task(
        task_name, device=device, quick_mode=False, batch_size=batch_size
    )
    task.setup()

    train_loader = _FlattenLoader(task.get_dataloader("train"))
    val_loader = _FlattenLoader(task.get_dataloader("val"))

    trainer_config = SystemTrainerConfig(
        max_epochs=epochs,
        batch_size=batch_size,
        val_batch_size=batch_size,
        device=device,
    )

    system = _build_system_from_flat_config(
        config.get("substrate", {}),
        config.get("geometry", {}),
        config.get("dynamics", {}),
        config.get("plasticity", {}),
        config.get("credit", {}),
        config.get("update", {}),
    )

    SystemTrainer(
        system=cast("System", system),  # JointSystem duck-types the trainer surface
        config=trainer_config,
        train_data=train_loader,
        val_data=val_loader,
    ).fit()
