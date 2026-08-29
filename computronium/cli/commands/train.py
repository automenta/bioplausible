"""Training commands for the CLI."""

import argparse

from computronium.cli.shared import logger

__all__ = ["add_train_subparsers", "run_training", "run_core_train", "run_from_yaml"]


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

    # Load YAML config
    cfg = OmegaConf.load(args.config)
    config = OmegaConf.to_container(cfg, resolve=True)

    # Extract components from flat YAML
    substrate_cfg = config.get("substrate", {})
    geometry_cfg = config.get("geometry", {})
    dynamics_cfg = config.get("dynamics", {})
    plasticity_cfg = config.get("plasticity", {})
    credit_cfg = config.get("credit", {})
    update_cfg = config.get("update", {})
    training_cfg = config.get("training", {})

    # Get training params - CLI --device overrides config
    device = getattr(args, "device", "auto")
    if device == "auto":
        device = training_cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = training_cfg.get("max_epochs", 10)
    batch_size = training_cfg.get("batch_size", 64)
    task_name = training_cfg.get("task", "mnist")

    # Create task and data loaders
    task = create_task(task_name, device=device, quick_mode=False)
    task.batch_size = batch_size
    task.setup()

    # Wrap data loaders to flatten input
    from torch.utils.data import DataLoader

    class _FlattenLoader:
        """Wrapper that flattens input tensors from a DataLoader."""

        def __init__(self, loader: DataLoader):
            self.loader = loader

        def __iter__(self):
            for x, y in self.loader:
                if x.dim() > 2:
                    x = x.view(x.size(0), -1)
                yield x, y

        def __len__(self):
            return len(self.loader)

    train_loader = _FlattenLoader(task.train_loader)
    val_loader = _FlattenLoader(task.val_loader)

    # Create SystemTrainer config
    trainer_config = SystemTrainerConfig(
        max_epochs=epochs,
        batch_size=batch_size,
        val_batch_size=batch_size,
        device=device,
    )

    # Build system from config (import substrate modules to register them)
    import computronium.ontology.credit  # noqa: F401
    import computronium.ontology.dynamics  # noqa: F401
    import computronium.ontology.geometry  # noqa: F401
    import computronium.ontology.substrate  # noqa: F401
    import computronium.ontology.update  # noqa: F401
    from computronium.ontology import (
        CreditAssignmentConfig,
        GeometryConfig,
        ParameterUpdateConfig,
        PlasticityConfig,
        StateDynamicsConfig,
        SubstrateConfig,
        substrate_from_config,
    )

    # Build configs from YAML
    substrate_config = SubstrateConfig(**substrate_cfg)
    geometry_config = GeometryConfig(**geometry_cfg)
    dynamics_config = StateDynamicsConfig(**dynamics_cfg)
    plasticity_config = PlasticityConfig(**plasticity_cfg) if plasticity_cfg else None
    credit_config = CreditAssignmentConfig(**credit_cfg)
    update_config = ParameterUpdateConfig(**update_cfg)

    # Instantiate components
    substrate = substrate_from_config(substrate_config)

    # Import geometry classes dynamically
    from importlib import import_module

    geometry_module = import_module("computronium.ontology.geometry")
    geometry_class = getattr(
        geometry_module, geometry_config.geometry_type.capitalize() + "Geometry"
    )
    geometry = geometry_class(geometry_config)

    # Import dynamics classes dynamically
    dynamics_module = import_module("computronium.ontology.dynamics")
    dynamics_class = getattr(
        dynamics_module, dynamics_config.dynamics_type + "Dynamics"
    )
    dynamics = dynamics_class(dynamics_config)

    # Create the system
    from computronium.ontology import System, SystemConfig

    system_config = SystemConfig(
        substrate=substrate_config,
        geometry=geometry_config,
        dynamics=dynamics_config,
        credit=credit_config,
        update=update_config,
    )

    system = System.from_configs(
        substrate=substrate,
        geometry=geometry,
        dynamics=dynamics,
        config=system_config,
    )

    trainer = SystemTrainer(system, train_loader, val_loader, trainer_config)
    trainer.fit()
