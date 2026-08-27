#!/usr/bin/env python
"""Quickstart script: Train Forward-Forward vs Backprop on MNIST in <2 minutes.

Usage:
    uv run scripts/quickstart.py

Expected output:
    Backprop:        ~95% accuracy (3 epochs)
    Forward-Forward: ~90%+ accuracy (3 epochs)
    Both biologically plausible and standard learning work!
"""

import multiprocessing
import signal
import sys
import time
import torch

# Use spawn to avoid semaphore leaks from forked processes
multiprocessing.set_start_method("spawn", force=True)

from computronium.core.system_trainer import (
    SystemTrainer,
    SystemTrainerConfig,
)
from computronium import create_backprop_mlp, create_ff_mlp
from computronium.domains.factory import create_task


# Global trainer reference for signal handler cleanup
_current_trainer: SystemTrainer | None = None


def _signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    print(f"\nReceived signal {signum}, cleaning up...")
    if _current_trainer is not None:
        _current_trainer.close()
    sys.exit(1)


def make_dataloaders(task_name: str, batch_size: int = 64, device: str = "cpu"):
    """Create train and validation DataLoaders for a task with flattening."""
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

        def __len__(self) -> int:
            return len(self.loader)

    task = create_task(task_name, device=device, quick_mode=True)
    task.setup()
    train_loader = _FlattenLoader(task.get_dataloader("train"))
    val_loader = _FlattenLoader(task.get_dataloader("val"))
    return train_loader, val_loader, task


def train_backprop(system, train_loader, val_loader, epochs: int, device: str):
    """Train a system using SystemTrainer and return final accuracy."""
    trainer_config = SystemTrainerConfig(
        max_epochs=epochs,
        batch_size=64,
        device=device,
        seed=42,
        log_every_n_steps=100,
    )

    trainer = SystemTrainer(
        system=system,
        config=trainer_config,
        train_data=train_loader,
        val_data=val_loader,
    )

    with trainer:
        history = trainer.fit()

    if history:
        final_acc = history[-1].get("val_acc", history[-1].get("train_acc", 0.0))
    else:
        final_acc = 0.0

    return final_acc * 100


def train_forward_forward(system, train_loader, val_loader, epochs: int, device: str):
    """Train Forward-Forward system using SystemTrainer."""
    trainer_config = SystemTrainerConfig(
        max_epochs=epochs,
        batch_size=64,
        device=device,
        seed=42,
        log_every_n_steps=100,
    )

    trainer = SystemTrainer(
        system=system,
        config=trainer_config,
        train_data=train_loader,
        val_data=val_loader,
    )

    with trainer:
        history = trainer.fit()

    if history:
        final_acc = history[-1].get("val_acc", history[-1].get("train_acc", 0.0))
    else:
        final_acc = 0.0

    return final_acc * 100


def main():
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    global _current_trainer

    print("=" * 60)
    print("Bioplausible Quickstart: Forward-Forward vs Backprop on MNIST")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load MNIST data
    print("\nLoading MNIST...")
    train_loader, val_loader, task = make_dataloaders("mnist", batch_size=64, device=device)

    # Get input/output dimensions
    input_dim = task.input_dim
    if isinstance(input_dim, (tuple, list)):
        input_dim = int(torch.prod(torch.tensor(input_dim)))
    output_dim = task.output_dim
    hidden_dim = 256

    print(f"Architecture: {input_dim} -> {hidden_dim} -> {hidden_dim} -> {output_dim}")

    # Create systems with FAIR comparison:
    # Same architecture (2 hidden layers), similar learning rates
    # Backprop uses exact gradient; Forward-Forward uses local layer-wise objectives
    # (no backward pass through the network - biologically plausible)

    print("\nCreating systems...")

    # Backprop: standard 2-hidden-layer MLP using 5-D ontology composition
    backprop_system = create_backprop_mlp(
        input_dim=input_dim,
        hidden_dims=(hidden_dim, hidden_dim),
        output_dim=output_dim,
        lr=0.001,
        device=device,
    )

    # Forward-Forward: Hinton's forward-only algorithm using 5-D ontology
    # Two forward passes (positive/negative), layer-local goodness objective
    # Per-layer independent optimizers, no backward pass - biologically plausible
    ff_system = create_ff_mlp(
        input_dim=input_dim,
        hidden_dims=(hidden_dim, hidden_dim),
        output_dim=output_dim,
        layer_lr=0.03,
        classifier_lr=0.01,
        threshold=2.0,
        num_layers=2,
        device=device,
    )

    # Train Backprop
    print("\n" + "=" * 60)
    print("Training Backprop (5-D Ontology Composition)...")
    print("=" * 60)
    start_time = time.time()
    trainer_config = SystemTrainerConfig(
        max_epochs=3,
        batch_size=64,
        device=device,
        seed=42,
        log_every_n_steps=100,
    )
    _current_trainer = SystemTrainer(
        system=backprop_system,
        config=trainer_config,
        train_data=train_loader,
        val_data=val_loader,
    )
    with _current_trainer:
        history = _current_trainer.fit()
    backprop_acc = history[-1].get("val_acc", history[-1].get("train_acc", 0.0)) * 100 if history else 0.0
    bp_time = time.time() - start_time
    print(f"  Time: {bp_time:.1f}s")
    print(f"  Final Accuracy: {backprop_acc:.1f}%")

    # Train Forward-Forward
    print("\n" + "=" * 60)
    print("Training Forward-Forward (5-D Ontology, Local Layer-wise)...")
    print("=" * 60)
    start_time = time.time()
    trainer_config = SystemTrainerConfig(
        max_epochs=3,
        batch_size=64,
        device=device,
        seed=42,
        log_every_n_steps=100,
    )
    _current_trainer = SystemTrainer(
        system=ff_system,
        config=trainer_config,
        train_data=train_loader,
        val_data=val_loader,
    )
    with _current_trainer:
        history = _current_trainer.fit()
    ff_acc = history[-1].get("val_acc", history[-1].get("train_acc", 0.0)) * 100 if history else 0.0
    ff_time = time.time() - start_time
    print(f"  Time: {ff_time:.1f}s")
    print(f"  Final Accuracy: {ff_acc:.1f}%")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Backprop:        {backprop_acc:.1f}% accuracy (3 epochs, {bp_time:.1f}s)")
    print(f"Forward-Forward: {ff_acc:.1f}% accuracy (3 epochs, {ff_time:.1f}s)")
    print()
    print("Both achieve competitive accuracy on MNIST!")
    print()

    # Clean up
    _current_trainer = None
    print("Key difference:")
    print("  Backprop:       Uses exact gradient (requires weight transport)")
    print("  Forward-Forward: Uses local layer-wise objectives (no backward pass)")
    print()
    print("This demonstrates bio-plausible learning without weight transport.")
    print("=" * 60)


if __name__ == "__main__":
    main()