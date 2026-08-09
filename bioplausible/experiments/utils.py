"""
Bioplausible Experimentation Utilities

Comprehensive utilities for experimentation, research, and discovery
of novel machine learning approaches using the Bioplausible framework.

Features:
- Model/Optimizer comparison utilities
- Experiment workflow helpers
- Hyperparameter search utilities
- Validation and benchmarking tools
"""

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from bioplausible.core.utils.device import get_device

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""

    model_name: str
    optimizer_name: str
    model_params: dict[str, object]
    optimizer_params: dict[str, object]

    # Performance metrics
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    test_accuracy: float = 0.0
    train_loss: float = 0.0
    val_loss: float = 0.0

    # Timing
    training_time: float = 0.0  # seconds
    steps_per_second: float = 0.0

    # Resource usage
    num_parameters: int = 0
    memory_peak_mb: float = 0.0

    # Additional metrics
    extra_metrics: dict[str, object] = field(default_factory=dict)

    def summary(self) -> str:
        """Get a summary string."""
        return (
            f"{self.model_name} + {self.optimizer_name}:\n"
            f"  Train Acc: {self.train_accuracy:.2f}%,"
            f" Val Acc: {self.val_accuracy:.2f}%\n"
            f"  Training Time: {self.training_time:.1f}s,"
            f" Steps/s: {self.steps_per_second:.1f}\n"
            f"  Parameters: {self.num_parameters:,}"
        )


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""

    model_name: str
    optimizer_name: str
    model_params: dict[str, object] = field(default_factory=dict)
    optimizer_params: dict[str, object] = field(default_factory=dict)

    # Training config
    epochs: int = 10
    batches_per_epoch: int = 100
    eval_batches: int = 20
    device: str = "auto"

    # Tracking
    track_metrics: bool = True
    verbose: bool = True


class ExperimentRunner:
    """
    Run controlled experiments for model/optimizer combinations.

    Example usage:
        runner = ExperimentRunner()

        # Run single experiment
        result = runner.run(
            model_name='looped_mlp',
            optimizer_name='smep',
            train_loader=train_loader,
            val_loader=val_loader,
        )

        # Compare multiple optimizers
        results = runner.compare_optimizers(
            model_name='looped_mlp',
            optimizer_names=['smep', 'smep_fast', 'muon_backprop'],
            train_loader=train_loader,
            val_loader=val_loader,
        )
    """

    def __init__(self, device: str = "auto"):
        self.device = device
        if device == "auto":
            self.device = str(get_device())

    def run(
        self,
        model_name: str,
        optimizer_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        test_loader: DataLoader | None = None,
        model_params: dict[str, object] | None = None,
        optimizer_params: dict[str, object] | None = None,
        epochs: int = 10,
        batches_per_epoch: int = 100,
        eval_batches: int = 20,
        verbose: bool = True,
    ) -> ExperimentResult:
        """
        Run a single experiment.

        Args:
            model_name: Name of model from Zoo.
            optimizer_name: Name of optimizer from Zoo.
            train_loader: Training data loader.
            val_loader: Validation data loader (optional).
            test_loader: Test data loader (optional).
            model_params: Override default model parameters.
            optimizer_params: Override default optimizer parameters.
            epochs: Number of training epochs.
            batches_per_epoch: Batches per epoch.
            eval_batches: Batches for evaluation.
            verbose: Print progress.

        Returns:
            ExperimentResult with metrics.
        """
        from bioplausible.core.registry import ComponentCategory, Registry

        # Get model and optimizer
        model_params = model_params or {}
        optimizer_params = optimizer_params or {}

        model_cls = Registry.get(ComponentCategory.MODEL, model_name)
        model = model_cls(**model_params)
        model = model.to(self.device)

        opt_params = list(model.parameters())
        try:
            opt_cls = Registry.get(ComponentCategory.OPTIMIZER, optimizer_name)
        except ValueError:
            opt_cls = Registry.get(ComponentCategory.PROPAGATOR, optimizer_name)
        try:
            optimizer = opt_cls(opt_params, model=model, **optimizer_params)
        except TypeError:
            optimizer = opt_cls(opt_params, **optimizer_params)

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Training loop
        start_time = time.time()
        total_steps = 0
        train_losses = []
        train_correct = 0
        train_total = 0

        model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_steps = 0

            for batch_idx, (x, y) in enumerate(train_loader):
                if batch_idx >= batches_per_epoch:
                    break

                x = x.to(self.device)
                y = y.to(self.device)

                # Flatten for MLP models
                if len(x.shape) > 2:
                    x = x.view(x.shape[0], -1)

                # Optimizer step
                try:
                    # MEP-style optimizers
                    optimizer.step(x=x, target=y)
                except TypeError:
                    # Standard PyTorch optimizers
                    output = model(x)
                    loss = nn.functional.cross_entropy(output, y)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                # Track metrics
                with torch.no_grad():
                    output = model(x)
                    loss = nn.functional.cross_entropy(output, y).item()
                    epoch_loss += loss
                    train_losses.append(loss)

                    pred = output.argmax(dim=1)
                    train_correct += (pred == y).sum().item()
                    train_total += y.shape[0]

                epoch_steps += 1
                total_steps += 1

            if verbose:
                avg_loss = epoch_loss / max(1, epoch_steps)
                logger.info("Epoch %d/%d - Loss: %.4f", epoch + 1, epochs, avg_loss)

        training_time = time.time() - start_time

        # Evaluation
        val_accuracy = 0.0
        val_loss = 0.0

        if val_loader is not None:
            val_accuracy, val_loss = self._evaluate(model, val_loader, eval_batches)

        test_accuracy = 0.0
        if test_loader is not None:
            test_accuracy, _ = self._evaluate(model, test_loader, eval_batches)

        # Create result
        result = ExperimentResult(
            model_name=model_name,
            optimizer_name=optimizer_name,
            model_params=model_params,
            optimizer_params=optimizer_params,
            train_accuracy=100.0 * train_correct / max(1, train_total),
            val_accuracy=val_accuracy,
            test_accuracy=test_accuracy,
            train_loss=np.mean(train_losses) if train_losses else 0.0,
            val_loss=val_loss,
            training_time=training_time,
            steps_per_second=total_steps / max(0.001, training_time),
            num_parameters=num_params,
        )

        if verbose:
            logger.info(result.summary())

        return result

    def _evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        max_batches: int = 20,
    ) -> tuple[float, float]:
        """Evaluate model on a data loader."""
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        batches = 0

        with torch.no_grad():
            for x, y in loader:
                if batches >= max_batches:
                    break

                x = x.to(self.device)
                y = y.to(self.device)

                if len(x.shape) > 2:
                    x = x.view(x.shape[0], -1)

                output = model(x)
                loss = nn.functional.cross_entropy(output, y).item()

                pred = output.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.shape[0]
                total_loss += loss
                batches += 1

        model.train()

        accuracy = 100.0 * correct / max(1, total)
        avg_loss = total_loss / max(1, batches)

        return accuracy, avg_loss

    def compare_optimizers(
        self,
        model_name: str,
        optimizer_names: list[str],
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        model_params: dict[str, object] | None = None,
        epochs: int = 5,
        verbose: bool = True,
    ) -> list[ExperimentResult]:
        """
        Compare multiple optimizers on the same model.

        Args:
            model_name: Name of model from Zoo.
            optimizer_names: List of optimizer names to compare.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            model_params: Model parameters.
            epochs: Training epochs per optimizer.
            verbose: Print progress.

        Returns:
            List of ExperimentResult, sorted by validation accuracy.
        """
        results = []

        for opt_name in optimizer_names:
            if verbose:
                logger.info(
                    "\n%s\nTesting optimizer: %s\n%s", "=" * 60, opt_name, "=" * 60
                )

            result = self.run(
                model_name=model_name,
                optimizer_name=opt_name,
                train_loader=train_loader,
                val_loader=val_loader,
                model_params=model_params,
                epochs=epochs,
                verbose=verbose,
            )
            results.append(result)

        # Sort by validation accuracy
        results.sort(key=lambda r: r.val_accuracy, reverse=True)

        if verbose:
            logger.info(
                "\n%s\nCOMPARISON RESULTS (sorted by val accuracy)\n%s",
                "=" * 60,
                "=" * 60,
            )
            for i, r in enumerate(results):
                logger.info("%d. %s: %.2f%%", i + 1, r.optimizer_name, r.val_accuracy)

        return results

    def compare_models(
        self,
        model_names: list[str],
        optimizer_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        optimizer_params: dict[str, object] | None = None,
        epochs: int = 5,
        verbose: bool = True,
    ) -> list[ExperimentResult]:
        """
        Compare multiple models with the same optimizer.

        Args:
            model_names: List of model names to compare.
            optimizer_name: Name of optimizer.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            optimizer_params: Optimizer parameters.
            epochs: Training epochs per model.
            verbose: Print progress.

        Returns:
            List of ExperimentResult, sorted by validation accuracy.
        """
        results = []

        for model_name in model_names:
            if verbose:
                logger.info(
                    "\n%s\nTesting model: %s\n%s", "=" * 60, model_name, "=" * 60
                )

            result = self.run(
                model_name=model_name,
                optimizer_name=optimizer_name,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer_params=optimizer_params,
                epochs=epochs,
                verbose=verbose,
            )
            results.append(result)

        # Sort by validation accuracy
        results.sort(key=lambda r: r.val_accuracy, reverse=True)

        if verbose:
            logger.info(
                "\n%s\nCOMPARISON RESULTS (sorted by val accuracy)\n%s",
                "=" * 60,
                "=" * 60,
            )
            for i, r in enumerate(results):
                logger.info("%d. %s: %.2f%%", i + 1, r.model_name, r.val_accuracy)

        return results


def quick_comparison(
    model_name: str = "looped_mlp",
    optimizer_names: list[str] | None = None,
    epochs: int = 3,
    verbose: bool = True,
) -> list[ExperimentResult]:
    """
    Quick comparison of optimizers on MNIST.

    This is a convenience function for rapid experimentation.

    Args:
        model_name: Model to test.
        optimizer_names: Optimizers to compare (default: all MEP).
        epochs: Training epochs.
        verbose: Print progress.

    Returns:
        List of results sorted by accuracy.
    """
    from bioplausible.data.vision import get_vision_dataset

    if optimizer_names is None:
        optimizer_names = ["smep", "smep_fast", "muon_backprop"]

    # Load MNIST
    train_loader, val_loader, _ = get_vision_dataset(
        dataset="mnist",
        batch_size=128,
        normalize=True,
    )

    runner = ExperimentRunner()

    return runner.compare_optimizers(
        model_name=model_name,
        optimizer_names=optimizer_names,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        verbose=verbose,
    )


def benchmark_model(
    model_name: str,
    optimizer_name: str = "smep",
    epochs: int = 10,
    verbose: bool = True,
) -> ExperimentResult:
    """
    Benchmark a model/optimizer combination on MNIST.

    Args:
        model_name: Model to benchmark.
        optimizer_name: Optimizer to use.
        epochs: Training epochs.
        verbose: Print progress.

    Returns:
        ExperimentResult with benchmark metrics.
    """
    from bioplausible.data.vision import get_vision_dataset

    train_loader, val_loader, test_loader = get_vision_dataset(
        dataset="mnist",
        batch_size=128,
        normalize=True,
    )

    runner = ExperimentRunner()

    return runner.run(
        model_name=model_name,
        optimizer_name=optimizer_name,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=epochs,
        verbose=verbose,
    )


__all__ = [
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentRunner",
    "benchmark_model",
    "quick_comparison",
]
