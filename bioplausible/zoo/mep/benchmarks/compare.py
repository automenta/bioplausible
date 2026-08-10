"""
Comprehensive MEP Benchmark Suite

Compares MEP optimizers against standard optimizers (Adam, SGD, Muon)
on various tasks and datasets.
"""

import argparse
import json
import pathlib
import time

import torch
from torch import nn

from bioplausible.core.logging import get_logger
from bioplausible.core.utils.device import get_device
from bioplausible.zoo.mep.benchmarks._shared import (
    BenchmarkConfig,
    EpochMetrics,
    OptimizerResult,
    cnn_classifier,
    get_dataloaders,
    get_input_dim,
    get_num_classes,
)
from bioplausible.zoo.mep.benchmarks.baselines import get_optimizer

__all__ = [
    "BenchmarkConfig",
    "EpochMetrics",
    "OptimizerResult",
    "evaluate",
    "get_dataloaders",
    "get_input_dim",
    "get_model",
    "get_num_classes",
    "logger",
    "main",
    "print_summary",
    "run_all_benchmarks",
    "run_benchmark",
    "save_results",
    "train_epoch",
]
logger = get_logger()


def get_model(config: BenchmarkConfig, input_dim: int, num_classes: int) -> nn.Module:
    """Get model for the specified architecture."""
    if config.model == "mlp":
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )
    elif config.model == "mlp_small":
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
    elif config.model == "cnn":
        if config.dataset == "cifar10":
            return cnn_classifier(3, 8, num_classes)
        return cnn_classifier(1, 7, num_classes)
    raise ValueError(f"Unknown model: {config.model}")


def train_epoch(
    model: nn.Module,
    optimizer: object,
    train_loader: DataLoader,
    device: torch.device,
    is_ep: bool,
    loss_fn: nn.Module,
) -> tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        if is_ep:
            # EP mode: optimizer handles forward pass
            optimizer.step(x=x, target=y)

            # Compute loss for tracking
            with torch.no_grad():
                output = model(x)
                loss = loss_fn(output, y)
        else:
            # Standard backprop mode
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = output.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module, test_loader: DataLoader, device: torch.device, loss_fn: nn.Module
) -> tuple[float, float]:
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = loss_fn(output, y)

        total_loss += loss.item() * x.size(0)
        pred = output.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


def run_benchmark(optimizer_name: str, config: BenchmarkConfig) -> OptimizerResult:
    """Run benchmark for a single optimizer."""
    device = get_device(config.device)

    # Get data
    train_loader, test_loader = get_dataloaders(config)

    # Get model
    input_dim = get_input_dim(config)
    num_classes = get_num_classes(config)
    model = get_model(config, input_dim, num_classes).to(device)

    # Get optimizer
    optimizer, is_ep = get_optimizer(
        optimizer_name,
        model,
        lr=config.lr,
        weight_decay=config.weight_decay,
        beta=0.5,
        settle_steps=10,
        settle_lr=0.05,
        loss_type=(
            "cross_entropy"
            if config.dataset in {"mnist", "fashion", "cifar10"}
            else "mse"
        ),
    )

    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    metrics = []
    start_time = time.time()

    for epoch in range(config.epochs):
        epoch_start = time.time()

        train_loss, train_acc = train_epoch(
            model, optimizer, train_loader, device, is_ep, loss_fn
        )
        val_loss, val_acc = evaluate(model, test_loader, device, loss_fn)

        epoch_time = time.time() - epoch_start

        metrics.append(
            EpochMetrics(
                epoch=epoch + 1,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                epoch_time=epoch_time,
            )
        )

        logger.info(
            "  %s Epoch %d/%d: Train Acc=%.4f, Val Acc=%.4f, Time=%.2fs",
            optimizer_name,
            epoch + 1,
            config.epochs,
            train_acc,
            val_acc,
            epoch_time,
        )

    total_time = time.time() - start_time

    return OptimizerResult(
        name=optimizer_name,
        metrics=metrics,
        total_time=total_time,
        best_val_acc=max(m.val_acc for m in metrics),
        final_train_acc=metrics[-1].train_acc if metrics else 0.0,
    )


def run_all_benchmarks(config: BenchmarkConfig) -> dict[str, OptimizerResult]:
    """Run benchmarks for all optimizers."""
    optimizers = ["sgd", "adam", "muon", "eqprop", "smep", "sdmep"]

    results = {}
    for opt_name in optimizers:
        logger.info("\n%s", "=" * 60)
        logger.info("Benchmarking: %s", opt_name.upper())
        logger.info("%s", "=" * 60)

        results[opt_name] = run_benchmark(opt_name, config)

    return results


def print_summary(results: dict[str, OptimizerResult]) -> None:
    """Print summary table of results."""
    logger.info("\n%s", "=" * 80)
    logger.info("BENCHMARK SUMMARY")
    logger.info("%s", "=" * 80)
    logger.info(
        "%-15s %-15s %-18s %-15s",
        "Optimizer",
        "Best Val Acc",
        "Final Train Acc",
        "Total Time (s)",
    )
    logger.info("%s", "-" * 80)

    # Sort by best validation accuracy
    sorted_results = sorted(
        results.items(), key=lambda x: x[1].best_val_acc, reverse=True
    )

    for name, result in sorted_results:
        logger.info(
            "%-15s %-15.4f %-18.4f %-15.2f",
            name,
            result.best_val_acc,
            result.final_train_acc,
            result.total_time,
        )

    logger.info("%s", "=" * 80)

    # Find best performer
    best = sorted_results[0]
    logger.info(
        "\U0001f3c6 Best performer: %s with %.2f%% validation accuracy",
        best[0].upper(),
        best[1].best_val_acc * 100,
    )


def save_results(results: dict[str, OptimizerResult], output_path: str) -> None:
    """Save results to JSON file."""
    data = {}
    for name, result in results.items():
        result_dict = {
            "name": result.name,
            "metrics": [m.to_dict() for m in result.metrics],
            "total_time": result.total_time,
            "best_val_acc": result.best_val_acc,
            "final_train_acc": result.final_train_acc,
        }
        data[name] = result_dict

    with pathlib.Path(output_path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    logger.info("Results saved to: %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="MEP Benchmark Suite")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fashion", "cifar10"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        choices=["mlp", "mlp_small", "cnn"],
        help="Model architecture",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument(
        "--subset-train", type=int, default=5000, help="Number of training samples"
    )
    parser.add_argument(
        "--subset-test", type=int, default=1000, help="Number of test samples"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output file for results",
    )

    args = parser.parse_args()

    config = BenchmarkConfig(
        dataset=args.dataset,
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        subset_train=args.subset_train,
        subset_test=args.subset_test,
        device=args.device,
    )

    logger.info("%s", "=" * 60)
    logger.info("MEP BENCHMARK SUITE")
    logger.info("%s", "=" * 60)
    logger.info("Dataset: %s", config.dataset)
    logger.info("Model: %s", config.model)
    logger.info("Epochs: %s", config.epochs)
    logger.info("Learning Rate: %s", config.lr)
    logger.info("Device: %s", config.device)
    logger.info("%s", "=" * 60)

    results = run_all_benchmarks(config)
    print_summary(results)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
