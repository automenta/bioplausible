"""
Tuned MEP Benchmark Suite

Runs benchmarks with hyperparameters tuned for each optimizer type.
EP methods need different hyperparameters than backprop methods.
"""

import argparse
import json
import pathlib
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

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
    "OPTIMIZER_CONFIGS",
    "BenchmarkConfig",
    "EpochMetrics",
    "OptimizerConfig",
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


@dataclass
class OptimizerConfig:
    """Hyperparameters for an optimizer."""

    lr: float
    beta: float = 0.5
    settle_steps: int = 10
    settle_lr: float = 0.05
    loss_type: str = "mse"
    ns_steps: int = 5
    gamma: float = 0.95
    error_beta: float = 0.9
    use_error_feedback: bool = True
    rank_frac: float = 0.2
    dion_thresh: int = 100000
    fisher_damping: float = 1e-3


# Tuned hyperparameters for different optimizers
OPTIMIZER_CONFIGS = {
    # Standard optimizers (backprop)
    "sgd": OptimizerConfig(lr=0.1),
    "adam": OptimizerConfig(lr=0.001),
    "adamw": OptimizerConfig(lr=0.001),
    "muon": OptimizerConfig(lr=0.02, gamma=0.95),
    # EP-based optimizers - OPTIMIZED for MNIST performance
    # Key insight: Higher beta and more settling steps dramatically improve convergence
    "eqprop": OptimizerConfig(
        lr=0.01,
        beta=0.5,  # Higher beta for stronger nudging
        settle_steps=30,  # More steps for proper settling
        settle_lr=0.15,  # Higher LR for faster convergence
        loss_type="mse",
        ns_steps=0,
        use_error_feedback=False,
    ),
    "smep": OptimizerConfig(
        lr=0.01,
        beta=0.5,
        settle_steps=30,
        settle_lr=0.15,
        loss_type="mse",
        ns_steps=5,
        gamma=0.95,
        use_error_feedback=False,
    ),
    "sdmep": OptimizerConfig(
        lr=0.01,
        beta=0.5,
        settle_steps=30,
        settle_lr=0.15,
        loss_type="mse",
        ns_steps=5,
        rank_frac=0.5,
        dion_thresh=200000,
        gamma=0.95,
        use_error_feedback=False,
    ),
    "local_ep": OptimizerConfig(
        lr=0.01,
        beta=0.2,  # Lower beta for local EP
        settle_steps=15,
        settle_lr=0.02,
        loss_type="mse",
        use_error_feedback=False,
    ),
    "natural_ep": OptimizerConfig(
        lr=0.01,
        beta=0.5,
        settle_steps=10,
        settle_lr=0.05,
        loss_type="mse",
        fisher_damping=1e-2,
        use_error_feedback=False,
    ),
}


def get_model(config: BenchmarkConfig, input_dim: int, num_classes: int) -> nn.Module:
    """Get model for the specified architecture.

    Identical to :func:`bioplausible.zoo.mep.benchmarks.compare.get_model`
    except for the ``mlp_small`` variant which uses a 256-unit hidden layer
    here (versus 128 in the un-tuned suite) so the tuned benchmark tracks
    the larger architecture its hyperparameters were calibrated against.
    """
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
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
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
    opt_config: OptimizerConfig,
) -> tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        if is_ep:
            # EP mode
            optimizer.step(x=x, target=y)
            with torch.no_grad():
                output = model(x)
                # Use MSE for EP training (more stable)
                if opt_config.loss_type == "mse":
                    target_onehot = F.one_hot(y, num_classes=output.shape[1]).float()
                    loss = F.mse_loss(output, target_onehot)
                else:
                    loss = F.cross_entropy(output, y)
        else:
            # Standard backprop
            optimizer.zero_grad()
            output = model(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = output.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module, test_loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = F.cross_entropy(output, y)

        total_loss += loss.item() * x.size(0)
        pred = output.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


def run_benchmark(optimizer_name: str, config: BenchmarkConfig) -> OptimizerResult:
    """Run benchmark for a single optimizer."""
    device = get_device(config.device)

    train_loader, test_loader = get_dataloaders(config)
    input_dim = get_input_dim(config)
    num_classes = get_num_classes(config)
    model = get_model(config, input_dim, num_classes).to(device)

    # Get tuned config
    opt_config = OPTIMIZER_CONFIGS.get(optimizer_name, OptimizerConfig(lr=config.lr))

    # Get optimizer with tuned hyperparameters
    optimizer, is_ep = get_optimizer(
        optimizer_name,
        model,
        lr=opt_config.lr,
        weight_decay=config.weight_decay,
        beta=opt_config.beta,
        settle_steps=opt_config.settle_steps,
        settle_lr=opt_config.settle_lr,
        loss_type=opt_config.loss_type,
        ns_steps=opt_config.ns_steps,
        gamma=opt_config.gamma,
        error_beta=opt_config.error_beta,
        use_error_feedback=opt_config.use_error_feedback,
    )

    metrics = []
    start_time = time.time()

    for epoch in range(config.epochs):
        epoch_start = time.time()

        train_loss, train_acc = train_epoch(
            model, optimizer, train_loader, device, is_ep, opt_config
        )
        val_loss, val_acc = evaluate(model, test_loader, device)

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


def run_all_benchmarks(
    config: BenchmarkConfig, optimizers: list[str] | None = None
) -> dict[str, OptimizerResult]:
    """Run benchmarks for all optimizers."""
    if optimizers is None:
        optimizers = ["sgd", "adam", "muon", "eqprop", "smep", "sdmep"]

    results: dict[str, OptimizerResult] = {}
    for opt_name in optimizers:
        logger.info("\n%s", "=" * 60)
        logger.info(
            "Benchmarking: %s (LR=%s)",
            opt_name.upper(),
            OPTIMIZER_CONFIGS[opt_name].lr,
        )
        logger.info("%s", "=" * 60)

        results[opt_name] = run_benchmark(opt_name, config)

    return results


def print_summary(results: dict[str, OptimizerResult]) -> None:
    """Print summary table of results."""
    logger.info("\n%s", "=" * 90)
    logger.info("BENCHMARK SUMMARY (Tuned Hyperparameters)")
    logger.info("%s", "=" * 90)
    logger.info(
        "%-15s %-15s %-18s %-15s %-10s",
        "Optimizer",
        "Best Val Acc",
        "Final Train Acc",
        "Total Time (s)",
        "LR",
    )
    logger.info("%s", "-" * 90)

    sorted_results = sorted(
        results.items(), key=lambda x: x[1].best_val_acc, reverse=True
    )

    for name, result in sorted_results:
        lr = OPTIMIZER_CONFIGS.get(name, OptimizerConfig(lr=0.01)).lr
        logger.info(
            "%-15s %-15.4f %-18.4f %-15.2f %-10.5f",
            name,
            result.best_val_acc,
            result.final_train_acc,
            result.total_time,
            lr,
        )

    logger.info("%s", "=" * 90)

    best = sorted_results[0]
    logger.info(
        "\U0001f3c6 Best performer: %s with %.2f%% validation accuracy",
        best[0].upper(),
        best[1].best_val_acc * 100,
    )

    # Show EP vs backprop comparison
    ep_opts = [
        r
        for r in sorted_results
        if r[0] in {"eqprop", "smep", "sdmep", "local_ep", "natural_ep"}
    ]
    bp_opts = [r for r in sorted_results if r[0] in {"sgd", "adam", "muon"}]

    if ep_opts and bp_opts:
        best_ep = ep_opts[0]
        best_bp = bp_opts[0]
        logger.info("\n\U0001f4ca EP vs Backprop:")
        logger.info(
            "   Best EP:     %s: %.2f%%",
            best_ep[0].upper(),
            best_ep[1].best_val_acc * 100,
        )
        logger.info(
            "   Best Backprop: %s: %.2f%%",
            best_bp[0].upper(),
            best_bp[1].best_val_acc * 100,
        )
        logger.info(
            "   Gap: %.2f%%", (best_bp[1].best_val_acc - best_ep[1].best_val_acc) * 100
        )


def save_results(results: dict[str, OptimizerResult], output_path: str) -> None:
    """Save results to JSON file."""
    data = {}
    for name, result in results.items():
        result_dict = {
            "name": result.name,
            "config": asdict(OPTIMIZER_CONFIGS.get(name, OptimizerConfig(lr=0.01))),
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
    parser = argparse.ArgumentParser(description="Tuned MEP Benchmark Suite")
    parser.add_argument(
        "--dataset", type=str, default="mnist", choices=["mnist", "fashion", "cifar10"]
    )
    parser.add_argument(
        "--model", type=str, default="mlp", choices=["mlp", "mlp_small", "cnn"]
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--subset-train", type=int, default=5000)
    parser.add_argument("--subset-test", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="tuned_benchmark_results.json")
    parser.add_argument(
        "--optimizers",
        type=str,
        nargs="+",
        default=None,
        help="Specific optimizers to benchmark",
    )

    args = parser.parse_args()

    config = BenchmarkConfig(
        dataset=args.dataset,
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        subset_train=args.subset_train,
        subset_test=args.subset_test,
        device=args.device,
    )

    logger.info("%s", "=" * 60)
    logger.info("TUNED MEP BENCHMARK SUITE")
    logger.info("%s", "=" * 60)
    logger.info("Dataset: %s", config.dataset)
    logger.info("Model: %s", config.model)
    logger.info("Epochs: %s", config.epochs)
    logger.info("Device: %s", config.device)
    logger.info("%s", "=" * 60)

    results = run_all_benchmarks(config, args.optimizers)
    print_summary(results)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
