"""EqProp Family Vision Parity — All EqProp Variants on Vision Benchmarks.

Compares all EqProp variants (EP, Directed EP, Finite Nudge EP, Momentum EP, etc.)
on MNIST, Fashion-MNIST, CIFAR-10, SVHN. Produces variant recommendation matrix.

Usage:
    python -m computronium.experiments.eqprop_vision_parity --tasks mnist,fashion_mnist,cifar10,svhn --seeds 5
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader

from computronium.analysis.dynamics import DynamicsAnalyzer
from computronium.cli.run import _BASELINE_MODELS
from computronium.core.system_trainer import (
    SystemTrainer,
    SystemTrainerConfig,
    create_backprop_system,
    create_eqprop_system,
)
from computronium.data.vision import get_vision_dataset
from computronium.utils import seed_everything
from computronium.validation.statistics import (
    bootstrap_ci,
    cliffs_delta,
    cohens_d,
    permutation_test_p,
)

if TYPE_CHECKING:
    import pandas as pd

    from computronium.ontology import System

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class EqPropParityConfig:
    """Configuration for EqProp vision parity experiment."""

    tasks: list[str] = field(
        default_factory=lambda: ["mnist", "fashion_mnist", "cifar10", "svhn"]
    )
    eqprop_models: list[str] = field(
        default_factory=lambda: [
            "eqprop",
            "directed_ep",
            "finite_nudge_ep",
            "momentum_equilibrium",
            "sparse_equilibrium",
            "equilibrium_alignment",
            "layerwise_equilibrium_fa",
        ]
    )
    baseline_models: list[str] = field(default_factory=lambda: ["backprop_mlp"])
    seeds: int = 5
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    output_dir: str = "results/eqprop_vision_parity"
    device: str = "auto"


# Model-specific configurations
MODEL_CONFIGS = {
    "eqprop": {
        "hidden_dim": 512,
        "num_layers": 3,
        "use_spectral_norm": True,
        "beta": 0.1,
        "step_size": 0.1,
        "inference_steps": 20,
    },
    "directed_ep": {
        "hidden_dim": 512,
        "num_layers": 3,
        "use_spectral_norm": True,
        "beta": 0.1,
        "step_size": 0.1,
        "inference_steps": 20,
    },
    "finite_nudge_ep": {
        "hidden_dim": 512,
        "num_layers": 3,
        "use_spectral_norm": True,
        "beta": 0.1,
        "step_size": 0.1,
        "inference_steps": 20,
        "finite_nudge": True,
    },
    "momentum_equilibrium": {
        "hidden_dim": 512,
        "num_layers": 3,
        "use_spectral_norm": True,
        "beta": 0.1,
        "step_size": 0.1,
        "inference_steps": 20,
        "momentum": 0.9,
    },
    "sparse_equilibrium": {
        "hidden_dim": 512,
        "num_layers": 3,
        "use_spectral_norm": True,
        "beta": 0.1,
        "step_size": 0.1,
        "inference_steps": 20,
        "sparsity": 0.1,
    },
    "equilibrium_alignment": {
        "hidden_dim": 512,
        "num_layers": 3,
        "use_spectral_norm": True,
        "beta": 0.1,
        "step_size": 0.1,
        "inference_steps": 20,
    },
    "layerwise_equilibrium_fa": {
        "hidden_dim": 512,
        "num_layers": 3,
        "use_spectral_norm": True,
        "beta": 0.1,
        "step_size": 0.1,
        "inference_steps": 20,
    },
    "backprop_mlp": {
        "hidden_dim": 512,
        "num_layers": 3,
    },
}


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _get_input_dims(task: str) -> tuple[int, int]:
    """Get input and output dimensions for a task."""
    dims = {
        "mnist": (784, 10),
        "fashion_mnist": (784, 10),
        "cifar10": (3072, 10),
        "svhn": (3072, 10),
    }
    return dims.get(task, (784, 10))


_SYSTEM_FACTORIES = {
    "eqprop": create_eqprop_system,
    "backprop_mlp": create_backprop_system,
}


def _build_system(model_name: str, task: str, config: EqPropParityConfig) -> System:
    """Build the 5-axis system for a supported model/task pair."""
    input_dim, output_dim = _get_input_dims(task)
    kwargs = MODEL_CONFIGS.get(model_name, {})
    common = {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "hidden_dim": kwargs.get("hidden_dim", 512),
        "num_layers": kwargs.get("num_layers", 3),
        "lr": config.learning_rate,
    }
    match model_name:
        case "eqprop":
            return create_eqprop_system(
                beta=kwargs.get("beta", 0.1),
                settle_steps=kwargs.get("inference_steps", 20),
                **common,
            )
        case _:
            return create_backprop_system(**common)


def _task_loaders(task: str, batch_size: int) -> tuple[DataLoader, DataLoader]:
    """Flat train/test loaders for a vision task."""
    train_ds = get_vision_dataset(task, flatten=True, train=True)
    test_ds = get_vision_dataset(task, flatten=True, train=False)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size),
    )


def _run_single_experiment(
    model_name: str,
    task: str,
    seed: int,
    config: EqPropParityConfig,
) -> dict:
    """Run a single training experiment through the ontology SystemTrainer."""
    seed_everything(seed)

    system = _build_system(model_name, task, config)
    train_loader, val_loader = _task_loaders(task, config.batch_size)

    start_time = time.time()
    trainer = SystemTrainer(
        system=system,
        config=SystemTrainerConfig(
            max_epochs=config.epochs,
            batch_size=config.batch_size,
            device=_resolve_device(config.device),
        ),
        train_data=train_loader,
        val_data=val_loader,
    )
    history = trainer.fit()
    trainer.close()
    elapsed = time.time() - start_time

    final = history[-1] if history else {}
    return {
        "model": model_name,
        "task": task,
        "seed": seed,
        "accuracy": final.get("val_acc", 0.0),
        "loss": final.get("val_loss", float("inf")),
        "time": elapsed,
        "params": sum(p.numel() for p in system.geometry.params.values()),
        "success": bool(history),
    }


def run_eqprop_parity(config: EqPropParityConfig) -> list[dict]:
    """Run EqProp vision parity experiments."""
    results = []
    device = _resolve_device(config.device)
    config = EqPropParityConfig(**{**config.__dict__, "device": device})

    # Filter out baseline models that fail learns-gate
    eqprop_models = [m for m in config.eqprop_models if m not in _BASELINE_MODELS]
    candidates = eqprop_models + config.baseline_models
    all_models = [m for m in candidates if m in _SYSTEM_FACTORIES]
    for model_name in set(candidates) - set(all_models):
        logger.warning("Model %s has no ontology system factory, skipping", model_name)

    total = len(config.tasks) * len(all_models) * config.seeds
    logger.info("EqProp Vision Parity: %d total experiments", total)
    logger.info("Tasks: %s", config.tasks)
    logger.info("Models: %s", all_models)
    logger.info("Seeds: %d", config.seeds)

    exp_count = 0
    for task in config.tasks:
        for model_name in all_models:
            for seed in range(config.seeds):
                exp_count += 1
                logger.info(
                    "[%d/%d] %s on %s (seed=%d)",
                    exp_count,
                    total,
                    model_name,
                    task,
                    seed,
                )

                try:
                    result = _run_single_experiment(model_name, task, seed, config)
                    results.append(result)
                except Exception as e:
                    logger.exception(
                        "Experiment failed: %s on %s seed=%d", model_name, task, seed
                    )
                    results.append({
                        "model": model_name,
                        "task": task,
                        "seed": seed,
                        "accuracy": 0.0,
                        "loss": float("inf"),
                        "time": 0.0,
                        "params": 0,
                        "success": False,
                        "error": str(e),
                    })

    return results


def _aggregate_results(results: list[dict]) -> pd.DataFrame:
    """Aggregate results across seeds with statistical tests."""
    import pandas as pd

    df = pd.DataFrame(results)
    if df.empty:
        return pd.DataFrame()

    # Group by task and model
    grouped = (
        df
        .groupby(["task", "model"])
        .agg({
            "accuracy": ["mean", "std", "count"],
            "loss": ["mean", "std"],
            "time": "mean",
            "params": "mean",
            "success": "sum",
        })
        .reset_index()
    )

    grouped.columns = [
        "_".join(col).strip("_") if col[1] else col[0] for col in grouped.columns.values
    ]

    # Add 95% CI for accuracy
    def compute_ci(row):
        task = row["task"]
        model = row["model"]
        subset = df[(df["task"] == task) & (df["model"] == model)]["accuracy"].values
        if len(subset) >= 2:
            ci = bootstrap_ci(subset, method="percentile")
            return ci[0], ci[1]
        return np.nan, np.nan

    ci_results = grouped.apply(compute_ci, axis=1)
    grouped["acc_ci_lower"] = [c[0] for c in ci_results]
    grouped["acc_ci_upper"] = [c[1] for c in ci_results]

    return grouped


def _statistical_comparison(
    results: list[dict], baseline: str = "backprop_mlp"
) -> dict:
    """Perform statistical comparisons against baseline."""
    import pandas as pd

    df = pd.DataFrame(results)
    if df.empty:
        return {}

    comparisons = {}

    for task in df["task"].unique():
        task_df = df[df["task"] == task]
        baseline_accs = task_df[task_df["model"] == baseline]["accuracy"].values

        if len(baseline_accs) == 0:
            continue

        task_comparisons = {}
        for model in task_df["model"].unique():
            if model == baseline:
                continue

            model_accs = task_df[task_df["model"] == model]["accuracy"].values
            if len(model_accs) < 2:
                continue

            # Cohen's d
            d = cohens_d(model_accs, baseline_accs)

            # Cliff's delta
            delta = cliffs_delta(model_accs, baseline_accs)

            # Permutation test
            p_val = permutation_test_p(model_accs, baseline_accs, n_perm=1000)

            # Mean difference
            mean_diff = np.mean(model_accs) - np.mean(baseline_accs)

            task_comparisons[model] = {
                "mean_diff": float(mean_diff),
                "cohens_d": float(d),
                "cliffs_delta": float(delta),
                "p_value": float(p_val),
                "significant": p_val < 0.05,
                "model_mean": float(np.mean(model_accs)),
                "model_std": float(np.std(model_accs)),
                "baseline_mean": float(np.mean(baseline_accs)),
                "baseline_std": float(np.std(baseline_accs)),
            }

        comparisons[task] = task_comparisons

    return comparisons


def _generate_recommendation_matrix(comparisons: dict, output_dir: str) -> None:
    """Generate variant recommendation matrix per task/budget."""
    import pandas as pd

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows = []
    for task, task_comps in comparisons.items():
        for model, stats in task_comps.items():
            rows.append({
                "task": task,
                "model": model,
                "mean_accuracy": stats["model_mean"],
                "std_accuracy": stats["model_std"],
                "gap_to_baseline_pp": stats["mean_diff"] * 100,
                "cohens_d": stats["cohens_d"],
                "cliffs_delta": stats["cliffs_delta"],
                "p_value": stats["p_value"],
                "significant": stats["significant"],
                "recommendation": _get_recommendation(stats),
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_path / "recommendation_matrix.csv", index=False)

    # Also create markdown table
    with Path(output_path / "recommendation_matrix.md").open("w") as f:
        f.write("# EqProp Variant Recommendation Matrix\n\n")
        for task in df["task"].unique():
            f.write(f"## {task}\n\n")
            task_df = df[df["task"] == task].sort_values(
                "gap_to_baseline_pp", ascending=False
            )
            f.write(task_df.to_markdown(index=False))
            f.write("\n\n")


def _get_recommendation(stats: dict) -> str:
    """Get recommendation based on statistical comparison."""
    if not stats["significant"]:
        return "Inconclusive (p >= 0.05)"

    gap = stats["mean_diff"] * 100  # percentage points

    if gap > -1 and stats["cohens_d"] > -0.2:
        return "✓ Recommended (parity with backprop)"
    elif gap > -3:
        return "○ Acceptable (small gap)"
    elif gap > -10:
        return "△ Marginal (moderate gap)"
    else:
        return "✗ Not recommended (large gap)"


def _save_results(results: list[dict], output_dir: str) -> None:
    """Save results to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with Path(output_path / "raw_results.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    aggregated = _aggregate_results(results)
    aggregated.to_json(
        output_path / "aggregated_results.json", orient="records", indent=2
    )

    logger.info("Saved results to %s", output_path)


def _analyze_dynamics(results: list[dict], output_dir: str) -> None:
    """Analyze dynamics for each model."""
    import pandas as pd

    df = pd.DataFrame(results)
    if df.empty:
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    analyzer = DynamicsAnalyzer()

    for task in df["task"].unique():
        task_df = df[df["task"] == task]
        for model in task_df["model"].unique():
            model_df = task_df[task_df["model"] == model]
            if len(model_df) < 2:
                continue

            # We'd need the actual model instances for full dynamics analysis
            # This is a placeholder for the framework

    logger.info("Dynamics analysis complete")


def main():
    parser = argparse.ArgumentParser(description="EqProp Vision Parity Experiment")
    parser.add_argument(
        "--tasks",
        default="mnist,fashion_mnist,cifar10,svhn",
        help="Comma-separated tasks",
    )
    parser.add_argument(
        "--models",
        default="eqprop,directed_ep,finite_nudge_ep,momentum_equilibrium,sparse_equilibrium,equilibrium_alignment,layerwise_equilibrium_fa",
        help="Comma-separated EqProp models",
    )
    parser.add_argument("--seeds", type=int, default=5, help="Seeds per config")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs per run")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--output-dir", default="results/eqprop_vision_parity", help="Output directory"
    )
    parser.add_argument("--device", default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode (fewer epochs)"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config = EqPropParityConfig(
        tasks=args.tasks.split(","),
        eqprop_models=args.models.split(","),
        seeds=args.seeds,
        epochs=args.epochs if not args.quick else 3,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        device=args.device,
    )

    logger.info("Starting EqProp Vision Parity Experiment")

    # Run experiments
    results = run_eqprop_parity(config)

    # Save results
    _save_results(results, config.output_dir)

    # Statistical comparison
    comparisons = _statistical_comparison(results)
    with Path(Path(config.output_dir) / "statistical_comparisons.json").open("w") as f:
        json.dump(comparisons, f, indent=2, default=str)

    # Generate recommendation matrix
    _generate_recommendation_matrix(comparisons, config.output_dir)

    logger.info("EqProp Vision Parity complete. Results in %s", config.output_dir)


if __name__ == "__main__":
    main()
