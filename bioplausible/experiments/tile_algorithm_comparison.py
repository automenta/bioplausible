"""Tile Algorithm Family Comparison — Fair Comparison on Same Substrate.

Compares PC vs EP vs FA vs TP vs Hebbian vs SNN on the same tile substrate,
isolating credit assignment mechanism as the only variable.

Usage:
    python -m bioplausible.experiments.tile_algorithm_comparison --tasks mnist,cifar10 --algorithms ep,fa,tp,pc,hebbian,snn,backprop --seeds 5
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from bioplausible.core.registry import ComponentCategory, Registry
from bioplausible.core.trainer import CoreTrainer, TrainerConfig
from bioplausible.utils import seed_everything
from bioplausible.validation.statistics import (
    bootstrap_ci,
    cliffs_delta,
    cohens_d,
    permutation_test_p,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class TileAlgorithmConfig:
    """Configuration for tile algorithm comparison."""

    tasks: list[str] = field(
        default_factory=lambda: ["mnist", "cifar10", "tiny_shakespeare"]
    )
    algorithms: list[str] = field(
        default_factory=lambda: ["ep", "fa", "tp", "pc", "hebbian", "snn", "backprop"]
    )
    substrates: list[str] = field(
        default_factory=lambda: ["tile"]
    )  # Could extend to other substrates
    seeds: int = 5
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    output_dir: str = "results/tile_algorithm_comparison"
    device: str = "auto"
    quick_mode: bool = False
    fixed_width: int = 128
    fixed_depth: int = 3


# Algorithm to model mapping (all using tile substrate)
ALGORITHM_MODELS = {
    "ep": "conv_tile",  # will use algorithm=ep
    "fa": "conv_tile_fa",
    "tp": "conv_tile_tp",
    "pc": "conv_tile_pc",
    "hebbian": "conv_tile_hebbian",
    "snn": "conv_tile_snn",
    "backprop": "backprop_mlp",
}

# For LM tasks
LM_ALGORITHM_MODELS = {
    "ep": "tile_lm",  # will use algorithm=ep
    "fa": "tile_lm",  # with algorithm=fa
    "tp": "tile_lm",  # with algorithm=tp
    "pc": "tile_lm",  # with algorithm=pc
    "hebbian": "tile_lm",  # with algorithm=hebbian
    "snn": "tile_lm",  # with algorithm=snn
    "backprop": "backprop_lm",
}


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _get_model_for_task(algorithm: str, task: str) -> tuple[str, dict]:
    """Get model name and algorithm-specific kwargs for task."""
    if task == "tiny_shakespeare":
        model_name = LM_ALGORITHM_MODELS.get(algorithm, algorithm)
        model_kwargs = {"algorithm": algorithm} if algorithm != "backprop" else {}
    else:
        model_name = ALGORITHM_MODELS.get(algorithm, algorithm)
        model_kwargs = {}  # algorithm is in model name for conv_tile variants

    return model_name, model_kwargs


def _create_trainer_config(
    algorithm: str,
    task: str,
    seed: int,
    config: TileAlgorithmConfig,
) -> TrainerConfig:
    """Create trainer config with fixed architecture (width/depth)."""
    model_name, algo_kwargs = _get_model_for_task(algorithm, task)

    # Fixed architecture for fair comparison
    base_kwargs = {
        "neurons_per_tile": config.fixed_width // 4,
        "tiles_per_layer": 4,
        "num_hidden_layers": config.fixed_depth,
        "learning_rate": config.learning_rate,
    }
    base_kwargs.update(algo_kwargs)

    # Task-specific dims
    if task in ("mnist", "fashion_mnist"):
        base_kwargs.update({
            "input_channels": 1,
            "input_size": 28,
            "num_classes": 10,
        })
    elif task == "cifar10":
        base_kwargs.update({
            "input_channels": 3,
            "input_size": 32,
            "num_classes": 10,
        })
    elif task == "tiny_shakespeare":
        base_kwargs.update({
            "vocab_size": 1000,
            "embed_dim": config.fixed_width,
            "num_layers": config.fixed_depth,
        })

    return TrainerConfig(
        model=model_name,
        task=task,
        epochs=config.epochs if not config.quick_mode else 3,
        batch_size=config.batch_size,
        optimizer_kwargs={"lr": config.learning_rate},
        model_kwargs=base_kwargs,
        device=config.device,
        quick_mode=config.quick_mode,
    )


def _run_single_experiment(
    algorithm: str,
    task: str,
    seed: int,
    config: TileAlgorithmConfig,
) -> dict:
    """Run a single algorithm comparison experiment."""
    seed_everything(seed)

    trainer_config = _create_trainer_config(algorithm, task, seed, config)

    # Verify model is registered
    try:
        Registry.get_metadata(ComponentCategory.MODEL, trainer_config.model)
    except KeyError:
        logger.warning("Model %s not registered, skipping", trainer_config.model)
        return {
            "algorithm": algorithm,
            "model": trainer_config.model,
            "task": task,
            "seed": seed,
            "accuracy": 0.0,
            "loss": float("inf"),
            "time": 0.0,
            "params": 0,
            "flops": 0,
            "memory_mb": 0,
            "success": False,
            "error": f"Model {trainer_config.model} not registered",
        }

    trainer = CoreTrainer(trainer_config)
    start_time = time.time()
    history = trainer.fit()
    elapsed = time.time() - start_time

    if not history:
        return {
            "algorithm": algorithm,
            "model": trainer_config.model,
            "task": task,
            "seed": seed,
            "accuracy": 0.0,
            "loss": float("inf"),
            "time": elapsed,
            "params": 0,
            "flops": 0,
            "memory_mb": 0,
            "success": False,
        }

    final = history[-1]
    return {
        "algorithm": algorithm,
        "model": trainer_config.model,
        "task": task,
        "seed": seed,
        "accuracy": final.val_acc if hasattr(final, "val_acc") else final.accuracy,
        "loss": final.val_loss if hasattr(final, "val_loss") else final.loss,
        "time": elapsed,
        "params": final.param_count if hasattr(final, "param_count") else 0,
        "flops": getattr(final, "flops", 0),
        "memory_mb": getattr(final, "memory_mb", 0),
        "success": True,
    }


def run_tile_algorithm_comparison(config: TileAlgorithmConfig) -> list[dict]:
    """Run tile algorithm family comparison."""
    device = _resolve_device(config.device)
    config = TileAlgorithmConfig(**{**config.__dict__, "device": device})

    results = []
    total = len(config.tasks) * len(config.algorithms) * config.seeds
    logger.info("Tile Algorithm Comparison: %d total experiments", total)
    logger.info("Tasks: %s", config.tasks)
    logger.info("Algorithms: %s", config.algorithms)
    logger.info(
        "Fixed architecture: width=%d, depth=%d", config.fixed_width, config.fixed_depth
    )

    exp_count = 0
    for task in config.tasks:
        for algorithm in config.algorithms:
            for seed in range(config.seeds):
                exp_count += 1
                logger.info(
                    "[%d/%d] %s on %s (seed=%d)",
                    exp_count,
                    total,
                    algorithm,
                    task,
                    seed,
                )

                result = _run_single_experiment(algorithm, task, seed, config)
                results.append(result)

    return results


def _analyze_algorithm_performance(results: list[dict]) -> dict:
    """Analyze and compare algorithm performance."""
    import pandas as pd

    df = pd.DataFrame(results)
    if df.empty:
        return {}

    df = df[df["success"]].copy()
    analysis = {}

    for task in df["task"].unique():
        task_df = df[df["task"] == task]
        task_analysis = {}

        # Overall statistics per algorithm
        for algorithm in task_df["algorithm"].unique():
            algo_df = task_df[task_df["algorithm"] == algorithm]
            accs = algo_df["accuracy"].values

            task_analysis[algorithm] = {
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy": float(np.std(accs)),
                "median_accuracy": float(np.median(accs)),
                "min_accuracy": float(np.min(accs)),
                "max_accuracy": float(np.max(accs)),
                "mean_params": float(algo_df["params"].mean()),
                "mean_time": float(algo_df["time"].mean()),
                "mean_flops": float(algo_df["flops"].mean())
                if "flops" in algo_df.columns
                else 0,
                "mean_memory": float(algo_df["memory_mb"].mean())
                if "memory_mb" in algo_df.columns
                else 0,
                "n_seeds": len(accs),
                "ci_95": list(bootstrap_ci(accs)) if len(accs) >= 2 else [0, 0],
            }

        # Pairwise comparisons
        algorithms = sorted(task_df["algorithm"].unique())
        comparisons = {}

        for i, algo1 in enumerate(algorithms):
            for algo2 in algorithms[i + 1 :]:
                df1 = task_df[task_df["algorithm"] == algo1]["accuracy"].values
                df2 = task_df[task_df["algorithm"] == algo2]["accuracy"].values

                if len(df1) >= 2 and len(df2) >= 2:
                    mean_diff = np.mean(df1) - np.mean(df2)
                    d = cohens_d(df1, df2)
                    delta = cliffs_delta(df1, df2)
                    p_val = permutation_test_p(df1, df2, n_permutations=1000)

                    comparisons[f"{algo1}_vs_{algo2}"] = {
                        "mean_diff": float(mean_diff),
                        "cohens_d": float(d),
                        "cliffs_delta": float(delta),
                        "p_value": float(p_val),
                        "significant": p_val < 0.05,
                        "winner": algo1 if mean_diff > 0 else algo2,
                    }

        # Ranking by accuracy
        ranked = sorted(task_analysis.items(), key=lambda x: -x[1]["mean_accuracy"])

        task_analysis["_ranking"] = [
            {"rank": i + 1, "algorithm": algo, **stats}
            for i, (algo, stats) in enumerate(ranked)
        ]
        task_analysis["_comparisons"] = comparisons

        analysis[task] = task_analysis

    return analysis


def _compute_bio_plausibility_scores(results: list[dict]) -> dict:
    """Compute bio-plausibility weighted scores."""
    import pandas as pd

    df = pd.DataFrame(results)
    if df.empty:
        return {}

    df = df[df["success"]].copy()

    # Bio-plausibility weights (from registry)
    bio_scores = {
        "ep": 0.8,
        "pc": 0.75,
        "fa": 0.7,
        "tp": 0.65,
        "hebbian": 0.6,
        "snn": 0.7,
        "backprop": 0.1,
    }

    scores = {}
    for task in df["task"].unique():
        task_df = df[df["task"] == task]
        task_scores = {}

        for algorithm in task_df["algorithm"].unique():
            algo_df = task_df[task_df["algorithm"] == algorithm]
            acc = algo_df["accuracy"].mean()
            bio = bio_scores.get(algorithm, 0.5)
            # Combined score: accuracy * bio_plausibility
            combined = acc * bio

            task_scores[algorithm] = {
                "accuracy": float(acc),
                "bio_plausibility": float(bio),
                "combined_score": float(combined),
            }

        # Rank by combined score
        ranked = sorted(task_scores.items(), key=lambda x: -x[1]["combined_score"])
        task_scores["_bio_ranking"] = [
            {"rank": i + 1, "algorithm": algo, **stats}
            for i, (algo, stats) in enumerate(ranked)
        ]

        scores[task] = task_scores

    return scores


def _save_results(results: list[dict], output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with Path(output_path / "raw_results.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    import pandas as pd

    df = pd.DataFrame(results)
    df.to_parquet(output_path / "results.parquet", index=False)

    logger.info("Saved results to %s", output_path)


def _generate_report(
    analysis: dict,
    bio_scores: dict,
    output_dir: str,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with Path(output_path / "algorithm_comparison_report.md").open("w") as f:
        f.write("# Tile Algorithm Family Comparison Report\n\n")

        f.write("## Performance Ranking (by Accuracy)\n\n")
        for task, task_analysis in analysis.items():
            f.write(f"### {task}\n\n")
            f.write(
                "| Rank | Algorithm | Accuracy | Std | Params | Time (s) | 95% CI |\n"
            )
            f.write(
                "|------|-----------|----------|-----|--------|----------|--------|\n"
            )
            for entry in task_analysis.get("_ranking", []):
                ci = entry.get("ci_95", [0, 0])
                f.write(
                    f"| {entry['rank']} | {entry['algorithm']} | "
                    f"{entry['mean_accuracy']:.4f} | {entry['std_accuracy']:.4f} | "
                    f"{entry['mean_params']:.0f} | {entry['mean_time']:.1f} | "
                    f"[{ci[0]:.4f}, {ci[1]:.4f}] |\n"
                )
            f.write("\n")

        f.write("## Pairwise Statistical Comparisons\n\n")
        for task, task_analysis in analysis.items():
            f.write(f"### {task}\n\n")
            f.write(
                "| Comparison | Mean Diff | Cohen's d | Cliff's δ | p-value | Significant |\n"
            )
            f.write(
                "|------------|-----------|-----------|-----------|---------|-------------|\n"
            )
            for comp, stats in task_analysis.get("_comparisons", {}).items():
                f.write(
                    f"| {comp} | {stats['mean_diff']:+.4f} | {stats['cohens_d']:.2f} | "
                    f"{stats['cliffs_delta']:.2f} | {stats['p_value']:.4f} | "
                    f"{'✓' if stats['significant'] else '✗'} |\n"
                )
            f.write("\n")

        f.write("## Bio-Plausibility Weighted Ranking\n\n")
        for task, task_scores in bio_scores.items():
            f.write(f"### {task}\n\n")
            f.write(
                "| Rank | Algorithm | Accuracy | Bio-Plausibility | Combined Score |\n"
            )
            f.write(
                "|------|-----------|----------|------------------|----------------|\n"
            )
            for entry in task_scores.get("_bio_ranking", []):
                f.write(
                    f"| {entry['rank']} | {entry['algorithm']} | "
                    f"{entry['accuracy']:.4f} | {entry['bio_plausibility']:.2f} | "
                    f"{entry['combined_score']:.4f} |\n"
                )
            f.write("\n")

        f.write("## Summary\n\n")
        f.write("**Key Findings:**\n\n")
        for task, task_analysis in analysis.items():
            ranking = task_analysis.get("_ranking", [])
            if ranking:
                best = ranking[0]
                f.write(
                    f"- **{task}**: Best algorithm is **{best['algorithm']}** "
                    f"(accuracy={best['mean_accuracy']:.4f}±{best['std_accuracy']:.4f})"
                )
                if len(ranking) > 1:
                    second = ranking[1]
                    gap = (best["mean_accuracy"] - second["mean_accuracy"]) * 100
                    f.write(f", {gap:.1f}pp ahead of {second['algorithm']}")
                f.write("\n")

        f.write("\n")
        f.write("**Bio-Plausibility Winners:**\n\n")
        for task, task_scores in bio_scores.items():
            ranking = task_scores.get("_bio_ranking", [])
            if ranking:
                best = ranking[0]
                f.write(
                    f"- **{task}**: **{best['algorithm']}** "
                    f"(combined score={best['combined_score']:.4f})"
                )
                f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Tile Algorithm Family Comparison")
    parser.add_argument(
        "--tasks",
        default="mnist,cifar10,tiny_shakespeare",
        help="Comma-separated tasks",
    )
    parser.add_argument(
        "--algorithms",
        default="ep,fa,tp,pc,hebbian,snn,backprop",
        help="Comma-separated algorithms",
    )
    parser.add_argument("--seeds", type=int, default=5, help="Seeds per config")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs per run")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--width",
        type=int,
        default=128,
        help="Fixed width (neurons_per_tile * tiles_per_layer)",
    )
    parser.add_argument(
        "--depth", type=int, default=3, help="Fixed depth (num_hidden_layers)"
    )
    parser.add_argument(
        "--output-dir",
        default="results/tile_algorithm_comparison",
        help="Output directory",
    )
    parser.add_argument("--device", default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument("--quick", action="store_true", help="Quick mode")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config = TileAlgorithmConfig(
        tasks=args.tasks.split(","),
        algorithms=args.algorithms.split(","),
        seeds=args.seeds,
        epochs=args.epochs if not args.quick else 3,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        fixed_width=args.width,
        fixed_depth=args.depth,
        output_dir=args.output_dir,
        device=args.device,
        quick_mode=args.quick,
    )

    logger.info("Starting Tile Algorithm Family Comparison")

    # Run experiments
    results = run_tile_algorithm_comparison(config)

    # Save results
    _save_results(results, config.output_dir)

    # Analyze performance
    analysis = _analyze_algorithm_performance(results)
    with Path(Path(config.output_dir) / "performance_analysis.json").open("w") as f:
        json.dump(analysis, f, indent=2, default=str)

    # Compute bio-plausibility scores
    bio_scores = _compute_bio_plausibility_scores(results)
    with Path(Path(config.output_dir) / "bio_plausibility_scores.json").open("w") as f:
        json.dump(bio_scores, f, indent=2, default=str)

    # Generate report
    _generate_report(analysis, bio_scores, config.output_dir)

    logger.info("Tile Algorithm Comparison complete. Results in %s", config.output_dir)


if __name__ == "__main__":
    main()
