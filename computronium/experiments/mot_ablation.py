"""Mixture-of-Tiles (MoT) Ablation — Dense vs Sparse Tile Routing.

Tests whether sparse routing helps or just adds overhead compared to dense tile routing.
Uses OptimizedLMEquiTile as the base.

Usage:
    python -m computronium.experiments.mot_ablation --tasks mnist,cifar10 --routing dense,sparse,topk --seeds 5
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from computronium.utils import seed_everything
from computronium.validation.statistics import (
    cohens_d,
    permutation_test_p,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class MoTAblationConfig:
    """Configuration for MoT ablation experiment."""

    tasks: list[str] = field(
        default_factory=lambda: ["mnist", "cifar10", "tiny_shakespeare"]
    )
    routing_modes: list[str] = field(
        default_factory=lambda: ["dense", "sparse", "topk", "random"]
    )
    tile_algorithms: list[str] = field(
        default_factory=lambda: ["ep", "fa", "pc", "hebbian"]
    )
    num_tiles: list[int] = field(default_factory=lambda: [4, 8, 16, 32])
    topk_values: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    seeds: int = 5
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    output_dir: str = "results/mot_ablation"
    device: str = "auto"
    quick_mode: bool = False


# Routing configurations
ROUTING_CONFIGS = {
    "dense": {"sparse_routing": False, "top_k": None},
    "sparse": {"sparse_routing": True, "top_k": 2},
    "topk": {"sparse_routing": True, "top_k": 4},
    "random": {"sparse_routing": True, "top_k": 2, "random_routing": True},
}


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _create_mot_config(
    routing_mode: str,
    tile_algorithm: str,
    num_tiles: int,
    top_k: int | None,
    task: str,
) -> dict:
    """Create model kwargs for MoT experiment."""
    config = {
        "algorithm": tile_algorithm,
        "neurons_per_tile": 64,
        "tiles_per_layer": num_tiles,
        "num_hidden_layers": 3,
    }
    config.update(ROUTING_CONFIGS.get(routing_mode, {}))

    if top_k is not None:
        config["top_k"] = top_k

    # Task-specific dims
    if task in ("mnist", "fashion_mnist"):  # ruff: ignore[literal-membership]
        config["input_dim"] = 784
        config["output_dim"] = 10
    elif task == "cifar10":
        config["input_dim"] = 3072
        config["output_dim"] = 10
    elif task == "tiny_shakespeare":
        config["input_dim"] = 256  # embed_dim
        config["output_dim"] = 256
        config["vocab_size"] = 1000

    return config


def _get_model_name(routing_mode: str, tile_algorithm: str) -> str:
    """Get registered model name."""
    # Try to find a model that supports MoT
    base_name = f"mot_{tile_algorithm}"
    return base_name


def _run_single_mot_experiment(
    routing_mode: str,
    tile_algorithm: str,
    num_tiles: int,
    top_k: int | None,
    task: str,
    seed: int,
    config: MoTAblationConfig,
) -> dict:
    """Run a single MoT experiment."""
    seed_everything(seed)

    model_kwargs = _create_mot_config(
        routing_mode, tile_algorithm, num_tiles, top_k, task
    )

    # Try to find a suitable registered model
    model_name = "tile_lm" if task == "tiny_shakespeare" else "conv_tile"
    # Add routing-specific suffix
    if routing_mode != "dense":
        model_name = f"{model_name}_{routing_mode}"

    from computronium.core.trainer import TrainerConfig

    trainer_config = TrainerConfig(
        model=model_name,
        task=task,
        epochs=config.epochs if not config.quick_mode else 3,
        batch_size=config.batch_size,
        optimizer_kwargs={"lr": config.learning_rate},
        model_kwargs=model_kwargs,
        device=config.device,
        quick_mode=config.quick_mode,
    )

    from computronium.core.trainer import CoreTrainer

    trainer = CoreTrainer(trainer_config)
    start_time = time.time()
    history = trainer.fit()
    elapsed = time.time() - start_time

    if not history:
        return {
            "routing_mode": routing_mode,
            "tile_algorithm": tile_algorithm,
            "num_tiles": num_tiles,
            "top_k": top_k,
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
        "routing_mode": routing_mode,
        "tile_algorithm": tile_algorithm,
        "num_tiles": num_tiles,
        "top_k": top_k,
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


def run_mot_ablation(config: MoTAblationConfig) -> list[dict]:
    """Run MoT ablation experiments."""
    device = _resolve_device(config.device)
    config = MoTAblationConfig(**{**config.__dict__, "device": device})

    results = []
    total = (
        len(config.tasks)
        * len(config.routing_modes)
        * len(config.tile_algorithms)
        * len(config.num_tiles)
        * max(1, len(config.topk_values))
        * config.seeds
    )
    logger.info("MoT Ablation: ~%d total experiments", total)

    exp_count = 0
    for task in config.tasks:  # ruff: ignore[too-many-nested-blocks]
        for routing_mode in config.routing_modes:
            for tile_algorithm in config.tile_algorithms:
                for num_tiles in config.num_tiles:
                    top_k_values = (
                        config.topk_values
                        if routing_mode in ("sparse", "topk", "random")  # ruff: ignore[literal-membership]
                        else [None]
                    )
                    for top_k in top_k_values:
                        for seed in range(config.seeds):
                            exp_count += 1
                            logger.info(
                                "[%d] %s/%s tiles=%d topk=%s on %s (seed=%d)",
                                exp_count,
                                routing_mode,
                                tile_algorithm,
                                num_tiles,
                                top_k,
                                task,
                                seed,
                            )

                            result = _run_single_mot_experiment(
                                routing_mode,
                                tile_algorithm,
                                num_tiles,
                                top_k,
                                task,
                                seed,
                                config,
                            )
                            results.append(result)

    return results


def _analyze_routing_efficiency(results: list[dict]) -> dict:  # ruff: ignore[too-many-locals]
    """Analyze sparse vs dense routing efficiency."""
    import pandas as pd

    df = pd.DataFrame(results)
    if df.empty:
        return {}

    df = df[df["success"]].copy()
    analysis = {}

    for task in df["task"].unique():
        task_df = df[df["task"] == task]
        task_analysis = {}

        # Compare dense vs each sparse mode
        dense_df = task_df[task_df["routing_mode"] == "dense"]
        if dense_df.empty:
            continue

        dense_acc = dense_df["accuracy"].mean()
        dense_params = dense_df["params"].mean()
        dense_time = dense_df["time"].mean()
        dense_flops = dense_df["flops"].mean() if "flops" in dense_df.columns else 0

        for routing_mode in ["sparse", "topk", "random"]:
            sparse_df = task_df[task_df["routing_mode"] == routing_mode]
            if sparse_df.empty:
                continue

            # Aggregate by algorithm and tiles
            for algorithm in sparse_df["tile_algorithm"].unique():
                for num_tiles in sparse_df["num_tiles"].unique():
                    subset = sparse_df[
                        (sparse_df["tile_algorithm"] == algorithm)
                        & (sparse_df["num_tiles"] == num_tiles)
                    ]
                    if subset.empty:
                        continue

                    sparse_acc = subset["accuracy"].mean()
                    sparse_params = subset["params"].mean()
                    sparse_time = subset["time"].mean()
                    sparse_flops = (
                        subset["flops"].mean() if "flops" in subset.columns else 0
                    )

                    # Efficiency metrics
                    acc_diff = (sparse_acc - dense_acc) * 100  # pp
                    param_ratio = (
                        sparse_params / dense_params if dense_params > 0 else 1
                    )
                    time_ratio = sparse_time / dense_time if dense_time > 0 else 1
                    flops_ratio = sparse_flops / dense_flops if dense_flops > 0 else 1

                    # Statistical significance
                    dense_accs = dense_df["accuracy"].values
                    sparse_accs = subset["accuracy"].values
                    if len(dense_accs) >= 2 and len(sparse_accs) >= 2:
                        p_val = permutation_test_p(
                            sparse_accs, dense_accs, n_permutations=500
                        )
                        d = cohens_d(sparse_accs, dense_accs)
                    else:
                        p_val = 1.0
                        d = 0.0

                    key = f"{algorithm}_tiles{num_tiles}"
                    task_analysis[f"{routing_mode}_{key}"] = {
                        "accuracy_diff_pp": float(acc_diff),
                        "param_ratio": float(param_ratio),
                        "time_ratio": float(time_ratio),
                        "flops_ratio": float(flops_ratio),
                        "p_value": float(p_val),
                        "cohens_d": float(d),
                        "significant": p_val < 0.05,
                        "sparse_accuracy": float(sparse_acc),
                        "dense_accuracy": float(dense_acc),
                    }

        analysis[task] = task_analysis

    return analysis


def _find_optimal_configs(results: list[dict]) -> dict:
    """Find optimal configurations per task."""
    import pandas as pd

    df = pd.DataFrame(results)
    if df.empty:
        return {}

    df = df[df["success"]].copy()
    optimal = {}

    for task in df["task"].unique():
        task_df = df[df["task"] == task]

        # Best by accuracy
        best_acc = task_df.loc[task_df["accuracy"].idxmax()]
        # Best by accuracy/param (efficiency)
        task_df["efficiency"] = task_df["accuracy"] / (task_df["params"] / 1e6 + 1e-6)
        best_eff = task_df.loc[task_df["efficiency"].idxmax()]
        # Best by accuracy/time
        task_df["speed_efficiency"] = task_df["accuracy"] / (task_df["time"] + 1e-6)
        best_speed = task_df.loc[task_df["speed_efficiency"].idxmax()]

        optimal[task] = {
            "best_accuracy": best_acc.to_dict(),
            "best_param_efficiency": best_eff.to_dict(),
            "best_speed_efficiency": best_speed.to_dict(),
        }

    return optimal


def _save_results(results: list[dict], output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with Path(output_path / "raw_results.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    import pandas as pd

    df = pd.DataFrame(results)
    df.to_parquet(output_path / "results.parquet", index=False)

    logger.info("Saved results to %s", output_path)


def _generate_report(
    routing_analysis: dict,
    optimal_configs: dict,
    output_dir: str,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with Path(output_path / "mot_ablation_report.md").open("w", encoding="utf-8") as f:
        f.write("# Mixture-of-Tiles (MoT) Ablation Report\n\n")

        f.write("## Routing Efficiency Analysis\n\n")
        for task, task_analysis in routing_analysis.items():
            f.write(f"### {task}\n\n")
            f.write(
                "| Routing | Algorithm | Tiles | Acc Δ (pp) | Param Ratio | Time Ratio | FLOPs Ratio | p-value | d |\n"
            )
            f.write(
                "|---------|-----------|-------|------------|-------------|------------|-------------|---------|---|\n"
            )
            for key, stats in sorted(
                task_analysis.items(),
                key=lambda x: -x[1].get("accuracy_diff_pp", 0),
            ):
                f.write(
                    f"| {key} | {stats.get('accuracy_diff_pp', 0):+.2f} | "
                    f"{stats.get('param_ratio', 1):.2f} | "
                    f"{stats.get('time_ratio', 1):.2f} | "
                    f"{stats.get('flops_ratio', 1):.2f} | "
                    f"{stats.get('p_value', 1):.4f} | "
                    f"{stats.get('cohens_d', 0):.2f} |\n"
                )
            f.write("\n")

        f.write("## Optimal Configurations\n\n")
        for task, configs in optimal_configs.items():
            f.write(f"### {task}\n\n")
            for name, config in configs.items():
                f.write(f"**{name}**: ")
                f.write(
                    f"routing={config.get('routing_mode', 'N/A')}, "
                    f"algo={config.get('tile_algorithm', 'N/A')}, "
                    f"tiles={config.get('num_tiles', 'N/A')}, "
                    f"topk={config.get('top_k', 'N/A')}, "
                    f"acc={config.get('accuracy', 0):.4f}, "
                    f"params={config.get('params', 0):.0f}, "
                    f"time={config.get('time', 0):.1f}s\n\n"
                )


def main():
    parser = argparse.ArgumentParser(description="Mixture-of-Tiles Ablation")
    parser.add_argument(
        "--tasks",
        default="mnist,cifar10,tiny_shakespeare",
        help="Comma-separated tasks",
    )
    parser.add_argument(
        "--routing", default="dense,sparse,topk,random", help="Routing modes"
    )
    parser.add_argument(
        "--algorithms", default="ep,fa,pc,hebbian", help="Tile algorithms"
    )
    parser.add_argument("--num-tiles", default="4,8,16,32", help="Number of tiles")
    parser.add_argument("--topk", default="1,2,4,8", help="Top-k values")
    parser.add_argument("--seeds", type=int, default=5, help="Seeds per config")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs per run")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--output-dir", default="results/mot_ablation", help="Output directory"
    )
    parser.add_argument("--device", default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument("--quick", action="store_true", help="Quick mode")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config = MoTAblationConfig(
        tasks=args.tasks.split(","),
        routing_modes=args.routing.split(","),
        tile_algorithms=args.algorithms.split(","),
        num_tiles=[int(n) for n in args.num_tiles.split(",")],
        topk_values=[int(k) for k in args.topk.split(",")],
        seeds=args.seeds,
        epochs=args.epochs if not args.quick else 3,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        device=args.device,
        quick_mode=args.quick,
    )

    logger.info("Starting MoT Ablation Experiment")

    # Run experiments
    results = run_mot_ablation(config)

    # Save results
    _save_results(results, config.output_dir)

    # Analyze routing efficiency
    routing_analysis = _analyze_routing_efficiency(results)
    with Path(Path(config.output_dir) / "routing_analysis.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(routing_analysis, f, indent=2, default=str)

    # Find optimal configs
    optimal_configs = _find_optimal_configs(results)
    with Path(Path(config.output_dir) / "optimal_configs.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(optimal_configs, f, indent=2, default=str)

    # Generate report
    _generate_report(routing_analysis, optimal_configs, config.output_dir)

    logger.info("MoT Ablation complete. Results in %s", config.output_dir)


if __name__ == "__main__":
    main()
