"""Feedback Alignment Depth Scaling — 10→1000 Layers.

Tests FA viability at extreme depths on MNIST and synthetic parity tasks.
Produces depth-scaling curves proving FA viability.

Usage:
    python -m computronium.experiments.fa_depth_scaling --depths 10,20,50,100,200,500,1000 --seeds 3
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

from computronium.analysis.scaling import fit_power_law
from computronium.core.registry import ComponentCategory, Registry
from computronium.core.trainer import CoreTrainer, TrainerConfig
from computronium.utils import seed_everything
from computronium.validation.statistics import bootstrap_ci

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class FADepthConfig:
    """Configuration for FA depth scaling experiment."""

    tasks: list[str] = field(default_factory=lambda: ["mnist", "synthetic"])
    depths: list[int] = field(default_factory=lambda: [10, 20, 50, 100, 200, 500, 1000])
    widths: list[int] = field(default_factory=lambda: [128, 256, 512])
    algorithms: list[str] = field(
        default_factory=lambda: ["fa", "backprop", "ep", "pc"]
    )
    seeds: int = 3
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 1e-3
    output_dir: str = "results/fa_depth_scaling"
    device: str = "auto"
    quick_mode: bool = False
    synthetic_samples: int = 5000
    synthetic_dim: int = 128
    synthetic_classes: int = 10


# Model configurations per algorithm
ALGO_CONFIGS = {
    "fa": {
        "model": "fa_mlp",
        "base_kwargs": {
            "use_spectral_norm": True,
            "feedback_type": "random",
            "learning_rate": 1e-3,
        },
    },
    "backprop": {
        "model": "backprop_mlp",
        "base_kwargs": {
            "use_spectral_norm": True,
            "learning_rate": 1e-3,
        },
    },
    "ep": {
        "model": "eqprop",
        "base_kwargs": {
            "use_spectral_norm": True,
            "beta": 0.1,
            "step_size": 0.1,
            "inference_steps": 20,
            "learning_rate": 1e-3,
        },
    },
    "pc": {
        "model": "predictive_coding",
        "base_kwargs": {
            "use_spectral_norm": True,
            "learning_rate": 1e-3,
        },
    },
}


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _get_task_dims(task: str, config: FADepthConfig) -> tuple[int, int]:
    """Get input/output dimensions for task."""
    if task == "mnist":
        return 784, 10
    elif task == "synthetic":
        return config.synthetic_dim, config.synthetic_classes
    return 784, 10


def _create_trainer_config(
    algorithm: str,
    task: str,
    depth: int,
    width: int,
    seed: int,
    config: FADepthConfig,
) -> TrainerConfig:
    """Create trainer config for algorithm at specific depth/width."""
    algo_cfg = ALGO_CONFIGS[algorithm]
    input_dim, output_dim = _get_task_dims(task, config)

    model_kwargs = algo_cfg["base_kwargs"].copy()
    model_kwargs.update({
        "input_dim": input_dim,
        "output_dim": output_dim,
        "hidden_dim": width,
        "num_layers": depth,
    })

    return TrainerConfig(
        model=algo_cfg["model"],
        task=task,
        epochs=config.epochs if not config.quick_mode else 3,
        batch_size=config.batch_size,
        optimizer_kwargs={"lr": config.learning_rate},
        model_kwargs=model_kwargs,
        device=config.device,
        quick_mode=config.quick_mode,
    )


def _run_single_experiment(
    algorithm: str,
    task: str,
    depth: int,
    width: int,
    seed: int,
    config: FADepthConfig,
) -> dict:
    """Run a single depth scaling experiment."""
    seed_everything(seed)

    trainer_config = _create_trainer_config(algorithm, task, depth, width, seed, config)

    # Verify model is registered
    try:
        Registry.get_metadata(ComponentCategory.MODEL, trainer_config.model)
    except KeyError:
        logger.warning("Model %s not registered, skipping", trainer_config.model)
        return {
            "algorithm": algorithm,
            "model": trainer_config.model,
            "task": task,
            "depth": depth,
            "width": width,
            "seed": seed,
            "accuracy": 0.0,
            "loss": float("inf"),
            "time": 0.0,
            "params": 0,
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
            "depth": depth,
            "width": width,
            "seed": seed,
            "accuracy": 0.0,
            "loss": float("inf"),
            "time": elapsed,
            "params": 0,
            "success": False,
        }

    final = history[-1]
    return {
        "algorithm": algorithm,
        "model": trainer_config.model,
        "task": task,
        "depth": depth,
        "width": width,
        "seed": seed,
        "accuracy": final.val_acc if hasattr(final, "val_acc") else final.accuracy,
        "loss": final.val_loss if hasattr(final, "val_loss") else final.loss,
        "time": elapsed,
        "params": final.param_count if hasattr(final, "param_count") else 0,
        "success": True,
    }


def run_fa_depth_scaling(config: FADepthConfig) -> list[dict]:
    """Run FA depth scaling experiments."""
    device = _resolve_device(config.device)
    config = FADepthConfig(**{**config.__dict__, "device": device})

    results = []
    total = (
        len(config.tasks)
        * len(config.algorithms)
        * len(config.depths)
        * len(config.widths)
        * config.seeds
    )
    logger.info("FA Depth Scaling: %d total experiments", total)
    logger.info("Depths: %s", config.depths)
    logger.info("Widths: %s", config.widths)
    logger.info("Algorithms: %s", config.algorithms)

    exp_count = 0
    for task in config.tasks:
        for algorithm in config.algorithms:
            for depth in config.depths:
                for width in config.widths:
                    for seed in range(config.seeds):
                        exp_count += 1
                        logger.info(
                            "[%d/%d] %s depth=%d width=%d on %s (seed=%d)",
                            exp_count,
                            total,
                            algorithm,
                            depth,
                            width,
                            task,
                            seed,
                        )

                        result = _run_single_experiment(
                            algorithm, task, depth, width, seed, config
                        )
                        results.append(result)

    return results


def _analyze_depth_scaling(results: list[dict]) -> dict:
    """Analyze depth scaling laws for each algorithm."""
    import pandas as pd

    df = pd.DataFrame(results)
    if df.empty:
        return {}

    df = df[df["success"]].copy()
    analysis = {}

    for task in df["task"].unique():
        task_df = df[df["task"] == task]
        task_analysis = {}

        for algorithm in task_df["algorithm"].unique():
            algo_df = task_df[task_df["algorithm"] == algorithm]
            algo_analysis = {}

            for width in algo_df["width"].unique():
                width_df = algo_df[algo_df["width"] == width]
                # Aggregate over seeds
                depth_agg = (
                    width_df
                    .groupby("depth")
                    .agg({
                        "accuracy": ["mean", "std"],
                        "loss": ["mean", "std"],
                        "params": "mean",
                        "time": "mean",
                    })
                    .reset_index()
                )
                depth_agg.columns = [
                    "_".join(col).strip("_") if col[1] else col[0]
                    for col in depth_agg.columns.values
                ]

                # Fit scaling law: accuracy vs depth
                depths = depth_agg["depth"].values
                accs = depth_agg["accuracy_mean"].values

                if len(depths) >= 3:
                    try:
                        # Fit power law: acc = a * depth^b + c
                        fit = fit_power_law(depths, accs, n_bootstrap=100)
                        algo_analysis[f"width_{width}"] = {
                            "scaling_law": {
                                "a": fit.a,
                                "b": fit.b,
                                "c": fit.c,
                                "r_squared": fit.r_squared,
                                "b_ci": fit.b_ci,
                            },
                            "depth_curve": depth_agg.to_dict("records"),
                        }
                    except Exception as e:
                        logger.warning(
                            "Scaling fit failed for %s/%s width=%d: %s",
                            task,
                            algorithm,
                            width,
                            e,
                        )

            task_analysis[algorithm] = algo_analysis

        analysis[task] = task_analysis

    return analysis


def _compute_parity_gaps(results: list[dict]) -> dict:
    """Compute FA vs Backprop parity gaps at each depth."""
    import pandas as pd

    df = pd.DataFrame(results)
    if df.empty:
        return {}

    df = df[df["success"]].copy()
    gaps = {}

    for task in df["task"].unique():
        task_df = df[df["task"] == task]
        task_gaps = {}

        for width in task_df["width"].unique():
            width_df = task_df[task_df["width"] == width]
            width_gaps = {}

            for depth in width_df["depth"].unique():
                depth_df = width_df[width_df["depth"] == depth]

                fa_accs = depth_df[depth_df["algorithm"] == "fa"]["accuracy"].values
                bp_accs = depth_df[depth_df["algorithm"] == "backprop"][
                    "accuracy"
                ].values

                if len(fa_accs) > 0 and len(bp_accs) > 0:
                    fa_mean = np.mean(fa_accs)
                    bp_mean = np.mean(bp_accs)
                    gap = (bp_mean - fa_mean) * 100  # percentage points

                    # Statistical test
                    from computronium.validation.statistics import permutation_test_p

                    p_val = permutation_test_p(fa_accs, bp_accs, n_permutations=500)

                    width_gaps[depth] = {
                        "fa_accuracy": float(fa_mean),
                        "backprop_accuracy": float(bp_mean),
                        "gap_pp": float(gap),
                        "p_value": float(p_val),
                        "fa_ci": bootstrap_ci(fa_accs) if len(fa_accs) >= 2 else (0, 0),
                        "bp_ci": bootstrap_ci(bp_accs) if len(bp_accs) >= 2 else (0, 0),
                    }

            task_gaps[f"width_{width}"] = width_gaps

        gaps[task] = task_gaps

    return gaps


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


def _generate_plots(results: list[dict], output_dir: str) -> None:
    """Generate depth scaling plots."""
    import pandas as pd

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return

    df = pd.DataFrame(results)
    if df.empty:
        return

    df = df[df["success"]].copy()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for task in df["task"].unique():
        task_df = df[df["task"] == task]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, width in enumerate(sorted(task_df["width"].unique())):
            if idx >= 4:
                break
            ax = axes[idx]
            width_df = task_df[task_df["width"] == width]

            for algorithm in sorted(width_df["algorithm"].unique()):
                algo_df = width_df[width_df["algorithm"] == algorithm]
                depth_agg = (
                    algo_df
                    .groupby("depth")["accuracy"]
                    .agg(["mean", "std"])
                    .reset_index()
                )

                if len(depth_agg) > 0:
                    ax.errorbar(
                        depth_agg["depth"],
                        depth_agg["mean"],
                        yerr=depth_agg["std"],
                        label=algorithm.upper(),
                        marker="o",
                        capsize=3,
                    )

            ax.set_xscale("log")
            ax.set_xlabel("Depth (layers)")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"{task.upper()} - Width {width}")
            ax.legend()
            ax.grid(True, which="both", ls="--", alpha=0.5)

        fig.suptitle(f"Depth Scaling: {task.upper()}", fontsize=14)
        fig.tight_layout()
        fig.savefig(
            output_path / f"depth_scaling_{task}.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

    # Parity gap plot
    for task in df["task"].unique():
        task_df = df[df["task"] == task]

        fig, axes = plt.subplots(1, len(task_df["width"].unique()), figsize=(14, 4))
        if len(task_df["width"].unique()) == 1:
            axes = [axes]

        for idx, width in enumerate(sorted(task_df["width"].unique())):
            ax = axes[idx]
            width_df = task_df[task_df["width"] == width]

            for depth in sorted(width_df["depth"].unique()):
                depth_df = width_df[width_df["depth"] == depth]

                fa_accs = depth_df[depth_df["algorithm"] == "fa"]["accuracy"].values
                bp_accs = depth_df[depth_df["algorithm"] == "backprop"][
                    "accuracy"
                ].values

                if len(fa_accs) > 0 and len(bp_accs) > 0:
                    gap = (np.mean(bp_accs) - np.mean(fa_accs)) * 100
                    ax.scatter(
                        [depth],
                        [gap],
                        s=100,
                        label=f"Depth {depth}" if idx == 0 else "",
                    )

            ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
            ax.set_xscale("log")
            ax.set_xlabel("Depth")
            ax.set_ylabel("Gap (BP - FA) %")
            ax.set_title(f"Width {width}")
            ax.grid(True, which="both", ls="--", alpha=0.5)

        fig.suptitle(f"FA vs Backprop Parity Gap: {task.upper()}", fontsize=14)
        fig.tight_layout()
        fig.savefig(
            output_path / f"parity_gap_{task}.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

    logger.info("Saved plots to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Feedback Alignment Depth Scaling")
    parser.add_argument(
        "--tasks", default="mnist,synthetic", help="Comma-separated tasks"
    )
    parser.add_argument(
        "--depths", default="10,20,50,100,200,500,1000", help="Comma-separated depths"
    )
    parser.add_argument(
        "--widths", default="128,256,512", help="Comma-separated widths"
    )
    parser.add_argument(
        "--algorithms", default="fa,backprop,ep,pc", help="Comma-separated algorithms"
    )
    parser.add_argument("--seeds", type=int, default=3, help="Seeds per config")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs per run")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--output-dir", default="results/fa_depth_scaling", help="Output directory"
    )
    parser.add_argument("--device", default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument("--quick", action="store_true", help="Quick mode")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config = FADepthConfig(
        tasks=args.tasks.split(","),
        depths=[int(d) for d in args.depths.split(",")],
        widths=[int(w) for w in args.widths.split(",")],
        algorithms=args.algorithms.split(","),
        seeds=args.seeds,
        epochs=args.epochs if not args.quick else 5,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        device=args.device,
        quick_mode=args.quick,
    )

    logger.info("Starting FA Depth Scaling Experiment")

    # Run experiments
    results = run_fa_depth_scaling(config)

    # Save results
    _save_results(results, config.output_dir)

    # Analyze depth scaling
    scaling_analysis = _analyze_depth_scaling(results)
    with Path(Path(config.output_dir) / "scaling_analysis.json").open("w") as f:
        json.dump(scaling_analysis, f, indent=2, default=str)

    # Compute parity gaps
    parity_gaps = _compute_parity_gaps(results)
    with Path(Path(config.output_dir) / "parity_gaps.json").open("w") as f:
        json.dump(parity_gaps, f, indent=2, default=str)

    # Generate plots
    _generate_plots(results, config.output_dir)

    logger.info("FA Depth Scaling complete. Results in %s", config.output_dir)


if __name__ == "__main__":
    main()
