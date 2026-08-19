"""MEP Preset Tournament — Factorized Ablation of MEP Components.

Tests all combinations of gradient×update×constraint×feedback factors in MEP
to identify factor importance and recommend presets.

Usage:
    python -m bioplausible.experiments.mep_tournament --tasks mnist,cifar10 --seeds 5
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from bioplausible.core.trainer import CoreTrainer, TrainerConfig
from bioplausible.utils import seed_everything

logger = logging.getLogger(__name__)


# =============================================================================
# MEP Factor Definitions
# =============================================================================

# MEP factors and their levels
MEP_FACTORS = {
    "gradient": ["bp", "fa", "direct", "kfac"],  # Gradient estimator
    "update": ["sgd", "adam", "muon", "shampoo"],  # Update rule
    "constraint": ["none", "spectral", "frobenius", "lipschitz"],  # Constraint type
    "feedback": ["symmetric", "random", "alignment", "none"],  # Feedback pathway
}

# Base MEP config
BASE_MEP_CONFIG = {
    "hidden_dim": 512,
    "num_layers": 3,
    "use_spectral_norm": True,
    "learning_rate": 1e-3,
}


@dataclass(frozen=True, slots=True)
class MEPConfig:
    """Configuration for a single MEP factor combination."""

    gradient: Literal["bp", "fa", "direct", "kfac"]
    update: Literal["sgd", "adam", "muon", "shampoo"]
    constraint: Literal["none", "spectral", "frobenius", "lipschitz"]
    feedback: Literal["symmetric", "random", "alignment", "none"]
    task: str
    seed: int
    epochs: int
    batch_size: int

    def model_name(self) -> str:
        """Generate model name from factors."""
        return f"mep_{self.gradient}_{self.update}_{self.constraint}_{self.feedback}"

    def model_kwargs(self) -> dict:
        """Generate model kwargs for trainer."""
        kwargs = BASE_MEP_CONFIG.copy()
        kwargs.update({
            "gradient_estimator": self.gradient,
            "update_rule": self.update,
            "constraint_type": self.constraint,
            "feedback_type": self.feedback,
        })
        # Task-specific dims
        if self.task in ("mnist", "fashion_mnist"):
            kwargs["input_dim"] = 784
            kwargs["output_dim"] = 10
        else:
            kwargs["input_dim"] = 3072
            kwargs["output_dim"] = 10
        return kwargs


@dataclass(frozen=True, slots=True)
class MEPExperimentConfig:
    """Configuration for MEP tournament experiment."""

    tasks: list[str] = field(default_factory=lambda: ["mnist", "cifar10"])
    factors: dict[str, list[str]] = field(default_factory=lambda: MEP_FACTORS)
    seeds: int = 5
    epochs: int = 20
    batch_size: int = 128
    output_dir: str = "results/mep_tournament"
    device: str = "auto"
    quick_mode: bool = False


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _generate_all_combinations(factors: dict[str, list[str]]) -> list[dict]:
    """Generate all factor combinations."""
    keys = list(factors.keys())
    values = list(factors.values())
    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))
    return combinations


def _run_single_mep_experiment(config: MEPConfig, device: str) -> dict:
    """Run a single MEP experiment."""
    seed_everything(config.seed)

    trainer_config = TrainerConfig(
        model="mep",  # Assuming MEP model is registered as "mep"
        task=config.task,
        epochs=config.epochs,
        batch_size=config.batch_size,
        optimizer_kwargs={"lr": BASE_MEP_CONFIG["learning_rate"]},
        model_kwargs=config.model_kwargs(),
        device=device,
    )

    try:
        trainer = CoreTrainer(trainer_config)
        start_time = time.time()
        history = trainer.fit()
        elapsed = time.time() - start_time

        if not history:
            return {
                **config.__dict__,
                "accuracy": 0.0,
                "loss": float("inf"),
                "time": elapsed,
                "params": 0,
                "success": False,
            }

        final = history[-1]
        return {
            **config.__dict__,
            "accuracy": final.val_acc if hasattr(final, "val_acc") else final.accuracy,
            "loss": final.val_loss if hasattr(final, "val_loss") else final.loss,
            "time": elapsed,
            "params": final.param_count if hasattr(final, "param_count") else 0,
            "success": True,
        }
    except Exception as e:
        logger.exception("MEP experiment failed: %s", config.model_name())
        return {
            **config.__dict__,
            "accuracy": 0.0,
            "loss": float("inf"),
            "time": 0.0,
            "params": 0,
            "success": False,
            "error": str(e),
        }


def run_mep_tournament(config: MEPExperimentConfig) -> list[dict]:
    """Run full MEP factor tournament."""
    device = _resolve_device(config.device)
    combinations = _generate_all_combinations(config.factors)

    logger.info("MEP Tournament: %d factor combinations", len(combinations))
    logger.info("Factors: %s", {k: len(v) for k, v in config.factors.items()})
    logger.info("Tasks: %s", config.tasks)
    logger.info("Seeds per combination: %d", config.seeds)

    results = []
    total = len(config.tasks) * len(combinations) * config.seeds
    exp_count = 0

    for task in config.tasks:
        for combo in combinations:
            for seed in range(config.seeds):
                exp_count += 1
                mep_config = MEPConfig(
                    **combo,
                    task=task,
                    seed=seed,
                    epochs=config.epochs if not config.quick_mode else 3,
                    batch_size=config.batch_size,
                )

                logger.info(
                    "[%d/%d] %s on %s (seed=%d)",
                    exp_count,
                    total,
                    mep_config.model_name(),
                    task,
                    seed,
                )

                result = _run_single_mep_experiment(mep_config, device)
                results.append(result)

    return results


def _analyze_factor_importance(results: list[dict]) -> dict:
    """Analyze factor importance using ANOVA and Sobol indices."""
    import pandas as pd

    df = pd.DataFrame(results)
    if df.empty:
        return {}

    # Only successful runs
    df = df[df["success"]].copy()
    if df.empty:
        return {}

    factor_cols = ["gradient", "update", "constraint", "feedback"]
    importance = {}

    for task in df["task"].unique():
        task_df = df[df["task"] == task]
        task_importance = {}

        for factor in factor_cols:
            # One-way ANOVA effect size (eta-squared)
            groups = [g["accuracy"].values for _, g in task_df.groupby(factor)]
            if len(groups) >= 2 and all(len(g) > 0 for g in groups):
                from scipy.stats import f_oneway

                try:
                    f_stat, p_val = f_oneway(*groups)
                    # Eta-squared
                    ss_between = sum(
                        len(g) * (np.mean(g) - np.mean(task_df["accuracy"])) ** 2
                        for g in groups
                    )
                    ss_total = sum(
                        (task_df["accuracy"] - np.mean(task_df["accuracy"])) ** 2
                    )
                    eta_squared = ss_between / ss_total if ss_total > 0 else 0
                    task_importance[factor] = {
                        "eta_squared": float(eta_squared),
                        "f_statistic": float(f_stat),
                        "p_value": float(p_val),
                        "significant": p_val < 0.05,
                    }
                except Exception:
                    task_importance[factor] = {
                        "eta_squared": 0.0,
                        "p_value": 1.0,
                        "significant": False,
                    }

        # Two-way interactions
        for f1, f2 in itertools.combinations(factor_cols, 2):
            try:
                # Create interaction groups
                task_df[f"{f1}_{f2}"] = task_df[f1] + "_" + task_df[f2]
                groups = [
                    g["accuracy"].values for _, g in task_df.groupby(f"{f1}_{f2}")
                ]
                if len(groups) >= 2 and all(len(g) > 0 for g in groups):
                    f_stat, p_val = f_oneway(*groups)
                    ss_between = sum(
                        len(g) * (np.mean(g) - np.mean(task_df["accuracy"])) ** 2
                        for g in groups
                    )
                    ss_total = sum(
                        (task_df["accuracy"] - np.mean(task_df["accuracy"])) ** 2
                    )
                    eta_squared = ss_between / ss_total if ss_total > 0 else 0
                    task_importance[f"{f1}×{f2}"] = {
                        "eta_squared": float(eta_squared),
                        "f_statistic": float(f_stat),
                        "p_value": float(p_val),
                        "significant": p_val < 0.05,
                    }
            except Exception:
                pass

        importance[task] = task_importance

    return importance


def _find_best_presets(results: list[dict], top_k: int = 5) -> dict:
    """Find best presets per task."""
    import pandas as pd

    df = pd.DataFrame(results)
    if df.empty:
        return {}

    df = df[df["success"]].copy()
    presets = {}

    for task in df["task"].unique():
        task_df = df[df["task"] == task]
        # Aggregate by factor combination
        factor_cols = ["gradient", "update", "constraint", "feedback"]
        grouped = (
            task_df
            .groupby(factor_cols)
            .agg({
                "accuracy": ["mean", "std", "count"],
                "time": "mean",
                "params": "mean",
            })
            .reset_index()
        )
        grouped.columns = [
            "_".join(col).strip("_") if col[1] else col[0]
            for col in grouped.columns.values
        ]

        # Sort by mean accuracy
        grouped = grouped.sort_values("accuracy_mean", ascending=False)
        presets[task] = grouped.head(top_k).to_dict("records")

    return presets


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


def main():
    parser = argparse.ArgumentParser(description="MEP Preset Tournament")
    parser.add_argument(
        "--tasks", default="mnist,cifar10", help="Comma-separated tasks"
    )
    parser.add_argument("--seeds", type=int, default=5, help="Seeds per combination")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs per run")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--output-dir", default="results/mep_tournament", help="Output directory"
    )
    parser.add_argument("--device", default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument("--quick", action="store_true", help="Quick mode")
    parser.add_argument(
        "--factors",
        help="JSON string of factors to test (default: all)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.factors:
        factors = json.loads(args.factors)
    else:
        factors = MEP_FACTORS

    config = MEPExperimentConfig(
        tasks=args.tasks.split(","),
        factors=factors,
        seeds=args.seeds,
        epochs=args.epochs if not args.quick else 3,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        device=args.device,
        quick_mode=args.quick,
    )

    logger.info("Starting MEP Preset Tournament")

    # Run experiments
    results = run_mep_tournament(config)

    # Save results
    _save_results(results, config.output_dir)

    # Analyze factor importance
    importance = _analyze_factor_importance(results)
    with Path(Path(config.output_dir) / "factor_importance.json").open("w") as f:
        json.dump(importance, f, indent=2, default=str)

    # Find best presets
    presets = _find_best_presets(results)
    with Path(Path(config.output_dir) / "best_presets.json").open("w") as f:
        json.dump(presets, f, indent=2, default=str)

    # Generate report
    _generate_report(importance, presets, config.output_dir)

    logger.info("MEP Tournament complete. Results in %s", config.output_dir)


def _generate_report(importance: dict, presets: dict, output_dir: str) -> None:
    """Generate markdown report."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with Path(output_path / "tournament_report.md").open("w") as f:
        f.write("# MEP Preset Tournament Report\n\n")

        f.write("## Factor Importance\n\n")
        for task, task_imp in importance.items():
            f.write(f"### {task}\n\n")
            f.write("| Factor | η² | F-stat | p-value | Significant |\n")
            f.write("|--------|-----|--------|---------|-------------|\n")
            for factor, stats in sorted(
                task_imp.items(), key=lambda x: -x[1].get("eta_squared", 0)
            ):
                f.write(
                    f"| {factor} | {stats.get('eta_squared', 0):.4f} | "
                    f"{stats.get('f_statistic', 0):.2f} | "
                    f"{stats.get('p_value', 1):.4f} | "
                    f"{'✓' if stats.get('significant', False) else '✗'} |\n"
                )
            f.write("\n")

        f.write("## Best Presets\n\n")
        for task, task_presets in presets.items():
            f.write(f"### {task}\n\n")
            f.write(
                "| Rank | Gradient | Update | Constraint | Feedback | Accuracy | Std | Time (s) |\n"
            )
            f.write(
                "|------|----------|--------|------------|----------|----------|-----|----------|\n"
            )
            for i, p in enumerate(task_presets, 1):
                f.write(
                    f"| {i} | {p['gradient']} | {p['update']} | {p['constraint']} | "
                    f"{p['feedback']} | {p['accuracy_mean']:.4f} | {p['accuracy_std']:.4f} | "
                    f"{p['time_mean']:.1f} |\n"
                )
            f.write("\n")


if __name__ == "__main__":
    main()
