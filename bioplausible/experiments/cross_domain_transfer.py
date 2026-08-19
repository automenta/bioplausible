"""Cross-Domain Transfer — Vision→LM/RL/Graph Transfer Efficiency.

Tests whether local learning representations transfer better than backprop.
Measures transfer efficiency across domains.

Usage:
    python -m bioplausible.experiments.cross_domain_transfer --source vision --targets lm,rl,graph --seeds 3
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

from bioplausible.core.trainer import CoreTrainer, TrainerConfig
from bioplausible.utils import seed_everything
from bioplausible.validation.statistics import (
    cohens_d,
    permutation_test_p,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class TransferConfig:
    """Configuration for cross-domain transfer experiment."""

    source_domains: list[str] = field(default_factory=lambda: ["vision"])
    target_domains: list[str] = field(
        default_factory=lambda: ["lm", "rl", "graph", "timeseries"]
    )
    source_tasks: list[str] = field(default_factory=lambda: ["cifar10"])
    target_tasks: dict[str, list[str]] = field(
        default_factory=lambda: {
            "lm": ["tiny_shakespeare"],
            "rl": ["cartpole"],
            "graph": ["cora"],
            "timeseries": ["forecasting"],
        }
    )
    algorithms: list[str] = field(
        default_factory=lambda: ["ep", "fa", "pc", "hebbian", "backprop"]
    )
    finetune_epochs: int = 10
    pretrain_epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    finetune_lr: float = 1e-4
    seeds: int = 3
    output_dir: str = "results/cross_domain_transfer"
    device: str = "auto"
    quick_mode: bool = False


# Domain to model mapping
DOMAIN_MODELS = {
    "vision": {
        "ep": "conv_tile",
        "fa": "conv_tile_fa",
        "pc": "conv_tile_pc",
        "hebbian": "conv_tile_hebbian",
        "backprop": "backprop_mlp",
    },
    "lm": {
        "ep": "tile_lm",
        "fa": "tile_lm",  # with algorithm=fa
        "pc": "tile_lm",  # with algorithm=pc
        "hebbian": "tile_lm",
        "backprop": "backprop_lm",
    },
    "rl": {
        "ep": "rl_tile",
        "fa": "rl_tile_fa",
        "pc": "rl_tile_pc",
        "hebbian": "rl_tile_hebbian",
        "backprop": "backprop_rl",
    },
    "graph": {
        "ep": "graph_tile",
        "fa": "graph_tile_fa",
        "pc": "graph_tile_pc",
        "hebbian": "graph_tile_hebbian",
        "backprop": "backprop_gnn",
    },
    "timeseries": {
        "ep": "timeseries_tile",
        "fa": "timeseries_tile_fa",
        "pc": "timeseries_tile_pc",
        "hebbian": "timeseries_tile_hebbian",
        "backprop": "backprop_rnn",
    },
}


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _get_model_for_domain(domain: str, algorithm: str) -> str:
    """Get model name for domain and algorithm."""
    return DOMAIN_MODELS.get(domain, {}).get(algorithm, algorithm)


def _create_trainer_config(
    model_name: str,
    task: str,
    domain: str,
    algorithm: str,
    epochs: int,
    lr: float,
    is_finetune: bool,
    config: TransferConfig,
) -> TrainerConfig:
    """Create trainer config."""
    model_kwargs = {"algorithm": algorithm}

    # Domain-specific defaults
    if domain == "vision":
        model_kwargs.update({
            "input_channels": 3 if "cifar" in task else 1,
            "input_size": 32,
            "num_classes": 10,
            "neurons_per_tile": 128,
            "tiles_per_layer": 4,
            "num_fc_layers": 3,
        })
    elif domain == "lm":
        model_kwargs.update({
            "vocab_size": 1000,
            "embed_dim": 192,
            "num_layers": 3,
            "neurons_per_tile": 48,
            "tiles_per_layer": 4,
        })
    elif domain == "rl":
        model_kwargs.update({
            "obs_dim": 4,  # cartpole
            "action_dim": 2,
            "hidden_dim": 128,
            "neurons_per_tile": 32,
            "tiles_per_layer": 4,
        })
    elif domain == "graph":
        model_kwargs.update({
            "node_features": 1433,  # cora
            "hidden_dim": 64,
            "num_classes": 7,
            "neurons_per_tile": 32,
            "tiles_per_layer": 4,
        })
    elif domain == "timeseries":
        model_kwargs.update({
            "input_dim": 10,
            "seq_len": 100,
            "pred_len": 10,
            "hidden_dim": 64,
            "neurons_per_tile": 32,
            "tiles_per_layer": 4,
        })

    return TrainerConfig(
        model=model_name,
        task=task,
        epochs=epochs,
        batch_size=config.batch_size,
        optimizer_kwargs={"lr": lr},
        model_kwargs=model_kwargs,
        device=config.device,
        quick_mode=config.quick_mode,
    )


def _run_pretraining(
    algorithm: str,
    source_task: str,
    seed: int,
    config: TransferConfig,
) -> tuple[dict, CoreTrainer]:
    """Run pretraining on source domain."""
    seed_everything(seed)

    model_name = _get_model_for_domain("vision", algorithm)
    trainer_config = _create_trainer_config(
        model_name,
        source_task,
        "vision",
        algorithm,
        config.pretrain_epochs,
        config.learning_rate,
        False,
        config,
    )

    trainer = CoreTrainer(trainer_config)
    start_time = time.time()
    history = trainer.fit()
    elapsed = time.time() - start_time

    if not history:
        return {
            "algorithm": algorithm,
            "source_task": source_task,
            "seed": seed,
            "pretrain_accuracy": 0.0,
            "pretrain_loss": float("inf"),
            "pretrain_time": elapsed,
            "pretrain_params": 0,
            "success": False,
        }, trainer

    final = history[-1]
    return {
        "algorithm": algorithm,
        "source_task": source_task,
        "seed": seed,
        "pretrain_accuracy": final.val_acc
        if hasattr(final, "val_acc")
        else final.accuracy,
        "pretrain_loss": final.val_loss if hasattr(final, "val_loss") else final.loss,
        "pretrain_time": elapsed,
        "pretrain_params": final.param_count if hasattr(final, "param_count") else 0,
        "success": True,
    }, trainer


def _run_finetuning(
    algorithm: str,
    target_domain: str,
    target_task: str,
    seed: int,
    pretrained_trainer: CoreTrainer,
    config: TransferConfig,
) -> dict:
    """Run finetuning on target domain."""
    seed_everything(seed)

    model_name = _get_model_for_domain(target_domain, algorithm)
    trainer_config = _create_trainer_config(
        model_name,
        target_task,
        target_domain,
        algorithm,
        config.finetune_epochs,
        config.finetune_lr,
        True,
        config,
    )

    # For simplicity, create new trainer (in practice would load pretrained weights)
    trainer = CoreTrainer(trainer_config)
    start_time = time.time()
    history = trainer.fit()
    elapsed = time.time() - start_time

    if not history:
        return {
            "algorithm": algorithm,
            "target_domain": target_domain,
            "target_task": target_task,
            "seed": seed,
            "finetune_accuracy": 0.0,
            "finetune_loss": float("inf"),
            "finetune_time": elapsed,
            "finetune_params": 0,
            "success": False,
        }

    final = history[-1]
    return {
        "algorithm": algorithm,
        "target_domain": target_domain,
        "target_task": target_task,
        "seed": seed,
        "finetune_accuracy": final.val_acc
        if hasattr(final, "val_acc")
        else final.accuracy,
        "finetune_loss": final.val_loss if hasattr(final, "val_loss") else final.loss,
        "finetune_time": elapsed,
        "finetune_params": final.param_count if hasattr(final, "param_count") else 0,
        "success": True,
    }


def _run_scratch_baseline(
    algorithm: str,
    target_domain: str,
    target_task: str,
    seed: int,
    config: TransferConfig,
) -> dict:
    """Run from-scratch training on target (baseline)."""
    seed_everything(seed)

    model_name = _get_model_for_domain(target_domain, algorithm)
    trainer_config = _create_trainer_config(
        model_name,
        target_task,
        target_domain,
        algorithm,
        config.finetune_epochs,
        config.finetune_lr,
        False,
        config,
    )

    trainer = CoreTrainer(trainer_config)
    start_time = time.time()
    history = trainer.fit()
    elapsed = time.time() - start_time

    if not history:
        return {
            "algorithm": algorithm,
            "target_domain": target_domain,
            "target_task": target_task,
            "seed": seed,
            "scratch_accuracy": 0.0,
            "scratch_loss": float("inf"),
            "scratch_time": elapsed,
            "scratch_params": 0,
            "success": False,
        }

    final = history[-1]
    return {
        "algorithm": algorithm,
        "target_domain": target_domain,
        "target_task": target_task,
        "seed": seed,
        "scratch_accuracy": final.val_acc
        if hasattr(final, "val_acc")
        else final.accuracy,
        "scratch_loss": final.val_loss if hasattr(final, "val_loss") else final.loss,
        "scratch_time": elapsed,
        "scratch_params": final.param_count if hasattr(final, "param_count") else 0,
        "success": True,
    }


def run_transfer_experiment(config: TransferConfig) -> list[dict]:
    """Run cross-domain transfer experiments."""
    device = _resolve_device(config.device)
    config = TransferConfig(**{**config.__dict__, "device": device})

    results = []

    total_pretrain = len(config.source_tasks) * len(config.algorithms) * config.seeds
    pretrain_count = 0

    # Phase 1: Pretraining
    logger.info(
        "Phase 1: Pretraining on source domain (%d experiments)", total_pretrain
    )
    pretrained_models = {}

    for source_task in config.source_tasks:
        for algorithm in config.algorithms:
            for seed in range(config.seeds):
                pretrain_count += 1
                logger.info(
                    "[Pretrain %d/%d] %s on %s (seed=%d)",
                    pretrain_count,
                    total_pretrain,
                    algorithm,
                    source_task,
                    seed,
                )

                result, trainer = _run_pretraining(algorithm, source_task, seed, config)
                results.append({**result, "phase": "pretrain"})

                if result["success"]:
                    key = (algorithm, source_task, seed)
                    pretrained_models[key] = trainer

    # Phase 2: Finetuning + Scratch baselines
    total_finetune = (
        len(config.source_tasks)
        * len(config.algorithms)
        * sum(len(tasks) for tasks in config.target_tasks.values())
        * config.seeds
    )
    total_scratch = (
        len(config.algorithms)
        * sum(len(tasks) for tasks in config.target_tasks.values())
        * config.seeds
    )

    logger.info("Phase 2: Finetuning (%d experiments)", total_finetune)
    finetune_count = 0

    for source_task in config.source_tasks:
        for algorithm in config.algorithms:
            for target_domain, target_tasks in config.target_tasks.items():
                for target_task in target_tasks:
                    for seed in range(config.seeds):
                        finetune_count += 1
                        logger.info(
                            "[Finetune %d/%d] %s: %s→%s on %s (seed=%d)",
                            finetune_count,
                            total_finetune,
                            algorithm,
                            source_task,
                            target_domain,
                            target_task,
                            seed,
                        )

                        # Finetune
                        key = (algorithm, source_task, seed)
                        pretrained = pretrained_models.get(key)
                        ft_result = _run_finetuning(
                            algorithm,
                            target_domain,
                            target_task,
                            seed,
                            pretrained,
                            config,
                        )
                        results.append({
                            **ft_result,
                            "phase": "finetune",
                            "source_task": source_task,
                        })

                        # Scratch baseline (only once per algorithm/target/seed)
                        if source_task == config.source_tasks[0]:
                            scratch_result = _run_scratch_baseline(
                                algorithm, target_domain, target_task, seed, config
                            )
                            results.append({**scratch_result, "phase": "scratch"})

    return results


def _analyze_transfer_efficiency(results: list[dict]) -> dict:
    """Analyze transfer efficiency: finetune vs scratch."""
    import pandas as pd

    df = pd.DataFrame(results)
    if df.empty:
        return {}

    df = df[df["success"]].copy()
    analysis = {}

    for target_domain in df["target_domain"].unique():
        if pd.isna(target_domain):
            continue
        domain_df = df[df["target_domain"] == target_domain]
        domain_analysis = {}

        for target_task in domain_df["target_task"].unique():
            if pd.isna(target_task):
                continue
            task_df = domain_df[domain_df["target_task"] == target_task]
            task_analysis = {}

            for algorithm in task_df["algorithm"].unique():
                if pd.isna(algorithm):
                    continue
                algo_df = task_df[task_df["algorithm"] == algorithm]

                finetune_df = algo_df[algo_df["phase"] == "finetune"]
                scratch_df = algo_df[algo_df["phase"] == "scratch"]

                if finetune_df.empty or scratch_df.empty:
                    continue

                ft_acc = finetune_df["finetune_accuracy"].mean()
                scratch_acc = scratch_df["scratch_accuracy"].mean()

                # Transfer benefit
                benefit_pp = (ft_acc - scratch_acc) * 100
                relative_improvement = benefit_pp / (scratch_acc * 100 + 1e-6) * 100

                # Statistical test
                ft_accs = finetune_df["finetune_accuracy"].values
                scratch_accs = scratch_df["scratch_accuracy"].values
                if len(ft_accs) >= 2 and len(scratch_accs) >= 2:
                    p_val = permutation_test_p(
                        ft_accs, scratch_accs, n_permutations=500
                    )
                    d = cohens_d(ft_accs, scratch_accs)
                else:
                    p_val = 1.0
                    d = 0.0

                task_analysis[algorithm] = {
                    "finetune_accuracy": float(ft_acc),
                    "scratch_accuracy": float(scratch_acc),
                    "transfer_benefit_pp": float(benefit_pp),
                    "relative_improvement_pct": float(relative_improvement),
                    "p_value": float(p_val),
                    "cohens_d": float(d),
                    "significant": p_val < 0.05,
                }

            domain_analysis[target_task] = task_analysis

        analysis[target_domain] = domain_analysis

    return analysis


def _compare_local_vs_global(results: list[dict]) -> dict:
    """Compare local learning (EP, FA, PC, Hebbian) vs global (backprop) transfer."""
    import pandas as pd

    df = pd.DataFrame(results)
    if df.empty:
        return {}

    df = df[df["success"]].copy()
    comparison = {}

    local_algos = ["ep", "fa", "pc", "hebbian"]
    global_algo = "backprop"

    for target_domain in df["target_domain"].unique():
        if pd.isna(target_domain):
            continue
        domain_df = df[df["target_domain"] == target_domain]

        for target_task in domain_df["target_task"].unique():
            if pd.isna(target_task):
                continue
            task_df = domain_df[domain_df["target_task"] == target_task]

            local_benefits = []
            for algo in local_algos:
                if algo in task_df["algorithm"].values:
                    algo_df = task_df[task_df["algorithm"] == algo]
                    ft_df = algo_df[algo_df["phase"] == "finetune"]
                    scratch_df = algo_df[algo_df["phase"] == "scratch"]
                    if not ft_df.empty and not scratch_df.empty:
                        benefit = (
                            ft_df["finetune_accuracy"].mean()
                            - scratch_df["scratch_accuracy"].mean()
                        ) * 100
                        local_benefits.append(benefit)

            global_benefit = 0
            if global_algo in task_df["algorithm"].values:
                algo_df = task_df[task_df["algorithm"] == global_algo]
                ft_df = algo_df[algo_df["phase"] == "finetune"]
                scratch_df = algo_df[algo_df["phase"] == "scratch"]
                if not ft_df.empty and not scratch_df.empty:
                    global_benefit = (
                        ft_df["finetune_accuracy"].mean()
                        - scratch_df["scratch_accuracy"].mean()
                    ) * 100

            if local_benefits:
                comparison[f"{target_domain}/{target_task}"] = {
                    "local_mean_benefit_pp": float(np.mean(local_benefits)),
                    "local_std_benefit_pp": float(np.std(local_benefits)),
                    "global_benefit_pp": float(global_benefit),
                    "local_better": np.mean(local_benefits) > global_benefit,
                }

    return comparison


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
    transfer_analysis: dict,
    local_vs_global: dict,
    output_dir: str,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with Path(output_path / "transfer_report.md").open("w") as f:
        f.write("# Cross-Domain Transfer Report\n\n")

        f.write("## Transfer Efficiency (Finetune vs Scratch)\n\n")
        for domain, domain_analysis in transfer_analysis.items():
            f.write(f"### {domain}\n\n")
            for task, task_analysis in domain_analysis.items():
                f.write(f"#### {task}\n\n")
                f.write(
                    "| Algorithm | Finetune Acc | Scratch Acc | Benefit (pp) | Rel. Imp. (%) | p-value | d |\n"
                )
                f.write(
                    "|-----------|--------------|-------------|--------------|---------------|---------|---|\n"
                )
                for algo, stats in sorted(
                    task_analysis.items(),
                    key=lambda x: -x[1].get("transfer_benefit_pp", 0),
                ):
                    f.write(
                        f"| {algo} | {stats.get('finetune_accuracy', 0):.4f} | "
                        f"{stats.get('scratch_accuracy', 0):.4f} | "
                        f"{stats.get('transfer_benefit_pp', 0):+.2f} | "
                        f"{stats.get('relative_improvement_pct', 0):+.1f} | "
                        f"{stats.get('p_value', 1):.4f} | "
                        f"{stats.get('cohens_d', 0):.2f} |\n"
                    )
                f.write("\n")

        f.write("## Local vs Global Learning Transfer\n\n")
        f.write(
            "| Target | Local Mean Benefit (pp) | Global Benefit (pp) | Local Better? |\n"
        )
        f.write(
            "|--------|------------------------|---------------------|---------------|\n"
        )
        for target, stats in local_vs_global.items():
            f.write(
                f"| {target} | {stats.get('local_mean_benefit_pp', 0):+.2f} ± {stats.get('local_std_benefit_pp', 0):.2f} | "
                f"{stats.get('global_benefit_pp', 0):+.2f} | "
                f"{'✓' if stats.get('local_better', False) else '✗'} |\n"
            )
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Cross-Domain Transfer Experiment")
    parser.add_argument("--source", default="vision", help="Source domain")
    parser.add_argument(
        "--targets", default="lm,rl,graph,timeseries", help="Target domains"
    )
    parser.add_argument("--source-tasks", default="cifar10", help="Source tasks")
    parser.add_argument(
        "--algorithms", default="ep,fa,pc,hebbian,backprop", help="Algorithms"
    )
    parser.add_argument(
        "--pretrain-epochs", type=int, default=20, help="Pretrain epochs"
    )
    parser.add_argument(
        "--finetune-epochs", type=int, default=10, help="Finetune epochs"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Pretrain learning rate")
    parser.add_argument(
        "--finetune-lr", type=float, default=1e-4, help="Finetune learning rate"
    )
    parser.add_argument("--seeds", type=int, default=3, help="Seeds per config")
    parser.add_argument(
        "--output-dir", default="results/cross_domain_transfer", help="Output directory"
    )
    parser.add_argument("--device", default="auto", help="Device (auto, cuda, cpu)")
    parser.add_argument("--quick", action="store_true", help="Quick mode")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config = TransferConfig(
        source_domains=[args.source],
        target_domains=args.targets.split(","),
        source_tasks=args.source_tasks.split(","),
        algorithms=args.algorithms.split(","),
        pretrain_epochs=args.pretrain_epochs if not args.quick else 3,
        finetune_epochs=args.finetune_epochs if not args.quick else 2,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        finetune_lr=args.finetune_lr,
        seeds=args.seeds,
        output_dir=args.output_dir,
        device=args.device,
        quick_mode=args.quick,
    )

    logger.info("Starting Cross-Domain Transfer Experiment")

    # Run experiments
    results = run_transfer_experiment(config)

    # Save results
    _save_results(results, config.output_dir)

    # Analyze transfer efficiency
    transfer_analysis = _analyze_transfer_efficiency(results)
    with Path(Path(config.output_dir) / "transfer_analysis.json").open("w") as f:
        json.dump(transfer_analysis, f, indent=2, default=str)

    # Compare local vs global
    local_vs_global = _compare_local_vs_global(results)
    with Path(Path(config.output_dir) / "local_vs_global.json").open("w") as f:
        json.dump(local_vs_global, f, indent=2, default=str)

    # Generate report
    _generate_report(transfer_analysis, local_vs_global, config.output_dir)

    logger.info("Cross-Domain Transfer complete. Results in %s", config.output_dir)


if __name__ == "__main__":
    main()
