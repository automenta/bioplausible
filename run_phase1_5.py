#!/usr/bin/env python
"""
Phase 1.5 HPO Experiment Runner — Architecture-Aware, Multi-Objective.

Key improvements over Phase 1:
- Fair comparisons: groups models by architecture (Conv spatial vs MLP flat)
- Multi-objective: maximize accuracy, minimize params, minimize epoch time
- Excludes underperformers from search space (archived to baselines/)
- Collects efficiency metrics: params, FLOPs, epoch_time, peak_memory
- Generates Pareto frontiers per architecture group
- EqProp models run the O(1) implicit-differentiation method
  (``gradient_method="equilibrium"``) by default — no BPTT unrolling, so
  CIFAR-10 trials fit in memory regardless of settle steps (FIX.md §44).

Usage:
    uv run python run_phase1_5.py          # Local run
    uv run python run_phase1_5.py --tier deep  # Full budget

Outputs:
    compute_phase1_5.db                 # SQLite with all Optuna studies
    results/phase1_5/portfolio.csv      # Ranked portfolio with Pareto status
    results/phase1_5/portfolio_conv.csv # Conv architecture group
    results/phase1_5/portfolio_mlp.csv  # MLP architecture group
    logs/experiment.log                 # Full run log
"""

import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration — EDIT HERE to change budgets, models, etc.
# ---------------------------------------------------------------------------

# Architecture groups for FAIR comparisons
ARCH_GROUPS = {
    "conv": {
        "description": "Convolutional spatial models (4D input, preserve spatial structure)",
        "models": [
            ("eqprop", "modern_conv_eqprop"),
            ("eqprop", "conv_eqprop"),
            ("eqprop", "graph_eqprop"),      # Graph conv
        ],
        "task": "cifar10",   # Primary task for conv models
        "budget_per_model": 30,
    },
    "mlp": {
        "description": "Flat MLP models (1D input, flattened)",
        "models": [
            ("backprop", "backprop_mlp"),
            ("eqprop", "eqprop_mlp"),
            ("eqprop", "neural_cube"),
            ("forward_only", "pepita"),
            ("hebbian", "deep_hebbian"),
            ("target_prop", "diff_target_prop"),
        ],
        "task": "cifar10",   # Primary task
        "budget_per_model": 50,
    },
}

# Models EXCLUDED from search space (archived as baselines)
# Source code preserved in bioplausible/zoo/models/ — only removed from HPO
EXCLUDED_MODELS = {
    # EqProp contrastive variants (nudge signal dies in deep layers)
    "eqprop",
    "directed_ep",
    "finite_nudge_ep",
    "momentum_equilibrium",
    "sparse_equilibrium",
    "equilibrium_alignment",
    "layerwise_equilibrium_fa",
    # FA hybrids (credit assignment fails end-to-end)
    "contrastive_feedback_alignment",
    "energy_guided_fa",
    "energy_minimizing_fa",
    # Hebbian (update rule plateaus)
    "hebbian_3d",
    "three_factor_hebbian",
    # Predictive Coding (requires per-graph propagator)
    "fabricpc_graph_pcn",
    # Spiking (requires surrogate-gradient BPTT)
    "spiking_stdp",
    # Diffusion (needs timestep in forward)
    "eqprop_diffusion",
    # Phase 1.5: Removed for poor CIFAR-10 performance / excessive compute
    "lazy_eqprop",  # Too slow (8-24s/epoch), implicit diff overhead
    "holomorphic_ep",  # Low accuracy (0.42), slow (6-9s/epoch)
    "forward_forward",  # Low accuracy (0.39), high params
    "hebbian_chain",  # Low accuracy (0.44), high params
}

CONFIG = {
    "db": "compute_phase1_5_v3.db",
    "tier": "overnight",            # "smoke" | "shallow" | "standard" | "deep" | "overnight"
    "interval": 60,               # Dashboard refresh seconds
    "emit_every": 20,             # Interim portfolio every N trials
    "seed": 42,
    "device": "auto",
    "tasks": ["cifar10"],         # Tasks to run (overnight: cifar10 only; full: cifar10,digits)
    # Multi-objective directions for Optuna
    "objectives": ["accuracy", "param_count", "epoch_time_s"],
    "directions": ["maximize", "minimize", "minimize"],
    # Pruning: remove trials that are clearly worse on all objectives
    "prune_worse_than_pareto": True,
    # Hard constraint: reject trials exceeding this param count (pressures minimization)
    # Relaxed vision floor (hidden_dim 32..256) lets equilibrium models sample
    # small configs; keep the cap tight so they must compete near backprop's size.
    "max_params": 200_000,
    # Use NSGA-II for true multi-objective Pareto optimization
    "sampler": "nsga2",
}

TIER_BUDGETS = {
    "smoke": {"conv": 5, "mlp": 3},
    "shallow": {"conv": 15, "mlp": 10},
    "standard": {"conv": 30, "mlp": 50},
    "deep": {"conv": 50, "mlp": 80},
    "overnight": {"conv": 30, "mlp": 50},  # Standard tier, cifar10 only
}


@dataclass
class ModelSpec:
    family: str
    name: str
    arch_group: str
    budget: int
    primary_task: str


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def resolve_models() -> list[ModelSpec]:
    """Resolve all models to run, grouped by architecture."""
    from bioplausible.cli.run import _model_compatible, _resolve_family_models

    # Map overnight -> standard for eval config, but use overnight budgets
    budget_tier = CONFIG["tier"]

    models = []
    for arch_name, arch_cfg in ARCH_GROUPS.items():
        budget_per = TIER_BUDGETS[budget_tier].get(arch_name, arch_cfg["budget_per_model"])
        primary_task = arch_cfg["task"]

        for cli_family, model_name in arch_cfg["models"]:
            # Verify model exists and is compatible
            reg_fam, fam_models = _resolve_family_models(cli_family)
            if model_name not in fam_models:
                logging.warning(
                    f"Model {model_name} not found in family {cli_family}, skipping"
                )
                continue
            if model_name in EXCLUDED_MODELS:
                logging.info(f"Excluding baseline model: {model_name}")
                continue
            if not _model_compatible(model_name, primary_task):
                logging.warning(
                    f"Model {model_name} not compatible with {primary_task}, skipping"
                )
                continue

            models.append(
                ModelSpec(
                    family=cli_family,
                    name=model_name,
                    arch_group=arch_name,
                    budget=budget_per,
                    primary_task=primary_task,
                )
            )

    logging.info(
        f"Resolved {len(models)} models across {len(ARCH_GROUPS)} architecture groups"
    )
    for arch_name in ARCH_GROUPS:
        arch_models = [m for m in models if m.arch_group == arch_name]
        logging.info(
            f"  {arch_name.upper()}: {len(arch_models)} models x {arch_models[0].budget if arch_models else 0} trials = {sum(m.budget for m in arch_models)} total"
        )
        for m in arch_models:
            logging.info(f"    - {m.name} ({m.family})")
    return models


def write_experiment_config(models: list[ModelSpec], path: str, eval_tier: str):
    """Write the experiment config as YAML for run_experiment.py."""
    import yaml

    # Group by family for run_experiment.py
    families = sorted(set(m.family for m in models))
    tasks = CONFIG.get("tasks", ["cifar10", "digits"])  # Configurable tasks

    # Per-model budgets so conv models (30) get more trials than MLP (20),
    # even when they share a family (e.g. eqprop). run_experiment.py supports
    # <family>.<model> keys in model_budgets.
    family_budgets = {}
    model_budgets: dict[str, int] = {}
    for m in models:
        family_budgets.setdefault(m.family, m.budget)
        model_budgets[f"{m.family}.{m.name}"] = m.budget

    config = {
        "families": families,
        "tasks": tasks,
        "budget": 20,  # fallback
        "tier": eval_tier,  # Use mapped eval tier
        "family_budgets": family_budgets,
        "model_budgets": model_budgets,
        "seed": CONFIG["seed"],
        # Phase 1.5 specific
        "phase": "1.5",
        "arch_groups": {
            k: [m.name for m in models if m.arch_group == k] for k in ARCH_GROUPS
        },
        "multi_objective": True,
        "objectives": CONFIG["objectives"],
        "directions": CONFIG["directions"],
        "max_params": CONFIG.get("max_params"),
        "sampler": CONFIG.get("sampler", "tpe"),
    }
    Path(path).write_text(yaml.dump(config, sort_keys=False))
    return path


def run_experiment(models: list[ModelSpec]):
    """Launch the round-robin runner as a subprocess."""
    # Map overnight -> standard for eval config
    eval_tier = "standard" if CONFIG["tier"] == "overnight" else CONFIG["tier"]
    tasks_str = ",".join(CONFIG.get("tasks", ["cifar10", "digits"]))

    cfg_path = write_experiment_config(models, "experiments/phase1_5_auto.yaml", eval_tier)

    cmd = [
        "uv",
        "run",
        "python",
        "-u",
        "run_experiment.py",
        "--config",
        cfg_path,
        "--db",
        CONFIG["db"],
        "--interval",
        str(CONFIG["interval"]),
        "--emit-every",
        str(CONFIG["emit_every"]),
        "--device",
        CONFIG["device"],
        "--seed",
        str(CONFIG["seed"]),
        "--task",
        tasks_str,
        "--budget-tier",
        eval_tier,
    ]

    logging.info("[PHASE1.5] Starting experiment: %s", " ".join(cmd))
    logging.info(
        "[PHASE1.5] Models: %d across %d arch groups", len(models), len(ARCH_GROUPS)
    )
    total_trials = sum(m.budget for m in models)
    logging.info("[PHASE1.5] Total trials: %d", total_trials)

    start = time.time()
    process = subprocess.Popen(cmd)
    try:
        process.wait()
    except KeyboardInterrupt:
        logging.warning("[PHASE1.5] Interrupted; terminating child...")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        logging.warning(
            "[PHASE1.5] Child stopped; studies persisted in %s. Re-run to resume.",
            CONFIG["db"],
        )
        return 130
    elapsed = time.time() - start
    logging.info(
        "[PHASE1.5] Completed in %.1f min (exit=%d)", elapsed / 60, process.returncode
    )
    return process.returncode


def build_portfolio(models: list[ModelSpec]):
    """Generate final portfolio CSVs with Pareto frontiers per architecture group."""
    logging.info("[PHASE1.5] Building final portfolio...")

    Path("results/phase1_5").mkdir(exist_ok=True)

    # Overall portfolio
    cmd_overall = [
        "uv",
        "run",
        "biopl-hpo",
        "portfolio",
        "--tasks",
        "cifar10,digits",
        "--output",
        "results/phase1_5/portfolio.csv",
        "--db",
        CONFIG["db"],
    ]
    result = subprocess.run(cmd_overall, capture_output=True, text=True)
    if result.returncode != 0:
        logging.warning("[PHASE1.5] Overall portfolio failed: %s", result.stderr)

    # Per-architecture-group portfolios
    for arch_name in ARCH_GROUPS:
        arch_models = [m.name for m in models if m.arch_group == arch_name]
        if not arch_models:
            continue
        studies_arg = ",".join(f"{arch_name}_{m}" for m in arch_models)
        cmd_arch = [
            "uv",
            "run",
            "biopl-hpo",
            "portfolio",
            "--tasks",
            "cifar10,digits",
            "--studies",
            studies_arg,
            "--output",
            f"results/phase1_5/portfolio_{arch_name}.csv",
            "--db",
            CONFIG["db"],
        ]
        result = subprocess.run(cmd_arch, capture_output=True, text=True)
        if result.returncode == 0:
            logging.info(
                "[PHASE1.5] %s portfolio written to results/phase1_5/portfolio_%s.csv",
                arch_name.upper(),
                arch_name,
            )
        else:
            logging.warning(
                "[PHASE1.5] %s portfolio failed: %s", arch_name, result.stderr
            )

    # Print summary
    subprocess.run(
        [
            "uv",
            "run",
            "biopl-hpo",
            "compare",
            "--family",
            "all",
            "--task",
            "cifar10",
            "--db",
            CONFIG["db"],
        ],
        check=False,
    )
    return 0


def main():
    setup_logging()
    Path("results/phase1_5").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("experiments").mkdir(exist_ok=True)

    logging.info("=" * 60)
    logging.info("  PHASE 1.5 TURNKEY EXPERIMENT")
    logging.info("  Architecture-aware, multi-objective HPO")
    logging.info("  Tier: %s", CONFIG["tier"])
    logging.info("=" * 60)

    models = resolve_models()
    if not models:
        logging.error("No models resolved. Check configuration.")
        sys.exit(1)

    rc = run_experiment(models)
    if rc == 0:
        build_portfolio(models)
        logging.info("[PHASE1.5] ✅ Experiment complete. Check results/phase1_5/")
    else:
        logging.error(
            "[PHASE1.5] ❌ Experiment failed (exit=%d). Re-run to resume.", rc
        )
        sys.exit(rc)


if __name__ == "__main__":
    main()
