#!/usr/bin/env python
"""
Turnkey Phase 1 HPO experiment runner.

Runs a complete, compute-matched HPO sweep across all bio-plausible families
on Digits + CIFAR-10, with a round-robin scheduler, live dashboard, and
crash-resume. Designed to finish in ~2 hours on a single GPU (RTX 3080 / A10G).

Usage:
    uv run python run_phase1.py          # Local run (uses local code)
    # Colab: see run_phase1_colab.py

Outputs:
    compute.db                    # SQLite with all Optuna studies
    results/portfolio.csv         # Final ranked portfolio (Scale/Hold/Eliminated)
    results/portfolio_interim.csv # Live updates every 10 trials
    logs/experiment.log           # Full run log
    results/portfolio_final.png   # Pareto frontiers (if pareto subcommand available)
"""

import logging
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration — EDIT HERE to change budgets, families, etc.
# ---------------------------------------------------------------------------
CONFIG = {
    "db": "compute.db",
    "tier": "standard",          # "smoke" | "shallow" | "standard" | "deep"
    "interval": 30,              # Dashboard refresh seconds
    "emit_every": 10,            # Interim portfolio every N trials
    # Families to include (CLI labels from bioplausible.cli.run.FAMILY_MAP)
    "families": [
        "backprop",
        "fa",
        "forward_only",
        "eqprop",
        "hebbian",
        "target_prop",
        "spiking",
        "predictive_coding",
    ],
    "tasks": ["digits", "cifar10"],
    # Per-family budget overrides (trials per model). Total wall-time ~2h on RTX 3080.
    "family_budgets": {
        "backprop": 30,         # Baseline needs stable reference
        "fa": 20,               # 11 models — keep modest for first pass
        "forward_only": 20,     # Only 2 models; can afford more
        "eqprop": 15,           # Many prune; 15 surviving is signal
        "hebbian": 15,
        "target_prop": 15,
        "spiking": 15,
        "predictive_coding": 15,
    },
    "seed": 42,
    "device": "auto",
}


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def write_yaml_config(path: str):
    """Write the experiment config as YAML for run_experiment.py."""
    import yaml

    config = {
        "families": CONFIG["families"],
        "tasks": CONFIG["tasks"],
        "budget": 20,  # default fallback
        "tier": CONFIG["tier"],
        "family_budgets": CONFIG["family_budgets"],
        "seed": CONFIG["seed"],
    }
    Path(path).write_text(yaml.dump(config, sort_keys=False))
    return path


def run_experiment():
    """Launch the round-robin runner as a subprocess."""
    cfg_path = write_yaml_config("experiments/phase1_auto.yaml")

    cmd = [
        "uv",
        "run",
        "python",
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
    ]

    logging.info("[PHASE1] Starting experiment: %s", " ".join(cmd))
    logging.info("[PHASE1] Budget summary:")
    for fam, b in CONFIG["family_budgets"].items():
        # Estimate trials (models per family from registry)
        if fam == "backprop":
            models = 3
        elif fam == "fa":
            models = 11
        elif fam == "forward_only":
            models = 2
        elif fam == "eqprop":
            models = 12
        else:
            models = 4  # typical
        total_trials = models * CONFIG["family_budgets"][fam]
        logging.info("  %s: %d models x %d = %d trials", fam, models, CONFIG["family_budgets"][fam], total_trials)

    start = time.time()
    # Stream output live so the dashboard prints in real time
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        logging.info(line.rstrip())
    process.wait()
    elapsed = time.time() - start
    logging.info("[PHASE1] Completed in %.1f min (exit=%d)", elapsed / 60, process.returncode)
    return process.returncode


def build_portfolio():
    """Generate final portfolio CSV from the completed studies."""
    logging.info("[PHASE1] Building final portfolio...")
    cmd = [
        "uv",
        "run",
        "biopl-hpo",
        "portfolio",
        "--tasks",
        "digits,cifar10",
        "--output",
        "results/portfolio.csv",
        "--db",
        CONFIG["db"],
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        logging.info("[PHASE1] Portfolio written to results/portfolio.csv")
        # Print summary
        subprocess.run(
            ["uv", "run", "biopl-hpo", "compare", "--family", "all", "--task", "digits", "--db", CONFIG["db"]],
            check=False,
        )
    else:
        logging.warning("[PHASE1] Portfolio build failed: %s", result.stderr)
    return result.returncode


def main():
    setup_logging()
    Path("results").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("experiments").mkdir(exist_ok=True)

    logging.info("=" * 60)
    logging.info("  PHASE 1 TURNKEY EXPERIMENT")
    logging.info("  Families: %s", ", ".join(CONFIG["families"]))
    logging.info("  Tasks: %s", ", ".join(CONFIG["tasks"]))
    logging.info("  Tier: %s", CONFIG["tier"])
    logging.info("=" * 60)

    rc = run_experiment()
    if rc == 0:
        build_portfolio()
        logging.info("[PHASE1] ✅ Experiment complete. Check results/portfolio.csv")
    else:
        logging.error("[PHASE1] ❌ Experiment failed (exit=%d). Re-run to resume.", rc)
        sys.exit(rc)


if __name__ == "__main__":
    main()