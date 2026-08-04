#!/usr/bin/env python
"""
Google Colab Phase 1 HPO Experiment Runner

Imports bioplausible from GitHub (autonull/bioplausible) and runs the same
turnkey Phase 1 experiment. Designed to run in a Colab GPU runtime.

Usage in Colab:
    # 1. Install uv
    !curl -LsSf https://astral.sh/uv/install.sh | sh
    !source $HOME/.local/bin/env
    # 2. Clone and run
    !git clone https://github.com/autonull/bioplausible.git
    %cd bioplausible
    !uv run python run_phase1_colab.py

Or copy-paste this script into a Colab cell and run.
"""

# ---------------------------------------------------------------------------
# Colab setup: install uv, clone repo, install deps
# ---------------------------------------------------------------------------
import os
import pathlib
import subprocess
import sys


def setup_colab():
    """Install uv and clone bioplausible if not already done."""
    # Check if we're in Colab
    in_colab = "google.colab" in sys.modules or os.getenv("COLAB_GPU") is not None

    if in_colab:
        # Install uv
        subprocess.run(
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            shell=True, check=True
        )
        os.environ["PATH"] = f"{os.path.expanduser('~/.local/bin')}:{os.environ['PATH']}"

        # Clone repo if not present
        if not pathlib.Path("bioplausible").exists():
            subprocess.run("git clone https://github.com/autonull/bioplausible.git", shell=True, check=True)

        os.chdir("bioplausible")
        # Sync deps (uv will use pyproject.toml + uv.lock)
        subprocess.run("uv sync", shell=True, check=True)
        print("✅ Colab setup complete")


# ---------------------------------------------------------------------------
# Configuration (same as local run_phase1.py)
# ---------------------------------------------------------------------------
CONFIG = {
    "db": "compute.db",
    "tier": "standard",
    "interval": 30,
    "emit_every": 10,
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
    "family_budgets": {
        "backprop": 30,
        "fa": 20,
        "forward_only": 20,
        "eqprop": 15,
        "hebbian": 15,
        "target_prop": 15,
        "spiking": 15,
        "predictive_coding": 15,
    },
    "seed": 42,
    "device": "auto",
}


def write_yaml_config(path: str):
    import yaml
    config = {
        "families": CONFIG["families"],
        "tasks": CONFIG["tasks"],
        "budget": 20,
        "tier": CONFIG["tier"],
        "family_budgets": CONFIG["family_budgets"],
        "seed": CONFIG["seed"],
    }
    from pathlib import Path
    Path(path).write_text(yaml.dump(config, sort_keys=False))
    return path


def run_colab_experiment():
    """Main entry point for Colab."""
    import logging
    import subprocess
    import time
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    # Setup Colab environment
    setup_colab()

    # GPU check
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Create dirs
    Path("results").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("experiments").mkdir(exist_ok=True)

    # Write config
    cfg_path = write_yaml_config("experiments/phase1_colab.yaml")

    # Run experiment
    cmd = [
        "uv", "run", "python", "run_experiment.py",
        "--config", cfg_path,
        "--db", "compute.db",
        "--interval", "30",
        "--emit-every", "10",
        "--device", "auto",
        "--seed", "42",
    ]

    print("=" * 60)
    print("  COLAB PHASE 1 EXPERIMENT")
    print(f"  Families: {len(CONFIG['families'])}")
    print("  Tasks: digits, cifar10")
    print("  Tier: standard")
    print("=" * 60)

    start = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"✅ Experiment complete in {elapsed / 60:.1f} min")
        # Build portfolio
        subprocess.run([
            "uv", "run", "biopl-hpo", "portfolio",
            "--tasks", "digits,cifar10",
            "--output", "results/portfolio.csv",
            "--db", "compute.db",
        ], check=False)
        print("📊 Portfolio saved to results/portfolio.csv")
    else:
        print(f"❌ Experiment failed (exit={result.returncode})")
        print("Re-run this cell to resume (studies persist in compute.db)")


if __name__ == "__main__":
    run_colab_experiment()
