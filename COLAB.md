# Google Colab Setup — Bioplausible Phase 1 Experiments

This guide runs the bioplausible Phase 1 HPO experiment on a Google Colab GPU
runtime (free T4, or accessed A100/V100). The code is imported from the public
repository: **https://github.com/autonull/bioplausible**

The Colab runner (`run_phase1_colab.py`) mirrors the local `run_phase1.py`
exactly — same families, budgets, dashboard, and crash-resume semantics.

---

## Quick Start (one cell)

Create a **new Colab notebook**, ensure the runtime is set to GPU
(`Runtime → Change runtime type → Hardware accelerator: GPU`), and paste this
into a single cell:

```python
# ---- Bioplausible Phase 1 on Colab ----
import subprocess, os

# 1. Install uv (Astral's Python package/venv manager)
subprocess.run("curl -LsSf https://astral.sh/uv/install.sh | sh", shell=True, check=True)
os.environ["PATH"] = f"{os.path.expanduser('~/.local/bin')}:{os.environ['PATH']}"

# 2. Clone the repo
if not os.path.exists("bioplausible"):
    subprocess.run("git clone https://github.com/autonull/bioplausible.git", shell=True, check=True)
os.chdir("bioplausible")

# 3. Sync deps (uses pyproject.toml + uv.lock)
subprocess.run("uv sync", shell=True, check=True)

# 4. Run the experiment
subprocess.run("uv run python run_phase1_colab.py", shell=True, check=True)
```

That's it. The runner prints a live progress dashboard, writes
`results/portfolio_interim.csv`, and produces the final `results/portfolio.csv`.

---

## What Happens

| Step | What runs |
|------|-----------|
| Install | `uv` + `git clone` + `uv sync` (deps from lockfile) |
| Device | Auto-detects `cuda` (T4/A100/V100) |
| Experiment | Round-robin HPO across 8 families × {digits, cifar10}, standard tier |
| Live stats | Dashboard every 30s: done/total, best acc, gap_pp, avg_t, ETA |
| Interim output | `results/portfolio_interim.csv` every 10 trials |
| Final output | `results/portfolio.csv` on completion |

---

## Running With Custom Config

The `CONFIG` dict at the top of `run_phase1_colab.py` controls everything:

```python
CONFIG = {
    "db": "compute.db",
    "tier": "standard",      # smoke / shallow / standard / deep
    "interval": 30,          # dashboard refresh seconds
    "emit_every": 10,        # interim portfolio every N trials
    "families": ["backprop", "fa", "forward_only", "eqprop",
                 "hebbian", "target_prop", "spiking", "predictive_coding"],
    "tasks": ["digits", "cifar10"],
    "family_budgets": {
        "backprop": 30, "fa": 20, "forward_only": 20, "eqprop": 15,
        "hebbian": 15, "target_prop": 15, "spiking": 15, "predictive_coding": 15,
    },
    "seed": 42,
    "device": "auto",
}
```

Edit it in the notebook (or the file after cloning) and re-run.

---

## Crash-Resume (Colab disconnects)

Colab runtimes disconnect after ~90 min idle or on session end. **Your results
are safe**: all Optuna studies persist in `compute.db` inside the repo.

To resume:
1. Re-run the Quick Start cell (reclone is skipped if `bioplausible/` exists).
2. The runner's `_ensure_studies` reuses existing studies (`load_if_exists`),
   so it continues from the previous trial counts instead of starting over.

> If the runtime was fully reset (filesystem wiped), the clone + sync will
> redo setup but your old `compute.db` is gone — start fresh.

---

## Estimated Wall-Time

On a **free T4 (16GB)**, the default Phase 1 config (standard tier, budgets
above) takes roughly **1.5–2.5 hours**. Use `"tier": "shallow"` or lower
`family_budgets` for a quicker (<1 hr) first pass if you're just probing.

---

## Viewing / Downloading Results

| Artifact | Location | How to view |
|----------|----------|-------------|
| Live dashboard | terminal cell | Scroll up / watch `emit` lines |
| Interim portfolio | `results/portfolio_interim.csv` | `!cat results/portfolio_interim.csv` |
| Final portfolio | `results/portfolio.csv` | `!cat results/portfolio.csv` |
| All studies | `compute.db` | `UV run biopl-hpo portfolio --db compute.db` |
| Full log | `logs/experiment.log` | `!tail -f logs/experiment.log` |

To download files to your machine:
```python
from google.colab import files
files.download("results/portfolio.csv")
files.download("compute.db")
```

---

## Requirements

- A Google account with Colab access.
- Recommendation: **GPU runtime** (T4/A100). CPU works but is drastically slower
  for the equilibrium/free-phase models.
- No prior install needed — Colab has Python 3; this guide installs `uv`.
