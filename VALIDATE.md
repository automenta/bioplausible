# VALIDATE.md — Grounded Actionable Validation Plan (Phase 0 + Phase 1)

**Status**: Sprints −1, 0, 1, 2, 3 complete. HPO infrastructure exists but **not wired to CLI**. Ready for Stage A (wiring) → Phase 0 → Phase 1.

**Goal**: Run compute-matched HPO across all 12+ propagator families on Digits + CIFAR-10, produce statistically rigorous parity comparisons, and identify which algorithms have genuine headroom.

---

## Reality Check: What Exists vs. What Plans Assume

| Assumed in original plan | Actual state | Action needed |
|---|---|---|
| `biopl-hpo` CLI | `cli/run.py::main` has `search` subcommand; **not registered** as console script | **Stage A.1**: Register as `biopl-hpo` |
| `biopl-hpo compare` | `hyperopt/comparison.py` has `compute_algorithm_rankings`; **no CLI** | **Stage A.2**: Add `compare` subcommand |
| Pareto plots | `optuna_bridge.get_pareto_trials` + `analysis/scaling.plot_scaling_curves` exist; **no glue** | **Stage A.3**: Add `pareto` subcommand |
| Track 10 = memory demo | Computes **theoretical** formulas only | **Stage B**: Modify to measure `torch.cuda.max_memory_allocated()` |
| `biopl-parity` | **Exists and works** | Use as-is |
| `biopl-failure-manifesto` | **Exists and works** | Use for negative results |

---

## Stage A: Wire Existing HPO to CLI (1 day, unblocks everything)

### A.1 Register `biopl-hpo` Console Script

**File**: `pyproject.toml` — add to `[project.scripts]`:
```toml
biopl-hpo = "bioplausible.cli.hpo:main"
```

**File**: `bioplausible/cli/hpo.py` — **new thin shim** (≈50 lines):
```python
"""CLI entry for HPO. Delegates to cli/run.py search logic."""
from bioplausible.cli.run import main as run_main

def main():
    import sys
    # Default to search subcommand if no args
    if len(sys.argv) == 1:
        sys.argv.append("search")
    run_main()

if __name__ == "__main__":
    main()
```

**Then extend `cli/run.py::search_parser`** to support Phase 0 flags:
```python
# In run.py, inside main() where search_parser is defined:
search_parser.add_argument("--family", choices=["eqprop", "forward_only", "feedback_alignment", "equitile", "hebbian", "predictive_coding", "target_prop", "spiking", "mep", "backprop", "all"], default="all")
search_parser.add_argument("--task", choices=["digits", "cifar10", "tiny_shakespeare", "mnist"], default="digits")
search_parser.add_argument("--budget", type=int, default=200, help="Optuna trials per model")
search_parser.add_argument("--seeds", type=int, default=5, help="Seeds for top-3 configs")
search_parser.add_argument("--method", choices=["bayesian", "random"], default="bayesian")
search_parser.add_argument("--output", type=str, help="JSONL output path")
```

**Verify**:
```bash
uv run biopl-hpo search --family eqprop --task digits --budget 10 --seeds 1
# Must complete without error, write trials to bioplausible.db
```

### A.2 Add `compare` Subcommand (Portfolio Ranking)

**Extend `cli/run.py`**:
```python
# New subparser
compare_parser = subparsers.add_parser("compare", help="Compare families, output ranking CSV")
compare_parser.add_argument("--studies", required=True, help="Comma-separated study names")
compare_parser.add_argument("--metric", default="accuracy", choices=["accuracy", "loss", "param_efficiency"])
compare_parser.add_argument("--output", required=True, help="Output CSV path")
```

**Implement `run_compare(args)`** using existing `hyperopt.comparison`:
```python
def run_compare(args):
    from bioplausible.hyperopt.storage import HyperoptStorage
    from bioplausible.hyperopt.comparison import (
        compute_algorithm_rankings, group_trials_by_family, generate_comparison_summary
    )
    from bioplausible.core.registry import ComponentCategory, Registry

    storage = HyperoptStorage("bioplausible.db")
    all_trials = []
    for study_name in args.studies.split(","):
        trials = storage.get_trials_by_study(study_name)
        all_trials.extend(trials)

    grouped = group_trials_by_family(all_trials)
    rankings = compute_algorithm_rankings(grouped, metric=args.metric)

    # Write CSV
    import csv
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "family", "best_value", "avg_value", "std_value", "n_trials", "best_trial_id"])
        for r in rankings:
            writer.writerow([r.rank, r.family, r.best_value, r.avg_value, r.std_value, r.n_trials, r.best_trial_id])

    print(generate_comparison_summary(rankings, baseline="backprop"))
```

**Verify**:
```bash
uv run biopl-hpo compare --studies eqprop_digits,fa_digits,backprop_digits --output results/portfolio_digits.csv
```

### A.3 Add `pareto` Subcommand (Pareto Frontier Plots)

**Extend `cli/run.py`**:
```python
pareto_parser = subparsers.add_parser("pareto", help="Generate Pareto frontier plots")
pareto_parser.add_argument("--study", required=True)
pareto_parser.add_argument("--output-dir", default="results/pareto")
pareto_parser.add_argument("--format", choices=["html", "png", "json"], default="html")
```

**Implement** using `optuna_bridge.get_pareto_trials` + `analysis.scaling.plot_scaling_curves`.

---

## Stage B: Fix Track 10 — Measured Memory (Critical for Phase 3)

**File**: `bioplausible/validation/tracks/scaling_tracks.py::track_10_memory_scaling`

**Change**: Replace theoretical calculation with **measured peak memory**:

```python
def measure_peak_memory(model, dataloader, device, epochs=1):
    """Actual measured peak memory during training."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    model.train()
    for epoch in range(epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            loss = model.train_step(x, y)["loss"]
            # For equilibrium models: model.settle(x, y) or equivalent
            loss.backward() if hasattr(model, "backward") else None
    
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
    else:
        import psutil
        peak_mb = psutil.Process().memory_info().rss / 1e6
    return peak_mb

def track_10_memory_scaling(verifier) -> TrackResult:
    # ... existing setup ...
    for depth in depths:
        model = LoopedMLP(...).to(device)
        # Measure actual memory
        train_loader = DataLoader(...)  # small subset for speed
        eqprop_peak = measure_peak_memory(model, train_loader, device)
        
        # Backprop baseline same depth
        bp_model = BackpropMLP(...).to(device)
        bp_peak = measure_peak_memory(bp_model, train_loader, device)
        
        results[depth] = {"eqprop": eqprop_peak, "backprop": bp_peak, "ratio": bp_peak / eqprop_peak}
    # ... rest unchanged ...
```

**Gate**: Track 10 must report **measured** MB and pass at depth 50 with ratio > 5×.

---

## Phase 0: HPO Infrastructure Validation (After Stage A+B)

### 0.1 Verify End-to-End Works

```bash
# 1. Search on one family (smoke test, 5 min)
uv run biopl-hpo search --family eqprop --task digits --budget 10 --seeds 1

# 2. Parity baseline (exists)
uv run biopl-parity --config-a backprop_mlp --config-b eqprop_mlp --task digits --epochs 5 --seed 0 --json

# 3. Full Phase 0 gate: 3 families × 200 trials × 5 seeds (GPU, ~2-4 hrs)
uv run biopl-hpo search --family eqprop --task digits --budget 200 --seeds 5
uv run biopl-hpo search --family forward_only --task digits --budget 200 --seeds 5
uv run biopl-hpo search --family feedback_alignment --task digits --budget 200 --seeds 5

# 4. Backprop baseline (same protocol)
uv run biopl-hpo search --family backprop --task digits --budget 200 --seeds 5
uv run biopl-hpo search --family backprop --task cifar10 --budget 200 --seeds 5
```

### 0.2 Family Groupings (Map `--family` to Search Spaces)

| Family | Search Space Keys (from `SEARCH_SPACES` in `hyperopt/search_space.py`) |
|--------|------------------------------------------------------------------------|
| `eqprop` | `eqprop_mlp`, `Holomorphic EqProp`, `Directed EqProp (Deep EP)`, `Finite-Nudge EqProp`, `Conv EqProp (CIFAR-10)` |
| `forward_only` | `forward_forward`, `pepita` |
| `feedback_alignment` | `standard_fa`, `adaptive_feedback_alignment`, `dfa_deep`, `direct_feedback_alignment_eqprop`, `energy_guided_fa`, `energy_minimizing_fa`, `layerwise_equilibrium_fa`, `equilibrium_alignment` |
| `equitile` | `equitile`, `EquiTile EP`, `LM EquiTile`, `RL EquiTile`, `Conv EquiTile` |
| `hebbian` | `hebbian_chain`, `hebbian_3d`, `three_factor_hebbian`, `deep_hebbian` |
| `predictive_coding` | `fabricpc_graph_pcn`, `predictive_coding_hybrid` |
| `target_prop` | `diff_target_prop` |
| `spiking` | `spiking_stdp` |
| `mep` | `smep`, `smep_fast`, `sdmep`, `local_ep`, `natural_ep`, `muon_backprop` |

**Implementation**: `cli/hpo.py` maps `--family` → list of model names → loops `run_search` per model.

### 0.3 Statistical Rigor (Per Algorithm, After HPO)

After HPO completes for a family, run **verification**:
```bash
# Get top-3 trial configs from study, re-run each with n=5 seeds
uv run biopl-hpo verify --study eqprop_digits --top-k 3 --seeds 5 --output results/eqprop_verified.jsonl
```
*(New `verify` subcommand needed in Stage A)*

### 0.4 Phase 0 Gate (Definition of Done)

- [ ] `biopl-hpo search` runs end-to-end on **eqprop**, **forward_only**, **feedback_alignment** on digits
- [ ] Each produces statistically valid parity comparison (n≥5 seeds on top-3 configs)
- [ ] Backprop baseline tuned with identical protocol (same budget, method, seeds)
- [ ] `biopl-hpo pareto` generates accuracy vs compute plots for each family
- [ ] `docs/hpo_protocol.md` written with exact search protocol (reproducible)

---

## Phase 1: Portfolio Revelation

### 1.1 Digits Completion (All 12+ Families)

```bash
# Runs all families sequentially (uses family map from 0.2)
uv run biopl-hpo search --family all --task digits --budget 200 --seeds 5
```

**Elimination Criterion** (implemented in `verify` step):
- Eliminated if: best tuned accuracy > 15 pp below best backprop baseline AND no structural regime advantage
- Survives if ANY of:
  - Tuned parity gap < 5 pp on digits
  - Tuned parity gap < 10 pp AND O(1) memory or forward-only structure (from registry `locality_level`)
  - Tuned parity gap < 10 pp AND enables continual learning (registry `family` in {eqprop, fa, hebbian, forward_only})

**Output**: `results/portfolio_digits.csv` via `biopl-hpo compare --studies ...`

### 1.2 CIFAR-10 Entry (Credibility Threshold)

For each survivor from digits:
```bash
# Automated: reads survivors CSV, runs CIFAR-10 HPO
uv run biopl-hpo search --family survivors --task cifar10 --budget 200 --seeds 5
```

**Architecture**: Fixed CNN (4-conv + 2-FC, ~500K params) from `experiments/presets.py`. If algorithm requires modification (e.g., EquiTile tiling), document and run backprop on same modified arch.

### 1.3 Portfolio Ranking Table (Final Phase 1 Artifact)

```bash
uv run biopl-hpo compare --studies eqprop_cifar10,fa_cifar10,backprop_cifar10,... --output results/portfolio_final.csv
```

Generates `results/portfolio_final_ranking.csv`:

| Rank | Algorithm | Digits Acc | CIFAR Acc | Parity Gap | Peak Mem | Wall Time | Regime Advantage | Status |
|------|-----------|------------|-----------|------------|----------|-----------|------------------|--------|
| 1 | ... | ... | ... | ... | ... | ... | ... | **Scale** |
| 2 | ... | ... | ... | ... | ... | ... | ... | **Scale** |
| 3 | ... | ... | ... | ... | ... | ... | ... | **Hold** |
| ... | ... | ... | ... | ... | ... | ... | ... | **Eliminated** |

**Status**: **Scale** → Phase 2; **Hold** → revisit; **Eliminated** → `biopl-failure-manifesto` entry.

### 1.4 Phase 1 Shareability Gates

**Level 1 (Internal — Team Continue Decision)**:
- [ ] ≥1 equilibrium algorithm: tuned parity gap < 5 pp on digits
- [ ] ≥1 algorithm (any family): tuned parity gap < 10 pp on CIFAR-10
- [ ] Portfolio ranking complete with elimination justifications
- [ ] All results reproducible via `biopl-hpo` with documented seeds

**Level 2 (Preprint-Worthy)**:
- [ ] ≥2 families: tuned parity gap < 5 pp on digits (n ≥ 10 seeds)
- [ ] ≥1 algorithm: tuned parity gap < 8 pp on CIFAR-10 (n ≥ 5 seeds)
- [ ] Compute-matched backprop baselines with identical search budgets
- [ ] Negative results documented with search budgets and best configs
- [ ] HPO protocol fully documented (`docs/hpo_protocol.md`)
- [ ] Effect sizes (Cohen's d) reported for all parity gaps

---

## Decision Gate: After Phase 1

**Question**: Does any algorithm achieve tuned parity gap < 10 pp on CIFAR-10?

| Answer | Action |
|--------|--------|
| **Yes** | Continue to Phase 2 (VALIDATE2.md) with top 2–3 algorithms |
| **No, but gap < 15 pp with clear regime advantage** | Continue to Phase 2, emphasize regime demo over raw accuracy |
| **No, and no regime advantage** | Publish negative result via `biopl-failure-manifesto`. Document search budgets. Reassess algorithmic approaches. |

---

## Implementation Checklist (What to Actually Build)

| Task | File(s) | Effort | Depends On |
|------|---------|--------|------------|
| A.1 Register `biopl-hpo` console script | `pyproject.toml`, `cli/hpo.py` (new) | 30 min | — |
| A.2 Add `--family`, `--budget`, `--seeds`, `--output` to `search` | `cli/run.py` | 1 hr | A.1 |
| A.3 Add `compare` subcommand | `cli/run.py` + `hyperopt/comparison.py` | 1 hr | A.1 |
| A.4 Add `verify` subcommand (top-k re-run n seeds) | `cli/run.py` + `hyperopt/experiment.py` | 1 hr | A.1 |
| A.5 Add `pareto` subcommand | `cli/run.py` + `optuna_bridge` + `analysis/scaling` | 1 hr | A.1 |
| B Fix Track 10 measured memory | `validation/tracks/scaling_tracks.py` | 2 hrs | — |
| 0.1 Write `docs/hpo_protocol.md` | `docs/hpo_protocol.md` (new) | 1 hr | A.1-A.5 |
| **Total Stage A+B** | | **~6-7 hrs** | — |

**After Stage A+B**: Phase 0/1 commands in this document become runnable.