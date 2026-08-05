# FIX2.md — Ideal Experiment Framework & Development Plan

**Goal**: Design a rigorous, configurable, and continuously escalating experiment framework that produces *convincing, publication-grade evidence* for bio-plausible learning parity with backprop. Every experiment must be a stepping stone to the next — no dead ends, no wasted compute, no debugging hell.

---

## 1. Ideal Results & Measurement Evidence Epistemology

### What "Convincing" Looks Like (Evidence Hierarchy)

| Level | Claim | Required Evidence | Experiment Type |
|-------|-------|-------------------|-----------------|
| **L0: Sanity** | Model runs, gradients flow, no NaNs | Single-batch overfit test (1 batch, 100% train acc) | 30-sec smoke test |
| **L1: Learning** | Model learns on trivial task | MNIST 98%+ in <2 min, 5 seeds | 5-min validation |
| **L2: Scaling** | Accuracy improves with compute | Depth/width sweeps, power-law fits (R² > 0.9) | 1-hr scaling sweep |
| **L3: Parity** | Matches BP at matched compute | FLOP-matched, wall-time-matched, 5 seeds, CIs | 4-hr parity suite |
| **L4: Superiority** | Beats BP in a regime | Lower memory, faster convergence, better OOD | Overnight campaign |
| **L5: Transfer** | Generalizes across domains | Vision→LM→RL→Graph, same algorithm | Multi-day campaign |

**Every experiment must output**: JSON with `task, model, seed, config, metrics, traces, artifacts` — machine-readable, no manual parsing.

---

## 2. Continuous Escalation Pipeline (Short → Long)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  TIER 0: SMOKE (30 sec)                                                     │
│  • Single batch overfit (1 batch, 10 steps, 100% train acc)                │
│  • Gradient norm check (no NaN, no explosion)                               │
│  • Memory footprint < threshold                                             │
│  → GATE: All registered models must pass before any real experiment        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  TIER 1: VALIDATION (5 min)                                                 │
│  • MNIST 98%+ in <2 min, 5 seeds                                            │
│  • Fashion-MNIST 90%+ in <3 min                                             │
│  • Gradient equivalence test (finite-diff vs local update)                  │
│  → GATE: Models failing TIER 1 are excluded from TIER 2+                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  TIER 2: SCALING SWEEP (1 hr)                                               │
│  • Depth: 2,4,8,16,32 layers                                                │
│  • Width: 64,128,256,512                                                    │
│  • Tasks: MNIST, Fashion-MNIST, CIFAR-10 (flattened)                       │
│  • Models: backprop, eqprop_mlp, neural_cube, deep_hebbian, standard_fa    │
│  • Output: Power-law fits (acc ~ params^α, depth^β, time^γ) + Pareto       │
│  → GATE: Models with α < threshold excluded from TIER 3                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  TIER 3: PARITY SUITE (4 hr)                                                │
│  • FLOP-matched: same forward+backward FLOPs per sample                    │
│  • Wall-time matched: same epoch time budget                               │
│  • Memory-matched: same peak GPU memory                                    │
│  • Tasks: MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100                        │
│  • 5 seeds, bootstrap CIs, effect sizes (Cohen's d)                        │
│  • Output: Publication-ready tables + Pareto plots + JSON                  │
│  → GATE: Models within 2% of BP accuracy advance to TIER 4                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│  TIER 4: SUPERIORITY CAMPAIGN (overnight)                                   │
│  • Regime: where BP fails (deep >50 layers, low memory, OOD, continual)    │
│  • Metrics: memory, convergence speed, OOD robustness, forgetting          │
│  • Models: EquiTile variants, MEP, progressive_locality                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key principle**: Each tier's output *feeds directly* into the next tier's config. No manual intervention.

---

## 3. Single Source of Truth: Experiment Config

**One YAML file per campaign** — no hardcoded values anywhere in bioplausible/.

```yaml
# experiments/campaign_parity_mlp.yaml
meta:
  name: "parity_mlp_cifar10"
  description: "BP vs bio-plausible MLP parity on CIFAR-10 (flattened)"
  created: "2026-08-05"
  git_commit: "{{git_commit}}"

compute:
  device: "auto"                    # auto / cuda:0 / cpu
  max_parallel: 1                   # studies in parallel
  max_wall_hours: 4                 # hard wall-clock limit

search_space:                       # Single source of truth for bounds
  hidden_dim: [16, 32, 64]
  num_layers: [1, 2, 4, 8]
  lr: [1e-4, 1e-2]                  # log-uniform
  batch_size: [64, 128, 256]
  # Model-specific overrides (merged at runtime)
  model_overrides:
    eqprop_mlp:
      beta: [0.01, 0.5]
      steps: [10, 20, 50]
    neural_cube:
      cube_size: [3, 4, 5]          # NOT hidden_dim

models:
  families:
    - backprop
    - eqprop
    - hebbian
    - target_prop
  excluded:                         # Explicit, not hidden in code
    - lazy_eqprop
    - holomorphic_ep
    - pepita

tasks:
  - name: mnist
    epochs: 10
    input_dim: 784
    num_classes: 10
  - name: fashion_mnist
    epochs: 10
    input_dim: 784
    num_classes: 10
  - name: cifar10
    epochs: 50                      # Proper convergence
    input_dim: 3072
    num_classes: 10

hpo:
  sampler: "nsga2"                  # Multi-objective
  objectives:
    - accuracy                      # maximize
    - param_count                   # minimize
    - epoch_time                    # minimize
  n_trials: 200                     # Per model per task
  n_startup_trials: 10
  max_params: 210_000               # Hard constraint (pre-sampling)
  prune_worse_than_pareto: true
  n_seeds: 5                        # Statistical rigor

resources:
  max_params: 210_000               # Hard cap (enforced at sample time)
  max_epoch_time_sec: 120           # Prune slow trials early
  early_stop_patience: 10           # Epochs without improvement

output:
  db: "results/parity_mlp_cifar10.db"
  artifacts_dir: "artifacts/parity_mlp_cifar10"
  log_level: "INFO"
  emit_every: 10                    # Interim portfolio

reproducibility:
  seed: 42
  capture_env: true                 # git, torch, cuda, deps hash
  artifact_hash: true               # Content-addressable checkpoints
```

**Rule**: If it's not in the YAML, it's not configurable. No `CONFIG` dicts in Python.

---

## 4. Refactoring Targets (Ambitious)

### 4.1 Single Experiment Runner (`run_experiment.py` → `biopl-run`)

```bash
# One command, fully configurable
biopl-run campaign_parity_mlp.yaml

# Resume from checkpoint
biopl-run campaign_parity_mlp.yaml --resume

# Dry run (validate config, estimate time)
biopl-run campaign_parity_mlp.yaml --dry-run

# Override any value from CLI
biopl-run campaign_parity_mlp.yaml --override models.families='["eqprop"]' --override hpo.n_trials=50
```

### 4.2 Unified Search Space (`HyperparameterMetamodel` → `SearchSpace`)

```python
# bioplausible/hyperopt/search_space.py
class SearchSpace:
    """Declarative, composable, serializable."""
    
    def __init__(self, base: dict, overrides: dict = None):
        self.base = base
        self.overrides = overrides or {}
    
    def for_model(self, model_name: str, task: str) -> dict:
        """Return Optuna-ready param distributions with model/task overrides applied."""
        # 1. Start with base space
        # 2. Apply model-specific overrides from YAML
        # 3. Apply task-specific constraints (e.g., max_params for CIFAR-10 input)
        # 4. Return frozen distributions (Optuna Categorical/Int/Float)
    
    def estimate_param_count(self, config: dict, model_name: str) -> int:
        """Static analysis: compute param count from config WITHOUT training."""
        # Used for pre-training max_params filter
```

**Eliminates**: Hardcoded `hyperparameter_metamodel.py` heuristics. The YAML *is* the search space.

### 4.3 Multi-Objective with Pre-Filtering

```python
# bioplausible/hyperopt/objectives.py
class MultiObjective:
    """True multi-objective with pre-training feasibility filter."""
    
    def __init__(self, objectives: list[str], directions: list[str], 
                 max_params: int, estimator: ParamEstimator):
        self.objectives = objectives
        self.directions = directions
        self.max_params = max_params
        self.estimator = estimator  # Computes param_count from config
    
    def filter_config(self, config: dict) -> bool:
        """Pre-training hard filter: reject configs exceeding max_params."""
        return self.estimator.estimate(config) <= self.max_params
    
    def evaluate(self, trial: optuna.Trial, metrics: dict) -> tuple:
        """Post-training evaluation (for objectives not statically known)."""
        return tuple(metrics[obj] for obj in self.objectives)
```

**Eliminates**: Post-training `max_params` pruning waste (45s per trial).

### 4.3 Unified Logging (Fix Duplicate Params)

```python
# bioplausible/logging/experiment_logger.py
class ExperimentLogger:
    """Single log entry per trial, structured JSON."""
    
    def log_trial_start(self, trial_id: int, model: str, task: str, 
                        config: dict, param_count: int, seed: int):
        self._write({"event": "trial_start", "trial_id": trial_id, 
                     "model": model, "task": task, "config": config,
                     "param_count": param_count, "seed": seed})
    
    def log_trial_end(self, trial_id: int, metrics: dict, 
                      status: str, wall_time: float):
        self._write({"event": "trial_end", "trial_id": trial_id,
                     "metrics": metrics, "status": status, "wall_time": wall_time})
    
    def log_epoch(self, trial_id: int, epoch: int, metrics: dict):
        self._write({"event": "epoch", "trial_id": trial_id, "epoch": epoch,
                     "metrics": metrics})
```

**Output**: One JSONL file per study — queryable, no duplicate params, no parse hacks.

---

## 5. Ideal Experiment Outputs (What We Actually Want)

### 5.1 Per-Campaign Artifacts

```
results/
└── parity_mlp_cifar10/
    ├── campaign.yaml                 # Exact config used
    ├── campaign.db                   # Optuna studies (SQLite)
    ├── metrics.jsonl                 # All trial metrics (JSONL)
    ├── pareto_frontier.csv           # Multi-objective Pareto
    ├── scaling_laws.json             # Power-law fits with CIs
    ├── pareto_plot.html              # Interactive Plotly
    ├── summary_table.md              # Publication-ready markdown
    ├── summary_table.csv             # Machine-readable
    └── artifacts/
        ├── trial_001/
        │   ├── config.json
        │   ├── metrics.json
        │   ├── checkpoints/
        │   └── traces/               # Energy, grad norms, etc.
        └── ...
```

### 5.2 Summary Table (What We Show The World)

| Model | Task | Acc (mean±CI) | Params (median) | Time/epoch (median) | Acc/Kparam | vs BP (Cohen's d) |
|-------|------|---------------|-----------------|---------------------|------------|-------------------|
| backprop | CIFAR-10 | 0.52 [0.50, 0.54] | 100K | 3.2s | 5.2 | — (ref) |
| eqprop_mlp | CIFAR-10 | 0.48 [0.46, 0.50] | 98K | 4.1s | 4.9 | -0.8 |
| neural_cube | CIFAR-10 | 0.47 [0.45, 0.49] | 84K | 3.8s | **5.6** | -0.5 |
| standard_fa | CIFAR-10 | 0.46 [0.44, 0.48] | 105K | 3.5s | 4.4 | -0.6 |

**This is what reviewers want to see. Everything else is noise.**

---

## 6. Implementation Sequence (No Big Bang)

### Phase A: Foundation (Week 1-2)
- [ ] `SearchSpace` class replacing `HyperparameterMetamodel`
- [ ] YAML config schema with validation (pydantic)
- [ ] `ParamEstimator` for static param counting
- [ ] `biopl-run` CLI with dry-run, resume, override

### Phase B: Logging & Runner (Week 2-3)
- [ ] `ExperimentLogger` (JSONL, no duplicates)
- [ ] `biopl-run` with dry-run, resume, override
- [ ] Multi-objective with pre-filtering
- [ ] TIER 0 smoke test gate

### Phase C: Tiered Experiments (Week 3-4)
- [ ] TIER 0/1: Smoke + MNIST validation
- [ ] TIER 2: Scaling sweep (depth/width)
- [ ] TIER 3: Parity suite (FLOP-matched)
- [ ] Auto-generated summary artifacts

### Phase D: Parity Suite (Week 4-5)
- [ ] `biopl-parity` CLI
- [ ] FLOP-matching logic
- [ ] Bootstrap CIs, effect sizes
- [ ] Auto-generated publication tables/plots

---

## 7. MLP-First Strategy (Conv Later)

**Decision**: Keep MLP-only until:
1. All MLP models pass TIER 0-2 consistently
2. `eqprop_mlp` or `neural_cube` within 2% of BP on CIFAR-10 (flattened) at matched FLOPs
3. Gradient equivalence verified for all EqProp variants
4. `modern_conv_eqprop` implementation audited for efficiency

**Rationale**: Conv adds 10× complexity (spatial dims, padding, pooling, batch norm interactions). If MLP isn't solid, Conv will inherit and amplify bugs.

---

## 8. Configurability Checklist (No More Debugging Hell)

| Aspect | Current Pain | Future State |
|--------|--------------|--------------|
| Change search space | Edit Python, find metamodel, debug | Edit YAML, `biopl-run --dry-run` |
| Change tasks | Edit Python, find ARCH_GROUPS | Edit YAML `tasks:` list |
| Change max_params | Edit Python, find CONST | Edit YAML `resources.max_params` |
| Add model | Edit Python, register, find exclusions | Add to YAML `models.families` |
| Change seeds | Edit Python | Edit YAML `hpo.n_seeds` |
| Add task | Edit Python, find task registry | Edit YAML `tasks:` list |
| Debug trial | Search logs, grep params | `jq '.config' metrics.jsonl` |
| Resume crashed run | Hope DB intact, guess state | `biopl-run --resume` |
| Compare runs | Manual CSV merge | `biopl-compare run1.db run2.db` |

---

## 9. Immediate Next Steps (This Week)

1. **Create `SearchSpace` class** — replace metamodel
2. **Define YAML schema** — pydantic models for validation
3. **Implement `ParamEstimator`** — static param counting per model
4. **Build `biopl-run` CLI** — dry-run, resume, override
5. **Delete `run_phase1_5.py`** — replaced by YAML + `biopl-run`
5. **Write TIER 0 smoke test** — gate for all models

---

## 10. Success Criteria for This Plan

| Metric | Target |
|--------|--------|
| Time from "new idea" to first result | < 5 min (TIER 0) |
| Time from "new model" to parity verdict | < 4 hr (TIER 3) |
| Config change → validated experiment | < 30 sec (dry-run) |
| Duplicate log entries | 0 |
| Manual config edits in Python | 0 |
| Publication-ready output per campaign | 1 command |

---

**This plan transforms "running experiments" into "running a discovery engine" — the core thesis of RESEARCH.md.**