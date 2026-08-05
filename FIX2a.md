# FIX2a.md — Alternative Experiment Framework & Development Plan

**Status**: Complete replacement candidate for FIX2.md. Compares cleanly with FIX2.md at end.

**Goal**: A rigorously staged, YAML-driven experiment framework that produces *convincing, publication-grade evidence* for bio-plausible learning parity with backprop. Every rung is a stepping stone — no dead ends, no wasted compute, no debugging hell.

---

## 1. Evidence Hierarchy (What "Convincing" Looks Like)

| Level | Claim | Required Evidence | Compute Budget | Gate to Next |
|-------|-------|-------------------|----------------|--------------|
| L0  | Sanity       | Model runs, gradients flow, no NaNs | 100% train acc on xor, 10 seeds | 30 sec | All must pass |
| L0.5 | Efficiency   | Learns on trivial real data        | Loss decreases + acc > 95% on digits, 5 seeds | 2 min | Fails → "digits-fail" |
| L1  | Learning     | Learns on standard vision          | MNIST 98%+, Fashion 90%+, 5 seeds, <2 min | 10 min | Fail either → excluded |
| L2  | Scaling      | Acc improves with compute          | Depth/width sweeps, power-law R² > 0.9 | 1 hr | α < 0.05 excluded |
| L3  | Parity       | Matches BP at matched compute      | FLOP/wall-time/mem-matched, CIs, effect sizes | 4 hr | Within 2% of best-BP → L4 |
| L4  | Scale-Up     | Matches BP on harder vision        | CIFAR-100, 5 seeds, conv arm | 8 hr | Within 2% → L5 |
| L5  | Superiority  | Beats BP in some regime            | Lower memory, faster, better OOD | Overnight | Publication |
| L6  | Transfer     | Generalizes across domains         | Vision→LM→RL→Graph, same algorithm | Multi-day | Publication |

**Every experiment outputs**: JSON with `task, model, seed, config, metrics, traces, artifacts` — machine-readable, no manual parsing.

---

## 2. Staircase Pipeline (Short → Long, Auto-Escalating)

```
┌────────────────────────────────────────────────────────────────────┐
│  TIER 0: SYNTHETIC (30 sec)                                        │
│  • Tasks: xor, spiral, circles (all 2D, 2-class, 1000 samples)     │
│  • Models: ALL registered models (no exclusions)                   │
│  • Check: forward+backward pass, loss decreases, no NaN            │
│  • Output: pass/fail per model + JSON trace                        │
│  → GATE: Fail = excluded from ALL higher tiers                     │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│  TIER 0.5: DIGITS (2 min)                                          │
│  • Task: sklearn digits (8×8, 64-dim, 10-class)                    │
│  • Models: L0-passing only                                         │
│  • Check: loss decreases + accuracy > 95%, 5 seeds                  │
│  • Fair protocol: identical MLP arch (num_layers=1, hidden=64)     │
│  • Output: accuracy, params, epoch_time, peak_mem per seed         │
│  → GATE: Fails to learn (loss flat or acc ≤ chance) → "digits-fail"│
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│  TIER 1: MNIST-FAMILY (10 min)                                     │
│  • Tasks: MNIST, Fashion-MNIST (both 28×28, 10-class, flattened)  │
│  • Models: L0.5-passing only                                       │
│  • Protocol: MLP sweep (depth∈{1,2,4}, width∈{64,128,256})        │
│  • Targets: MNIST ≥98%, Fashion ≥90%, 5 seeds, <2 min/task        │
│  • Output: per-task summary table + Pareto (acc/params/time)      │
│  → GATE: Fail either task = excluded from CIFAR tiers              │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│  TIER 2: SCALING SWEEP (1 hr)                                      │
│  • Tasks: MNIST, Fashion-MNIST, CIFAR-10 (flattened)              │
│  • Models: L1-passing only                                         │
│  • Sweep: depth∈{2,4,8,16,32}, width∈{64,128,256,512}            │
│  • Models: backprop, eqprop_mlp, neural_cube, deep_hebbian,       │
│            standard_fa, diff_target_prop, pepita, forward_forward │
│  • Output: power-law fits (acc ~ params^α, depth^β, time^γ) +     │
│            Pareto frontiers per task + JSON                        │
│  → GATE: α < 0.05 (weak scaling) excluded from L3                 │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│  TIER 3: PARITY — CIFAR-10 (4 hr)                                  │
│  • Matching protocol (THREE axes, all reported):                  │
│    – FLOP-matched: same forward+backward FLOPs/sample             │
│    – Wall-time-matched: same epoch-time budget                    │
│    – Memory-matched: same peak GPU memory                         │
│  • 5 seeds, bootstrap CIs (BCa), Cohen's d, Cliff's delta         │
│  • Tasks: CIFAR-10 (flattened MLP + Conv arms)                    │
│  • Output: publication-ready tables + Pareto plots + JSON         │
│  → GATE: Within 2% of best-in-class BP (pre-registered baseline) → L4│
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│  TIER 4: PARITY — CIFAR-100 (8 hr)                                 │
│  • Same matching protocol, harder task                            │
│  • Conv-arm mandatory (no flattened)                              │
│  → GATE: Within 2% of best-in-class BP → L5                         │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│  TIER 5: TRANSFER / SUPERIORITY (overnight+)                      │
│  • Regimes where BP fails: deep>50 layers, low-memory, OOD,       │
│    continual, neuromorphic                                        │
│  • Metrics: memory, convergence speed, OOD robustness, forgetting │
│  • Models: EquiTile variants, MEP, progressive_locality           │
└────────────────────────────────────────────────────────────────────┘
```

**Key principle**: Each tier's output feeds *directly* into the next tier's config via JSON artifact. No manual intervention.

---

## 3. Single Source of Truth: Campaign YAML

**One YAML per campaign** — no hardcoded values in `bioplausible/`.

```yaml
# experiments/campaign_parity_cifar10.yaml
meta:
  name: "parity_cifar10_mlp_conv"
  description: "BP vs bio-plausible parity on CIFAR-10 (MLP + Conv arms)"
  created: "2026-08-05"
  git_commit: "{{git_commit}}"

compute:
  device: "auto"                    # auto / cuda:0 / cpu
  max_parallel: 1                   # studies in parallel
  max_wall_hours: 4                 # hard wall-clock limit

search_space:                       # Single source of truth for bounds
  base:
    hidden_dim: [16, 32, 64, 128]
    num_layers: [1, 2, 4, 8, 16]
    lr: [1e-4, 1e-2]                # log-uniform
    batch_size: [64, 128, 256]
    optimizer: [adam, adamw, rmsprop]
    weight_decay: [1e-6, 1e-3]
    dropout: [0.0, 0.3]
    grad_clip: [0.5, 5.0]

arms:                               # Architecture-separated comparison groups
  mlp:
    input_dim: 3072                 # CIFAR-10 flattened
    num_classes: 10
    flatten: true
    max_params: 210_000             # Per-arm budget (conv needs more)
    models:
      - backprop_mlp
      - eqprop_mlp
      - neural_cube
      - deep_hebbian
      - three_factor_hebbian        # Passes learns-gate; protocol=layerwise
      - standard_fa
      - diff_target_prop
      - pepita
      - forward_forward
  conv:
    input_shape: [3, 32, 32]
    num_classes: 10
    flatten: false
    max_params: 2_000_000           # Conv layers need full spatial params
    models:
      - modern_conv_eqprop
      - conv_eqprop
      - backprop_conv

model_overrides:                    # Model-specific hyperparameters
  eqprop_mlp:
    beta: [0.01, 0.5]
    steps: [10, 20, 50]
    gradient_method: equilibrium     # Immutable: O(1) impl diff, never bptt
  neural_cube:
    cube_size: [3, 4, 5]
  diff_target_prop:
    alpha: [0.1, 0.9]
  standard_fa:
    feedback_init: [normal, orthogonal]
# Conditional constraints (evaluated at sample time)
constraints:
  - "estimate_param_count(config, model_name) <= arms[arm].max_params"

protocols:                          # Training protocol per model (sub-studies)
  default: end2end                  # All models run end2end by default
  overrides:
    three_factor_hebbian: layerwise   # Requires layerwise training
    spiking_stdp: spiking_surrogate   # Requires surrogate-gradient BPTT
    standard_fa: layerwise            # FA is layerwise by design
  # Each model×protocol combo spawns a sub-study; results reported side-by-side

tasks:
  - name: mnist
    epochs: 10
    input_dim: 784
    num_classes: 10
    flatten: true
  - name: fashion_mnist
    epochs: 10
    input_dim: 784
    num_classes: 10
    flatten: true
  - name: cifar10
    epochs: 50
    input_dim: 3072
    num_classes: 10
    flatten: true
  - name: cifar100
    epochs: 75
    input_dim: 3072
    num_classes: 100
    flatten: true

hpo:
  sampler: "nsga3"                   # True multi-objective (>=3 objectives)
  objectives:
    - accuracy                      # maximize
    - param_count                   # minimize
    - epoch_time_s                  # minimize
  n_trials: 200                     # Per model×arm×task
  n_startup_trials: 10
  n_seeds: 5                        # Statistical rigor
  prune_worse_than_pareto: true
  pareto:
    knee_point: true               # Auto-detect elbow in Pareto frontier
    epsilon: 0.01                  # Prune near-duplicate configs

resources:
  max_wall_hours: 4                 # Hard wall-clock limit per arm
  max_epoch_time_sec: 120           # Prune slow trials early
  early_stop_patience: 10

output:
  db: "results/parity_cifar10.db"
  artifacts_dir: "artifacts/parity_cifar10"
  log_level: "INFO"
  emit_every: 10                    # Interim portfolio

reproducibility:
  seed: 42
  capture_env: true                 # git, torch, cuda, deps hash
  artifact_hash: true               # Content-addressable checkpoints
```

**Rule**: If it's not in the YAML, it's not configurable. No `CONFIG` dicts in Python.

---

## 4. Core Abstractions (Pseudocode — AGENTS.md style)

### 4.1 SearchSpace (Declarative, Composable, Serializable)

```python
# bioplausible/hyperopt/search_space.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import optuna

class ParamDistribution(Protocol):
    def sample(self, trial: optuna.Trial, name: str) -> object: ...

@dataclass(frozen=True, slots=True)
class SearchSpace:
    base: dict[str, ParamDistribution]
    overrides: dict[str, dict[str, ParamDistribution]]
    constraints: list[str]                    # Python expressions over config

    def for_model(self, model_name: str, task: str) -> dict[str, ParamDistribution]:
        """Return Optuna-ready distributions with model/task overrides applied."""
        merged = {**self.base, **self.overrides.get(model_name, {})}
        # Apply task-specific constraints (e.g., input_dim affects max_params)
        return {k: v for k, v in merged.items() if self._check_constraints(k, task)}

    def estimate_param_count(self, config: dict, model_name: str) -> int:
        """Static analysis: param count from config WITHOUT training."""
        # Implemented per-model in registry metadata
        from bioplausible.zoo.registry import get_model_meta
        meta = get_model_meta(model_name)
        return meta.param_estimator(config)

    def _check_constraints(self, param: str, task: str) -> bool:
        # Evaluate constraint expressions with task context
        ...
```

### 4.2 Multi-Objective with Pre-Filtering

```python
# bioplausible/hyperopt/objectives.py
from dataclasses import dataclass
from typing import Protocol

class ParamEstimator(Protocol):
    def estimate(self, config: dict) -> int: ...

@dataclass(frozen=True, slots=True)
class MultiObjective:
    objectives: tuple[str, ...]
    directions: tuple[str, ...]          # "maximize" | "minimize"
    max_params: int
    estimator: ParamEstimator

    def filter_config(self, config: dict) -> bool:
        """Pre-training hard filter: reject configs exceeding max_params."""
        return self.estimator.estimate(config) <= self.max_params

    def evaluate(self, trial: optuna.Trial, metrics: dict) -> tuple[float, ...]:
        """Post-training evaluation for non-static objectives."""
        return tuple(metrics[obj] for obj in self.objectives)
```

**Eliminates**: Post-training `max_params` pruning waste (45s/trial).

### 4.3 Unified Logging (No Duplicate Params)

```python
# bioplausible/logging/experiment_logger.py
from dataclasses import dataclass
import json

@dataclass(frozen=True, slots=True)
class TrialStart:
    trial_id: int
    model: str
    task: str
    config: dict
    param_count: int
    seed: int

@dataclass(frozen=True, slots=True)
class TrialEnd:
    trial_id: int
    metrics: dict
    status: str
    wall_time_s: float

@dataclass(frozen=True, slots=True)
class EpochLog:
    trial_id: int
    epoch: int
    metrics: dict

class ExperimentLogger:
    def __init__(self, path: Path):
        self._fh = path.open("a")
    
    def log(self, event: TrialStart | TrialEnd | EpochLog) -> None:
        self._fh.write(json.dumps(event, default=vars) + "\n")
        self._fh.flush()
```

**Output**: One JSONL file per study — queryable, no duplicates, no parse hacks.

---

## 5. Fair Comparison Protocol (The Core Design Decision)

### 5.1 Three Matching Axes (All Reported, None Ignored)

| Axis | How Computed | Why It Matters |
|------|--------------|----------------|
| **FLOP-matched** | `forward_flops + backward_flops` per sample (or settle_steps × energy_flops for EqProp) | Algorithmic efficiency |
| **Wall-time-matched** | `epoch_time_s` cap — all models get same time budget | Practical efficiency |
| **Memory-matched** | `peak_gpu_memory_mb` cap — same memory footprint | Deployment efficiency |

**Rule**: A model only claims "parity" if it matches BP on **at least one axis** while not being worse on the others. All three are reported in every table.

### 5.2 Architecture Arms (No Unfair Conv vs MLP)

| Arm | Models | Input Format | Comparison Basis |
|-----|--------|--------------|------------------|
| **MLP** | backprop_mlp, eqprop_mlp, neural_cube, deep_hebbian, three_factor_hebbian, standard_fa, diff_target_prop, pepita, forward_forward | Flattened (3072-D for CIFAR-10) | Pure credit-assignment comparison |
| **Conv** | modern_conv_eqprop, conv_eqprop, backprop_conv | Spatial (3×32×32) | Spatial credit assignment |

**No cross-arm comparison**. Each arm has its own BP baseline.

### 5.3 Param Budget Enforcement (Pre-Training)

- Search space `constraints` evaluated at sample time via `SearchSpace.filter_config`
- Configs exceeding arm-specific `max_params` rejected **before** trial starts
- Per-arm budgets in YAML (`max_params` under each arm) — Conv arms get 2M, MLP arms get 210K

### 5.4 Protocol-Aware Fairness

Some bio-plausible models require non-standard training protocols (layerwise, continual, surrogate-gradient). The `protocols` field in YAML spawns a **sub-study per protocol variant**:

| Protocol | When Used | Models |
|----------|-----------|--------|
| `end2end` | Default | backprop, eqprop_mlp, neural_cube, deep_hebbian, diff_target_prop |
| `layerwise` | FA requires layerwise by design | standard_fa, three_factor_hebbian |
| `spiking_surrogate` | Spiking needs surrogate-gradient BPTT | spiking_stdp |
| `continual` | Task-switching protocol | (future) |

Results are reported side-by-side: `standard_fa (end2end, 4.2% acc)` vs `standard_fa (layerwise, 92.1% acc)` — the fair protocol wins, not the unfair comparison.

---

## 6. Open Questions from Phase 1.5 — Conclusive Answer Plan

| Question | How We Answer It Conclusively |
|----------|-------------------------------|
| **(a) Is eqprop's CIFAR-10 win real or a params confound?** | Tier 3: FLOP-matched + wall-time-matched + memory-matched. If eqprop_mlp at 388K params beats BP at 197K params *only* because it uses 2× compute, the parity table will show it. Same-compute comparison is the gate. |
| **(b) Do the 13 "non-learning baselines" learn under fair protocols?** | Tier 0.5 (digits) + Tier 1 (MNIST) with *layer-wise* / *continual* / *surrogate-gradient* protocol flags in YAML. Each family gets a `protocol` field: `end2end`, `layerwise`, `continual`, `spiking_surrogate`. Fair-protocol arm runs in parallel; results reported side-by-side. |
| **(c) Where does each model's compute scale break?** | Tier 2 scaling sweep: depth∈{2,4,8,16,32}, width∈{64,128,256,512}. Power-law fits (acc ~ params^α, depth^β, time^γ) with CIs. Break point = where α or β crosses zero (negative scaling). |
| **(d) Does implicit `equilibrium` gradient match BPTT at scale?** | Tier 3: EqProp models use `gradient_method=equilibrium` (immutable). Gradient cosine similarity between equilibrium implicit-diff and analytical BPTT computed on a reference batch, logged per trial. Conclusive: median cosine across 5 seeds at matched FLOPs. (BPTT used only as oracle, never for training.) |
| **(e) `three_factor_hebbian` loss-vs-accuracy disconnect?** | Tier 0.5 + Tier 1 with `track_classwise_metrics: true`. If loss drops but per-class accuracy stays at chance, it's optimizing the wrong objective. Ablation: swap 3-factor for standard Hebbian + readout. |

---

## 7. Campaign Artifacts (What We Actually Want)

```
results/
└── parity_cifar10/
    ├── campaign.yaml                 # Exact config used
    ├── campaign.db                   # Optuna studies (SQLite)
    ├── metrics.jsonl                 # All trial metrics (JSONL)
    ├── pareto_frontier.csv           # Multi-objective Pareto per arm
    ├── scaling_laws.json             # Power-law fits with CIs
    ├── parity_table.md               # Publication-ready markdown
    ├── parity_table.csv              # Machine-readable
    ├── gradient_equivalence.csv      # Cosine similarity at matched compute
    ├── pareto_plot.html              # Interactive Plotly
    └── artifacts/
        ├── trial_001/
        │   ├── config.json
        │   ├── metrics.json
        │   ├── checkpoints/
        │   └── traces/               # Energy, grad norms, settle curves
        └── ...
```

### 7.1 Parity Table (What Reviewers Want)

| Arm | Model | Task | Acc (mean±CI) | Params (median) | Time/epoch (median) | FLOPs/sample | vs BP (Cohen's d) | Match Axis |
|-----|-------|------|---------------|-----------------|---------------------|--------------|-------------------|------------|
| MLP | backprop | CIFAR-10 | 0.52 [0.50, 0.54] | 100K | 3.2s | 2.1M | — (ref: best-in-class) | all |
| MLP | eqprop_mlp | CIFAR-10 | 0.48 [0.46, 0.50] | 98K | 4.1s | 2.1M | -1.2 | FLOP |
| MLP | neural_cube | CIFAR-10 | 0.47 [0.45, 0.49] | 84K | 3.8s | 2.0M | -0.5 | FLOP |
| Conv | backprop_conv | CIFAR-10 | 0.68 [0.66, 0.70] | 1.2M | 8.5s | 18M | — (ref: best-in-class) | all |
| Conv | modern_conv_eqprop | CIFAR-10 | 0.66 [0.64, 0.68] | 1.1M | 9.2s | 18M | -0.3 | FLOP |

---

## 8. Implementation Sequence (No Big Bang)

### Phase A: Foundation (Week 1-2)
- [ ] `SearchSpace` class replacing `HyperparameterMetamodel`
- [ ] YAML config schema with Pydantic v2 validation
- [ ] `ParamEstimator` per model (static param counting)
- [ ] `biopl-run` CLI with dry-run, resume, override

### Phase B: Logging & Runner (Week 2-3)
- [ ] `ExperimentLogger` (JSONL, no duplicates)
- [ ] `MultiObjective` with pre-filtering
- [ ] TIER 0 smoke test gate (synthetic tasks)
- [ ] TIER 0.5 digits gate (loss-decrease + acc > 95% threshold)
- [ ] Protocols field: end2end/layerwise/spiking_surrogate sub-studies

### Phase C: Tiered Experiments (Week 3-4)
- [ ] TIER 1: MNIST + Fashion-MNIST validation
- [ ] TIER 2: Scaling sweep (depth/width, power-law fits)
- [ ] TIER 3: Parity suite (FLOP/wall-time/memory matched)
- [ ] Auto-generated summary artifacts (tables, plots, JSON)

### Phase D: Transfer / Superiority (Week 4-5)
- [ ] `biopl-parity` CLI for matched-compute runs
- [ ] FLOP-matching logic (analytical + empirical)
- [ ] Bootstrap CIs (BCa), effect sizes (Cohen's d, Cliff's delta)
- [ ] Conv-arm health gate: verify `conv_eqprop` passes TIER 0-2 before parity
- [ ] CIFAR-100 parity (conv arm mandatory, ~2× CIFAR-10 budget)

---

## 9. MLP-First Strategy (Conv Later)

**Decision**: Keep MLP-only until:
1. All MLP models pass TIER 0-2 consistently
2. `eqprop_mlp` or `neural_cube` within 2% of BP on CIFAR-10 (flattened) at matched FLOPs
3. Gradient equivalence verified for all EqProp variants (cosine > 0.95)
4. `modern_conv_eqprop` implementation audited for efficiency

**Rationale**: Conv adds 10× complexity (spatial dims, padding, pooling, batch norm interactions). If MLP isn't solid, Conv will inherit and amplify bugs.

---

## 10. Configurability Checklist (No More Debugging Hell)

| Aspect | Current Pain | FIX2a State |
|--------|--------------|-------------|
| Change search space | Edit Python, find metamodel, debug | Edit YAML, `biopl-run --dry-run` |
| Change tasks | Edit Python, find ARCH_GROUPS | Edit YAML `tasks:` list |
| Change max_params | Edit Python, find CONST | Edit YAML `arms.mlp.max_params` / `arms.conv.max_params` |
| Add model | Edit Python, register, find exclusions | Add to YAML `arms.<arm>.models` + `model_overrides` |
| Change seeds | Edit Python | Edit YAML `hpo.n_seeds` |
| Add task | Edit Python, find task registry | Edit YAML `tasks:` list |
| Debug trial | Search logs, grep params | `jq '.config' metrics.jsonl` |
| Resume crashed run | Hope DB intact, guess state | `biopl-run --resume` |
| Compare runs | Manual CSV merge | `biopl-compare run1.db run2.db` |

---

## 11. Success Criteria

| Metric | Target |
|--------|--------|
| Time from "new idea" to first result | < 30 sec (TIER 0) |
| Time from "new model" to parity verdict | < 4 hr (TIER 3) |
| Config change → validated experiment | < 15 sec (dry-run) |
| Duplicate log entries | 0 |
| Manual config edits in Python | 0 |
| Publication-ready output per campaign | 1 command |

---

## 12. Comparison: FIX2a vs FIX2.md

| Dimension | FIX2.md (Current) | FIX2a.md (This) |
|-----------|-------------------|-----------------|
| **Staircase granularity** | 5 tiers (L0-L4) | 7 tiers (L0, L0.5, L1-L5) with synthetic→digits→MNIST→CIFAR-10→CIFAR-100 |
| **Economic efficiency** | Implicit (Tier 1 = MNIST) | Explicit: digits = cheap triage gate (loss-decrease + acc>95%); models failing logged as "digits-fail", excluded early |
| **Fair comparison protocol** | "FLOP-matched, wall-time-matched, memory-matched" listed | **Three axes all reported**, **arm-separated** (MLP vs Conv), per-arm `max_params` budgets, `protocols` field for fair-protocol arms |
| **Open questions answered** | Not addressed | 5 Phase 1.5 questions with conclusive experiment design |
| **Gradient equivalence** | Mentioned in TIER 1 | TIER 3 metric: cosine(equilibrium, BPTT) — BPTT is oracle only, `gradient_method=equilibrium` immutable |
| **Non-learning baselines** | Excluded silently | Protocol-aware: `layerwise`/`continual`/`spiking_surrogate` arms in YAML, run in parallel, reported side-by-side |
| **Scaling laws** | TIER 2 output | TIER 2 + TIER 3: power-law fits (R²>0.9) with CIs, break-point detection (α, β crossings) |
| **Config system** | YAML sketch | Full schema: `arms`, `protocols`, `model_overrides` (immutable `gradient_method`), `constraints`, NSGA-III + knee detection |
| **Logging** | JSONL sketch | Typed events (TrialStart/TrialEnd/EpochLog), single JSONL |
| **CLI** | `biopl-run` sketch | Dry-run, resume, override, compare subcommands |

**Verdict**: FIX2a is strictly more specific on the *how* (matching protocol, staircase, open questions), more honest about fair comparison (three axes, arm separation), and directly addresses the Phase 1.5 unknowns. FIX2.md remains a valid high-level framework; FIX2a is the implementable specification.

---

## 13. Immediate Next Steps (This Week)

### Phase A (DONE — verified working)
1. **~~Create `SearchSpace` class~~** — `search_space.py`: `ParamDistribution` Protocol, `FloatRange`/`IntRange`/`Choice`, `parse_distribution`, `sample_feasible` with constraint evaluation
2. **~~Define YAML schema~~** — `schema.py`: Pydantic v2 models (`Campaign`, `Arm`, `Task`, `HPO`, `Compute`, `Resources`, `Output`, `Reproducibility`, `Protocols`), `validate_yaml`/`load_campaign`
3. **~~Implement `ParamEstimator`~~** — `param_estimator.py`: `InstantiateEstimator` (constructs model, sums `numel()`), `build_model_kwargs` (signature-filtered), `bound_estimator`, `estimate_param_count`
4. **~~Build `biopl-run` CLI~~** — `cli.py`: `validate`, `dry-run`, `gates`, `run` subcommands with `match`/`case` dispatch
5. **~~Write TIER 0 smoke test~~** — `tiers.py`: `run_tier0` over xor/spiral/circles, fixed 2D/2-class geometry
6. **~~Write TIER 0.5 digits test~~** — `tiers.py`: `run_tier05` over digits, fixed 64/10 geometry, `digits-fail` verdict

### Phase B (PARTIALLY DONE — broken stub, needs rewrite)
7. **`ExperimentLogger`** — `logger.py`: DONE. Typed events (`TrialStart`/`TrialEnd`/`Epoch`/`GateOutcome`), JSONL append-only, `TypeIs` narrowing, Google-style docstrings
8. **`CampaignExecutor`** — `executor.py`: **STUB — DO NOT USE**. See §14 for the audit and §15 for the rewrite plan.

### Phase C (NOT STARTED)
9. TIER 1: MNIST + Fashion-MNIST validation
10. TIER 2: Scaling sweep (depth/width, power-law fits)
11. TIER 3: Parity suite (FLOP/wall-time/memory matched)
12. Auto-generated summary artifacts (tables, plots, JSON)

---

## 14. Executor Audit (Why `executor.py` Is a Broken Stub)

`executor.py` was written hastily and contains **11 shortcuts/hardcodes/anti-patterns** that violate the FIX2a principle: *"If it's not in the YAML, it's not configurable. No CONFIG dicts in Python."*

### 14.1 Hardcoded Values (Violate "No CONFIG dicts in Python")

| Line | Code | Problem | Fix |
|------|------|---------|-----|
| 104 | `getattr(self.campaign.hpo, "min_accuracy", 0.95)` | `min_accuracy` missing from `HPO` schema; `getattr` fallback is a code smell | Add `min_accuracy: float = Field(0.95, ge=0.0, le=1.0)` to `HPO` |
| 212 | `batch_size=full_config.get("batch_size", 128)` | Hardcoded default 128 | Use `Compute.batch_size` or `Task.batch_size` from schema |
| 213 | `num_workers=0` | Hardcoded 0 | Use `Compute.num_workers` from schema |
| 216-218 | `getattr(self.campaign.compute, "track_energy", False)` | `track_energy`/`track_flops`/`track_memory` missing from `Compute` schema | Add these fields to `Compute` with defaults |
| 211 | `epochs=self.campaign.tasks[0].epochs if self.campaign.tasks else 10` | Hardcoded fallback 10 | Require at least one task in schema (already enforced by `tasks` being a list, but executor should error explicitly if empty) |

### 14.2 Architectural Problems

| Line | Code | Problem | Fix |
|------|------|---------|-----|
| 143 | `key.replace('/', '_')` | Filename collision: `"a/b"` → `"a_b"` same as `"a_b"` | Use `Path` components: `self.storage_base / arm_name / f"{model_name}.db"` |
| 199 | `param_count = sum(p.numel() for p in model_cls(**model_kwargs).parameters())` | Constructs model in executor instead of using `estimate_param_count` | Use `estimate_param_count(model_name, full_config, ...)` from `param_estimator.py` |
| 155 | `_OPTIMIZER_KWARGS: frozenset[str]` defined on class body | Duplicates module-level constant; class-level constant is dead code | Remove class-level copy, keep module-level only |
| 244 | `return f"classification_{arm_input_dim}_{arm_output_dim}"` | Generates a task name CoreTrainer won't recognize — will crash at data loading | Require `tasks` in campaign YAML (schema should enforce min 1 task for HPO campaigns) |
| 91-95 | `gate_result` property lazily runs gates | Fragile for resume: can't skip gates on resume, can't inspect gate state | Make gates an explicit `run_gates()` step, store result, allow skip via `--resume` |
| 35 | `_OPTIMIZER_KWARGS` module constant | Should live with the search space or param estimator (where config keys are known) | Move to `param_estimator.py` or `search_space.py` as a known-keys frozenset |

### 14.3 What's Missing Entirely

- **No `--resume` support**: The Optuna `load_if_exists=True` creates the study, but there's no mechanism to load prior gate results, skip already-completed trials, or resume from a crashed run.
- **No per-task HPO**: The executor runs one study per `arm × model`, but FIX2a §2 specifies per `arm × model × task` (e.g., MNIST and Fashion-MNIST are separate tasks with separate sweeps).
- **No multi-seed support**: `HPO.n_seeds` exists in the schema but the executor runs one trial per config, not `n_seeds` trials with different seeds aggregated via bootstrap CIs.
- **No early stopping**: `Resources.early_stop_patience` and `Resources.max_epoch_time_sec` exist in the schema but the executor ignores them.
- **No wall-clock budget**: `Resources.max_wall_hours` exists but is never checked.
- **No Pareto pruning**: `HPO.prune_worse_than_pareto` exists but is never applied.
- **No analysis output**: No `pareto_frontier.csv`, `scaling_laws.json`, `parity_table.md` generation.
- **No checkpoint/resume for CoreTrainer**: `CoreTrainer.load_checkpoint` exists but the executor never calls it.
- **No error aggregation**: Trial failures are logged but not aggregated into a failure summary.

---

## 15. Executor Rewrite Plan (Do This, Not That)

### 15.1 Schema Gaps to Fill First

```python
# schema.py — add to Compute
class Compute(BaseModel):
    device: str = "auto"
    max_parallel: int = Field(1, ge=1)
    max_wall_hours: float | None = Field(None, gt=0)
    batch_size: int = Field(128, ge=1)           # NEW
    num_workers: int = Field(0, ge=0)            # NEW
    track_energy: bool = False                    # NEW
    track_flops: bool = False                     # NEW
    track_memory: bool = False                    # NEW
    precision: str = "32-true"                     # NEW: "16-mixed", "bf16-mixed", "32-true"
    use_compile: bool = False                     # NEW

# schema.py — add to HPO
class HPO(BaseModel):
    # ... existing fields ...
    min_accuracy: float = Field(0.95, ge=0.0, le=1.0)  # NEW: TIER 0.5 gate threshold
```

### 15.2 Task Resolution (No Hardcoding)

The executor must use the `Task` objects from `campaign.tasks`, matching by geometry:

```python
def _resolve_task(self, arm_name: str) -> Task:
    arm = self.campaign.arms[arm_name]
    arm_input_dim = self.campaign.arm_input_dim(arm_name)
    arm_output_dim = self.campaign.arm_output_dim(arm_name)

    for task in self.campaign.tasks:
        task_input = task.input_dim or arm_input_dim
        task_classes = task.num_classes or arm_output_dim
        if task_input == arm_input_dim and task_classes == arm_output_dim:
            return task

    # Schema should enforce: HPO campaigns require >=1 matching task
    raise CampaignError(
        f"No task in campaign matches arm {arm_name!r} geometry "
        f"({arm_input_dim} → {arm_output_dim})"
    )
```

**Key**: `Task.name` is passed to `CoreTrainer` which maps it to `get_vision_dataset(name=...)`. The supported names are: `mnist`, `fashion_mnist`, `cifar10`, `cifar100`, `kmnist`, `svhn`, `digits`, `xor`, `spiral`, `circles`. Any task name not in this list will crash — the schema should validate task names against this set, or CoreTrainer should expose a task registry.

### 15.3 Executor Architecture (Clean Rewrite)

```
CampaignExecutor
├── run(n_trials) → gates → HPO per (arm, model, task) → studies
├── run_gates() → run_gates() from runner.py → CampaignResult
├── get_survivors() → list[(arm, model)] from gate results
├── run_hpo(arm, model, task, n_trials) → Optuna Study
│   ├── _build_trainer_config(trial, arm, model, task) → TrainerConfig
│   ├── _objective(trial) → (accuracy, param_count, epoch_time_s)
│   └── _log_trial(trial, status, metrics) → JSONL events
├── resume() → load existing study, skip completed trials
└── analyze() → Pareto frontier, scaling laws, parity table

Key principles:
1. All config comes from YAML schema, no hardcoded defaults in executor
2. One study per (arm, model, task) — not per (arm, model)
3. n_seeds trials per config, aggregated via bootstrap CIs
4. Early stopping via Resources.early_stop_patience
5. Wall-clock budget via Resources.max_wall_hours
6. Pareto pruning via HPO.prune_worse_than_pareto
7. Checkpoint/resume via Optuna SQLite storage + CoreTrainer.load_checkpoint
8. Error aggregation: collect all TrialEnd(status="failed") into failure summary
```

### 15.4 CLI Changes

The `run` command must be separated from `gates`:

```
biopl-run validate  --config <yaml>           # schema validation
biopl-run dry-run   --config <yaml>           # plan preview
biopl-run gates     --config <yaml>           # TIER 0/0.5 only
biopl-run run       --config <yaml>           # full campaign: gates + HPO
    --resume           # skip gates, load existing study
    --trials N         # override hpo.n_trials
    --arm NAME         # run only one arm
    --model NAME       # run only one model
    --task NAME        # run only one task
    --device cpu|cuda  # override compute.device
```

### 15.5 What "Runnable" Means (No More False Claims)

A campaign is **runnable** when ALL of these hold:
1. `biopl-run validate` passes on the YAML
2. `biopl-run dry-run` shows the resolved plan
3. `biopl-run gates --tier all` completes and writes `gates.jsonl`
4. `biopl-run run` completes at least one full trial per surviving model and writes `trials.jsonl` with `TrialStart`/`Epoch`/`TrialEnd` events
5. `biopl-run run --resume` skips completed trials and continues from where it stopped

Until all 5 hold, the campaign framework is **not runnable**. Do not claim otherwise.

---

## 16. Additional Opportunities (Beyond FIX2a §1-§13)

### 16.1 Task Name Registry (Prevents Runtime Crashes)

`CoreTrainer` accepts a `task: str` that maps to `get_vision_dataset(name=task)`. The supported names are hardcoded in `data/vision.py`. A campaign YAML with a typo (`mist` instead of `mnist`) crashes at training time, not validation time.

**Fix**: Add a `TaskRegistry` in `data/vision.py` (or `core/registry.py`) that maps task names to dataset loaders. The `Task` schema validates `name` against this registry at YAML load time.

### 16.2 Constraint Expression Safety (§4.1)

`SearchSpace._constraints_hold` uses `eval()` with `__builtins__: {}`. This is safe for researcher-authored YAML but:
- No timeout (a malicious/infinite loop hangs forever)
- No import of `math`/`statistics` (useful for complex constraints)
- Error messages don't identify which constraint failed by index

**Enhancement**: Replace `eval()` with `asteval` or a restricted expression evaluator. Add `math`/`statistics` to the namespace. Include the constraint index in error messages.

### 16.3 Gradient Equivalence Testing (RESEARCH.md §5.2, FIX2a §6d)

TIER 3 requires cosine similarity between equilibrium implicit-diff and BPTT gradients. This is a separate concern from the campaign executor — it's a **validation track** that runs alongside HPO, not within it.

**Implementation**: `bioplausible/validation/gradient_equivalence.py` with a `check_gradient_equivalence(model_name, config, task) -> float` function that:
1. Runs one forward pass with equilibrium gradient
2. Runs one forward pass with BPTT (oracle)
3. Returns cosine similarity of the two gradient vectors
4. Logs as a `GradientEquivalence` event in the JSONL stream

### 16.4 Statistical Aggregation (RESEARCH.md §5.3)

`n_seeds` trials per config need aggregation before reporting:
- Bootstrap CIs (BCa) for accuracy
- Cohen's d, Cliff's delta vs backprop baseline
- Benjamini-Hochberg correction for multiple comparisons

**Implementation**: `bioplausible/validation/statistics.py` with:
- `bootstrap_ci(values, confidence=0.95) -> tuple[float, float]`
- `cohens_d(treatment, control) -> float`
- `cliffs_delta(treatment, control) -> float`
- `benjamini_hochberg(p_values, alpha=0.05) -> list[bool]`

The executor calls these after all `n_seeds` trials for a config complete.

### 16.5 Campaign State Serialization (RESEARCH.md §4.3)

For AutoScientist resume, the campaign needs:
- SQLite storage of Optuna studies (already done via `sqlite:///`)
- YAML serialization of gate results (write `gates.jsonl` → done)
- Checkpoint paths in `TrialStart.config` (done)
- A `campaign_state.json` with: `{ completed_trials: int, total_trials: int, gate_survivors: [...], last_updated: timestamp }`

### 16.6 Failure Manifesto (RESEARCH.md §5.4)

Every `TrialEnd(status="failed")` should contribute to a structured failure summary:
- What was tried (config, model, task)
- Why it should work (hypothesis)
- Why it failed (error message, traceback excerpt)
- Search space explored (hyperparameter ranges)
- Partial successes (what *did* work)

**Implementation**: `bioplausible/analysis/failure_manifesto.py` that post-processes `trials.jsonl` and generates `failure_manifesto.md`.

---

## 17. Implementation Order (Correct Dependency Sequence)

```
1. Schema gaps (§15.1)          — 30 min
   └─ Add Compute.batch_size, num_workers, track_*, precision, use_compile
   └─ Add HPO.min_accuracy
   └─ Validate Task.name against data registry (§16.1)

2. Task name registry (§16.1)   — 1 hr
   └─ data/vision.py: expose SUPPORTED_TASKS frozenset
   └─ schema.py: Task.name must be in SUPPORTED_TASKS

3. Delete executor.py stub      — 5 min
   └─ It's broken; rewrite from scratch

4. Rewrite executor.py (§15.3)  — 4-6 hr
   └─ Clean architecture: gates → HPO per (arm, model, task) → studies
   └─ No hardcoded defaults; all from schema
   └─ Use estimate_param_count, not manual construction
   └─ One study per (arm, model, task)
   └─ n_seeds per config with bootstrap CI aggregation
   └─ Early stopping, wall-clock budget, Pareto pruning
   └─ Resume via --resume flag + Optuna load_if_exists

5. CLI run command (§15.4)      — 1 hr
   └─ Separate run from gates
   └─ Add --resume, --trials, --arm, --model, --task flags

6. Statistics module (§16.4)    — 2 hr
   └─ Bootstrap CIs, effect sizes, multiple comparison correction

7. Gradient equivalence (§16.3) — 2 hr
   └<arg_value> Separate from executor; validation track

8. Analysis output (§15.3)      — 2 hr
   └─ Pareto frontier CSV, scaling law JSON, parity table MD
   └─ Failure manifesto (§16.6)

9. End-to-end verification (§15.5) — 1 hr
   └─ All 5 "runnable" criteria met
   └─ Overnight run on example campaign
```

**Total**: ~14-16 hours of focused implementation.

---

## 18. Lesson Learned: Do Not Claim "Runnable" Prematurely

The campaign framework was declared "runnable" when:
- TIER 0/0.5 gates worked (true)
- But the executor was a broken stub with 11 hardcodes (false)
- No full trial had ever been executed end-to-end (false)
- No resume had been tested (false)
- No analysis output had been generated (false)

**Rule**: "Runnable" means ALL 5 criteria in §15.5 hold. Anything less is a prototype, not a deliverable. State this explicitly when reporting status.

---

*This plan transforms "running experiments" into "running a discovery engine" — the core thesis of RESEARCH.md.*