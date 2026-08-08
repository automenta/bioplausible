# EXPERIMENT_PLAN6.md — Revised Action Plan

## Root Cause

`BioModel.__init__` reads `learning_rate`/`beta`/`max_steps` from `**kwargs` (defaulting to 0.001/0.2/30) instead of `self.config.learning_rate`/`self.config.beta`/`self.config.max_steps`. The constructor system (`build_model_config` + `construct_model`) **correctly builds the config with sampled values** — models just ignore it.

This affects ALL bio families (eqprop, hebbian, spiking, target_prop, fa variants, forward_forward). Only FA and Pepita work because they hardcode working defaults.

## Actions (Minimal, No Deletion)

### 1. Fix Hyperparam Routing (3 lines)
**File:** `bioplausible/core/model.py` → `BioModel.__init__`

After `self.config = config`, add:
```python
self.learning_rate = self.config.learning_rate
self.beta = self.config.beta
self.max_steps = self.config.max_steps
```

That's it. All models now read hyperparams from their config.

### 2. Eqprop Family: One Shared Energy-Contrastive Engine
**New file:** `bioplausible/zoo/models/eqprop/_energy.py` — `EquilibriumMLP(EqPropModel)`
- Single-hidden recurrent MLP: `h ← σ(W_in x + W_rec h)`
- Settle: `settle_single_state` (contractive forward-only, spectral-norm freeze, early convergence) — ~5 ms/step
- Train step: Scellier & Bengio energy contrastive (free settle → 5-step nudged with `−β·v·W_out` → `Δw = (∇E_nudged − ∇E_free)/β` + `W_out` from loss). **No optimizer. Self-contained.**
- Six thin registered subclasses (keep registrations):
  - `StandardEqProp` (variant="plain")
  - `DirectedEP` (variant="feedback")
  - `FiniteNudgeEP` (variant="plain")
  - `LazyEqProp` (variant="plain")
  - `MomentumEquilibrium` (variant="momentum")
  - `SparseEquilibrium` (variant="sparse")

**Replace 6 files** with thin re-exports from `_energy.py`:
- `standard_eqprop.py`, `deep_ep.py`, `finite_nudge_ep.py`, `lazy_eqprop.py`, `mom_eq.py`, `sparse_eq.py`

Keep working eqprop models unchanged: `graph_eqprop`, `holomorphic_ep`, `eqprop_mlp`, `conv_eqprop`, `modern_conv_eqprop`, `neural_cube`, `eqprop_diffusion`.

### 3. Fix NeuralCube Budget Matcher
**File:** `scripts/broad_sweep.py` → `_match_param_budget`
Add `cube_size` to width-axis search (like `hidden_channels` for conv).

### 4. Sweep Integration
**File:** `scripts/broad_sweep.py`
- Remove `_SETTLE_STEP_CAPS` (engine is fast).
- Update `_eqprop_gradient_method`: all eqprop models now have working `train_step` → all get `gradient_method="equilibrium"` (energy-contrastive).

### 5. Remove `lr` Alias
**File:** `bioplausible/core/construction.py`
- Delete `_KNOB_ALIASES["lr"] = "learning_rate"` and its handling in `_normalize`. Standardize on `learning_rate`.

### 6. Update Tests
| Test | Change |
|------|--------|
| `test_config_knobs.py` | Remove `lr` alias test; assert `learning_rate` from config directly |
| `test_broad_sweep.py` | Replace eqprop gradient_method tests with energy-contrastive expectations |
| `test_settle_speed.py` | Test `train_step` learns; remove `forward(beta=...)` checks |
| `test_eqprop_learns.py` | Ensure it tests the new energy-contrastive path |

## Files to Modify (Only These)

| File | Action |
|------|--------|
| `bioplausible/core/model.py` | Add 3 lines in `BioModel.__init__` |
| `bioplausible/zoo/models/eqprop/_energy.py` | **NEW** — shared engine + 6 subclasses |
| `bioplausible/zoo/models/eqprop/standard_eqprop.py` | Replace → re-export |
| `bioplausible/zoo/models/eqprop/deep_ep.py` | Replace → re-export |
| `bioplausible/zoo/models/eqprop/finite_nudge_ep.py` | Replace → re-export |
| `bioplausible/zoo/models/eqprop/lazy_eqprop.py` | Replace → re-export |
| `bioplausible/zoo/models/eqprop/mom_eq.py` | Replace → re-export |
| `bioplausible/zoo/models/eqprop/sparse_eq.py` | Replace → re-export |
| `bioplausible/zoo/models/eqprop/__init__.py` | No change (imports work) |
| `scripts/broad_sweep.py` | Fix budget matcher (cube_size); remove settle caps; update eqprop gradient method |
| `bioplausible/core/construction.py` | Remove `lr` alias |
| 6 test files | Update assertions to match new convention |

## Verification (Run in Order)

```bash
# 1. Constructor sanity
uv run pytest tests/unit/experiment/test_config_knobs.py -q --no-cov

# 2. Eqprop engine learns + fast (GPU)
uv run pytest tests/unit/experiment/test_eqprop_learns.py -q --no-cov

# 3. Model surface contracts
uv run pytest tests/unit/models/test_eqprop_models.py -q --no-cov

# 4. Full unit regression
uv run pytest tests/unit/ -q --no-cov

# 5. GPU smoke: one probe per fundamental eqprop model, 2 epochs, 32k budget
```

## What the Sweep Will Then Surface

With hyperparams correctly routed and eqprop fast/learning, the sweep across `--families fa,hebbian,forward_only,predictive_coding,spiking,target_prop,eqprop` will produce honest liveness/performance data for every bio family — the first time the framework actually tests them with their sampled hyperparams.
<tool_call>
<function=write>
<parameter=content>
# EXPERIMENT_PLAN6.md — Revised Action Plan

## Root Cause

`BioModel.__init__` reads `learning_rate`/`beta`/`max_steps` from `**kwargs` (defaulting to 0.001/0.2/30) instead of `self.config.learning_rate`/`self.config.beta`/`self.config.max_steps`. The constructor system (`build_model_config` + `construct_model`) **correctly builds the config with sampled values** — models just ignore it.

This affects ALL bio families (eqprop, hebbian, spiking, target_prop, fa variants, forward_forward). Only FA and Pepita work because they hardcode working defaults.

## Actions (Minimal, No Deletion)

### 1. Fix Hyperparam Routing (3 lines)
**File:** `bioplausible/core/model.py` → `BioModel.__init__`

After `self.config = config`, add:
```python
self.learning_rate = self.config.learning_rate
self.beta = self.config.beta
self.max_steps = self.config.max_steps
```

That's it. All models now read hyperparams from their config.

### 2. Eqprop Family: One Shared Energy-Contrastive Engine
**New file:** `bioplausible/zoo/models/eqprop/_energy.py` — `EquilibriumMLP(EqPropModel)`
- Single-hidden recurrent MLP: `h ← σ(W_in x + W_rec h)`
- Settle: `settle_single_state` (contractive forward-only, spectral-norm freeze, early convergence) — ~5 ms/step
- Train step: Scellier & Bengio energy contrastive (free settle → 5-step nudged with `−β·v·W_out` → `Δw = (∇E_nudged − ∇E_free)/β` + `W_out` from loss). **No optimizer. Self-contained.**
- Six thin registered subclasses (keep registrations):
  - `StandardEqProp` (variant="plain")
  - `DirectedEP` (variant="feedback")
  - `FiniteNudgeEP` (variant="plain")
  - `LazyEqProp` (variant="plain")
  - `MomentumEquilibrium` (variant="momentum")
  - `SparseEquilibrium` (variant="sparse")

**Replace 6 files** with thin re-exports from `_energy.py`:
- `standard_eqprop.py`, `deep_ep.py`, `finite_nudge_ep.py`, `lazy_eqprop.py`, `mom_eq.py`, `sparse_eq.py`

Keep working eqprop models unchanged: `graph_eqprop`, `holomorphic_ep`, `eqprop_mlp`, `conv_eqprop`, `modern_conv_eqprop`, `neural_cube`, `eqprop_diffusion`.

### 3. Fix NeuralCube Budget Matcher
**File:** `scripts/broad_sweep.py` → `_match_param_budget`
Add `cube_size` to width-axis search (like `hidden_channels` for conv).

### 4. Sweep Integration
**File:** `scripts/broad_sweep.py`
- Remove `_SETTLE_STEP_CAPS` (engine is fast).
- Update `_eqprop_gradient_method`: all eqprop models now have working `train_step` → all get `gradient_method="equilibrium"` (energy-contrastive).

### 5. Remove `lr` Alias
**File:** `bioplausible/core/construction.py`
- Delete `_KNOB_ALIASES["lr"] = "learning_rate"` and its handling in `_normalize`. Standardize on `learning_rate`.

### 6. Update Tests
| Test | Change |
|------|--------|
| `test_config_knobs.py` | Remove `lr` alias test; assert `learning_rate` from config directly |
| `test_broad_sweep.py` | Replace eqprop gradient_method tests with energy-contrastive expectations |
| `test_settle_speed.py` | Test `train_step` learns; remove `forward(beta=...)` checks |
| `test_eqprop_learns.py` | Ensure it tests the new energy-contrastive path |

## Files to Modify (Only These)

| File | Action |
|------|--------|
| `bioplausible/core/model.py` | Add 3 lines in `BioModel.__init__` |
| `bioplausible/zoo/models/eqprop/_energy.py` | **NEW** — shared engine + 6 subclasses |
| `bioplausible/zoo/models/eqprop/standard_eqprop.py` | Replace → re-export |
| `bioplausible/zoo/models/eqprop/deep_ep.py` | Replace → re-export |
| `bioplausible/zoo/models/eqprop/finite_nudge_ep.py` | Replace → re-export |
| `bioplausible/zoo/models/eqprop/lazy_eqprop.py` | Replace → re-export |
| `bioplausible/zoo/models/eqprop/mom_eq.py` | Replace → re-export |
| `bioplausible/zoo/models/eqprop/sparse_eq.py` | Replace → re-export |
| `bioplausible/zoo/models/eqprop/__init__.py` | No change (imports work) |
| `scripts/broad_sweep.py` | Fix budget matcher (cube_size); remove settle caps; update eqprop gradient method |
| `bioplausible/core/construction.py` | Remove `lr` alias |
| 6 test files | Update assertions to match new convention |

## Verification (Run in Order)

```bash
# 1. Constructor sanity
uv run pytest tests/unit/experiment/test_config_knobs.py -q --no-cov

# 2. Eqprop engine learns + fast (GPU)
uv run pytest tests/unit/experiment/test_eqprop_learns.py -q --no-cov

# 3. Model surface contracts
uv run pytest tests/unit/models/test_eqprop_models.py -q --no-cov

# 4. Full unit regression
uv run pytest tests/unit/ -q --no-cov

# 5. GPU smoke: one probe per fundamental eqprop model, 2 epochs, 32k budget
```

## What the Sweep Will Then Surface

With hyperparams correctly routed and eqprop fast/learning, the sweep across `--families fa,hebbian,forward_only,predictive_coding,spiking,target_prop,eqprop` will produce honest liveness/performance data for every bio family — the first time the framework actually tests them with their sampled hyperparams.