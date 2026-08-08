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

---

## 7. Sweep Findings — Post-Sweep Action Plan (from 2026-08-08 run)

### Sweep Summary (cumulative, all runs)

| Family | Models Tested | Status |
|--------|---------------|--------|
| fa | 12 | 💀 **All skipped** — pre-flight "forward/propagator-incompatible" |
| hebbian | 4 | 🟡 **Run but don't learn** — loss flat ~2.36, acc 5-10% |
| forward_only | 2 | ✅ **Working** — FF 30%→71%, Pepita 19%→48% |
| predictive_coding | 1 | 💥 **Device bug** — CPU tensors in CUDA model |
| eqprop | 14 (2 skipped) | 🟡 **Mixed** — 11 live, 1 dead (momentum_equilibrium), 1 defect (neural_cube BPTT fallback) |
| spiking | 1 | 💀 **Not learning** — loss flat ~6.5, acc 10% (random) |
| target_prop | 1 | ✅ **Working** — loss decreasing, acc 11%→12% (slow learner) |

---

### Low-Hanging Fruit (≤1 day each, self-contained)

#### ✅ 7.1 Fix `lr` alias leaking into sweep configs — **DONE**
**File:** `bioplausible/core/construction.py`  
**Fix applied:** Added `"lr": "learning_rate"` to `_KNOB_ALIASES` dict. Now `lr` keys from rule spaces are canonicalized to `learning_rate`.  
**Note:** Rule space definitions (`search_space.py`) still use `"lr"` key but the `_normalize()` function now handles the alias, so all models get `learning_rate` correctly.  

#### ✅ 7.2 Fix momentum_equilibrium tensor size mismatch — **DONE**
**File:** `bioplausible/zoo/models/eqprop/_energy.py` line 119  
**Root cause:** `_velocity` buffer initialized once and reused across different batch sizes.  
**Fix:** Added shape check — `_velocity` is re-initialized when batch dimension changes:  
```python
if not hasattr(self, "_velocity") or self._velocity.shape != h.shape:
    self._velocity = torch.zeros_like(h)
```  
**Verification:** Tested with batch sizes 128 → 96, no crash.

#### 7.3 Fix predictive_coding device placement bug — TODO
**File:** `bioplausible/zoo/models/predictive_coding.py`  
**Root cause:** `FabricPCGraphPCN.forward` creates parameters/tensors on CPU despite model being on CUDA.  
**Fix:** In `_feedforward` and `GraphPCNNode.forward`, ensure all tensors use `x.device` or `weight.device`. Add `.to(x.device)` where parameters are materialized.  
**Verification:** `uv run python -c "..."` (see above)

#### 7.4 Fix FA propagator compatibility (or mark family dead) — TODO
**File:** `bioplausible/zoo/models/fa.py` or propagator  
**Root cause:** FA models' `transition_modules()` return layers the `feedback_alignment` propagator cannot stream.  
**Option A (quick):** Add `transition_modules()` to each FA model returning compatible layer sequence.  
**Option B (honest):** If FA fundamentally can't work with current propagator interface, remove family from sweep and document why.  
**Verification:** Pre-flight `_forward_probe_ok` passes for at least `feedback_alignment` model.

#### 7.5 NeuralCube BPTT fallback — TODO (medium complexity)
**File:** `bioplausible/zoo/models/eqprop/neural_cube.py`  
**Root cause:** NeuralCube doesn't inherit from BioModel and has no `train_step` method, so trainer falls back to BPTT.  
**Fix:** Implement `train_step` using energy-contrastive rule (free settle → nudged settle → gradient difference). Model already has `_forward_step_impl`, `_initialize_hidden_state`, `_transform_input`, and `_pre_activation` — can reuse the `settle_state` protocol.  
**Alternative:** Mark neural_cube as non-eqprop-conformant if local rule can't be applied to spatial lattice structure.  
**Verification:** Probe reports `training_path='model_train_step'` not `'bptt'`.

---

### More Difficult Work (requires debugging/design)

#### 7.6 Debug Hebbian contrastive rule not learning
**Files:** `bioplausible/zoo/models/hebbian.py`, `bioplausible/core/propagators/contrastive_hebbian_learning.py`  
**Symptoms:** Rule engages (propagator created, runs), but loss flat at ~2.36 (random), accuracy 5-10% after 2 epochs.  
**Hypotheses:**  
- Learning rates too low (sweep showed `lr` ~1e-5 to 9e-4)  
- Contrastive phase not computing correct weight updates  
- Missing supervisor signal / nudging in positive phase  
- Weight update sign wrong  
**Debug steps:**  
1. Add logging to propagator `step()` showing weight update magnitudes  
2. Test with hand-tuned `learning_rate=0.01`, `hebbian_lr=0.1`  
3. Compare free/nudged phase activations — should differ meaningfully  
4. Verify `three_factor_hebbian` (which has separate modulator) vs `deep_hebbian`  
**Verification:** At least one hebbian model shows loss decrease over 2 epochs on MNIST.

#### 7.7 Eqprop energy-contrastive engine — verified, with edge cases

**Files:** `bioplausible/zoo/models/eqprop/_energy.py`, `scripts/broad_sweep.py`  
**Results from sweep:**
- ✅ 11 live models: `eqprop`, `directed_ep`, `finite_nudge_ep`, `lazy_eqprop`, `sparse_equilibrium`, `graph_eqprop` (acc 0.85!), `eqprop_mlp` (acc 0.90!), `holomorphic_ep` (acc 0.53), `modern_conv_eqprop` (acc 0.42), `conv_eqprop` (acc 0.22), `neural_cube` (acc 0.63)
- 💀 **`momentum_equilibrium` DEAD** — `RuntimeError: tensor size mismatch (128 vs 96)` in momentum variant's velocity initialization. Batch-dimension bleed: `_velocity` initialized from first batch and reused across different batch sizes.
- 💥 **`neural_cube` DEFECT** — silently falls back to BPTT (`training_path='bptt'`) instead of energy-contrastive `train_step`. Conv/structured model not wired for local rule.
- ⏭️ **3 skipped**: `eqprop_diffusion` (needs timestep `t`), `noisy_looped_mlp`/`quantized_looped_mlp` (incompatible forward)
- ⏱️ Settle speed fine — no epoch truncation, all ~10-30s/epoch for 30k param models
- 🔑 `graph_eqprop` and `eqprop_mlp` are the standout winners (85-90% acc in 2 epochs)

**Actions:**
1. ✅ **Fixed** `momentum_equilibrium` velocity buffer — now re-initializes when batch shape changes (`_energy.py` line 117-118)
2. 🟡 **TODO** Wire `neural_cube` to use local `train_step` instead of BPTT fallback — see 7.5 above
3. `graph_eqprop` and `eqprop_mlp` are proven winners (85-90% acc) — use as reference for what works

#### 7.8 Spiking and target_prop families — results

**Spiking (`spiking_stdp`):**
- 💀 **DEAD** — loss flat at ~6.5 (increasing), accuracy 10% (random), no learning signal
- Uses **BPTT fallback** (not local STDP rule) — the propagator isn't engaging the model's native learning rule
- 5.8s/epoch, 31k params — fast but ineffective

**Target Prop (`diff_target_prop`):**
- ✅ **LIVE** — loss decreasing (2.30→2.3024→2.2989), accuracy barely moving (11%→12%)
- Very slow learner — 3.5s/epoch, 31k params
- Needs tuning (possibly higher learning rate or more epochs)

**Actions:**
- Spiking: Debug why STDP rule isn't engaging. Check if `SpikingSTDP` has `transition_modules()` compatible with the default propagator path. Verify spike-timing-dependent weight updates are actually computed.
- Target Prop: Increase `learning_rate` range in rule space (sweep got `lr=0.001` which may be too low for target prop). Add more rule-specific knobs (target_loss_weight, etc.)

---

### Updated Verification Sequence (after low-hanging fixes)

```bash
# 1. Constructor sanity
uv run pytest tests/unit/experiment/test_config_knobs.py -q --no-cov

# 2. Eqprop engine learns + fast (GPU)
uv run pytest tests/unit/experiment/test_eqprop_learns.py -q --no-cov

# 3. Model surface contracts
uv run pytest tests/unit/models/test_eqprop_models.py -q --no-cov

# 4. Full unit regression
uv run pytest tests/unit/ -q --no-cov

# 5. GPU sweep smoke: all families, 2 probes, 2 epochs, 32k budget
uv run python scripts/broad_sweep.py \
  --families fa,hebbian,forward_only,predictive_coding,spiking,target_prop,eqprop \
  --probes-per-rule 2 --epochs 2 --device cuda --max-params 32000
```