# EXPERIMENT_PLAN6.md — Consolidated State (PLAN5 + fixes)

## What Was Fixed in PLAN5 (Complete)

| Asset | Status |
|-------|--------|
| **Training-Path Telemetry** | **LIVE** — `CoreTrainer` records `training_path` per step (`energy`, `model_train_step`, `propagator`, `implicit_equilibrium`, `bptt`); driver surfaces in probe metrics. |
| **EnergyTracker Throttle** | **LIVE** — Heavy metrics (activation/weight sparsity) computed **once per probe**, cached on model, reused; standalone use still eager. Spatial dummy fix works for conv models. |
| **BPTT Opt-Out Default** | **LIVE** — `allow_bptt_fallback=False` for bio families; loud warning + `training_path='bptt'` on silent fallback; sweep flags as DEFECT. |
| **Spectral-Norm Power-Iteration Knob** | **LIVE** — `spectral_norm_power_iterations` on `BioModel`/eqprop; passed through `build_model_kwargs`. |
| **Deep Local O(1) Memory** | **LIVE** — `DeepHebbianChain.train_step` runs local Oja rule (no BPTT): **14.8 MB @100L** vs backprop 47.3 MB; CHL propagator rewritten `no_grad` streaming: **27.3 MB @100L** vs backprop 47.3 MB. |
| **EqProp GPU Memory vs BP** | **LIVE** — Implicit equilibrium (`gradient_method="equilibrium"`) O(1) in steps undercuts unrolled BPTT on same arch; telemetry labels `implicit_equilibrium`; lock-in tests pass. |
| **DirectedEP Early-Convergence** | **LIVE** — Forward honors config `convergence_start`/`convergence_threshold` (no longer hardcoded full steps). Probe 0.9s (was slow/no-converge). |
| **Conv Spatial Fix** | **LIVE** — `_estimate_activation_sparsity` builds spatial dummies for conv models; `conv_eqprop` / `modern_conv_eqprop` build & train. |
| **Conv GroupNorm Divisibility** | **LIVE** — `_derive_conv_channels` rounds `hidden_channels` up to multiple of 8 (GroupNorm groups). |
| **Domain Filtering** | **LIVE** — Sweep filters models by task domain (e.g. MNIST→VISION only); LM transformers (`backprop_transformer_lm`), custom_stacked_model excluded automatically. |
| **Sweep Self-Diagnosis** | **LIVE** — Per-probe `training_path` surfaced; bio family probes falling back to BPTT flagged `DEFECT` in report. |
| **EnergyTracker Spatial Dummy** | **LIVE** — `_build_spatial_dummy` infers 4D `(1,C,H,W)` for conv/spatial models; conv models no longer break in activation-sparsity estimation. |

---

## Verification (All Green)

```bash
# Core new tests
uv run pytest tests/unit/experiment/test_energy_tracker_throttle.py -q  # 3 passed
uv run pytest tests/unit/experiment/test_training_path.py -q            # 4 passed
uv run pytest tests/unit/experiment/test_sweep_defect_flag.py -q        # 6 passed
uv run pytest tests/unit/experiment/test_bptt_opt_out.py -q             # 3 passed
uv run pytest tests/unit/experiment/test_deep_hebbian_o1_memory.py -q   # 2 passed (GPU)
uv run pytest tests/unit/experiment/test_eqprop_learns.py -q            # 2 passed (GPU)
uv run pytest tests/unit/experiment/test_eqprop_memory_advantage.py -q  # 2 passed (GPU, xfail removed)
```

---

## Remaining / Known Issues

| Issue | Priority | Notes |
|-------|----------|-------|
| **Parameter-Matched Comparison** | High | Sweep / memory tests currently compare at matched depth/width; fair comparison = fixed `param_count ≈ 8k` budget, measure `peak_memory_mb` AND `final_acc` at that budget. Needs sweep config / probe normalization. |
| **DirectedEP NaN Loss** | Medium | `directed_ep` on MNIST (784-dim, 146k params, contrastive) produces `Train Loss=nan`. Likely beta/lr too high for 784-dim contrastive; needs sweep-space clamping (`beta` lower, `hebbian_lr` lower) or migration to `equilibrium` path. |
| **ForwardForward NaN** | Medium | `forward_forward` probe hits `Train Loss=nan` (10s/epoch). Investigate / clamp sweep space. |
| **Conv Reshape (expected 4 got 2) Residual** | Low | EnergyTracker spatial dummy fixed, but if a conv model's `_initialize_hidden_state` still unpacks 4D, ensure model `input_format="spatial"` is set and `_adapt_input` passes 4D. `conv_eqprop` now works; verify `modern_conv_eqprop`. |
| **ModernConvEqProp Channel Bloat** | Low | `modern_conv_eqprop` got both `hidden_dim` and `hidden_channels` (242k params) due to `**kwargs` catch-all. Either drop `hidden_dim` from derived config for conv, or model should ignore redundant. |
| **NaN Divergence Sweep Guard** | Medium | Add `_check_numerical_health` gate in sweep (already in trainer) to prune NaN runs early and log as defect. |
| **Sweep Defect: custom_stacked_model** | Low | Needs `layers_config` — skip via domain filter or add derivation. |

---

## Next Actions (Priority Order)

### Done this cycle (root-cause fixes, not patches)

1. **Single construction layer** (`bioplausible/core/construction.py`). The
   phantom-drift root cause: `build_model_kwargs` built eqprop models via loose
   kwargs, so `beta`/`learning_rate`/`max_steps` landed in `config.extra`
   (ignored) and every probe trained with identical defaults (identical loss
   across all sampled configs — that is why nothing we tuned ever changed).
   Now `ModelConfig`'s fields are the reflection-derived knob schema, no aliases,
   and `construct_model()` is the one path used by trainer/estimator/finders/
   probe. `phantom_knobs()` surfaces unconsumed knobs as defects.
2. **Liveness-gate fix**: trainer backfills train loss/acc for propagator paths
   (FA/hebbian `step()` returned `None` → `Train Loss=0` → every rule looked
   "dead"). The gate now sees real loss.
3. **NaN guard** across all training paths (raise fast + flag `nan_divergence`);
   eqprop `learning_rate` capped at 5e-3 (measured DirectedEP divergence
   threshold).
4. **Parameter-matched sweep**: `--max-params N` rematches width to a fixed
   budget (static estimator, no training), minimises un-budgetable models, flags
   `over_budget`.
5. **Self-diagnosis defect flags**: `bptt_fallback`, `nan_divergence`,
   `phantom_knobs=[…]`, `over_budget=N` surfaced per model/family; `param_count`
   in the report.
6. **Fast no-op unit tests** (~2 s, fake trainer/driver/estimator) lock in all of
   the above without GPU loops.

### Remaining

- **DirectedEP / DeepDFAEqProp settling speed**: equilibrium-settling forwards
  (`for _ in range(max_steps)` over spectral-norm layers) dominate epoch time
  (e.g. ~53 s/epoch for 7.5k-param DeepDFAEqProp). Needs per-model settle-step
  capping / convergence early-stop for the shallow sweep.
- **Pre-existing staircase test failures**: 6 tests fail on clean HEAD
  (`staircase.py:322` formats a string error with `%.4f`, and `run_probe` passes
  `propagator=` to fake drivers that don't accept it). Unrelated to this work.

---

## Key Files Modified

| File | Purpose |
|------|---------|
| `bioplausible/core/trainer.py` | Training-path telemetry, BPTT opt-out warning, dispatch order (propagator before model.train_step), implicit_equilibrium path. |
| `bioplausible/core/energy.py` | Throttle heavy metrics once/probe; `_build_spatial_dummy` for conv models; pass `None` to `_estimate_activation_sparsity`. |
| `bioplausible/core/model.py` | `spectral_norm_power_iterations` param, `apply_spectral_norm` uses it. |
| `bioplausible/core/config.py` | `spectral_norm_power_iterations` field in `ModelConfig`. |
| `bioplausible/zoo/models/hebbian.py` | `DeepHebbianChain.train_step` = local Oja rule (O(1) memory, no BPTT). |
| `bioplausible/zoo/propagators/hebbian.py` | `ContrastiveHebbianLearning.step` = `no_grad` streaming free/clamped, per-layer outer-product, O(1) memory. |
| `bioplausible/experiment/probe.py` | Driver surfaces `training_path`; threads `allow_bptt_fallback`; `ProbeResult` carries `training_path`. |
| `bioplausible/experiment/param_estimator.py` | `_derive_conv_channels` rounds `hidden_channels` to multiple of 8; conv-channel derivation gated on model signature. |
| `scripts/broad_sweep.py` | Domain filtering (`_task_domain`); `_models_in_family` filtered by domain; conv hidden_channels multiple-of-8 clamp. |
| `bioplausible/zoo/models/eqprop/deep_ep.py` | `forward()` honors `convergence_start`/`convergence_threshold` (early stop enabled). |
| `bioplausible/core/energy.py` | `_build_spatial_dummy` for conv models; `_estimate_activation_sparsity` builds spatial dummy when `sample_input=None`. |

---

## Verification Commands

```bash
# Quick smoke
uv run pytest tests/unit/experiment/test_energy_tracker_throttle.py tests/unit/experiment/test_training_path.py tests/unit/experiment/test_sweep_defect_flag.py tests/unit/experiment/test_bptt_opt_out.py -q --no-cov

# GPU lock-in
uv run pytest tests/unit/experiment/test_deep_hebbian_o1_memory.py tests/unit/experiment/test_eqprop_learns.py tests/unit/experiment/test_eqprop_memory_advantage.py -q --no-cov

# Regression suite (core + experiment + eqprop models)
uv run pytest tests/unit/core/test_core_trainer.py tests/unit/experiment/ tests/unit/models/test_eqprop_models.py -q --no-cov
```