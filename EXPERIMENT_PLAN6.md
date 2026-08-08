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
| **Parameter-Matched Comparison** | **DONE** | `_match_param_budget` now searches the conv model's real width axis `hidden_channels` (seeded from sampled `hidden_dim`, GroupNorm-rounded), so `modern_conv_eqprop`/`conv_eqprop` fit a 32k budget instead of building 0.6–3.8 M params. Measured at fixed `param_count≈budget` + `final_acc` at that budget. |
| **DirectedEP NaN Loss** | Medium | `directed_ep` on MNIST (784-dim, 146k params, contrastive) produces `Train Loss=nan`. Likely beta/lr too high for 784-dim contrastive; needs sweep-space clamping (`beta` lower, `hebbian_lr` lower) or migration to `equilibrium` path. |
| **ForwardForward NaN** | Medium | `forward_forward` probe hits `Train Loss=nan` (10s/epoch). Investigate / clamp sweep space. |
| **Conv Reshape (expected 4 got 2) Residual** | **DONE** | `_derive_conv_channels` now always derives `input_channels` for conv models (even when `hidden_channels` is present), so the param counter / budget matcher can build `conv_eqprop`; `input_format="spatial"` + `_adapt_input` pass 4D. |
| **ModernConvEqProp Channel Bloat** | **DONE** | Param matcher binds `hidden_channels` within a 32k budget (was 242k–3.8 M via `**kwargs` catch-all). |
| **NaN Divergence Sweep Guard** | **DONE** | `_check_numerical_health` in trainer + sweep-level NaN check (`_probe_runs` flags `nan_divergence`, excludes from ok/liveness). |
| **Sweep Defect: custom_stacked_model** | Low | Needs `layers_config` — skip via domain filter or add derivation. |
| **Sweep compatibility gate** | **DONE** | `_forward_probe_ok` pre-flights a bare `forward` + bio propagator step before training; `eqprop_diffusion` (needs `t`) and `hebbian_3d` (CHL can't stream conv3d) are skipped with a logged reason (SWEEP_FAILURES #5/#6). |
| **Phantom-knob noise** | **DONE** | `_prune_phantom_knobs` prunes family-space knobs a model cannot consume, so healthy probes no longer flag `phantom_knobs=[...]` (SWEEP_FAILURES #2). |
| **Settle-speed quarantine** | **HIGH** | Contrastive settling models (`eqprop`, `lazy_eqprop`, `momentum_equilibrium`, `sparse_equilibrium`, `finite_nudge_ep`) all hit `epoch_time_truncated` on GPU: the settle loop (~18k sequential iterations/epoch) can't finish a probe inside the 30s epoch budget. Truncated logs prove they WERE learning (acc 0.7–0.8) — the gate never counted it. Needs: convergence early-stop that actually fires + per-model settle-step cap for shallow probes (SWEEP_FAILURES #8). |
| **eqprop_mlp loss-flat** | Medium | LoopedMLP (implicit `EquilibriumFunction`, O(1)) completes probes but loss doesn't decrease (acc 0.19, path not `bptt`). Equilibrium-adjoint gradient learning-quality issue, distinct from StandardEqProp's speed failure. |
| **DirectedEP NaN Loss** | Medium | `directed_ep` on MNIST (784-dim, contrastive) produces `Train Loss=nan` — reproduced on GPU sweep (1/2 probes). Needs sweep-space clamping (`beta` lower, `max_steps` lower) or migration to `equilibrium` path. |
| **neural_cube over_budget** | Low | `--max-params` matcher can't bind `neural_cube`'s width axis (`cube_size`) → still `over_budget=52618` on GPU. It learns (acc 0.92) but is excluded from ok/liveness. |

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

- **Settle-speed quarantine (HIGH — next session P0)**: contrastive settling
  models (`eqprop`, `lazy_eqprop`, `momentum_equilibrium`, `sparse_equilibrium`,
  `finite_nudge_ep`) never complete a probe (`epoch_time_truncated`) because the
  bidirectional settle loop (~18k sequential iterations/epoch with
  free+nudged phases × `max_steps`) overruns the epoch budget. The truncated
  logs show genuine learning (acc 0.7–0.8), so this is a probe-speed defect,
  not a liveness verdict. Fix plan:
  1. Instrument `settle_activations_list` to measure real steps-before-
     early-stop and the settle/step/energy time split.
  2. Make convergence early-stop actually fire (sampled `convergence_start` ≤10
     + `1e-4` threshold rarely triggers; spectral-norm layers converge in a few
     steps) and/or cap settle steps per-model in `_SHALLOW_CAPS` for the
     shallow breadth probe.
  3. Re-run `families=eqprop` (GPU) and require these models to complete 2 real
     epochs (honest liveness), not just be un-truncated.
- **DirectedEP / DeepDFAEqProp settling speed**: equilibrium-settling forwards
  (`for _ in range(max_steps)` over spectral-norm layers) dominate epoch time
  (e.g. ~53 s/epoch for 7.5k-param DeepDFAEqProp). Needs per-model settle-step
  capping / convergence early-stop for the shallow sweep.
- **eqprop_mlp loss-flat**: LoopedMLP completes probes but loss doesn't decrease
  over 2 epochs (acc 0.19, path not `bptt`). Isolate whether the
  `EquilibriumFunction` adjoint gradient is wrong/oversmall at sampled lrs.
- **directed_ep NaN**: reproduced on GPU (1/2 probes). Clamp `beta`/`max_steps`
  or route to `equilibrium` path.
- **neural_cube over_budget**: `_match_param_budget` can't bind `cube_size`
  axis. Add `cube_size` to the matcher's width-axis search.
- **Pre-existing staircase test failures**: 6 tests fail on clean HEAD
  (`staircase.py:322` formats a string error with `%.4f`, and `run_probe` passes
  `propagator=` to fake drivers that don't accept it). Unrelated to this work.

---

## Fresh-Session Start Plan (batch at end of session, per decision)

1. **P0 — settle-speed fix** (see Remaining). Then re-run `families=eqprop`
   GPU sweep; confirm `eqprop`/`lazy`/`momentum`/`sparse`/`finite_nudge` get
   honest liveness verdicts, `eqprop_mlp` loss decreases, `directed_ep` not NaN.
2. **P1 — bio families breadth sweep** (GPU): `--families
   fa,hebbian,forward_only,predictive_coding,spiking,target_prop`,
   `--probes-per-rule 1 --epochs 2 --device cuda --max-params 32000
   --max-epoch-time 30`. Capture live/dead landscape + `bptt_fallback`
   self-diagnosis + NaN/over_budget/phantom defects (skip `backprop`; it is the
   comparison baseline, not a bio family).
3. **P2 — batch docs**: append all results to `SWEEP_FAILURES.md` + this plan.

### GPU eqprop verification snapshot (2026-08-08, `families=eqprop`, 2 epochs, 32k budget, 30s/epoch)

| model | verdict | acc | defects |
|-------|---------|-----|---------|
| graph_eqprop | LIVE | 0.85 | none |
| holomorphic_ep | LIVE | 0.53 | none |
| conv_eqprop | LIVE | 0.22 | none |
| eqprop_mlp | not live | 0.19 | none (loss flat) |
| modern_conv_eqprop | not live | 0.11 | none |
| eqprop | dead | — | epoch_time_truncated |
| lazy_eqprop | dead | — | epoch_time_truncated |
| momentum_equilibrium | dead | — | epoch_time_truncated |
| sparse_equilibrium | dead | — | epoch_time_truncated |
| finite_nudge_ep | dead | — | epoch_time_truncated |
| directed_ep | dead | — | nan_divergence |
| neural_cube | dead | — | over_budget=52618 |
| eqprop_diffusion | skipped | — | (needs `t`) |

Report: `logs/broad_sweep_mnist.json`. Family `live=True` (3/12 live).

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