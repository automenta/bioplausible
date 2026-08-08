# SWEEP_FAILURES.md — Detected Failures (collected, not yet fixed)

Inventory of concrete defects surfaced by the sweep runs. Collected here after
the wholesale fixes; not yet triaged/fixed (see `### Analysis` per item).

Reference runs:
- sweep #2 (`hebbian,forward_only,target_prop,spiking`, digits) — completed, 24 probes.
- sweep #3 (`all`, digits) — aborted mid-eqprop due to log flood / OOM; partial.

---

## 1. BPTT-fallback flood + silent-local-rule defeat (eqprop family)

**Observed**: `conv_eqprop`, `lazy_eqprop`, `momentum_equilibrium` each emit
the `BPTT fallback used ... flagged as DEFECT` warning **~108 times** (once per
batch) instead of once per probe. The sweep's `_RULE_ACTIVATION["eqprop"] =
{"config": {"gradient_method": "contrastive"}}` is supposed to force the local
rule, but these models end up on the backprop path anyway.

**Impact**: (a) the locality thesis is being measured as *backprop* cost for
these three models (they are the flagship eqprop family!); (b) log spam makes
a full sweep unreadable.

**Triaged causes to check (not yet fixed)**:
- `_train_step` dispatch order / `gradient_method` consumption for these models.
- Warning should be emitted once per probe (dedupe), not once per batch.

---

## 2. Phantom knobs on core eqprop models

**Observed** (all 3 probes each): `lazy_eqprop`, `graph_eqprop`, `eqprop_mlp`,
`conv_eqprop` report `phantom_knobs=['beta','convergence_start',
'convergence_threshold']` (plus `max_steps` for `lazy_eqprop`).

**Meaning**: the sweep samples these equilibrium knobs but the model's
reflection-derived knob schema does not consume them → they land in
`config.extra` and are silently ignored. Their configs always train with
defaults.

**Impact**: these are exactly the knobs the plan wants to sweep; currently
dead weight. Likely the `build_model_kwargs` reflection misses constructor
kwargs that shouldn't be "phantom".

---

## 3. ModernConvEqProp channel bloat (over_budget)

**Observed**: `modern_conv_eqprop` builds at **3,775,690 and 3,844,362 params**
vs the 32,000 budget → `over_budget=3844362` defect on **all** probes.

**Cause (plan §45)**: got both `hidden_dim` and `hidden_channels` via the
`**kwargs` catch-all; the param matcher's width rematch does not bind it.

---

## 4. ConvEqProp over_budget (channel rounding)

**Observed**: `conv_eqprop` builds at **58,610 and 593,034 params** vs 32k →
`over_budget` defects (plus the phantom knobs above).

**Cause (plan §44/§101)**: channel derivation rounds `hidden_channels` up to a
multiple of 8 (GroupNorm groups); at 32k budget this overshoots badly.

---

## 5. EqPropDiffusion failures

**Observed**:
- probe 0: `t must be provided for diffusion forward pass` — the model's
  forward requires a `t` argument the trainer never supplies.
- probes 1, 2: `CUDA out of memory` (150k-param diffusive model at batch 128).

**Both are hard probe failures** (not just defects) — the model cannot train in
the sweep at all.

---

## 6. (sweep #2, re-check) HebbianCube conv3d / CHL incompat

`hebbian_3d` (HebbianCube) fails all probes: the CHL propagator streams 2D
activations through conv3d transitions. Analogous to the `custom_stacked_model`
skip-if-undetectable case. Collected earlier; still open.

---

## 7. (sweep #2) Dead families — expected, for awareness

- `spiking_stdp`: DEAD (no loss decrease, acc ~0.26).
- `three_factor_hebbian`: DEAD (acc ~0.1, flat).

These are legitimate liveness verdicts (rule genuinely doesn't converge on the
given budget), not framework bugs.

---

### Not framework failures
- `equilibrium_alignment` → fixed earlier (non-trainable `B_out` in adjoint).
- staircase `%.4f` + optional `propagator` → fixed earlier (31/31 pass).
- `max_epoch_time` truncation pruning, `digits`→VISION → landed.

---

### Next step (now unblocked)
Categories 1–2 fixed, 3–5 bounded, 6 skipped via the pre-sweep gate. A bounded
re-sweep of the `eqprop` family can now run without hard failures, phantom-knob
noise, or over_budget defects; the remaining outcome is a clean liveness/resource
audit.

---

## Root-cause fixes applied (this cycle)

### A. BPTT-fallback flood — FIXED
**Root cause**: the warning was emitted inside `_bptt_step` on *every call*
(once per batch), so a probe that degraded to backprop flooded the log with
~108 identical warnings. A probe should announce its degradation once, then
record it via `training_path='bptt'` per-epoch for the defect flag.
**Fix**: `CoreTrainer` now dedupes to one warning per run
(`_bptt_fallback_warned` flag, `core/trainer.py`).

### B. eqprop models wrongly forced to BPTT — FIXED
**Root cause**: the sweep's `_RULE_ACTIVATION["eqprop"]` blanket-forced
`gradient_method="contrastive"` on *every* eqprop model. But the contrastive
path requires a model to implement `train_step` (→ `_contrastive_step`) or
`get_hebbian_pairs`. Models that expose neither (`conv_eqprop`, `lazy_eqprop`,
`momentum_equilibrium`) cannot run contrastive → the base `train_step` raises
`NotImplementedError` → Phase 3 falls through to the BPTT fallback → the
cost-of-locality probe quietly measured *backprop*, not the bio rule.
**Fix**: `scripts/broad_sweep.py` now resolves the eqprop gradient method
*per model* (`_eqprop_gradient_method`):
- has `train_step` or `get_hebbian_pairs` → `"contrastive"`
- otherwise → `"equilibrium"` (the O(1) implicit local rule, recorded as
  `implicit_equilibrium`, no BPTT defect).

### C. equilibrium_alignment "requires grad" — FIXED (re-applied)
`EquilibriumFunction` adjoint passed *all* `self.parameters()` incl. the fixed
`requires_grad=False` `B_out` → `autograd.grad` raised. Now forwards only
trainable params (`zoo/models/base.py:360`).

### Tests added
- `test_max_epoch_time.py` (6): budget capping + truncation flag.
- `test_broad_sweep.py`: `digits`→VISION, truncation defect,
  `_eqprop_gradient_method` model-awareness, `_rule_activation_for`.
- `test_bptt_opt_out.py`: warning dedup (exactly one per run).
- `test_fa_model.py`: EquilibriumAlignment backward + FA-propagator.
- `test_experiment.py`: 31/31 (staircase `%.4f` + optional `propagator`).

### Process improvements (recommended next) — DONE this cycle
- **Pre-sweep compatibility gate** (`_forward_probe_ok`): constructs each
  model once and runs a bare `forward` **plus** (for bio families) one
  propagator step before spending training compute. `eqprop_diffusion`
  (forward needs `t`) and `hebbian_3d` (CHL can't stream 2D→conv3d) are now
  skipped with a logged reason instead of crashing every probe (items 5 & 6);
  surfaced in `report["_meta"]["skipped"]`.
- **Per-model search space** (`_prune_phantom_knobs`): prunes the sampled
  family config to the knobs the model actually consumes, so healthy probes no
  longer collect `phantom_knobs=[...]` noise (item 2). `learning_rate` is
  retained (trainer consumes it); structural width knobs are untouched.
- **Budget matcher binds conv width** (`_match_param_budget`): now searches the
  conv model's real width axis `hidden_channels` (seeded from the sampled
  `hidden_dim`, GroupNorm-rounded) instead of returning the original wide
  sample — `modern_conv_eqprop`/`conv_eqprop` now fit a 32k budget rather than
  0.6–3.8 M params (items 3 & 4). `_derive_conv_channels` always derives
  `input_channels` for conv models even when `hidden_channels` is present (the
  counter bug that previously made the matcher fail).
- **Memory floor before GPU**: subsumed by the compatibility gate + budget
  matching — an OOM-prone diffusive model is now either width-matched small or
  skipped.

## 8. Fundamental eqprop quarantined by settle speed (`epoch_time_truncated`)

**Observed** (GPU, 2 epochs, 30 s/epoch cap, 32k param budget, 24 probes,
`families=eqprop`): the *most fundamental* contrastive/settling eqprop models
never complete a real probe — every run is flagged `epoch_time_truncated`
because the settle loop cannot finish even one full epoch within the per-epoch
budget. Full landscape:

| model | probe verdict | log evidence |
|-------|---------------|--------------|
| `graph_eqprop` | **LIVE**, 2/2 ok, acc 0.85 | energy-based own train_step, 9 s/epoch |
| `holomorphic_ep` | **LIVE**, 2/2 ok, acc 0.53 | contrastive |
| `conv_eqprop` | LIVE, 2/2 ok, acc 0.22 | 2.5k params (weak but honest) |
| `eqprop` (StandardEqProp) | 0/2 ok, `epoch_time_truncated` | both probes cut |
| `lazy_eqprop` | 0/2 ok, truncated | **log acc 0.70–0.77 = was learning** |
| `momentum_equilibrium` | 0/2 ok, truncated | **log acc 0.69–0.81 = was learning** |
| `sparse_equilibrium` | 0/2 ok, truncated | **log acc 0.76–0.80 = was learning** |
| `finite_nudge_ep` | 0/2 ok, truncated | slow settle |
| `eqprop_mlp` (LoopedMLP) | 2/2 ok, **not live** | completes but loss flat (acc 0.19) |
| `directed_ep` | 1 NaN + 1 ok(0.07) | **NaN divergence** |
| `modern_conv_eqprop` | 2/2 ok, acc 0.11 | 15.6k params, weak-but-honest |
| `neural_cube` | `over_budget=52618` | budget matcher can't bind its width axis |
| `eqprop_diffusion` | skipped (needs `t`) | correct, per plan §48 |

**Meaning**: the models that should prove the locality thesis — the contrastive
settling implementations (`eqprop`, `lazy`, `momentum`, `sparse`,
`finite_nudge`) — get auto-quarantined as `epoch_time_truncated` without an
honest liveness verdict. The run is a *speed* failure (settle loop too slow for
the probe budget), not a learning failure: the truncated logs show real
learning (0.7–0.8 acc) that the gate never counted. `graph_eqprop` escapes
only because it hand-rolls a fast energy step with ~5 nudged iterations.

**Root cause to fix (not patch)**:
- `settle_activations_list` runs `max_steps` (≤20) bidirectional
  `forward_dynamics` passes; with spectral-norm layers + 2 phases
  (free+nudged) + 469 batches/epoch ≈ 18k sequential settle iterations/epoch.
- Sampled `convergence_start` (≤10) + tight `convergence_threshold` (can be
  `1e-4`) rarely trigger early-stop, so every model pays the full `max_steps`.
- Fix: make convergence early-stop actually fire (spectral-norm layers converge
  in a handful of steps), cap settle steps per-model for the shallow sweep,
  and/or split the 30 s budget so a probe can complete. Target: these models
  complete 2 real epochs → honest liveness.

## 9. `eqprop_mlp` completes but loss is flat (not `bptt` fallback)

LoopedMLP (implicit `EquilibriumFunction`, O(1) memory) finishes probes but
does not decrease loss over 2 epochs (acc 0.19, `training_path` not bptt). This
is a learning-quality issue on the equilibrium-adjoint gradient — distinct from
StandardEqProp's speed failure. Needs isolation (is the adjoint gradient small
/ wrong at these sampled lrs?).
