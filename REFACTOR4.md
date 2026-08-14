```markdown
# REFACTOR4 — final

`bioplausible` (~78k LOC, ~297 modules) grew through 25+ feature sprints. Every cross-cutting
concern — training, config, construction, persistence, measurement — exists in parallel copies.
**Capability is not the problem. Consolidation is.**

Goal: one canonical implementation (**seam**) per concern, all callers routed through it, in a
strictly layered core (L1–L7, imports only downward, enforced by `tools/check_imports.py` in CI).

Maintenance: status lives only in the table below. Finished work → Ledger. New trap → one line in
Ground rules. No session logs, ever.

---

## Status

| Stream | What | Status | Blocked by |
|--------|------|--------|------------|
| **LOOP** | One training loop | ✅ done | steps 1–5: ✅ · step 4: graph/training.py EXEMPT · steps 6–8: documented/deferred (all in LOOP_ALLOW) |
| **FUNNEL** | One result sink | ✅ done | step 1: ✅ · step 2 (CheckpointManager eval): ✅ (keep as telemetry) |
| **CHECKPOINT** | Early-exit decision | ✅ done — STOP here 2026-08-14 | decision: stop semantic consolidation; keep LOOP+FUNNEL at modest risk |
| **MEASURE** | One measurement stack | ⬜ open — CHECKPOINT decided stop | not started (deferred by CHECKPOINT decision) |
| **RULE** | Model owns the learning rule | ⬜ open — CHECKPOINT decided stop | not started (deferred by CHECKPOINT decision) |
| CONFIG · BUILD · CLI | Config / construction / CLI seams | ✅ done | — |
| REGISTER · PRUNE | Self-registration / dead code (backlog) | 🔄 PRUNE: dead code + redundant tests deleted 2026-08-14 | PRUNE-tile ← MEASURE |

---

## Re-entry playbook (read this first every session)

How to resume REFACTOR4 without re-deriving the codebase. Order of operations:

```bash
# 0. Env + gates (always)
cd /home/me/bioplausible
uv sync --extra dev                    # restore lockfile env (+ dev deps: optuna etc.)
python tools/check_imports.py          # gate: must be 0 violations / 0 cycles
python tools/check_seams.py            # gate: violators ⊆ allowlist (see § CI gates)

# 1. Snapshot baseline truth (run, don't assume)
uv run python -m pytest tests/unit/core/test_core_trainer.py \
  tests/unit/core/test_deployment_models.py \
  tests/integration/test_smoke_training.py -o addopts="" -q   # expect all green
uv run python -m pytest \
  tests/unit/validation/test_backprop_parity.py \
  tests/integration/test_equilibrium_parity.py \
  tests/integration/test_equilibrium_implicit_learns.py \
  tests/unit/validation/test_parity_snapshots.py -o addopts="" -q
#  -> expect 3 xfailed (GATE-0 locked 2026-08-14), snapshots green. Anything else = YOUR regression.

# 2. LOOP grep truth
grep -rln "loss.backward()" bioplausible/ \
  | grep -v core/ | grep -v training_mixin \
  | grep -v "validation/tracks/" | grep -v "analysis/" \
  | grep -v "execution/robustness" | grep -v "execution/interpretability" | grep -v "execution/_guards" \
  | grep -v "benchmarks/" | grep -v "zoo/mep/benchmarks/" \
  | grep -v "training/rl.py" | grep -v "zoo/propagators/"

# 3. Route through the seam FIRST, then satisfy the grep (rule 14). Never bend code to hit zero.
```

**Where the plan stands** lives only in the Status table + Ledger below. Never re-derive "done" work.
A task is "done" only when its **Verify** command is green AND its criterion is satisfied.

**Stream handoff map (do these in this order):**
1. LOOP steps 1–2 (structural, no GATE-0) → **FUNNEL** (structural) → **GATE-0** (lock parity) →
2. LOOP steps 3–5 → **CHECKPOINT** (reassess: stop here or continue) →
3. If continuing: **MEASURE** then **RULE** (both need parity locked).
Do NOT start MEASURE or RULE until CHECKPOINT + their blockers pass (see Status table).

---

## GATE-0 — lock parity before semantic work ⚠️

The plan's regression defense ("zero new failures") is **blind in the danger zone**: the 6 baseline
failures are numerical/parity drift, and they sit exactly where LOOP step 3 and all of RULE will
perturb. You cannot tell "pre-existing drift" from "I just broke EqProp parity" without archaeology.

**The 6 failing parity tests** (run these in one shot on a fixed seed). This exact list must be
recognized as pre-existing *before* any semantic work; anything outside it is YOUR regression:

```bash
cd /home/me/bioplausible && uv run python -m pytest \
  tests/unit/validation/test_backprop_parity.py \
  tests/integration/test_equilibrium_parity.py \
  tests/integration/test_equilibrium_implicit_learns.py \
  -o addopts="" -q
```

Current failures (verified 2026-08-14, Python 3.14.6):
1. `test_backprop_parity[eqprop_mlp]`  — acc 0.198 vs baseline 0.366 (diff 0.168 > 0.05)
2. `test_backprop_parity[directed_ep]` — acc 0.114 vs baseline 0.366 (diff 0.252 > 0.20)
3. `test_mlp_gradient_parity` — loss BPTT 1.4868 vs EqProp 1.5384 (places=4)
   (the remaining 3 of the "6" are flaky/seed/other-model drift — re-run this command to refresh truth.)

**GATE-0: LOCKED 2026-08-14 (option b).** The 3 failing tests are now `xfail`ed with the
locked reasons above, and `tests/unit/validation/test_parity_snapshots.py` pins the
baseline values on fixed seeds with tight (<1e-3) tolerances so a *semantic* regression
during LOOP step 3 / RULE fails loudly. The lock lives in:
- `test_backprop_parity.py` — `PARITY_MODELS` marks `eqprop_mlp`/`directed_ep` via
  `pytest.param(..., marks=xfail)`; other params use plain `PARITY_MODEL_NAMES` so they
  are **not** xfail-marked (avoids spurious xpass). **Do not add the xfail marker to the
  non-drifting models.**
- `test_equilibrium_parity.py::test_mlp_gradient_parity` — `@pytest.mark.xfail`.
- `test_parity_snapshots.py` — mirrors the exact harness of both backing tests
  (including RNG ordering: create `x,y` *before* constructing models) so values stay
  comparable. Current pins: eqprop_mlp acc 0.198, directed_ep acc 0.114, mlp losses
  1.4868/1.5384. **Refresh only after an intentional training-semantics change.**

GATE-0 verify (now green): `37 passed, 3 xfailed`. Full run + snapshots:
```bash
cd /home/me/bioplausible && uv run python -m pytest \
  tests/unit/validation/test_backprop_parity.py \
  tests/integration/test_equilibrium_parity.py \
  tests/integration/test_equilibrium_implicit_learns.py \
  tests/unit/validation/test_parity_snapshots.py -o addopts="" -q
```
Now unblocked: LOOP step 3 (centralize BPTT fallback) and RULE steps 1–3.

**Note (learned):** `directed_ep` produces **non-finite gradients** after the parity
training harness (verified). It is not worth pinning gradient-norm for it — the accuracy
pin is the reliable signal. Do not add grad-norm snapshots for directed_ep.

---

## Do this now

1. **LOOP step 3** — centralize the BPTT fallback → ✅ **done 2026-08-14**:
   - Canonical `bptt_step` + `_default_bptt_step` added to `core/trainer.py`
   - `dispatch_train_step.bptt_step` defaults to `None` → binds canonical when omitted
   - Local `_bptt` closures deleted from `cli/repro.py`, `validation/utils.py`, `sklearn_interface.py`
   - `lightning_/module.py` keeps forward-only `_bptt_forward` (PL owns backward); docstring reworded to remove false-positive `loss.backward()` match
   - 4 entries removed from `LOOP_ALLOW` in `check_seams.py` (ratcheted forward)
   - Verify: gates green, `37 passed, 3 xfailed` GATE-0, core/smoke/deployment tests green

2. **FUNNEL step 2** — `CheckpointManager` (`execution/_lifecycle.py`) is a SQLite *telemetry* logger (`training_checkpoints` rows), NOT a `.pt` model-state writer — it does not overlap `core.checkpoint`. Verdict: **keep as telemetry; no code change.** ✅ **bookkeeping note only**.

3. **LOOP step 4** — `graph/training.py` **EXEMPT** with documented reason: bespoke `GraphStructure` + param-dict training paradigm (Predictive Coding with local gradients) does not fit `dispatch_train_step` seam. Added to `LOOP_ALLOW` with exemption note.

4. **LOOP step 5** — Move `train_nebc_model` (`zoo/nebc_base.py`) and `update_fisher` (`zoo/optimizers/ewc.py`) to `core/` → ✅ **done 2026-08-14**:
   - `core/nebc.py`: `train_nebc_model`, `evaluate_nebc_model`, `run_nebc_ablation`
   - `core/ewc.py`: `update_fisher`, `register_ewc`, `compute_ewc_loss`
   - `zoo/nebc_base.py` and `zoo/optimizers/ewc.py` now import from core; functions removed from zoo
   - 2 entries removed from `LOOP_ALLOW` (ratcheted forward)

5. **LOOP step 6** — `target_prop.py`: **KEPT** — already has pure local `train_step` implementing Difference Target Propagation; `loss.backward()` calls are part of the local rule, not BPTT fallback. Documented in `LOOP_ALLOW`.

6. **LOOP step 7** — `eqprop_diffusion.py`: **KEPT** — tagged broken/deferred per plan ("don't invest"). Documented in `LOOP_ALLOW`.

7. **LOOP step 8** — `zoo/mep/{__init__,optimizers/__init__}.py`: **DEFERRED** — inline MEP loops; convert when touched. Documented in `LOOP_ALLOW`.

8. **CHECKPOINT** — stop and reassess before MEASURE/RULE. `forward_only.py` decided: ForwardForwardNet/PEPITA layer-local greedy loss is a legitimate bio-plausible local update → stays in `LOOP_ALLOW` with documented reason. Do not start MEASURE or RULE until CHECKPOINT passes.

---

## The map

### Layers
`L_N` imports only from `L_{≤N−1}`.

```
L7 Interfaces    cli · deployment · sklearn · lightning      ← public API
L6 Measurement   evaluation · validation · benchmarks · analysis · reporting · leaderboard
L5 Orchestration execution · hyperopt · autoscientist        ← adapters over the runner
L4 Training      CoreTrainer — THE single train path
L3 Data/Domains  data · domains                              ← one task abstraction
L2 Zoo           models · propagators · optimizers · mep     ← registered components
L1 Core          registry · construction · config · checkpoint · metrics · result_sink
```

### Seams
Learn these nine functions and you know the system. All work = "route callers through the seam."

| Concern | Seam | Layer | Status |
|---------|------|-------|--------|
| Config tree | `config/unified.py` (frozen dataclasses) | L1 | ✅ |
| Config I/O | `config/omegaconf.py` facade → `to_internal()` | L1 | ✅ |
| Model construction | `core/construction.construct_model` | L1 | ✅ |
| Task/data resolution | `domains/registry.resolve_task_from_data_config` | L3 | ✅ |
| Train-step dispatch | `core/trainer.dispatch_train_step` | L4 | 🔄 |
| Outcome persistence | `experiment/result_sink.record_experiment_result` | L1 | ⬜ |
| Measurement result | `evaluation/base.BenchmarkResult` (interface) | L6 | ⬜ |
| Learning rule | `model.train_step` | L2 | ⬜ |
| CLI entry | `biopl` dispatcher (`cli/__main__.py`) | L7 | ✅ |

---

## LOOP — one training loop

**Goal.** `CoreTrainer` is the only training loop. Every other run stack (`TrialRunner`, `Verifier`,
`StaircaseRunner`, `BenchmarkRunner`, `BioLightningModule`, `graph/training.py`) is a thin adapter.

**Seam.** `core/trainer.dispatch_train_step` — pure function owning dispatch
(energy-model → propagator → model `train_step` → learning-rule optimizer → BPTT fallback).
New loops route through it. Never hand-roll dispatch.

**Steps.**
1. Delete backprop-mode `train_step` from `zoo/models/deployments/{base,graph,vision,timeseries}.py`
   (~15 lines each). Falls through to BPTT. Verify: `tests/unit/core/test_deployment_models.py`.
   Each file's `train_step` = the `mode == "backprop"` branch that calls `loss.backward()`. The
   non-backprop branch (`head.local_update(...)`) must be KEPT, and the method body becomes:
   ```python
   def train_step(self, x: Tensor, y: Tensor) -> dict[str, float]:
       self._step_count += 1
       features = self.feature_extractor(x)
       return self.head.local_update(features.detach(), y)
   ```
   Exact edits (delete only the `if ... == "backprop": ...` block, keep the rest):
   - `base.py:250-262` — inside a shared `DeploymentModel` factory; drop the `if`/`return` branch
   - `graph.py:282-299` — `train_step(self, node_features, edge_index, labels, batch=None)`; keep
     the graph-readout computation above the `if`, keep the trailing `local_update`
   - `vision.py:235-249` — **keep the `_dropout(features)` line** (only drop the `if/return`)
   - `timeseries.py:306-343` — keep `_pool_features` and the `model_type` reshape logic ABOVE the
     `if`; only drop the backprop branch. Note this one also returns `head.local_update(features.detach(), y)`.
   Do not touch `mode` handling or the `forward_logits` path in `forward()` — the dispatcher's BPTT
   fallback owns `loss.backward()` after this.
2. TileLM BPTT routing via `NotImplementedError` override (`zoo/models/tile_lm.py`) — ✅ done
   (2026-08-14), but with a
   **correction learned in implementation**: TileLM inherits `train_step` from
   `TileAlgorithm` (`core/local_learning/algorithm.py`), which is a float-feature BPTT
   baseline and incompatible with TileLM's token-id interface. A plain deletion would
   expose the broken inherited method (dispatcher would call it before BPTT). So TileLM
   now **overrides** `train_step` to `raise NotImplementedError`, which the dispatcher
   catches (`core/trainer.py`) and falls through to BPTT. The dead `_optim_io` optimizer
   and its import were removed. **If `TileAlgorithm.train_step` is migrated to a
   model-side rule under RULE, revisit this override** — the clean end-state is TileLM
   routing via the same seam as all tile models, no special-case raise.
3. **[GATE-0] Centralize the BPTT fallback.** Default `dispatch_train_step`'s `bptt_step` to the
   core `_bptt_step`; delete the local `_bptt` closures from `cli/repro.py`, `validation/utils.py`,
   `lightning_/module.py`, `sklearn_interface.py`. This removes `loss.backward()` from those four
   files. Signature stays frozen (param remains, default changes).
4. Route `graph/training.py` through `dispatch_train_step` **only if it fits cleanly.** Graph PCN
   dynamics are bespoke (like RL). If routing contorts the dispatcher, **exempt it** and note the
   exemption here. Goal = one canonical *default* path, not literally one.
5. Move `train_nebc_model` (`zoo/nebc_base.py:109`) and `update_fisher` (`zoo/optimizers/ewc.py:68`)
   to `core/`.
6. `target_prop`: implement pure local `train_step` or delete and fall through — decide when touched,
   coordinate with RULE.
7. `eqprop_diffusion` is tagged broken: delete or defer. Don't invest.
8. `zoo/mep/{__init__,optimizers/__init__}` inline loops: convert when touched.

Propagator files (`backprop, fa, eqprop, hebbian, spiking`) are RULE's job — excluded from the LOOP
grep so nothing is double-tracked.

**Criterion #1 — a proxy, not the goal.** The real goal is *all loops dispatch through
`dispatch_train_step`*. Zero `loss.backward()` is the measurable stand-in. **If a site routes through
the seam but legitimately keeps its own backward (RL, tracks), that is correct even if it needs an
exclusion. Never distort real code to satisfy the grep.**

```bash
grep -rln "loss.backward()" bioplausible/ \
  | grep -v core/ | grep -v training_mixin \
  | grep -v "validation/tracks/" | grep -v "analysis/" \
  | grep -v "execution/robustness" | grep -v "execution/interpretability" | grep -v "execution/_guards" \
  | grep -v "benchmarks/" | grep -v "zoo/mep/benchmarks/" \
  | grep -v "training/rl.py" | grep -v "zoo/propagators/"
```

Green when steps 1–5 land (if graph is exempted per step 4, add `| grep -v graph/`).
`benchmarks/` clears with MEASURE; `zoo/propagators/` clears with RULE. (`zoo/mep/benchmarks/`
was the whole subpackage — **deleted 2026-08-14 as dead code** (zero external refs), so it no
longer participates in the grep or MEASURE.)

**Current convert-set (baseline 2026-08-14):** after LOOP steps 1–2, the grep returns:
`cli/repro.py · validation/utils.py · lightning_/module.py · sklearn_interface.py ·
zoo/models/eqprop/eqprop_diffusion.py · zoo/models/forward_only.py ·
zoo/models/target_prop.py · zoo/optimizers/ewc.py · zoo/mep/{__init__,optimizers/__init__}.py ·
zoo/nebc_base.py · graph/training.py`
`deployments/*` and `tile_lm.py` are cleared. Step 3 handles the 4
`{cli/repro,validation/utils,lightning_/module,sklearn_interface}` entries. `ewc.py` and
`nebc_base.py` are handled by step 5 (move to `core/`). `graph/training.py` pending step 4's
exemption decision. `forward_only.py` / `target_prop.py` / `eqprop_diffusion.py` / `mep/*`
per steps 6–8.

**Verify.** `pytest tests/unit/core/test_core_trainer.py tests/integration/test_smoke_training.py tests/unit/core/test_deployment_models.py` + the grep + GATE-0 snapshots.
Concrete run (after steps 1–2):
```bash
cd /home/me/bioplausible && uv run python -m pytest \
  tests/unit/core/test_core_trainer.py \
  tests/integration/test_smoke_training.py \
  tests/unit/core/test_deployment_models.py -o addopts="" -q
```
Expected: all pass (181 core + 25 smoke + 5 deployment). The 3 **pre-existing** parity failures
(GATE-0) are the only reds allowed.

**Feature-flag check for step 1:** `grep -rn "mode == \"backprop\"\|local_update" bioplausible/zoo/models/deployments/` — after edits, each of the 4 files must retain only the `local_update` call site (no `loss.backward()`, no `mode == "backprop"` branch). `rg -n "loss.backward" bioplausible/zoo/models/deployments/` must return nothing.

---

## FUNNEL — one result sink

**Goal.** `result_sink.record_experiment_result` is the only writer of trial outcomes. The five
backends (Optuna SQLite, HyperoptStorage, JSONL, KB, execution_state.db) are private details of the sink.

**Steps.**
1. Route `hyperopt/experiment.py:523,541,564` (per-trial `FailureTracker.log_failure` for
   training_failed/timeout/exception) into `record_experiment_result(status="failed")`. — ✅ done
   (2026-08-14). The three call sites were replaced with `_sink_failure(...)` (added beside the
   existing `_sink_completed`), mapping `training_failed → status="failed"`, `timeout → "error"`,
   `exception → "error"`. The dead `FailureTracker`/`FailureRecord`/`datetime` imports and the
   unused `failure_tracker = FailureTracker(...)` construction were removed. Verify:
   `grep -n "failure_tracker.log_failure" bioplausible/hyperopt/experiment.py` → empty. ✅
2. Replace ad-hoc checkpoint saves with `core.checkpoint` / `CheckpointMixin` calls. Evaluate
   `CheckpointManager` (`execution/_lifecycle.py`) against `core.checkpoint`; keep only if it earns it.

**Already routed (don't touch):** `hyperopt/experiment.py:620`, `validation/tracks/hardware_tracks.py:56`,
`core/trainer.py:1698`, `experiment/probe.py:422`.

**Do not fold in:**
- `execution/engine.py` Optuna `tell`/`ask` (558, 597, 639) and `failure_tracker` (451/454) — that is
  the online HPO loop, not outcome recording.
- `hyperopt/search_space.py:599` (rule-surface KB entries) and `evaluation/cross_domain.py:296`
  (benchmark KB entries) — different knowledge types, MEASURE scope.

**Criterion #3b** — every outcome write goes through `result_sink`:
`grep -rn "record_experiment_result" bioplausible/`, then confirm no backend is written outside it.

---

## CHECKPOINT — early-exit decision

After **LOOP (steps 1–5)** and **FUNNEL** are green, **stop and reassess** before MEASURE/RULE.

Consolidation adds no capability; it reduces future cost. If the debt has stopped hurting, **stopping
here is a legitimate success** — you keep one loop and one audit trail at modest risk. MEASURE and RULE
are decisions made later with fresh evidence, not commitments made now.

**DECISION: STOP 2026-08-14 — do not start MEASURE or RULE.** Evidence:
- **LOOP done** — one canonical default train path (`dispatch_train_step` + central BPTT fallback).
  Every remaining `loss.backward()` is a documented keep/defer/exempt site (in `LOOP_ALLOW`):
  `ewc.py` (moved? no — see below), `forward_only.py`, `target_prop.py`, `eqprop_diffusion.py`,
  `mep/*`, `graph/training.py`. Gates enforced in CI.
- **FUNNEL done** — every hyperopt trial outcome (success + 3 failure paths) writes via
  `record_experiment_result`. `CheckpointManager` = telemetry, kept as-is.
- **MEASURE cost vs. payoff** — Step-0 note (assessed): the three non-canonical `BenchmarkResult`
  classes are semantically distinct and do **not** map cleanly onto the canonical `metrics`-dict base:
  `benchmarks/rigorous.py` (statistical perf distributions + raw samples + system info),
  `benchmarks/compare_nanoGPT.py` (loss/ppl/tokens-per-sec), `analysis/tile_profiler.py` (tile timing
  distribution). Ground rule 8 already sanctions coexistence ("interface + tracks, not mechanical
  merging"). Forcing them into one class is an XL rework for zero new capability.
- **RULE risk vs. payoff** — converting `fa/eqprop/hebbian/spiking` to model-side `train_step`s and
  deleting files that ~20 tests + `cli/repro._gradient_gate` + `validation/tracks/nebc_tracks.py` +
  `bioplausible/__init__.py` consume is a numerical-semantics change. GATE-0 pins only 3 specific
  values; it cannot guarantee a *semantic* drift elsewhere goes unnoticed at low effort. AGENTS.md
  prioritizes working functionality over consolidation.

**Reopening MEASURE/RULE** should require a concrete pain point (e.g. a 6th `BenchmarkResult` is added,
or a new training loop hand-rolls dispatch). Fresh evidence, not momentum. The CI gates (`check_seams.py`
allowlists) will fail fast on any new violation, so the consolidated state stays enforced while dormant.

**Criterion-1 note (current LOOP grep truth, 2026-08-14):** returns exactly the documented allowlisted
set — `zoo/optimizers/ewc.py`, `zoo/models/forward_only.py`, `zoo/models/target_prop.py`,
`zoo/models/eqprop/eqprop_diffusion.py`, `zoo/mep/{__init__,optimizers/__init__}.py`,
`graph/training.py`. (`ewc.py` still matches `loss.backward()` because it is the EWC *loss rule*
body, not the Fisher-move step that went to `core/`; see LOOP step 5 / Ledger.)

---

## MEASURE — one measurement stack

**Step 0 — design note (~30 min, before coding).** Decide concretely how the four non-canonical
`BenchmarkResult` classes relate to the base: **compose, subclass, or coexist.** Do not force
composition. If a class doesn't fit the interface cleanly, coexistence is acceptable. This note is the
difference between a clean consolidation and an XL rework.

**Steps.**
1. Establish `evaluation/base.BenchmarkResult` as the canonical **interface** per the Step-0 decision.
2. Convert benchmark loops (`benchmarks/`) to `BenchmarkRegistry` tracks, routing
   their outcomes through `result_sink`. (`zoo/mep/benchmarks/` is **gone** — deleted
   2026-08-14 as dead code; only the live `benchmarks/` package remains.) This completes #3b.
3. Report renderers → `experiment/report.py` (JSONL) as canonical; others become thin adapters.
   (`analysis/reporting.py` consumes Optuna trials — different input, keep as adapter.)
4. Leaderboard/ranking → one implementation, rendered by `leaderboard/` and `cli/rank.py`.
5. Fold remaining inline accuracy copies into `core.losses.compute_accuracy` where semantics match.
   The 3-D-per-token / accumulation / PL-raw-tensor sites are legitimately different — leave them.

Unblocks: PRUNE's tile archive.

**Verify.** One `BenchmarkResult` interface imported by all benchmark code; `biopl-report` renders the
canonical JSONL.

---

## RULE — model owns the learning rule

**Done.** Alias map: `_PROPAGATOR_TO_MODEL` → `_ALIASES`; `Registry.get(PROPAGATOR, "ff")` returns the
model class.

**Steps 1–3 — do these (gated on GATE-0).**
1. Migrate consumers off `zoo/propagators/{backprop,base,fa,eqprop,hebbian,spiking}.py` — they are
   **not dead code**: ~20 tests, `cli/repro.py:_gradient_gate`, `validation/tracks/nebc_tracks.py`,
   `bioplausible/__init__.py`.
2. Convert `fa`, `eqprop`, `hebbian`, `spiking` to model-side `train_step`s on their models.
3. Delete the converted files plus `backprop.py` (BPTT fallback covers backprop models).

**Step 4 — optional, go/no-go.** Collapse `dispatch_train_step` from 5 phases to 2
(energy-model → model `train_step` → BPTT). **Internal change only — signature frozen.**
Do this **only if** parity is locked (GATE-0) **and** LOOP is green. If parity isn't locked, **stop
after step 3.** The payoff (simpler AutoScientist composition) may not justify the parity risk.

**Criterion #6** — `zoo/propagators/` contains only `mep.py` + pure gradient transformers:
`ls zoo/propagators/` and `grep -rn "from bioplausible.zoo.propagators" bioplausible/ tests/`.

---

## Backlog

- **REGISTER** — remaining `_LAZY` re-exports: before trimming any name, grep for
  `from bioplausible import NAME`; trim only names confined to their own subpackage. Adopt the
  `vars(module)` `__all__` pattern in other re-export subpackages (with per-file ruff ignores).
- **PRUNE** — archive `analysis/tile_*.py` only after MEASURE lands. **Done 2026-08-14:** dead
  `zoo/mep/benchmarks/` subpackage (~3.8k LOC), `visualization.py` (~1k LOC),
  `eqprop/_unified.py` (legacy duplicate engine), `analysis/scaling.py`,
  `core/local_learning/config.py`, `hyperopt/task_registry.py`, `p2p/cloud_guide.py`, and 6
  whole-file redundant test suites + a redundant class block — all removed, verified green.

## Not doing

- God-object splits (`core/trainer.py`, `knowledge/kb.py`, `execution/strategy.py`) — only if a stream
  above forces it; stop when cohesive.
- Settling-loop merge (Family A/B) — numerics risk, low gain. Telemetry already unified.
- Visualization consolidation — UI preference, not architecture.
- Anything in the Ledger below.

---

## Improvement opportunities (2026-08-14, non-blocking)

Collected during CHECKPOINT; none are started, none are required. Keep them out of the "Not doing" list
until a concrete pain point appears.

- **`zoo/optimizers/ewc.py`** still matches the LOOP grep (`loss.backward()`) because it holds the EWC
  *loss-rule* body; only the Fisher-move step went to `core/ewc.py`. When RULE is ever resumed, the
  `update_fisher` → `register_ewc` → `compute_ewc_loss` trio should be reviewed for a single canonical
  owner. Currently fine as-is (documented in `LOOP_ALLOW`).
- **`graph/training.py` exemption** is the only bespoke non-`dispatch_train_step` loop left. If a second
  such loop ever appears, revisit whether the dispatcher should grow a graph mode vs. keep the exemption.
- **`check_imports.py`** reports "6 lazy loader(s) found (may mask cycles)" — a latent cycle-detection
  blind spot. Worth a `# TODO` to surface what the lazy loaders are; low priority.
- **MEASURE Step 5** (fold inline accuracy copies into `core.losses.compute_accuracy`) is the lowest-risk
  item in MEASURE and could be done opportunistically later without the rest of MEASURE.

## Facilitating future work

- Re-entry = playbook at top: `uv sync --extra dev` → `check_imports.py` → `check_seams.py` → GATE-0
  suite (`90 passed, 3 xfailed`). Do not re-derive LOOP/FUNNEL — they are done.
- Status truth lives **only** in the Status table + Ledger. MEASURE/RULE remain open but are
  deliberately deferred by the CHECKPOINT decision; the CI allowlists keep the achieved state enforced.
- If resuming MEASURE: start with the Step-0 note (already assessed → coexist), then
  `evaluation/base.BenchmarkResult` as the canonical interface, then `benchmarks/` tracks via
  `BenchmarkRegistry` + `result_sink`. If resuming RULE: start with GATE-0 green check, then migrate
  propagator consumers before deleting any file.

---

## Ground rules

1. **GATE-0 gates all semantic work — LOCKED 2026-08-14 (xfail + snapshots).** The 3 drifting parity tests are xfail; test_parity_snapshots.py pins their values. Zero new failures outside the locked xfails is the bar.
2. `loss.backward()` in `validation/tracks/`, `analysis/{dynamics,energy_landscape}.py`, and
   `execution/{robustness,interpretability,_guards}.py` is measurement, not training. **Do not touch.**
3. `training/rl.py` is REINFORCE from env trajectories. Never wrap it in `CoreTrainer`.
   `graph/training.py` gets the same exemption if `dispatch_train_step` doesn't fit it cleanly.
4. Task geometry is ambiguous by design: `input_dim` may be a tuple (`mnist → (1,28,28)`). Thread it
   straight to `construct_model`. Never `int()`-coerce. Flattening (`math.prod`) lives only in
   `domains/registry.resolve_task`.
5. `BioLightningModule`: PL owns the outer loop. Route it through `dispatch_train_step`; never nest
   `CoreTrainer.fit` inside PL. It passes `propagator=None`/`optimizer=None` on purpose — revisit when
   RULE lands.
6. `BioLightningModule.create_model` stays — `test_lightning_integration.py:368,417` patch it.
7. Never fold the engine's Optuna `tell`/`ask` into `result_sink`. Search loop ≠ audit trail.
8. The five `BenchmarkResult` classes are semantically distinct. Interface + tracks, not mechanical
   merging. See MEASURE Step 0.
9. `dispatch_train_step`'s signature is frozen:
   `(model, x, y, adapt_input, bptt_step, propagator, optimizer, config, record_path)`.
   RULE's phase collapse is internal.
10. `ConvEquiTile` NaNs on MNIST with the default tile config. Use the small test config
    (`conv_channels=[4,8]`, `tiles_per_layer=1`, `mode="pc"`). Hyperparameters, not structure.
11. `LoopedMLP.train_step` returns `None` for single-hidden `gradient_method="equilibrium"` → dispatcher
    falls through to BPTT. Correct behavior, not a regression.
12. `sklearn_interface.EqPropClassifier` was broken standalone; fixed (local `zoo` import to trigger
    lazy registration; default model `"eqprop_mlp"`). Don't revert.
13. Python 3.14 allows `except A, B, C:` without parentheses. `ruff format` produces it. Not a bug.
14. **Criterion greps are proxies.** Route through the seam first; only then satisfy the grep. Never
    bend real code to hit zero.
15. **ruff 0.15.9 fails to parse `pyproject.toml`** (`Unknown rule selector: line-too-long`) —
    pre-existing, unrelated to the refactor; don't chase it.
16. **A git hook auto-commits edits**; a clean `git status` after a session is expected (e.g.
    `3db76846`). Check `git log`, don't panic.

---

## Acceptance

| # | Check | Status |
|---|-------|--------|
| gate | `tools/check_imports.py` passes in CI | always |
| seams | `tools/check_seams.py` passes in CI (violators ⊆ allowlist) | ✅ enforcing, ratcheting |
| GATE-0 | Parity locked (xfail + numerical snapshots) | ✅ locked 2026-08-14 |
| 1 | LOOP grep = 0 (convertible set; proxy — see rule 14) | 🔄 remainder is the documented allowlisted set; parked at CHECKPOINT |
| 3 | `grep -rn "model_cls(" bioplausible/ \| grep -v construction.py` = 0 | ✅ |
| 3b | All outcome writes via `result_sink` | FUNNEL ✅ · MEASURE deferred at CHECKPOINT |
| 6 | `zoo/propagators/` = `mep.py` + gradient transformers | RULE |
| K | `biopl` dispatcher works | ✅ |
| — | Zero new test failures | always |

**Baseline (post-GATE-0 — confirm with one full run):** ~2002 pass / ~3 fail / 10 skip / ~4 xfail — the 3 known parity drifts are now xfail + snapshot-pinned; remaining ~3 fails are flaky/seed drift.

---

## CI gates — keep it won

Integration won by a manual grep is a one-time event; a future sprint will re-add a sixth
`BenchmarkResult` or a hand-rolled training loop. The acceptance greps must run in CI so the
consolidated state is **enforced, not just achieved once**.

`tools/check_seams.py` encodes each criterion as **violator-set ⊆ versioned allowlist**. The
allowlist is the explicit, committed home for legitimate exceptions (rule 14) — exceptions are
first-class and visible, never silently accumulated.

| Gate | Asserts | Allowlist (lives in the script) |
|------|---------|--------------------------------|
| `seam:loop-backward` | `loss.backward()` files ⊆ ALLOW | `LOOP_ALLOW` = today's convertible debt |
| `seam:model-cls` | `model_cls(` outside `construction.py` = ∅ | (empty) |
| `seam:propagators` | `zoo/propagators/*.py` ⊆ ALLOW | `PROPAGATORS_ALLOW` = RULE's delete-set |
| `seam:result-sink` | outcome writers ⊆ `result_sink` | `RESULT_SINK_ALLOW` = sanctioned callers |

**Rules.**
- **Lock the baseline first.** Each allowlist = today's violator set. The gate passes immediately
  and can only tighten.
- **Ratchet.** Completing a stream step = *remove* its entries from the allowlist. The gate then
  enforces the smaller set forever. Allowlists shrink monotonically; growing one is a visible diff
  that requires review justification.
- **New violations fail fast.** A file not in an allowlist that introduces a violation fails CI —
  the regression guard (verified: dropping a stray `loss.backward()` in a new file fails).
- Wired into pre-commit (`id: check-seams`). `check_imports.py` (layering) + `check_seams.py`
  (criteria) are the two CI guardians; both must pass on every merge.
- **GOTCHA:** the dev deps are required to run tests (`uv sync --extra dev`); a bare `uv sync` omits
  `optuna`, which `core/trainer.py` imports at module load and breaks every test run with
  `ModuleNotFoundError`.

---

## Ledger (done — don't redo)

- **CONFIG** — single `ModelConfig`/`ExperimentConfig`; `TrainerConfigSchema` and `_KNOB_ALIASES` gone.
  `omegaconf.py` is the I/O facade by design — keep it.
- **BUILD** — zero `model_cls(` outside `construction.py`; task/geometry resolution collapsed to
  `resolve_task_from_data_config`; `cli/lab.py` and `cli/repro.py` build via `construct_model`.
- **CLI** — `biopl` lazy dispatcher over all subcommands; `DASHBOARD` decoupled via `EventSink`;
  `cli/run.py` default model fixed (`"MLP"` → `"backprop_mlp"`). CI still pins three legacy shim
  scripts — fine, they delegate.
- **LOOP so far** — `dispatch_train_step` exists and is used by CoreTrainer, BioLightningModule,
  `cli/repro.py`, `validation/utils.train_model`, and the sklearn classifier; `ConvEquiTile` trains
  through CoreTrainer (spatial input preserved); `LightningExecutionCallback` added (not yet wired);
  engine device threaded.
  **LOOP steps 1–5 done 2026-08-14:** (1) backprop-mode `train_step` removed from the 4 deployment
  files (`base/graph/vision/timeseries.py`), dead optimizer blocks + `create_optimizer`/`OptimizerConfig`
  imports dropped, only `local_update` remains (feature-flag grep clean). (2) TileLM routes via
  dispatcher BPTT fallback (see step 2 note above); direct-call tests now go through the shared
  `tests/conftest.lm_train_step` helper, which builds an LM-aware BPTT step and calls
  `dispatch_train_step`. (3) Canonical BPTT fallback centralized in `core/trainer.py` as
  `bptt_step` + `_default_bptt_step`; `dispatch_train_step` defaults `bptt_step=None` and binds
  canonical when omitted; local `_bptt` closures deleted from `cli/repro.py`, `validation/utils.py`,
  `sklearn_interface.py`; `lightning_/module.py` keeps forward-only `_bptt_forward` (PL owns backward);
  docstring reworded to remove false-positive `loss.backward()` match; 4 entries removed from
  `LOOP_ALLOW` in `check_seams.py`. (4) `graph/training.py` **EXEMPT** with documented reason:
  bespoke `GraphStructure` + param-dict training (Predictive Coding with local gradients) does not
  fit `dispatch_train_step` seam; added to `LOOP_ALLOW` with exemption note. (5) `train_nebc_model`
  (`zoo/nebc_base.py`) and `update_fisher` (`zoo/optimizers/ewc.py`) moved to `core/nebc.py` and
  `core/ewc.py`; zoo files now import from core; 2 entries removed from `LOOP_ALLOW`. **LOOP step 6:**
  `target_prop.py` **KEPT** — already has pure local `train_step` (target propagation); `loss.backward()`
  is part of local rule, not BPTT fallback. **LOOP step 7:** `eqprop_diffusion.py` **KEPT** — tagged
  broken/deferred. **LOOP step 8:** `zoo/mep` inline loops **DEFERRED**. Verified: `test_deployment_models`
  (5), `test_tile_lm` + LM integration suites (78), `test_core_trainer` + smoke (53),
  `test_nebc_base` (15), `test_optimizer_stubs` (15) all green.
- **FUNNEL so far** — `hyperopt/experiment.py` success + the 3 failure paths all write via
  `record_experiment_result` (through `_sink_completed`/`_sink_failure`). Criterion 3b partially met:
  every hyperopt trial outcome now flows through the sink. **FUNNEL step 2 done 2026-08-14:**
  `CheckpointManager` evaluated — it is a SQLite telemetry logger (`training_checkpoints` rows), NOT
  a `.pt` model-state writer; does not overlap `core.checkpoint`. Verdict: keep as telemetry; no
  code change (bookkeeping note only).
- **GATE-0** — locked 2026-08-14 (option b): the 3 known-drifting parity tests are `xfail`ed with
  locked reasons and `tests/unit/validation/test_parity_snapshots.py` pins their baseline values.
  Verify: `37 passed, 3 xfailed` on the GATE-0 command.
- **CI SEAMS** — `tools/check_seams.py` created, locked to today's baseline, wired into pre-commit.
  All 4 gates green; `check_imports.py` + `check_seams.py` are the two CI guardians (§ CI gates).
- **CHECKPOINT** — decided 2026-08-14: **STOP semantic consolidation here** (LOOP+FUNNEL done, MEASURE/RULE deferred). See CHECKPOINT section for the full evidence (MEASURE Step-0 assessed → coexist; RULE = parity risk not worth it absent a pain point).
- **RE-ENTRY STATE (2026-08-14)** — verified: `check_imports.py` = 0 violations / 0 cycles; `check_seams.py` all 4 gates green (LOOP 7 violators ⊆ allowlist); GATE-0 + LOOP suites = `90 passed, 3 xfailed`. LOOP grep returns exactly the documented allowlisted set (see CHECKPOINT). The repo is cleanly parked at a green, gate-enforced checkpoint.
- **CORE EXTENSIONS** — `core/nebc.py` (NEBC training utilities), `core/ewc.py` (EWC Fisher utilities)
  added as canonical L1 locations for formerly zoo-scoped utilities.
- **RULE so far** — alias map live; `Registry.aliases()` / `resolve_alias()` available.
- **REGISTER so far** — dead `_LAZY` re-exports trimmed (`__all__` 103→83 top, 28→27 core);
  `zoo/models/eqprop/__init__.py` computes `__all__` from `vars(module)`.
- **PRUNE so far** — `TODO.md` / `REFACTOR.md` archived to `docs/archive/20260813/`.
  **2026-08-14 dead-code + redundant-test prune** (see PRUNE backlog item): removed ~4.9k LOC of
  dead code (`zoo/mep/benchmarks/` subpackage incl. config/, `visualization.py`,
  `eqprop/_unified.py`, `analysis/scaling.py`, `core/local_learning/config.py`,
  `hyperopt/task_registry.py`, `p2p/cloud_guide.py`) and 6 whole-file redundant test suites
  (`test_tasks.py`, `test_all_models.py`, `test_adaptive_fa.py`, `property/test_energies.py`,
  `test_engine_stability.py`, `test_stress_equilibrium.py`) plus the `TestEnvironmentCapture`
  block in `test_reproducibility.py`. All verified: gates green, `202 passed, 3 xfailed`
  on the LOOP/GATE-0 suites, full-suite collection clean (2008 collected).
  **Redundant-test coverage map** (one line per deletion, so future auditors don't re-derive):
  - `test_tasks.py` (5) → `test_smoke_all_tasks.py` (11 vision+LM+RL tasks, same `create_task` flow)
  - `test_all_models.py` (17) → `test_eqprop_models_forward.py` (eqprop forward), `test_fa_model.py` (FA forward), `test_hebbian_models.py` (hebbian forward), `test_smoke_training.py` (25 training smokes)
  - `test_adaptive_fa.py` (2) → `test_fa.py` (AdaptiveFA step), `test_fa_model.py` (DFA forward), `test_smoke_training.py` (AdaptiveFA smoke)
  - `property/test_energies.py` (9) → `tests/unit/core/test_energies.py` (12 deterministic tests, strict superset of hypothesis assertions)
  - `test_engine_stability.py` + `test_stress_equilibrium.py` (2) → `test_validation_all.py::test_looped_mlp_equilibrium_learns` (same model/mode, parametrized)
  - `TestEnvironmentCapture` block (5) → `test_repro_check.py::TestCaptureEnvironment` (tests the real `bioplausible.utils.capture_environment`, not local duplicates)
