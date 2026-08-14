# REFACTOR — Canonical Plan (v3)

> **How to use this file.** It is written so a fresh session is productive in under two minutes: read **§0 Orient**, then jump to the **Frontier** item you're working. You should not need to read the whole file, and you should not need to reconstruct the system from history. Each work brief is self-contained.
>
> **Maintenance contract.** Status lives in exactly one place (§0.3). When you finish an item, flip its status there and move its brief to §5 Ledger. Do not append session logs — fold learnings into the relevant brief's **Gotchas** or into §6 Reference.

---

## 0 · ORIENT — read this first

### 0.1 What this is
`bioplausible` (~78k LOC, ~297 modules) is a biologically-plausible ML framework. It grew via 25+ feature sprints, so every cross-cutting concern (training, config, persistence, measurement, construction) exists in several parallel copies. **Capability is not the problem — consolidation is.**

### 0.2 The one-sentence goal
A strict dependency-layered core with **exactly one implementation of every cross-cutting concern** — a canonical **seam** for each, with all callers routed through it.

### 0.3 Status at a glance *(the only status table — keep it true)*

| ID | Work stream | Former pillar | Status | Blocked by |
|----|-------------|---------------|--------|-----------|
| **LOOP** | One training loop | A | 🔄 in progress | — |
| **FUNNEL** | One result/persistence funnel | E | ⬜ open | LOOP |
| **MEASURE** | One measurement stack | D | ⬜ open | FUNNEL |
| **RULE** | One learning-rule interface (propagator→model) | G | ⬜ open | LOOP |
| **REGISTER** | Self-registration | L | ⚠️ partial | — |
| **PRUNE** | Dead-code removal | J | ⚠️ partial | MEASURE |
| CONFIG | One config hierarchy | B | ✅ done | — |
| BUILD | One construction funnel | C | ✅ done | — |
| CLI | One CLI entry point | K | ✅ done | — |

Legend: ✅ done · 🔄 in progress · ⬜ open · ⚠️ partial · ⛔ deferred

### 0.4 Start here (next concrete action)
Work **LOOP**. The two unblocked steps from the previous revision are **done** (validation tracks → seam; engine device threading). Next: **§2.1 step 1** — delete backprop-mode `train_step` in `zoo/models/deployments/base.py:250-257` (~15 lines). Steps 4 & 6 (propagator deletion/migration) advance both LOOP #1 and RULE #6 — coordinate with RULE.

### 0.5 Baseline (what "green" means)
- **Tests:** 2002 pass / 6 fail / 10 skip / 1 xfail. The 6 fails are documented numerical/parity drifts, unrelated to this refactor. **Zero new failures is the bar.**
- **Pyright:** 0 errors (strict).
- **Ruff:** 0.16 parses config; ~2k pre-existing warnings are backlog, not blockers.

---

## 1 · THE MAP

### 1.1 Layers
**Rule:** `L_N` may import only from `L_{≤N−1}`. Enforced in CI by `tools/check_imports.py` (import-DAG gate). The current graph violates this in places; the refactor removes the violations.

```
┌─────────────────────────────────────────────────────────────────┐
│ L7  Interfaces  : CLI · deployment · sklearn · lightning        │  public API
├─────────────────────────────────────────────────────────────────┤
│ L6  Measurement : evaluation · validation · benchmarks ·        │  one BenchmarkResult,
│                   analysis · reporting · leaderboard           │  one report renderer
├─────────────────────────────────────────────────────────────────┤
│ L5  Orchestration: execution · hyperopt · autoscientist        │  adapters over runner
├─────────────────────────────────────────────────────────────────┤
│ L4  Training     : CoreTrainer (THE single train path)         │  runners become adapters
├─────────────────────────────────────────────────────────────────┤
│ L3  Data/Domains : data · domains                                │  one task abstraction
├─────────────────────────────────────────────────────────────────┤
│ L2  Zoo          : models · propagators · optimizers · mep      │  registered components
├─────────────────────────────────────────────────────────────────┤
│ L1  Core         : registry · construction · config ·           │  zero upward imports
│                    checkpoint · metrics · result_sink ·         │
│                    tile substrate · local_learning              │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Where things live (directory guide)
| Dir(s) | Layer | Contains |
|--------|-------|----------|
| `core/` | L1 | registry, construction, config, checkpoint, metrics, result_sink, trainer, tile substrate, local_learning |
| `zoo/` | L2 | models, propagators, optimizers, mep |
| `data/`, `domains/` | L3 | datasets, task abstraction |
| `execution/`, `hyperopt/`, `autoscientist/` | L5 | orchestration, HPO |
| `evaluation/`, `validation/`, `benchmarks/`, `analysis/`, `leaderboard/` | L6 | measurement, reporting |
| `cli/`, `deployment/`, `lightning_/`, `sklearn_interface.py` | L7 | interfaces |

### 1.3 The seams *(the central mental model)*
A **seam** is the single canonical implementation of a concern. All work in this refactor is "route callers through the seam." Learn these and you know the system.

| Concern | Canonical seam | Layer | Status |
|---------|----------------|-------|--------|
| Config tree | `config/unified.py` (frozen dataclasses) | L1 | ✅ |
| Config I/O | `config/omegaconf.py` facade → `to_internal()` | L1 | ✅ |
| Model construction | `core/construction.construct_model` | L1 | ✅ |
| Task/data resolution | `domains/registry.resolve_task_from_data_config` | L3 | ✅ |
| **Train-step dispatch** | `core/trainer.dispatch_train_step` | L4 | 🔄 |
| **Outcome persistence** | `experiment/result_sink.record_experiment_result` | L1 | ⬜ |
| **Measurement result** | `evaluation/base.BenchmarkResult` (interface) | L6 | ⬜ |
| **Learning rule** | `model.train_step` | L2 | ⬜ |
| CLI entry | `biopl` dispatcher (`cli/__main__.py`) | L7 | ✅ |

---

## 2 · THE FRONTIER — active work, in priority order

Each brief is self-contained: goal → seam → state → next steps → gotchas → criterion → verify.

---

### 2.1 · LOOP — one training loop *(former Pillar A · XL · 🔄 in progress)*

**Goal.** `CoreTrainer` is the only training loop. The other 6 run stacks (`TrialRunner`, `Verifier`, `StaircaseRunner`, `BenchmarkRunner`, `BioLightningModule`, `graph/training.py`) become thin adapters over `fit()`/`train_epoch()`.

**Seam.** `core/trainer.dispatch_train_step` — a module-level pure function owning the 5-phase routing (energy-model → learning-rule propagator → model `train_step` → learning-rule optimizer → BPTT fallback). Both `CoreTrainer._train_step` and `BioLightningModule.training_step` route through it; PL stays the outer loop. **All future loop conversions must route through this seam, never hand-roll dispatch.**

**Win.** ~2,500 lines removed; one place for bug fixes/features (mixed precision, compile, energy tracking).

**Done so far.**
- `dispatch_train_step` shared seam exists and is used by both CoreTrainer and BioLightningModule.
- Deployment-model migration: `ConvEquiTile` trains through CoreTrainer (spatial input preserved via `input_format="spatial"`); `LightningExecutionCallback` added.
- `cli/repro.py:_train_one_epoch` now uses `dispatch_train_step`. **Note:** this did *not* advance criterion #1 — its `_bptt` fallback `loss.backward()` remains (non-`train_step` families like `fa` legitimately use BPTT).
- **Validation tracks delegate to the seam.** `validation/utils.train_model` now routes each epoch through `dispatch_train_step` (BPTT fallback closure retained; identical behavior). All declarative `track_*` specs call `train_model` unchanged, so execution now flows through the seam without touching the ~40 specs. This mirrors the `cli/repro.py` pattern — does *not* reduce the criterion-#1 file count (the `_bptt_step` closure keeps `loss.backward()`).
- **Engine device threaded.** `ExecutionEngine.__init__` gained a `device: str = "auto"` param (resolved once via `get_device()` → `self.device: str`); `_get_train_loader` and `_get_val_loader` now pass `device=self.device` instead of hardcoded `"cpu"`. Default `"auto"` preserves all existing callers.
- **`sklearn_interface.EqPropClassifier._train_step` routed through the seam.** It hand-rolled the exact dispatch logic (zero-grad → `model.train_step` → BPTT fallback). Now delegates to `dispatch_train_step`; behavior preserved (verified both paths: `backprop_mlp` → BPTT fallback `{"loss","accuracy"}`; `eqprop_mlp` → model `train_step` metrics). BPTT fallback closure still holds `loss.backward()` (counts toward criterion #1's ~40).

**The hard part.** Criterion #1 requires removing `loss.backward()` from **~40 files** outside `core/`+`training_mixin`. These are **not mechanical duplicates**—each is a distinct algorithm with legitimate reasons for owning its backward pass. The work is **architectural rerouting** through `dispatch_train_step`, not deletion.

**Categorized hit list (verified by grep):**

| Category | Files | Strategy |
|----------|-------|----------|
| **Model-side `train_step` with `loss.backward()` (convert to pure local update OR delete `train_step` to fall through to BPTT fallback)** | | |
| EquiTile deployments (backprop mode) | `zoo/models/deployments/base.py:250-257` | Delete `train_step` for `mode="backprop"`; falls through to BPTT |
| TileLM (BPTT mode) | `zoo/models/tile_lm.py:228-246` | Delete `train_step`; it's `mode="backprop"` anyway |
| Difference Target Prop | `zoo/models/target_prop.py:111-175` | Implement pure target-prop (no autograd) in `train_step` OR delete `train_step` + use model-side propagator (per RULE) |
| ForwardForwardNet classifier head | `zoo/models/forward_only.py:161-163` | Classifier uses `supervised_step`→backward; move to separate optimizer or keep as legitimate local update |
| EqPropDiffusion (tagged **broken**) | `zoo/models/eqprop/eqprop_diffusion.py:104-129` | Low priority—tagged broken; delete or fix later |
| **Propagators (per RULE §2.4: convert to model-side `train_step` OR delete—let BPTT fallback handle backprop)** | | |
| Backprop propagator | `zoo/propagators/backprop.py:65` | **Delete**—backprop models fall through to BPTT fallback |
| FA / EqProp / Hebbian / Spiking propagators | `zoo/propagators/{fa,eqprop,hebbian,spiking}.py` | Convert to model-side `train_step` on respective models, then delete propagator |
| **Graph training (used by `predictive_coding.py:FabricPCGraphPCN.train_step`)** | | |
| `train_backprop` / `train_pcn` | `graph/training.py:145,346` | Delegate to `CoreTrainer` instead of hand-rolled loops |
| **Diagnostic / measurement loops (EXPLICITLY PERMITTED — do NOT convert; they measure specific semantics)** | | |
| Validation tracks (6 files) | `validation/tracks/{scaling,hardware,application,tradeoff,nebc,architecture_comparison}_tracks.py` | **KEEP**—gradient-flow/Lipschitz, memory-geometry, Fisher, EWC, thermal-noise, plain-BPTT-intent, forward-only smoke checks |
| Analysis diagnostics | `analysis/dynamics.py`, `analysis/energy_landscape.py` | **KEEP**—measurement tools, not training loops |
| Adversarial/interpretability | `execution/{robustness,interpretability,_guards}.py` | **KEEP**—gradient analysis (FGSM/PGD/saliency/IG), not training |
| Benchmarks | `benchmarks/{rigorous,efficiency_analysis}.py` | Convert to `BenchmarkRegistry` tracks (**blocked on MEASURE**) |
| MEP benchmarks (7 files) | `zoo/mep/benchmarks/*.py` | Convert to `BenchmarkRegistry` tracks (**blocked on MEASURE**) |
| **Utilities (move to `core/`)** | | |
| `train_nebc_model` | `zoo/nebc_base.py:109` | Move to `core/`; used by tests |
| `update_fisher` | `zoo/optimizers/ewc.py:68` | Move to `core/` or accept pre-computed gradients |
| **Already routed / not applicable** | | |
| `sklearn_interface.py` | Fixed—routes through seam; BPTT fallback in `core/` | **DONE** |
| `lightning_/module.py` | Uses `dispatch_train_step` | **DONE** |
| `training/rl.py` | RL (REINFORCE from env trajectories) | **NOT CONVERTING**—architecturally inappropriate |

**Acceptance criterion #1.** 0 `loss.backward()` in the *convertible* set (i.e. excluding explicitly-permitted and blocked files):

```bash
grep -rln "loss.backward()" bioplausible/ \
  | grep -v core/ | grep -v training_mixin \
  | grep -v "validation/tracks/" \
  | grep -v "analysis/" \
  | grep -v "execution/robustness" \
  | grep -v "execution/interpretability" \
  | grep -v "execution/_guards" \
  | grep -v "benchmarks/" \
  | grep -v "zoo/mep/benchmarks/" \
  | grep -v "training/rl.py" \
  → 0 files
```

(Currently ~40 in the convertible set. The `dispatch_train_step` BPTT fallback in `core/trainer.py:_bptt_step` is **allowed**—it's in `core/`. MEASURE-blocked files in `benchmarks/` and `zoo/mep/benchmarks/` are tracked separately; `training/rl.py` is not converting.)

**Verify.** `pytest tests/unit/core/test_core_trainer.py tests/integration/test_smoke_training.py tests/unit/core/test_deployment_models.py`; run the grep above.

**Gotchas (item-specific).**
- **`validation/utils.py` line-count / module identity:** the module is the shared training executor for all validation tracks. Keep it as the single place track specs call — do not duplicate its seam-routing in each track. The `X`/`y` uppercase arg names and `100.0` magic comparisons are pre-existing ruff warnings (backlog, not blockers).
- **LoopedMLP parity path:** `LoopedMLP.train_step` returns `None` for single-hidden-layer `gradient_method="equilibrium"` models, so `dispatch_train_step` falls through to BPTT for them — identical to the old hand-rolled loop. Multi-layer/contrastive/`kernel` models route through the model's own `train_step`. This is the correct seam behavior, not a regression.
- **`test_backprop_parity[eqprop_mlp]` and `[directed_ep]` are pre-existing parity drifts** (fail identically on clean baseline; verified via `git stash`). Do not chase them as LOOP regressions.
- `BioLightningModule.create_model` was kept because `tests/integration/test_lightning_integration.py:368,417` patch `bioplausible.lightning_.module.create_model`. Deleting it requires updating those tests first.
- `BioLightningModule` passes `propagator=None`/`optimizer=None` to the dispatcher and steps its own optimizer externally — this suppresses Phase-4 so its bio-optimizers keep `model.train_step → opt.step()` semantics. Re-evaluate once RULE lands (Phase-4 is their intended home).
- The EnergyModel dispatch branch is guarded by `config is not None`: CoreTrainer enables `_make_ebm_trainer`; BioLightningModule (no config) lets EnergyModels fall through to model `train_step`. Give the module a minimal config facade to unlock the energy path if needed.
- `LightningExecutionCallback` (`execution/callbacks.py`) is infrastructure, **not wired** — no production consumer yet. It's the bridge any remaining PL-logging work should consume.
- `ConvEquiTile` default tile config (`neurons_per_tile=64`, `tiles_per_layer=4`, `mode="backprop"`) diverges on MNIST (NaN); the small test config (`conv_channels=[4,8]`, `tiles_per_layer=1`, `mode="pc"`) trains stably — a hyperparameter issue, not structural. Keep the small config for tests. Other deployment models (Graph/TimeSeries/RL EquiTile) use scalar `input_dim` and work unchanged.
- `dispatch_train_step` is typed `dict[str, object]` by design: PL's automatic path must return a *tensor* loss; CoreTrainer's paths yield floats (cast to `dict[str, float]`). New callers must pick the right typing.
- **Engine device threading pattern** (`engine.py`) mirrors `hyperopt/experiment.py:_select_device` and `core/trainer.py:_resolve_device` — all use `core/utils/device.get_device()`. The engine stores `self.device: str` and passes it to `resolve_task_from_data_config`.
- **`sklearn_interface.EqPropClassifier` fixed (was broken standalone).** Two latent defects resolved: (a) `_initialize` now imports `bioplausible.zoo` (local, `# ruff: ignore[unused-import]`) to trigger the deliberately-lazy component registration — previously `Registry.get(...)` raised `Unknown category`; (b) default `model_name` changed from the unregistered `"EqProp MLP"` to registered `"eqprop_mlp"` (backwards-compat NONE per AGENTS.md). `fit()`/`partial_fit()` now run end-to-end through the `_train_step` seam (verified: `backprop_mlp` 30-epoch → 98% train acc on synthetic). No tests reference this class, so the fix carries zero new-failure risk.
- **Validation track inline loops are EXPLICITLY PERMITTED.** The `loss.backward()` calls in `validation/tracks/{scaling,hardware,application,tradeoff,nebc,architecture_comparison}_tracks.py` and `analysis/{dynamics,energy_landscape}.py` measure specific semantics (Lipschitz, memory geometry, Fisher, EWC, thermal noise, plain BPTT cost, forward-only smoke). **Converting them changes what they measure. Do not touch.**
- **Execution gradient-analysis tools are EXPLICITLY PERMITTED.** `execution/{robustness,interpretability,_guards}.py` use `loss.backward()` for FGSM/PGD attacks, saliency maps, integrated gradients, and gradient health checks — not for training. Keep as-is.
- **RL training is NOT CONVERTING.** `training/rl.py` uses REINFORCE from environment trajectories (no DataLoader). CoreTrainer adapter is architecturally inappropriate.

---

### 2.2 · FUNNEL — one result & persistence funnel *(former Pillar E · M · ⬜ open)*

**Goal.** `result_sink.record_experiment_result` is the **only** writer of trial outcomes. The 5 backends (Optuna SQLite, HyperoptStorage, JSONL Report, KB, execution_state.db) become private implementation details of the sink.

**Seam.** `experiment/result_sink.record_experiment_result`. One artifact loader already exists: `core/checkpoint.load_checkpoint` + `find_trial_artifact(trial_id)`. Ad-hoc saves → `CheckpointMixin`/`core.checkpoint`. Evaluate `CheckpointManager` (`execution/_lifecycle.py`) against `core.checkpoint`.

**Win.** ~700 lines removed; split-brain audit trails eliminated.

**Risk assessment — do NOT do this mechanically.** The `execution/engine.py` Optuna `study.tell/ask` (lines 558, 597, 639) and `state.failure_tracker.log_failure` (451) are the engine's **online HPO loop itself**, not outcome recording — folding them into `record_experiment_result` would conflate the search loop with the KB audit trail. `result_sink` already owns KB + FailureTracker and is called from hyperopt, validation tracks, trainer, and probe. **Treat the remaining unification as architectural, not mechanical.**

**Audit (done — split-brain map for the next session).** `record_experiment_result` is already called from `hyperopt/experiment.py:620`, `validation/tracks/hardware_tracks.py:56`, `core/trainer.py:1698`, `experiment/probe.py:422`. Remaining out-of-sink writers, classified:
- **Genuine outcome-recording (FUNNEL candidates — route through the sink):** `hyperopt/experiment.py:523,541,564` write per-trial `FailureTracker.log_failure` (training_failed/timeout/exception). These ARE trial outcomes duplicating `record_experiment_result(status="failed")` — NOT the engine's online HPO loop. Consolidating requires the sink's failure path to capture `failure_type`, `trial_id`, `stack_trace` (it already does — `_record_failure` maps status→type and reads `extra["tier"]`/`extra["error"]`).
- **Semantically-distinct KB writes (do NOT fold into the outcome sink):** `hyperopt/search_space.py:599` records *rule-surface* validator entries (phantom detection); `evaluation/cross_domain.py:296` records *benchmark* entries — different knowledge types than experiment outcomes; folding needs schema extension (MEASURE-scope).
- **Online HPO loop (do NOT fold):** `execution/engine.py:454` (failure_tracker) + `558,597,639` (Optuna tell/ask), per risk assessment.

**Acceptance criterion #3.** `record_experiment_result` called by execution, hyperopt, validation, mep-benchmarks; all five backends written only from `result_sink`.

**Verify.** `grep -rn "record_experiment_result" bioplausible/`; confirm no backend is written from outside the sink.

---

### 2.3 · MEASURE — one measurement & reporting stack *(former Pillar D · XL · ⬜ open, depends on FUNNEL)*

**Goal.** Collapse parallel measurement ecosystems into one canonical `evaluation` package.

**Seam.** `evaluation/base.BenchmarkResult` (interface) + one canonical JSONL renderer (`experiment/report.py`).

**Win.** ~4,000 lines removed; findings share schema/CIs/renderers. **Unblocks** LOOP step 2 (MEP benchmarks) and PRUNE (tile archive).

**Scope.**
- `BenchmarkResult` ×5 (`evaluation/base.py`, `rigorous.py`, `compare_nanoGPT.py`, `tile_profiler.py`, `mep/runner.py`) — **do NOT mechanically merge** (semantically distinct). Establish `evaluation/base.BenchmarkResult` as the canonical *interface*; the others become Tracks/composites over it.
- Report renderers ×5 → one canonical JSONL renderer; others become thin adapters.
- Benchmark loops → registry-driven `BenchmarkRegistry` tracks (declarative, not new loops).
- Metrics: `core.losses.compute_accuracy` is canonical; fold remaining inline copies (only legitimately-different sites remain: 3-D per-token, accumulation, PL raw tensors).
- Leaderboard/ranking ×3 → one implementation in `leaderboard/` + `cli/rank.py`.

**Risk assessment.** `experiment/reporting.render_report` (JSONL) is already the canonical `biopl-report` renderer; `analysis/reporting.generate_experiment_report` consumes Optuna trials (different input, not a duplicate). The 5 `BenchmarkResult`s are semantically distinct by design.

**Verify.** One `BenchmarkResult` interface imported everywhere; `biopl-report` renders the canonical JSONL.

---

### 2.4 · RULE — one learning-rule interface *(former Pillar G · M · ⬜ open, depends on LOOP)*

**Goal.** The model owns the learning rule: one `train_step(x, y) -> dict` per algorithm. `ComponentCategory.PROPAGATOR` shrinks to pure gradient transformers (Muon, spectral norm, EWC).

**Seam.** `model.train_step`.

**Win.** ~800 lines removed; one interface; AutoScientist composition simplifies.

**Done so far.** Alias map complete (`_PROPAGATOR_TO_MODEL` → `_ALIASES`; `Registry.get(PROPAGATOR, "ff")` returns the model class).

**Remaining.**
- Collapse `CoreTrainer._train_step` (via `dispatch_train_step`) 5→2 phases: `energy-model` → `model.train_step` → `BPTT`. Delete phases 2 & 4 (explicit propagator / learning-rule optimizer).
- Convert `zoo/propagators/{eqprop,fa,hebbian,backprop,spiking}.py` to model-side `train_step`s, or delete.

**Risk assessment (blocking).** `zoo/propagators/{backprop,base,fa,eqprop,hebbian,spiking}.py` are **NOT dead code** — heavily imported by ~20 tests, `cli/repro.py:_gradient_gate`, `validation/tracks/nebc_tracks.py`, and `bioplausible/__init__.py`. The "or delete" path is unavailable without first migrating those consumers to model-side `train_step` (risky).

**Acceptance criterion #6.** `zoo/propagators/` contains only `mep.py` and pure-gradient-transform submodules.

**Verify.** `ls zoo/propagators/`; `grep -rn "from bioplausible.zoo.propagators" bioplausible/ tests/`.

---

## 3 · THE BACKLOG — low priority

### 3.1 · REGISTER — self-registration *(former Pillar L · M · ⬜ open, low)*
Done: `zoo/models/eqprop/__init__.py` auto-computes `__all__` from `vars(module)`; registry has `aliases()` + `resolve_alias()`.
Progress: **trimmed dead `_LAZY` re-exports** — removed **21** from `bioplausible/__init__.py`/`core/__init__.py` (`.py`: `zoo_models/optimizers/propagators/sparsity`; dangling `EqPropTrainer`; and subpackage-only re-exports `LLMHypothesisGenerator`, `DEFAULT_CONFIGS`, `BenchmarkRegistry`, `EvaluatorBase`, `run_cross_domain_benchmark`, `DEFAULT_KB`, `BioPredictionWriter`, `BenchmarkResult`, `BenchmarkSuiteConfig`, `BenchmarkSuiteResult`, `CrossDomainBenchmarkSuite`, `MetricSuite`, `evaluate_model_on_task`, `get_benchmark`, `list_benchmarks`, `create_knowledge_base`). **`__all__` 103→83 (top) and 28→27 (core).** Each removed name is referenced only within its own subpackage `__init__` + defining module (+ tests) — never via `from bioplausible import X` (verified by grep). `BenchmarkResult` (the §1.3 MEASURE seam) remains reachable as `evaluation.base.BenchmarkResult` — only the redundant top-level re-export was dropped. No `from bioplausible import *` consumers; 470 tests green across evaluation/cross-domain/knowledge/result-sink/cli/analysis/zoo/registry/refactor.
**Method:** for each `_LAZY` name, check `grep -rln "\bNAME\b"` across `bioplausible/`+`tests/`; if references are confined to the name's own subpackage (no `from bioplausible import NAME` / `bioplausible.NAME`), the top-level re-export is dead. **Do not trim** names used via `from bioplausible import` (e.g. `CoreTrainer`, `smep`, `muon_backprop`, `smep_fast`, `zoo`) — verified those are the only top-level imports in tests.
Remaining: other leaf re-export subpackages adopt the `vars(module)` pattern with per-file `ruff` ignores. Any remaining `_LAZY` names (e.g. `KnowledgeBase`, `KnowledgeEntry`) must be checked for `from bioplausible import` usage before trimming.
**Win:** adding a model/rule = one registration decorator, nothing else to touch.

### 3.2 · PRUNE — dead-code removal *(former Pillar J · S · ⚠️ partial)*
Done: `TODO.md` + `REFACTOR.md` archived to `docs/archive/20260813/`.
Remaining: `analysis/tile_*.py` legacy systems are superseded by `evaluation/` + `mep/benchmarks`. **Do NOT archive until MEASURE lands.**

---

## 4 · DEFERRED — explicitly not now ⛔
- **God-object decomposition (O):** `core/trainer.py`, `knowledge/kb.py`, `execution/strategy.py` — split only when LOOP/FUNNEL/MEASURE touch them; cap effort; stop when cohesive.
- **Settling loop merge (I):** Family A/B convergence loops — high numerical risk, low gain. Telemetry unification already done.
- **Visualization stack consolidation:** 4 stacks — UI preference, not an architectural flaw.
- **Micro-consolidation remainder (M):** ~12 inline accuracy folds (3-D/accumulation/PL) are legitimately different; `count_parameters` + seeding already done.

---

## 5 · THE LEDGER — completed work (don't redo)

### CONFIG — one config hierarchy *(former Pillar B · XL · ✅)*
Single `ModelConfig`, single `ExperimentConfig`; `TrainerConfigSchema` and `_KNOB_ALIASES` removed. `omegaconf.py` resolved as the I/O-boundary facade (mutable OmegaConf YAML types with `to_internal()` seams into the frozen `unified.py` tree) — **keep it, do not re-delete.**

### BUILD — one construction funnel *(former Pillar C · M · ✅)*
`grep -rn "model_cls(" bioplausible/ | grep -v construction.py` → 0. Task/geometry resolution collapsed onto `domains/registry.resolve_task_from_data_config`. `cli/lab.py` and `cli/repro.py` construct only via `construct_model`. Tail folded into LOOP.

### CLI — one entry point *(former Pillar K · M · ✅)*
`biopl` lazy dispatcher over `run | report | parity | repro | hpo | audit | frontier | rank | lab`; `DASHBOARD` global decoupled via `EventSink` (`execution/events.py`); 14 dispatch tests green.
*Optional add-ons:* adopt `biopl` in CI (`.github/workflows/ci.yml:31,33,59` still references `biopl-registry-audit`, `biopl-repro-check`, `eqprop-verify` — they're thin shims now; update CI/docs or keep shims). **Fixed:** `cli/run.py:1422` `core-train --model` default changed `"MLP"` → `"backprop_mlp"` (the plan note said `cli/lab.py`, but that subcommand has `--model required=True`; the unregistered default was in `run.py`). Verified `backprop_mlp`+`mnist` fits via CoreTrainer; `"MLP"` raised `Unknown model`. **Dormant, not a bug:** `config/defaults.py` `"model": {"name": "MLP", ...}` entries in `DEFAULT_CONFIGS` have **no consumers** (grep of `get_named_config`/`get_default_config`/`DEFAULT_CONFIGS` outside `config/` returns nothing) — nothing constructs a model from them today; leave unless a future consumer surfaces.

---

## 6 · REFERENCE

### 6.1 Global invariants (always true)
- **Task geometry is ambiguous by design.** `TaskProtocol.input_dim` is typed `int | None` but concrete tasks return *tuples* (e.g. `mnist → (1,28,28)`). The resolution seam (`resolve_task_from_data_config`) must thread geometry **straight through** to `construct_model` (matching `_build_runconfig_model`) — never `int()`-coerce it. Flattening (`math.prod`) lives only in `domains/registry.resolve_task` (the scheduler's geometry view).
- **`_create_model` needs no `int()` coercion.** `_setup_data` seeds `model_kwargs["input_dim"]` from the task; `_create_model` uses a `None` check that preserves tuples. Any caller hand-building `TrainerConfig` must pass a task name (so geometry resolves) or include `input_dim`/`output_dim` in `model_kwargs`.
- **`dispatch_train_step` is the single train-step seam.** New loops must route through it.

### 6.2 Environment / toolchain gotchas
- **`python -m bioplausible.cli <cmd>` shows the wrong `prog`** ("python3 -m bioplausible.cli") vs the installed `biopl` script ("biopl rank"). Cosmetic runpy vs. entry-script difference; verify via `uv run biopl ...`, not `python -m`.
- **argparse `--help` raises `SystemExit(0)`** before the adapter body runs — a dispatcher calling an adapter `main` directly must catch `SystemExit` and remap `exc.code`, or the CLI exit status is wrong under the console script.
- **Python 3.14 allows `except A, B, C:`** (tuple-of-exceptions form) without parentheses — old-style clauses are valid; `ruff format` strips redundant parens back to it. Do not "fix" them.

### 6.3 Acceptance checklist
| # | Criterion | Status |
|---|-----------|--------|
| gate | Import-DAG checker passes in CI | required always |
| 1 | `loss.backward()` outside `core/`+`training_mixin` = 0 (convertible set only; see §2.1 for exclusions) | ⬜ ~40 files (LOOP) |
| 3 | `model_cls(` outside construction = 0 | ✅ |
| 3b | No split-brain persistence (all writes via `result_sink`) | ⬜ (FUNNEL) |
| 6 | `zoo/propagators/` = `mep.py` + gradient transformers only | ⬜ (RULE) |
| K | `biopl` dispatcher works | ✅ |
| — | Zero new test failures beyond the 6 pre-existing drifts | required always |

---

## Why this structure (design notes)

I reorganized around **how a session actually works**, not around history. Concretely:

1. **Killed the pillar letters.** `A–O` are opaque until read. Work streams are now named by their outcome (`LOOP`, `FUNNEL`, `MEASURE`, `RULE`, `REGISTER`, `PRUNE`) with a crosswalk to the old pillar letters so nothing referencing "Pillar A" is lost.
2. **Introduced "seams" as the central mental model** (§1.3). The fastest way to know this system is to know its ~9 canonical functions. That table *is* the orientation.
3. **Single source of status** (§0.3). The old file repeated status in a table, in prose, in acceptance criteria, and in session logs — a drift hazard. Now there's one table; the maintenance contract says to keep it true.
4. **Self-contained work briefs.** Each Frontier item carries its own goal, seam, state, next steps, gotchas, criterion, and verify command — so a session editing `validation/tracks` reads only §2.1, not the whole file.
5. **Deleted session logs.** Their useful content was folded into brief **Gotchas** (item-specific) or §6 Reference (global). Logs are history; they forced every session to skim past the past.
6. **Explicit "Start here"** (§0.4) so no session burns cycles deciding what to do next.

