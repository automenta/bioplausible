# REFACTOR2: Remaining Work — Strategic Consolidation Plan

## Core Philosophy
The codebase (~78k LOC, ~297 modules) has 7 parallel training stacks, 4 config hierarchies, 5 `BenchmarkResult` classes, and 5 persistence layers. **Capability is not the problem — consolidation is.** Target: a strict dependency-layered core with exactly one implementation of every cross-cutting concern.

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

**Layering rule:** L_N may import from L_{≤N−1} only. Enforced by Pillar N gate (`tools/check_imports.py`).

---

## Remaining Pillars (Ordered by Value/Effort)

### 1. B — Single Config Hierarchy (XL, High risk, blocks A/C/E)  ✅ COMPLETE (verified 2026-08-13)
**Problem:** 4+ duplicate hierarchies, same-named classes.
- `ModelConfig` ×2 (`unified.py`, `omegaconf.py` renamed)
- `ExperimentConfig` ×2 (`unified.py`, `omegaconf.py` renamed)
- `TrainerConfigSchema` (Pydantic, zero prod consumers) — delete or auto-generate via `TypeAdapter`
- `config/omegaconf.py` mirror — delete; keep `unified.py` as single I/O pair
- `_KNOB_ALIASES` in `construction.py` — shrink to zero once all sites emit canonical names

**Verified:** `grep "class ModelConfig"` == 1 (`config/unified.py:123`); `ExperimentConfig` == 1
(`unified.py:328`); `TrainerConfigSchema` and `_KNOB_ALIASES` no longer exist. `omegaconf.py` was
resolved into the intended *I/O-boundary facade* design (mutable OmegaConf YAML document types with
`to_internal()` seams into the frozen `unified.py` tree) rather than deletion — this is the correct
"single I/O pair" outcome and should be kept, not re-deleted.

**Target:** One compositional tree in `config/unified.py`:
```python
@dataclass(frozen=True, slots=True)
class ModelConfig: ...
@dataclass(frozen=True, slots=True)
class OptimizerConfig: ...
@dataclass(frozen=True, slots=True)
class DataConfig: ...
@dataclass(frozen=True, slots=True)
class TrainLoopConfig: ...
@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    model: ModelConfig
    optimizer: OptimizerConfig
    data: DataConfig
    train_loop: TrainLoopConfig
    hardware: HardwareConfig
    seed: int
    tags: tuple[str, ...]
```

**Win:** ~1,200 lines removed; drift bugs eliminated; `phantom_knobs` provably exhaustive.

---

### 2. C — Single Construction Funnel (M, Medium, unblocks A)
**Status:** Model instantiation funnel essentially complete. **Remaining:**
- **Task/geometry resolution collapse:** `create_task`/`resolve_task`/`trainer._setup_data`/`engine._get_train_loader` → single `DataConfig → DomainTask` resolution in `domains/registry`.
- **Acceptance criterion #3:** `grep -rn "model_cls(" bioplausible/ | grep -v construction.py` → zero instantiation sites (only `construct_model` calls and `.build` for tile/deployment).

**Progress (2026-08-13):**
- **Acceptance criterion #3 SATISFIED.** The last two direct-instantiation sites now route through
  the canonical `construct_model` funnel:
  - `cli/lab.py:44` — was `model_cls(input_dim=..., output_dim=...)`; now `construct_model(...)` with
    `{"hidden_dim": 64, "num_layers": 2}` defaults. Also fixed a latent Python-2-style
    `except RuntimeError, ValueError, TypeError:` clause (valid syntax in 3.14, left as-is) and the
    import ordering. Verified end-to-end: `inspect_model` builds + forward-passes `backprop_mlp` on
    `xor` (scalar input) and `mnist` (spatial `(1,28,28)` tuple input). Spatial tuple `input_dim`
    must be passed **straight through** (do NOT `int()` it) — matches `_build_runconfig_model`
    (`core/trainer.py:1711`).
  - `cli/repro.py:125` (pre-existing uncommitted) — was `model_cls.build(...)`; now `construct_model`.
- **Pre-existing uncommitted Pillar C work in the tree (left intact, verified green):**
  `core/construction.py` adds `_is_deployment_model()` and routes the deployment `BioModel` family
  (`conv_equitile`/`rl_equitile`/etc.) through the substrate `build` branch; `repro.py` consumes it;
  `tests/unit/experiment/test_config_knobs.py` opts `conv_equitile` out of the depth audit (its depth
  map is offset by 2). `pytest tests/unit/experiment/test_config_knobs.py` → 13 passed.

**Win:** ~600 lines removed; one error message for "unknown task/model"; lossless checkpoint round-trips.

---

### 3. A — Single Training Path (XL, High, depends on B/C)
**Problem:** 7 parallel run stacks (CoreTrainer, TrialRunner, Verifier, StaircaseRunner, BenchmarkRunner, BioLightningModule, graph/training.py). All re-implement: `train_step` dispatch, model instantiation, device/seed, metrics, checkpointing.

**Target:** `CoreTrainer` is the ONLY training loop. Everything else is a thin adapter:
- `TrialRunner`, `Verifier`, `StaircaseRunner`, `BenchmarkRunner` → construct `CoreTrainer`, call `fit()`/`train_epoch()`.
- `BioLightningModule` → callback/plugin on `CoreTrainer` (PL-specific needs = `ExecutionCallback` implementations). Delete its `create_model` helper.
- `graph/training.py` custom PCN/BPTT loops — leave alone unless trivially adaptable.
- Validation tracks keep declarative `track_*` spec but delegate execution to shared runner.

**Acceptance criterion #1:** `grep -rl "loss.backward()" bioplausible/` outside `core/` & `training_mixin` returns nothing.

**Win:** ~2,500 lines removed; one place for bug fixes/features.

---

### 4. E — Single Result & Persistence Funnel (M, Medium, depends on A)
**Problem:** 5 write paths for outcomes (Optuna SQLite, HyperoptStorage, JSONL Report, KB, execution_state.db). `result_sink.record_experiment_result` is the canonical funnel but engine, verifier, mep-benchmarks write around it.

**Target:** `result_sink` is the ONLY writer. All backends become private implementation details.
- `record_experiment_result` owns: Optuna `tell`, `hyperopt_logs`, JSONL `Report`, KB upsert, failure log.
- One artifact loader: `core/checkpoint.load_checkpoint` + `find_trial_artifact(trial_id)` (already done).
- Ad-hoc checkpoint saves → `CheckpointMixin`/`core.checkpoint` calls.
- Evaluate `CheckpointManager` in `execution/_lifecycle.py` against `core.checkpoint`.

**Acceptance criterion #3:** `record_experiment_result` called by execution, hyperopt, validation, mep-benchmarks; all five backends written only from `result_sink`.

**Win:** ~700 lines removed; split-brain audit trails eliminated.

---

### 5. D — Single Measurement & Reporting Stack (XL, High, depends on E)
**Problem:** Parallel measurement ecosystems.
- `BenchmarkResult` ×5 (`evaluation/base.py`, `rigorous.py`, `compare_nanoGPT.py`, `tile_profiler.py`, `mep/runner.py`) — **do not mechanically merge** (semantically distinct). Establish `evaluation/base.BenchmarkResult` as canonical interface; others become Tracks/composites.
- Report renderers ×5 → one canonical JSONL renderer (`experiment/report.py`); others become thin adapters.
- Benchmark loops → registry-driven `BenchmarkRegistry` tracks (declarative, not new loops).
- Metrics: `core.losses.compute_accuracy` is canonical; fold remaining inline copies (only legitimately different sites remain: 3-D per-token, accumulation, PL raw tensors).
- Leaderboard/ranking ×3 → one implementation in `leaderboard/` + `cli/rank.py`.

**Win:** ~4,000 lines removed; findings share schema/CIs/renderers.

---

### 6. G — Propagator/Model Unification (M, Medium, depends on A/F)
**Status:** Alias map done (`_PROPAGATOR_TO_MODEL` → `_ALIASES`; `Registry.get(PROPAGATOR, "ff")` returns model class). **Remaining:**
- `CoreTrainer._train_step` 5→2 phase collapse: `energy-model` (structural `match`) → `model.train_step` → `BPTT`. Delete phases 2 & 4 (explicit propagator/learning-rule optimizer).
- Convert `zoo/propagators/{eqprop,fa,hebbian,backprop,spiking}.py` to model-side `train_step`s or delete.
- `ComponentCategory.PROPAGATOR` shrinks to pure gradient transformers (Muon, spectral norm, EWC).

**Acceptance criterion #6:** `zoo/propagators/` contains only `mep.py` and pure-gradient-transform submodules.

**Win:** ~800 lines removed; one interface; AutoScientist composition simplifies.

---

### 7. K — CLI & Interface Hygiene (M, Low)
**Status:** `DASHBOARD` global decoupled via `EventSink` protocol (`execution/events.py`). **Remaining:**
- **`biopl` dispatcher:** Consolidate 13 console scripts + `cli/run.py` 6-subcommand monolith into one `argparse` subcommand dispatcher (`run | report | parity | repro | hpo | audit | frontier | rank`). Each `cli/` module becomes thin adapter over Pillars A–F APIs.

**Win:** Clean public API boundary; headless CI/sweeps work.

---

### 8. L — Self-Registration (M, Low)
**Status:** `zoo/models/eqprop/__init__.py` auto-computes `__all__` from `vars(module)`. Registry has `aliases()` + `resolve_alias()`. **Remaining:**
- Reduce `bioplausible/__init__.py` and `core/__init__.py` `_LAZY` maps to declared shortlist of public API.
- Other leaf re-export subpackages (e.g. `fa.py`-style) adopt `vars(module)` pattern with per-file `ruff` ignores.

**Win:** Adding a model/rule = one registration decorator; nothing else to touch.

---

### 9. J — Dead Code Tail (S, Low)
- `analysis/tile_*.py` legacy systems → superseded by `evaluation/` + `mep/benchmarks` (post-Pillar D).
- `TODO.md`, `REFACTOR.md`, stale `docs/` → archive out of tree.

---

## Deprioritized (Explicitly Not Now)
- **God-Object Decomposition (O):** `core/trainer.py`, `knowledge/kb.py`, `execution/strategy.py` — split only when Pillars A/E/D touch them; cap effort; stop when cohesive.
- **Settling Loop Merge (I):** Family A/B convergence loops — high numerical risk, low gain. Telemetry unification done.
- **Visualization Stack Consolidation:** 4 stacks — UI preference, not architectural flaw.
- **Micro-Consolidation Remainder (M):** ~12 inline accuracy folds (3-D/accumulation/PL) are legitimately different; `count_parameters` + seeding done.

---

## Execution Sequence & Metrics

| Phase | Pillars | Metric |
|-------|---------|--------|
| **1. Foundations** | B (config) → C (construction tail) | `grep "class ModelConfig"` = 1; `grep "model_cls("` outside construction = 0 |
| **2. Core Unification** | A (training) → E (persistence) → D (measurement) | `grep "loss.backward()"` outside core = 0; 100% outcomes via `result_sink`; 1 `BenchmarkResult` interface |
| **3. Clarity** | G (propagator/model) → K (CLI) → L (self-reg) | 0 propagator-only loops; `biopl` dispatcher works; 1 registration decorator |

**Acceptance Criteria (all must pass):**
1. Import-DAG checker passes in CI (Pillar N gate).
2. `CoreTrainer` sole owner of BPTT/optimizer step logic.
3. No split-brain persistence (all writes via `result_sink`).
4. Zero new test failures beyond the 6 pre-existing numerical/parity drifts.
5. `grep` criteria from criteria #1–#6 all satisfied.

---

## Current Baseline
**Full suite:** 2002 pass / 6 fail / 10 skip / 1 xfail (6 failures = documented numerical/parity drift, unrelated to refactor).
**Lint gate:** Functional (ruff 0.16 parses config; ~2k pre-existing warnings are backlog, not blocker).
**Pyright:** 0 errors strict mode.

---

## Session Log (2026-08-13)

### Completed
- **Pillar B — verified COMPLETE** (single `ModelConfig`, single `ExperimentConfig`, `TrainerConfigSchema`
  and `_KNOB_ALIASES` gone; `omegaconf.py` resolved as I/O-boundary facades, keep).
- **Pillar C acceptance criterion #3 — satisfied** (`grep -rn "model_cls(" | grep -v construction.py` = 0).
  Both `cli/lab.py` and `cli/repro.py` now construct exclusively via `construct_model`.
- `cli/lab.py` hardened: import ordering (E402), `cast` to `nn.Module` (ruff + pyright clean, 0 errors),
  added `hidden_dim`/`num_layers` defaults so the lab actually constructs models that declare them.

### Findings for future work
- **Python 3.14 allows `except A, B, C:`** (tuple-of-exceptions form) without parentheses — old-style
  clauses are valid and `ruff format` will strip redundant parens back to it. Do not "fix" them; they are
  not bugs.
- **Task geometry is ambiguous by design:** `TaskProtocol.input_dim` is typed `int | None` but concrete
  tasks return *tuples* (e.g. `mnist → (1,28,28)`). The single task-resolution seam must thread geometry
  straight through to `construct_model` (matching `_build_runconfig_model`), never `int()`-coerce it.
  `domains/registry.resolve_task` already flattens via `math.prod` for scheduling; unify this policy.
- **`cli/lab.py` `args.model="MLP"` is NOT a registered name** — the registry's model names are
  `backprop_mlp`, `eqprop`, `forward_forward`, etc. The CLI's default is misleading; a future Pillar K pass
  should map a friendly shortlist or error clearly.
- **Engine loader duplication (Pillar C tail):** `engine._get_train_loader`/`_get_val_loader`
  (`execution/engine.py:824,839`) and `trainer._setup_data` (`core/trainer.py:514`) independently call
  `create_data_loaders` / `_setup_lm_data`. The trainer already keeps a `self.task_obj` (LM path only).
  Collapsing all three onto one `DataConfig → DomainTask` resolver in `domains/registry` is the remaining
  Pillar C work — but it is high-risk (touches the CoreTrainer data path) and should be done with targeted
  training tests, not the full suite.

### Acceptance-criteria status
- Criterion #1 (`loss.backward()` outside `core/`+`training_mixin` = 0): **NOT done** — ~35 legit sites
  remain (mep benchmarks, propagators, deployment models, validation tracks). Pillar A scope.
- Criterion #3 (`model_cls(` outside construction = 0): **DONE**.
- Criterion #6 (`zoo/propagators/` = only `mep.py` + gradient transformers): **NOT done** — `backprop.py`,
  `base.py`, `eqprop.py`, `fa.py`, `hebbian.py`, `spiking.py` still present. Pillar G scope.