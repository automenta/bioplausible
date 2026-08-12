# REFACTOR2: Toward an Ideal Architecture — Consolidation, Layering & Single-Source-of-Truth

## Architecture Vision

The codebase (~94k LOC, ~290 modules) has grown through 25+ sprint-style feature additions. The result is 7 parallel training stacks, 4 parallel config hierarchies, 5 duplicate `BenchmarkResult` classes, 4 visualization stacks, and 5 persistence layers. **Capability is not the problem — consolidation is.**

The ideal target architecture is **a strict dependency-layered core with exactly one implementation of every cross-cutting concern**:

```
┌─────────────────────────────────────────────────────────────────┐
│ L7  Interfaces  : CLI · deployment · demo · sklearn · lightning │  live on the public API
├─────────────────────────────────────────────────────────────────┤
│ L6  Measurement : evaluation · validation · benchmarks ·        │  one BenchmarkResult,
│                   analysis · reporting · leaderboard           │  one report renderer
├─────────────────────────────────────────────────────────────────┤
│ L5  Orchestration: execution · hyperopt · autoscientist        │  adapters over the runner
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

**Layering rule:** L_N may import from L_{≤N−1} only. The current module graph violates this everywhere (zoo imports building blocks that live in core, but core.trainer imports zoo; `domains` wraps `data` while `data` is round-tripped through domains; `experiment.reporting` reaches into `execution.synthesizer`).

### Seven principles guiding every pillar

1. **One implementation per concern.** If a concept has a canonical home (config, metrics, checkpoint, result, parameter count), there is exactly one implementation and every other site is an adapter/call into it.
2. **The public API is the boundary.** Libraries expose `TrainerConfig/ModelConfig/Registry/CoreTrainer/BenchmarkResult`; applications (demo, CLI, notebook) consume only those.
3. **Construction is single-sourced.** `construct_model` is the only way a model is built anywhere — finders, trainers, delegates, validators all call it.
4. **The model is the learning rule.** One `train_step(x, y) -> dict` interface per algorithm; no parallel propagator <-> model duality.
5. **Results flow through one funnel.** Every experiment result (success or failure) transits `result_sink`, which owns persistence to all backends.
6. **Configs compose, never duplicate.** A small set of nested dataclasses; identical field names mean identical meanings.
7. **Dead code is deleted.** No "legacy", "seeded", "evolve bridge", or plan-N remnants remain reachable.

---

## Pillar A — Single Training Path (Runner Unification)

### The problem: 7 parallel run stacks
| Runner | Location | Role |
|--------|----------|------|
| `CoreTrainer` | `core/trainer.py:298` | canonical training loop |
| `TrialRunner` / `run_single_trial_task` | `hyperopt/experiment.py:43` | execution/hyperopt per-trial runner |
| `_TaskTrainer` | `domains/trainer.py:72` | task adapter over CoreTrainer |
| `BioLightningModule` + PL `Trainer` | `lightning_/` | Pytorch-Lightning path (re-implements `_train_step` dispatch, `create_model`) |
| `Verifier` | `validation/core.py:20` | track runner (own multi-seed/epoch loop) |
| `StaircaseRunner` + `CoreTrainerDriver` | `experiment/` | probe campaign runner |
| `BenchmarkRunner`, `RigorousBenchmark` | `benchmarks/`, `analysis/tile_profiler.py` | benchmark loops |

All re-implement: `model.train_step → bio-optimizer → BPTT` dispatch, model instantiation, device/seed management, metric compute, checkpointing. `BioLightningModule.training_step` (`lightning_/module.py:121-186`) is a near-copy of `CoreTrainer._train_step` (`core/trainer.py:1123-1173`).

### Target: one loop, many thin adapters
`CoreTrainer` is the only place that runs a step. Everything else is an **adapter**:

- `TrialRunner`, `Verifier`, `StaircaseRunner`, `BenchmarkRunner`, `RigorousBenchmark` → construct a `CoreTrainer` (or a `TrainLoopConfig`) and call `fit()`/`train_epoch()`.
- `BioLightningModule` → is a **callback/plugin** registered on a `CoreTrainer`, not a parallel stack. PL-specific needs (precision, prediction writing) become `ExecutionCallback` implementations. Delete the duplicated `create_model` helper in favor of `construct_model`.
- The duplicated 3× `create_model` helpers (`lightning_/module.py:22`, `execution/robustness.py:33`, `hyperopt/experiment.py:259`) → all become `construct_model` calls.
- Validation *tracks* keep their declarative `track_*` spec but delegate execution to the shared runner.

### Win
- **Eliminates ~2,500 lines** of parallel loops and duplicate instantiation helpers.
- One place to fix bugs, one place to add features (mixed precision, compile, energy tracking).
- New training variants (memory-efficient kernels, distributed) change `CoreTrainer`, not five stacks.

### Key files
`core/trainer.py` · `hyperopt/experiment.py` · `domains/trainer.py` · `lightning_/*` · `validation/core.py` · `experiment/probe.py`, `experiment/staircase.py` · `benchmarks/*` · `validation/tracks/*`

---

## Pillar B — Single Config Hierarchy

### The problem: 4+ duplicate hierarchies, same-named classes
- `ModelConfig` **defined twice**: `config/unified.py:123` and `config/schema.py:58` (± `core/trainer.py` `TrainerConfig`).
- `ExperimentConfig` **defined twice**: `config/unified.py:328` and `config/schema.py:180`.
- `load_config` **defined twice** with different signatures: `config/unified.py:274` and `config/__init__.py:127`.
- Pydantic `TrainerConfigSchema` (`config/__init__.py:58`) mirrors `TrainerConfig` field-for-field (`core/trainer.py:101-193`).
- Also overlapping: `TileAlgorithmConfig`, `LocalLearningConfig`, `DeploymentConfig` family (5 configs in `zoo/models/deployments/`), `BenchmarkConfig` ×4, `RunConfig` family.
- `config/schema.py` exists only as an OmegaConf bridge with `to_internal()` conversion — a parallel mirror that must be kept in sync by hand.

### Target: one canonical hierarchy, no mirrors
A single compositional tree in `config/unified.py`:

```python
@dataclass(frozen=True, slots=True)
class ModelConfig:     # architecture + training knobs (already the canonical knob schema
    ...                #   because construction.py reflects on its fields)
@dataclass(frozen=True, slots=True)
class OptimizerConfig: ...
@dataclass(frozen=True, slots=True)
class DataConfig:      # task, batch_size, num_workers, transforms ...
@dataclass(frozen=True, slots=True)
class TrainLoopConfig: # epochs, batches_per_epoch, grad_clip, early_stop, energy, compile ...
@dataclass(frozen=True, slots=True)
class ExperimentConfig:  # model + optimizer + data + train_loop + hardware + reproducibility
    model: ModelConfig
    optimizer: OptimizerConfig
    data: DataConfig
    train_loop: TrainLoopConfig
    hardware: HardwareConfig
    seed: int
    tags: tuple[str, ...]
```

- **Delete** `config/schema.py` mirror and `config/__init__.py` `load_config` shadow; keep `config/unified.py` `load_config`/`save_config` as the single I/O pair (OmegaConf structured round-trip already proven for frozen dataclasses — see `config/unified.py` module docstring).
- `TrainerConfig` becomes a thin alias/preset over `ExperimentConfig` (one-line `field(default_factory=...)` wiring), keeping the public import path stable.
- `TileAlgorithmConfig` + `LocalLearningConfig` → `ModelConfig.extra` (tile-specific knobs already live there); deployment domain configs fold into `ModelConfig` or a small validated subset.
- `TrainerConfigSchema` (Pydantic) → generated automatically from `ExperimentConfig` at the boundary, never hand-maintained. Pydantic v2 supports `TypeAdapter`/`model_validate` on dataclasses directly — no schema mirror needed.
- The alias layer (`steps`→`max_steps`, `lr`→`learning_rate`) in `construction._KNOB_ALIASES` shrinks to zero: with one config there is one name.

### Win
- **Eliminates ~1,200 lines** of mirrors/converters and an entire class of drift bugs.
- `phantom_knobs` becomes provably exhaustive: every tunable is a `ModelConfig` field.
- OmegaConf stays the single serialization format.

---

## Pillar C — Single Construction & Reconstruction Layer

### The problem: reconstruction logic scattered ~10×
`core/construction.construct_model` is canonical but widely bypassed:
`hyperopt/experiment.py:259`, `lightning_/module.py:22`, `execution/robustness.py:33`, `cli/repro.py:43`, `deployment/ModelLoader.load_from_config:346`, `experiment/probe.py:223`, `zoo/mep/benchmarks/runner.py`, `domains/factory.create_task`, `execution/engine._get_train_loader`, `core/trainer._setup_data` (line 520).

Task/geometry resolution is also re-heuristic'd: `domains/factory.create_task` name-matching (91-140) vs `domains/registry.resolve_task` vs the trainer's `match` on dataset names vs `engine._get_train_loader`.

### Target
- Every model instantiation goes through `construct_model` (or `ExperimentConfig → model`). Helpers like `create_model()` in lightning/robustness disappear; callers use `construct_model(Registry.get(MODEL, name), ...)`.
- `create_task`/`resolve_task`/data-setup collapse into one `DataConfig → DomainTask` resolution in `domains` — the trainer, engine, and CLI all call it; the trainer's `match self.config.task:` block moves into `domains.registry`.
- `DeploymentConfig` reconstruction and `repro._instantiate` both call the same builder.

### Win
- **~600 lines** of heuristics removed; one `ValueError` message for "unknown task/model".
- Serialization round-trips (checkpoint → run) become lossless by construction.

---

## Pillar D — Single Measurement & Reporting Stack

### The problem: parallel measurement ecosystems
- **`BenchmarkResult` defined 5×**: `analysis/tile_profiler.py:845`, `zoo/mep/benchmarks/runner.py:83`, `evaluation/base.py:190`, `benchmarks/rigorous.py:251`, `benchmarks/compare_nanoGPT.py:323`.
- **Pareto-computation 3×**: `analysis/results.py:207`, `experiment/reporting.py:100`, `hyperopt/frontier.py`.
- **Failure-manifesto 3×**: `analysis/failure_manifesto.py`, `experiment/reporting.py:128`, `execution/synthesizer._analyze_failures:527`.
- **Power-law fitting 2×**: `analysis/scaling.py:18` vs `hyperopt/scaling_law.py`.
- **Report renderers 5×**: `analysis/reporting.py` (Optuna DB), `experiment/reporting.py` (JSONL), `execution/synthesizer.py` (pandas), `validation/notebook.py`, `leaderboard/generator.py`.
- **Ranking/leaderboard 3×**: `analysis/results.get_rankings`, `leaderboard/generator.py`, `cli/rank.py`.
- **Visualization stacks 4×**: `visualization.py`, `execution/dashboard.py`, `zoo/mep/benchmarks/visualization.py`, `demo/charts.py`.
- **6+ benchmark/trial-runner loops** in `mep/benchmarks/*` alone (5 distinct implementations each with its own `create_model`/`get_dataloader`/`evaluate`/`BenchmarkResult`).

### Target: one canonical `evaluation` package
- **One `BenchmarkResult`** in `evaluation/base.py`; all benchmark modules import it.
- **One `metrics` module**: fold `core/metrics.py`, `hyperopt/metrics.py`, `evaluation/base.py` metric fns, and the ~12 inline `(logits.argmax(1) == y).float().mean()` copies into `core/metrics.py` + `core/losses.py` (accuracy is in `core/losses.compute_accuracy`). One `count_parameters` in `core/utils` (already exists) used by synthesizer, hyperopt, estimator.
- **One report renderer** over one canonical result format. Choose JSONL `experiment/report.py` as the canonical Result (it already has CIs via `validation.statistics`); `analysis/reporting.py` becomes a thin Optuna-DB adapter that *renders* the same report from the same data model. Delete the pandas `synthesizer` report path or reduce it to a consumer.
- **One benchmark suite** = `evaluation/` (registry-driven `BenchmarkRegistry`, `EvaluatorBase`, `MetricSuite`). `benchmarks/rigorous.py`, `mep/benchmarks/*`, `analysis/tile_*`, `validation/tracks` become **tracks of `BenchmarkRegistry`** — declarative entries, not new loops.
- **One leaderboard/ranking** implementation; `leaderboard/` and `cli/rank.py` render it.

### Win
- **Eliminates ~4,000 lines** of reporting/benchmark reimplementations.
- Results are comparable across sources because they share a schema, CIs, and renderers.
- Researchable: a *finding* has a single canonical representation.

---

## Pillar E — Single Result & Persistence Funnel

### The problem: 5 write paths for trial outcomes
1. Optuna SQLite (`trials`/`study`/`trial_values`)
2. `HyperoptStorage.hyperopt_logs` (`hyperopt/storage.py`)
3. JSONL `experiment/Report` (`experiment/report.py`)
4. `knowledge/kb.db` (`knowledge/kb.py`)
5. `execution_state.db` failure log (`execution/_state.py`) — **separate** from `experiment/result_sink.py:_record_failure:191`

`result_sink.record_experiment_result` (`experiment/result_sink.py:82`) is *meant* to be the funnel (success → KB, failure → FailureTracker) and probes/hyperopt already partially call it — but execution engine, validation verifier, and mep-benchmarks write around it.

Also duplicated: artifacts zip-extraction (`engine._get_weights_context:776` vs `hyperopt/experiment._load_transfer_weights:84`), checkpoint save paths (8+ ad-hoc implementations despite `core/checkpoint_mixin.py`). And `knowledge/seed.py` defines a **second `KnowledgeBase`** class (unused dead code).

### Target
- **`result_sink` is the only writer of trial outcomes.** All five persistence backends are private details of `result_sink`'s implementation. `record_experiment_result` gains a `report`/`weights_path`/`checkpoint` payload and owns: Optuna `tell`, `hyperopt_logs`, JSONL `Report`, KB upsert, failure log.
- Delete `knowledge/seed.py` second `KnowledgeBase`; keep the real `kb.py` API.
- **One artifact loader** (`core/checkpoint.load_checkpoint` + a small `find_trial_artifact(trial_id)` helper) used by engine and hyperopt.
- Ad-hoc checkpoint saves become `CheckpointMixin`/`core.checkpoint` calls; the `CheckpointManager` SQLite buffer in `execution/_lifecycle.py` stays only if it adds value over `core.checkpoint`.

### Win
- **Eliminates ~700 lines** of duplication and, critically, **split-brain audit trails**: one record = one canonical row across all backends.
- Failure logging finally consistent (engine currently logs to a different store than result_sink).

---

## Pillar F — EqProp Consolidation (from v1, retained)

### Problem recap
6+ registered EqProp models (`StandardEqProp`, `DirectedEP`, `FiniteNudgeEP`, `LazyEqProp`, `MomentumEquilibrium`, `SparseEquilibrium`) all share `zoo/models/eqprop/_energy.py` `EquilibriumMLP` and differ only by `variant`. Plus `LoopedMLP` (alias), `BackpropMLP`, `MemoryEfficient*`, `NeuralCube`, `TransformerEqProp`, `ConvEqProp`, `GraphEqProp` — a deep directory of near-duplicate registrations.

### Target
**_energy.py_ `EquilibriumMLP` is the only registered eqprop model.** `variant` (and `nudge_steps`, `sparse_ratio`, `momentum`, `feedback_gain`, `feedback_init_gain`, `w_rec_init/gain`, `update_scale{,_by_depth}`) are `ModelConfig.extra` knobs — most already appear in `RULE_SPACES["eqprop"]`. Named lookups (`"directed_ep"`, `"sparse_equilibrium"`) keep working via a tiny alias table in the registry, mapping to `("eqprop", {variant: ...})` so search space + sweep + existing configs still resolve.

Non-MLP eqprop architectures (conv, transformer, graph, cube, diffusion, homeostatic, mem-efficient, holomorphic, ternary) are *architecturally distinct* and legitimately separate — but each should be **thin**: a registered subclass overriding only `_build_layers`/`forward_dynamics`, inheriting `train_step`, weights init, and construction. The `variant` mechanism generalizes to an architecture registry.

### Win
- **~500 lines** of registration boilerplate gone; single debugging target.
- Adding "eqprop with slightly-different dynamics" = one config knob, not a new file.

---

## Pillar G — Propagator/Model Unification (from v1, retained)

### The problem (confirmed with evidence)
`core/registry.py:300 _PROPAGATOR_TO_MODEL` hard-codes the duality: `ff`, `pepita`, `target_prop`, `difference_target_prop`, `predictive_coding` are *models* that exist under propagator lookup only to produce a confusing cross-reference error. Meanwhile real propagators (`EqProp`, `FeedbackAlignment`, `Hebbian`, `SpikingSTDP`) still **re-implement settling** the corresponding models already own.

### Target
**The model owns the learning rule**: one `train_step(x, y) -> dict` per algorithm. `ComponentCategory.PROPAGATOR` shrinks to *pure* gradient transformers that operate on any model without owning the forward (Muon, spectral norm projection, EWC reg). The `LearningRuleOptimizer` base class and its 2-phase/4-phase trainer dispatch collapse:

- `CoreTrainer._train_step` goes from 5 phases to **2**: `energy-model` (structural `match`) → `model.train_step` → `BPTT` (standard optimizer). Phases 2 & 4 (explicit propagator=, learning-rule optimizer=) are deleted.
- `zoo/propagators/{eqprop,fa,hebbian,backprop,spiking}.py` → deleted or converted to model-side `train_step`s.
- `_PROPAGATOR_TO_MODEL` becomes the *only* lookup — a compatibility map, not an error message.

### Win
- **~800 lines** removed; one interface to learn and one to teach.
- AutoScientist composition (`requires`/`provides` capability check) simplifies: query MODEL by `credit_assignment_type`, no propagator branch.

---

## Pillar H — Single Search Space & Constraint Source

### The problem
- Two spaces in `hyperopt/search_space.py`: `SEARCH_SPACES` (`SearchSpace` class, `sample()`, model-keyed, coarse ranges) vs `RULE_SPACES` (rule-keyed, continuous, used by Optuna + P0a gate) — **different ranges for the same parameter** (`eqprop.learning_rate`: `(1e-5,1e-2)` vs `(1e-2,5e-1)`).
- `ALGORITHM_FAMILY_CONSTRAINTS` in `execution/_guards.py:175` re-declares family→knob restrictions that overlap `RULE_SPACES`.
- `create_constrained_optuna_config` exported in two namespaces (`execution._guards` and `hyperopt.__init__`).

### Target
- **Delete `SEARCH_SPACES` and the `SearchSpace` class.** `RULE_SPACES` is the single source; `get_search_space(model_name)` resolves model→rule via registry `family` metadata and returns the rule's space.
- Move `ALGORITHM_FAMILY_CONSTRAINTS` into `hyperopt/search_space.py` next to `RULE_SPACES` (same file = same canonical family knowledge), then have `_guards` import it.
- Keep **one** public face for `create_constrained_optuna_config` (in `hyperopt`).

### Win
- **~250 lines** removed; one range per hyperparameter per rule, guaranteed consistent between sampling, P0a audit, and constraint injection.
- Removes the `lazy __getattr__` hack for `create_constrained_optuna_config` in `hyperopt/__init__.py`.

---

## Pillar I — Settling Utility Unification (from v1, retained)

### The problem
`zoo/_settling.py` has `settle_single_state` (Family A) and `settle_activations_list` (Family B) implementing the same loop twice (iteration, early convergence, gradient checkpointing, spectral-norm freeze, trajectory/dynamics). `settle_state`, `_inf_norm_converged`, and `energy_gradient_descent` are yet more partially-overlapping primitives.

### Target
One **`SettleState` protocol** with two adapters (`TensorState`, `ActivationsListState`) and one `settle()` driving convergence/dynamics/checkpoint/SN-freeze uniformly. `EquilibriumFunction` keeps its implicit-differentiation role but reuses the protocol's `_step` plumbing.

### Win
- **~200 lines** removed; uniform convergence telemetry across all equilibrium families (important for the researchability goal).

---

## Pillar J — Dead Code & Legacy Removal

Confirmed dead/legacy subtrees to delete (with tests), after grepping for last consumers:

| Target | Evidence |
|--------|----------|
| `execution/evolve_evaluator.py` | `engine.py:516` "ASI-Evolve integration removed" |
| `knowledge/seed.py` | second `KnowledgeBase`, unused |
| `data/transforms.py` | orphaned — nothing imports it |
| `experiments/` (+ `presets.py`, `utils.py`) | superseded by `domains/` + `experiment/` |
| `campaign/` | only `__init__.py`; migrated to `experiment/schema.py` |
| `archive/` | dead |
| `hyperopt/parallel_runner.py:_worker_process_task` | near-identical twin of `_wrapped_worker:113` |
| `analysis/tile_*.py` legacy systems | superseded by `evaluation/` + `mep/benchmarks` (post-Pillar D) |
| `hyperopt/comparator.py` vs `comparison.py` | merge into one comparison module |
| `TODO.md`, `REFACTOR.md`, stale `docs/` | archive out of tree |

Also resolve the **circular-import hacks** (`execution/__init__.py:20` lazy loader, many `lazy getattr` pattern) — with layering fixed (Pillar A + Pillar B), the import graph becomes a DAG and lazy init disappears.

---

## Pillar K — Demo, CLI & Interface Hygiene (from v1, retained/expanded)

### Target
- **`demo/` moves out of the package** to a sibling repo (or is excluded in `pyproject.toml` `exclude`), consuming only public API. Also removes NiceGUI/Plotly from any package import surface. **Note:** the demo header says it lives in package tree but is not part of `bioplausible.*` — verify `setuptools.find` already excludes it and make exclusion explicit.
- **CLI consolidation**: 13 console scripts + 4 overlapping run loops + 3 "report" entry points. Introduce a **one-command dispatcher** (`biopl` with `run | report | parity | repro | hpo | audit | frontier | rank` subcommands) backed by a tiny argframework — no new deps (stdlib `argparse` subparsers are enough). Each `cli/` module becomes a thin adapter over `Pillars A-F`'s canonical APIs. Delete `cli/run.py`'s 6-subcommand monolith in favor of dispatch + shared `_resolve_targets`.
- `sklearn_interface.py` stays but calls `construct_model`/`CoreTrainer` (it already does).
- **`DASHBOARD` global singleton** (`execution/dashboard.py:349`) — decouple: decision modules (`strategy.py`, `engine.py`) accept an `EventSink` protocol (dashboard = one implementation); remove the global import from decision logic. This unblocks UI-free use (headless sweeps).

---

## Pillar L — Self-Registration Eliminates Hardcoded Top-Level Repetition

### The problem
`bioplausible/__init__.py` (200 lines) and `core/__init__.py` (57 lines) hand-maintain `_LAZY` maps; `core/registry.py` maintains `_PROPAGATOR_TO_MODEL`; `zoo/models/eqprop/__init__.py` hand-lists 40+ exports; `SEARCH_SPACES` name list drifts from `RULE_SPACES`.

### Target
- **Metadata-driven discovery**: with `family`/`alias` metadata on `ComponentMetadata` (already exists), the top-level `__init__` can be *generated* or reduced to a declared shortlist of the real public API. Registry gains `aliases()` and reverse-lookup (`get(name)` → alias chain) so named lookups (`"directed_ep"`) survive consolidation.
- `__all__` list-comprehensions computed from module contents (`vars(module)`) for zoo subpackages instead of hand-written 40-item lists.

### Win
- Adding a model/rule = one registration decorator; **nothing else** to touch (no `__init__` edit, no `_PROPAGATOR_TO_MODEL` edit, no `__all__` edit).

---

## Implementation Roadmap

Pillars are ordered by value/effort and by dependency (each row de-risks the next).

| # | Pillar | Primary Win | Effort | Risk | Blocks |
|---|--------|-------------|--------|------|--------|
| 1 | **J** Dead code removal | unblock everything; safest first | S | None | all |
| 2 | **B** Single config hierarchy | removes drift at the root; unblocks A/C/E | XL | High | A, C, E |
| 3 | **C** Single construction layer | one way to build | M | Med | A |
| 4 | **A** Single training path | 2.5k lines, one loop | XL | High | B, C |
| 5 | **H** Single search space | consistency | S | Low | B |
| 6 | **F** EqProp consolidation | 500 lines, one engine | M | Low | B |
| 7 | **I** Settling unification | 200 lines, uniform telemetry | M | Low | F |
| 8 | **G** Propagator/model unification | 800 lines, one interface | M | Med | A, F |
| 9 | **E** Single result funnel | audit correctness | M | Med | A |
| 10 | **D** Single measurement/reporting stack | 4k lines | XL | High | E |
| 11 | **L** Self-registration | zero-touch extensibility | M | Low | B, F, G |
| 12 | **K** Demo/CLI/dashboard hygiene | boundaries | M | Low | A, B |

**Suggested execution sequence:** J → B → C (foundations), then **F → I → G** (zoo purity), then **A → E → H** (training + persistence), then **D** (measurement), then **L, K** (extensibility + boundary). Each pillar ships with `ruff format && ruff check && pyright && pytest --cov` green.

---

## Acceptance Criteria (single-sourced codebase)

1. **One training loop**: `CoreTrainer._train_step` has ≤3 branches (energy-model / model `train_step` / BPTT); all other runners are adapters. `grep -rl "loss.backward()" bioplausible/` outside `core/` & `training_mixin` returns nothing.
2. **One config tree**: `grep -rn "class ModelConfig" bioplausible/` → exactly 1 hit (`config/unified.py`); `class ExperimentConfig` → 1; `def load_config` → 1.
3. **One construction path**: `grep -rn "model_cls(" bioplausible/` outside `core/construction.py` returns no *instantiation* sites — only `construct_model` calls.
4. **One `BenchmarkResult`** and **one result funnel**: `record_experiment_result` is called by execution, hyperopt, validation, and mep-benchmarks; all five persistence backends written only from `result_sink`.
5. **One search space**: `grep -rn "SEARCH_SPACES\b"` → 0 hits.
6. **zoo purity**: `zoo/propagators/` contains only `mep.py` and pure-gradient-transform submodules; `zoo/models/eqprop/` holds one registered engine + architecture subclasses.
7. **Acyclic import graph**: `execution/__init__.py` and `hyperopt/__init__.py` lazy-loaders deleted; import-time side effects limited to registry decorators.
8. **No global UI mutation from decision code**: `strategy.py`/`engine.py` route events through an injected `EventSink`.
9. **Dead code absent**: `execution/evolve_evaluator.py`, `knowledge/seed.py`, `campaign/`, `experiments/`, `data/transforms.py`, `search_space.SearchSpace` gone.
10. **Full suite green**, including parity/validation/hyperopt cross-checks; AutoScientist end-to-end smoke run.

---

## Current Status & Progress Log

Last updated: 2026-08-12. Baseline when this log began: **13 pre-existing test
failures** (2003 collected) — all unrelated to the refactor and still present.

| Pillar | Status | Notes |
|--------|--------|-------|
| J | **partial** | Safe deletions done (commit `c1a68b3`); see log below. |
| B | **partial** | Dead `ExperimentSchema`/`load_config` duplicate removed (commit `8bb4727`). Full merge deferred — see findings. |
| H | **partial** | Removed `hyperopt` lazy `__getattr__` re-export (commit `c1a68b3`). `SEARCH_SPACES`/`SearchSpace` deletion **blocked by p2p** — see findings. |
| C, A, D, E, F, G, I, K, L | not started | — |

### Completed work

**Pillar J (commit `c1a68b3`)**
- Deleted `execution/evolve_evaluator.py` — zero consumers (ASI-Evolve bridge).
- Deleted `campaign/` package — re-export shim; `experiment.schema` is canonical.
- Deleted `bioplausible/experiments/` (presets.py, utils.py) — orphaned; superseded by `domains/` + `experiment/`.
- Deleted dead `_worker_process_task` twin from `hyperopt/parallel_runner.py` (near-copy of `_wrapped_worker`); moved `ExperimentTask` to `TYPE_CHECKING`.
- Stripped the duplicate (JSON-file-based) `KnowledgeBase` from `knowledge/seed.py`; kept `KNOWLEDGE_BASE_SEED` data. Removed `SEED_KB` lazy re-export + `get_default_kb` from `knowledge/__init__.py`.
- Removed `hyperopt/__init__.py` lazy `__getattr__` for `create_constrained_optuna_config`/`get_constrained_search_space` (dead re-export — nothing consumed them via `hyperopt`); `execution/engine.py` now imports them from `execution._guards` directly.

**Pillar B micro-win (commit `8bb4727`)**
- Deleted dead `ExperimentSchema` + `config/__init__.py:127 load_config` (a second `load_config` definition with zero consumers — YAML+ExperimentSchema loader). `config/unified.py` `load_config`/`save_config` is now the only I/O pair. Removed now-unused `yaml`/`pathlib`/`ValidationError` imports.

### Findings that change the plan (important for future work)

1. **`data/transforms.py` is NOT orphaned** (plan Pillar J table is wrong on this row): it is imported by `data/vision.py`, `domains/vision.py`, and `zoo/mep/benchmarks/continual_learning.py` (`build_transform`, `normalization`, `create_dataloader`, `MNIST_TRANSFORM`). Do not delete.
2. **`hyperopt/comparator.py` vs `comparison.py` are NOT duplicates** (plan Pillar J row wrong): `comparator.py` is frontier-comparison (`compare_frontiers`, `FrontierComparison`, `OperatingPointMatch`); `comparison.py` is multi-algorithm ranking (`AlgorithmRanking`, `ComparisonStudy`, `compute_algorithm_rankings`). Different consumers (`analysis/results.py`+`cli/run.py` vs `hyperopt/__init__.py`+tests). No merge.
3. **Pillar B merge is bigger than the plan implies and partially blocked by tests.** `config/schema.py` classes are **facades, not mirrors** of `config/unified.py`: schema `ModelConfig` (name/kwargs/compile/compile_mode) vs unified `ModelConfig` (name/input_dim/output_dim/hidden_dims/extra) differ field-for-field, and schema `ExperimentConfig` carries the OmegaConf structured section types (`DatasetConfig`, `TrainerConfig`, `LightningConfig`, …). `tests/unit/test_refactor2_bugfixes.py` (`test_config_schema_*`, ~3 tests) asserts `schema.py` contents exist as non-frozen dataclasses with a single resolver registration — deleting the file breaks those tests. **Recommended path**: migrate the OmegaConf facade classes into `unified.py` (keep names), make `config/__init__.py` re-export from unified, then delete `schema.py`; update `test_refactor2_bugfixes` accordingly. Needs a dedicated session (XL, high-risk).
4. **Pillar H `SEARCH_SPACES`/`SearchSpace` deletion is blocked by p2p** (plan didn't account for it): `p2p/evolution.py` uses `list(SEARCH_SPACES.keys())`, `get_search_space()`, `space.sample()`, `space.apply_constraints()`, and (latent-broken) `space.crossover()`/`space.mutate()` — those last two don't exist on `SearchSpace` (only `sample`+`apply_constraints`), so p2p's crossover/mutate paths would `AttributeError` at runtime. Options: (a) keep `SearchSpace` as the genetic-op vehicle but build it from `RULE_SPACES` via `get_search_space(model)` resolving model→rule through registry `family` metadata; (b) reimplement p2p sampling against raw `RULE_SPACES` dicts. Tests: `tests/integration/test_p2p_constraints.py` constructs `SearchSpace` directly.
5. **Pre-existing unrelated breakage** (not from this refactor): `bioplausible/execution/cli.py:12` imports `ReportOrchestrator` from `analysis/reporting.py` which no longer defines it → `bioplausible.execution.cli` is unimportable. `tests/integration/test_onnx.py` calls `LoopedMLP(...)` and `EquilibriumMLP.forward()` with 7 args (signature drift). `test_triton_kernel` + `test_equilibrium_parity` + several `backprop_parity`/`biology_axioms` acc-drift tests fail at baseline. NEBC `test_cannot_instantiate_base` expects abstract TypeError but class is concrete. These should be triaged separately.

### Facilitation for future work

- **Test baseline**: full suite = `uv run pytest -q --no-cov`; 1990 pass / 13 fail pre-refactor. Targeted fast check for config/construction work: `pytest tests/unit/core/test_config_*.py tests/unit/test_refactor2_bugfixes.py tests/unit/execution tests/unit/analysis`.
- **Lint baseline**: repo has ~2100 pre-existing `ruff check` errors (mostly non-empty-init-module, long lines, typing-only imports). Keep edits to touched files clean; do not chase the global baseline.
- `config/unified.py` already documents the proven OmegaConf frozen-dataclass round-trip (module docstring) — the single serialization path is ready to build Pillar B on.
- `core/construction.construct_model` is already the canonical builder used by trainer/estimator/finders/probe; Pillar C's remaining `create_model` helpers (`lightning_/module.py:22`, `execution/robustness.py:33`) have **different signatures** (name+kwargs, and are `unittest.mock.patch` targets) vs `construct_model` (sampled-config + required dims) — consolidate by adapting call sites, not by deleting.

---

## Non-Goals (kept in scope discipline)

- No new learning algorithms, features, or research capability.
- No re-tune of default hyperparameters (values move, meanings don't).
- No change to the tile substrate graph/settling kernels or the MEP math.
- P2P federation and the `acceleration/` CUDA/Triton backends are islands; they are *targets* of the same construction/result refactors but their internals are untouched.
- Gradual, per-pillar merges only; no big-bang rewrite.

---

## Definition of Done for the *Ideal* Architecture

The refactor is complete when a new contributor can:

1. **Add a new learning rule** by writing one class with a `train_step` + one `register_model` decorator (nothing else).
2. **Run any experiment** (probe, hyperopt trial, validation track, rigorous benchmark) by constructing `ExperimentConfig` and passing it to the single runner.
3. **Read any result** (JSONL, Optuna DB, KB) as one canonical schema with one renderer.
4. **Trace a hyperparameter** from `RULE_SPACES` → `ModelConfig` → `construct_model` → model attribute without crossing a name alias.

That is the ideal: fewer, deeper, composable building blocks — each with exactly one home.