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
- `ModelConfig` **defined twice**: `config/unified.py:123` and `config/omegaconf.py` (renamed `ExperimentModelConfig` — was `config/schema.py:58`).
- `ExperimentConfig` **defined twice**: `config/unified.py:328` and `config/omegaconf.py` (renamed `ExperimentSchemaConfig` — was `config/schema.py:180`).
- `TrainerConfigSchema` (Pydantic, `config/__init__.py:58`) mirrors `TrainerConfig` field-for-field (`core/trainer.py:101-193`) — **zero production consumers** (only `tests/unit/core/test_config_schema.py`).
- Also overlapping: `TileAlgorithmConfig`, `LocalLearningConfig`, `DeploymentConfig` family (5 configs in `zoo/models/deployments/`), `BenchmarkConfig` ×4, `RunConfig` family.
- `config/omegaconf.py` exists only as an OmegaConf bridge with `to_internal()` conversion — a parallel mirror that must be kept in sync by hand.

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

- **Delete** `config/omegaconf.py` mirror; keep `config/unified.py` `load_config`/`save_config` as the single I/O pair (OmegaConf structured round-trip already proven for frozen dataclasses — see `config/unified.py` module docstring).
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
- **Pareto-computation 3×** → **UNIFIED** into `hyperopt.metrics.non_dominated_indices` (commit `fa62672`); was `analysis/results.py`, `experiment/reporting.py`, `hyperopt/frontier.py`.
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

### Status: **DONE** (verified)
- The 6 variant models already live as thin `EquilibriumMLP` subclasses in `zoo/models/eqprop/_energy.py:624-710`, differing only by a class-level `variant`. The directory's other eponymous files are 5-line re-export shims. Verified green: 62 tests across `test_eqprop*.py` + `test_settling_memory.py`.
- The plan's roadmap row is stale; Pillar F requires no code change.

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

### Status: **DONE** (committed `c777549`)
- Deleted the ~245-line `SEARCH_SPACES` dict and the old heuristic `get_search_space`. New resolution in `hyperopt/search_space.py`: a model whose *name* is a `RULE_SPACES` key uses that rule verbatim (identical to the P0a constructor gate); otherwise the model's registered `family` maps through `_FAMILY_TO_RULE` (backprop/baseline→backprop, eqprop→eqprop, fa/feedback_alignment→feedback_alignment, target_prop→target_prop, forward_only/mep→forward_forward); registered families without a rule (hebbian, equitile, tile, predictive_coding, spiking, hybrid) get a small honest `_FALLBACK_SPACE` instead of a divergent curated grid.
- `get_available_models()` (registry-driven, via `_registered_families`) replaces `list(SEARCH_SPACES.keys())` in `p2p/evolution.py` "new architecture" discovery, so a sampled config always carries a constructible registered model name.
- `SearchSpace.apply_constraints` mapping extended to a list-of-pairs so both the legacy `steps` and the `RULE_SPACES` `max_steps` param conventions get clamped.
- Removed the `SEARCH_SPACES` re-export from `hyperopt/__init__.py`. Zero `SEARCH_SPACES` hits remain in `bioplausible/` or `tests/`.
- `SearchSpace` class itself is kept: it hosts the GA operators (`sample`/`crossover`/`mutate`/`apply_constraints`) used by the p2p island, and `tests/integration/test_p2p_constraints.py` exercises it directly.

### Win
- **~250 lines** removed; one range per hyperparameter per rule, guaranteed consistent between sampling, P0a audit, and constraint injection.
- Removes the `lazy __getattr__` hack for `create_constrained_optuna_config` in `hyperopt/__init__.py`.

---

## Pillar I — Settling Utility Unification (from v1, retained)

### The problem
`zoo/_settling.py` has `settle_single_state` (Family A) and `settle_activations_list` (Family B) implementing the same loop twice (iteration, early convergence, gradient checkpointing, spectral-norm freeze, trajectory/dynamics). `settle_state`, `_inf_norm_converged`, and `energy_gradient_descent` are yet more partially-overlapping primitives.

### Status: **PARTIAL** (uniform telemetry added, commit `c32e15f`)
- `settle_single_state` (Family A) now reports the same dynamics surface as `settle_activations_list` (Family B): added `steps_taken` / `converged` / `settle_time_s` to its dynamics dict, alongside the existing `deltas` / `final_delta`. Tracks the step counter through the SN-freeze `warmup`/`main_loop` split and captures the `_inf_norm_converged` break.
- Convergence loops are NOT merged (Family A uses inf-norm `_inf_norm_converged`; Family B uses max-relative per-layer norm) — only the reporting surface is now uniform, per the documented low-risk first step.

### Target (remaining)
One **`SettleState` protocol** with two adapters (`TensorState`, `ActivationsListState`) and one `settle()` driving convergence/dynamics/checkpoint/SN-freeze uniformly. `EquilibriumFunction` keeps its implicit-differentiation role but reuses the protocol's `_step` plumbing.

### Win
- **~200 lines** removed; uniform convergence telemetry across all equilibrium families (important for the researchability goal).

---

## Pillar J — Dead Code & Legacy Removal

### Status: **MAJORITY DONE** (commits `c1a68b3`, `6ac0583`, `8bb4727`, `2e147c2`, `5e5d5a2`, `1fcd637`)

| Target | Status | Evidence |
|--------|--------|----------|
| `execution/evolve_evaluator.py` | **DELETED** | `engine.py:516` "ASI-Evolve integration removed" |
| `knowledge/seed.py` second `KnowledgeBase` | **DELETED** | Kept `KNOWLEDGE_BASE_SEED` data only |
| `campaign/` | **DELETED** | Only `__init__.py`; migrated to `experiment/schema.py` |
| `experiments/` (+ `presets.py`, `utils.py`) | **DELETED** | Superseded by `domains/` + `experiment/` |
| `archive/` | **NOT IN TREE** | Does not exist in-tree |
| `hyperopt/parallel_runner.py:_worker_process_task` | **DELETED** | Near-identical twin of `_wrapped_worker:113` |
| `hyperopt/comparator.py` vs `comparison.py` | **NOT DUPLICATES** | Different consumers: `comparator.py`=frontier-comparison; `comparison.py`=multi-algorithm ranking. **Do not merge.** |
| `data/transforms.py` | **KEEP** | Imported by `data/vision.py`, `domains/vision.py`, `zoo/mep/benchmarks/continual_learning.py` — **NOT orphaned** (plan table was wrong). |
| `execution/cli.py` | **DELETED** | Zero consumers; imported nonexistent `ReportOrchestrator` making package unimportable |
| `hyperopt/__init__.py` lazy `__getattr__` | **REMOVED** | Dead re-export for `create_constrained_optuna_config`/`get_constrained_search_space` |
| `config/__init__.py:127 load_config` | **DELETED** | Second `load_config` definition with zero consumers |
| `NEBCBase` abstract contract | **FIXED** | Added `@abstractmethod _build_layers` (was `ABC` with no abstracts) |

### Remaining
- `analysis/tile_*.py` legacy systems → superseded by `evaluation/` + `mep/benchmarks` (post-Pillar D)
- `TODO.md`, `REFACTOR.md`, stale `docs/` → archive out of tree

### Win
- **~1,500 lines** removed; unblocks import graph DAG (no more lazy-loader hacks).

---

## Pillar K — Demo, CLI & Interface Hygiene (from v1, retained/expanded)

### Target
- **`demo/` moves out of the package** to a sibling repo (or is excluded in `pyproject.toml` `exclude`), consuming only public API. Also removes NiceGUI/Plotly from any package import surface. **Note:** the demo header says it lives in package tree but is not part of `bioplausible.*` — verify `setuptools.find` already excludes it and make exclusion explicit.
- **CLI consolidation**: 13 console scripts + 4 overlapping run loops + 3 "report" entry points. Introduce a **one-command dispatcher** (`biopl` with `run | report | parity | repro | hpo | audit | frontier | rank` subcommands) backed by a tiny argframework — no new deps (stdlib `argparse` subparsers are enough). Each `cli/` module becomes a thin adapter over `Pillars A-F`'s canonical APIs. Delete `cli/run.py`'s 6-subcommand monolith in favor of dispatch + shared `_resolve_targets`.
- `sklearn_interface.py` stays but calls `construct_model`/`CoreTrainer` (it already does).
- **`DASHBOARD` global singleton** (`execution/dashboard.py:349`) — decouple: decision modules (`strategy.py`, `engine.py`) accept an `EventSink` protocol (dashboard = one implementation); remove the global import from decision logic. This unblocks UI-free use (headless sweeps).

### Win
- Clean public API boundary; no UI framework deps in library code.
- Headless CI/sweeps no longer pull dashboard machinery.

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

## Pillar M — Micro-Consolidation: the ~60 hand-rolled one-liners

### The problem (newly quantified — highest raw repetition in the tree)
Three canonical primitives already exist but are bypassed at dozens of sites, so the same one-liner is copied with silent *semantic drift*:

1. **Accuracy (~40 sites)** — `core.losses.compute_accuracy` (handles one-hot/reshaped/3-D targets) is canonical, yet `zoo/models/{fa.py,forward_only.py,target_prop.py,spiking.py,hebbian.py,base.py}`, `core/ebm.py`, `core/training_mixin.py`, `domains/*.py`, `lightning_/module.py`, `validation/utils.py`, `validation/tracks/*`, `execution/robustness.py`, `zoo/mep/benchmarks/*`, `zoo/models/eqprop/_contrastive.py` all hand-roll `(logits.argmax(dim=1) == y).float().mean()`. **Drift hazard:** some sites return a 0–1 ratio, others `× 100` (e.g. `validation/utils.py:107`, `nebc_base.py:113`, `application_tracks.py:194`), some `.item()`, some call `argmax(-1)` vs `argmax(1)` — two different correctness bugs depending on target layout. Every site should call `core.losses.compute_accuracy`.
2. **`count_parameters` (~17 sites)** — `bioplausible.utils.count_parameters` (handles `_orig_mod` unwrap + trainable-only) is canonical, yet `deployment.py:122`, `benchmarks/rigorous.py:435/491/514`, `benchmarks/compare_nanoGPT.py:221/511`, `hyperopt/experiment.py:432`, `evaluation/{base,cross_domain}.py`, `core/{trainer,profiling,spectral_mixin}.py`, `zoo/models/tile_lm.py`, `zoo/models/eqprop/eqprop_lm_variants.py` all re-`sum(p.numel() for p in model.parameters())`. The compiled-model unwrap (`_get_model_for_processing`) is silently lost at most sites.
3. **Seeding (two divergent APIs + ~10 raw sites)** — `bioplausible.utils.seed_everything(seed, device)` (returns an env fingerprint, refuses silent CPU fallback; used by `cli/repro`, `cli/parity`, `experiment/probe`) *and* `core.utils.seeds.set_all_seeds(seed, deterministic)` (sets `use_deterministic_algorithms`; used by `cli/run`, `core/trainer`, `benchmarks`) seed the same RNGs with incompatible behavior/shapes. `cli/repro.py:183` and `validation/{utils,core,gradient_check}.py`, `training/rl.py`, `sklearn_interface.py`, `domains/{timeseries,scientific}.py`, `zoo/mep/benchmarks/{niche_benchmarks,ewc_baseline,continual_learning}.py` still call raw `torch.manual_seed(seed) + np.random.seed(seed)`.

### Target
- **One `compute_accuracy`** — fold the ~40 inline accuracy one-liners; where the caller genuinely needs a `×100` or score-ratio form, add `scale=100`/`as_score` params to the canonical fn instead of re-implementing. One target-layout rule, one return convention.
- **One `count_parameters`** — all sites call `bioplausible.utils.count_parameters`; the `_orig_mod` unwrap and trainable-only semantics become uniform.
- **One seeding API** — merge `seed_everything` and `set_all_seeds` into a single `core.utils.seeds` entry point (one returns the fingerprint + determinism flag; the other is a thin wrapper), then move every raw `torch.manual_seed` site onto it.

### Win
- **~250 lines** of near-identical arithmetic removed and, more importantly, **three silent-drift classes eliminated**: accuracy target-layout bugs, param-count-with-compiled-model undercounts, and seed-mismatch repro failures.
- Highest *researchability* win per line: correctness of a finding depends on these three primitives being right and uniformly computed.

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
| 13 | **M** Micro-consolidation (accuracy/params/seed) | ~60 one-liners, 3 drift classes | M | Low | none |

**Suggested execution sequence:** J → B → C (foundations), then **M → F → I → G** (micro-DRY + zoo purity), then **A → E → H** (training + persistence), then **D** (measurement), then **L, K** (extensibility + boundary). Pillar M is deliberately placed early: it is low-risk, dependency-free, and the accuracy/count/seed consolidation unifies the exact primitives that Pillars A/D/G touch. Each pillar ships with `ruff format && ruff check && pyright && pytest --cov` green.

---

## Acceptance Criteria (single-sourced codebase)

1. **One training loop**: `CoreTrainer._train_step` has ≤3 branches (energy-model / model `train_step` / BPTT); all other runners are adapters. `grep -rl "loss.backward()" bioplausible/` outside `core/` & `training_mixin` returns nothing.
2. **One config tree**: `grep -rn "class ModelConfig" bioplausible/` → exactly 1 hit (`config/unified.py`); `class ExperimentConfig` → 1; `def load_config` → 1.
3. **One construction path**: `grep -rn "model_cls(" bioplausible/` outside `core/construction.py` returns no *instantiation* sites — only `construct_model` calls.
4. **One `BenchmarkResult`** and **one result funnel**: `record_experiment_result` is called by execution, hyperopt, validation, and mep-benchmarks; all five persistence backends written only from `result_sink`.
5. **One search space**: `grep -rn "SEARCH_SPACES\\b"` → 0 hits.
6. **zoo purity**: `zoo/propagators/` contains only `mep.py` and pure-gradient-transform submodules; `zoo/models/eqprop/` holds one registered engine + architecture subclasses.
7. **Acyclic import graph**: `execution/__init__.py` and `hyperopt/__init__.py` lazy-loaders deleted; import-time side effects limited to registry decorators.
8. **No global UI mutation from decision code**: `strategy.py`/`engine.py` route events through an injected `EventSink`.
9. **Dead code absent**: `execution/evolve_evaluator.py`, `knowledge/seed.py`, `campaign/`, `experiments/`, `data/transforms.py`, `search_space.SearchSpace` gone.
10. **Full suite green**, including parity/validation/hyperopt cross-checks; AutoScientist end-to-end smoke run.

---

## Current Status & Progress Log

Last updated: 2026-08-13 (this revision). Baseline when this log began:
**13 pre-existing test failures** (2003 collected) — all unrelated to the
refactor; since then 3 stale-test failures were fixed + the config relocation
shipped, so the current full-suite baseline is **2008 pass / 6 fail / 10 skip /
1 xfail** (the 6 remaining are all the documented numerical/parity drift — see
finding #5).

### Completed work (all sessions)

**Pillar J — Dead code removal (commits `c1a68b3`, `6ac0583`, `8bb4727`, `2e147c2`, `5e5d5a2`, `1fcd637`)**
- Deleted `execution/evolve_evaluator.py`, `campaign/`, `experiments/`, `hyperopt/parallel_runner.py:_worker_process_task`, `execution/cli.py`.
- Stripped duplicate `KnowledgeBase` from `knowledge/seed.py`; kept `KNOWLEDGE_BASE_SEED` data.
- Removed `hyperopt/__init__.py` lazy `__getattr__` for `create_constrained_optuna_config`/`get_constrained_search_space`.
- Deleted dead `config/__init__.py:127 load_config` (second definition).
- Fixed `NEBCBase` abstract contract (added `@abstractmethod _build_layers`).
- Fixed stale eqprop `FixedTrial` test (`5e5d5a2`) — added missing `RULE_SPACES` knobs.
- Fixed latent p2p crash: implemented `SearchSpace.crossover`/`mutate` (`2e147c2`).
- **Note**: `data/transforms.py` is NOT orphaned (imported by vision/continual_learning). `hyperopt/comparator.py` vs `comparison.py` are NOT duplicates (different domains). Plan table corrected.

**Pillar B — Single config hierarchy: first step done (uncommitted, this session)**
- `config/schema.py` mirror eliminated — the critical **name-collision blocker** resolved.
- New `bioplausible/config/omegaconf.py` holds the mutable OmegaConf-facing *document formats* (the former `schema.py` contents), with the two name-colliding classes renamed: `ModelConfig → ExperimentModelConfig` and `ExperimentConfig → ExperimentSchemaConfig`.
- Deleted `bioplausible/config/schema.py` — the parallel mirror is gone.
- `load_config`/`save_config` were already single-sourced in `unified.py`.
- All 5 direct consumers updated (`config/__init__.py`, `defaults.py`, `analysis/ablation.py`, `core/trainer.py` docstring, `tests/integration/test_phase0.py`).
- Pinned tests updated in `tests/unit/test_refactor2_bugfixes.py`.
- **Acceptance criterion #2 now fully met:** `grep -rn "class ModelConfig"` → 1 hit (`unified.py:123`); `class ExperimentConfig` → 1 (`unified.py:328`); `def load_config` → 1 (`unified.py:274`) after renaming the unrelated `zoo/mep/benchmarks/runner.py` `load_config` → `load_benchmark_config`.
- Verified: whole-package import clean; 2025 tests collect with 0 import errors; targeted runs green; full suite = 2008 pass / 6 fail (all pre-existing, finding #5).
- **Remaining Pillar B work:** (a) Delete `TrainerConfigSchema` (Pydantic, zero prod consumers) or regenerate via `TypeAdapter(TrainerConfig)`. (b) Eliminate `_KNOB_ALIASES` in `construction.py`. (c) Fold `unified.ExperimentConfig` into the facade's sectioned shape or vice versa.

**Pillar I — Settling unification: first step done (commit `c32e15f`)**
- Uniform Family A telemetry: `settle_single_state` now reports same dynamics surface as Family B (`steps_taken`, `converged`, `settle_time_s`).
- Convergence loops NOT merged (different convergence criteria) — only reporting surface unified.

**Pillar D — Metrics & Pareto sub-goals done (commits `5cb626f`, `fa62672`, `77428fc`)**
- `evaluation/base.py` `accuracy_fn` → delegates to canonical `core/losses.compute_accuracy`.
- `validation/tracks/tradeoff_tracks.py` local `count_parameters` deleted; imports canonical `bioplausible.utils.count_parameters`.
- `zoo/models/backprop.py` `BackpropTransformerLM.count_parameters` → delegates to canonical `utils.count_parameters`.
- Added `hyperopt.metrics.non_dominated_indices` — single generic non-dominated filter (per-axis `maximize` + `tol`).
- Routed all three frontier sinks to it: `analysis.results.compute_pareto_frontier`, `experiment/reporting.pareto_frontier`, `hyperopt.frontier.pareto_frontier`.
- Deleted the now-dead `_dominates` helper.
- Semantics proven identical to all three originals over 900 randomized cases; unit tests in `tests/unit/test_hyperopt_metrics.py` lock the shared behavior and the three delegate contracts.

**Pillar H — Single search space (committed `c777549`, prior session)**
- Deleted `SEARCH_SPACES` (~245 lines) and old `get_search_space`. Resolution now family/rule-driven off `RULE_SPACES`.
- `SearchSpace` class kept for p2p GA operators (`sample`/`crossover`/`mutate`/`apply_constraints`).

**Pillar F — EqProp consolidation (verified done, no code change needed)**
- The 6 variant models already collapsed into thin `EquilibriumMLP` subclasses in `zoo/models/eqprop/_energy.py`. Verified green: 62 tests.
- Plan's roadmap row was stale; corrected to DONE.

**ONNX export fix (this session; Pillar A territory — real bug)**
- `bioplausible/utils.py export_to_onnx`: ONNX/`torch.onnx.export` tracing resolves every `forward` default and passes them positionally, so `EquilibriumMLP.forward(x, beta=0.0, target=None, steps=None, *, return_trajectory, return_dynamics)` got 6 args → `TypeError`. Fixed by wrapping the model in a new `_InferenceOnly(nn.Module)` adapter whose `forward(x)` exposes only the tensor; export also now creates parent directories. Both `tests/integration/test_onnx.py` tests pass (previously 1 TypeError-escaped-skip + 1 skip).

**Pillar M — Micro-consolidation: accuracy one-liner sweep (this session, uncommitted)**
- Added a `scale: int = 1` param to the canonical `core.losses.compute_accuracy` (returns a 0-1 ratio by default; `scale=100` for percent), so the ~40 inline copies can be folded without any re-implementation.
- Swept **23 sites across 17 files** to `compute_accuracy` (2-D multiclass logits with index targets): `validation/{utils,tracks/application_tracks}.py`, `zoo/nebc_base.py`, `zoo/models/{spiking,target_prop,forward_only,hebbian,fa,base}.py`, `zoo/models/eqprop/{_contrastive,neural_cube}.py`, `domains/base.py`, `sklearn_interface.py`, `core/{ebm,model,training_mixin}.py`, `core/local_learning/{algorithm,task}.py`, `execution/robustness.py`.
- The `× 100` percent sites (`validation/utils.py:108`, `application_tracks.py:195`, `nebc_base.py:114`) use `scale=100`; the ratio sites pass no scale.
- **Deliberately left unchanged** (different semantics — NOT equivalent to `compute_accuracy`): `lightning_/module.py` (PL logs a raw tensor, not `.item()`), `zoo/models/eqprop/graph_eqprop.py:154` (masked logits/y), `zoo/models/hebbian.py:477` (count accumulation, not a fraction), and the 3-D-per-token `logits.argmax(dim=-1)` sites already routed through `reshape_for_cross_entropy`-style logic. The remaining 3-D `argmax(-1)` in `core/{model,training_mixin}` were folded safely because their targets are 1-D index (2-D logits in practice).
- Verified: 0 net new lint errors (527 E/F/I before and after), pyright 0 errors, and targeted suites green — `tests/unit/models/*`, `test_robustness`, `test_core_trainer`/`test_energy_model`, `tests/unit/domains`, `test_backprop_parity_smoke`, `test_reproducibility`, `test_smoke_training`, `test_zoo_integration`, all of `tests/unit/validation` (exception: the 2 pre-existing `backprop_parity` numerics-drift failures).

**Pillar B — Single config: dead `TrainerConfigSchema` deleted (this session, uncommitted)**
- Deleted the Pydantic `TrainerConfigSchema` + `validate_trainer_config` from `config/__init__.py` (~78 lines) and the only consumer `tests/unit/core/test_config_schema.py`. Verified zero production consumers via grep; removed the now-unused `pydantic`/`typing.Any` imports and both `__all__` entries. Whole-package import clean; `tests/unit/core/test_config_{unified,defaults}.py` green.
- This resolves Pillar B remaining sub-goal **(a)** ("delete TrainerConfigSchema ... zero prod consumers"). The Pydantic schema is *superset* of `TrainerConfig`; deleting it is safe because nothing in production loads configs through it (OmegaConf `validate_config` is the live gate). Sub-goals **(b)** `_KNOB_ALIASES` elimination and **(c)** folding `unified.ExperimentConfig` into the facade remain, both blocked on the deeper config unification (XL).

**Pillar B — Single config: dead `TrainerConfigSchema` deleted (this session, uncommitted)**
- Deleted the Pydantic `TrainerConfigSchema` + `validate_trainer_config` from `config/__init__.py` (~78 lines) and the only consumer `tests/unit/core/test_config_schema.py`. Verified zero production consumers via grep; removed the now-unused `pydantic`/`typing.Any` imports and both `__all__` entries. Whole-package import clean; `tests/unit/core/test_config_{unified,defaults}.py` green.
- This resolves Pillar B remaining sub-goal **(a)** ("delete TrainerConfigSchema ... zero prod consumers"). The Pydantic schema is *superset* of `TrainerConfig`; deleting it is safe because nothing in production loads configs through it (OmegaConf `validate_config` is the live gate). Sub-goals **(b)** `_KNOB_ALIASES` elimination and **(c)** folding `unified.ExperimentConfig` into the facade remain, both blocked on the deeper config unification (XL).

**Pillar K — broken `biopl-scientist` console script fixed (this session, uncommitted; finding #11 resolved)**
- `pyproject.toml` declared `biopl-scientist = "bioplausible.execution.cli:main_scientist"`, but `execution/cli.py` was deleted in `6ac0583`, so the installed script was a live broken public entry point (`ModuleNotFoundError` on import).
- Refactored the `__main__` block of `bioplausible/execution/engine.py` into a real `main(argv=None) -> int` function (the AutoScientist launcher: `--report/--dir/--tier-limit`), added the `collections.abc.Sequence` import, and repointed the script to `bioplausible.execution.engine:main`. Verified `uv run biopl-scientist --help` works and `tests/unit/execution/*` stay green.

**Pillar A — `graph/training.py` duplicate BPTT+eval loops consolidated (this session, uncommitted; finding #12 partial)**
- `train_backprop` and `train_pcn` were ~150 lines of near-identical epoch scaffolding (manual `loss.backward(); optimizer.step()`, per-epoch averaging, a `torch.no_grad()` eval loop).
- Extracted shared `_train_loop(...)` (epoch + train-loss/acc averaging + test-set eval) and `_collect_trainable(...)` helpers; the two `train_*` functions are now thin wrappers differing only in the per-batch `step_fn` (`_backprop_step` vs a `_pcn_step` closure) and `eval_fn`. The dead `param_to_key` map in `train_backprop` was dropped (it existed only to force `requires_grad`).
- Public signatures, `graph/__init__` exports, and the `_feedforward` import (used by `zoo/models/predictive_coding.py:143`) are unchanged. Preserved exact per-batch ordering for PC (compute loss *after* `optimizer.step()`), so logged metrics match observably.
- Net lint: 23 → 19 errors in the file; 0 new pyright errors. `tests/graph/test_training.py` (8) + `tests/unit/models/test_predictive_coding_model.py` (17) green.
- **Note:** this de-duplicates the loops but does NOT yet migrate them to `CoreTrainer` — `graph/training.py` still calls `loss.backward()` directly, so acceptance criterion #1's full "no `loss.backward()` outside `core/`" remains open. The full graph path should eventually become a `CoreTrainer`/`train_epoch` adapter (Pillar A).

**Pillar A — `CoreTrainer._validate` promoted to public `validate` (this session, uncommitted; finding #13 first half)**
- The `_TaskTrainer` adapter (`domains/trainer.py:141`) reached into the private `CoreTrainer._validate`. Renamed `_validate` → `validate` on `CoreTrainer` (`core/trainer.py:1268`), updated the internal call (`core/trainer.py:922`) and the adapter call, and updated the two test suites that poked the private name (`tests/integration/test_trainer_coverage.py`, `tests/unit/test_refactor2_bugfixes.py`). No `._validate(` references remain anywhere.
- This is the API-hygiene half of finding #13: adapters now consume a public validation API. Verified: 0 new lint errors, pyright 0 errors, `test_core_trainer` + `tests/unit/domains` + `test_trainer_coverage` + `test_refactor2_bugfixes` green.
- **Note:** the `train_*`/`val_*` metric re-keying in `_TaskTrainer.train_epoch` is *not* consolidated further — it is the adapter's contract for `hyperopt` callers (no other producer uses that exact shape today), so folding it into `CoreTrainer` would change `CoreTrainer`'s widely-used `loss`/`accuracy` return contract. Left as-is deliberately.

### Findings that change the plan (important for future work)

0. **Pillar H decision resolved (option a, as recommended).** `SEARCH_SPACES` is gone and `get_search_space` is family/rule-driven off `RULE_SPACES`. Caution: registry family metadata is the mapping basis, and it is coarser than rule keys — e.g. registered families are only `{backprop, eqprop, equitile, fa, forward_only, hebbian, predictive_coding, spiking, target_prop, tile}`; there is **no registered family** named `neural_cube`/`pepita`/`feedback_alignment`/`forward_forward`, yet `RULE_SPACES` has those keys. Resolution handles this by preferring the rule key when the model *name* is a rule key, so `neural_cube`/`pepita`/`feedback_alignment`/`forward_forward` still get their own (P0a-consistent) spaces. A cleaner long-term fix: align registry `family` metadata with rule keys (Pillar F/G territory).

1. **`data/transforms.py` is NOT orphaned** (plan Pillar J table was wrong on this row): it is imported by `data/vision.py`, `domains/vision.py`, and `zoo/mep/benchmarks/continual_learning.py` (`build_transform`, `normalization`, `create_dataloader`, `MNIST_TRANSFORM`). Do not delete.

2. **`hyperopt/comparator.py` vs `comparison.py` are NOT duplicates** (plan Pillar J row wrong): `comparator.py` is frontier-comparison (`compare_frontiers`, `FrontierComparison`, `OperatingPointMatch`); `comparison.py` is multi-algorithm ranking (`AlgorithmRanking`, `ComparisonStudy`, `compute_algorithm_rankings`). Different consumers (`analysis/results.py`+`cli/run.py` vs `hyperopt/__init__.py`+tests). No merge.

3. **Pillar B foundation is now DONE (this session); the XL merge remains.** The `config/schema.py` facade module was relocated to `config/omegaconf.py` with its two name-colliding classes renamed, all 5 direct consumers updated, `schema.py` deleted, and pinned tests updated — see Completed work. `config/__init__.py` still re-exports the public facade names via aliased imports, so the public API (`bioplausible.config.ModelConfig` = the OmegaConf facade) is unchanged. What remains of Pillar B is the deeper unification: (a) `TrainerConfigSchema` (Pydantic, `config/__init__.py:58`) + `validate_trainer_config` have **zero production consumers** (only `tests/unit/core/test_config_schema.py`) and the schema is a *superset* of `TrainerConfig` (adds `track_flops`, `save_checkpoints`, `use_wandb`, `wandb_project`, `deterministic`, `seed`, `device`, `tags`, `extra`) — so it is either dead-code-deletable (Pillar J) or regenerateable via `TypeAdapter(TrainerConfig)` (the plan's "never hand-maintained" goal); (b) the `_KNOB_ALIASES` layer in `core/construction.py` (`steps`→`max_steps`, `lr`→`learning_rate`); (c) folding `unified.ExperimentConfig` (frozen, description/tags leaf) into the facade's sectioned shape, or vice versa. Any of these can now proceed without the name-collision blocker.

4. **Pillar H done** — see Completed work. (This supersedes the old finding #4; the `crossover`/`mutate` latent crash remains fixed from `2e147c2`.)

5. **Pre-existing unrelated breakage** (not from this refactor; partially fixed):
   - **FIXED** `bioplausible/execution/cli.py` imported a nonexistent `ReportOrchestrator` (`6ac0583`) — deleted as dead code; package import graph is now clean.
   - **FIXED** stale eqprop `FixedTrial` test (`5e5d5a2`) and non-abstract `NEBCBase` (`1fcd637`).
   - **FIXED** `tests/integration/test_onnx.py` (this session) — forward-signature tracing + parent-dir creation; see Completed work.
   - **FIXED** `test_finite_nudge_execution`, `test_smoke_training::test_directed_ep`, `::test_finite_nudge_ep` (this session) — `metrics is None` because single-hidden eqprop `train_step` returns None under default `gradient_method="equilibrium"` (an O(1)-implicit vestige; `EquilibriumMLP` is not an `EnergyModel`, so no trainer phase consumes it). Tests now use `gradient_method="contrastive"` to exercise the model's own `train_step` contract.
   - **OPEN** (full-suite `--no-cov` run, this session: **1996 pass / 9 fail / 10 skip / 1 xfail** — down from the 13-failure baseline): the 6 remaining are accuracy/parity drift or kernel mismatch, all training/numerics-dependent and out of scope:
     - `test_equilibrium_parity::test_mlp_gradient_parity` (BPTT vs EqProp loss gap)
     - `test_triton_kernel::test_triton_match` (Triton kernel vs PyTorch numerical mismatch — `acceleration/` island, Non-Goals)
     - `tests/property/biology/test_biology_axioms.py::test_ep_gradient_matches_bptt[eqprop_mlp]` and `::test_deq_gradients_match_bptt_wired_up` (EP-BPTT cosine < 0.5)
     - `tests/unit/validation/test_backprop_parity.py::test_backprop_parity[eqprop_mlp]` and `[directed_ep]` (bio acc vs backprop baseline gap > tolerance)

6. **Pillar F is already done in code — the plan's roadmap row is stale.** The 6 variant models were already collapsed into thin `EquilibriumMLP` subclasses in `zoo/models/eqprop/_energy.py` (verified green; see Completed work). The plan's roadmap still lists F as pending (step #6) and the status table said "not started" — corrected. The remaining Pillar F "nice-to-have" (an architecture registry so named *non-MLP* eqprop variants are thin subclasses overriding only `_build_layers`/`forward_dynamics`) is an optimization, not a correctness gap; defer unless a new architecture is added. Similarly the `_PROPAGATOR_TO_MODEL` alias work (Pillar G) is untouched and remains the real open work in the zoo.

7. **Pillar D is best entered via the metrics/`count_parameters` seam** (this session consolidated `accuracy_fn` and `tradeoff_tracks.count_parameters` to `core`). Next low-risk D sub-goals in ascending size: (a) fold the remaining inline `(logits.argmax(dim=1) == y).float().mean()` accuracy copies and the ~4 `count_parameters` variants (`validation/tracks/tradeoff_tracks.py` done; `zoo/models/backprop.py:230` and `benchmarks/efficiency_analysis.py:91` are method wrappers that can call `utils.count_parameters`); (b) the `BenchmarkResult` unification (5 classes); (c) the report renderer consolidation. Each is independently shippable.

8. **Pareto dominance is now unified (this session, commit `fa62672`).** Added `hyperopt.metrics.non_dominated_indices` — a single generic non-dominated filter (per-axis `maximize` + `tol`) — and routed all three frontier sinks to it: `analysis.results.compute_pareto_frontier` (3 obj), `experiment.reporting.pareto_frontier` (2 obj + config-key dedup), `hyperopt.frontier.pareto_frontier` (4 obj + accuracy eps). Deleted the now-dead `_dominates`. Semantics proved identical to all three originals over 900 randomized cases; unit tests in `tests/unit/test_hyperopt_metrics.py` lock the shared behavior and the three delegate contracts. Remaining Pillar D: `BenchmarkResult` unification (5 classes) and the report-renderer consolidation.

9. **Pillar C's remaining `create_model` helpers are the plan's stated mock.patch targets.** `lightning_/module.py:22` and `execution/robustness.py:33` both do `Registry.get(MODEL, name) → cls(**kwargs)` with slightly different defaults (robustness adds `hidden_dim`/`num_layers`/`.to(device)`). The canonical builder `core/construction.construct_model(model_cls, config, input_dim, output_dim)` differs in signature (sampled-config + required dims) and is `nyi` on the loose-kwargs path these callers use, so adapting them is *not* a mechanical rename. Two options: (i) keep them as thin per-site adapters that build a scalar config dict and call `construct_model` behind the module-level patchable name (call sites unchanged); (ii) delete and rewrite the ~6 consuming tests. Given `test_lightning_integration.py:368/417` and `test_robustness.py:17` patch these symbols directly, option (i) with green tests is the low-risk path — but it ships near-zero line reduction, so Pillar C should be bundled with the trainer/lightning adapter work (Pillar A) rather than attempted in isolation.

10. **Pillar I (settling unification) is numerics-sensitive — leave until F is fully tested or bundle with G.** `_settling.py` already routes P1 single-hidden models through `settle_state` (protocol `EquilibriumSettleProtocol`); `settle_single_state` (Family A) and `settle_activations_list` (Family B) share no convergence logic yet. The protocol unification is the correct next step but should be done when the full EqProp/propagator test matrix is green to catch numerical regressions. Bundle with Pillar G (propagator/model unification) since both touch the settling path.

11. **Broken console-script entry point — `biopl-scientist` references a deleted module (regression from `6ac0583`).** **RESOLVED (this session, uncommitted).** `pyproject.toml` still declared `biopl-scientist = "bioplausible.execution.cli:main_scientist"`, but `execution/cli.py` was deleted as dead code in commit `6ac0583` (its `main_reporter`/`_run_reporter` imported a nonexistent `ReportOrchestrator`). Verified: `import bioplausible.execution.cli` → `ModuleNotFoundError`. So the `biopl-scientist` console script was **installed but unusable**. **Fix applied:** refactored `engine.py`'s `__main__` into `main(argv=None)` and repointed the script to `bioplausible.execution.engine:main` (the AutoScientist launcher), per the recommended option (a). `uv run biopl-scientist --help` verified working.

12. **`graph/training.py` has two near-identical BPTT+eval loops (Pillar A evidence).** **PARTIALLY RESOLVED (this session, uncommitted).** Both `train_gradient_descent`-style paths (feedforward `_feedforward` branch and the settle/inference `train_pcn` branch) duplicated the same epoch loop: manual `loss.backward(); optimizer.step()`, per-epoch averaging, and a `torch.no_grad()` eval loop with `correct += (...).sum()`. The two loops now collapse into one shared `_train_loop`/`_collect_trainable` helper (the only difference is the per-batch `step_fn`/`eval_fn`: `_feedforward` vs `infer.settle`) — see Completed work. Together with `training/rl.py` and `zoo/models/deployments/{base,vision,graph,timeseries}.py`, the remaining direct `loss.backward()` sites still violate acceptance criterion #1 and belong in the Pillar A adapter sweep, but the intra-file duplication that finding #12 described is gone.

13. **`_TaskTrainer.train_epoch` re-implements metric normalization + validation (Pillar A tail).** **PARTIALLY RESOLVED (this session, uncommitted).** `domains/trainer.py:120` already delegates training to `CoreTrainer.from_task`, but its `train_epoch` hand-rolls the `train_loss`/`train_acc` re-keying and a manual `self._trainer._validate(1)` call. The private-API violation is fixed: `CoreTrainer._validate` is now public `validate` (see Completed work). The metric-shape re-keying remains but is deliberately left as the adapter's `hyperopt` contract (only producer of the `train_*` shape) — folding it into `CoreTrainer` would change its widely-consumed `loss`/`accuracy` return contract, so it is deferred unless/until Pillar A unifies all adapter metric shapes under one helper.

14. **Accuracy fold done (this session); `count_parameters` follow-up is the remaining Pillar M seam.** The inline-accuracy sweep is complete (see Completed work). ~20 raw `sum(p.numel() for p in model.parameters())` sites remain across `analysis/energy_landscape.py`, `cli/lab.py`, `hyperopt/experiment.py`, `core/{profiling,spectral_mixin,trainer}.py`, `zoo/models/eqprop/eqprop_lm_variants.py`, `zoo/models/tile_lm.py`, `evaluation/{base,cross_domain}.py`, `deployment.py`, `benchmarks/{rigorous,compare_nanoGPT}.py`, `experiment/param_estimator.py`. `bioplausible.utils.count_parameters` (trainable-only default + `_orig_mod` compile unwrap) is strictly more robust, but folding the `benchmarks/*`/`evaluation/*` parity sites is deferred: `trainable_only=True` could shift counts on models with frozen params (spectral-norm wrappers, pretrained embeddings), which risks benchmark parity drift and is not cheaply verifiable. The `efficiency_analysis.py` breakdown and `hebbian.py:477` accumulation are genuinely *different* counting semantics (per-component attribution / running count) and should NOT be folded.

15. **Pillar B remaining sub-goals are now (b) and (c) only.** With `TrainerConfigSchema` deleted (sub-goal a done this session), the only remaining Pillar B work is `_KNOB_ALIASES` elimination (`core/construction.py`: `steps`→`max_steps`, `lr`→`learning_rate`) and folding `unified.ExperimentConfig` into the OmegaConf facade. Both are blocked on the same XL config-unification step and are independent of sub-goal (a), which is now fully resolved.

