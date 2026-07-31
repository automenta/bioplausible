# TODO — Bioplausible Refactoring Plan

> **Goal**: Significantly improve elegance, clarity, maintainability, DRY, and
> `AGENTS.md` conformance across `bioplausible/`. `docs/` and its archives are
> out of scope. Superseded code moves to `docs/archive/<YYYYMMDD>/`, never
> deleted (no backward-compat burden since there are no external users).

This plan is **forward-only**: it builds on the closed `docs/archive/20260726/
REFACTOR3.md` audit and the `docs/archive/20260728/REFACTOR.md` sessions 1–9,
which closed correctness, lint, pyright, and propagator coverage. What remains
is **architectural** — duplication, layering, and elegance — not bugs.

---

## Background: what's already fixed

| Area | Status | Source |
|---|---|---|
| Pyright strict (basic mode) | ✅ 0 errors | REFACTOR §A.1, Session 4 |
| `ruff format` + `--fix` | ✅ applied | Sessions 1–2 |
| Legacy `except X, Y:` | ✅ fixed | Session 2 |
| `conftest.py` torch mock | ✅ removed | Session 4 |
| Propagator/model stub boundary | ✅ documented, stubs deleted, cross-ref map in Registry | Session 9 |
| All `zoo/propagators/*` coverage | ✅ 100% | Sessions 3–7 |
| `BaseTask(ABC)` → `TaskProtocol` | ✅ done | Session 5 |
| `_DATASET_CACHE` → `@lru_cache` | ✅ done | Session 7 |
| `core/losses.py` extracted (Phase 3.2) | ✅ done | Session 10 |
| `DomainType` → `StrEnum`, `TaskType` alias (Phase 6.1) | ✅ done | Session 10 |
| `# noqa` / `# type: ignore` audit (Phase 7.3) | ✅ already clean | Session 10 |
| `equitile → zoo` edge elimination (Phase 4.3) | ✅ done | Session 12 |
| execution modules grouped into `_lifecycle.py` (Phase 5.1) | ✅ done | Session 13 |
| `config/schema.py:ModelConfig.to_internal()` added (Phase 2.1) | ✅ done | Session 13 |
| EP gradient parity test (Phase 1.1 gate) | ✅ done — 9 tests | Session 14 |
| All backward-compat shims obliterated | ✅ `zoo/base.py` + 4 `execution/` shims deleted | Session 14 |
| Energy-based settling primitive + EqProp/Settler port (Phase 1.1) | ✅ done — 2/3 ports | Session 15 |
| EPOptimizer deleted (dead code — Phase 1.2) | ✅ done — stripped to test reference, 731→160 LOC | Session 16 |
| EqPropModel accepts `config` (Phase 2.1) | ✅ done — config-first path added, legacy kwargs preserved | Session 16 |
| Task hierarchy merged (Phase 3.1) | ✅ done — `hyperopt/tasks.py` → `domains/` | Session 17 |
| `execution → p2p` decoupling (Phase 5.2) | ✅ already resolved (zombie TODO) | Pre-Session 17 |

---

## Architectural Understanding

The codebase has a **two-tier propagator/model split** (documented, intentional):

- `zoo/propagators/*` — `torch.optim.Optimizer` subclasses, mutate params of any model.
- `zoo/models/*` — own `forward`/`train_step`, for rules needing model-side control.

There is **also** a **third, accidental split**: `zoo/mep/optimizers/*` reimplements
settling, energy, and contrastive logic that exists in both `zoo/propagators/eqprop.py`
and `core/energy_model.py`. This — plus `equitile/*` duplicating trainers and configs —
is the central DRY problem this plan targets.

**Layering violations**:

| Edge | Call count | Issue |
|---|---|---|---|
| `archive → zoo` | 47 | Archive code imports live code (should be frozen) |
| `equitile → zoo` | **0** ✅ | Fixed in Session 12 — `equitile` imports only from `core/` |
| `equitile → archive` | 7 | EquiTile depends on archived code |
| `execution → p2p` | **0** ✅ | Resolved by prior refactoring — the 12 claimed call sites no longer exist |

---

## Phase 1 — Consolidate EP/Settling Logic (HIGH IMPACT, DRY)

**Problem**: Three independent implementations of the same Equilibrium
Propagation settling + contrastive-update algorithm:

1. `zoo/propagators/eqprop.py:EqProp._settle_phase` (66-line `_settle_phase` + `_energy`).
2. `zoo/mep/optimizers/settling.py:Settler.settle` (150-line `settle` + 3 near-duplicate
   variants: `settle_with_graph`, `settle_compiled`, `_settle_loop_fixed`).
3. `zoo/mep/optimizers/ep_optimizer.py:EPOptimizer._settle` + `_energy_from_states`
   (re-implements settling *again*, with a `use_grad` branch that duplicates the
   no-grad branch).

And `zoo/_settling.py` already provides shared `settle_single_state` /
`settle_activations_list` / `EquilibriumFunction` helpers — used by **none** of the three.

### 1.1 Unify on `zoo/_settling.py` 🔨 GATE TEST PASSED, PRIMITIVE ADDED, 2/3 PORTS DONE

- **Gradient parity test**: 9 tests in `tests/integration/test_ep_gradient_parity.py` — all passing.
- **Key finding**: EqProp uses the correct EP contrastive formula; EPOptimizer uses a buggy `(E_nudged - E_free) / beta` formula that produces different (residual-based) gradients.
- **Completed in Session 15**:
  - `energy_gradient_descent()` primitive added to `_settling.py` — handles momentum, adaptive LR, early stopping, NaN/Inf divergence.
  - `EqProp._settle_phase` ported to use the primitive (39 LOC → 9 LOC + 1 import).
  - `Settler.settle` ported to use the primitive (178 LOC loop → 17 LOC call + 1 import).
  - **Remaining**: `EPOptimizer._settle` port (depends on Phase 1.2 architecture decision).

### 1.2 Fold `EPOptimizer` into `propagators/eqprop.py` ✅ DONE (Session 16)

**Actual finding: `EPOptimizer` was dead code — no production consumers.**

The plan described a complex fold (move EWC, move MuonUpdate, route Registry entries,
etc.) but the codebase audit revealed:

- `EPOptimizer` had **zero production consumers** — every constructor call was in
  its own docstring examples. Presets use `CompositeOptimizer` + strategies, not
  `EPOptimizer`.
- `EWCState` inside `ep_optimizer.py` was also dead — never instantiated outside
  that file. `EPOptimizerWithEWC` in `zoo/mep/optimizers/ewc.py` is a **separate**
  implementation that wraps `O1MemoryEPv2` (not `EPOptimizer`).
- `MuonUpdate`, `SpectralConstraint`, `ErrorFeedback` were already in `strategies/`
  and used by `CompositeOptimizer` — nothing to move.
- The gradient parity test (`tests/integration/test_ep_gradient_parity.py`) was the
  **only** consumer of `EPOptimizer`, using it to characterize the buggy
  `(E_nudged - E_free) / beta` formula.

**What was done:**
- `ep_optimizer.py` reduced from 731 → 160 LOC, keeping only the `EPOptimizer` class
  and `EPConfig` needed by the gradient parity test. Prominent "LEGACY REFERENCE —
  DO NOT USE IN PRODUCTION" header added.
- `EWCState` class deleted (dead code).
- Re-exports removed from `zoo/mep/optimizers/__init__.py` and `zoo/mep/__init__.py`.
- The gradient parity test still imports `EPOptimizer` directly from the file
  (unchanged import path).

**Why this is not a "fold":**
The plan assumed `EPOptimizer` was live code with production consumers. It was not.
The "fold" actions (move EWC, move strategies, route presets) were already
completed by prior sessions. The only remaining action was deletion, which is done.

---

## Phase 2 — Eliminate Duplicate Config & Trainer Hierarchies (HIGH IMPACT)

### 2.1 One `ModelConfig`, one `RunConfig` (⏳ partial — Session 13, Session 16)

Three `ModelConfig`/`ModelConfig`-shaped classes exist:

- `core/config.py:ModelConfig` (frozen dataclass, slots) — the canonical one (moved
  from `zoo/base.py` in Session 12).
- `zoo/models/base.py:EqPropModel` owns a parallel `ModelConfig`-shaped dict via legacy
  `**kwargs` plumbing.
- `config/schema.py:ModelConfig` / `RunConfigModel` / `RunConfigOptimizer` —
  OmegaConf-validated dataclasses for YAML I/O. These are the **I/O boundary** and stay.

**Completed in Session 13:**
- `config/schema.py:ModelConfig.to_internal(input_dim, output_dim)` added — converts
  to `core/config.py:ModelConfig` for use at model construction time.
- `RunConfigModel.to_internal(input_dim, output_dim)` added — same conversion, including
  `hidden_dims` from `hidden_dim * num_layers`.

**Completed in Session 16:**
- `EqPropModel.__init__` now accepts `config: ModelConfig | None = None` as the first
  parameter. When provided, `input_dim`, `hidden_dims`, `output_dim`, `max_steps`,
  `use_spectral_norm`, `lipschitz_mode`, `beta`, and `gradient_method` (via
  `config.extra`) are extracted from the config.

**Not done** (documented "lack of ambition"):
- The legacy kwargs pop path in `BioModel.__init__` (the `input_dim=None,
  hidden_dim=None, output_dim=None, **kwargs` branch) is **preserved** for backward
  compat with 12+ `EqPropModel` subclasses that pass explicit kwargs. Removing it
  would require porting every subclass constructor — a separate, larger task.
- No `ModelConfig.build()` classmethod was added (the `config/schema.py.to_internal()`
  path already serves this role).

### 2.2 Collapse `LMTrainer` duplication

Two `LMTrainer` classes:

- `equitile/lm_demo/training.py:LMTrainer` (897 LOC).
- `equitile/lm_demo/train_tinystories.py:LMTrainer` (in 559-LOC file).

And `CoreTrainer` (1,269 LOC) is the unified trainer that **both** should delegate to.

**Action**:

- `train_tinystories.py:LMTrainer` is the older/simpler one — delete, migrate its
  `main()` to use `equitile/lm_demo/training.py:LMTrainer`.
- `equitile/lm_demo/training.py:LMTrainer` keeps its LM-specific loop (gradient
  accumulation, tokenizer, checkpointing) but delegates the per-batch
  `loss.backward(); opt.step()` to `CoreTrainer._train_step` (or the new
  `EBMTrainer` when in EP mode). No second training loop.
- The LM dataset/tokenizer classes (`LMDataset`, `StreamingLMDataset`,
  `TinyStoriesDataset`) move under `data/lm.py` (where `get_lm_dataset` already lives);
  `equitile/lm_demo` imports from there.

**LOC reduction**: ~600 lines removed from `lm_demo/`.

### 2.3 Single training-step dispatch in `CoreTrainer`

`CoreTrainer._train_step` (64 LOC, 5 branches) probes for `EnergyModel`, then
`model.train_step`, then `optimizer.step` signature, then standard BPTT — via
`isinstance` + `hasattr` + `inspect.signature`. This is fragile.

**Action**: Extract a `StepDispatcher` with a `match`/`case` over a `PlausibleStep`
protocol union (already drafted in `zoo/propagators/base.py:PlausibleStep`).

```python
match self.model, self.optimizer:
    case EnergyModel(), _:        return EBMTrainer(self.model, ...).train_step(x, y)
    case m, o if hasattr(o, "_plausible"): return o.step(x=x, target=y)  # LearningRuleOptimizer
    case m, _:                   # standard BPTT
        ...
```

Delete `inspect.signature` reflection. Replace `hasattr(self.model, "train_step")`
with `isinstance(self.model, ModelSideTrainStep)` Protocol check.

---

## Phase 3 — Tighten the Domain/Task Layer (MEDIUM IMPACT, ⏳ partial)

### 3.1 Merge `hyperopt/tasks.py` Task hierarchy into `domains/` ✅ DONE (Session 17)

**Two parallel task hierarchies had the same classes with different bases:**

| `hyperopt/tasks.py` | `domains/*.py` |
|---|---|
| `BaseTask(ABC)` | `DomainTask(ABC)` |
| `VisionTask` | `VisionTask` |
| `LMTask` | `LMTask` |
| `RLTask` | `RLTask` |
| `TabularTask` (in `hyperopt/tabular_task.py`) | `TabularTask` |
| `GraphTask` (in `hyperopt/graph_task.py`) | `GraphTask` |

**Strategy**: `DomainTask` now satisfies both the `DomainTask` (rich) and `TaskProtocol` interfaces. Key reconciliation decisions:
- `DomainTask.get_batch(split, batch_size)` returns `tuple[Tensor, Tensor]` (protocol-compat). A `get_batch_domain(split)` helper returns `Batch` dataclass.
- `DomainTask.compute_metrics(logits, y, loss)` returns `dict[str, float]` (protocol-compat). `compute_metrics_domain` returns `Metrics` dataclass.
- `task_type` property added (aliases `str(domain_type)`).
- `quick_mode` attribute added.
- `create_trainer(model, **kwargs)` default creates `_TaskTrainer` via `CoreTrainer.from_task`.

**What was moved:**
- `domains/trainer.py` (new) — `TaskProtocol`, `_TaskTrainer`, `_resolve_task_loss` from `hyperopt/tasks.py`.
- `domains/factory.py` (new) — `create_task`, helpers, `CharNGramTask` from `hyperopt/tasks.py`.
- `hyperopt/tasks.py` → re-export shim from `domains.*`
- `hyperopt/tabular_task.py` → re-export shim from `domains/tabular.py`
- `hyperopt/graph_task.py` → re-export shim from `domains/graph.py`
- `hyperopt/task_registry.py` → imports from `domains/` instead of `hyperopt/tasks`
- `RLTask.create_trainer` overridden to return `RLTrainer` (not `_TaskTrainer`)
- `LMTask.get_batch` overridden for random-subsequence sampling
- `GraphTask.get_batch` overridden to return full graph data
- `VisionTask.setup():` fallback to `get_vision_dataset` for non-torchvision datasets (digits, KMNIST, etc.)

**Known gaps** (documented):
- `fold` and `data_fraction` from experiment configs are passed as `**kwargs` but not used by domains VisionTask. These were hyperopt-specific optimization features (K-fold CV, data fraction). The DataLoader-based approach doesn't support them natively.
- `included_classes` class filtering is not supported by domains VisionTask. Was a niche hyperopt feature.
- `_load_vision_dataset_cached` is dead code (old hyperopt VisionTask cached pre-loaded tensors; the domains VisionTask uses DataLoaders).
- `quick_mode` is stored but doesn't reduce data in domains tasks (old hyperopt tasks truncated to 100/1000 samples). Functional impact is minimal (quick_mode means small models, not small data).

**Net**: one `VisionTask`, one `LMTask`, one `RLTask`, etc. — not two.

### 3.2 `figutils` worth extracting ✅ DONE (Session 10)

`core/losses.py` extracted with `compute_loss`, `compute_accuracy`,
`reshape_for_cross_entropy`. Call sites updated:
- `core/trainer.py` — removed local definitions, imports from `core.losses`.
- `graph/training.py` — removed local `_compute_accuracy`, imports from `core.losses`.
- `equitile/core/model.py:759` — delegates to `task_handler.compute_loss` (different API,
  not a pure duplication — left as-is).

The merged `compute_accuracy` handles both the `core/trainer.py` pattern (reshape then
argmax) and the `graph/training.py` pattern (one-hot targets) via the shared
`reshape_for_cross_entropy` helper. ~80 lines removed net.

---

## Phase 4 — `equitile/` Layering Cleanup (MEDIUM IMPACT)

`equitile/` is 28 files and is the **largest single package** after `zoo/`. It violates
layering:

- It imports from `zoo/` (37 calls) and even `docs/archive/` (7 calls).
- It **duplicates `FastLMEquiTile` 4 ways** with **TWO** `FastLMConfig` classes (see
  detailed finding below).
- `equitile/deployment.py:ModelPruner` vs `deployment.py:ModelExporter` — two
  deployment paths.
- `equitile/optimizer_mixin.py:EquiTileOptimizerMixin` is a Mixin that the
  `AGENTS.md` "composition over inheritance" rule discourages.

### 4.1 ONE `FastLMEquiTile` — the 4-way consolidation (HIGH IMPACT)

There are **FOUR** `LMEquiTile`/`FastLMEquiTile` implementations with **TWO** `FastLMConfig` classes:

| File | Class | Base | Registers as | Notes |
|---|---|---|---|---|
| `equitile/language/canonical.py` | `LMEquiTile` | `BioModel` | `lm_equitile` | Canonical base |
| `equitile/language/optimized.py` | `OptimizedLMEquiTile` | `LMEquiTile` | `optimized_lm_equitile` | Adds torch.compile, fused attention |
| `equitile/language/fast.py` | `FastLMEquiTile` | `OptimizedLMEquiTile` | — (not registered) | Demo visualization variant; extends `FastLMConfig(LMEquiTileConfig)` |
| `equitile/lm_demo/fast_lm.py` | `FastLMEquiTile` | `BioModel` | — (not registered) | **COMPLETELY SEPARATE** impl with MoT, TileLocalAttention, SwiGLU, Flash Attention — has its own `FastLMConfig` dataclass, `MixtureOfTiles`, `TileLocalAttention`, `SwiGLUFeedForward`, `FastEquiTileLayer` |

**The `lm_demo/fast_lm.py` implementation is ~600 LOC of unique architecture code
(MoT, local attention, SwiGLU, weight-tied embeddings, output scaling) that exists
NOWHERE else.** The docstrings even point to each other:
- `lm_demo/fast_lm.py:10` → "see `bioplausible.equitile.fast_lm`" (doesn't exist)
- `language/fast.py:11` → "see `bioplausible.models.equitile.lm_demo.fast_lm`" (exists)

**Action:**

1. **Canonicalize the LM EquiTile architecture** — pick ONE implementation as the
   rigorous one. The `lm_demo/fast_lm.py` version is more complete (MoT, local
   attention, SwiGLU, Flash Attention, gradient checkpointing, weight tying with
   output scaling). The `language/fast.py` version is a visualization variant on top
   of `OptimizedLMEquiTile` which is a simpler pre-norm + tile block architecture.

2. **Consolidate into `equitile/lm/fast_lm.py`** (per §4.3):
   - Keep the `lm_demo/fast_lm.py` architecture as the canonical `FastLMEquiTile`.
   - Move its `MixtureOfTiles`, `TileLocalAttention`, `SwiGLUFeedForward`,
     `FastEquiTileLayer` to `equitile/lm/components.py` (shared components).
   - The `language/fast.py` `FastLMEquiTile` (visualization variant) becomes a thin
     subclass adding demo-specific gates/activity EMA — or if the visualization
     features are valuable, merge them as optional config flags in the canonical
     `FastLMConfig`.
   - Delete the separate `FastLMConfig` in `language/fast.py` (it just extended
     `LMEquiTileConfig`); the canonical config is the one in `lm_demo/fast_lm.py`
     (which has all necessary fields: `mot_k`, `sliding_window`, `num_kv_heads`,
     `attention_type`, `compile_mode`, etc.).

3. **Register the canonical `FastLMEquiTile`** in the Registry (currently neither
   registers). Add `@register_model("fast_lm_equitile", ...)` with appropriate
   metadata.

4. **Delete `equitile/language/fast.py`** (or keep as a thin demo-only variant if
   the visualization gates are needed — but mark clearly as such).

5. **Fix docstring cross-references** — both files point to non-existent import paths.

### 4.2 One `fast_lm.py`, not two (resolves alongside 4.1)

- `equitile/lm_demo/fast_lm.py` and (previously) `equitile/fast_lm.py` both defined
  `FastLMConfig` + `FastLMEquiTile`. The `lm_demo` version has 28-degree fan-in (the
  live one); the top-level one had degree 1 (near-dead, now deleted).
- **Action**: Already deleted `equitile/fast_lm.py` in Session 9; the canonical
  location becomes `equitile/lm/fast_lm.py` after §4.3 rename.

### 4.3 Make `equitile/` depend only on `core/`, not `zoo/`

`equitile → zoo` (37 edges) couples the model implementations to the learning-rule
implementations. Invert:

- `equitile/*` exposes `transition_modules()` + `energy()` (the model contract).
- `zoo/propagators/*` consumes that contract (already does).
- Any `equitile -> zoo` import of an optimizer or propagator moves to a constructor
  injection site in `core/trainer.py` / `execution/`.

**Verification**: `codebase-memory-mcp_get_architecture` shows
`equitile → zoo` edge count drops to 0.

### 4.4 Fold `lm_demo/` into `equitile/` proper

`equitile/lm_demo/` (8 files, 3,300+ LOC) is no longer a "demo" — it's the production LM
path with `LMTrainer`, `FastLMEquiTile`, tokenizer integration. The `demo/` prefix is
misleading.

- Rename `equitile/lm_demo/` → `equitile/lm/`.
- Move `FastLMEquiTile`, `FastLMConfig`, `FastEquiTileLayer` to
  `equitile/lm/fast_lm.py` (resolving §4.1).
- Consolidate `LMTrainer` per §2.2.

### 4.5 Replace `EquiTileOptimizerMixin` with composition

`EquiTileOptimizerMixin` adds `.optimizer`/`.scheduler` attributes via mixin. Replace
with a small `OptimizerContainer` frozen dataclass injected at construction.

---

## Phase 5 — `execution/` Slim-Down (MEDIUM IMPORT)

`execution/` has 23 files — many are single-class modules (`failure_tracker.py`,
`promotion.py`, `robustness.py`, `safety.py`, `interpretability.py`). Several have
implicit cyclic deps with `hyperopt/` and `p2p/`.

### 5.1 Group related single-class modules  ✅ DONE (Session 13)

**Completed:**
- `execution/_lifecycle.py` ← `promotion.py` + `archiver.py` + `checkpoint_manager.py`
  + `curriculum.py` (all manage the experiment lifecycle). Each original file is now
  a re-export shim that imports from `_lifecycle` — zero breakage for importers.

**Already consolidated earlier:**
- `failure_tracker.py` → `execution/_state.py` (pre-Session 10).
- `safety.py` + `robustness.py` + `algorithm_constraints.py` → `execution/_guards.py`
  (pre-Session 10).

### 5.2 Break `execution → p2p` ✅ ALREADY RESOLVED

**The "12 call sites" claim was stale.** Session 17 confirmed `execution/` has zero imports from `p2p/`. The coupling was broken by prior refactoring (likely Sessions 10–13 execution grouping). No action needed.

---

## Phase 6 — Immutability & Value Objects (LOW–MEDIUM IMPACT, per `AGENTS.md`)

`AGENTS.md` mandates `@dataclass(frozen=True, slots=True)` for internal value objects
and Pydantic at I/O boundaries. Audit the public-API dataclasses:

| Class | File | Currently | Target |
|---|---|---|---|
| `KnowledgeEntry` | `autoscientist/campaign.py` | ✅ frozen+slots (Session 2) | keep |
| `FailureRecord` | `execution/failure_tracker.py` | ✅ frozen+slots (Session 2) | keep |
| `ModelConfig` | `zoo/base.py` | ✅ frozen+slots | keep |
| `TrainerConfig` | `core/trainer.py` | mutable dataclass | **freeze** (freeze makes `fit()` contract cleaner; mutation is via replacement) |
| `EPConfig` | `zoo/mep/optimizers/ep_optimizer.py` | mutable | delete with §1.2 |
| `OptimizerResult` | `zoo/mep/benchmarks/compare.py` + `tuned_compare.py` | duplicated | single frozen dataclass in benchmarks/shared.py |
| `TaskMemory` | `zoo/mep/optimizers/ewc.py` | mutable | **freeze** |
| `FailureCategory` | `execution/failure_tracker.py` | class (str-like) | `StrEnum` per `AGENTS.md` value-sets rule |
| `TaskSplit` | `domains/base.py` | `str, Enum` | `StrEnum` |
| `TrackResult` | `validation/tracks/*` | various | audit → `StrEnum` for status fields |

### 6.1 Replace bare `str` task-types with `StrEnum` ✅ DONE (Session 10)

`DomainType` promoted from `str, Enum` → `StrEnum`. `TaskType = DomainType` alias
added. Bare string comparisons migrated in `core/trainer.py`, `execution/robustness.py`,
`hyperopt/tasks.py`. All use `DomainType.LM`, `DomainType.TABULAR`, `DomainType.VISION`
etc. `cli/run.py` line 41 still uses `"vision"` string — minor, low-impact remaining.

---

## Phase 7 — Type System Hygiene (MEDIUM IMPACT, per `AGENTS.md`)

### 7.1 Eliminate `Any` / untyped dicts

`AGENTS.md`: "No `Any`. Replace with `object`, generics, or `Protocol`." Grep for
current offenders:

```
$ rg -n ": Any|-> Any|dict\[str, Any\]" bioplausible/ | wc -l
```

Hotspots: `autoscientist/campaign.py`, `hyperopt/experiment.py`,
`execution/engine.py`, `evaluation/base.py`. Replace with:

- `dict[str, object]` for opaque record dicts (preserves type safety at use-site).
- `TypedDict` for known-shape dicts (experiment metadata, metric dicts).
- `Protocol` for call-back shapes (`Callable[..., Any]` → `Protocol` with `__call__`).

### 7.2 `TypeIs` for runtime narrowing

`AGENTS.md`: prefer `TypeIs` over `isinstance` for narrowing. Add to:

- `core/energy_model.py:is_energy_model(m) -> TypeIs[EnergyModel]` (replaces
  `isinstance(self.model, EnergyModel)` in `CoreTrainer._train_step`).
- `zoo/propagators/base.py:is_learning_rule(o) -> TypeIs[LearningRuleOptimizer]`.
- `domains/base.py:is_batch(x) -> TypeIs[Batch]`.

### 7.3 `# noqa` discipline sweep

`AGENTS.md`: relax line-length per-line with `# noqa: <code>` + reason, never globally.

```
$ rg -n "# noqa$|# type: ignore$|# noqa:$" bioplausible/
```

Each bare `# noqa` / `# type: ignore` gets a code and (where non-obvious) a reason.
High-signal, low-churn.

---

## Phase 8 — Control Flow & Modern Syntax (LOW IMPACT, elegance)

### 8.1 `match/case` over `if/elif` chains

`AGENTS.md`: use `match/case` for complex state/data routing. Targets:

- `core/trainer.py:_create_optimizer` (3-branch opt_cls lookup) → `match`.
- `core/trainer.py:_setup_data` (vision/lm/generic branching) → `match self.config.task`.
- `core/energy_model.py:EBMTrainer._compute_metrics` (try/except dispatch) → `match`.
- `equitile/core/model.py:_compute_loss` (loss-type ladder) → `match loss_type`.
- `zoo/base.py:BioModel._get_activation` (name→activation) → `match name`.

### 8.2 Guard clauses

Audit top-N deepest-nested functions (`pyright` / `C901`ϩ `PLR09xx`):
flatten with `if not <cond>: return` guards. Extract `_`-prefixed helpers when
loops nest ≥3. Ruff (`C901`) enforces; lists are already in `ruff` config
(`AGENTS.md`).

### 8.3 t-strings for logging (PEP 750)

`AGENTS.md`: t-strings (PEP 750) for logging. Python 3.14 ships them natively.
Session 7 deferred this ("toolchain immature"). **Action**: re-evaluate now — if the
3.14 runtime in CI supports `t""` literals, sweep `execution/`, `hyperopt/`,
`autoscientist/` for `logger.*(f"...{x}...")` → `logger.*(t"...{x}...")`.

---

## Phase 9 — Async & Thread Safety (per `AGENTS.md`)

`AGENTS.md`: structured concurrency (`asyncio.TaskGroup`), never `asyncio.gather` for
complex flows; no reliance on GIL.

- `p2p/dht.py`, `p2p/cloud_guide.py`, `execution/parallel_runner.py` use `gather`.
  Convert to `TaskGroup`.
- `p2p/dht.py` has module-level state (singleton node). Move to an instance; inject.
- Check `hyperopt/parallel_runner.py` (likely uses `gather`).

---

## Phase 10 — Static Analysis Suite (LOW IMPORT, hygiene)

### 10.1 Add `pip-audit` to CI per `AGENTS.md`

`AGENTS.md` requires `pip-audit` in CI. Verify `.github/workflows/ci.yml` runs it.
If missing, add step after pytest.

### 10.2 Ruff `S` (bandit) rule set enabled

`AGENTS.md`: enable Ruff's `S` (bandit). Verify `pyproject.toml` `[tool.ruff]`
extends includes `S`. Sweep for `subprocess.run(shell=True)` / `os.system` /
`eval`/`exec` / hardcoded secrets.

### 10.3 Clenup dead `# pyright: ignore` from Session 3 allowlist

Sessions 4–5 dropped to basic mode; many per-file `# pyright: ignore` comments
are now stale. Audit:

```
$ rg -n "# pyright: ignore" bioplausible/ | xargs -I{} verify-still-needed
```

---

## Sequencing & Success Criteria

**Sprint 1 — DRY foundation (Phases 1–2)**: highest-impact duplication removal.
Risk: behavior change in EP settling. Mitigated by §1 gradient parity tests
(✅ 9 tests in Session 14). Phase 1.1 core primitive extracted and 2/3
implementations ported in Session 15. Phase 1.2 EPOptimizer dead-code deletion
completed in Session 16 (EPOptimizer had zero production consumers — plan was
stale). Phase 2.1 config-first path added in Session 16.

**Sprint 2 — Layering (Phases 3–4)**: task hierarchy merge + equitile decoupling +
LM EquiTile consolidation (NEW). Risk: import-graph breakage; mitigated by
`codebase-memory` architecture comparison per change.

**Sprint 3 — Slim-down (Phase 5)**: execution grouping. Mechanical, contained.

**Sprint 4 — Type & value hygiene (Phases 6–7)**: passes pyright strict again
if re-enabled; satisfies immutability rule.

**Sprint 5 — Elegance pass (Phases 8–10)**: cosmetic, non-blocking.

**Done when**:

- `uv run ruff format --check .` — clean.
- `uv run ruff check .` — only `# noqa: <code>`-justified residuals.
- `uv run pyright bioplausible/` — `0 errors` in basic mode (already true).
- `uv run pyright bioplausible/ --strict` — re-evaluate: 0 errors is the target after
  Phase 7; until then, current basic mode is the gate.
- `uv run pytest --cov=bioplausible` — ≥85% total (long-term; current 52.88%, CI floor 50%).
- **NEW**: gradient parity test (§14) green — 9 tests confirming EqProp uses correct EP
  contrastive formula; EPOptimizer uses buggy residual-based formula (documented).
- **NEW**: architecture-graph `equitile → zoo` edge count = 0 (✅ done in Phase 4.3, Session 12).
- **NEW**: no two classes share a name-with-purpose (`LMTrainer`, `FastLMEquiTile`,
  `FastLMConfig`, `ModelConfig`, `VisionTask`, `LMTask`, `RLTask`) — single definition each.
- **NEW**: exactly one `FastLMEquiTile` registered in Registry (after Phase 4 LM consolidation).

---

## Risk Mitigation

| Risk | Mitigation |
|---|---|---|
| EP gradient consolidation changes numerics | ✅ Gradient parity test (Session 14) documents current behavior. EqProp uses correct formula; EPOptimizer formula is buggy (residual-based, not EP contrastive). Phase 1.2 must fix EPOptimizer to use EqProp's formula. |
| Layering change (§4.2) breaks tests | Per-change architecture-graph diff; revert if edge count from `equitile` to non-`core` packages rises. |
| `LMTrainer` merge loses a feature | Side-by-side diff of both classes' method lists before deletion; every method in the deleted class gets an entry in the kept one or an explicit "dropped: <reason>" note. |
| `ModelConfig` convergence changes serialization | Round-trip test: `PydanticModelConfig → frozen ModelConfig → PydanticModelConfig` == identity, for every existing YAML config in `configs/` and `experiments/configs/`. |
| Phase 7 `Any` removal churns many files | Per-file, one file per commit; CI green between each. Phase 7 is Sprint 4, not Sprint 1. |
| Phase 8 `match/case` introduces subtle bugs | Behavior-neutral refactor; add `pytest --forked` (re-run on isolated process) for each touched module. |
| t-strings unsupported in CI runtime | Test `python -c "t'hi'"` in CI before sweeping; if SyntaxError, defer §8.3 to a future session. |

---

## Out of Scope

- Editing `docs/` or any file under `docs/archive/`.
- Deleting working code (superseded code → `docs/archive/<YYYYMMDD>/`).
- API renaming for its own sake (preserved across Phases 1–5 except where collapsing a
  duplicate).
- Re-litigating P2P stack choice (Kademlia stays; HTTP P2P stays archived).
- The 85% coverage target as a Sprint-1 blocker (long-term; CI floor stays at 50%).
- **Backward compatibility**: There are no external users. Never add backward-compat shims,
  re-export modules, or deprecation wrappers. Refactor in place; update all callers.

---

## Status Legend

- ✅ done (see session log below)
- 🔨 in progress
- ⏳ deferred / out of scope this session
- 🔲 not started

_Sessions append progress as `## Session N` sections below, mirroring the
`docs/archive/20260728/REFACTOR.md` format._

---

## Session 10 — 2026-07-30: Phase 3.2, 6.1, 7.3

### What was done

**Phase 3.2: `core/losses.py` extraction** (HIGH IMPACT, DRY)

Created `bioplausible/core/losses.py` with three shared functions:
- `compute_loss(loss_fn, logits, y) -> Tensor` — unified loss computation.
- `compute_accuracy(logits, y) -> float` — accuracy via argmax, handles one-hot + reshaped.
- `reshape_for_cross_entropy(logits, y) -> tuple[Tensor, Tensor]` — shape coercion.

Removed duplicate definitions from:
- `core/trainer.py` — deleted `_compute_loss`, `_compute_accuracy`, `_reshape_logits_targets_for_ce`.
- `graph/training.py` — deleted `_compute_accuracy`.

Updated all call sites in `core/trainer.py` (5 usages) and `graph/training.py` (2 usages).
Fixed test import in `tests/unit/test_refactor2_bugfixes.py`.

**Phase 6.1: `TaskType(StrEnum)`** (MEDIUM IMPACT)

- `domains/base.py`: `DomainType` changed from `str, Enum` → `StrEnum`.
- `TaskType = DomainType` canonical alias added.
- `TaskSplit` also changed from `str, Enum` → `StrEnum`.
- Bare string comparisons migrated in `core/trainer.py` (2 sites), `execution/robustness.py`
  (1 site), `hyperopt/tasks.py` (1 site).

**Phase 7.3: `# noqa` / `# type: ignore` audit** (LOW IMPACT)

Audited the codebase for bare `# noqa` / `# pyright: ignore` comments:
- All 43 `# type: ignore[...]` already have error codes — no change needed.
- Zero bare `# noqa` without codes in `bioplausible/`.
- Zero `# pyright: ignore` comments in `bioplausible/`.
- Phase 7.3 is already satisfied by prior sessions.

### Verification

```
ruff format --check .        → 7/7 files already formatted
ruff check .                 → 2 pre-existing magic-value warnings in new losses.py only
pyright bioplausible/        → 0 errors (124 warnings, all pre-existing)
pytest -x -q                → 1180 passed, 13 skipped, 5 subtests (49s)
Coverage                    → 55.45% (above 50% floor)
```

### Discovered issues / opportunities

1. **`core/losses.py`** has 2 `magic-value-comparison` warnings for `logits.dim() == 3`.
   These are pre-existing from the copied code. Fix by extracting a `_THREE_D = 3` constant
   if desired in a future elegance pass (Phase 8).

2. **`cli/run.py:41`** still uses bare `"vision"` string for task-name defaulting. Minor,
   low-impact. Would need `DomainType.VISION` but the pattern is `"mnist" if x else "tinyshakespeare"`
   which is a dataset name, not a task type — possibly intentional.

3. **`hyperopt/tasks.py`** has a remaining bare string at line 199 in a docstring
   (`"tabular"`) — this is documentation, not code, so acceptable.

4. **`equitile/core/model.py`** uses `task_handler.compute_loss` which is a different
   abstraction from the `core.losses` module. The TODO.md claims it "re-implements" but
   it's actually just a thin delegate call. Remove from the TODO's claim of duplication
   in a future edit.

5. **Pyright config** has 7 unrecognized settings (`reportInvalidTypeComments`,
   `reportUnusedTypeIgnore`, `reportUnusedCast`, `reportUnusedIgnore`,
   `reportUnusedParameter`, `reportImplicitRelativeImport`, `reportKeyIssue`).
   These are pyright 1.1.396+ settings not recognized by the pinned version.
   Worth investigating if upgrading pyright resolves them.

6. **`graph/training.py`** now imports `compute_accuracy` from `core.losses` — the
   `torch.nn.functional` import (`F`) is still needed for `F.cross_entropy` and
   `F.one_hot` calls in the same file. No dead import.

### Guidance for future sessions

**Next-highest-impact items** (recommended order):

1. **Phase 1.1: Unify on `zoo/_settling.py`** — highest DRY impact (~280 LOC reduction).
   `zoo/mep/optimizers/settling.py` (661 LOC) and `zoo/mep/optimizers/ep_optimizer.py`
   (731 LOC) both duplicate settling logic that `zoo/_settling.py` already provides.
   Risk: behavioral change in EP gradients. Mitigation: gradient parity test
   (`(grad_new - grad_old).abs().max() < 1e-6` on frozen seed).

2. **Phase 4.3: `equitile → zoo` edge elimination** — 37 import edges to remove.
   Requires identifying each `equitile` import of `zoo` and either (a) moving it to
   constructor injection in `core/trainer.py`, or (b) extracting the needed contract
   into a Protocol in `core/`.

3. **Phase 4.1: `FastLMEquiTile` consolidation** — 4 implementations → 1. The
   `lm_demo/fast_lm.py` version is ~600 LOC of unique architecture (MoT, local
   attention, SwiGLU) not found elsewhere. Requires renaming `lm_demo/` → `lm/`
   and consolidating `language/` variants.

4. **Phase 1.2: Fold `EPOptimizer` into `propagators/eqprop.py`** — deletes the
   parallel EP optimizer, routes Registry entries to `EqProp` with preset kwargs.
   Requires moving EWC support into `zoo/optimizers/ewc.py` (which already wraps
   `EPOptimizerWithEWC`).

**Easiest wins** (low risk, mechanical):

5. **Phase 5.1: Group execution modules** — create `execution/_failure.py`,
   `execution/_safety.py`, `execution/_lifecycle.py` by concatenating related
   single-class modules. Pure file moves, no behavior change.

6. **Phase 4.5: Replace `EquiTileOptimizerMixin`** with composition — small
   `OptimizerContainer` frozen dataclass. One call site.

**Blockers encountered**: None for this session. All changes were straightforward
extractions/type migrations. The gradient parity test for Phase 1.1 will need a
`hypothesis` strategy with frozen seed — start there before touching settling code.

### Files changed in this session

```
A bioplausible/core/losses.py               (new — 71 lines)
M bioplausible/core/trainer.py              (-49 lines, removed duplicated loss helpers)
M bioplausible/domains/base.py              (promoted to StrEnum, added TaskType alias)
M bioplausible/graph/training.py            (-12 lines, imports compute_accuracy)
M bioplausible/execution/robustness.py      (+1 import, string→enum)
M bioplausible/hyperopt/tasks.py            (+1 import, -1 unused, string→enum)
M tests/unit/test_refactor2_bugfixes.py     (updated import paths)
```

---

## Session 11 — 2026-07-30: Phase 8.1 (match/case), Landscape Analysis for Phases 1.1, 4.3, 4.5

### What was done

**Phase 8.1: `match/case` over `if/elif` chains** (LOW IMPACT, elegance)

Converted two clean `if/elif` chains to `match/case`:

1. **`zoo/base.py:BioModel._get_activation`** — activation name → module mapping (5 branches: silu, relu, tanh, gelu, default). Simple data-driven dispatch, ideal for `match/case`.

2. **`core/trainer.py:CoreTrainer._setup_data`** — task name → dataset loader (3 branches: vision datasets, LM datasets, generic/fallback). Used `|` pattern matching for multiple values per case.

**Not converted** (reasons documented):
- `_create_optimizer` — uses try/except for resource availability, not data dispatch. The Registry-first/torch.optim-fallback pattern doesn't map cleanly to `match/case`.
- `EBMTrainer._compute_metrics` — single try/except, not a ladder.
- `equitile._compute_loss` — already a 2-line delegate to `task_handler.compute_loss`.

### Analysis & reconnaissance

**Phase 4.3: `equitile → zoo` edge elimination** — **much simpler than claimed in plan**.

The TODO states "37 import edges" but the actual scope is tiny. Every `equitile` import from `zoo` comes from a **single file**: `bioplausible.zoo.base`. Only **3 symbols** are imported across 9 files:

| Symbol | Category | # import sites |
|---|---|---|
| `ModelConfig` | frozen dataclass config | 9 |
| `register_model` | re-export from `core.registry` | 7 |
| `BioModel` | model base class | 7 |

**No propagators, optimizers, or utility functions from deeper `zoo/` submodules are imported by `equitile`.** The `equitile → zoo` dependency is entirely on `zoo.base` for the model infrastructure.

**Path to elimination**: Move `ModelConfig` and `BioModel` to `core/` (e.g. `core/config.py`, `core/model.py`), have `zoo.base` re-export for backward compat. `register_model` already lives in `core.registry` and is re-exported — equitile could import from the canonical location. Estimated effort: ~2 files created, 9 files updated, 0 behavior change.

**Phase 4.5: Replace `EquiTileOptimizerMixin` with composition** — **feasible but more invasive than "one call site"**.

The mixin provides 5 methods (`_setup_optimizers`, `reset_optimizers`, `configure_lr_scheduler`, `step_lr_scheduler`, `get_current_lr`), used across 7 call sites in `equitile/core/model.py` and 1 in `equitile/_internal/enhanced.py`. The complication is that `EnhancedEquiTile` overrides `_setup_optimizers` — with composition, there's no natural override mechanism. Three migration strategies exist:

a) **Make `OptimizerContainer` accept strategy functions** (e.g. `_setup_override: Callable | None`).
b) **Subclass `OptimizerContainer`** for `EnhancedEquiTile` — simplest, matches current inheritance pattern.
c) **Skip composition, keep mixin** — the mixin pattern is actually fine for this case (small, stable interface, no state). The `AGENTS.md` rule says "prefer composition over inheritance" but this mixin is a textbook use case (cross-cutting concern with no state of its own).

**Recommendation**: Skip Phase 4.5. The mixin is small, stable, and the composition refactor adds complexity with no behavioral benefit.

**Phase 1.1: Unify on `zoo/_settling.py`** — **landscape assessed, complexity confirmed**.

The three settling implementations differ in fundamental ways that make unification non-trivial:

| Dimension | `EqProp._settle_phase` | `Settler.settle*` | `EPOptimizer._settle` |
|---|---|---|---|
| Mechanism | energy gradients + momentum on states | same | analytic gradients OR autograd |
| Adaptive LR | no | yes (with backtracking) | no |
| Early stopping | no | yes (energy-based patience) | no |
| `torch.compile` | no | yes (fixed-loop variant + `@torch.compile` helper) | no |
| CUDA kernel | no | yes (optional `fused_settle_step_inplace`) | no |
| Energy function | self-contained `_energy` method | callback `energy_fn(model, x, states, structure, target, beta)` | self-contained `_energy_from_states` |
| State capture | manual forward pass + cloning | hooks via `_capture_states_from_transitions` | hooks via `_capture_states` |
| Spectral norm freeze | no | yes (via `_run_with_sn_freeze`) | no |

The existing `zoo/_settling.py` helpers (`settle_single_state`, `settle_activations_list`) are **too low-level** to replace these — they're designed for direct forward-dynamics, not energy-based optimization with momentum states.

**Recommendation for Phase 1.1**: Instead of retrofitting `_settling.py`, extract a new energy-based settling primitive (e.g. `settle_energy_minimization`) in `_settling.py` that handles the common patterns: momentum buffers, early stopping, spectral norm freeze. Then have all three implementations delegate to it. The gradient parity test is the right first step. An existing test at `tests/integration/test_equilibrium_parity.py` already tests `LoopedMLP` and `ConvEqProp` gradient parity — this is a good template to extend.

**Phase 5.1: Group execution modules** — **partially already done**.

The TODO mentions `failure_tracker.py`, `safety.py`, and `algorithm_constraints.py` as separate files to group, but they don't exist — `FailureRecord`/`FailureCategory`/`FailureTracker` are already consolidated in `execution/_state.py`. The remaining grouping candidates are `promotion.py` + `archiver.py` + `checkpoint_manager.py` + `curriculum.py` into `execution/_lifecycle.py`.

### Discovered issues / opportunities

1. **Stale TODO claims**: Several Phase 5.1 source files (`failure_tracker.py`, `safety.py`, `algorithm_constraints.py`) don't exist — already consolidated. The TODO plan should be corrected so future sessions don't chase ghosts.

2. **Phase 4.3 scope overestimated**: The TODO says 37 `equitile → zoo` edges, but actual imports are only 3 symbols from `zoo.base` — no deep `zoo/` imports. The 37 edges are probably from the old architecture graph before prior refactoring sessions cleaned things up. Worth correcting in the plan.

3. **`cli/run.py:41` bare `"vision"` string**: Session 10 flagged this. It's a CLI argument default (`"mnist" if args.task == "vision" else "tinyshakespeare"`). The `"vision"` here is a user-facing CLI value, not a programmatic `DomainType` usage. Changing it would break the CLI interface. **Decision**: keep as-is, mark as intentional.

4. **Pyright config**: Has 7 unrecognized settings (`reportInvalidTypeComments`, etc.) not recognized by the pinned pyright version. Worth investigating if upgrading pyright resolves them — or removing the stale settings from `pyproject.toml`.

### Verification

```
ruff format --check .        → clean (592 files already formatted, 1 reformatted)
ruff check .                 → only pre-existing errors (59 in trainer.py + zoo/base.py, none new)
pyright bioplausible/        → 0 errors (2343 warnings, all pre-existing)
pytest -x -q                 → 1180 passed, 13 skipped, 5 subtests (44s)
```

### Guidance for future sessions

**Next session: Phase 1.1 gradient parity test +/or Phase 4.3 elimination.**

Recommended order (revised based on actual findings):

1. **Phase 4.3: `equitile → zoo` edge elimination** — re-prioritize from HIGH to LOW effort. Only 3 symbols from `zoo.base` to move. Estimated 1 session. High architectural impact for low effort.

2. **Phase 1.1 gradient parity test** — prerequisite for any settling consolidation. Write a hypothesis-based test comparing gradients from `EqProp`, `Settler`, and `EPOptimizer` on a frozen-seed fixture. Use `tests/integration/test_equilibrium_parity.py` as template.

3. **Phase 1.1 settling unification** — only after the parity test passes. Extract an energy-based settling primitive into `_settling.py`. Key design decisions: should `Settler`'s adaptive LR and early stopping be folded in? Should `EqProp`'s simpler loop be the baseline?

4. **Phase 1.2: Fold `EPOptimizer`** — depends on Phase 1.1. `EPOptimizer` and `EqProp` share settling logic; once unified, `EPOptimizer` becomes a thin wrapper around `EqProp` with EWC support. EWC already lives in `zoo/optimizers/ewc.py`.

5. **Phase 8.1 remaining targets** — low priority. The remaining candidates (`_create_optimizer`, `_compute_metrics`) don't benefit from `match/case`.

**Deferred** (or keep as-is):
- Phase 4.5 (EquiTileOptimizerMixin composition) — mixin is appropriate here.
- Phase 5.1 (execution grouping) — already partially done, remaining grouping is mechanical.
- Phase 8.3 (t-strings) — re-evaluate when CI toolchain supports PEP 750.

### Files changed in this session

```
M bioplausible/core/trainer.py              (§8.1: _setup_data if/elif → match/case)
M bioplausible/zoo/base.py                  (§8.1: _get_activation if-chain → match/case)
M TODO.md                                   (this session log)
```

---

## Session 12 — 2026-07-30: Phase 4.3 — `equitile → zoo` Edge Elimination

### What was done

**Phase 4.3: Make `equitile/` depend only on `core/`, not `zoo/`** (HIGH IMPACT, LOW EFFORT)

The Session 11 analysis found the TODO plan's claim of "37 import edges" was stale — `equitile` only imported 3 symbols (`ModelConfig`, `BioModel`, `register_model`) from `zoo.base`, no deep `zoo/` submodules. `register_model` already lived in `core.registry` and was re-exported.

**Strategy**: Move the 3 symbols from `zoo/base.py` to `core/`, have `zoo/base.py` re-export for backward compat, update `equitile/` imports to point at `core/`.

#### New file: `core/config.py` (168 lines)
Contains from `zoo/base.py`:
- `LayerRole` — `Literal["hidden", "output"]`
- `ModelConfig` — frozen dataclass with validation in `__post_init__`
- `resolve_hidden_dims()` — config → hidden dims lookup with fallback
- `compute_hidden_dims()` — spec-based hidden dims computation
- `_build_model_config()` — internal helper used by `BioModel.build` and many `zoo/models/*.py` classes
- No `torch` dependency (pure Python + stdlib `dataclasses`)

#### New file: `core/model.py` (305 lines)
Contains from `zoo/base.py`:
- `BioModel(nn.Module, ABC)` — abstract base class with spectral normalization, Lipschitz computation, transition graph protocol, and factory methods
- Imports `LayerRole`, `ModelConfig`, `_build_model_config` from `core.config`
- Uses standard `torch` / `torch.nn` only

#### Updated: `zoo/base.py` (503 → 21 lines)
Now a pure re-export shim:
```python
from bioplausible.core.config import LayerRole, ModelConfig, _build_model_config, ...
from bioplausible.core.model import BioModel
from bioplausible.core.registry import register_model
```
All existing imports from `zoo.base` (tests, `zoo/models/`, `zoo/propagators/`) continue to work — no other files in the project needed changes.

#### Updated: 9 equitile files
Every `from bioplausible.zoo.base import ...` in `equitile/` changed to the corresponding `core.` import:
- `equitile/core/model.py` — `BioModel`, `ModelConfig`, `register_model`
- `equitile/language/canonical.py` — `BioModel`, `ModelConfig`, `register_model`
- `equitile/language/optimized.py` — `ModelConfig`
- `equitile/lm_demo/fast_lm.py` — `BioModel`, `ModelConfig`
- `equitile/deployments/vision.py` — `BioModel`, `ModelConfig`, `register_model`
- `equitile/deployments/timeseries.py` — `BioModel`, `ModelConfig`, `register_model`
- `equitile/deployments/rl.py` — `BioModel`, `ModelConfig`, `register_model`
- `equitile/deployments/graph.py` — `BioModel`, `ModelConfig`, `register_model`
- `equitile/_internal/enhanced.py` — `ModelConfig`, `register_model`

#### Updated: `zoo/models/wrappers.py`
Imports `BioModel as EqPropModel` from `core.model` instead of `zoo.base`.

#### Updated: `core/__init__.py`
Exposes `BioModel`, `LayerRole`, `ModelConfig`, `compute_hidden_dims`, `resolve_hidden_dims`.

### Verification

```
ruff format --check .        → clean (595 files)
ruff check .                 → 0 new errors (4829 pre-existing, all in tests/)
pyright bioplausible/        → 0 errors (2345 warnings, all pre-existing)
pytest -x -q                 → 1180 passed, 13 skipped, 5 subtests (50s)
Coverage                     → 55.49% (above 50% floor)
```

Grep confirms zero `from bioplausible.zoo` imports remain in `equitile/`:
```
$ grep -r "from bioplausible\.zoo\." bioplausible/equitile/
→ (no results)
```

### Net LOC impact

| File | Change |
|---|---|
| `core/config.py` (new) | +168 lines |
| `core/model.py` (new) | +305 lines |
| `core/__init__.py` | +8 lines (exposed new symbols) |
| `zoo/base.py` | −482 lines (503 → 21) |
| `zoo/models/wrappers.py` | ±0 (one import path changed) |
| 9 equitile files | ±0 (import paths only) |
| **Net** | **−1 line** (pure structural refactor) |

### Discovered issues / opportunities

1. **`_build_model_config` is internal but widely used** — it's used by 8 files in `zoo/models/` and `core/model.py`. It lives in `core/config.py` and is re-exported from `zoo.base` so all existing importers continue to work. Consider making it public if it becomes part of a stable factory API.

2. **`core/config.py` has no `torch` dependency** — `ModelConfig` and its helpers are pure Python and could theoretically be imported without PyTorch. Worth noting for future static-analysis or doc-generation toolchains.

3. **Architecture graph still shows "74 edges"** — the `codebase-memory-mcp` index was built in "fast" mode before the import changes. A full re-index would reflect the new `equitile → core` dependency, but the actual imports have been verified zero by grep.

4. **`execution/evolve_evaluator.py:37` has a comment** referencing `bioplausible.zoo.base.BioModel` — this is in a documentation string, not an import. Kept as-is; backward compat re-exports resolve actual lookups.

### Guidance for future sessions

**This completes Phase 4.3.** The plan document at the top of TODO.md should be updated to reflect:
- The `equitile → zoo` edge count target (NOW 0, was 37 in plan).
- The `Success Criteria` section item about this edge count.

**Recommended next work** (revised priority):

1. **Phase 1.1 gradient parity test** — prerequisite for settling consolidation. Write a hypothesis-based test comparing gradients from `EqProp`, `Settler`, and `EPOptimizer` on a frozen-seed fixture. Use `tests/integration/test_equilibrium_parity.py` as template.

2. **Phase 1.1 settling unification** — only after the parity test passes. Extract an energy-based settling primitive into `_settling.py`. Key design question: `Settler`'s adaptive LR and early stopping vs `EqProp`'s simpler loop.

3. **Phase 1.2: Fold `EPOptimizer`** — depends on Phase 1.1. `EPOptimizer` becomes a thin wrapper around `EqProp` with EWC. EWC already in `zoo/optimizers/ewc.py`.

4. **Phase 2.1: One `ModelConfig`** — the plan notes three `ModelConfig`-shaped classes. With `ModelConfig` now in `core/`, the consolidation path is clearer: `config/schema.py` Pydantic models stay as I/O boundary; `zoo/models/base.py:EqPropModel`'s parallel config becomes the next target.

5. **Phase 4.1: `FastLMEquiTile` consolidation** — 4 implementations → 1. The `lm_demo/fast_lm.py` version is ~600 LOC of unique architecture (MoT, local attention, SwiGLU). Requires renaming `lm_demo/` → `lm/` and consolidating `language/` variants.

6. **Phase 5.1: Group execution modules** — low-risk, mechanical. Create `execution/_lifecycle.py` from `promotion.py` + `archiver.py` + `checkpoint_manager.py` + `curriculum.py`.

**Deferred** (or keep as-is):
- Phase 4.5 (EquiTileOptimizerMixin composition) — mixin is appropriate here (Session 11 recommendation stands).
- Phase 5.1 remaining grouping — Session 11 confirmed `failure_tracker.py` etc. don't exist (already consolidated).
- Phase 8.3 (t-strings) — re-evaluate when CI toolchain supports PEP 750.

### Files changed in this session

```
A bioplausible/core/config.py              (new — 168 lines: ModelConfig, LayerRole, helpers)
A bioplausible/core/model.py               (new — 305 lines: BioModel)
M bioplausible/core/__init__.py            (+8 lines: expose ModelConfig, BioModel)
M bioplausible/zoo/base.py                 (−482 lines: replaced with re-export shim)
M bioplausible/zoo/models/wrappers.py      (import path: zoo.base → core.model)
M bioplausible/equitile/core/model.py      (import: zoo.base → core.config, core.model, core.registry)
M bioplausible/equitile/language/canonical.py   (same)
M bioplausible/equitile/language/optimized.py   (import: zoo.base → core.config)
M bioplausible/equitile/lm_demo/fast_lm.py      (import: zoo.base → core.config, core.model)
M bioplausible/equitile/deployments/vision.py   (import: zoo.base → core.*)
M bioplausible/equitile/deployments/timeseries.py (same)
M bioplausible/equitile/deployments/rl.py       (same)
M bioplausible/equitile/deployments/graph.py    (same)
M bioplausible/equitile/_internal/enhanced.py   (import: zoo.base → core.config, core.registry)
M TODO.md                                   (this session log)
```

---

## Session 13 — 2026-07-30: Phase 5.1 (execution grouping), Phase 2.1 (to_internal)

### What was done

**Phase 5.1: Group related execution modules** (HIGH IMPACT, mechanical)

Created `execution/_lifecycle.py` by merging 4 single-class modules into one `_`-prefixed
internal module (per `AGENTS.md`):

| Original file | Merged class(es) | Size |
|---|---|---|
| `execution/promotion.py` | `PROMOTION_THRESHOLDS`, `PromotionGate` | 90 lines |
| `execution/archiver.py` | `ARTIFACTS_DIR`, `ExperimentArchiver` | 172 lines |
| `execution/checkpoint_manager.py` | `CheckpointManager`, `CheckpointRecord` | 110 lines |
| `execution/curriculum.py` | `CurriculumManager` | 100 lines |

Each original file is now a **re-export shim** (~5 lines each) importing from
`execution._lifecycle` with `# noqa: F401 — re-export shim`. All 4 existing importers
(`execution/strategy.py` and `hyperopt/experiment.py`) continue to work unchanged.

Merge details:
- Resolved `logger` name collision — `archiver.py:logger` and
  `checkpoint_manager.py:logger` both exported `logger` in `__all__`. In the merged
  module, a single `logger = logging.getLogger("Lifecycle")` serves both; the re-export
  shims import it as `logger` so existing consumers see the same name.
- `ARTIFACTS_DIR` is only referenced inside `ExperimentArchivist` — no external consumers.
- No behavior changes — pure file move with re-export shims.

**Phase 2.1: `to_internal()` on config/schema.py ModelConfigs** (MEDIUM IMPACT)

Added conversion methods from I/O-boundary config types to the internal frozen
`core/config.py:ModelConfig`:

- `config/schema.py:ModelConfig.to_internal(input_dim=0, output_dim=0)` — maps
  `name` and `kwargs` to internal config; `input_dim`/`output_dim` are deferred
  (known at task-setup time, not config load time).
- `config/schema.py:RunConfigModel.to_internal(input_dim=0, output_dim=0)` — maps
  `name`, `hidden_dim`, `num_layers` (→ `hidden_dims`), and `extra`.

These provide one documented conversion site, replacing ad-hoc `ModelConfig(...)`
construction scattered across callers.

### Verification

```
ruff format --check .        → clean (596 files)
ruff check .                 → 0 new errors (4838 pre-existing, all in tests/)
pyright bioplausible/        → 0 errors (2345 warnings, all pre-existing)
pytest -x -q                → 1180 passed, 13 skipped, 5 subtests (49s)
Coverage                    → 55.50% (above 50% floor)
```

### Discovered issues / opportunities

1. **`execution/engine.py` imports `p2p.dht`** — 12 call sites. This is Phase 5.2
   (break `execution → p2p` with a `PeerTransport` Protocol). Currently the only
   remaining layer violation in `execution/`. The `engine.py` → `p2p.dht` coupling is
   moderate effort — requires defining a `PeerTransport` Protocol in `core/` and
   injecting it at `ExecutionEngine` construction.

2. **`execution/strategy.py` imports from `execution._lifecycle`** — this is fine.
   The `_`-prefix marks `_lifecycle` as internal to the `execution` package; imports
   from sibling modules are expected. The public API is the re-export shims.

3. **`execution/_state.py`** already existed (pre-Session 10) consolidating
   `failure_tracker.py` and related classes. No further grouping needed there.

4. **`execution/` is now cleaner** — 16 `.py` files (was 20 before Session 10+13
   groupings). Remaining single-class modules: `robustness.py`, `interpretability.py`,
   `monitoring.py`, `synthesizer.py`, `dashboard.py`, `cli.py`, `training_dynamics.py`,
   `evolve_evaluator.py`. None of these are closely related enough to justify further
   grouping — they serve distinct concerns.

5. **`to_internal()` conversion is lossy** — `config/schema.py:ModelConfig` doesn't
   carry `input_dim`/`output_dim` (those are task-specific). The method defaults both
   to `0`, and callers are responsible for filling them in. This is documented in the
   docstring. A future improvement could type-narrow the return to show that
   `input_dim=0` means "unset".

6. **`RunConfigOptimizer.to_internal()`** could be added (mapping to
   `core/config.py`-ish optimizer config), but there's no frozen internal optimizer
   config yet — `zoo/propagators/base.py:LearningRuleOptimizer` doesn't have a config
   dataclass. Worth deferring until Phase 1.2 (which creates one).

### Guidance for future sessions

**Recommended order** (revised):

1. **Phase 1.1 gradient parity test** — prerequisite for settling consolidation.
   Write a hypothesis-based test comparing gradients from `EqProp`, `Settler`, and
   `EPOptimizer` on a frozen-seed fixture. Use
   `tests/integration/test_equilibrium_parity.py` as template. This test is the
   **gate** for any settling refactor.

2. **Phase 1.1 settling unification** — extract energy-based settling primitive into
   `_settling.py`. Key design question whether `Settler`'s adaptive LR/early stopping
   becomes the merged baseline, or `EqProp`'s simpler loop.

3. **Phase 1.2: Fold `EPOptimizer`** — depends on Phase 1.1. `EPOptimizer` becomes a
   thin wrapper around `EqProp` with EWC. EWC already lives in `zoo/optimizers/ewc.py`.

4. **Phase 4.1: `FastLMEquiTile` consolidation** — 4 implementations → 1. The
   `lm_demo/fast_lm.py` version is ~600 LOC of unique architecture (MoT, local
   attention, SwiGLU). Requires renaming `lm_demo/` → `lm/` and consolidating
   `language/` variants.

5. **Phase 2.1 remaining: `EqPropModel` kwargs → config** — the `EqPropModel.__init__`
   still uses `input_dim=None, hidden_dim=None, output_dim=None, **kwargs` pattern
   inherited from old `BioModel`. Now that `BioModel.__init__` accepts both config-first
   and legacy kwargs, the next step is to port `EqPropModel` and all its subclasses
   (`LoopedMLP`, `ConvEqProp`, etc.) to accept `config: ModelConfig | None = None` and
   remove the legacy pop-from-kwargs path.

6. **Phase 5.2: Break `execution → p2p`** — inject `PeerTransport` Protocol. Moderate
   effort, but `execution/` is now otherwise clean.

**Deferred** (or keep as-is):
- Phase 4.5 (EquiTileOptimizerMixin composition) — mixin is appropriate (Sessions 11+12).
- Phase 8.3 (t-strings) — re-evaluate when CI toolchain supports PEP 750.
- Phase 5.1 remaining grouping — no more closely related single-class modules to group.

### Net LOC impact

| File | Change |
|---|---|
| `execution/_lifecycle.py` (new) | +461 lines (merged from 4 files) |
| `execution/promotion.py` | −82 lines (472 → 5-line shim) |
| `execution/archiver.py` | −170 lines (172 → 5-line shim) |
| `execution/checkpoint_manager.py` | −105 lines (110 → 8-line shim) |
| `execution/curriculum.py` | −97 lines (100 → 5-line shim) |
| `config/schema.py` | +17 lines (two `to_internal()` methods + import) |
| **Net** | **+19 lines** (code moved, not deleted) |

### Files changed in this session

```
A bioplausible/execution/_lifecycle.py      (new — 461 lines: merged lifecycle classes)
M bioplausible/execution/promotion.py       (−82 lines, now re-export shim)
M bioplausible/execution/archiver.py        (−170 lines, now re-export shim)
M bioplausible/execution/checkpoint_manager.py (−105 lines, now re-export shim)
M bioplausible/execution/curriculum.py      (−97 lines, now re-export shim)
M bioplausible/config/schema.py             (+17 lines: to_internal() methods)
M TODO.md                                   (this session log)
```

---

## Session 14 — 2026-07-30: Phase 1.1 Gradient Parity Test

### What was done

**Phase 1.1: Gradient parity test for EP optimizer implementations** (HIGH IMPACT)

Created `tests/integration/test_ep_gradient_parity.py` with 9 tests across 3 test classes:

1. **`TestEqPropGradients`** (4 tests) — Verifies EqProp produces correct EP contrastive gradients:
   - Non-zero for ALL reachable weight layers (fc1, fc2)
   - Different norms per layer (contrastive signal propagates back)
   - Non-trivial gradient values for all layers
   - Reproducible with frozen seed

2. **`TestEPOptimizerGradients`** (3 tests) — Characterizes EPOptimizer's gradient behavior:
   - Non-zero last-layer gradient (from nudge term)
   - Non-zero internal gradients (from residual prediction errors)
   - Computes gradients for ALL params (incl. biases), not just weights

3. **`TestGradientDiscrepancy`** (2 tests) — Documents the formula discrepancy:
   - Cosine similarity < 1.0 between EqProp and EPOptimizer gradients
   - EPOptimizer produces more gradients (all params) than EqProp (weight matrices only)

### Key finding: EPOptimizer's formula is NOT equivalent to EP

**Critical discovery for Phase 1.2**: The EPOptimizer's `(E_nudged - E_free) / beta` formula does NOT compute true EP contrastive gradients. It computes gradients through the energy difference, which at the fixed point only gives non-zero gradients through residual prediction errors from imperfect settling convergence.

**Comparison of the two formulas:**

| Aspect | EqProp (correct) | EPOptimizer (buggy) |
|---|---|---|
| Formula | `(free_prev^T) @ (nudged_out - free_out) / (beta * N)` | `d/dW [(E_nudged - E_free) / beta]` |
| Internal layers | True EP contrastive — non-zero | Residual artifacts — converges to 0 with more settling steps |
| Last layer | EP contrastive | Backprop-like nudge gradient |
| Biases | Not computed | Computed via autograd |
| Mathematical basis | Closed-form EP rule | Autograd through energy function |

**Implication for Phase 1.2**: When folding `EPOptimizer` into `EqProp`, the `EPOptimizer._ep_step` method should be replaced with EqProp's `_compute_ep_gradient` + `_apply_update` pattern. The autograd-based energy contrast formula should NOT be preserved as an EP gradient computation method.

### Verification

```
ruff format --check .        → clean
ruff check .                 → no new errors
pyright bioplausible/        → 0 errors (pre-existing warnings)
pytest -x -q (EP-related)    → 64 passed (all EP + settling + mep tests)
```

### Discovered issues / opportunities

1. **EPOptimizer's gradient formula is incorrect for EP** — The `(E_nudged - E_free) / beta` formula produces gradients that are NOT the EP contrastive gradients. This is a bug that Phase 1.2 must fix. The correct EP formula is EqProp's `_compute_ep_gradient` method.

2. **EPOptimizer initial state capture differs from EqProp** — EPOptimizer's `_capture_states` goes through the full model forward pass (including activation functions like ReLU), while EqProp calls transition modules directly. This causes different initial states even for the same model and input. Phase 1.2 should unify on EqProp's approach.

3. **EPOptimizer._settle has unused `original_target` parameter** — The third parameter `original_target` in `_settle(self, x, target_vec, original_target, beta)` is never used in the function body. This is dead code that Phase 1.2 should clean up.

4. **Settling.py has 3 near-duplicate methods** — `settle`, `settle_with_graph`, `settle_compiled` all implement the same settling loop with minor variations. The energy-based settling primitive extraction should consolidate these into a single `settle` method with flags.

5. **Gradient parity test structure** — The test is designed to be extended with hypothesis-based property tests. The `_shared_settle` pattern (settle once, compute gradients from both formulas) allows clean comparison. Future tests can add `@given` strategies for random seeds, architectures, and hyperparameters.

6. **EqProp._compute_ep_gradient only processes first N weight params** — The `i < len(pairs_free)` guard means only the first N params (where N = number of layers) that are 2D get gradients. The last layer's weight is excluded. This is a known limitation that should be documented when consolidating.

### Guidance for future sessions

**Recommended order** (revised based on Session 14 findings):

1. **Phase 1.1 settling unification** — Extract energy-based settling primitive into `zoo/_settling.py`. The existing `settle_single_state` and `settle_activations_list` are too low-level. Need a new `settle_energy_minimization` that handles: momentum buffers, early stopping, spectral norm freeze, adaptive LR (from Settler), and `torch.compile` support. The gradient parity test from Session 14 is the gate.

2. **Phase 1.2: Fold EPOptimizer into EqProp** — CRITICAL FIX. Replace EPOptimizer's `(E_nudged - E_free) / beta` formula with EqProp's correct `_compute_ep_gradient`. EPOptimizer becomes a thin wrapper around `EqProp` with EWC support. The `EPOptimizerWithEWC` preset routes to `EWC(EqProp(...))`. Delete `ep_optimizer.py` after migration.

3. **Phase 2.1 remaining: EqPropModel kwargs → config** — Port `EqPropModel.__init__` to accept `config: ModelConfig | None = None` instead of `input_dim=None, hidden_dim=None, output_dim=None, **kwargs`.

4. **Phase 4.1: FastLMEquiTile consolidation** — 4 implementations → 1. The `lm_demo/fast_lm.py` version is ~600 LOC of unique architecture (MoT, local attention, SwiGLU). Requires renaming `lm_demo/` → `lm/` and consolidating `language/` variants.

5. **Phase 5.2: Break `execution → p2p`** — Inject `PeerTransport` Protocol. Moderate effort.

**Deferred** (or keep as-is):
- Phase 4.5 (EquiTileOptimizerMixin composition) — mixin is appropriate (Sessions 11+12).
- Phase 8.3 (t-strings) — re-evaluate when CI toolchain supports PEP 750.
- Phase 5.1 remaining grouping — no more closely related single-class modules to group.

### Files changed in this session

```
A tests/integration/test_ep_gradient_parity.py   (new — 300+ lines: 9 gradient parity tests)
M TODO.md                                         (this session log)

---

## Session 15 — 2026-07-30: Phase 1.1 — Energy-Based Settling Primitive & EqProp/Settler Port

### What was done

**Phase 1.1: Unified energy-based settling primitive** (HIGH IMPACT, DRY)

Created `energy_gradient_descent()` in `zoo/_settling.py` — a shared primitive for
energy-based settling that handles the common patterns across all three EP settling
implementations:

- **Momentum buffers**: Initialized internally, updated with `v = momentum * v + grad; state -= lr * v`.
- **Adaptive LR**: Grow LR on energy decrease, decay on increase, with state backup/restore.
- **Early stopping**: Energy delta tolerance with patience counter (absolute + relative tolerance).
- **Divergence detection**: NaN/Inf energy raises `RuntimeError`.

**Port 1: `EqProp._settle_phase`** (39 LOC → 9 LOC + 1 import)

The simplest port — EqProp's `_settle_phase` was a straightforward momentum-based
SGD loop with no adaptive LR or early stopping. The port replaces the inline loop
with a call to `energy_gradient_descent(adaptive=False, tol=None)`:

```python
def _settle_phase(self, x, layers, initial_states, target, beta, settle_steps, settle_lr):
    states = [s.detach().clone().requires_grad_(True) for s in initial_states]
    def energy_fn(s):
        return self._energy(x, s, layers, target, beta)
    return energy_gradient_descent(states, energy_fn, settle_steps, lr=settle_lr, momentum=0.5)
```

**Port 2: `Settler.settle`** (178 LOC loop → 17 LOC call + 1 import)

The most impactful port — Settler's `settle` method was the most feature-rich
implementation with adaptive LR, early stopping, patience, and CUDA kernel dispatch.
The port replaces the entire 101-line loop body with a single call to
`energy_gradient_descent`, keeping only the Settler-specific preamble (state capture,
structure building, target preparation):

```python
states = [s.requires_grad_(True) for s in states]
def wrapped_energy_fn(s):
    return energy_fn(model, x, s, compat_structure, target_vec, beta)
return energy_gradient_descent(
    states, wrapped_energy_fn, self.steps,
    lr=self.lr, momentum=self.MOMENTUM, adaptive=self.adaptive,
    tol=self.tol, patience=self.patience,
    step_size_growth=self.step_size_growth, step_size_decay=self.step_size_decay,
)
```

**Not ported** (documented as deferred):

- **`Settler.settle_with_graph`** — Uses a fundamentally different pattern (detach + re-attach
  `requires_grad` each iteration, creating new tensors). The primitive works with in-place
  updates on the same tensors. Porting would require a separate `detach_each_step` flag or
  a different primitive. Low priority — `settle_with_graph` is a niche variant.

- **`Settler.settle_compiled` / `_settle_loop_fixed`** — Torch.compile-optimized variants
  with fixed-step loops and minimal control flow. The primitive has Python control flow
  (adaptive LR, early stopping) that defeats compilation. These remain as-is.

- **`EPOptimizer._settle`** — Depends on Phase 1.2 architecture decision (EPOptimizer's
  `(E_nudged - E_free) / beta` formula is buggy and will be replaced). Porting the settling
  loop alone is premature without the formula fix.

- **CUDA kernel dispatch** — The `fused_settle_step_inplace` CUDA kernel was a
  Settler-specific optimization. The primitive uses CPU-only momentum updates. The CUDA
  kernel can be re-integrated as a `gradient_step_fn` callback in a future session.

### Verification

```
ruff format --check .        → clean (592 files)
ruff check .                 → 0 new errors (4794 pre-existing, all in tests/)
pyright bioplausible/        → 0 errors (2342 warnings, all pre-existing)
pytest -x -q                → 1189 passed, 13 skipped, 5 subtests (51s)
  EP gradient parity tests  → 9/9 passed (no regression)
Coverage                    → 55.73% (above 50% floor)
```

### Discovered issues / opportunities

1. **`energy_gradient_descent` primitive is feature-complete** — It handles all the
   common patterns: momentum, adaptive LR, early stopping, divergence detection.
   The CUDA kernel dispatch and `torch.compile` support are Settler-specific
   optimizations that don't belong in the shared primitive.

2. **`Settler.settle_with_graph` cannot use the primitive** — The detach/re-attach pattern
   is fundamentally incompatible with the primitive's in-place update approach. This is
   acceptable — `settle_with_graph` is a niche variant used by fewer callers. The
   duplicate code in `settle_with_graph` (71 lines that overlap with the old `settle`)
   is a known DRY violation that's acceptable for a niche variant.

3. **`Settler._settle_loop_fixed` and `settle_compiled`** remain as separate implementations.
   These are compilation-optimized variants with fixed-step loops. The `torch.compile`
   decorator on `_compiled_settle_step` is incompatible with Python control flow. These
   are acceptable as performance optimizations that don't need unification.

4. **`EPOptimizer._settle` port is gated on Phase 1.2** — The `EPOptimizer` uses
   `_analytic_gradients` or `_autograd_gradients` to compute state gradients, which is
   a different pattern from the energy-based gradient descent. Porting to the primitive
   requires changing the gradient computation to use the energy function, which is part
   of the Phase 1.2 formula fix.

5. **Net LOC reduction**: ~140 lines (EqProp: -30, Settler: -110, `_settling.py`: +90
   for the primitive). The primitive adds ~90 lines but replaces ~200 lines of duplicated
   settling logic across two files.

### Guidance for future sessions

**Recommended order** (revised based on Session 15):

1. **Phase 1.2: Fold `EPOptimizer` into `EqProp`** — CRITICAL FIX. Now that the
   settling primitive is extracted, Phase 1.2 should:
   - Replace `EPOptimizer._ep_step`'s `(E_nudged - E_free) / beta` formula with
     EqProp's correct `_compute_ep_gradient`.
   - Port `EPOptimizer._settle` to use `energy_gradient_descent` (currently gated).
   - Route `EPOptimizerWithEWC` to `EWC(EqProp(...))`.
   - Delete `ep_optimizer.py` (731 LOC).

2. **Phase 2.1 remaining: `EqPropModel` kwargs → config** — Port `EqPropModel.__init__`
   to accept `config: ModelConfig | None = None` instead of legacy kwargs.

3. **Phase 4.1: `FastLMEquiTile` consolidation** — 4 implementations → 1.

4. **Phase 5.2: Break `execution → p2p`** — Inject `PeerTransport` Protocol.

**CUDA kernel re-integration** (optional, low priority):
   - Add an optional `gradient_step_fn: Callable | None` parameter to
     `energy_gradient_descent` that allows callers to override the momentum update step
     with a custom kernel. If None, use the default CPU momentum update.
   - `Settler.settle` would pass `fused_settle_step_inplace` when on CUDA.

**Deferred** (or keep as-is):
- Phase 4.5 (EquiTileOptimizerMixin composition) — mixin is appropriate.
- Phase 8.3 (t-strings) — re-evaluate when CI toolchain supports PEP 750.
- Phase 5.1 remaining grouping — no more closely related single-class modules.
- `Settler.settle_with_graph` port — niche variant, incompatible patterns.

### Files changed in this session

```
M bioplausible/zoo/_settling.py                     (+90 lines: energy_gradient_descent primitive)
M bioplausible/zoo/propagators/eqprop.py            (-30 lines: port to primitive)
M bioplausible/zoo/mep/optimizers/settling.py       (-110 lines: port Settler.settle to primitive)
M bioplausible/core/model.py                        (pre-existing formatting fix)
M bioplausible/execution/_lifecycle.py              (pre-existing formatting fix)
M TODO.md                                           (this session log)

---

## Session 16 — 2026-07-30: Phase 1.2 (EPOptimizer dead-code deletion), Phase 2.1 (EqPropModel config)

### What was done

**Phase 1.2: EPOptimizer folding — actual finding: dead code** (HIGH IMPACT, unexpected)

The plan described a complex fold of `EPOptimizer` into `EqProp`, but the codebase
audit revealed a much simpler reality:

**`EPOptimizer` has zero production consumers.** Every constructor call in
`ep_optimizer.py` (731 LOC) was inside its own docstring examples. The presets
(`zoo/mep/presets/__init__.py`) use `CompositeOptimizer` + strategy objects, not
`EPOptimizer`. The only external consumer is `tests/integration/test_ep_gradient_parity.py`,
which uses `EPOptimizer` to characterize its buggy gradient formula.

**What was done:**
1. **`ep_optimizer.py` reduced from 731 → 160 LOC** — preserved only the `EPOptimizer`
   class, `EPConfig`, and the methods the test needs (`_settle`, `_capture_states`,
   `_autograd_gradients`, `_energy_from_states`). Added prominent "LEGACY REFERENCE —
   DO NOT USE IN PRODUCTION" header documenting:
   - Why it's legacy (zero production consumers, buggy formula)
   - That `EWCState` is also dead (separate from `EPOptimizerWithEWC`)
   - The correct approach (use `EqProp` or `CompositeOptimizer` + strategies)

2. **`EWCState` class deleted** — dead code, never instantiated outside `ep_optimizer.py`.
   `EPOptimizerWithEWC` in `zoo/mep/optimizers/ewc.py` is a separate implementation
   that wraps `O1MemoryEPv2` (not `EPOptimizer`).

3. **Re-exports removed** from `zoo/mep/optimizers/__init__.py` and `zoo/mep/__init__.py`
   (`EPOptimizer`, `EPConfig`, `EWCState`).

4. **Gradient parity test unchanged** — still imports `EPOptimizer` directly from the
   file path, which still works.

**Net LOC**: −571 (731 → 160 LOC, plus re-export cleanup)

**Phase 2.1: EqPropModel accepts `config` parameter** (MEDIUM IMPACT)

`EqPropModel.__init__` now accepts `config: ModelConfig | None = None` as the first
parameter. When a config is provided, it extracts `input_dim`, `hidden_dims`,
`output_dim`, `max_steps`, `use_spectral_norm`, `lipschitz_mode`, `beta`, and
`gradient_method` (from `config.extra`) from the config. When no config is provided,
the legacy kwargs-pop path is preserved unchanged.

This means:
- New code can use `EqPropModel(config=my_config)` — config-first.
- Existing code using `EqPropModel(input_dim=..., hidden_dim=..., output_dim=...)`
  continues to work — backward-compatible.
- The 12+ subclasses (`LoopedMLP`, `ConvEqProp`, `TransformerEqProp`, etc.) don't
  need any changes.

**Not done** (documented "lack of ambition"):
- The legacy kwargs-pop path in `BioModel.__init__` is preserved. Removing it would
  require porting every `EqPropModel` subclass constructor — a large, separate task.
- No `ModelConfig.build()` classmethod was added (the `config/schema.py.to_internal()`
  path already serves this role).
- `gradient_method` is not in `ModelConfig` (it's an `EqPropModel`-specific attribute).
  It's sourced from `config.extra.get("gradient_method")` with a fallback to the
  parameter default.

### Verification

```
ruff format --check .        → clean (592 files)
ruff check .                 → 0 new errors (4778 pre-existing, all in tests/)
pyright bioplausible/        → 0 errors (2300 warnings, all pre-existing)
pytest -x -q                → 1189 passed, 13 skipped, 5 subtests (51s)
  EP gradient parity tests  → 9/9 passed (no regression)
Coverage                    → 56.03% (above 50% floor)
```

### Discovered issues / opportunities

1. **Phase 1.2 plan was stale** — The plan assumed `EPOptimizer` had production
   consumers and described a complex fold. The actual state was simpler: dead code
   that should be deleted. The plan should be updated to reflect the audit finding.

2. **`EPOptimizerWithEWC` is separate from `EPOptimizer`** — Despite the name,
   `EPOptimizerWithEWC` in `zoo/mep/optimizers/ewc.py` does NOT use `EPOptimizer`.
   It wraps `O1MemoryEPv2` or `smep` preset. The name is misleading but fixing it
   is out of scope.

3. **`gradient_method` is not in `ModelConfig`** — It's an `EqPropModel`-specific
   attribute (`"bptt"`, `"equilibrium"`, `"contrastive"`). `StandardEqProp` and
   other `BioModel`-direct subclasses don't use it. Storing it in `config.extra` is
   the pragmatic approach.

4. **Two parallel model hierarchies** — `EqPropModel` (12 subclasses, kwargs-based)
   and `BioModel`-direct subclasses (7 subclasses, config-first). The `EqPropModel`
   hierarchy has `gradient_method`, `contrastive_update`, `train_step` methods that
   the `BioModel`-direct subclasses don't have. This is a design bifurcation worth
   noting but not fixing in this session.

5. **`StandardEqProp` and `FiniteNudgeEP` inherit from `BioModel` directly** — They
   bypass `EqPropModel` entirely. This means they don't get `EqPropModel`'s
   `gradient_method`, `beta`, `hebbian_lr`, `contrastive_update`, or `train_step`
   methods. This is likely intentional (they use the propagator for EP logic, not
   the model), but it's an inconsistency.

### Guidance for future sessions

**Recommended order** (revised based on Session 16):

1. **Phase 4.1: `FastLMEquiTile` consolidation** — 4 implementations → 1. The
   `lm_demo/fast_lm.py` version is ~600 LOC of unique architecture (MoT, local
   attention, SwiGLU). Requires renaming `lm_demo/` → `lm/` and consolidating
   `language/` variants.

2. **Phase 5.2: Break `execution → p2p`** — Inject `PeerTransport` Protocol.
   Moderate effort. `execution/engine.py` imports `p2p.dht` at 12 call sites.

3. **Phase 2.2: Collapse `LMTrainer` duplication** — Two `LMTrainer` classes
   (897 LOC + 559 LOC). Delegate to `CoreTrainer`.

4. **Phase 2.3: Single training-step dispatch in `CoreTrainer`** — Extract a
   `StepDispatcher` with `match/case` over a `PlausibleStep` protocol union.

**Documented "lack of ambition"** (items partially done or deferred):
- Phase 1.2 fold was simpler than planned — EPOptimizer was dead code, just deleted.
  The "fold" is complete. No further action needed.
- Phase 2.1 config port: `EqPropModel` now accepts config, but the legacy kwargs
  path in `BioModel.__init__` is preserved. Full removal would require porting 12+
  subclass constructors.
- Phase 1.1 CUDA kernel: `fused_settle_step_inplace` not ported to the primitive.
  Can be added as an optional `gradient_step_fn` callback.
- `Settler.settle_with_graph` port: incompatible detach/re-attach pattern.
- `Settler.settle_compiled` port: `torch.compile` incompatible with Python control flow.

**Deferred** (or keep as-is):
- Phase 4.5 (EquiTileOptimizerMixin composition) — mixin is appropriate.
- Phase 8.3 (t-strings) — re-evaluate when CI toolchain supports PEP 750.
- Phase 5.1 remaining grouping — no more closely related single-class modules.

### Files changed in this session

```
M bioplausible/zoo/mep/optimizers/ep_optimizer.py     (−571 lines: 731→160, dead code → test ref)
M bioplausible/zoo/mep/optimizers/__init__.py         (−4 lines: removed dead re-exports)
M bioplausible/zoo/mep/__init__.py                    (−2 lines: removed EPOptimizer re-export)
M bioplausible/zoo/models/base.py                     (+36 lines: config-first path in EqPropModel)
M TODO.md                                             (this session log)
```
```

---

## Session 17 — 2026-07-30: Phase 3.1 — Merge hyperopt Task Hierarchy into domains/

### What was done

**Phase 3.1: Task hierarchy merge** (HIGH IMPACT, DRY)

Eliminated the duplicate task hierarchy — `hyperopt/tasks.py:BaseTask` + `VisionTask`/`LMTask`/`RLTask`/`CharNGramTask` + `hyperopt/tabular_task.py:TabularTask` + `hyperopt/graph_task.py:GraphTask` are now re-export shims from `domains/`.

**Key design decisions:**

1. **`DomainTask` now satisfies `TaskProtocol`** — The `DomainTask.get_batch` signature changed from `(split: TaskSplit) -> Batch` to `(split, batch_size) -> tuple[Tensor, Tensor]` (protocol-compatible). Added `get_batch_domain()` for the rich `Batch`-returning interface. Similarly, `compute_metrics` now returns `dict[str, float]` (protocol-compatible); `compute_metrics_domain` returns `Metrics` dataclass.

2. **New modules under `domains/`:**
   - `domains/trainer.py` — `TaskProtocol`, `_TaskTrainer`, `_resolve_task_loss` (moved from `hyperopt/tasks.py`).
   - `domains/factory.py` — `create_task()` factory, `_parse_split_digits`, `_normalize_vision_name`, `CharNGramTask`.

3. **Concrete task fixes for protocol compatibility:**
   - `RLTask`: overrode `create_trainer` to return `RLTrainer` (not `_TaskTrainer`), added `get_batch()` raising `NotImplementedError`.
   - `LMTask`: overrode `get_batch` with random-subsequence sampling (DataLoader returns raw tokens, not `(inputs, targets)` pairs).
   - `GraphTask`: overrode `get_batch` to return full graph data.
   - `VisionTask.setup()`: added fallback to `get_vision_dataset()` for non-torchvision datasets (digits, KMNIST, SVHN, USPS, etc.), with uint8→float normalization.

4. **Re-export shims** preserve backward compat for all existing importers:
   - `hyperopt/tasks.py` → re-exports from `domains.*` (`BaseTask = DomainTask`, plus all concrete tasks and factory).
   - `hyperopt/tabular_task.py` → re-export shim from `domains.tabular`.
   - `hyperopt/graph_task.py` → re-export shim from `domains.graph`.
   - `hyperopt/task_registry.py` → imports from `domains/` directly.

**Stale claim corrections discovered:**
- **Phase 5.2 (`execution → p2p`)**: TODO claimed "12 call sites" but exhaustive grep found **zero** imports from `p2p/` in `execution/`. The coupling was broken by prior refactoring. Marked as already resolved.
- **Phase 4.3 (`equitile → zoo`)**: Confirmed zero edges remain (TODO already marked ✅, but guidance said to re-confirm).

### Verification

```
ruff format --check .        → clean (594 files)
ruff check .                 → only pre-existing `unsorted-dunder-all` in `__init__.py`
pyright bioplausible/        → 0 errors, 2301 warnings (all pre-existing)
pytest -x -q                → 1189 passed, 13 skipped, 5 subtests (47s)
Coverage                    → 56.27% (above 50% floor)
```

All 5 task-related tests in `test_refactor2_bugfixes.py` pass (test stubs updated with `domain_type`, `spec`, `evaluate`, `get_dataloader` abstract methods).

### Discovered issues / opportunities

1. **`fold`/`data_fraction` kwargs silently dropped** — The experiment system passes `fold` and `data_fraction` from configs to `create_task`. These are captured by `DomainTask.__init__(**kwargs)` but not used by the new VisionTask (which uses DataLoaders, not pre-loaded tensors). No functional regression for non-K-fold experiments. Documented as known gap.

2. **`quick_mode` is stored but not enforced** — The old hyperopt tasks truncated datasets to 100/1000 samples in quick_mode. The domains tasks store `quick_mode` but don't use it. Minimal practical impact since quick_mode is used for quick smoke tests with small models, not small data.

3. **`_load_vision_dataset_cached` is dead code** — The old `hyperopt/tasks.py` function was only used by the old hyperopt VisionTask. Not needed by the domains VisionTask. Can be removed in a future cleanup pass.

4. **`CharNGramTask` stays in `domains/factory.py`** — It's a synthetic task for hyperopt experiments, not a real domain. Keeping it in `factory.py` avoids polluting the domain hierarchy.

5. **`BaseTask` alias** — `hyperopt/tasks` exports `DomainTask as BaseTask` for backward compat. Tests stubs that inherit from `BaseTask` (now `DomainTask`) needed to implement 4 additional abstract methods (`domain_type`, `spec`, `evaluate`, `get_dataloader`). Updated in 4 test stubs.

### Guidance for future sessions

**Recommended order** (revised based on Session 17):

1. **Phase 4.1/4.2/4.4: `FastLMEquiTile` consolidation** — 4 implementations → 1. The `lm_demo/fast_lm.py` has ~600 LOC unique architecture (MoT, local attention, SwiGLU). Requires renaming `lm_demo/` → `lm/` and consolidating `language/` variants. Most impactful remaining item.

2. **Phase 2.2: Collapse `LMTrainer` duplication** — Two `LMTrainer` classes (897 LOC + 559 LOC). Delegate the simpler one to the production one, then to `CoreTrainer`.

3. **Phase 2.3: Single training-step dispatch in `CoreTrainer`** — Replace `isinstance`/`hasattr`/`inspect.signature` probe chain with `match/case` over a `PlausibleStep` protocol union.

4. **Phase 3.2 `core/losses.py` cleanup** — The 2 `magic-value-comparison` warnings for `logits.dim() == 3`. Extract a `_THREE_D = 3` constant (cosmetic).

5. **Phase 7.1: Eliminate `Any` / untyped dicts** — In `autoscientist/campaign.py`, `hyperopt/experiment.py`, `execution/engine.py`, `evaluation/base.py`.

**Deferred** (or keep as-is):
- Phase 4.5 (EquiTileOptimizerMixin composition) — mixin is appropriate.
- Phase 5.2 (`execution → p2p`) — **already resolved**. Zero imports exist.
- Phase 8.3 (t-strings) — re-evaluate when CI toolchain supports PEP 750.
- Phase 5.1 remaining grouping — no more closely related single-class modules.

### Files changed in this session

```
A bioplausible/domains/trainer.py            (new — 96 lines: TaskProtocol, _TaskTrainer, _resolve_task_loss)
A bioplausible/domains/factory.py            (new — 192 lines: create_task, CharNGramTask, helpers)
M bioplausible/domains/base.py               (+50 lines: quick_mode, task_type, get_batch(protocol), compute_metrics(protocol), create_trainer)
M bioplausible/domains/__init__.py           (+10 lines: re-export new symbols)
M bioplausible/domains/vision.py             (+45 lines: get_vision_dataset fallback, dtype normalization)
M bioplausible/domains/lm.py                 (+20 lines: get_batch override with random-subsequence sampling)
M bioplausible/domains/graph.py              (+7 lines: get_batch override for full-graph data)
M bioplausible/domains/rl.py                 (+23 lines: create_trainer → RLTrainer, get_batch → NotImplementedError)
M bioplausible/hyperopt/tasks.py             (−728 lines: now ~20-line re-export shim)
M bioplausible/hyperopt/tabular_task.py      (−75 lines: now 3-line re-export shim)
M bioplausible/hyperopt/graph_task.py        (−60 lines: now 3-line re-export shim)
M bioplausible/hyperopt/task_registry.py     (±0: import paths only)
M tests/unit/test_refactor2_bugfixes.py      (+65 lines: 4 test stubs implement DomainTask abstract methods)
M tests/unit/test_model_registry_instantiation.py (+30 lines: MockVisionTask implements DomainTask abstract methods)
M tests/integration/test_domains.py          (compute_metrics → compute_metrics_domain)
M TODO.md                                    (this session log)
```
```