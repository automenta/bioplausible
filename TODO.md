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

---

## Architectural Understanding

The codebase has a **two-tier propagator/model split** (documented, intentional):

- `zoo/propagators/*` — `torch.optim.Optimizer` subclasses, mutate params of any model.
- `zoo/models/*` — own `forward`/`train_step`, for rules needing model-side control.

There is **also** a **third, accidental split**: `zoo/mep/optimizers/*` reimplements
settling, energy, and contrastive logic that exists in both `zoo/propagators/eqprop.py`
and `core/energy_model.py`. This — plus `equitile/*` duplicating trainers and configs —
is the central DRY problem this plan targets.

**Layering violations observed** (from architecture graph):

| Edge | Call count | Issue |
|---|---|---|
| `archive → zoo` | 47 | Archive code imports live code (should be frozen) |
| `equitile → zoo` | 37 | EquiTile reaches into zoo internals |
| `equitile → archive` | 7 | EquiTile depends on archived code |
| `execution → p2p` | 12 | Execution coupled to p2p transport |

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

### 1.1 Unify on `zoo/_settling.py`

- Make `zoo/_settling.py` the **single settling primitive**.
- Port `EqProp`, `Settler`, `EPOptimizer` to call `settle_activations_list`.
  - `EqProp._settle_phase` → delegates to `settle_activations_list`.
  - `Settler.settle*` family collapses to one `Settler.settle` that calls the shared
    loop; `settle_with_graph`/`settle_compiled`/`_settle_loop_fixed` deleted (graph mode
    becomes a `return_graph=True` flag; compiled mode becomes a `compile=True` flag).
  - `EPOptimizer._settle` → delegates to `Settler` (which delegates to `_settling`).
- Add `_settling.set_energy_fn` that builds the standard MSE-consistency + nudge energy
  from a `loss_type` and `transition_modules` list; both `EqProp._energy` and
  `EPOptimizer._energy_from_states` call it.

**Verification**: `pytest tests/test_eqprop.py tests/test_mep_integration.py
tests/test_propagator_eqprop.py` — all green; gradient parity test in §7.2 below.

**LOC reduction**: ~400 lines deleted, ~120 added → net −280.

### 1.2 Fold `EPOptimizer` into `propagators/eqprop.py`

`EPOptimizer` is advertised as "unified" but is a parallel implementation with its own
config, buffers, structure-building, and EWC integration. `EqProp` + `AdamEqProp` already
cover the EP surface. Action:

- Move EWC support (`EWCState`, `consolidate_task`) into `zoo/optimizers/ewc.py:EWC`
  (which already wraps `EPOptimizerWithEWC`) — keep EWC **as a wrapper**, not duplicated
  inside the EP optimizer.
- Move `MuonUpdate`, `SpectralConstraint`, `ErrorFeedback` composition into the
  `update_strategy` slot of `EqProp` (already exists: `EqProp.update_strategy`).
- Delete `ep_optimizer.py`; route Registry entries to `EqProp` with preset kwargs for
  `smep`, `smep_fast`, `O1MemoryEP`. Presets live in `zoo/mep/presets/` (already there).
- `EPOptimizerWithEWC` becomes `EWC(EqProp(...))` — the `EWC` wrapper already accepts
  any propagator.

**Verification**: every `test_mep_integration.py` test passes against the new wiring;
presets in `zoo/mep/presets/__init__.py` construct.

---

## Phase 2 — Eliminate Duplicate Config & Trainer Hierarchies (HIGH IMPACT)

### 2.1 One `ModelConfig`, one `RunConfig`

Three `ModelConfig`/`ModelConfig`-shaped classes exist:

- `zoo/base.py:ModelConfig` (frozen dataclass, slots) — the canonical one.
- `zoo/models/base.py:EqPropModel` owns a parallel `ModelConfig`-shaped dict.
- `config/schema.py:ModelConfig` / `RunConfigModel` / `RunConfigOptimizer` —
  Pydantic models for YAML I/O. These are the **I/O boundary** and stay.

**Action**:

- `config/schema.py` keeps the Pydantic `RunConfig*` models (I/O boundary — correct per
  `AGENTS.md`).
- `zoo/base.py:ModelConfig` stays the **internal frozen dataclass**.
- `zoo/models/base.py:EqPropModel` constructors accept the `config/schema.py` Pydantic
  `ModelConfig` and convert to the frozen `zoo/base.py:ModelConfig` via a
  `to_internal()` method on the Pydantic model. One conversion site.
- Delete the ad-hoc `kwargs` plumbing in `BioModel.__init__` (the `input_dim=None,
  hidden_dim=None, output_dim=None, **kwargs` legacy path). Force the config-first path;
  direct-dim construction goes through a `ModelConfig.build()` classmethod.

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

## Phase 3 — Tighten the Domain/Task Layer (MEDIUM IMPACT)

### 3.1 Merge `hyperopt/tasks.py` Task hierarchy into `domains/`

Two parallel task hierarchies do the same thing:

| `hyperopt/tasks.py` | `domains/*.py` |
|---|---|
| `BaseTask(ABC)` | `DomainTask(ABC)` |
| `VisionTask` | `VisionTask` |
| `LMTask` | `LMTask` |
| `RLTask` | `RLTask` |
| `TabularTask` (in `hyperopt/tabular_task.py`) | `TabularTask` |
| `GraphTask` (in `hyperopt/graph_task.py`) | `GraphTask` |

The `domains/*` version uses `Batch`/`TaskSplit`/`DomainTask` (cleaner, Protocol-adjacent).
The `hyperopt/tasks.py` version is older with `BaseTask.__init__` carrying dataset caches.

**Action**: `domains/base.py:DomainTask` becomes the single base; `TaskProtocol`
(completed Session 5) is the structural interface. Migrate `hyperopt/*_task.py` concrete
classes to subclass `DomainTask`. Delete `hyperopt/tasks.py:BaseTask` and the duplicate
`VisionTask`/`LMTask`/`RLTask` definitions there. Hyperopt factories keep their
`create_task` match (already refactored to `match/case` in Session 2) but instantiate
from `domains/*`.

**Net**: one `VisionTask`, one `LMTask`, one `RLTask`, etc. — not two.

### 3.2 `figutils` worth extracting

`_compute_loss` / `_compute_accuracy` are defined in `core/trainer.py:71`,
duplicated in `graph/training.py:25`, and re-implemented in
`equitile/core/model.py:759`. Extract once:

```python
# bioplausible/core/losses.py
def compute_loss(loss_fn, logits, y) -> Tensor: ...
def compute_accuracy(logits, y) -> float: ...
def reshape_for_cross_entropy(logits, y) -> tuple[Tensor, Tensor]: ...
```

All three call sites import from `core/losses.py`. ~80 lines removed.

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

### 5.1 Group related single-class modules

- `execution/_failure.py` ← `failure_tracker.py` + `analysis/failure_manifesto.py`
  (both track failures).
- `execution/_safety.py` ← `safety.py` + `robustness.py` + `algorithm_constraints.py`
  (all encode invariants/constraints).
- `execution/_lifecycle.py` ← `promotion.py` + `archiver.py` + `checkpoint_manager.py`
  + `curriculum.py` (all manage the experiment lifecycle).

Each `_`-prefixed per `AGENTS.md` (internal modules). Re-export old names from
`execution/__init__.py` for compatibility during migration.

### 5.2 Break `execution → p2p`

`execution/engine.py` imports `p2p.dht` directly (12 calls). Inject a
`PeerTransport` Protocol; the engine receives it at construction. p2p stays optional.

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

### 6.1 Replace bare `str` task-types with `StrEnum`

`task_type` strings ("vision", "lm", "tabular", "rl", "graph", "timeseries",
"scientific") are compared as bare strings across `hyperopt/`, `core/trainer.py`,
`domains/`. Promote to `domains/base.py:TaskType(StrEnum)`. Match/case over the enum
in all dispatch sites.

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
Risk: behavior change in EP settling. Mitigated by §1 parity tests.

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
- **NEW**: gradient parity test (§1.1 verification) green — EP grads identical before/after consolidation.
- **NEW**: architecture-graph `equitile → zoo` edge count = 0 (after Phase 4.2).
- **NEW**: no two classes share a name-with-purpose (`LMTrainer`, `FastLMEquiTile`,
  `FastLMConfig`, `ModelConfig`, `VisionTask`, `LMTask`, `RLTask`) — single definition each.
- **NEW**: exactly one `FastLMEquiTile` registered in Registry (after Phase 4 LM consolidation).

---

## Risk Mitigation

| Risk | Mitigation |
|---|---|
| EP gradient consolidation changes numerics | Property test: assert `(grad_new - grad_old).abs().max() < 1e-6` on a frozen-seed fixture across `(EqProp, Settler, EPOptimizer)` for 50 random small models (hypothesis strategy). Ship this test before any consolidation PR. |
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

---

## Status Legend

- 🔲 not started
- 🔨 in progress
- ✅ done (append session log below)

_Sessions append progress as `## Session N` sections below, mirroring the
`docs/archive/20260728/REFACTOR.md` format._