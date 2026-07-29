# Refactoring TODO — Bioplausible Codebase

> **Goal**: Improve elegance, clarity, DRY, maintainability, and alignment with `@AGENTS.md`.
> **Scope**: `bioplausible/`, `tests/`, `examples/`, `experiments/`, `configs/`. Excludes `docs/` and archives.
> **Strategy**: Prioritize *architectural* improvements with high blast radius over tedious line-level lint work (which automated tooling handles).

---

## Theoretical North Star

**Millidge et al. (2022)** — *"Backpropagation at the Infinitesimal Inference Limit of Energy-Based Models: Unifying Predictive Coding, Equilibrium Propagation, and Contrastive Hebbian Learning"* ([arXiv:2206.02629](https://arxiv.org/abs/2206.02629)).

**Key insight**: Predictive Coding (PC), Equilibrium Propagation (EP), and Contrastive Hebbian Learning (CHL) are all instances of a *single* framework — energy-based models (EBMs) at the infinitesimal inference limit — where backpropagation emerges as the linearized gradient of the energy at free-phase equilibrium. The differences are specific choices of energy function and nudging protocol, *not* fundamental algorithmic distinctions.

**Implication for this codebase**: The current architecture treats PC, EP, and CHL as three separate families (`zoo/models/predictive_coding.py`, `zoo/models/eqprop/*`, `zoo/propagators/hebbian.py`) with duplicated settling loops, duplicated contrastive-update logic, and duplicated energy computation. **Unifying these under a shared `EnergyBasedModel` abstraction would eliminate the largest source of duplication in the codebase and align the architecture with the underlying theory.**

**μPC / muP output-node scaling** (via [FabricPC `mupc_output_fix`](https://github.com/trueagi-io/FabricPC/compare/main...matthewbehrend/mupc_output_fix)) — The output-layer weighting in predictive-coding / energy-based models must **not** include the √L factor that appears in the hidden-layer scaling. The current `graph/` code (adapted from FabricPC) and the `zoo/base.py` spectral-norm initialization both apply a uniform scaling that is incorrect for output nodes. This must be fixed to match the μPC paper.

---

## Current State (from `TODO0.md`)

| Phase | Focus | Status | Net Δ |
|-------|-------|--------|-------|
| 0 | Archive dead code, syntax, print→logging | ✅ | −7,909 |
| 1 | Unify registries, frozen dataclasses | ✅ | −41 |
| 2 | Core type safety (eliminate `Any`) | ✅ | +40 |
| 3 | Algorithmic dedup | 🟡 partial | +81 |
| 4 | Full type hardening | ⏳ | — |

**Tests**: 1,117 passed, 15 skipped · **Coverage**: 55% (floor=40%) · **Pyright**: 0 errors, 0 new.

### Session 7 Progress (2026-07-29)

| Item | Status | Details |
|------|--------|---------|
| **A.4 μPC output scaling fix** | ✅ | `ModelConfig.output_scaling_mode`, `BioModel.apply_spectral_norm(layer_role=...)`, updated 10+ callers across zoo/models/ |
| **A.1 EnergyModel protocol** | ✅ | `core/energy_model.py` — `EnergyModel` Protocol + `EBMTrainer` |
| **A.3 Energy function library** | ✅ | `core/energies.py` — 6 shared energy functions |
| **F.2 Pyright errors** | ✅ | Fixed `deployment.py:717` (missing `InferenceRequest` def) + `hyperopt/graph_task.py:28-32` (missing `import os`) |
| **CI gate** | ✅ | `ruff format` — clean · `ruff check` — 5447 pre-existing warnings (all `@typing.override` / PLR6301, not new) · `pyright` — **0 errors** (was 5) · `pytest` — 1,117 passed, 15 skipped |

**Key diff**: +3 new files (`core/energy_model.py`, `core/energies.py`, `InferenceRequest` fix), ~10 modified. Zero test regressions.

---

## Phase A — Unified Energy-Based Model Framework (HIGH IMPACT)

*This is the single highest-impact architectural change. It eliminates the deepest duplication and aligns the codebase with the theory.*

### A.1 Create `core/energy_model.py` — Shared EBM Base

**Problem**: Three algorithm families implement the same "settle → compute energy → contrastive update" loop independently:

| Family | File(s) | Lines | Settling | Contrastive Update |
|--------|---------|-------|----------|--------------------|
| EqProp | `zoo/models/eqprop/*.py` (12+ files) | ~2,500 | `settle_activations_list` / `EquilibriumFunction` | `_contrastive_step` (3 models) |
| Predictive Coding | `zoo/models/predictive_coding.py`, `graph/inference.py` | ~400 | `InferenceSGD.settle` | inline in `train_step` |
| CHL / Hebbian | `zoo/propagators/hebbian.py` | ~200 | none (single-step) | `ContrastiveHebbianLearning` |

**Solution**: A single `EnergyBasedModel` protocol/base class in `core/energy_model.py`:

```python
class EnergyModel(Protocol):
    """Unified interface for energy-based learning algorithms.

    All of PC, EP, and CHL satisfy this protocol. The trainer
    selects the nudging protocol and energy function; the model
    provides settle dynamics and energy computation.
    """

    def energy(self, x: Tensor, y: Tensor | None) -> Tensor:
        """Total free energy at current state."""
        ...

    def settle(
        self, x: Tensor, steps: int, beta: float = 0.0, y: Tensor | None = None
    ) -> None:
        """Iterate internal states toward equilibrium (free or nudged)."""
        ...

    def contrastive_update(
        self, free_state: State, nudged_state: State, beta: float, lr: float
    ) -> None:
        """Apply weight update from free/nudged state difference."""
        ...
```

**Benefits**:
- Eliminates ~600–900 lines of duplicated settling/update logic
- Makes the PC↔EP↔CHL equivalence *structurally visible* in the code
- New EBM variants become trivial: implement `energy()`, `settle()`, `contrastive_update()`
- Enables a single `EBMTrainer` that handles all three families (replaces special-cased `train_step` in 23+ models)

**Migration path** (non-breaking):
1. Define `EnergyModel` protocol + `EBMTrainer` in `core/energy_model.py`
2. Have existing models implement the protocol (duck-typing, no inheritance required)
3. Add an opt-in `ebm_train()` path in `CoreTrainer` that uses `EBMTrainer` when model satisfies `EnergyModel`
4. Migrate models one at a time; legacy `train_step` remains as fallback

### A.2 Unify `graph/` Module with `zoo/_settling.py`

**Problem**: Two parallel settling implementations:
- `zoo/_settling.py` — `settle_activations_list`, `settle_single_state`, `EquilibriumFunction` (autograd-compatible)
- `graph/inference.py` — `InferenceSGD.settle` (FabricPC-derived, manual activity updates)

**Solution**: `graph/inference.py` implements the `EnergyModel` protocol (A.1), delegating to `zoo/_settling.py` for the actual settling loop. Eliminates ~80 lines of duplicated settling in `InferenceSGD`.

### A.3 Shared Energy Function Library

**Problem**: Every EBM model defines its own energy inline:
- `zoo/models/eqprop/standard_eqprop.py` — implicit (layer predictions)
- `zoo/mep/optimizers/energy.py` — `EnergyFunction` class (MEP-specific)
- `graph/inference.py` — `||a_child - f_parent(a_parent, θ)||²`
- `zoo/models/predictive_coding.py` — `mse_loss(top_down(upper), lower)`

**Solution**: `core/energies.py` with a small library of energy functions:

```python
def prediction_error_energy(
    activities: list[Tensor],
    predictions: list[Tensor],
    weights: list[Tensor] | None = None,
) -> Tensor: ...

def supervised_energy(
    logits: Tensor, targets: Tensor, loss_fn: Callable[..., Tensor]
) -> Tensor: ...

def hybrid_energy(
    activities: list[Tensor],
    predictions: list[Tensor],
    logits:.Tensor, targets: Tensor,
    supervised_weight: float = 1.0,
) -> Tensor: ...
```

**Estimated savings**: ~200–400 lines across all EBM models.

### A.4 μPC Output-Node Scaling Fix

**Problem**: The FabricPC `mupc_output_fix` branch corrects the output-layer weight scaling for μPC (Maximum Update Parameterization for PC networks):

> The output-node weighting must **NOT** include the √L factor that is applied to hidden nodes. The output layer's update magnitudes are governed by a different scaling than hidden layers in the μPC framework. Applying the uniform √L factor to output nodes causes the output gradients to be off by a factor of √(L_out) relative to the μPC prescription, where L_out is the output layer's fan-in.

**Affected code**:
- `graph/initialization.py` — weight initialization scales all layers uniformly (√L factor)
- `zoo/base.py` — `apply_spectral_norm` applies uniform spectral norm across all layers
- `equitile/topology.py` — `build_layered` initializes all tiles with the same fan-in scaling
- `zoo/models/eqprop/*` — spectral norm applied uniformly to all layers including output

**Solution**:
1. Add a `layer_role: Literal["input", "hidden", "output"]` parameter to initialization/spectral-norm functions
2. Output layers skip the √L scaling factor (or apply a corrected scaling: `scale_output = scale_hidden / math.sqrt(L_output_fan_in)`)
3. Add property to `ModelConfig`: `output_scaling_mode: Literal["uniform", "mupc"] = "mupc"`
4. Default to `"mupc"` to align with the paper; `"uniform"` available for backward compatibility / ablation

**Verification**: Add a hypothesis-style test asserting that, at initialization, the gradient magnitude ratio between hidden and output layers matches the μPC prescription (no √L on output).

**Priority**: **HIGH** — this is a correctness fix, not just refactoring. Current models using spectral norm have suboptimal output-layer learning dynamics.

---

## Phase B — Consolidate Monolithic Modules (HIGH IMPACT)

### B.1 Split `equitile/` Mega-Module

**Problem**: `equitile/` contains **28 files** (~8,000 lines) implementing:
- Core model (`core.py` — 1,240 lines)
- 3 LM variants (`language.py` 1,192, `language_optimized.py` 687, `fast_lm.py` 613)
- Distributed training (`distributed.py` 994, `multigpu.py` 950)
- Profiling (`profiler.py` 1,076)
- RL, timeseries, vision, research, deployment, builder, dynamics, enhanced, async
- CLI demos (`lm_demo/` — 8 files)

This is **4× larger than any other package** and has no clear internal boundary.

**Solution**: Reorganize into focused sub-packages:

```
equitile/
├── __init__.py              # Public API only
├── core/                    # Model + config + topology + kernels
│   ├── model.py             # EquiTile, EquiTileEP (from core.py)
│   ├── config.py            # All configs (already consolidated)
│   ├── topology.py          # TileGraph, TileState
│   └── kernels.py           # compute_* functions
├── training/                # Training infrastructure
│   ├── optimizer_mixin.py
│   ├── task_handler.py
│   ├── distributed.py       # Merged distributed + multigpu (B.2)
│   └── async_execution.py
├── language/                # LM variants
│   ├── __init__.py
│   ├── components.py         # Shared TileAttention, TileFeedForward (Phase B.4)
│   ├── canonical.py         # language.py content
│   ├── optimized.py          # language_optimized.py content
│   └── fast.py               # fast_lm.py content
├── analysis/                # Profiling + dynamics + research
│   ├── profiler.py
│   ├── dynamics.py
│   └── research.py
├── deployments/             # RL, timeseries, vision, deployment
│   ├── rl.py
│   ├── timeseries.py
│   ├── vision.py
│   └── deployment.py
└── _internal/               # Builder, enhanced, state_types, utils
    ├── builder.py
    ├── enhanced.py
    └── state_types.py
```

**Note**: `lm_demo/` should move to `examples/equitile_lm/` (it's demo code, not library code).

**Impact**: The current flat structure makes it impossible to understand EquiTile's architecture without reading 28 files. The reorganization makes the module boundaries explicit and discoverable.

### B.2 Merge `distributed.py` + `multigpu.py` → One Module

**Problem**: `distributed.py` (994 lines) and `multigpu.py` (950 lines) = **1,944 lines** of overlapping distributed-training code:
- `TileCommunicator` (distributed.py) vs `NCCLCommunicator` (multigpu.py) — same NCCL primitives, different class names
- `DistributedEquiTile` vs `MultiGPUEquiTile` — same training loop, different wrapper

**Solution**:
1. Extract NCCL primitive wrappers → `equitile/_nccl.py` (~200 lines, both files currently duplicate `all_reduce`, `broadcast`, etc.)
2. Single `DistributedEquiTile` class; `MultiGPUEquiTile` becomes a deprecated alias or thin config wrapper
3. Single `TileCommunicator` with `backend: Literal["nccl", "gloo"]` parameter
4. Deprecate `NCCLCommunicator` (alias for backward compat)

**Estimated savings**: ~400–500 lines.

### B.3 Merge Reduction: `execution/` (23 files)

**Problem**: The `execution/` package has **23 files** for the AutoScientist agent — many are tiny single-concern modules:

| File | Lines | Concern |
|------|-------|---------|
| `task.py` | ~80 | `ExperimentTask` dataclass |
| `state.py` | ~100 | DB state wrapper |
| `decisions.py` | ~60 | `DecisionLogger` |
| `failure_tracker.py` | ~90 | `FailureRecord` + tracker |
| `safety.py` | ~50 | Safety checks |
| `algorithm_constraints.py` | ~50 | Constraint helpers |
| `experiment_checks.py` | ~60 | Experiment validators |

**Solution**: Consolidate related concerns:
- `execution/_state.py` — `ExperimentState` + `DecisionLogger` + `FailureTracker` (all DB-adjacent)
- `execution/_guards.py` — `Safety` + `algorithm_constraints` + `experiment_checks` (all validation)
- `execution/engine.py` — `ExecutionEngine` (the agent loop)
- `execution/strategy.py` — Keep (already substantial)
- `execution/resources.py` — Keep
- `execution/dashboard.py` — Keep

**Estimated savings**: ~6 files → 3 files; ~150 lines from reduced import boilerplate.

**Alternative**: Leave as-is if the current 23-file structure aids testing/isolation. The primary cost is *navigation*, not duplication. **Reassess after B.1.**

### B.4 Shared LM Components (`equitile/_components.py`)

**Problem**: Three LM files (`language.py`, `language_optimized.py`, `fast_lm.py`) each implement:
- `TileAttention` (multi-head causal attention as tile)
- `TileFeedForward` (GLU/SwiGLU FFN as tile)
- `PositionalEncoding` (learned or RoPE)
- `CausalMask` (buffer registration)

**Note from TODO0.md**: Previous analysis found these are "divergent architectures, max savings ~50-100 lines." However, with the A.1 `EnergyModel` unification, the *shared* components become more valuable because they can be reused across all three variants with a common training interface.

**Solution**: Extract `TileAttention`, `TileFeedForward`, `PositionalEncoding`, `CausalMask` → `equitile/language/components.py`. The three LM variants retain their unique training loops but share building blocks.

**Estimated savings**: ~150–250 lines.

---

## Phase C — Trainer & Pipeline Architecture (HIGH IMPACT)

### C.1 Single Trainer for All Learning Rules

**Problem**: The codebase has **multiple training pathways**:
- `CoreTrainer` (`core/trainer.py`) — handles standard backprop + custom `train_step`
- `_TaskTrainer` (`hyperopt/tasks.py`) — thin wrapper around `CoreTrainer.from_task`
- `RLTrainer` (`training/rl.py`) — completely separate RL loop
- `run_from_runconfig` — yet another path with inline trainer selection
- `ExecutionEngine` (`execution/engine.py`) — wraps all of the above with retry/circuit-breaker
- `run_pl_trial` (`lightning_/experiment.py`) — PyTorch Lightning path

**The consequence**: `_train_step` in `CoreTrainer` has a 3-way conditional dispatch (model.train_step / optimizer.step with target / standard forward-backward), and every new learning rule requires touching this method.

**Solution**: Unify around the `EnergyModel` protocol (A.1) and a thin `Trainer` abstraction:

```python
class Trainer:
    """Single training loop for all models.

    Dispatch:
    - EnergyModel → EBMTrainer (settle + contrastive update)
    - Has train_step → delegate to model
    - Else → standard forward + loss.backward()
    """

    def train_epoch(self) -> dict[str, float]:
        match self.model:
            case EnergyModel():
                return self._ebm_epoch()
            case _ if hasattr(self.model, "train_step"):
                return self._custom_step_epoch()
            case _:
                return self._backprop_epoch()
```

**Benefits**:
- Eliminates `_TaskTrainer` wrapper (it just adds metric renaming)
- `run_from_runconfig` simplifies to `Trainer.fit()`
- `ExecutionEngine` no longer needs special-cased Lightning path
- New learning rules: implement `EnergyModel` → automatically work with all trainers

**Estimated savings**: ~300–500 lines across `core/trainer.py`, `hyperopt/tasks.py`, `execution/engine.py`.

### C.2 Config Validation at I/O Boundary (Pydantic)

**Problem**: `TrainerConfig` is a mutable dataclass with `dict[str, Any]` fields (OmegaConf compatibility). There is **no runtime validation** of config values — invalid configs fail late with confusing errors (e.g., `model_kwargs={"input_dim": "seven"}` fails at `model = model_cls(**...)`).

**AGENTS.md mandate**: *"Pydantic v2 at I/O boundaries for runtime validation."*

**Solution**: Add a Pydantic `TrainerConfigSchema` that validates configs at the YAML/dict boundary, then converts to the OmegaConf-compatible `TrainerConfig`:

```python
class TrainerConfigSchema(BaseModel):
    model: str
    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    # ... all fields with validation constraints
```

**Benefits**: Fail-fast with clear error messages. `dict[str, Any]` stays on `TrainerConfig` (OmegaConf boundary), but the *input* is validated.

### C.3 Checkpoint Format Standardization

**Problem**: Three different checkpoint formats:
- `CoreTrainer._save_checkpoint` — `{"epoch", "model_state_dict", "optimizer_state_dict", "metrics", "config", "global_step"}`
- `EquiTile.save_checkpoint` — `EquiTileStateDict` (TypedDict with `{"model_state_dict", "task_type", "config", "training", "optim_*", "lr_scheduler"}`)
- `ExecutionEngine` — stores Optuna trial artifacts in zip files

**Solution**: A single `Checkpoint` TypedDict + `save_checkpoint(path, model, optimizer, ...)` / `load_checkpoint(path)` in `core/checkpoint.py`:

```python
class Checkpoint(TypedDict):
    model_state_dict: dict[str, Tensor]
    optimizer_state_dict: dict[str, object] | None
    scheduler_state_dict: dict[str, object] | None
    config: dict[str, object]
    epoch: int
    global_step: int
    metrics: dict[str, object]
    metadata: dict[str, object]
```

**Estimated savings**: ~100 lines + eliminates a class of load-failure bugs.

---

## Phase D — Automated Code Quality (TOOLING, NOT MANUAL EDITS)

*These are achieved through automated refactoring tools and global search/replace, NOT an endless series of individual edits.*

### D.1 Eliminate `Any` via Codemod

**Command**: Use `ruff`'s `UP` (pyupgrade) + a custom codemod script:
```bash
# Replace `dict[str, Any]` → `dict[str, object]` everywhere EXCEPT OmegaConf-structured dataclasses
ruff check --select UP --fix .
# Then manually verify OmegaConf boundaries (TrainerConfig fields must stay `Any`)
grep -rn "from typing import Any" bioplausible/ --include="*.py" | \
  grep -v "test" | grep -v "__pycache__"
```

**Files with `Any`** (from TODO0.md, non-core):
- `equitile/config.py` (6) — OmegaConf boundary, keep `Any`
- `equitile/builder.py` (~8) — `object`
- `equitile/research.py` (~12) — `object`
- `equitile/deployment.py` (~15) — `object` + fix missing `InferenceRequest` import
- `equitile/benchmarks/*.py` (~20) — `object`
- `zoo/models/eqprop/*.py` (5–10 each) — `object`
- `hyperopt/graph_task.py` (4) — fix missing `os` import + `object`
- `validation/tracks/*.py` (~30) — `TypedDict` for metrics dicts

**Approach**: Write a one-shot codemod script (`scripts/refactor_any_to_object.py`) that:
1. Finds all `from typing import Any` imports
2. Replaces `Any` with `object` in annotations (except in OmegaConf-structured classes)
3. Removes unused `Any` imports
4. Runs `ruff check --fix .` to clean up

**Then**: Verify with `pyright` — the 5 pre-existing errors should resolve (2 are missing imports caught by this pass).

### D.2 Add `__all__` via Script

```bash
# For each bioplausible/**/*.py without __all__:
#   1. Parse top-level public names (classes, functions, assignments)
#   2. Generate __all__ = ["Name1", "Name2", ...]
#   3. Insert after the last import or module docstring
python scripts/add_all_exports.py bioplausible/
```

**Exclude** `_`-prefixed modules (internal) and `__init__.py` (handled separately — should re-export only public API per AGENTS.md).

### D.3 t-string Migration for Logging (PEP 750)

**Current state**: All core files use `%s`-style deferred logging (correct per AGENTS.md). Non-core files (equitile demos, execution, hyperopt) use f-strings:

```bash
grep -rn 'logger\.\(info\|warning\|error\|debug\)(".*{.*}"' bioplausible/ --include="*.py" | grep -v "f'"
```

**Approach**: Python 3.14+ supports t-strings. A codemod can convert f-string logging calls to t-strings:
```python
# Before (insecure for untrusted inputs)
logger.info(f"Task {task.name} failed: {e}")
# After (PEP 750 t-string)
logger.info(t"Task {task.name} failed: {e}")
```

**Note**: t-strings are a *superset* of `%s`-style — they provide the same deferred interpolation but with full expression power. Evaluate whether to standardize on t-strings across the board (replacing `%s` style) or keep `%s` for simple cases.

---

## Phase E — Test Architecture (HIGH LEVERAGE)

### E.1 Test Suite Reorganization

**Problem**: 114 test files in a flat `tests/` directory. No clear organization, no boundary between unit/integration/property/slow tests.

**Solution**:
```
tests/
├── conftest.py              # Shared fixtures (E.2)
├── unit/                    # Fast, isolated, no GPU, no data download
│   ├── core/
│   │   ├── test_registry.py
│   │   ├── test_energy_model.py    # New: EnergyModel protocol tests
│   │   └── test_checkpoint.py
│   ├── models/
│   │   ├── test_settling.py
│   │   ├── test_energies.py        # New: energy function tests
│   │   └── test_build_helpers.py
│   └── data/
│       └── test_task_factory.py
├── integration/             # Model + trainer + real (small) data
│   ├── test_eqprop_models.py   (merged from test_eqprop*.py)
│   ├── test_fa_models.py       (merged from test_fa*.py)
│   ├── test_equitile.py        (merged from test_equitile*.py)
│   └── test_trainer.py
├── property/                # Hypothesis property-based tests
│   ├── test_hidden_dims.py
│   ├── test_energy_convergence.py
│   └── test_registry_query.py
└── slow/                    # @pytest.mark.slow — full epochs, real datasets
    ├── test_mnist_full.py
    └── test_lm_full.py
```

**Approach**: Move files in one batch (git mv preserves history). Update `pyproject.toml` testpaths.

### E.2 Shared Fixtures (`conftest.py`)

```python
@pytest.fixture(scope="session")
def synthetic_classification():
    """Deterministic synthetic classification data for all fast tests."""
    torch.manual_seed(42)
    X = torch.randn(200, 64)
    y = (X.sum(dim=1) > 0).long() % 10
    return X, y


@pytest.fixture
def equitile_small(synthetic_classification) -> EquiTile:
    """Minimal 2-layer EquiTile for fast unit tests."""
    return EquiTile(input_dim=64, output_dim=10, num_layers=2, tiles_per_layer=2)


@pytest.fixture
def mnist_quick_task() -> VisionTask:
    """MNIST task in quick_mode (100 samples, no download)."""
    return VisionTask("mnist", quick_mode=True)


@pytest.fixture
def eqprop_model() -> StandardEqProp:
    """Minimal EqProp model for settling/contrastive tests."""
    config = ModelConfig(name="test", input_dim=64, output_dim=10, max_steps=5)
    return StandardEqProp(config=config)
```

**Benefit**: Eliminates ~500 lines of duplicated setup across 100+ test files. Current tests each have `torch.manual_seed(42); X = torch.randn(...); ...` blocks.

### E.3 Property-Based Tests (Hypothesis)

**Targets** (pure logic, deterministic, fast):

| Module | Property to Test |
|--------|-----------------|
| `core/energies.py` (new) | Energy is non-negative; energy decreases during settling |
| `zoo/_settling.py` | `settle_activations_list` converges for contractive dynamics; trajectory length == steps |
| `core/registry.py` | `_QueryFilter.matches` is monotonic (adding constraints only removes results) |
| `zoo/base.py` | `compute_hidden_dims(n, k) == [k] * min(n, max_layers)` for all (n, k) |
| μPC scaling (A.4) | Output gradient magnitude ratio is independent of depth (after fix) |
| `graph/topology.py` | `topological_order` is valid for all DAGs; raises for all cyclic graphs |

### E.4 Coverage Floor → 85%

**Current**: 55% (floor=40% in `pyproject.toml`).

**Action**:
1. Raise `--cov-fail-under=85` in `pyproject.toml` (CI gate)
2. Add tests for uncovered core:
   - `core/energy_model.py` (new — 100% coverage from property tests)
   - `core/checkpoint.py` (new — save/load round-trip tests)
   - `core/registry.py` — compatibility check, export_yaml
   - `zoo/_settling.py` — convergence, trajectory shapes, EquilibriumFunction backward
3. Integration tests for the unified `EBMTrainer` (A.1) — ensures PC/EP/CHL all train

**Do NOT** chase coverage in:
- `execution/` — AutoScientist is integration-tested via `test_scientist*.py`
- `equitile/lm_demo/` — demo code, should move to `examples/`
- `p2p/` — DHT networking, hard to unit test

---

## Phase F — Dependency & Build Hygiene

### F.1 Optional Dependencies Audit

**Problem**: `pyproject.toml` has **29 required dependencies** including heavy packages (`PyQt6`, `pyqtgraph`, `transformers`, `datasets`, `onnx`, `onnxscript`, `fastapi`, `uvicorn`, `kademlia`) that are not needed for core usage.

**Solution**: Move to optional dependency groups:

```toml
[project]
dependencies = [
    "torch>=2.0",
    "numpy",
    "tqdm",
    "rich",
    "pydantic>=2.0",
    "omegaconf>=2.3",
]

[project.optional-dependencies]
vision = ["torchvision", "scikit-learn"]
lm = ["transformers", "datasets", "tokenizers"]
ui = ["PyQt6", "pyqtgraph", "matplotlib", "seaborn"]
p2p = ["kademlia", "uvicorn", "fastapi"]
export = ["onnx", "onnxscript"]
hyperopt = ["optuna", "tabulate"]
analysis = ["pandas", "scipy", "matplotlib"]
gym = ["gymnasium"]
knowledgebase = ["gpytorch", "botorch", "sympy"]
graphs = ["torch-geometric>=2.5", "networkx"]
spiking = ["snnTorch>=0.8"]
llm = ["openai>=1.0"]
dev = ["pytest>=8.0", "pytest-cov", "pytest-xdist", "pytest-qt",
      "pytest-mock", "hypothesis>=6.0", "ruff>=0.6", "pyright>=1.1",
      "pre-commit", "build", "twine", "pip-audit"]
full = ["bioplausible[vision,lm,ui,p2p,export,hyperopt,analysis,gym]"]
```

**Benefit**: `pip install bioplausible` no longer pulls 2+ GB of packages for users who just want the core models.

### F.2 Fix Pre-Existing Pyright Errors

Two files have **known broken imports** (ocumented in TODO0.md):
- `deployment.py:717` — `InferenceRequest` undefined (missing import)
- `hyperopt/graph_task.py:28-32` — `os` undefined (missing import)

**Action**: Add the missing imports. These should be caught by D.1's codemod pass.

---

## Phase G — Documentation

### G.1 Architecture Decision Records (ADRs)

**Location**: `docs/adr/` (new, NOT in archive)

**Priority ADRs** (write these as part of the refactoring):
1. **ADR-001**: Unified Energy-Based Model framework (Phase A) — documents the PC/EP/CHL unification per Millidge et al.
2. **ADR-002**: μPC output-node scaling (A.4) — documents the fix and the FabricPC provenance
3. **ADR-003**: Single Trainer for all learning rules (C.1)
4. **ADR-004**: Optional dependency groups (F.1)
5. **ADR-005**: `Any` elimination strategy & OmegaConf boundary (from TODO0.md Phase 2)

### G.2 AGENTS.md Rules → Automated Enforcement

Verify that every `@AGENTS.md` rule is enforced by tooling, not memory:

| Rule | Enforcement |
|------|-------------|
| `No Any` | `ruff` custom check / pyright config |
| `No print()` | `ruff` `T20` (flake8-print) |
| `No f-string logging` | `ruff` `G004` (flake8-logging-format) |
| `match/case over if/elif` | `ruff` `RET` (flake8-return) + review |
| `Protocol over ABC` | pyright `reportGeneralTypeIssues` + review |
| `frozen=True, slots=True` | pyright + review |
| `__all__` on public modules | `ruff` `F401` (unused import) catches missing |
| `t-strings for logging` | `ruff` `G004` + manual for t-string upgrade |

**Action**: Add missing `T20`, `G004`, `RET` to `[tool.ruff.lint]` select list if not already present.

---

## Execution Plan

### Dependency Chain

```
A.4 (μPC fix) ──────► (independent, can ship first)
     │
A.1 (EnergyModel) ──► A.2 (graph unification) ──► A.3 (energy library)
     │                                          │
     ▼                                          ▼
C.1 (Unified Trainer) ──────────────────► C.2 (Pydantic config)
     │
     ▼
C.3 (Checkpoint std.)

B.1 (equitile split) ──► B.2 (distributed merge) ──► B.4 (LM components)
     │
B.3 (execution consolidate) — independent

D.1 (Any codemod) ──► D.2 (__all__ script) ──► D.3 (t-string migration)

E.1 (test reorg) ──► E.2 (fixtures) ──► E.3 (property tests) ──► E.4 (coverage 85%)
```

### Session Log & Remaining Work

| Session | Focus | Status | Est. Days | Impact |
|---------|-------|--------|-----------|--------|
| **1** | **A.4** — μPC output scaling fix | ✅ Done | 0.5 | **Critical correctness fix** |
| **2** | **A.1 + A.3** — EnergyModel + energies | ✅ Done | 2–3 | **Eliminates deepest duplication** |
| **3** | **C.1** — Unified Trainer using `EnergyModel` | ⏳ Next | 1–2 | **Simplifies all training paths** |
| **4** | **B.2** — Merge distributed/multigpu | 🔲 | 1 | −400 lines |
| **5** | **D.1** — `Any` → `object` codemod | 🔲 | 0.5 | Type safety |
| **6** | **E.1 + E.2** — Test reorg + fixtures | 🔲 | 1 | Test velocity |
| **7** | **A.2** — Unify `graph/` with `zoo/_settling.py` | 🔲 | 0.5 | Completes A |
| **8** | **B.1** — equitile/ reorganization | 🔲 | 1–2 | Navigation clarity |
| **9** | **E.3 + E.4** — Property tests + coverage 85% | 🔲 | 1–2 | Quality gate |
| **10** | **C.2 + C.3** — Pydantic config + checkpoint std. | 🔲 | 1 | I/O robustness |
| **11** | **F.1** — Optional deps split | 🔲 | 0.5 | Install footprint |
| **12** | **D.2 + D.3 + G.1** — `__all__`, t-strings, ADRs | 🔲 | 1 | Polish |

**Total remaining**: ~9–13 days. Next critical session is **#3 (C.1)** — wiring the `EnergyModel` protocol into `CoreTrainer`.

---

## Verification Gates (After Each Session)

```bash
# Fast loop (during development)
ruff format . && ruff check --fix .
pyright .
pytest -q --no-cov                          # ~45s

# Full gate (before commit)
pytest --cov=bioplausible --cov-fail-under=85  # ~4min
pip-audit                                    # security
```

**Phase-specific**:
- After A.1: `grep -r "def train_step" bioplausible/zoo/models/ | wc -l` should decrease as models adopt `EnergyModel`
- After A.4: New test asserts μPC output gradient scaling (property test)
- After C.1: `_TaskTrainer` removed; `run_from_runconfig` simplified
- After D.1: `grep -r "from typing import Any" bioplausible/ | grep -v test | grep -v __pycache__ | wc -l` → 0 (except OmegaConf boundary)

---

## What This Plan Deliberately Does NOT Include

1. **Larger lint-style issues**: `@typing.override` suggestions (5,313 from TODO0.md), line-length adjustments, import sorting — these are `ruff`'s job.
2. **Converging 12 `_build_layers` implementations**: Architectures are genuinely divergent (Linear vs Conv3d vs GCNConv vs Hebbian). CPR <50 lines. Not worth the abstraction risk.
3. **Aggressive elimination of runtime-mutable dataclasses** (`TrainerConfig`, `DistributedConfig`): These are mutable because callers mutate them. Freezing would require refactoring all callers. Low ROI.
4. **Rewriting tests in `docs/archive/`**: Excluded by scope.
5. **Re-implementing the FabricPC graph module from scratch**: The current `graph/` code is adapted from FabricPC and works. A.2 unifies its settling with `zoo/_settling.py`; that's sufficient.

---

*This plan supersedes TODO0.md Phases 3–4. Phases 0–2 (completed) remain the foundation.*

---

## Session 7 Handoff Notes (2026-07-29)

### What Was Done

1. **Phase A.4** (`zoo/base.py` + 10+ callers in `zoo/models/`):
   - Added `output_scaling_mode: Literal["uniform", "mupc"]` to `ModelConfig` (default `"mupc"`)
   - Added `layer_role: LayerRole = "hidden"` parameter to `BioModel.apply_spectral_norm()`
   - Output layers with `output_scaling_mode="mupc"` rescale weights to remove the √L fan-in factor
   - Updated all layer-build loops in `standard_eqprop.py`, `mom_eq.py`, `sparse_eq.py`, `predictive_coding.py`, `fa.py` (6 classes), and `wrappers.py`

2. **Phase A.1** — Created `core/energy_model.py`:
   - `EnergyModel` Protocol with `energy()`, `settle()`, `contrastive_update()`
   - `EBMTrainer` class with free/nudge/contrastive loop and BPTT fallback
   - Runtime-checkable (`@runtime_checkable`) — models satisfy structurally, no inheritance needed

3. **Phase A.3** — Created `core/energies.py`:
   - `prediction_error_energy`, `supervised_energy`, `hybrid_energy`, `contrastive_energy`, `mse_energy`, `node_energy`

4. **Phase F.2** — Fixed 5 pre-existing Pyright errors:
   - `deployment.py:717` — defined missing `InferenceRequest` dataclass
   - `hyperopt/graph_task.py:28-32` — added `import os`

### What's Blocking Session 3 (C.1 — Unified Trainer)

The `EnergyModel` protocol and `EBMTrainer` exist but are **not wired** into `CoreTrainer`. The current `_train_step` in `core/trainer.py` (line 834) still uses:
```python
if hasattr(self.model, "train_step"): ...
elif inspect.signature(...): ...
else: ...
```

**Next step**: Add a `match/case` dispatch before the existing checks:
```python
match self.model:
    case EnergyModel():
        return EBMTrainer(...).train_step(x, y)
```

This is ~10 lines of code. The complexity is deciding how `EBMTrainer` gets its hyperparams (lr, free_steps, beta, etc.) — either from `TrainerConfig` or from the model's config.

### A.2 (graph/ unification) Is Simplified

Since we already have `zoo/_settling.py` with `settle_single_state()` and `settle_activations_list()`, and `graph/inference.py` has its own `InferenceSGD.settle()`, Phase A.2 is now just:
1. Have `InferenceSGD` implement `EnergyModel.settle()` by delegating to `zoo/_settling.settle_activations_list()`
2. Delete the duplicated loop in `graph/inference.py`

### Ruff Warnings to Ignore

The 5,447 remaining `ruff check` warnings are all `@typing.override` suggestions (PLE, PLC, PLR) — **not actionable**. They come from a `ruff` rule (`PLE`?) that flags every method override as needing `@typing.override`, inflating the count. If quieting them is desired, add `"PLE", "PLC", "PLR"` exclusions for the specific patterns.

### Pre-Existing Issues (Unrelated to Refactoring)

1. **`test_onnx.py` warnings**: Tensor attributes assigned during export should be registered as buffers. This is a real issue in `equitile/core.py` and `equitile/kernels.py` but is out of scope.
2. **`torch.jit.script` deprecation**: 14 warnings across `zoo/_settling.py` and `graph/`. Python 3.14+ compatibility requires migrating to `torch.compile`.
3. **`sklearn.datasets` NumPy 2.5 deprecation**: In `test_new_domains.py` — Python 3.14 / NumPy 2.5 changed `.shape` assignment behavior. Pre-existing, not blocking.

### Test Coverage Gap for A.1/A.3

New code (`core/energy_model.py`, `core/energies.py`) has **zero tests**. Before raising coverage to 85% (E.4), add:
- `tests/unit/core/test_energy_model.py` — Test `EnergyModel` protocol structural typing, `EBMTrainer` fallback
- `tests/unit/core/test_energies.py` — Test each energy function with known inputs/outputs, verify non-negativity

These are quick to write and would add ~2% coverage by themselves.
