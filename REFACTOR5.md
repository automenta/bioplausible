```markdown
# REFACTOR5 — Consolidation Completion & Architecture Hardening

**Context**: REFACTOR4 achieved LOOP/FUNNEL/GATE-0/CI SEAMS with MEASURE/RULE structurally complete via relocation. The codebase is green, gate-enforced, and parked at a clean checkpoint. REFACTOR5 addresses the **remaining structural debt** that was explicitly deferred, exempted, or assessed as "coexist" — plus the duplication patterns identified in the architecture review.

**Philosophy**: AGENTS.md priorities — working functionality > consolidation. No semantic changes to training dynamics. Every change routes through an existing seam or adds a new one with a frozen signature. Allowlists ratchet only when a violation is genuinely eliminated.

---

## Status Summary

| Stream | State | Blockers |
|--------|-------|----------|
| **LOOP** | Core done; 4 allowlisted sites remain | `zoo/mep` inline loops, `target_prop`, `eqprop_diffusion`, `graph/training`, `zoo/optimizers/ewc.py` |
| **FUNNEL** | Complete | — |
| **MEASURE** | Complete (coexist decision) | 3 distinct `BenchmarkResult` classes kept as-is |
| **RULE** | Structural relocation done; phase collapse gated | 5→2 dispatch collapse only if research need forces it |
| **REGISTER** | Complete (top `_LAZY` 84→5) | — |
| **PRUNE** | Complete (dead code + redundant tests) | — |
| **NEW: STRATEGY DEDUP** | **Active** | `core/optimization/strategies` ↔ `zoo/mep/optimizers/strategies` |
| **NEW: EQPROP UNIFICATION** | **Active** | 3 engine variants: `EquilibriumMLP` / `LoopedMLP` / `EqPropKernel` |
| **NEW: OPTIMIZER SURFACE** | **Active** | `register_optimizer` (learning-rule) vs `CompositeOptimizer` (MEP presets) |
| **NEW: EXPERIMENT CACHING** | **Active** | Data/model construction per-probe without reuse |
| **NEW: ROOT HYGIENE** | **Active** | 12+ `.db` files in project root |

---

## Work Streams

### 1. STRATEGY DEDUP — Unify Gradient/Update/Constraint/Feedback Strategies

**Problem**: Two parallel hierarchies implementing the same concepts:

| Location | Purpose | Key Classes |
|----------|---------|-------------|
| `core/optimization/strategies/` | Core optimizer strategies | `GradientStrategy`, `UpdateStrategy`, `ConstraintStrategy`, `FeedbackStrategy` |
| `zoo/mep/optimizers/strategies/` | MEP preset strategies | `GradientStrategy`, `UpdateStrategy`, `ConstraintStrategy`, `FeedbackStrategy` |

**Decision**: `core/optimization/strategies` is the canonical L1 location. `zoo/mep/optimizers/strategies` becomes a thin re-export/adapter layer.

**Actions**:
1. **Audit equivalence** — For each strategy pair (e.g., `MuonUpdate` in both), verify identical signatures and behavior. Document any semantic differences.
2. **Consolidate implementations** — Move unique MEP strategies to `core/optimization/strategies` (e.g., `DionUpdate`, `FisherUpdate`, `NaturalGradient`, `ErrorFeedback`, `NoFeedback`, `NoConstraint`, `SpectralConstraint`).
3. **Update MEP imports** — `zoo/mep/optimizers/__init__.py` and `zoo/mep/presets/__init__.py` import from `core.optimization.strategies`.
4. **Delete duplicate files** — Remove `zoo/mep/optimizers/strategies/{gradient,update,constraint,feedback,base}.py`.
5. **Verify** — `smep`, `sdmep`, `local_ep`, `natural_ep`, `muon_backprop`, `smep_fast` presets produce identical optimizer instances.

**Allowlist impact**: None (internal consolidation).

---

### 2. EQPROP UNIFICATION — Single Engine with Backend Selection

**Problem**: Three EqProp implementations with overlapping capability:

| Engine | Location | Memory | Backend | Use Case |
|--------|----------|--------|---------|----------|
| `EquilibriumMLP` | `zoo/models/eqprop/_energy.py` | O(N) activations | PyTorch autograd | Layered, consolidated engine |
| `LoopedMLP` | `zoo/models/eqprop/looped_mlp.py` | O(1) implicit / O(N) contrastive | PyTorch | Facade over `EquilibriumMLP`; 3-arg legacy ctor |
| `EqPropKernel` | `acceleration/kernels.py` | **O(1) contrastive** | NumPy/CuPy/Triton | No autograd; FPGA/neuromorphic path |

**Decision**: `EquilibriumMLP` is the canonical PyTorch engine. `LoopedMLP` remains the registered facade (`eqprop_mlp`). `EqPropKernel` becomes a selectable **backend** via `TrainerConfig.target_hardware` or a new `backend` field, not a separate model class.

**Actions**:
1. **Add backend routing to `LoopedMLP`** — Extend `ModelConfig.extra` or `TrainerConfig` with `eqprop_backend: Literal["pytorch", "kernel", "triton"]`.
2. **Wire `EqPropKernel` into `CoreTrainer`** — In `dispatch_train_step`, when `model.backend == "kernel"` and `gradient_method == "contrastive"`, delegate to `EqPropKernel.train_step` (convert tensors ↔ arrays at boundary).
3. **Unify hyperparameters** — Ensure `beta`, `settle_steps`, `settle_lr`, `gamma`, `max_steps` map 1:1 between `ModelConfig` and `EqPropKernel` constructor.
4. **Expose Triton path** — `EqPropKernel` already has `TritonEqPropOps`; make it the default GPU path when `HAS_TRITON` and `use_gpu`.
5. **Deprecate implicit-equilibrium path for kernel backend** — The O(1) implicit path (`gradient_method="equilibrium"`) is PyTorch-only; kernel backend uses contrastive Hebbian exclusively.
6. **Verify parity** — Run GATE-0 suite + `test_equilibrium_parity.py` with `backend="kernel"` on MNIST; accuracy within 1% of PyTorch path.

**Allowlist impact**: None (new backend, existing models unchanged).

---

### 3. OPTIMIZER SURFACE — Single Learning-Rule Optimizer Protocol

**Problem**: Two incompatible calling conventions for learning-rule optimizers:

| Surface | Location | Signature | Used By |
|---------|----------|-----------|---------|
| `LearningRuleOptimizer` (Protocol) | `core/local_learning/rules/base.py` | `step(self, x, target)` | `CoreTrainer.dispatch_train_step`, `bioplausible.__init__` lazy exports |
| `CompositeOptimizer` | `zoo/mep/optimizers/composite.py` | `step(self)` (internal state) | `zoo/mep/presets` (`smep`, `sdmep`, etc.) |

**Decision**: `LearningRuleOptimizer` Protocol is the canonical L1 interface. `CompositeOptimizer` adapts to it via a wrapper.

**Actions**:
1. **Create adapter** — `core/local_learning/rules/composite_adapter.py`: `CompositeOptimizerAdapter(LearningRuleOptimizer)` wraps `CompositeOptimizer`, implements `step(x, target)` by running the composite's internal phases.
2. **Update MEP presets** — `zoo/mep/presets/__init__.py` returns `CompositeOptimizerAdapter(composite)` instead of raw `CompositeOptimizer`.
3. **Unify registration** — Register MEP presets via `register_optimizer` (canonical) with `credit_assignment_type="equilibrium"`. Remove duplicate registration paths.
4. **Verify** — `CoreTrainer` with `optimizer="smep"` works identically; `_is_learning_rule_optimizer` guard passes for adapted instances.

**Allowlist impact**: None (adapter preserves external behavior).

---

### 4. EXPERIMENT CACHING — Eliminate Per-Probe Redundancy

**Problem**: Each probe in `StaircaseRunner` re-instantiates:
- `DomainTask` + `DataLoader` (dataset download, transform, split)
- Model via `construct_model` (weights, buffers, optimizer state)

**Decision**: Add caching layer in `ProbeDriver` (L5) with explicit invalidation keys.

**Actions**:
1. **Dataset cache** — `ProbeDriver.__init__` accepts `dataset_cache: DatasetCache | None`. `DatasetCache` keyed by `(task_name, data_kwargs_hash, batch_size, num_workers, seed)`. Returns `(train_loader, val_loader, task_spec)` tuple.
2. **Model construction cache** — `ProbeDriver` accepts `model_cache: ModelCache | None`. `ModelCache` keyed by `(model_name, config_hash, input_dim, output_dim, device)`. Returns `nn.Module` with fresh parameters (state dict not cached).
3. **Integrate into `CoreTrainerDriver`** — Before `Registry.get` + `construct_model`, check cache. On cache miss, construct and store.
4. **Invalidation** — Cache key includes `config_hash` (SHA256 of canonical config via `config_key()`). Any hyperparameter change = new entry.
5. **Memory bound** — `ModelCache` uses `weakref.WeakValueDictionary` + LRU eviction (max 32 entries default).
6. **Verify** — Run a 10-model × 5-config × 3-seed campaign; measure probe startup time (target: <500ms cold, <50ms warm).

**Allowlist impact**: None (internal optimization).

---

### 5. ROOT HYGIENE — Artifact Organization

**Problem**: 12+ `.db` files in project root:
```
bioplausible.db, bioplausible2.db, bioplausible_kb.db,
compute_phase1_5.db, compute_phase1_5_v2.db, compute_phase1_5_v3.db,
compute_phase1_5_v3.backup.db, dummy.db, execution_state.db,
smoke.db, smoke_p15.db
```

**Decision**: All experiment artifacts → `artifacts/` (already exists). SQLite databases are experiment outputs, not source code.

**Actions**:
1. **Move existing DBs** — `mv *.db artifacts/` (preserve git history with `git mv`).
2. **Update connection strings** — Search for hardcoded `"bioplausible.db"`, `"sqlite:///bioplausible.db"`:
   - `hyperopt/storage.py:HyperoptStorage.__init__`
   - `cli/run.py:_DB_PATH`, `_STORAGE_URL`
   - `execution/_lifecycle.py:CheckpointManager`
   - `experiment/result_sink.py`
   - `bioplausible/execution/engine.py`
   - Replace with `artifacts/bioplausible.db` or environment variable `BIOPL_DB_DIR`.
3. **Add `.gitignore`** — `artifacts/*.db`, `artifacts/*.db-*` (WAL/SHM).
4. **Verify** — All HPO, checkpoint, and KB operations write to `artifacts/`; tests pass.

**Allowlist impact**: None (path change only).

---

### 6. LOOP ALLOWLIST — Resolve Remaining Exemptions

**Current `LOOP_ALLOW` entries** (from `tools/check_seams.py`):

| File | Reason | Resolution Target |
|------|--------|-------------------|
| `zoo/mep/__init__.py` | Inline MEP training loop (deferred) | Convert when touched (STRATEGY DEDUP enables this) |
| `zoo/mep/optimizers/__init__.py` | Inline MEP training loop (deferred) | Convert when touched (STRATEGY DEDUP enables this) |
| `zoo/models/target_prop.py` | Pure local `train_step` (target propagation) | **Keep** — legitimate bio-plausible local rule |
| `zoo/models/eqprop/eqprop_diffusion.py` | Tagged broken/deferred | **Delete** — no tests, no consumers, marked broken |
| `zoo/optimizers/ewc.py` | EWC loss-rule body (Fisher step moved to core) | Move `compute_ewc_loss` to `core/ewc.py`; zoo file becomes re-export |
| `graph/training.py` | Bespoke GraphStructure + param-dict (PCN) | **Keep** — documented exemption |

**Actions**:
1. **Delete `eqprop_diffusion.py`** — Confirm zero imports (`grep -r "eqprop_diffusion" bioplausible/ tests/`). Remove from `zoo/models/eqprop/__init__.py` exports.
2. **Move EWC loss rule** — `zoo/optimizers/ewc.py:compute_ewc_loss` → `core/ewc.py`; `zoo/optimizers/ewc.py` becomes `from bioplausible.core.ewc import compute_ewc_loss, register_ewc, update_fisher`.
3. **Convert MEP inline loops** — After STRATEGY DEDUP, `zoo/mep` presets return `CompositeOptimizerAdapter`; `zoo/mep/__init__.py` and `optimizers/__init__.py` lose their training loops. Remove from `LOOP_ALLOW`.
4. **Ratchet allowlist** — Edit `tools/check_seams.py:LOOP_ALLOW` to remove cleared entries.

**Verify**: `tools/check_seams.py` passes with smaller allowlist; GATE-0 suite green.

---

### 7. RULE PHASE COLLAPSE — Gate Decision

**Status**: Gated (high-risk, low-payoff). Current `dispatch_train_step` has 5 phases:
1. Energy model (`EnergyModel` → `EBMTrainer`)
2. Explicit propagator (`LearningRuleOptimizer`)
3. Model `train_step`
4. Learning-rule optimizer (`LearningRuleOptimizer`)
5. BPTT fallback

**Proposed 2-phase**: Energy model → Model `train_step` → BPTT.

**Decision**: **Do not implement unless a concrete research need forces it.** The 5-phase dispatcher correctly handles:
- Models with implicit equilibrium backward (return `None` from `train_step`)
- Models with contrastive `train_step` (EP, PC, TP, FA, Hebbian)
- Learning-rule optimizers as propagator OR optimizer
- Plain BPTT models

**Trigger to revisit**: A new algorithm family cannot be expressed without contorting the 5-phase dispatch, OR AutoScientist composition logic becomes unmaintainable due to phase complexity.

---

## Implementation Order

```
1. ROOT HYGIENE          (independent, zero risk)
2. STRATEGY DEDUP        (enables MEP conversion, internal only)
3. OPTIMIZER SURFACE     (depends on STRATEGY DEDUP)
4. LOOP ALLOWLIST        (depends on STRATEGY DEDUP + OPTIMIZER SURFACE)
5. EQPROP UNIFICATION    (depends on OPTIMIZER SURFACE for backend routing)
6. EXPERIMENT CACHING    (independent, L5 only)
7. RULE PHASE COLLAPSE   (gated — only if triggered)
```

---

## Verification Gates (CI-Enforced)

| Gate | Command | Must Pass |
|------|---------|-----------|
| Import layering | `uv run python tools/check_imports.py` | 0 violations / 0 cycles |
| Seam criteria | `uv run python tools/check_seams.py` | Violators ⊆ allowlist |
| GATE-0 parity | `uv run pytest tests/unit/validation/test_backprop_parity.py tests/integration/test_equilibrium_parity.py tests/integration/test_equilibrium_implicit_learns.py tests/unit/validation/test_parity_snapshots.py -o addopts="" -q` | 37 passed, 3 xfailed |
| Core trainer | `uv run pytest tests/unit/core/test_core_trainer.py tests/integration/test_smoke_training.py tests/unit/core/test_deployment_models.py -o addopts="" -q` | All green |
| MEP integration | `uv run pytest tests/integration/test_mep_integration.py -o addopts="" -q` | All green |
| Full suite | `uv run pytest --cov=bioplausible --cov-report=term-missing --cov-fail-under=55 -p no:warnings` | ≥55% coverage, 0 new failures |

---

## Non-Goals (Per AGENTS.md & REFACTOR4 Ground Rules)

- God-object splits (`core/trainer.py`, `knowledge/kb.py`, `execution/strategy.py`)
- Settling-loop merge (Family A/B) — numerics risk, low gain
- Visualization consolidation — UI preference
- Forced `BenchmarkResult` merge — coexistence sanctioned (ground rule 8)
- Backwards compatibility — NONE per AGENTS.md

---

## Re-Entry Protocol

```bash
cd /home/me/bioplausible
uv sync --extra dev
uv run python tools/check_imports.py
uv run python tools/check_seams.py
uv run pytest tests/unit/validation/test_backprop_parity.py \
  tests/integration/test_equilibrium_parity.py \
  tests/integration/test_equilibrium_implicit_learns.py \
  tests/unit/validation/test_parity_snapshots.py -o addopts="" -q
# Expect: 37 passed, 3 xfailed (GATE-0 locked)
```

Status truth lives **only** in this document's Status table + Ledger. No session logs.
```