```markdown
# REFACTOR6 — Structural Debt Assessment (2026-08-14)

**Context**: REFACTOR5 completed all consolidation streams (LOOP/FUNNEL/MEASURE/RULE/
REGISTER/PRUNE/STRATEGY/OPTIMIZER/CACHING/ROOT-HYGIENE) and enabled the GPU kernel
backend. The REFACTOR5 "Non-Goals" section named three structural items for a future
REFACTOR6: god-object splits, `BenchmarkResult` merge, and settling-loop merge. This
document is the **assessment** of those items (per AGENTS.md: working functionality >
consolidation; every change routes through a seam or adds a frozen-signature one).

**Decision**: All three assessed items are **KEPT as-is** (safe to defer). The codebase
stays green. No risky split is forced. Triggers to revisit are documented below.

---

## 1. God-Object Splits — KEEP (high-risk, low-payoff)

| God object | Size | Nature |
|-----------|------|--------|
| `core/trainer.py` (`CoreTrainer`) | 1918 lines, 72 methods | Interleaved lifecycle: setup → data/model/propagator/optimizer → fit → `dispatch_train_step` → checkpointing → callbacks → history |
| `knowledge/kb.py` (`KnowledgeBase`) | 1205 lines, 15 classes | Cohesive SQLite-backed facade; 30+ methods are distinct public read/query surfaces |
| `execution/strategy.py` | 1098 lines | Orchestration strategy state machine |

**Why KEEP**:
- `CoreTrainer`'s methods share instance state and are called in a strict order;
  extracting any subset touches the *working training loop* — a mistake silently
  changes training dynamics. AGENTS.md explicitly prioritizes working functionality
  over consolidation, and REFACTOR5 already non-goaled this.
- `KnowledgeBase` is not a god-object in the problematic sense — it is one cohesive
  data-access unit; splitting it would fragment a natural boundary, not reduce
  duplication.

**Safe seams IF revisited** (low-risk, mechanical, do one at a time with a frozen
signature):
1. **Checkpointing mixin**: `_should_save_checkpoint` / `_save_checkpoint` /
   `_check_early_stopping` / `_save_history` / `load_checkpoint` are already thin
   wrappers over `core/checkpoint.py`. Move them to a `CheckpointMixin`
   (`core/checkpoint_mixin.py` — already exists as a layer entry) delegating to the
   same `core/checkpoint` functions. Pure move, no behavior change. *If any split is
   done, start here.*
2. **Hardware facade logic**: `_apply_hardware` / `_hardware_meta_for` (already
   extracted as a helper this session) could move to a small `_hardware.py` module.
   Very low risk.
3. **`TrainerConfig` / `LMTrainingConfig`** are already separate classes in the file;
   could move to `config/` if desired (low value).

**Trigger to revisit**: a new training feature requires editing `CoreTrainer` in a way
that the 72-method monolith makes unreasonably error-prone, OR a god-object split is
needed to enable a downstream seam (e.g., a new trainer variant).

---

## 2. `BenchmarkResult` Merge — KEEP (coexist sanctioned, ground rule 8)

There are **4 distinct** `BenchmarkResult` classes with incompatible schemas:

| File | Layer | Schema purpose |
|------|-------|----------------|
| `evaluation/base.py` | L1 | Light metrics holder: `model_name/task_name/metrics/params_count/flops/energy/wall_time/peak_memory/metadata` |
| `benchmarks/rigorous.py` | L6 | Statistical run: `throughput_stats/time_per_epoch_stats/memory_mb/final_train_loss/val_loss/val_ppl/system_info/parameter_count` |
| `analysis/tile_profiler.py` | L6 | Throughput timing: `batch_size/mean_time_ms/std_time_ms/min_time_ms/max_time_ms/throughput_samples_per_sec` |
| `benchmarks/compare_nanoGPT.py` | L6 | LM training: `model_name/parameter_count/train_loss/val_loss/train_ppl/val_ppl/tokens_per_sec/memory_mb/training_time_sec` |

**Why KEEP**: These are not duplicates — they are per-domain result records consumed by
independent benchmark runners. A forced single class would need either a bag-of-fields
superset (defeats typing) or a Protocol (adds indirection with no consolidation value).
REFACTOR5 ground rule 8 already sanctioned coexistence.

**Safe seam IF revisited**: introduce a shared `BenchmarkResultProtocol` (structural)
in `evaluation/base.py` and have the others satisfy it structurally — only if a
consumer genuinely needs to accept all four. Low priority; not needed today.

---

## 3. Settling-Loop Merge (Family A/B + MEP Settler) — KEEP (numerics risk)

`core/local_learning/settling.py` already consolidates two settling families:
- **Family A** — single-hidden-state: `settle_single_state` (`EqPropModel` subclasses)
- **Family B** — activations-list: `settle_activations_list` (`StandardEqProp`,
  `DirectedEP`, etc.)

Separately, `zoo/mep/optimizers/settling.py::Settler` is a **distinct gradient-based
energy optimizer** with its own adaptive LR, patience, momentum, and softmax
temperature — used by MEP presets.

**Why KEEP**: Family A and B are already unified under one module with shared
convergence helpers (the consolidation REFACTOR4/5 wanted is done). The MEP `Settler`
is a genuinely different optimizer; folding it into the core settling primitives is
the "numerics risk, low gain" item REFACTOR5 flagged. It could subtly change MEP
preset training dynamics.

**Trigger to revisit**: if the MEP `Settler`'s adaptive steps are shown to be a strict
superset of `settle_single_state`'s, and a single parametrized primitive can express
both with a frozen signature — only then unify, with gradient-parity as the gate.

---

## Status

| Item | Decision | Risk | Action |
|------|----------|------|--------|
| God-object splits | KEEP | High | Defer; safe seams documented (checkpoint mixin first) |
| `BenchmarkResult` merge | KEEP | Medium | Coexist (ground rule 8); Protocol only if a consumer needs all four |
| Settling-loop merge | KEEP | High (numerics) | Defer; MEP `Settler` stays separate |

**Re-entry is unchanged and green**: `check_imports.py` exit 0, `check_seams.py` exit 0,
full suite green (2002 passed, 0 failed, 65.56% cov), kernel/triton/caching/hardware
tests pass.

**Real remaining work** is not these splits — it is the deferred kernel tuning
(REFACTOR5 improvement opportunities #8/#9): close the large-batch time gap and the
~5pt accuracy gap so the GPU kernel beats BPTT on both axes.
```
