# Bioplausible Short-Term Development Plan

**Goal**: Stabilize the codebase so that passing unit tests *are* the viability proof. No lengthy experiments, no UI/CLI work, no demos until the foundation is solid.

**Principle**: If it takes >30 seconds to run, it's not a unit test. If it requires real data download, it's not a unit test. The test suite must pass in <60s on CPU.

**References**:
- `RESEARCH.md` — full roadmap (deferred: Phases 1-10)
- `RESEARCH.pre.md` — refactoring prerequisites (Tiers 1-4 + Appendix A)

---

## Session Log

### 2026-07-31 — Sprint 1 Core Implemented (13 files + 1 new)

**Done this session:** tasks 1.1–1.9 and 1.11–1.14 (everything except 1.10 snapshot tests).

**Gate status after session:**
```bash
uv run ruff format --check .        # PASS (594 files)
uv run ruff check bioplausible/     # 2525 errors (baseline was 2521; +4 net, all plan-mandated TRY003)
uv run pyright .                    # 0 errors, 2290 warnings (unchanged)
uv run pytest tests/unit/ tests/property/ -q --no-cov   # 763 passed, 1 skipped in ~23s (CPU)
```
Whole-repo `ruff check` / coverage are pre-existing failures, not regressions.

**Net ruff delta explained (+4):**
- kb.py +5 `TRY003` (`raise KnowledgeBaseError("...") from e`) and model.py +2 `TRY003` (`raise LoadStateError(...) from e`) — **plan-mandated**; codebase already has ~10 unsuppressed TRY003 in model.py, so consistent with house style.
- Offsets: `_relax` no longer `complex-structure` (C901), `too-many-locals` gone, `_state.py` one `try-consider-else`/try-clause resolved, `TC003` fixed by TYPE_CHECKING import. All other touched files at baseline.

**Discovered during work (fixes embedded in this session):**
- `create_dynamic_config` (equitile/core/config.py) forwarded the *identical* `**kwargs` to both `TileGrowthConfig` and `DynamicEquiTileConfig` — latent bug. Now split via `fields(TileGrowthConfig)`.
- `LazyStats.reset()` mutated `self` — incompatible with frozen dataclass. Now `@staticmethod` returning a fresh `LazyStats`; callers updated (`lazy_eqprop.py`, `scaling_tracks.py: model.stats = model.stats.reset()`).
- `TrainingMetrics.to_dict()` used `self.__dict__` — broken under `slots=True`. Switched to `asdict(self)`.
- `_QueryFilter` needs `_predicates: tuple[_Predicate, ...]` as a **quoted-free** forward ref: Python 3.14 PEP 649 lazy annotations make `tuple[_Predicate, ...]` legal before `_Predicate` is defined (verified).
- `CreditAssignmentType` Literal needs `"backpropagation"` and `"local"` in addition to the 6 members listed in the table below — they are real values at registration sites.
- `pickle` import in model.py: ruff 0.16 inline suppression syntax is `# ruff: ignore[rule-name]` (code form `[S403]` triggers RUF100, `-- reason` suffix must follow on the same line).
- Logger placement in benchmark files: keep `import logging` in stdlib block; assign `logger = logging.getLogger(...)` only *after* all imports (avoids E402).
- kb.py early `return None` guards (e.g. no-data paths) live *inside* the narrowed try blocks, so the new `KnowledgeBaseError` chaining does not change behavior on those paths — 42 kb tests green.

**Remaining Sprint 1:**
- **1.10 snapshot tests** for the extracted helpers (never started).
- True CI gate cleanup (whole-repo ruff → 0, coverage → ≥50%) is a separate, larger effort; today's work only guarantees "no new violations from Sprint 1 code" modulo the +4 TRY003.

---

## Sprint 1: Foundation Hardening (Week 1-2)
*Only work that makes the test suite faster, stricter, and more trustworthy.*

### Must-Do (CI Correctness Blockers)

| # | Task | Ref | Done |
|---|------|-----|------|
| 1.1 | Add `bioplausible/core/exceptions.py` with domain hierarchy (`BioplausibleError`, `ConfigError`, `RegistryError`, `CheckpointError`, `KnowledgeBaseError`, `TrialExecutionError`, `PropagatorError`, `TileGraphError`) | pre.md 1.1 | ☑ |
| 1.2 | Narrow 3 broad `except Exception` sites to specific types + chain raise domain errors (`equitile/core/model.py`×2, `knowledge/kb.py`×5) | pre.md 2.2 | ☑ |
| 1.3 | Replace `print()` with `logging` in 4 equitile benchmark files (library code only) | pre.md 2.1 | ☑ |
| 1.4 | Fix bare `except X, Y:` → `except (X, Y):` in 12 files | pre.md 2.3 | ☑ |
| 1.5 | Add `@contextmanager _connect(db_path)` helper; migrate `execution/_state.py` 12 methods to use it | pre.md 1.6 | ☑ |

### High-Impact Architecture (Testability Enablers)

| # | Task | Ref | Done |
|---|------|-----|------|
| 1.6 | Refactor `_QueryFilter.matches` → predicate dispatch table (`_Predicate` Protocol + frozen dataclass per axis). Makes `matches()` a one-liner; each predicate independently testable with `hypothesis`. | pre.md 1.2 | ☑ |
| 1.7 | Extract `_relax` → `_step_with_tolerance`, `_measure_change`, `_check_convergence` (each <20 LOC, pure). | pre.md 1.3 | ☑ |
| 1.8 | Extract `_apply_hebbian_updates` → `_propagate_errors_backward`, `_compute_weight_updates`, `_apply_weight_updates` (each <25 LOC). | pre.md 1.3 | ☑ |
| 1.9 | Convert `EquiTile._get_activation` (5-way) and `train_step` (3-way on closed `Literal`) to `match`/`case`. | pre.md 1.4 | ☑ |
| 1.10 | Add **snapshot tests** for extracted helpers: deterministic seed → fixed tensor output. Guard all future refactors. | pre.md Snapshot Tests | ☐ |

### Type System Quick Wins (Ride Along)

| # | Task | Ref | Done |
|---|------|-----|------|
| 1.11 | `credit_assignment_type: str` → `Literal["gradient","equilibrium","hebbian","target","forward-only","spiking"]` (+ `"backpropagation"`, `"local"` seen in the wild) | pre.md 3.2 | ☑ |
| 1.12 | `**kwargs: Any` → `**kwargs: object` in 6 factory functions | pre.md 3.1 | ☑ |
| 1.13 | `frozen=True, slots=True` on `TrainingMetrics`, `LazyStats`, `TileTask` | pre.md 3.4 | ☑ |
| 1.14 | `builtins.list` → `list` in 6 annotations (`registry.py`) | pre.md 3.5 | ☑ |

### Sprint 1 Gate (Must Pass Before Any Other Work)
```bash
uv run ruff format --check . && uv run ruff check . && uv run pyright . && uv run pytest --cov -x --tb=short
```
- Zero ruff violations — **NOT met** (pre-existing ~2521 project-wide; this session kept it at +4 net, all plan-mandated TRY003)
- Zero pyright errors — **met** (0 errors; 2290 warnings pre-existing, relaxed in pyproject)
- Coverage ≥ baseline — **NOT met** (whole-repo coverage 16.82% vs required 50%; unit+property ~763 tests pass)
- All tests pass in **<60s on CPU** (no GPU, no downloads) — **met** (~23s for unit+property)
- Remaining gate cleanup (whole-repo ruff + coverage) is a Sprint 1.5 effort, tracked below.

---

## Sprint 2: Fast Parity & Registry as Unit Tests (Week 2-3)
*Parity suite = fast unit tests with synthetic data. Registry audit = fast instantiation checks.*

### Parity Unit Test Suite (Synthetic, 1-2 Epochs Max)

| # | Task | Ref | Done |
|---|------|-----|------|
| 2.1 | Create `tests/unit/validation/test_backprop_parity.py` — **not** a benchmark runner, a `pytest` module | RESEARCH.md 0.1 | ☐ |
| 2.2 | Synthetic classification data fixture (64-dim, 10 classes, 200 samples) — zero I/O | RESEARCH.md 0.1 | ☐ |
| 2.3 | Parameterized test: each bio-plausible model vs backprop on identical MLP (1 hidden layer) | RESEARCH.md 0.1 | ☐ |
| 2.4 | Metric assertions: accuracy within 5% of backprop, FLOPs tracked, memory tracked, deterministic seeds | RESEARCH.md 0.1 | ☐ |
| 2.5 | Target models: `eqprop_mlp`, `directed_ep`, `standard_fa`, `forward_forward`, `pepita`, `equitile` (ConvEquiTile on 32×32 synthetic) | RESEARCH.md 0.1 | ☐ |
| 2.6 | Run time: **<30s total** for full parity suite (1 epoch, batch=32, synthetic) | — | ☐ |

### Registry Audit Unit Test

| # | Task | Ref | Done |
|---|------|-----|------|
| 2.7 | Create `tests/unit/validation/test_registry_audit.py` | RESEARCH.md 0.2 | ☐ |
| 2.8 | Test: every registered component instantiates, runs `forward()` on dummy tensor, metadata fields match implementation | RESEARCH.md 0.2 | ☐ |
| 2.9 | Test: deterministic output with fixed seed for all 80+ components | RESEARCH.md 0.2 | ☐ |
| 2.10 | Run time: **<15s total** | — | ☐ |

### Reproducibility Unit Test

| # | Task | Ref | Done |
|---|------|-----|------|
| 2.11 | Create `tests/unit/validation/test_reproducibility.py` | RESEARCH.md 0.3 | ☐ |
| 2.12 | Test: fixed seed → identical model weights, identical loss trajectory (5 steps) | RESEARCH.md 0.3 | ☐ |
| 2.13 | Test: environment capture (git commit, torch version, deps hash) serializes correctly | RESEARCH.md 0.3 | ☐ |

### Sprint 2 Gate
```bash
uv run pytest tests/unit/validation/ -x --tb=short
```
- All parity tests pass (accuracy within 5% of backprop on synthetic)
- All registry tests pass (100% components instantiable + deterministic)
- All reproducibility tests pass
- **Full validation suite <45s on CPU**

---

## Sprint 3: Property-Based Tests for Core Logic (Week 3)
*Replace manual test cases with `hypothesis` strategies for pure functions. This IS the viability proof.*

### Property Test Targets (Pure Functions Only)

| # | Target | Properties to Verify | Ref |
|---|--------|---------------------|-----|
| 3.1 | `_QueryFilter` predicates | `matches(meta)` ↔ predicate logic equivalence; commutativity of filter composition | pre.md 1.2 |
| 3.2 | `core.config.resolve_hidden_dims` / `compute_hidden_dims` | Idempotence, monotonicity in `num_layers`, `hidden_dim=None` → `[]` | pre.md A1 |
| 3.3 | `acceleration.kernels` (matmul, transpose, outer product) | Numerical equivalence to PyTorch reference; shape invariants | pre.md 6.4 |
| 3.4 | `Registry.register` + `Registry.get` round-trip | `get(register(x)) == x`; metadata preserved; name collision handling | pre.md A9 |
| 3.5 | `knowledge.kb.KnowledgeEntry` serialization | `from_dict(to_dict(entry)) == entry`; vector embedding determinism | pre.md A12 |
| 3.6 | `equitile.core.config.EquiTileConfig.validate()` | Invalid configs raise; valid configs don't; field bounds respected | pre.md A11 |
| 3.7 | `domains.base.DomainSpec` / `Batch` / `Metrics` | Round-trip serialization; `Batch.to(device)` preserves metadata | pre.md A2 |

### Sprint 3 Gate
```bash
uv run pytest tests/property/ -x --tb=short
```
- All property tests pass (1000+ examples each via `hypothesis`)
- No flaky tests (deterministic seeds)
- **Property suite <30s on CPU**

---

## Sprint 4: Test Infrastructure & CI (Week 3-4)
*Make the test suite the single source of truth for viability.*

### CI Pipeline Hardening

| # | Task | Done |
|---|------|------|
| 4.1 | `.github/workflows/ci.yml`: `ruff format --check` → `ruff check` → `pyright` → `pytest --cov --maxfail=5` (unit + property only) | ☐ |
| 4.2 | Coverage floor: `--cov-fail-under=50` (per `pyproject.toml`), trending to 85% | ☐ |
| 4.3 | Separate `slow` mark for integration tests (excluded from default CI) | ☐ |
| 4.4 | Nightly workflow: runs `tests/slow/` (real data, full epochs) — results to artifact store, not gate | ☐ |

### Test Organization Cleanup

| # | Task | Done |
|---|------|------|
| 4.5 | Move all real-data/download tests to `tests/slow/` (currently mixed in `integration/`) | ☐ |
| 4.6 | Ensure `tests/unit/` has zero I/O, zero GPU, zero network | ☐ |
| 4.7 | Ensure `tests/property/` uses only `hypothesis` strategies, no fixtures with side effects | ☐ |
| 4.8 | Add `conftest.py` synthetic fixtures: `synthetic_batch`, `synthetic_vision_task`, `synthetic_lm_task` | ☐ |

### Sprint 4 Gate
```bash
uv run pytest tests/unit/ tests/property/ --maxfail=1 -q
```
- **<45s on CPU** (CI runner)
- Coverage ≥ 50%
- Zero flakes in 5 consecutive runs

---

## What Is Explicitly NOT In This Plan

| Deferred | Reason |
|----------|--------|
| Full training experiments (Sprints 3-5 of old plan) | Code still changing; experiments belong in RESEARCH.md Phase 1+ after foundation solid |
| CLI (`biopl-scientist --demo`, `biopl-parity`, etc.) | Passing unit tests = viability proof; CLI is polish |
| Colab notebooks / leaderboard / failure gallery | Recruitment artifacts; build after test suite is bulletproof |
| AutoScientist LLM integration / campaign persistence | Requires stable execution engine; Sprint 1-4 stabilize the engine |
| Cross-domain benchmarks (LM, RL, Graph, TimeSeries) | Need stable domain tasks first; domain tasks need stable registry |
| Neuromorphic / distributed / P2P | Explicitly deferred in RESEARCH.md |
| Config unification (A1), TaskProtocol (A2), PersistenceIndex (A3) | Appendix A items — fold into Sprint 1-4 when touching those files |

---

## Viability Proof = Passing Test Suite

| Audience | What They See |
|----------|---------------|
| **Developer** | `git clone && uv sync && uv run pytest` → green in <60s, no setup |
| **Researcher** | `tests/unit/validation/test_backprop_parity.py` — bio-plausible models within 5% of backprop on synthetic data, deterministic, no GPU needed |
| **Contributor** | Clear test patterns: unit (fast, isolated), property (exhaustive, pure), slow (real data, opt-in) |

---

## Success Metrics (End of Sprint 4)

| Metric | Target |
|--------|--------|
| Unit + property test time (CPU) | <60s |
| Ruff violations | 0 |
| Pyright errors | 0 |
| Coverage (unit + property) | ≥50% |
| Parity accuracy (synthetic, 1 epoch) | Bio-plausible within 5% of backprop |
| Registry instantiation | 100% of 80+ components |
| Determinism | 100% components reproducible |
| Flaky tests | 0 in 5 consecutive runs |

---

## After Sprint 4: RESEARCH.md Phase 0 Complete

Only then consider:
1. **Phase 1** — Full experiments (GPU, real data, multi-epoch) via `tests/slow/`
2. **Phase 4** — AutoScientist (stable execution engine + KB)
3. **Adoption** — CLI, Colabs, leaderboards

**The test suite is the product until Sprint 4 gates pass.**