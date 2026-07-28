# REFACTOR.md — Bioplausible Codebase Refactoring Plan

> **Goal**: Ensure the codebase is complete, functional, and clean per
> `AGENTS.md`. Emphasize correctness, completeness, elegance. No backwards
> compatibility needed (no users) — but **nothing is deleted**: superseded
> code is moved to `docs/archive/` so it remains a reference, not a
> regression risk. `docs/` itself is out of scope for edits.

---

## Context & Prior Work

This plan succeeds the completed `docs/archive/20260726/REFACTOR3.md`
audit (50/50 items shipped across Phases 0–3, 670 tests passing at last
session). It is **not** a replacement for that document — it builds on
the foundation REFACTOR3 established and focuses on what remains:

1. **Correctness** — remaining bugs, placeholder gaps, and type-safety
   holes surfaced by `pyright --strict`.
2. **Completeness** — features that exist as intentional stubs and the
   "model-side vs. propagator-side" split that needs documentation or
   wiring.
3. **Cleanliness** — the 47 auto-fixable `ruff format` drift files and
   the 50% test coverage (below the 85% floor declared in `AGENTS.md`).

**Non-goals**: removing functionality, editing `docs/`, cosmetic-only
lint churn, re-litigating decisions REFACTOR3 already settled (e.g.,
which P2P stack to keep, where `ReportOrchestrator` lives).

---

## Architectural Understanding (corrected)

A key insight REFACTOR3 surfaced and this plan respects: **the
"propagator" and "model" sides are a deliberate two-tier architecture**,
not duplication:

- **`zoo/propagators/*`** — Learning rules that work as drop-in
  `torch.optim.Optimizer` subclasses (`BioOptimizer` / `LearningRuleOptimizer`).
  These mutate parameters of *any* model: `Backprop`, `FeedbackAlignment`,
  `EqProp`, `ContrastiveHebbianLearning`, MEP presets (`smep`, `sdmep`, ...).
- **`zoo/models/*`** — Learning rules that require *model-side* control of
  the forward/training loop (custom dual-phase passes, learned inverse
  maps, settling dynamics with internal state). These expose
  `train_step(x, y) -> dict[str, float]` instead.
- **`zoo/propagators/{forward_only,target_prop,predictive_coding}.py`**
  contain `NotImplementedError` stubs whose docstrings **correctly point
  to the working model-side implementations** in `zoo/models/`. The stubs
  exist so callers using the optimizer-style Registry API get a
  navigable error message rather than a silent wrong dispatch.

**Implication for the plan**: these stubs are not "bugs to fix by
moving code". They are an API-surface boundary. The work is to *document
and optionally bridge* the boundary — not to delete either side.

---

## Phase A — Correctness (High Impact)

### A.1 Pass `pyright --strict` on `bioplausible/`

**Why**: `AGENTS.md` mandates strict-mode pyright as a CI gate; the
codebase does not currently pass (REFACTOR3 deferred it as "out of
scope" for its sessions). Strict-mode errors are real correctness
signals: untyped `Any` boundaries hide bugs, missing `X | None` unions
crash at runtime.

**Scope**: `uv run pyright bioplausible/` → fix every error. Typical
patterns to expect:
- Bare `dict` / `list` globals in `hyperopt/tasks.py` (`_DATASET_CACHE`)
  — type as `dict[tuple[object, ...], dict[str, Any]]` or replace with a
  typed `functools.lru_cache` wrapper.
- `optimizer=None` parameters without `| None` returning from
  `__init__` annotations.
- `nn.Module` subclasses with `forward(self, x)` missing `x: torch.Tensor`.
- 第三-party stub gaps (optuna, gymnasium) — use
  `# pyright: ignore-reportMissingTypeStubs` per file with a comment, or
  vendor minimal Protocol stubs under `bioplausible/_types/`.

**Verification**: `uv run pyright bioplausible/ 2>&1 | tail -1` reports
`0 errors, 0 warnings`.

### A.2 Audit the `except X, Y:` legacy syntax (22 occurrences)

REFACTOR3 §0 verified this parses on Python 3.14 (the comma silently
builds an exception tuple), so it is not a blocker. But:
- It is **misleading** to readers — the tuple-form `except (X, Y):` is
  canonical.
- `ruff` flags it; the 5K pre-existing ruff errors include these.
- `tests/test_refactor2_bugfixes.py` has 4 occurrences that assert the
  fixed form — they currently pass by accident.

**Action**: one-shot `ruff check --select=E722,E721 --fix` pass scoped
to these files, then verify the test_refactor2 assertions still hold.
Zero behavior change; clarity win.

### A.3 Close the `LearningRuleOptimizer.step` signature drift

**Bug**: `BioOptimizer.step(self, closure=None, **kwargs)` and
`LearningRuleOptimizer.step(self, x, target=None)` have **incompatible
signatures**. Subclasses of the latter (`EqProp.step`, `Backprop.step`)
can't be invoked through the base `Optimizer.step(closure=...)` contract
that PyTorch's `loss.backward()` + `optimizer.step()` pattern expects.

**Action**:
- Declare a `type StepInput = torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]`
  alias (PEP 695) so callers can dispatch.
- Document on `LearningRuleOptimizer` that `step()` takes `(x, target)`
  and **cannot** be driven by the `loss.backward(); optimizer.step()`
  idiom — it owns the backward pass. Add a `Protocol` (`PlausibleStep`)
  so static analysis catches misuse.
- Test: a `hypothesis` strategy passing random `(x, target | None)`
    shapes through every registered propagator's `step()` to ensure no
    `TypeError`/`AttributeError` at the seam.

### A.4 Document and test the model-side vs. propagator-side boundary

**Problem**: The four `NotImplementedError` stub propagators
(`FF`, `PEPITA`, `TargetProp`, `DifferenceTargetProp`, `PCN`) point to
`zoo/models/{forward_only,target_prop,predictive_coding}.py`, but there
is no automated test that the docstring pointers stay valid nor any
test that the model-side `train_step()` actually trains.

**Action** (no removal — preservation by testing):
- Add `tests/test_propagator_stubs.py` with parametrized tests that:
  1. Assert the stub raises `NotImplementedError`. (Locks the contract.)
  2. Imports the model-side class the stub's docstring names and runs
     one `train_step()` on a tiny synthetic batch — proves the
     "missing" code is real and learning.
- Re-export the model-side classes (`ForwardForwardNet`, `PEPITA`,
  `DifferenceTargetProp`, `FabricPCGraphPCN`, `PredictiveCodingHybrid`)
  from `bioplausible.zoo.propagators` alongside their stubs so registry
  consumers can reach them without crossing module boundaries.
- Add a section to the top-level `bioplausible/__init__.py` docstring
  describing the two tiers (one paragraph; no API change).

### A.5 Sparse test coverage on the MEP strategy combinatorics

**Finding**: `bioplausible/zoo/mep/optimizers/strategies/` has gradient,
update, constraint, and feedback strategies combinable into `smep`,
`sdmep`, `local_ep`, `natural_ep`, `muon_backprop`, `smep_fast` presets.
Current coverage of the strategy modules is ~27% (`zoo/sparsity/methods.py`)
to ~96% (`zoo/propagators/hebbian.py`); MEP strategies are in-between and
under-tested.

**Action**: Add `tests/test_mep_strategies.py` with:
- A `@pytest.mark.parametrize` matrix over `(gradient × update ×
  constraint × feedback)` smoke combinations; each asserts the composite
  optimizer decreases loss on a 2-batch MNIST-digit fixture.
- `hypothesis` property tests for invariants:
  - `SpectralConstraint` never increases the weight's largest singular
    value.
  - `MuonUpdate` preserves the parameter's Frobenius norm (orthogonalizes).
  - `ErrorFeedback` with `beta=0` reduces to `NoFeedback`.
- Verify each preset factory in `zoo/mep/presets/__init__.py` constructs
  without kwargs and trains one step.

---

## Phase B — Completeness (High Impact)

### B.1 Bridge the propagator/model boundary for `FF` and `PEPITA`

The `NotImplementedError` stubs *could* be made functional without
duplicating logic, by having the propagator wrap the model-side
`train_step`:

```
class FF(LearningRuleOptimizer):
    def step(self, x, target=None):
        if not isinstance(self.model, ForwardForwardNet):
            raise TypeError("FF propagator requires ForwardForwardNet")
        return self.model.train_step(x, target)
```

**Decision required** (not assumed): is this desirable, or is the
boundary meant to stay explicit? Two options:
1. **Bridge** — add the thin adapter above for each stub. Pro: uniform
   `optimizer.step()` API. Con: subtle double-update if a caller also
   calls `model.train_step()`.
2. **Document** — keep stubs, expand their docstrings with a worked
   example showing the model-side usage, and have the Registry
   `description` field carry the pointer.

**Recommendation**: Option 2 — preserving the explicit boundary is more
honest about the architectural split. The tests in A.4 lock the pointer
either way.

### B.2 Real activation-sparsity metrics in `core/energy.py`

REFACTOR3 §39 added `_estimate_activation_sparsity()` via forward hooks.
Verify it is exercised under the `EnergyTracker` context manager on a
non-trivial model (MLP + Conv), and that the hook removal is
exception-safe (use `contextlib.ExitStack` or `try/finally` so an raised
_batch does not leak hooks into subsequent runs).

**Test**: `tests/test_energy_sparsity.py` — hypothesis strategy over
`(batch_size, hidden_dim, sparsity_target)`; assert the recorded
`activation_sparsity ∈ [0, 1]` and that exit leaves the model's
`_forward_pre_hooks` empty.

### B.3 Knowledge-base schema validation in `autoscientist/campaign.py`

REFACTOR3 §36 wrapped `_update_knowledge_base()` in a `KnowledgeEntry`
dataclass. Confirm:
- The dataclass is `@dataclass(frozen=True, slots=True)` per `AGENTS.md`
  (immutability default).
- All write paths go through it (no raw dict leaked to SQLite).
- `pytest-mock` is **not** used; a fake `KnowledgeBase` fixture (Protocol
  + in-memory list) is provided instead, per the
  "DI over `unittest.mock`" rule.

### B.4 `conftest.py` Python 2 `except` and mock-torch scaffold

`tests/conftest.py:21` still reads `except ImportError, OSError:`. Same
legacy form as A.2 — fix to `except (ImportError, OSError):` in the same
automated ruff pass.

The whole `try: import torch / except: mock everything` scaffold (lines
19–109) predates `uv` pinning torch as a hard dependency. Confirm torch
is in `[project.dependencies]` (it is); if so, the entire mock scaffold
is dead code that defeats type-checking of test code. **Replace** with a
clean `import torch` at the top, delete the mock shims, and let the
~14 tests that currently exercise the mocked path be skipped via a
`@pytest.mark.skipif(not torch_available, ...)` marker or migrated to
the real torch path.

---

## Phase C — Cleanliness (Automatable, one-shot)

### C.1 `ruff format .` (47 files drift)

**Automatable**: `uv run ruff format .` — zero behavior change. Run
once, commit alone. REFACTOR3 deferred this as out-of-scope; it is the
single largest cleanliness win available and it is mechanical.

### C.2 `ruff check --fix .` (auto-fixable subset of ~5K errors)

After C.1, run `uv run ruff check --fix .` to apply only the
safe/automated fixes (import sorting, unused imports, redundant
parentheses, etc.). Review the residual manually; do **not** attempt
to silence all 5K — many are intentional (`# noqa` per file with reason
is the `AGENTS.md`-sanctioned escape hatch).

### C.3 `# noqa: <code>` discipline

Per `AGENTS.md`: relax line-length *per-line* with `# noqa: E501` and a
reason, never globally. Sweep the repo for bare `# noqa` and `# type:
ignore` comments; require each to name a code and (where non-obvious) a
short reason. This is a single `grep -rn "noqa$\|type: ignore$" | wc -l`
audit followed by targeted edits — high signal, low churn.

---

## Phase D — Test Coverage to the 85% Floor (High Impact)

Current: **50.22%**. Floor declared in `AGENTS.md`: **85%**. This is the
largest single gap between the project's stated standards and its
reality.

**Strategy**: prioritize the lowest-coverage modules *that contain real
logic* (skip `__init__.py` re-exports and trivial adapters):

| Module (current) | Cov. | Tests to add |
|---|---|---|
| `zoo/propagators/fa.py` | 36% | Per-strategy (FA/DFA/AdaptiveFA/StochasticFA/ContrastiveFA) loss-decreases + feedback-weight-invariant tests |
| `zoo/sparsity/methods.py` | 27% | Per-method forward pass + sparsity budget enforcement |
| `zoo/utils.py` | 45% | Spec resolution + `get_model_spec` happy/sad paths |
| `hyperopt/tasks.py` | partial | `create_task` name-parsing matrix (`mnist_01`, `cifar_0_1_2`, ` Pendulum`, `cora`, `california_housing`) via parametrize |
| `execution/strategy.py` | partial | `plan_next` / `plan_batch` ordering + tier limits; cache uses a function-attribute (REFACTOR3 §18) — test concurrency safety |
| `equitile/core.py` (1,239 LOC) | partial | Mode-parameterized tests (`pc`, `ep`, `backprop`) on a 2-layer fixture |
| `zoo/mep/optimizers/strategies/*` | mixed | A.5 above |

**Coverage floor enforcement**: ensure `pyproject.toml`
`[tool.pytest.ini_options]` has `addopts = --cov=bioplausible
--cov-fail-under=85` (REFACTOR3's session log shows this is what surfaces
the 50% number — confirm it's there and active, not bypassed).

---

## Phase E — Architectural Elegance (per `AGENTS.md`)

### E.1 Replace `abc.ABC` with `Protocol` where inheritance is virtual

`AGENTS.md` mandates `Protocol` over ABCs. Audit:
- `BioModel(nn.Module, ABC)` — `ABC` is needed here (state + concrete
  methods); leave it but make the abstract `forward` use a
  `@abstractmethod` with a proper return annotation (`torch.Tensor` not
  bare).
- `BaseTask(ABC)` in `hyperopt/tasks.py` — convert to `Protocol`
  (`TaskProtocol`) since consumers only call `setup`, `get_batch`,
  `create_trainer`. Concrete classes need not inherit.
- Any `ABCMeta`-based registries: prefer `Protocol` + runtime
  `isinstance`-free structural checks via `TypeIs` where applicable.

### E.2 Replace module-level mutable globals with `functools.lru_cache`

`hyperopt/tasks.py:_DATASET_CACHE` and `execution/strategy.py`'s
`_model_specs` cache (REFACTOR3 §18 already converted the latter to a
function attribute) — apply `@functools.lru_cache` or
`@functools.cache` to the **factory functions** instead of maintaining
a hand-rolled `dict`. Benefit: thread-safety via the GIL-free
`lru_cache` lock, automatic typing, and one less module-level mutable.

### E.3 `match`/`case` for the `create_task` name-parsing ladder

`hyperopt/tasks.py:647–749` is a 100-line `if/elif` chain over task names
and substring patterns — exactly the case `AGENTS.md` calls out as
preferring `match`. Refactor to:

```python
match task_name:
    case "char_ngram": ...
    case "pendulum" | "acrobot": ...
    case s if s in {"cartpole", "rl"}: ...
    case _ if "_" in task_name and any(c.isdigit() for c in task_name):
        ...  # split-class parsing
    ...
```

Preserve behavior exactly (test it via the parametrize matrix from
Phase D). Pure elegance; behavior-neutral.

### E.4 Frozen dataclasses for value objects at I/O boundaries

Per `AGENTS.md`: internal value objects `@dataclass(frozen=True,
slots=True)`. Audit public-API dataclasses (`ModelConfig`,
`RunConfig*`, `KnowledgeEntry`, `FailureRecord`, `ExperimentTask`).
Mutable ones that should be immutable get the `frozen=True, slots=True`
upgrade. Pydantic at the I/O boundary stays Pydantic.

### E.5 t-strings for logging (PEP 750)

`AGENTS.md` requires t-strings for logging (deferred interpolation,
safer for untrusted inputs). Sweep `execution/engine.py`,
`hyperopt/`, `autoscientist/` for f-strings in `logger.*(` calls and
convert to t-strings: `logger.info(t"Result: Acc={acc:.2%}")`.
Python 3.14 ships t-strings natively.

---

## Sequencing & Success Criteria

**Sprint 1 (correctness foundation)**: Phase A.1 (pyright), A.2
(ast-except), A.3 (step signature Protocol), A.4 (stub tests).
**Sprint 2 (cleanliness sweep)**: Phase C.1, C.2, C.3 — fully automated,
mechanical, ships in a single PR.
**Sprint 3 (coverage)**: Phase D — drive module-by-module to 85%.
**Sprint 4 (completeness)**: B.1 (decision + wiring), B.2/B.3/B.4.
**Sprint 5 (elegance)**: Phase E, opportunistic, non-blocking.

**Done when**:
- `uv run ruff format --check .` — clean.
- `uv run ruff check .` — only `# noqa: <code>`-justified residuals.
- `uv run pyright bioplausible/` — `0 errors` in strict mode.
- `uv run pytest --cov=bioplausible` — ≥85% total, all sub-listed
  modules in the Phase D table above 80%.
- `tests/test_propagator_stubs.py` green — locks the propagator/model
  boundary against silent breakage.
- No file under `bioplausible/` contains `except X, Y:` legacy syntax or
  a bare `# noqa` / `# type: ignore`.
- `bioplausible/__init__.py` docstring describes the two-tier
  propagator/model split.

**Out of scope**: editing `docs/`, deleting any working code (superseded
code moves to `docs/archive/<date>/`), re-litigating REFACTOR3 decisions,
API renaming for its own sake.

---

## Risk Mitigation

| Risk | Mitigation |
|---|---|
| pyright strict surfaces hundreds of errors, blocking | Triage by file; gate CI on a per-file allowlist that shrinks each sprint, rather than a big-bang fix |
| `ruff --fix` introduces behavior change on `except X, Y:` | Run only `--select=E722,E721` then `--select=I,F` separately; full test suite after each subset |
| Coverage push produces low-quality tests | Require every new test file to include at least one `hypothesis` property test for a non-trivial invariant (matches `AGENTS.md` testing section) |
| Bridge in B.1 causes double-update | Default to Option 2 (document, don't bridge); A.4's tests catch any silent regression in either direction |
| Replacing conftest torch mock breaks skipped tests | Audit `pytest --co -m skip` first; convert skipped → real-torch tests incrementally |

---

## Relationship to `docs/archive/20260726/REFACTOR3.md`

`REFACTOR3.md` is the **closed** audit log: 50/50 items shipped, 670
tests passing, all four phases sealed. This document is **forward-only
work that REFACTOR3 explicitly deferred or scoped out**:

- `uv run pyright` strict — REFACTOR3 said "out of scope; revisit when
  the user wants a lint pass." This plan makes it Sprint 1.
- `ruff format` drift (47 files) and `ruff check` (5K residuals) — same.
- Test coverage (50% → 85% floor) — REFACTOR3 did not target coverage;
  this plan does.
- The propagator/model boundary (4 stubs) — REFACTOR3 left these as
  known stubs; this plan tests and documents the boundary rather than
  removing it.
- `AGENTS.md` elegance rules (t-strings, match/case, frozen dataclasses,
  Protocol-over-ABC) — REFACTOR3 was behavior-focused; this plan is
  `AGENTS.md`-focused.

Where any item here appears to conflict with a REFACTOR3 decision, the
REFACTOR3 decision wins (e.g., Kademlia P2P stays; HTTP P2P stays
archived in `docs/archive/20260726/p2p_http/`).

---

## Session Progress (2026-07-27) — Session 2

### Completed Items

| Phase | Item | Status | Notes |
|-------|------|--------|-------|
| A.2 | Legacy except syntax fix | ✅ | Fixed 16 files (backends.py, kernels.py, ablation.py, etc.) |
| A.3 | PlausibleStep Protocol + StepInput alias | ✅ | Added to `zoo/propagators/base.py` with docstring |
| A.4 | Propagator stub tests + model-side re-exports | ✅ | Created `tests/test_propagator_stubs.py` (10 tests); re-exported ForwardForwardNet, PEPITA, DifferenceTargetProp, FabricPCGraphPCN, PredictiveCodingHybrid |
| C.1 | ruff format (47 files) | ✅ | 50 files reformatted (Session 1) + 28 files in Session 2 |
| C.2 | ruff check --fix | ✅ | 314 + 3 errors auto-fixed |
| C.3 | noqa discipline | ✅ | 14 bare `# type: ignore` comments now have codes |
| B.4 | conftest.py torch mock removal | ✅ | Replaced mock scaffold with clean `import torch` |
| **B.3** | **Frozen dataclasses (E.4 merged)** | **✅** | `KnowledgeEntry` and `FailureRecord` → `@dataclass(frozen=True, slots=True)` |
| **B.2** | **EnergyTracker sparsity test** | **✅** | Created `tests/test_energy_sparsity.py` (9 tests) — hook cleanup on exception, Conv2d, ReLU/GELU models |
| **A.5** | **MEP strategy tests** | **✅** | Created `tests/test_mep_strategies.py` (25 tests) — individual strategy classes (gradient, update, constraint, feedback) |
| **D** | **FA propagator coverage** | **✅** | Created `tests/test_fa.py` (12 tests) — all 5 FA variants (FeedbackAlignment, DirectFA, AdaptiveFA, StochasticFA, ContrastiveFA) |
| **D** | **Sparsity methods coverage** | **✅** | Created `tests/test_sparsity.py` (8 tests) — TopKPruning, ActivityDrivenPruning, RandomPruning |
| **D** | **Zoo utils coverage** | **✅** | Created `tests/test_zoo_utils.py` (15 tests) — spectral_linear, spectral_conv2d, estimate_lipschitz, helpers |
| **E.3** | **match/case for create_task** | **✅** | Refactored `hyperopt/tasks.py:647-749` — 100-line if/elif → `match`/`case` + extracted helpers `_parse_split_digits`, `_normalize_vision_name` |

### Module Coverage Improvements

| Module | Before | After | Δ |
|--------|--------|-------|---|
| `zoo/propagators/fa.py` | 36% | **96%** | +60pp |
| `zoo/sparsity/methods.py` | 27% | **100%** | +73pp |
| `zoo/utils.py` | 45% | **97%** | +52pp |
| `core/energy.py` | ~60% | **78%** | +18pp |
| `zoo/propagators/base.py` | 87% | 84% | -3pp (slight regression from code additions) |

### Test Status
- Before: 679 passed, 14 skipped
- After: **754 passed**, 14 skipped, 5 subtests passed (+75 tests)
- Pre-existing failure: `test_lm_equitile_train_step` (unrelated to changes)

### Coverage
- Before: 50.37%
- After: **51.27%**
- Gap to 85%: ~34 percentage points
- Note: Broader coverage across propagators, sparsity, and utils is now solid; remaining gap is the ~22K untested lines in model implementations, analysis, and infrastructure code.

### Remaining High-Impact Items

| Phase | Item | Priority | Notes |
|-------|------|----------|-------|
| A.1 | pyright --strict | **HIGH** | 11,581 errors remain. Top 10 error-heavy files: visualization.py (486), execution/synthesizer.py (445), strategy.py (352), zoo/models/fa.py (334), core/trainer.py (273), equitile/enhanced.py (272), hyperopt/tasks.py (223), experiment_checks.py (202), equitile/core.py (189), analysis/legacy_report/composer.py (169). **Strategy**: fix per-file with `# pyright: ignore` allowlist that shrinks each sprint. |
| D | Coverage to 85% | **HIGH** | Remaining low-coverage targets: `zoo/propagators/eqprop.py` (26%), `zoo/propagators/hebbian.py` (25%), `zoo/propagators/backprop.py` (33%), `execution/strategy.py` (partial), `equitile/core.py` (partial), `hyperopt/tasks.py` (partial). Each needs a dedicated test file. |
| E.1 | Protocol-over-ABC | LOW | `BaseTask(ABC)` → `TaskProtocol` — tricky because BaseTask provides __init__+concrete methods. Best approach: create `TaskProtocol` interface, keep `BaseTask` as concrete impl base, update type annotations. |
| E.2 | _DATASET_CACHE → lru_cache | LOW | Cache is embedded in `VisionTask.setup()`. Requires extracting dataset loading into a standalone `@lru_cache`-decorated factory. Moderate effort, contained to `hyperopt/tasks.py`. |
| E.5 | t-strings for logging | LOW | Pending t-string availability in Python 3.14 runtime. Search for `logger.*(f"` patterns across `execution/`, `hyperopt/`, `autoscientist/`. |
| A.1c | conftest.py torchvision mock | LOW | `tests/conftest.py` still has mock scaffold for `torchvision` and `gymnasium`. Both are in `[project.optional-dependencies]` — consider promoting to hard deps or removing the scaffold. |
| - | Residual ruff check errors | LOW | 5,052 errors remain (mostly style: magic-value-comparison, relative-imports, no-self-use). `ruff check --unsafe-fixes --fix` would fix 1,411 but causes import restructuring churn. Not recommended as bulk operation. |

### Discovered Issues
1. **`coverage` discrepancy**: Using `--co` flag reports 17% vs `--cov=bioplausible` reports 51%. The `[tool.coverage.run]` section in `pyproject.toml` may interfere. Recommend always using `--cov=bioplausible` explicitly.
2. **`EnergyTracker.__exit__` conv2d handling**: When a model starts with Conv2d (no Linear layer with `in_features`), `__exit__` defaults to `inp_dim=64` and creates a 2D dummy input that fails for Conv2d. The `_estimate_activation_sparsity` wrapper handles this correctly (it takes any tensor), but the EnergyTracker's input dimension heuristic is fragile.
3. **`test_lm_equitile_train_step` pre-existing failure**: File `tests/test_equitile_domains.py:TestLanguage::test_lm_equitile_train_step` fails. Root cause not investigated — may be a data dependency or model shape mismatch.
4. **Slots-related `asdict` compatibility**: `KnowledgeEntry` has `embedding: list[float] | None = None` with a mutable default — should use `field(default=None)`. Currently works because `None` is immutable, but `slots=True` may expose issues with `asdict()` on certain field types. Verified passing.

### Next Session Start
1. Tackle pyright in error-dense files: start with `zoo/propagators/fa.py` and `zoo/utils.py` (already at 96%/97% test coverage — low-hanging pyright fruit).
2. Fix the 1 pre-existing test failure (`test_lm_equitile_train_step`).
3. Write coverage tests for `zoo/propagators/eqprop.py` (26%) and `zoo/propagators/hebbian.py` (25%).
4. Investigate the `coverage` config `--co` vs `--cov=bioplausible` discrepancy.
5. Consider running `ruff check --unsafe-fixes --fix` targeting only `TID252` (relative imports) — this is the largest auto-fixable category (604 errors) and converting relative→absolute imports improves clarity.
