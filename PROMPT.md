# PROMPT.md — Curated Menu of Reusable Prompts

> Synthesized from 263+ sessions. Each entry is a generalized, variable-parameterized "ultimate" prompt. Replace `<VARIABLES>` with project-specific values.

---

## Table of Contents

1. [Continuation / Refinement](#1-continuation--refinement)
2. [Plan Creation](#2-plan-creation)
3. [Codebase Exploration](#3-codebase-exploration)
4. [Bug Hunting & Code Quality Audit](#4-bug-hunting--code-quality-audit)
5. [Mechanical Code-Fix (Lint / Pattern Replacement)](#5-mechanical-code-fix)
6. [Test Optimization](#6-test-optimization)
7. [Test Creation / Coverage](#7-test-creation--coverage)
8. [Code Refactoring](#8-code-refactoring)
9. [Validation & Quality-Gate](#9-validation--quality-gate)
10. [Documentation](#10-documentation)

---

## 1. Continuation / Refinement

**Use when:** Resuming a multi-session implementation. The agent needs context on what's done, what remains, and quality guardrails.

```
Continue implementing the `@<PLAN_FILE>` plan. Before wrapping-up:

1. **Detail progress** directly in `<PLAN_FILE>` — update status tables, mark completed items, note new discoveries.
2. **New issues or work** — add any newly-discovered issues to the plan file, not only in chat.
3. **Remaining-work hints** — include notes for the next session: which files to read next, which patterns to watch for, which tests are fragile.
4. **Prioritize** significant architectural work ahead of cosmetic lint.
5. **No shortcuts** — if something was deferred, mark it as deferred with a reason; don't silently skip it.
6. **Test consideration** — before running tests, profile for optimization opportunities (reduce iterations, model sizes, batch sizes).
7. **If unsure** — stop and ask for clarification rather than guessing.
```

**Optional addenda** (append as needed):
- `Pay attention to deleted files — ensure no stale imports remain.`
- `Verify re-export shims are absent; <GUIDELINES_FILE> stipulates no backwards compatibility.`
- `Don't start <NEXT_PLAN_FILE> until <PLAN_FILE> is complete.`
- `Don't begin implementing — we're still developing the plan.`

---

## 2. Plan Creation

**Use when:** Kicking off a new refactoring/development phase. Produces a living plan document to be iteratively refined before execution.

```
Develop `<PLAN_FILE>`, our next refactoring plan.

**Background:** This plan builds on <PRIOR_PLAN_FILE(S)> (summarize what was already completed). What remains is: <SCOPE_DESCRIPTION — e.g., "architectural: duplication, layering, elegance">.

**Scope:** `<PACKAGE_DIR>/` source code only. `<DOCS_DIR>/` and archives are out of scope. Superseded code moves to `<ARCHIVE_DIR>/`, never deleted. No external users — no backwards compatibility needed.

**Approach:**
1. Study `<README_FILE>` and the codebase thoroughly before anything else.
2. Identify the central DRY problem — find 2–3 implementations of the same algorithm across different files.
3. Map layering violations — which packages import from which.
4. Prioritize phases by impact: correctness → DRY/unification → elegance → cosmetic.
5. Include a validation gate for each phase (tests that must pass before proceeding).
6. Include a status table tracking what's already done.
7. State explicitly: NO backwards compatibility is a plan constraint.
8. Don't perform any actual changes yet — create the plan document only.

**Constraints:**
- Line counts are NOT important.
- Verify dead code isn't actually dead before archiving.
- Exclude time/effort estimates.
- This plan should be iteratively developed before execution.
```

**Iterative refinement follow-ups** (use in subsequent turns):
- `Review the plan and deeply consider whether it's what we should actually do. Freely revise anything.`
- `Write an addendum with all details discussed, including overlooked items.`
- `Don't just be honest about the boundaries — transcend them.`

---

## 3. Codebase Exploration

**Use when:** Research-only investigation before touching code. Read-only; no edits.

```
Deeply explore the codebase at `<PROJECT_ROOT>` to understand the existing architecture before <SPECIFIC_TASK>.

Deliver a structured report covering:

1. **Project structure** — all top-level directories, key modules, configuration files (`<BUILD_CONFIG>`, etc.)
2. **Target files** — read and summarize:
   - `<FILE_PATH_1>` — focus: <CLASS/METHOD_NAMES>
   - `<FILE_PATH_2>` — focus: <CLASS/METHOD_NAMES>
   For each: 2–3 sentence summary of role, key public API (classes, functions, signatures), external dependencies.
3. **Duplication & dead code** — duplicate implementations, unused functions, commented-out code, TODO/FIXME markers.
4. **Guideline violations** — check against `<GUIDELINES_FILE>`: old-style typing, f-strings in logging, ABC vs Protocol, mutable dataclasses, `Any` usage, `os.path` vs pathlib.
5. **Layering / dependency edges** — which packages import from which; circular imports.
6. **Hardcoded assumptions** — specific optimizer interfaces, training-loop patterns, magic numbers that would need adaptation.
7. **Test landscape** — existing test files, patterns, fixtures, coverage gaps for the target modules.

Exclude: `<EXCLUDE_DIRS>` (e.g., docs/, __pycache__, .git).
Be very thorough — check every `.py` file in scope. Return findings as a detailed structured report. Do NOT write code unless asked.
```

**Targeted deep-dive variant** (for a single module):
```
Read `<FILE_PATH>` and give me:
1. Public API: classes, functions, signatures
2. Key methods on the main class(es)
3. External dependencies
4. What a good set of smoke tests would look like

Also read `<TEST_FILE>` and summarize: existing tests, patterns, fixtures, what's already covered.
```

---

## 4. Bug Hunting & Code Quality Audit

**Use when:** Read-only investigation for bugs, correctness issues, or quality violations. No edits.

```
Search the codebase at `<PROJECT_ROOT>/<PACKAGE_DIR>` for correctness and quality issues. Do NOT edit any files. Report findings with exact file:line references, organized by severity.

**Correctness (CRITICAL/HIGH):**
1. Silent exception swallowing: `except Exception: pass` or `except: pass` in source (not tests)
2. Undefined name references: leftover references to renamed/moved/deleted symbols
3. Stale imports from deleted/moved modules: `from <OLD_MODULE_PATH>` — search ALL files (source, tests, examples)
4. Mismatched function signatures in overrides
5. Algorithmic: division by zero, NaN gradient potential (log/exp/sqrt, 1/x), in-place tensor ops (.add_(), .mul_()), `.detach()` without `.clone()` followed by mutation, `@torch.no_grad()` on methods needing gradients, eval/train mode not restored
6. Logic bugs in: <CRITICAL_FILES_LIST>

**Quality (MEDIUM/LOW):**
7. Backwards-compat shims: aliases (`X = Y`), DeprecationWarning, `__getattr__` forwarding, `try/except ImportError` for old paths
8. `<GUIDELINES_FILE>` violations: `Any` usage, old-style `typing.List/Dict/Optional`, `os.path` (use pathlib), `print()` in production, ABC instead of Protocol, mutable dataclasses, `# type: ignore` without error code
9. Mutable default arguments, `import *`, `__all__` incomplete
10. Dead code: unused functions, unreachable branches, commented-out blocks, files >500 lines with no test coverage

**Output format:** Concise table: `file:line | issue | severity | suggested fix`. Cap at ~25 items. Order by severity. Only include REAL bugs for CRITICAL/HIGH; style issues only for MEDIUM/LOW.
```

**Optional scoped variants** (append to narrow focus):
- `Search specifically for backwards-compatibility patterns in ALL .py files under <PACKAGE_DIR>/ (not docs, not tests).`
- `Audit test quality: flakiness (time.sleep, unseeded random, float == without approx), redundancy, deprecated patterns, slow tests, conftest.py fixtures.`
- `Survey the `<SUBPACKAGE>` package specifically for duplication patterns and inconsistencies between similar classes.`

---

## 5. Mechanical Code-Fix

**Use when:** Systematic, pattern-based fixes across many files (lint errors, API migrations, style conversions). One prompt per directory group or pattern type.

```
Fix ALL instances of `<PATTERN_DESCRIPTION>` in `<DIR_LIST>` under `<PROJECT_ROOT>/<PACKAGE_DIR>/`.

**Step 1 — Assess:**
Run `<LINT_OR_GREP_COMMAND>` first to see the full scope. (e.g., `flake8 <dirs> --count`, `grep -rn "<PATTERN>" <dirs> --include="*.py"`)

**Step 2 — Fix each file:**
For each file:
1. READ the file first.
2. Apply the transformation:
   - `<OLD_PATTERN>` → `<NEW_PATTERN>`
   - (List specific mappings, e.g.:)
     - `os.path.join(a, b)` → `Path(a) / b`
     - `print(...)` → `logger.info(...)` with `%s`-style formatting
     - `from __future__ import annotations` → (remove line)
     - `except Exception: pass` → `logger.warning(...)` / `logger.exception(...)`
3. Add necessary imports (`import logging`, `from pathlib import Path`, etc.) if not present.
4. Remove imports that become unused after the fix.
5. Clean up resulting blank lines.

**Step 3 — Verify:**
Run `<LINT_OR_GREP_COMMAND>` again. Confirm count is 0 (or all instances resolved).
Run `<FORMATTER_COMMAND>` (e.g., `ruff format . && ruff check --fix .`).

**Constraints:**
- Do NOT modify any test logic or assertions — only fix the target pattern.
- Do NOT modify model/algorithm logic.
- Preserve `tqdm.write()` calls (distinct from `print`).
- Use `%s`-style formatting for logger calls (not f-strings).
- Use Edit tool for surgical changes, not Write tool.

**Report:** Summary of changes per file + final verification count.
```

**Common instantiations of `<PATTERN_DESCRIPTION>`:**
| Pattern | Old | New |
|---------|-----|-----|
| Flake8 errors | E501, F401, F841, F821, W293 | Fixed per rule |
| Print → logging | `print(...)` | `logger.info/warning/error(...)` |
| os.path → pathlib | `os.path.join/exists/dirname` | `Path() / .exists() / .parent` |
| `__future__` removal | `from __future__ import annotations` | (delete line) |
| Compat shim removal | aliases, deprecation wrappers | (delete code) |
| Silent except | `except Exception: pass` | `logger.warning/exception(...)` |
| Float equality in tests | `== 0.5` | `== pytest.approx(0.5)` |
| f-string logging | `logger.info(f"...")` | `logger.info("...", var)` |

---

## 6. Test Optimization

**Use when:** Tests are slow and profiling data is available. Speed up without losing coverage.

```
Optimize `<TEST_FILE>` to make tests faster WITHOUT losing coverage.

**Profiling data (slowest tests):**
- `<test_name_1>`: <N>s
- `<test_name_2>`: <N>s
- ...

**Optimization opportunities:**
1. Reduce iteration/epoch counts (e.g., `range(100)` → `range(20)`)
2. Reduce model sizes (embed_dim, hidden_dim, num_layers, neurons_per_tile)
3. Reduce batch sizes / sequence lengths / dataset sizes
4. Replace `time.sleep(N)` with poll-based or deterministic waits
5. Move expensive setup to session/module-level fixtures (server/process spawns)
6. Skip GPU-only tests on CPU (if conditional, leave alone)

**Constraints:**
- DO NOT change assertion semantics — tests must still test what they test.
- DO NOT delete tests or add skip markers.
- Tests must still pass after changes.
- `<GUIDELINES_FILE>`: no comments unless explaining "why", no dead code.

**Procedure:**
1. READ the file first.
2. Use Edit tool for surgical changes.
3. Run before/after:
   `<TEST_RUNNER> <TEST_FILE> -q --tb=short -p no:cov -p no:warnings --durations=10 2>&1 | tail -20`

**Report:** Initial time + count → investigation findings (bottleneck) → changes made (each edit) → final time + count → biggest speedup.
```

---

## 7. Test Creation / Coverage

**Use when:** Creating new test files for low-coverage modules.

```
Create test file(s) in `<TEST_DIR>/` to increase coverage on low-coverage modules. Follow patterns in existing tests (e.g., `<REFERENCE_TEST_FILE>`). Verify with: `<TEST_RUNNER> --cov=<PACKAGE> --cov-report=term-missing`.

**Context:** `<FORMATTER>` will be run separately. Python <VERSION>, torch, numpy available. Use `torch.manual_seed(42)` for reproducibility.

## File: `<TEST_DIR>/test_<NAME>.py`
**Target:** `<MODULE_PATH>` (currently <X>% coverage, <Y> LOC)
**Classes/functions to test:**
- `<ClassName>` — <description>
- `<function_name>` — <description>

**Tests to include:**
- `<test_name>` — <what to assert>
- `<test_name>` — <what to assert>
- A hypothesis property test for `<INVARIANT>`

**Rules:**
- Import from the correct module path: `from <PACKAGE>.<MODULE> import <CLASS>`
- Use small fixtures (tiny tensors, 2-layer models)
- Follow existing test patterns in the codebase
- No network calls, no dataset downloads
```

---

## 8. Code Refactoring

**Use when:** Concrete code transformations — the actual implementation work. Three sub-forms depending on the type of change.

### 8a. Mechanical Pattern Replacement

```
Mechanically refactor `<PATTERN_NAME>` across the codebase.

**Shared helper (already defined in `<HELPER_PATH>`):**
```python
<HELPER_CODE>
```

**Pattern:** Replace:
```python
<OLD_PATTERN_EXAMPLE>
```
With:
```python
<NEW_PATTERN_EXAMPLE>
```

**Files to update:**
1. `<FILE_PATH_1>` — <N> occurrences at lines <X, Y, Z>
2. `<FILE_PATH_2>` — <M> occurrences at lines <A, B>
...

**Import to add:** `from <IMPORT_PATH> import <HELPER_NAME>`

**After all edits, run:** `<TEST_RUNNER> --no-cov -q -x 2>&1 | tail -5`
If tests fail, show the error output and fix.
```

### 8b. Structural Refactoring (Deduplication / Extraction / Unification)

```
Refactor `<SCOPE_DESCRIPTION>` to eliminate duplication.

**Goal:** Extract shared logic into `<NEW_HELPER_PATH>` and reduce N implementations to thin wrappers.

**New helper(s) to create:**
- `<function_name>()` — <description, parameters, return>
- `<function_name>()` — <description>

**Classes/files to refactor:**
1. `<CLASS_1>` in `<FILE_1>` — currently <N> lines of `<WHAT>`; reduce to <M> lines calling the helper.
2. `<CLASS_2>` in `<FILE_2>` — variant: <DESCRIBE_VARIANT_PARAMETER, e.g., "uses use_conj=True">
3. ...

**Constraints:**
- Preserve all public API signatures.
- Handle variants via parameters (not separate code paths).
- Add docstrings to new helpers.
- No backwards compatibility needed.

**Verify:** `<TEST_RUNNER> --no-cov -q -x` and `<TYPE_CHECKER> <PACKAGE_DIR>`
```

### 8c. File/Directory Operations (Moves, Merges, Import Migration)

```
Execute the following structural changes in `<PROJECT_ROOT>`:

**Task 1: <MOVE/MERGE/DESCRIPTION>**
- Source: `<SOURCE_PATH>`
- Target: `<TARGET_PATH>`
- Method: <cp -r / git mv / concatenate>
- After: verify all files present; list resulting structure.

**Task 2: Fix all imports affected by the move.**
- Replace `from <OLD_IMPORT_PATH>` with `from <NEW_IMPORT_PATH>` across ALL files.
- Run: `grep -rn "from <OLD_IMPORT_PATH>" <PROJECT_ROOT> --include="*.py"`
- For each file: read, replace all instances, also fix `import <OLD_MODULE>` forms.
- Update `<BUILD_CONFIG>` packages section if applicable.
- Update `__init__.py` exports.

**Task 3: Verify.**
- `<TEST_RUNNER> --no-cov -q -x`
- `<TYPE_CHECKER> <PACKAGE_DIR>`
- `grep -rn "<OLD_IMPORT_PATH>" <PROJECT_ROOT> --include="*.py"` → should return 0 results.

**Return:** List of all files modified and a summary.
```

---

## 9. Validation & Quality-Gate

**Use when:** Verifying a plan or implementation is complete, correct, and clean. End-of-phase gate.

```
Ensure `@<PLAN_FILE>` is completely implemented and clean (following `<GUIDELINES_FILE>` guidelines).

**Checklist:**
1. **Functionality retained** — all capabilities present in some usable form since the refactoring began. Nothing silently lost.
2. **Plan completeness** — every item in `<PLAN_FILE>` is marked done or explicitly deferred with reason. No orphaned TODOs.
3. **No backwards compatibility** — remove any trace: aliases, deprecation warnings, `__getattr__` forwarding, re-export shims, `try/except ImportError` for old paths.
4. **Bug scan** — search for concrete bugs introduced by the refactoring (stale imports, undefined names, signature mismatches). Report as table: `file:line | bug | fix`.
5. **Test health** — all tests pass. Before running, profile for optimization opportunities (reduce iterations, model sizes). Add tests for any fixes made.
6. **Guideline conformance** — `<GUIDELINES_FILE>` violations resolved in touched files.
7. **Documentation** — `<README_FILE>` reflects current state; no stale references to moved/deleted modules.

**Focus on** significant, impactful work — not cosmetic lint.
**Exclude:** `<DOCS_DIR>/` (do not read or edit).

Before wrapping-up, detail in `<PLAN_FILE>`: progress made, new discoveries, remaining-work hints for the next session.
```

---

## 10. Documentation

**Use when:** Writing or revising the project README as the single source of truth.

```
Revise `<README_FILE>` as the primary (and only necessary) documentation and guide to all codebase functionality.

**Process:**
1. Thoroughly study the codebase and the previous version of `<README_FILE>` (in git history).
2. Ensure all details, functionality, and components are present in some form.
3. Ensure each major component maps to at least one main source file path.

**Content requirements:**
- Begin with a compelling introduction/overview conveying the full potential impact of the system.
- Comprehensive list of algorithms, optimizers, architectures (the "Zoo").
- Flow-chart diagrams of automated processes (if applicable).
- Project structure overview.

**Style constraints:**
- Technologically optimistic and inspiring, yet factual/honest.
- Egotistically sterile: no "us", "our", "we".
- Moderate use of emojis.
- Exclude: numeric results (likely to change), code snippets, volatile implementation details.
- Don't feature any single algorithm disproportionately.

**Exclude:** `<DOCS_DIR>/` and archives. Non-README documentation is archived, not maintained.
```

---

## Appendix: Recurring Directives

These constraints appeared across all prompt categories. Append as needed:

| Directive | When to use |
|-----------|-------------|
| No backwards compatibility — no external users | Plan creation, refactoring, validation |
| Archive to `<ARCHIVE_DIR>/`, never delete | Plan creation, file operations |
| Follow `<GUIDELINES_FILE>` | All code-touching prompts |
| Be very thorough; check every `.py` file | Exploration, audits |
| Report with exact `file:line` references | Audits, bug hunts |
| Run before/after tests with `<TEST_RUNNER>` | All code changes |
| Read the file first before editing | Mechanical fixes, refactoring |
| Do NOT modify test logic/assertions | Lint fixes in test directories |
| Use Edit tool (surgical), not Write tool | All code changes |
| `<TEST_RUNNER>` is the canonical test command | Always (never bare `pytest`) |
| Prioritize architectural work over cosmetic lint | Continuation, validation |
| If unsure, stop and ask | All prompts |

---

## Appendix: Variable Reference

| Variable | Example value | Description |
|----------|--------------|-------------|
| `<PROJECT_ROOT>` | `/home/me/myproject` | Absolute path to repo root |
| `<PACKAGE_DIR>` | `bioplausible` | Importable Python package name |
| `<PLAN_FILE>` | `REFACTOR3.md` | Current plan document |
| `<GUIDELINES_FILE>` | `AGENTS.md` | Coding standards/conventions file |
| `<README_FILE>` | `README.md` | Project readme |
| `<TEST_DIR>` | `tests/` | Test directory |
| `<TEST_RUNNER>` | `uv run pytest` | Canonical test invocation |
| `<FORMATTER>` | `ruff format . && ruff check --fix .` | Auto-formatter + linter |
| `<TYPE_CHECKER>` | `pyright` | Static type checker |
| `<BUILD_CONFIG>` | `pyproject.toml` | Build/packaging config |
| `<ARCHIVE_DIR>` | `docs/archive/20260801/` | Date-stamped archive for superseded code |
| `<DOCS_DIR>` | `docs/` | Documentation directory (usually out of scope) |
| `<EXCLUDE_DIRS>` | `docs/, __pycache__, .git, examples/` | Dirs to skip in searches |
| `<LINT_OR_GREP_COMMAND>` | `flake8 <dirs> --count` | Assessment command |
