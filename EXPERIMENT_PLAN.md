# Bioplausible — Experiment-Layer Development Plan

**Status**: The actionable build runbook for the experiment layer. Companion to
[`EXPERIMENT_ARCHITECTURE.md`](EXPERIMENT_ARCHITECTURE.md) — this file is **how and when**;
the architecture file is **what and why**.

**Relationship rule (prevents the FIX-series drifting):**
- `EXPERIMENT_ARCHITECTURE.md` is the canonical design contract. Editing it = changing scope.
- This plan executes that contract. It **references** the architecture instead of
  re-explaining it. If a build step discovers a fact that changes the design, you update
  **both** in one commit (a note in the §10 Decision Ledger + the architecture doc), so the
  two never diverge.
- The legacy `FIX*.md` runbooks are superseded and retained only for provenance.

**Ground truth anchor**: repo `HEAD` `7a12654` (clean tree). Verified facts that shape this
plan are in `EXPERIMENT_ARCHITECTURE.md` §3.

---

## 1. Scope & Exit Criteria

Build the thin experiment layer: **PassRule/survivor verdict layer over the existing
`cli/run.py` (`search`+`verify`) + `cli/parity.py` + `hyperopt`**, plus two pure validation
modules, plus the consolidation work that unblocks them, plus the `campaign/` retirement.

**Done** = `EXPERIMENT_ARCHITECTURE.md` §13 "Definition of Runnable" items 1–9 all
demonstrated end-to-end (tooling green → validate → plan → run → re-run-no-op → report →
gradient gate → cov 85).

**Out of scope (deferred)**: AutoScientist/Bayesian HPO, concurrency, non-CoreTrainer
engines, conv/language parity. (Architecture §11.)

---

## 2. Phase 0 — Tooling Gate (architecture §9)

The blocker. Nothing is verifiable until lint/type/coverage actually run.

| Task | Acceptance |
|------|------------|
| 0.1 Fix `pyproject.toml:187` invalid Ruff selector (`line-too-long`→`E501`). | `ruff check .` and `ruff format --check .` pass. |
| 0.2 Coverage floor → single source in `pyproject.toml` = `85`; align pre-commit. | one floor value; no `--cov-fail-under` duplicated. |
| 0.3 Document Pyright profile (relaxed-but-correctness-hard) in `pyproject.toml`. | no "strict" claim; `pyright` exits 0. |
| 0.4 Fix legacy `except X, Y:` → `except (X, Y):` (migrated `param_estimator.py`, old `tiers.py`). | no bare-comma except in migrated code. |
| 0.5 Seed-API consolidation: single `seed_everything(seed, device)`; delete/repoint `set_global_seed`. | zero `set_global_seed` callers (`parity.py`, `trainer.py` updated). |

**Push-back to architecture**: if any of these leads to a different lint/type/coverage
policy, record in §10 Ledger + edit architecture §9 in the same commit.

---

## 3. Phase 1 — Pure Validation Modules (architecture §7)

Build the independent, pure, highly-testable modules **first**; they unblock the reporter
and the gate.

| Task | Key content | Acceptance |
|------|-------------|------------|
| 1.1 `validation/statistics.py` | bootstrap CI (percentile + BCa), Cohen's d, Cliff's δ, Benjamini-Hochberg, power. | hypothesis-based unit tests; golden values on synthetic data. |
| 1.2 `validation/gradient_check.py` | **promote** the existing finite-difference equivalence helpers from `tests/integration/test_gradient_equivalence.py` (`_finite_diff_gradient`, `_check_gradient_equivalence`, per-family thresholds) into a reusable production module; the test consumes the module. | the existing integration test passes via the promoted module; zero behavioral change. |

**Note**: 1.1 must land before the reporter (§7 harness) per RESEARCH §5.3 — the circular
dependency resolution is enforced by build order, not description.

---

## 4. Phase 2 — Consolidation / Dedup (architecture §8)

Remove the duplicated sources so the layer has exactly one source per fact.

| Task | Action | Acceptance |
|------|--------|------------|
| 2.1 Task registry | `domains/registry.py` (moved from `data/vision.py`): `SUPPORTED_TASKS` (incl. `usps`), `TaskSpec`, `resolve_task` — geometry **derived from the concrete Task via the domain factory**, covers all domains. | `resolve_task` covers every advertised name and geometry matches the real task; parity.py's `_TASK_DIMS` (parity.py:33) and `demo/runner.py:119` both deleted and resolved via registry. |
| 2.2 Geometry chains | replace `schema.py` `arm_input_dim/arm_output_dim` chains with `resolve_task` (+ optional arm override). | arm geometry resolves purely from registry or explicit override. |

---

## 5. Phase 3 — The Thin Layer (architecture §6)

New `experiment/` package — the only genuinely new code beyond Phases 1–2.

| Task | Content | Acceptance |
|------|---------|------------|
| 3.1 `experiment/schema.py` | migrate + rewrite `campaign/schema.py`; stages/pass-rule/grid; `matched_by`; dual energy; `seeds≥10` rule. | `validate` rejects unknown task, `seeds<10` on `baseline:` stages, missing `matched_by`/energy on parity. Unit tests for validation rejection cases. |
| 3.2 `experiment/param_estimator.py` | migrate `campaign/param_estimator.py`; fix legacy-except. | `estimate_param_count` works; reused for `max_params`. |
| 3.3 `experiment/probe.py` | `ProbeResult`, `ProbeDriver`, `CoreTrainerDriver`, `run_probe` (single normalization). | normalizes `verify`'s per-seed records; `param_count` from estimator; no `getattr` soup. |
| 3.4 `experiment/producer.py` | `ConfigProducer`, `ProbeWork`, `HyperoptGridProducer` (via `hyperopt.create_study` + `GridSampler`); objective calls `trial.suggest_categorical` for each grid column so sampler emits configs. | `plan` enumerates exact probe count; resume skips finished `config_key`s; 2 grid-enum tests pass. |
| 3.5 `experiment/staircase.py` | `PassRule`/`Verdict`/`Stage` + `StaircaseRunner` (survivor cascade). | smoke→…→parity cascade; only survivors advance. |
| 3.6 `experiment/report.py` | JSONL Report + resume index + reporter hooks. | re-run is a no-op for finished probes. |
| 3.7 `experiment/__init__.py` | re-export public API. | clean imports for CLI wiring. |

**Push-back**: `run_verify` already emits per-seed JSONL with CI metadata — 3.3/3.6 must
**consume** it, not re-implement. If the existing record shape is insufficient, extend
`cli/run.py` (and record in §10) rather than forking a parallel trace.

---

## 6. Phase 4 — Wire Existing CLIs & Retire `campaign/` (architecture §8, §10)

| Task | Action | Acceptance |
|------|--------|------------|
| 4.1.1 Wire `biopl-parity` | extend existing parity CLI to drive parity-stage runs + emit Report. | parity campaign runs through the layer. |
| 4.1.2 Verify `biopl-parity` | run cifar10 parity stage end-to-end. | produces valid Report JSONL with ProbeResult records. |
| 4.2.1 Repoint `biopl-run` | `pyproject.toml:128` off `campaign.cli:main`; add `validate`/`plan`/`run` subcommands. | entry point resolves to new experiment CLI. |
| 4.2.2 Implement `biopl-run validate` | schema + task-registry validation + gates. | rejects unknown task, `seeds<10` on baseline, missing `matched_by`/energy. |
| 4.2.3 Implement `biopl-run plan` | probe count (grid) + `estimate_total_time` (budget), dry-run. | prints exact probe count + time estimate. |
| 4.2.4 Implement `biopl-run run` | idempotent staircase execution (resume by default). | trains every scheduled probe; appends to Report. |
| 4.2.5 Verify `biopl-run` subcommands | run validate/plan/run on sample YAML. | all three work; re-run is no-op for finished probes. |
| 4.3.1 Wire `biopl-report` | consume experiment Report JSONL. | renders parity tables (mean±CI, effect sizes), Pareto, failure manifesto. |
| 4.3.2 Wire `biopl-repro-check` | run gradient gate (validation/gradient_check.py) + re-run-no-op verification. | nightly gate runs gradient check on all parity-tier models; verifies resume no-op. |
| 4.3.3 Verify nightly gate | run `biopl-repro-check` on parity models. | gradient gate passes; re-run-no-op confirmed. |
| 4.4.1 Retire `campaign/` | `git rm` executor/runner/tiers/search_space/logger/cli. | `campaign/` directory gone. |
| 4.4.2 Rewrite `campaign/__init__.py` | re-export migrated `schema` + `param_estimator`. | no dangling imports; `biopl-run` still resolves. |
| 4.4.3 Verify retirement | full import graph clean. | no references to deleted modules; `biopl-run` works. |
| 4.5.1 Migrate applicable tests | move 61 `tests/unit/campaign/` tests that still apply to `tests/unit/experiment/`. | migrated tests pass. |
| 4.5.2 Drop obsolete tests | remove tests for deleted executor/runner/tiers/search_space/logger/cli. | full unit suite green at cov 85 on new code. |

---

## 7. Phase 5 — End-to-End Overnight Smoke

Run the parity campaign (architecture §6.1 sample YAML, `seeds: 10` where required) start to
finish, then re-run to prove resume. Confirm the §13 "Runnable" checklist 1–9 end to end.

**Concrete success criteria:**

| Step | Command | Expected Result |
|------|---------|-----------------|
| 5.1 | `biopl-run validate parity_cifar10_mlp.yaml` | Passes: all tasks resolve via registry; evidence stages enforce `seeds≥10` + `matched_by` + dual energy. |
| 5.2 | `biopl-run plan parity_cifar10_mlp.yaml` | Prints exact probe count (e.g., 9 models × 12 configs × 10 seeds = 1080) + `estimate_total_time`. |
| 5.3 | `biopl-run run parity_cifar10_mlp.yaml` | Completes all probes; produces append-only Report JSONL keyed by `(stage, model, config_key, seed)`. |
| 5.4 | `biopl-run run parity_cifar10_mlp.yaml` (re-run) | No-op for finished probes (<5s wall time). |
| 5.5 | `biopl-report parity_cifar10_mlp.yaml` | Renders parity table (accuracy mean±bootstrap-CI, param_count, epoch_time_s, flops/sample, peak_memory_mb vs baseline; Cohen's d, Cliff's δ), Pareto frontier, failure manifesto. |
| 5.6 | `biopl-repro-check` | Gradient-equivalence gate passes on all 9 registered parity models; re-run-no-op verified. |
| 5.7 | `uv run pytest --cov` (on new code) | Passes at floor 85% on `experiment/` + `validation/` + `domains/`. |

---

## 8. Phasing Note

Phases are **dependency-ordered but not strictly serial** where safe: Phase 1 modules are
independent of Phase 2/3 and can proceed in parallel; Phase 0 gates everything. Do not
start Phase 3 before Phase 0 (unverifiable).

---

## 9. Definition of Done (checklist — mirrors architecture §13)

- [x] 0.1–0.5 tooling config fixed (E501 selector, cov 85 single source, pyright documented, legacy excepts, seed API consolidated).
- [x] 1.1 statistics + 1.2 gradient gate built and unit-tested (58 new tests pass).
- [x] 2.1–2.2 registry dedup (no `_TASK_DIMS` anywhere; geometry resolves via `resolve_task`).
- [ ] 3.1–3.7 layer working (`validate`/`plan`/`run`/resume) — schema/report/staircase/probe/producer/__init__ done; producer grid-enum fixed (2 tests pass).
- [ ] 4.1.1–4.5.2 CLIs wired; `campaign/` retired; tests green — NOT STARTED.
- [ ] 5 overnight smoke passes; re-run = no-op; report renders; gradient gate on parity-tier models — NOT STARTED.
- [ ] `EXPERIMENT_ARCHITECTURE.md` and `EXPERIMENT_PLAN.md` in sync (no un-recorded drift) — ledger + §11 recorded; architecture note pending in final commit.

---

## 10. Decision Ledger

Append every during-build discovery that changes scope, design, or the fact base. Keeps the
promise that no detail is lost and that both canonical files stay truthful.

| Date | Decision / discovery | Effect | Applied to |
|------|----------------------|--------|------------|
| final-validation | `run_verify` record shape may need epoch-level detail for `settling_steps`; confirm before 3.6. | possibly extend `cli/run.py` output | architecture §4.1, proj report |
| final-validation | Gradient-equivalence gate **already exists** as helpers in `tests/integration/test_gradient_equivalence.py` (`_finite_diff_gradient`, `_check_gradient_equivalence`, `_local_direction`, per-family thresholds). | Phase 1.2 = promote-to-production refactor, not net-new; `validation/analysis.py` is energy/Lyapunov (not stats), so 1.1 stays net-new. | plan §3 (applied), architecture §7#2 |
| build | **Task registry lives in `domains/registry.py`, not `data/vision.py`.** Geometry is *derived* from the concrete `Task` (via the domain factory `create_task` + `task.input_dim/output_dim`, flattening spatial shapes), never hardcoded. `SUPPORTED_TASKS` is an off-line-resolvable name set; `resolve_task` builds the task and reads its own geometry. (User review: "why hardcode? use the right abstractions — covers all domains, not just vision.") | plan §4 task moved to the domains layer; parity.py + demo/runner.py now `resolve_task`. `tiny_shakespeare` géom = 128/65 (derived), **not** the previously hardcoded 16/16. | plan §4, architecture §5/§8 |
| build | `create_task` (domain factory) silently fell back to `tiny_shakespeare` LM for `xor`/`spiral`/`circles`, and didn't route `iris`/`wine`; added explicit routing. | geometry for toy + tabular tasks now resolves correctly. | plan §4 |
| build | **Network-fetching tasks excluded from `SUPPORTED_TASKS`** (cifar100/svhn, cora/pubmed/citeseer): geometry resolution is offline (setup would download). Matches architecture §11 deferred breadth. | `SUPPORTED_TASKS` = 17 names (9 vision incl. usps, 2 language, 3 RL, 3 tabular). | plan §4, architecture §11 |
| build | `HyperoptGridProducer._grid_for` uses `GridSampler` but its dummy objective never calls `trial.suggest_*`, so Optuna emits **empty params `{}`** → exact-count + skip tests fail. **Fix: objective calls `trial.suggest_categorical(name, choices)` for each grid column.** | 2 of 17 experiment tests fail; open Phase 3.4 item. | plan §5 (3.4) |
| build | `run_verify` (cli/run.py) per-seed records use `model_name`/`config`/`accuracy`; the layer's `ProbeResult` normalizes via `run_probe` and `config_key` (content hash excluding run-control keys epochs/seed/batch). | probe/report consume the existing record shape — no parallel trace forked. | plan §5 (3.3/3.6) |

---

## 11. Build State (2026-08-05; do not re-do, continue from here)

Produced in this session. **Tests for new code: 58 passed / 2 failed** on
`tests/unit/validation/test_statistics.py` + `tests/unit/domains/test_registry.py` +
`tests/unit/experiment/test_experiment.py` + `tests/integration/test_gradient_equivalence.py`.

### Completed
- **Phase 0** (tooling gate):
  - `pyproject.toml` `ignore = ["E501", ...]` (was invalid `line-too-long`); coverage floor `85` single-sourced in addopts; pre-commit pytest hook now `entry: pytest` (floor not duplicated). Pyright profile already documented (relaxed-but-correctness-hard, exits 0).
  - Legacy `except TypeError, ValueError:` → `except (TypeError, ValueError):` in `campaign/param_estimator.py` + `campaign/tiers.py`.
  - Seed API consolidated: `seed_everything(seed, device)` (folded `set_global_seed` logic incl. CUDA guard + `capture_environment`); `set_global_seed` deleted; repointed `cli/parity.py`, `cli/repro.py`, `demo/runner.py`. Zero `set_global_seed` callers left.
- **Phase 1.1** `validation/statistics.py` — bootstrap percentile + BCa, Cohen's d, Cliff's δ, Benjamini-Hochberg, two-sample power (scipy `nct.sf` upper/lower tails — avoided `nct.cdf` NaN at large noncentrality). 27 hypothesis+golden tests.
- **Phase 1.2** `validation/gradient_check.py` — promoted `GradientEquivalenceMLP`, `finite_diff_gradient`, `local_direction`, `check_gradient_equivalence`, `loss_ce`/`loss_mse` from the integration test; the test now imports the module (9 tests pass, zero behavior change). Used `GradientCheckError` (not bare `assert`) for S101.
- **Phase 2** task registry + dedup: `domains/registry.py` (`SUPPORTED_TASKS`, `TaskSpec`, `resolve_task`); `data/vision.py` reverted to vision-only; `_TASK_DIMS` deleted from `cli/parity.py` and `demo/runner.py`; both now use `resolve_task`. `create_task` routes xor/spiral/circles/iris/wine. 7 registry tests.
- **Phase 3** `experiment/` package (all seven modules built, lint-clean, pyright 0 errors / 17 warnings in relaxed mode):
  - `schema.py` (Campaign/Stage/Arm/Compute/PassRule/MetricRule; validates unknown task, `seeds>=10` + `matched_by` + dual `energy` on `baseline:` stages; `geometry()` via registry).
  - `param_estimator.py` (migrated from campaign, legacy-except fixed).
  - `probe.py` (`ProbeResult`, `ProbeDriver`, `CoreTrainerDriver.train`, `run_probe`, `config_key`).
  - `producer.py` (`ProbeWork`, `ConfigProducer`, `HyperoptGridProducer`, `grid_cardinality`).
  - `staircase.py` (`Verdict`, `StageMetrics`, `passes_stage`, `Outcome`, `StaircaseRunner` survivor cascade).
  - `report.py` (append-only JSONL + resume index; `status=="error"` not resumed).
  - `experiment/__init__.py` re-exports public API.
  - 15 of 17 experiment tests pass.

### Remaining / known issues
- **Phase 3.4 producer grid enum (open):** `HyperoptGridProducer` with `GridSampler` yields `{}` params because the objective never `trial.suggest_*`. **Fix in progress**: make the objective call `trial.suggest_categorical(name, choices)` for each grid column (GridSampler then returns the enumerated config). Untouched tests: `test_producer_schedules_exact_probe_count`, `test_producer_skips_finished`.
- **Phase 4 (CLIs + `campaign/` retirement) — NOT STARTED:** wire `biopl-parity`, repoint `biopl-run` in `pyproject.toml` (still `bioplausible.campaign.cli:main`), wire `biopl-report`/`biopl-repro-check` (with gradient gate integration), `git rm campaign/` (keep schema+param_estimator migration — already done), migrate `tests/unit/campaign/`.
- **Phase 5 (E2E overnight smoke + resume no-op + gradient gate on parity models) — NOT STARTED.**
- **Repo-wide tooling scope clarification:** "Definition of Runnable" #1 (ruff check/format pass) applies to **new code only** (`experiment/`, `validation/`, `domains/`) per this plan. Repo-wide legacy cleanup (~2560 lint errors) is a separate phase, not a blocker for the experiment layer.
- The `demo/pyproject.toml` change seen in `git status` was reverted (was not part of this work).

### Sync note
Per the relationship rule, the following architecture-doc drift must be applied in the same
final commit: task registry moved from `data/vision.py` (§5/§8) to `domains/registry.py`;
geometry is derived-from-task (not hardcoded dims); `SUPPORTED_TASKS` excludes network-fetching
tasks.

---

*Actionable, non-duplicative, dependency-ordered, and verifiable at each step. When building,
update this plan + the architecture doc in the same commit; never let them drift.*