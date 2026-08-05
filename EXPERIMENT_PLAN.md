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
| 2.1 Task registry | `data/vision.py`: `SUPPORTED_TASKS` (incl. `usps`), `TaskSpec`, `resolve_task`. | `resolve_task` covers every name; parity.py's `_TASK_DIMS` (parity.py:33) and `demo/runner.py:119` both deleted and resolved via registry. |
| 2.2 Geometry chains | replace `schema.py` `arm_input_dim/arm_output_dim` chains with `resolve_task` (+ optional arm override). | arm geometry resolves purely from registry or explicit override. |

---

## 5. Phase 3 — The Thin Layer (architecture §6)

New `experiment/` package — the only genuinely new code beyond Phases 1–2.

| Task | Content | Acceptance |
|------|---------|------------|
| 3.1 `experiment/schema.py` | migrate + rewrite `campaign/schema.py`; stages/pass-rule/grid; `matched_by`; dual energy; `seeds≥10` rule. | `validate` rejects unknown task, `seeds<10` on `baseline:` stages, missing `matched_by`/energy on parity. |
| 3.2 `experiment/param_estimator.py` | migrate `campaign/param_estimator.py`; fix legacy-except. | `estimate_param_count` works; reused for `max_params`. |
| 3.3 `experiment/probe.py` | `ProbeResult`, `ProbeDriver`, `CoreTrainerDriver`, `run_probe` (single normalization). | normalizes `verify`'s per-seed records; `param_count` from estimator; no `getattr` soup. |
| 3.4 `experiment/producer.py` | `ConfigProducer`, `ProbeWork`, `HyperoptGridProducer` (via `hyperopt.create_study` + `GridSampler`). | `plan` enumerates exact probe count; resume skips finished `config_key`s. |
| 3.5 `experiment/staircase.py` | `PassRule`/`Verdict`/`Stage` + `StaircaseRunner` (survivor cascade). | smoke→…→parity cascade; only survivors advance. |
| 3.6 `experiment/report.py` | JSONL Report + resume index + reporter hooks. | re-run is a no-op for finished probes. |

**Push-back**: `run_verify` already emits per-seed JSONL with CI metadata — 3.3/3.6 must
**consume** it, not re-implement. If the existing record shape is insufficient, extend
`cli/run.py` (and record in §10) rather than forking a parallel trace.

---

## 6. Phase 4 — Wire Existing CLIs & Retire `campaign/` (architecture §8, §10)

| Task | Action | Acceptance |
|------|--------|------------|
| 4.1 Wire `biopl-parity` | extend existing parity CLI to drive parity-stage runs + emit Report. | parity campaign runs through the layer. |
| 4.2 Repoint `biopl-run` | `pyproject.toml:128` off `campaign.cli:main`; add `validate`/`plan`/`run`. | `biopl-run plan/validate/run` all work. |
| 4.3 `biopl-report` / `biopl-repro-check` | wire to Report + gradient gate. | report renders; nightly gate runs. |
| 4.4 Retire `campaign/` | `git rm` executor/runner/tiers/search_space/logger/cli; migrate schema+param_estimator. | `campaign/` gone; `__init__` rewritten; no dangling imports; `biopl-run` still resolves. |
| 4.5 Tests | migrate the 61 `tests/unit/campaign/` tests that still apply; drop the rest. | full unit suite green at cov 85. |

---

## 7. Phase 5 — End-to-End Overnight Smoke

Run the parity campaign (architecture §6.1 sample YAML, `seeds: 10` where required) start to
finish, then re-run to prove resume. Confirm the §13 "Runnable" checklist 1–9 end to end.

---

## 8. Phasing Note

Phases are **dependency-ordered but not strictly serial** where safe: Phase 1 modules are
independent of Phase 2/3 and can proceed in parallel; Phase 0 gates everything. Do not
start Phase 3 before Phase 0 (unverifiable).

---

## 9. Definition of Done (checklist — mirrors architecture §13)

- [ ] 0.1–0.5 tooling green (ruff/pyright/cov 85)
- [ ] 1.1 statistics + 1.2 gradient gate unit-tested
- [ ] 2.1–2.2 registry dedup (no `_TASK_DIMS`, no geometry chains)
- [ ] 3.1–3.6 layer working; `validate`/`plan`/`run`/resume correct
- [ ] 4.1–4.5 CLIs wired; `campaign/` retired; tests green
- [ ] 5 overnight smoke passes; re-run = no-op; report renders; gradient gate on parity-tier models
- [ ] `EXPERIMENT_ARCHITECTURE.md` and `EXPERIMENT_PLAN.md` in sync (no un-recorded drift)

---

## 10. Decision Ledger

Append every during-build discovery that changes scope, design, or the fact base. Keeps the
promise that no detail is lost and that both canonical files stay truthful.

| Date | Decision / discovery | Effect | Applied to |
|------|----------------------|--------|------------|
| final-validation | `run_verify` record shape may need epoch-level detail for `settling_steps`; confirm before 3.6. | possibly extend `cli/run.py` output | architecture §4.1, proj report |
| final-validation | Gradient-equivalence gate **already exists** as helpers in `tests/integration/test_gradient_equivalence.py` (`_finite_diff_gradient`, `_check_gradient_equivalence`, `_local_direction`, per-family thresholds). | Phase 1.2 = promote-to-production refactor, not net-new; `validation/analysis.py` is energy/Lyapunov (not stats), so 1.1 stays net-new. | plan §3 (applied), architecture §7#2 |

---

*Actionable, non-duplicative, dependency-ordered, and verifiable at each step. When building,
update this plan + the architecture doc in the same commit; never let them drift.*
