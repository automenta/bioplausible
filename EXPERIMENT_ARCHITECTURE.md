# Bioplausible — Experiment-Layer Architecture (Canonical)

**Status**: Single source of truth for the bioplausible experiment layer. Supersedes the
speculative FIX chain (FIX2a, FIX2b, FIX3b, FIX3c) as the authoritative contract. Every
surviving decision is carried forward below **with provenance** (§2), so no detail from the
history is silently dropped. Content here is written against, and verified against, repo
`HEAD` (`7a12654`, 647 commits, clean tree).

**Design principle that governs everything in this doc**: *the new engine stands on the
infrastructure that already exists* — `bioplausible/cli/` and `bioplausible/hyperopt/` are
**reused and deduplicated**, never re-created. The only genuinely new code is a thin
survivor-cascade verdict layer plus two pure validation modules.

---

## 1. Goal & Non-Goals

**Goal**: a YAML-driven experiment layer producing *publication-grade parity evidence*
(bio-plausible vs backprop), runnable **unattended overnight**, with a small, auditable,
deduplicated core. Directly serves RESEARCH.md Phase 0.1, 0.3, 5.2, 5.3 and cheapens Phase 1.

**Non-goals (explicitly out of v1)**: Bayesian/AutoScientist HPO as the core path;
concurrent probe scheduling; non-`CoreTrainer` engines; full Phase-0 breadth (conv
/ language) — each deferred, with re-entry paths (§11).

---

## 2. Provenance — what each FIX established (nothing dropped)

| Source | Durable contribution carried forward |
|--------|--------------------------------------|
| FIX2a | The original (over-engineered) YAML campaign + Optuna machinery. Contributions **rejected** here: multi-objective HPO as reporting optimizer, pareto pruning in runner, per-`(arm,model,task)` SQLite studies, `eval()` constraints, hardcoded tier geometry. |
| FIX2b | The **architecture**: probe/staircase/report decomposition; content-addressed resume; task-registry geometry; structured `PassRule`; parity axes recorded per probe; report-only statistics. |
| FIX3b | Corrections to false repo premises (executor is **tracked**, tree is clean, ruff **broken**, legacy-`except` present); the **tooling gate**; `ProbeDriver` & `ConfigProducer` flex seams; the **research-validity gates** (n≥10 seeds, compute-match contract, dual energy model, gradient-equivalence P0, declared JSONL-vs-SQLite divergence). |
| FIX3c | The **re-anchoring audit**: `cli/run.py`, `hyperopt/`, `cli/parity.py`, `_TASK_DIMS` ×2, dual seed API all **already exist** — the engine must reuse them, and `campaign/` must be retired, not grown. |
| FIX3d (this) | **Convergence**: `cli/run.py::run_verify` already re-runs top-k configs across n seeds and emits per-seed JSONL with CI/effect-size metadata — so the "engine" collapses to a **thin PassRule/survivor verdict layer over the existing search+verify+parity+SQLite surface**, with no new probe/report/SQLite engine. |

Net: scope has converged from a full rewrite (~20 hr) to a thin layer + two validation
modules (~8–10 hr). This convergence is the *result* of the audit, not a reduction of the
goal.

---

## 3. Verified Repository Ground Truth (fact base)

Checked against `HEAD`; keep these fixed in mind — they are why the plan looks the way it does:

| Fact | Evidence |
|------|----------|
| `executor.py` is **tracked** (not untracked) and exported | added in `45b3544`; `campaign/__init__.py` imports `CampaignExecutor/TrialContext/run_campaign` |
| Working tree is clean (no "pending improvements" to preserve) | `git status --porcelain` empty |
| Ruff **cannot run**: invalid selector `line-too-long` | `pyproject.toml:187` → "Unknown rule selector" |
| Coverage floor mismatch | `pyproject` `--cov-fail-under=50` vs pre-commit `85` vs AGENTS.md `85` |
| Pyright is *not* strict (deliberately relaxed) | `pyproject.toml:219` relaxes `report*` rules; hard-correctness rules stay `error` |
| Legacy `except X, Y:` present | `param_estimator.py:84`, `tiers.py:121` |
| **Two** live seed APIs | `utils.py:21 seed_everything`, `utils.py:39 set_global_seed` |
| `_TASK_DIMS` duplicated twice | `cli/parity.py:33`, `demo/runner.py:119` |
| Full experiment runner already exists | `cli/run.py` (1694 lines): `train/core-train/from-config/search/compare/verify/pareto/portfolio/list/benchmark`; SQLite `--db`; `verify` runs n seeds → per-seed JSONL with CI |
| Parity CLI exists | `cli/parity.py` → `biopl-parity` (`pyproject.toml:134`) |
| Repro/report CLIs exist | `biopl-repro-check`, `biopl-report`, `biopl-registry-audit`, `biopl-scientist`, `biopl-failure-manifesto` |
| HPO is Optuna (GridSampler available) | optuna 4.7; `hyperopt/{optuna_bridge,eval_tiers,search_space}.py` |
| Evaluation tiers + budgeting exist | `hyperopt.eval_tiers`: `PatientLevel`, `EVALUATION_TIERS`, `estimate_total_time` |
| All parity-arm models registered | `backprop_mlp`(looped_mlp:282), `eqprop_mlp`(:43), `standard_fa`(fa:835), `diff_target_prop`(target_prop:36), `forward_forward`(:43), `pepita`(:193), `deep_hebbian`(hebbian:76), `three_factor_hebbian`(:300), `neural_cube`(neural_cube:37) |
| 61 campaign unit tests pass | `uv run pytest tests/unit/campaign` → 61 passed |

---

## 4. Design Invariants (non-negotiable)

1. **No parallel experiment stack.** The staircase is a thin layer *over* `cli/run.py` +
   `hyperopt` + `cli/parity.py`; it never re-implements training loop, grid, storage, or report.
2. **One source per fact.** Task geometry (`resolve_task`), seeding (`seed_everything`),
   config identity (`config_key`), statistics (`validation/statistics.py`).
3. **Runner never prunes.** Pareto / effect sizes are computed from completed probes by the
   reporter only.
4. **Zero `getattr(field, fallback)`** in the layer — every field required-or-defaulted in
   schema.
5. **Every correctness claim is gate-enforced, not asserted** (§8 "Definition of Runnable").
6. **Grid is a sampler** — implemented as Optuna `GridSampler` via `hyperopt`, so Bayesian /
   AutoScientist producers drop in later with zero architecture change.

---

## 5. Glossary

| Term | Definition |
|------|-----------|
| **Campaign** | One YAML document = ordered list of **Stages**. |
| **Stage** | One rung: a **Task**, a **config grid**, a seed count, a **PassRule**. Maps onto `hyperopt.eval_tiers.PatientLevel`. |
| **Probe** | `(model, task, config, seed)` → one trainings run → a metrics record. Training happens only in `cli` (CoreTrainer path). |
| **ProbeResult** | Normalized per-probe metrics record (schema in §6.2). |
| **Verdict** | **PASS / REJECT** for a model after a Stage. |
| **Survivor** | A model that **PASS**ed every preceding Stage; only survivors run the next. |
| **Run** | One Campaign execution → a **Report**. |
| **Report** | Append-only JSONL of probes = artifact + resume index. |
| **Task registry** | `name → TaskSpec` in `domains/registry.py` (single source of task facts). |
| **Baseline** | Frozen reference (e.g. `backprop_mlp`) the reporter compares against. |
| **ProbeDriver** | Thin adapter over the existing training path (default = CoreTrainer via `cli`). |
| **ConfigProducer** | Adapter over the existing Optuna/`hyperopt` sampling path (default = `GridSampler`). |

Avoided as *new* names (they exist as reuse targets, RESTRICTED to underlying infra):
*tier*→`PatientLevel`, *study*→`hyperopt`, *SQLite*→`cli.run --db`; and dropped outright:
*level, gate, sub-study, sample, digits-fail, snapshot*.

---

## 6. Component Specifications (the full detail, preserved)

### 6.1 Schema (`experiment/schema.py` — migrated + rewritten from `campaign/schema.py`)

```yaml
meta:      { name: parity_cifar10_mlp, created: "2026-08-05" }

compute:
  device: auto                 # auto|cpu|cuda:0
  num_workers: 0
  track:   { flops: true, memory: true, energy: false }

arms:
  mlp:
    max_params: 210_000
    models: [backprop_mlp, eqprop_mlp, neural_cube, deep_hebbian,
             three_factor_hebbian, standard_fa, diff_target_prop,
             pepita, forward_forward]

stages:                        # the staircase; each maps to a PatientLevel
  - { name: smoke,          task: xor,          epochs: 3,  seeds: 1,
      configs: { hidden_dim: [16, 32], num_layers: [2] },
      pass: { acc: {op: ">=", value: 0.90}, min_seed_ok: 1 } }
  - { name: digits,         task: digits,       epochs: 5,  seeds: 5,
      configs: { hidden_dim: [64], num_layers: [1] },
      pass: { acc: {op: ">=", value: 0.95}, epoch_time_s: {op: "<=", value: 120} } }
  - { name: mnist,          task: mnist,        epochs: 10, seeds: 5,
      configs: { hidden_dim: [64, 128, 256], num_layers: [1, 2, 4] },
      pass: { acc: {op: ">=", value: 0.98} } }
  - { name: fashion_mnist,  task: fashion_mnist,epochs: 10, seeds: 5,
      configs: { hidden_dim: [64, 128, 256], num_layers: [1, 2, 4] },
      pass: { acc: {op: ">=", value: 0.90} } }
  - { name: parity,         task: cifar10,      epochs: 30, seeds: 10,   # n≥10 floor
      configs: { hidden_dim: [64, 128, 256], num_layers: [1, 2, 4] },
      baseline: backprop_mlp,
      matched_by: { equal_budget: max_params,
                    reported: [wall_time_s, forward_flops, backward_flops, settling_steps] },
      energy: [gpu_tdp_x_util, op_count],        # RESEARCH §0.1: report both
      pass: {} }

reproducibility: { seed: 42, capture_env: true }
```

Rules: task geometry inherited from the registry (never redeclared except epochs/seeds/
configs); every runner field required-or-defaulted (no `getattr`); `pass` rules may carry
`aggregate` (default `median`); evidence stages (`baseline:` present) **reject `seeds < 10`**
at validate time.

### 6.2 ProbeResult

```python
@dataclass(frozen=True, slots=True)
class ProbeResult:
    model: str
    task: str
    config: dict[str, object]   # noun
    config_key: str             # content hash of config, for idempotence
    seed: int
    status: str                 # "ok" | "error"
    final_acc: float = 0.0
    final_train_loss: float = 0.0
    epoch_time_s: float = 0.0
    param_count: int = 0
    forward_flops: int = 0
    backward_flops: int = 0
    peak_memory_mb: float = 0.0
    wall_time_s: float = 0.0
    error: str = ""             # message when status == "error" (failure manifesto)
```

`run_verify` (existing) already produces the per-seed source records; the layer normalizes
them into `ProbeResult` once. `param_count` comes from `experiment/param_estimator.py`
(migrated from `campaign/`).

### 6.3 PassRule & Verdict (no eval)

```python
@dataclass(frozen=True, slots=True)
class MetricRule:
    metric: Literal["acc", "epoch_time_s", "loss", "flops", "memory"]
    op: Literal[">=", "<=", ">", "<"]      # "==" excluded (float trap)
    value: float
    aggregate: Literal["median", "mean", "min"] = "median"

class Verdict(StrEnum):
    PASS = "PASS"
    REJECT = "REJECT"
```

A model **PASSES** a Stage iff, for **every** `MetricRule`: the `aggregate` over its seeds
satisfies the rule **and** `ok`-seed count ≥ `min_seed_ok` (default 1). Non-finite/errored
seeds never satisfy `>=`. `max_params` enforced at schedule time via `estimate_param_count`.

### 6.4 ProbeDriver (flex seam over the existing path)

```python
@runtime_checkable
class ProbeDriver(Protocol):
    """Narrow adapter over the existing training path."""
    def train(self, *, model, task, config, seed, device,
              epochs, track, checkpoint_dir) -> list[TrainingMetrics]: ...
```

`CoreTrainerDriver` owns the `CoreTrainer`/`cli.run.run_verify` integration; `run_probe` is
the single normalization point. `TrainingMetrics` already carries `train_accuracy`,
`val_accuracy`, `epoch_time`, `forward_flops`, `backward_flops`, `peak_memory_mb`,
`train_loss` (trainer.py:186-204). Future NumPy/Triton/spiking engines implement this
interface — no core change.

### 6.5 ConfigProducer (grid as a sampler, through `hyperopt`)

```python
@runtime_checkable
class ConfigProducer(Protocol):
    def schedule(self, stage: Stage, survivors: list[str]) -> Iterator[ProbeWork]: ...

@dataclass(frozen=True, slots=True)
class ProbeWork:
    model: str
    config: dict[str, object]
    config_key: str

class HyperoptGridProducer:
    """Grid = hyperopt.create_study(sampler=GridSampler(search_space)).

    Reuses hyperopt's study/resume machinery. Deterministic (fixed grid order, seeded
    study); probe count enumerable from the GridSampler search space (exact `plan`);
    resume via study.ask()/tell() + the content-addressed Report (skip finished probes).
    
    **Implementation note**: the study's objective must call `trial.suggest_categorical(name, choices)`
    for each grid column so that GridSampler emits the enumerated configs (not empty `{}`).
    """
```

### 6.6 StaircaseRunner (the thin layer)

```python
class StaircaseRunner:
    def __init__(self, campaign, report,
                 driver: ProbeDriver = CoreTrainerDriver(),
                 producer: ConfigProducer = HyperoptGridProducer()): ...

    def run(self) -> None:
        survivors = self._initial_models()
        for stage in self.campaign.stages:
            outcomes = self._run_stage(stage, survivors)   # schedule -> probes -> verdicts
            survivors = [o.model for o in outcomes if o.verdict is Verdict.PASS]
            self.report.record_stage(stage.name, outcomes)
```

`_run_stage`: for each `ProbeWork` from `producer.schedule`, run `driver`/`run_probe` (via
`cli.run.search`+`verify`), record the `ProbeResult`, then compute verdicts. That is the
entire layer. Budget comes from `hyperopt.eval_tiers` (`estimate_total_time`) + grid
cardinality.

### 6.7 Report, Resume, Idempotence

Append-only JSONL keyed by `(stage, model, config_key, seed)` = artifact + resume index.
On launch, skip probes recorded `status != "error"` → crash-resume, incremental extension,
exact reproducibility. Storage reuses `cli.run --db` (SQLite) for the Optuna study layer and
the JSONL Report for the experiment trace; the JSONL-vs-SQLite divergence from RESEARCH §4.3
is **declared** (both satisfy resume/versioning; documented, not silent).

### 6.8 Reporter

Post-processing over the Report (run via existing `biopl-report`): parity tables (accuracy
mean±bootstrap-CI, param_count, epoch_time_s, flops/sample, peak_memory_mb vs baseline;
Cohen's d, Cliff's δ), Pareto frontier, failure manifesto. Uses `validation/statistics.py`
and `hyperopt.pareto`/`cli.run.pareto`.

---

## 7. Correctness Gates (RESEARCH §0.1 / §5.2 / §5.3)

1. **Statistics-first**: `validation/statistics.py` (bootstrap percentile+BCa, Cohen's d,
   Cliff's δ, Benjamini-Hochberg, power) is built **before** the reporter — resolves
   RESEARCH §5.3's circular-dependency note.
2. **Gradient-equivalence P0**: `validation/gradient_check.py` — **promoted from** the
   existing finite-difference helpers in `tests/integration/test_gradient_equivalence.py`
   (`_finite_diff_gradient`, `_check_gradient_equivalence`, per-family thresholds), not
   built from scratch; the gate runs per family before a model is admitted to the parity
   tier (RESEARCH §5.2), invoked by `biopl-repro-check`.
3. **n≥10 seed floor** on `baseline:` stages (validate-time).
4. **Compute-matched contract** (`matched_by`) required on parity stages; disclosed in report
   (RESEARCH §0.1's FLOPs-matching ambiguity).
5. **Dual energy model** (`gpu_tdp_x_util` + `op_count`/Horowitz) recorded per probe.

---

## 8. Reuse / Retire / Dedup Map

**Reuse (existing, extended not re-created):**
- `cli/run.py` — `search`, `verify` (n seeds + per-seed JSONL + CI), `pareto`, `compare`, `portfolio`, `--db`.
- `cli/parity.py` (`biopl-parity`), `cli/repro.py` (`biopl-repro-check`), `biopl-report`.
- `hyperopt/` — `create_study`, `GridSampler`, `eval_tiers.PatientLevel`, `estimate_total_time`, `run_single_trial_task`.
- `core/trainer.py` — `CoreTrainer`, `TrainingMetrics`.

**Dedup (delete in favor of one source):**
- `_TASK_DIMS` ×2 (`cli/parity.py:33`, `demo/runner.py:119`) and `schema.py` geometry chains → `resolve_task`.
- `set_global_seed` → folded into `seed_everything(seed, device)` (`utils.py`).
- Freeform `eval()` constraints (`search_space.py:222`) → `PassRule`.

**Retire (`git rm`, tracked; repoint entry points):**
- `campaign/` — migrate `schema.py` → `experiment/schema.py`, `param_estimator.py` →
  `experiment/param_estimator.py`; delete `executor.py`, `runner.py`, `tiers.py`,
  `search_space.py`, `logger.py`, `cli.py`. Repoint `biopl-run` (`pyproject.toml:128`) off
  `campaign.cli:main`. Mirror in `tests/unit/campaign/` (migrate the 61 passing tests that
  still apply).

**New (only genuinely new code):**
- `experiment/{schema,param_estimator,probe,producer,staircase,report,__init__}.py`
- `validation/{statistics,gradient_check}.py`

---

## 9. Tooling Gate (Step 0 — the blocker)

1. `pyproject.toml:187` — invalid `line-too-long` → `E501` (ruff can't run until fixed).
2. Coverage floor → single source in `pyproject.toml`, set to `85` (align pre-commit).
3. Document Pyright's relaxed-but-correctness-hard profile (don't claim "strict").
4. Legacy `except X, Y:` → `except (X, Y):` at migrated `param_estimator.py` (and the old
   `tiers.py`).
5. Seed-API consolidation (one `seed_everything`).

---

## 10. CLI Surface (extend existing; add nothing new)

```
biopl-parity            # EXISTING — extend to drive the parity campaign stages + emit Report
biopl-run validate      # repointed biopl-run: schema + task-registry validation + gates
biopl-run plan          # probe count (grid) + estimate_total_time (budget), dry
biopl-run run           # idempotent staircase execution (resume by default)
biopl-report            # EXISTING — parity/Pareto/failure from the Report
biopl-repro-check       # EXISTING — nightly gate (gradient-equivalence + run-resume no-op)
```

---

## 11. Deferred (explicit re-entry paths)

| Capability | Re-entry path |
|------------|---------------|
| Bayesian / AutoScientist sampling | `TPESampler`/`NSGAIISampler` `ConfigProducer` (same §6.5 interface) |
| Concurrent probe scheduling | flag behind `ProbeDriver`/`ConfigProducer`, after determinism proven |
| Non-`CoreTrainer` engines (NumPy/Triton/spiking) | `ProbeDriver` impl, no core change |
| Conv / language / cross-domain parity (full §0.1 breadth) | new Stages + registry tasks; declared post-MLP milestone |

---

## 12. Implementation Sequence (dependency-ordered, each step verifiable)

```
0. Tooling gate           1 hr   §9              (verify: ruff/pyright now pass)
1. statistics.py          3 hr   §7#1            (verify: unit tests green)
2. gradient_check.py      2 hr   §7#2            (verify: gate passes on registered models)
3. Task registry          1.5 hr §8-dedup        (verify: parity.py resolves via resolve_task)
4. Seed consolidation     0.5 hr §8-dedup        (verify: no set_global_seed callers left)
5. Schema rewrite         2 hr   §6.1            (verify: validate rejects seeds<10, unknown task)
6. producer.py            1.5 hr §6.5            (verify: plan enumerates exact grid; GridSampler objective emits configs)
7. StaircaseRunner        2.5 hr §6.6            (verify: run + re-run no-op resume)
8. Wire CLIs + retire     2.5 hr §8/§10          (verify: biopl-parity/report/repro-check work; gradient gate integrated)
9. E2E overnight smoke    1 hr   §13
                   Total ≈ 15 hr, of which <9 hr is new code
```

---

## 13. Definition of "Runnable" (verified, not claimed)

1. `ruff check` / `format --check` pass on **new code** (`experiment/`, `validation/`, `domains/`).  2. `pyright` zero errors on
`experiment/`+`validation/`.  3. `biopl-run validate` passes: every `stages[].task` resolves
via registry; evidence stages enforce `seeds≥10` + `matched_by` + dual-energy.
4. `biopl-run plan` prints exact probe count + `estimate_total_time`.  5. `biopl-run run`
trains every scheduled probe and appends to the Report.  6. `biopl-run run` again = no-op
for finished probes.  7. `biopl-report` renders parity/Pareto/failure.  8. Every parity-tier
model passed the gradient gate.  9. `uv run pytest --cov` passes at floor 85 on new code; `biopl-repro-check`
rail runs the parity ladder for 1 epoch nightly.

---

## 14. Research Alignment

Serves Phase 0.1 (parity via `biopl-parity`+staircase), 0.3 (reproducibility:
`seed_everything`, `capture_env`, `biopl-repro-check`), 5.2 (gradient gate as P0), 5.3
(statistics-first resolves the circular-dependency note). `ConfigProducer` reuses `hyperopt`
so Phase 4 AutoScientist is a drop-in sampler producer; `ProbeDriver` is the seam for
Phase 8 engines. Each declared divergence (MLP-only Phase-0 scope, JSONL-vs-SQLite,
compute-match tradeoff) is explicit so the evidence ladder stays honest.

---

*Canonical. Every decision here is either verified against the repo or explicitly declared
as a divergence. This document is the implementation contract; changes to it are changes to
scope, not fixes. Stop iterating on successors; iterate on this file.*
