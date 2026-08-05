# FIX2b.md — The Experiment Layer, Massively Simplified

**Status**: Successor to FIX2a.md. Discards FIX2a's HPO/Optuna machinery and its tier
naming tangle in favor of one linear, declarative staircase. Every byte of the old
design that was sound is kept; everything that added complexity without buying
evidence is cut.

**Goal**: A YAML-driven experiment layer that produces *publication-grade parity
evidence* for bio-plausible learning vs backprop — runnable **unattended overnight** —
with a small, auditable, deduplicated core.

---

## 1. The Simplification Thesis

FIX2a suffered from over-engineering that produced bugs and confusion:

| FIX2a machinery | Why it's cut in FIX2b | FIX2b replacement |
|-----------------|------------------------|-------------------|
| Optuna HPO (NSGA-II/III, `n_trials`, `n_startup`, sampler, directions) | Multi-objective HPO optimizes for the *reporting* tiers, not the early tiers. It multiplied compute (200×5 seeds) and broke the budget math. | **Explicit config grids** in YAML. Deterministic, auditable, budget-computable. HPO is a *deferred optional sampler*, never the core. |
| Multi-objective pareto pruning, knee detection, `prune_worse_than_pareto` | Reporting-only concern snuck into the runner. | Reporter computes Pareto from completed probes; runner never prunes. |
| Per-`(arm, model, task)` Optuna SQLite studies + resume | A whole DB subsystem to support resume. | **Content-addressed report**: probe already on disk ⇒ skip. No DB. |
| `tiers.py` hardcoded geometry (`TIER0_INPUT_DIM` …) | Duplicates task facts. | **Task registry** is the single source of task geometry. |
| Freeform `eval()` constraint language | Unsafe, unparseable, name-collision footgun (FIX2a §3 example crashed). | **Structured `PassRule`** + a numeric `max_params` budget. No code exec. |
| `value lock` `» "tier0" / "tier0.5" / Tiers L0–L6 / gates / digits-fail'` | Parallel naming taxonomies, confusing both humans and code. | One word: **Stage**. One verdict vocabulary: **PASS / REJECT**. |
| `min_accuracy` via `getattr` fallback, batch_size/num_workers hardcoded | Violated "Everything in YAML." | Required schema fields with no fallback hacks. |
| `track_energy/flops/memory` hardcoded `False` in gates | Made the parity axes unmeasurable. | Tracking flags flow from schema; probes **record** all axes. |

**Net effect**: the runner is ~5 small modules, no external optimizer dependency, no
database, no eval. What remains is only what generates or aggregates evidence.

---

## 2. Glossary (Chosen Terminology)

Pick a term once; use it everywhere.

| Term | Definition |
|------|-----------|
| **Campaign** | One YAML document describing an ordered list of **Stages** (the "what to measure"). |
| **Stage** | One rung of the staircase: one **Task**, a **config grid**, a seed count, and a **PassRule** verdict. |
| **Probe** | One measurement: `(model, task, config, seed)` → one training run → a `ProbeResult` (metrics record). Probes are the **only** place training happens. |
| **ProbeResult** | The normalized metrics from a probe: `final_acc`, `final_train_loss`, `epoch_time_s`, `param_count`, `forward_flops`, `backward_flops`, `peak_memory_mb`, `seed`, `status`. |
| **Verdict** | How a model is judged after a Stage: **PASS** or **REJECT**. |
| **Survivor** | A model that has received **PASS** on every preceding Stage. Only survivors run the next Stage. |
| **Run** | One execution of a Campaign → produces a **Report**. |
| **Report** | The outputs: a JSONL trace of every probe + summary tables (per-stage survivors, parity table). The Report is the single artifact; it is also the resume index. |
| **Task registry** | A mapping `name → {loader, input_dim, num_classes, flatten}` in `data/`. The single source of task facts; YAML `Task` only overrides epochs. |
| **Baseline** | The frozen reference (e.g. `backprop_mlp` on the parity stage) the reporter compares others against. |

Deliberately **avoided**: *tier, level, gate, sub-study, trial, study, sample,
digits-fail, snapshot*. They were overloaded or duplicated the above.

---

## 3. Reversion & Clean-Up (Step 0)

The committed `HEAD` already contains a sound, tested foundation
(`schema.py`, `param_estimator.py`, `search_space.py`, `logger.py`, plus `runner.py`,
`tiers.py`, `cli.py`). The working-tree changes are genuine improvements (TypeIs
logger, geometry constants, legacy-`except` fixes, 11 new tests — **61 pass**, ruff clean
on all committed files). The broken `executor.py` is **untracked**.

**Action (no `git revert` needed):**
```
rm bioplausible/campaign/executor.py          # untracked debris — delete, don't commit
git add bioplausible/campaign tests/unit/campaign   # keep the sound foundation + fixes
git commit -m "refactor(campaign): harden FIX2a foundation (no executor)"
```
No history is rewritten; the clean foundation is preserved wholesale. The plan going
forward **edits** the kept modules and **adds** the new experiment layer described §5,
§6, §7.

---

## 4. Internal Bioplausible Changes (facilitate the experiment layer)

These are small, high-value changes outside `experiment/` that remove duplication the
experiment layer would otherwise have to re-implement.

### 4.1 Task registry (single source of task facts)
`data/vision.py` already knows every task via `get_vision_dataset`/`_get_dataset_class`.
Expose it:
```python
# data/vision.py
SUPPORTED_TASKS: frozenset[str] = frozenset({
    "mnist", "fashion_mnist", "cifar10", "cifar100", "kmnist",
    "svhn", "digits", "xor", "spiral", "circles",
})

@dataclass(frozen=True, slots=True)
class TaskSpec:
    input_dim: int                # None → dynamic (e.g. language)
    num_classes: int | None
    flatten: bool
    loader: Callable[..., Dataset]   # resolves name → dataset factory
```
A `resolve_task(name) -> TaskSpec` looks up geometry. **`TaskSpec` is authoritative**;
an arm that omits geometry inherits from the task, removing `arm_input_dim`+
`arm_output_dim` heuristic chains (§A3 of the analysis).

### 4.2 Typed training face (isolate CoreTrainer)
The experiment layer must not reach into CoreTrainer's dict-based config hacks. Add a
thin, fully-typed adapter:
```python
# experiment/probe.py (owns the only CoreTrainer import in the layer)
def run_probe(
    *, model: str, task: str, config: dict[str, object],
    seed: int, device: str, epochs: int,
    track: TrackFlags, checkpoint_dir: Path,
) -> ProbeResult: ...
```
It constructs `TrainerConfig` (mapping config→optimizer kwargs etc. in one place), calls
`CoreTrainer(config).fit()`, and **normalizes** the returned `list[TrainingMetrics]`
into a `ProbeResult` (handling `train_accuracy` absence, `val_accuracy` fallback, and
flops/mem extraction exactly once — no `getattr` soup in every caller).

### 4.3 Global seeding (RESEARCH §0.3)
Add `utils/reproducibility.py::seed_everything(seed)` wrapping torch/numpy/random/cuda/
cudnn so `n_seeds` reproduction is a one-liner. Replace inline `torch.manual_seed` in
`core/trainer.py` with a call to it.

### 4.4 Arm vs task geometry contract
An `Arm` may set `input_shape`/`input_dim` **only** to override the task; otherwise
`resolve_geometry(arm, task)` returns the task's `TaskSpec`. One function, no fallback
chains, validated at load time (§16.1 analysis).

---

## 5. Schema (Single Source of Truth, No Fallback Hacks)

```yaml
# experiments/campaign_parity_cifar10.yaml
meta:
  name: parity_cifar10_mlp
  created: "2026-08-05"

compute:                      # ALL fields required or defaulted in schema — no getattr
  device: auto                # auto|cpu|cuda:0  (resolves cuda if available)
  num_workers: 0
  track:
    flops: true               # parity needs all three ON
    memory: true
    energy: false

arms:
  mlp:                        # geometry optional: inherits from task by default
    max_params: 210_000
    models: [backprop_mlp, eqprop_mlp, neural_cube, deep_hebbian,
             three_factor_hebbian, standard_fa, diff_target_prop,
             pepita, forward_forward]
  # conv arm deferred to a later campaign (MLP-first, FIX2a §9)

stages:                       # THE staircase — linear, ordered
  - name: smoke
    task: xor                 # geometry from task registry
    epochs: 3
    seeds: 1
    configs:                  # explicit grid (no HPO in v1)
      hidden_dim: [16, 32]
      num_layers: [2]
    pass: { acc: {op: ">=", value: 0.90}, n_seed_agree: {op: ">=", value: 1} }

  - name: digits
    task: digits
    epochs: 5
    seeds: 5
    configs:
      hidden_dim: [64]
      num_layers: [1]
    pass:
      acc: {op: ">=", value: 0.95}
      epoch_time_s: {op: "<=", value: 120}

  - name: mnist
    task: mnist
    epochs: 10
    seeds: 5
    configs: { hidden_dim: [64, 128, 256], num_layers: [1, 2, 4] }
    pass:
      acc: {op: ">=", value: 0.98}

  - name: fashion_mnist
    task: fashion_mnist
    epochs: 10
    seeds: 5
    configs: { hidden_dim: [64, 128, 256], num_layers: [1, 2, 4] }
    pass:
      acc: {op: ">=", value: 0.90}

  - name: parity                # the evidence tier
    task: cifar10
    epochs: 30
    seeds: 5
    configs: { hidden_dim: [64, 128, 256], num_layers: [1, 2, 4] }
    baseline: backprop_mlp      # frozen reference for the reporter
    pass: {}                    # no verdict — everything that finished is reportable

tasks: []                       # empty: all task facts come from the registry

reproducibility:
  seed: 42
  capture_env: true
```

**Schema rules:**
- `Task` geometry is **inherited from the registry** (`stages[].task`), never redeclared
  in YAML except `epochs`/`seeds`/`configs`.
- Every field the runner needs is a **required schema field with a default** — zero
  `getattr(x, field, fallback)` anywhere.
- `configs` is an explicit cross-product grid. Each Stage runs
  `|models| × |configs| × seeds` probes, **compute-computable before launch**.

---

## 6. Probe & Stage Harness

### 6.1 ProbeResult (the normalized measurement contract)

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

### 6.2 PassRule (verdict, no eval)

```python
@dataclass(frozen=True, slots=True)
class MetricRule:
    metric: Literal["acc", "epoch_time_s", "loss", "flops", "memory"]
    op: Literal[">=", "<=", ">", "<", "=="]
    value: float
```
A model **PASSES** a Stage iff, for **every** rule R:
- `aggregate` of the probe metric over that model's seeds (default `median`) satisfies R,
- **and** the count of successful (`status == "ok"`) seeds ≥ `min_seed_ok` (default 1).

Non-finite or errored seeds never satisfy `>=`. `max_params` budget is enforced at
**schedule** time: any config whose `param_count > arm.max_params` is skipped, not trained.

### 6.3 Runner (small, linear)

```python
# experiment/staircase.py
class StaircaseRunner:
    def __init__(self, campaign: Campaign, report: Report): ...

    def run(self) -> None:
        survivors = self._initial_models()
        for stage in self.campaign.stages:
            outcomes = self._run_stage(stage, survivors)   # probes -> verdicts
            survivors = [o.model for o in outcomes if o.verdict is Verdict.PASS]
            self.report.record_stage(stage.name, outcomes)
```

**`_run_stage`** loops `model × config × seed`, calls `run_probe`, writes each
`ProbeResult` to the Report, then computes verdicts. That's the entire runner.

---

## 7. Report, Resume, and Idempotence

**Report** = append-only JSONL of `ProbeResult` events, one per probe, keyed by
`(stage, model, config_key, seed)`. This is both the data artifact **and** the resume
index — no Optuna DB, no separate state file.

**Resume** = on launch, scan the existing Report; skip any probe whose key is already
recorded with `status != "error"`. This gives:
- crash-resume for free (re-run the same command),
- incremental extension (add a Stage to YAML, re-run — only the new stage computes),
- exact reproducibility of already-computed probes (content-addressed config).

Satisfies RESEARCH §4.3 "campaign persistence & resume" more simply than a database.

**Reporter** (post-processing, separate from runner):
- *Parity tables*: per arm/model/task — `accuracy mean±bootstrap-CI`, `param_count`,
  `epoch_time_s`, `flops/sample`, `peak_memory_mb`, vs `baseline` (Cohen's d, Cliff's δ).
- *Pareto frontier*: accuracy vs param_count vs time across the parity stage's probes
  (computed from results; runner never prunes).
- *Failure manifesto* (§16.6): aggregate `status == "error"` propagation raises.

All statistics live in `bioplausible/validation/statistics.py` (bootstrap CI, Cohen's d,
Cliff's delta, BH correction) — a pure, separately-testable module imported by the
reporter only.

---

## 8. CLI Surface

```
biopl-run validate  --config <yaml>            # schema + task-registry validation
biopl-run plan      --config <yaml>            # print probe count + budget, dry (no compute)
biopl-run run       --config <yaml>            # execute; idempotent (resume by default)
    --device cpu|cuda                          # override compute.device
    --stage NAME                               # run only one stage
biopl-run report    --config <yaml>            # render parity/Pareto/failure tables
```
No `gates`/`run` alias split, no `--tier` enum — the "staircase" is just the ordered
`stages:` list. `plan` replaces `dry-run` and pre-checks budget before any GPU time.

---

## 9. Statistics & Matching Axes (kept, isolated)

- The parity **axes** (flops, wall-time, memory) are **recorded per probe** (tracking ON
  via `compute.track`), not hardcoded off. The reporter computes "which axis matches
  baseline" and flags it; the runner only records.
- `n_seeds` aggregation uses the median per model, and the reporter adds bootstrap CIs /
  effect sizes — the runner stays aggregation-agnostic.

---

## 10. What's Deferred (explicitly out of v1)

| Capability | Why deferred | Re-entry path |
|------------|--------------|---------------|
| HPO / AutoScientist sampling | Not needed for early tiers; over-engineered | Optional `configs: {sampler: ...}` later — probes are already config-agnostic, so a sampler is a drop-in `configs` producer, no architecture change |
| Cross-domain transfer (L6) | Separate milestone | New stage + task from registry |
| Matching-axis *enforcement* | Reporter flags first; enforce later | Pure logic in reporter |
| Spiking / manual-protocol sub-studies | MLP-first (§9) | Add `protocol` field to Stage when needed |

Deferral is deliberate: **the core never grows; stages/probes/configs carry all
variability.**

---

## 11. Module Map (final)

```
bioplausible/
  data/vision.py            # + SUPPORTED_TASKS, TaskSpec, resolve_task (internal change)
  utils/reproducibility.py  # seed_everything (internal change)
  campaign/
    schema.py               # KEEP, simplify: stages/pass-rule/grid; drop min_accuracy
    param_estimator.py      # KEEP (clean, use estimate_param_count for budget)
    search_space.py         # KEEP, shrink: keep FloatRange/IntRange/Choice; retire constraints-eval
    logger.py               # KEEP (typed JSONL)
    probe.py                # NEW: run_probe + ProbeResult + TrainerConfig mapping
    staircase.py            # NEW: Stage harness + StaircaseRunner
    report.py               # NEW: append-only JSONL + resume index + reporter
    cli.py                  # EDIT: validate/plan/run/report
  validation/
    statistics.py           # NEW: bootstrap CI, Cohen's d, Cliff's delta, BH
  # runner.py, tiers.py     # ABSORBED into staircase.py/probe.py; delete both
```

**Deduplication wins**: `runner.py`(run_gates/tier logic) + `tiers.py`(gate functions)
+ `executor.py`(broken) → collapse into `probe.py` + `staircase.py`. `arm_input_dim`
heuristic chains → `resolve_task`. Config→`TrainerConfig` mapping→`run_probe`. Reported
stats→`validation/statistics.py`. Eval-constraints→structured `PassRule`.

---

## 12. Implementation Sequence (simplified)

```
1. Internal changes           2 hr   §4: task registry, seed_everything, resolve_task
2. Schema rewrite             2 hr   §5: stages/pass-rule/grid; drop HPO fields
3. probe.py                   2 hr   §6.1,§6.2: run_probe + ProbeResult
4. report.py                  2 hr   §7: JSONL + resume index
5. staircase.py               3 hr   §6.3: stage harness + verdicts
6. cli.py (validate/plan/run) 2 hr   §8: wire plan preview + idempotent run
7. Delete runner.py/tiers.py/ 30m    absorb into probe/staircase
8. Reporter + statistics      3 hr   §9: parity tables, Pareto, failure manifesto
9. E2E overnight smoke        1 hr   §13
                         Total ≈ 17–18 hr
```

---

## 13. Definition of "Runnable" (verified, not claimed)

A FIX2b campaign is runnable when **all** hold:
1. `biopl-run validate` passes and validates every `task` against the registry.
2. `biopl-run plan` prints an exact probe count and a compute time estimate.
3. `biopl-run run` trains every scheduled probe and appends them to the Report.
4. `biopl-run run` **again** is a no-op for already-finished probes (resume works).
5. `biopl-run report` renders parity/Pareto/failure outputs from the Report.
6. `uv run pytest` (campaign + validation) passes; `ruff check`/`format`/`pyright` clean.

Nothing is claimed runnable before 1–6 are demonstrated end to end.

---

## 14. Research Alignment

This directly serves RESEARCH.md's early milestones (Phase 0.1 parity suite, 0.3
reproducibility, Phase 1 experiments) while **removing** the machinery that made FIX2a
un-runnable overnight. The staircase (`smoke → digits → mnist → fashion → parity`) is the
Phase-0 evidence ladder; `biopl-run report` is the "1-command publication output"
milestone. HPO/AutoScientist (Phase 4) slots in later as an optional `configs` producer
without touching the core.

---

*The staircase, made small enough to trust, and large enough to matter.*
