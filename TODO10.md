# TODO10.md — Active Plan: The Library, Proven Continuously

> **Opened 2026-09-02.** Successor to [TODO9.md](TODO9.md) (R9 stress trials
> landed claim-grade; leftovers parked in the deferred register below).
> Research catalog: [RESEARCH3.md](RESEARCH3.md).
>
> **Identity decision (2026-09-02, v4):** **Computronium is an ML library whose
> every claim is a live demonstration.** Import it, compose a learning system
> from the 6-axis ontology, train it, see it work. v3 said "tests are the
> examples"; v4 completes the move: **tests are the evidence system.** A claim
> stands only while the current code re-demonstrates it, on demand, in under a
> minute. Verification is continuous, not archival. Development depends on
> capabilities the suite shows — never on numbers stored in files. Results too
> subtle to demonstrate live are not library claims; they belong to the
> research track (RESEARCH3), where registered scale and preregistration are
> the right instruments.
>
> **Prime directive:** *Nothing is claimed that the suite does not re-show at
> HEAD. The demo suite is the proof; the README quotes it; everything else is
> history or hypothesis.*
>
> **Why not artifact-centric (v3, superseded same day):** v3 built figures and
> captions on registered JSON artifacts. Audit (2026-09-02) found: 4 of 5
> result files record no git-commit provenance; the Z3 optimizer-hygiene
> defect (RESEARCH3 PR-1 — carried Adam momentum across the freeze boundary)
> was fixed in code at some unrecorded point relative to the registered runs;
> and standing type-level anomalies sit in the exact pipeline modules beneath
> the trials. Any one of these makes "the artifact is ground truth"
> unverifiable. Rather than re-verify the past, the plan re-centers on the
> present: evidence regenerated at HEAD on every commit cannot have a
> provenance problem.
>
> **State:** OPEN — R10.2 in design. All five registered artifacts exist under
> `benchmark_results/` (schemas verified 2026-09-02); they are demoted to
> historical corroboration — context in RESULTS.md, never load-bearing. The
> compositional API has no test that *demonstrates* it. R10.2.1/R10.2.2 were
> pre-flighted 2026-09-02 (one throwaway probe, then reverted — no
> implementation has landed): the audit findings are recorded in R10.2.1,
> R10.2.2, and Register C. D3 and D4 were pre-flighted the same day
> (run_trial/memory-wall probes — throwaway scripts, also reverted): D3 has
> a visible regime (R10.2.4), D4's verdict semantics and depth requirement
> are pinned (R10.2.5). No new experiments are
> commissioned this round. Termination criterion unchanged in spirit: **if
> it works it will be obvious.**

---

## 🎯 The Demonstration Table (what the suite shows, live)

Each row is a library capability. The claim is what the test shows at HEAD —
not what a file once recorded. A reader runs the suite and *watches* each row
demonstrate itself.

| # | Capability | Demo test | What the runner sees | Registered corroboration (history, not claim) |
|---|------------|-----------|----------------------|------------------------------------------------|
| D1 | Six-axis composition is real | `test_demo_compose_6axis.py` | A system composed from all six axes trains; its config round-trips to an identical system; the null-plasticity forward matches the 5-D path | EqProp parity & energy proofs (L4-locked continuously) |
| D2 | One trainer, every credit rule | `test_demo_swap_credit.py` | Three credit rules through byte-identical wiring except one constructor argument — all three learn | Deep-credit registered study |
| D3 | The M-axis swap matters | `test_demo_swap_plasticity.py` | Routing visibly retains what null forgets across a task switch — designed so the gap is seen, not computed | Retention registered study (16 seeds) |
| D4 | The memory profiler is honest | `test_demo_memory_budget.py` | The backprop-profiled arm simply **cannot run** under a tight budget (walled, deterministically); the O(1)-memory arm runs | Memory-budget registered study |
| D5 | Frozen θ is a guarantee, bitwise | `test_demo_z3_frozen_theta.py` | θ's hash identical across the whole freeze→adapt→switch→restore run; restored ψ reproduces stage-A accuracy *exactly* | Z3 registered study |

**Rules of the round:**

- **Visibility standard.** A demo is designed so its effect is *seen* —
  categorical (hashes, verdicts, walls) or large at toy scale. If an effect
  needs statistics to be visible at demo scale, the demo is redesigned for a
  regime where it doesn't — or the claim leaves the front page entirely. A
  guard band survives a noisy contrast; only a redesign or a demotion makes it
  self-evident.
- **Corroboration never carries.** Registered numbers appear only as labeled
  history in RESULTS.md ("the same effect, at preregistered scale, in runs you
  can inspect"). No figure caption, README line, or test docstring leans on
  one.
- **Provenance still recorded.** Every demo/figure run emits its git commit +
  config hash into its output record. Costs nothing; keeps the research track
  honest. (The registered artifacts' provenance gap is recorded, not fixed —
  fixing the past is nobody's job when the present regenerates itself.)

**Sequencing.** **R10.2.1 (API audit) is the first move** — nothing imports
cleanly from the package root until `create_task` and `JointSystemTrainer`
are fixed; R10.2.0's protocol governs throughout. R10.2.2 (flagship) next —
R10.2.7's README block derives from it; R10.2.3–.6 follow, each consuming a
design decision from R10.2.0. R10.1 consumes the run records R10.2.2–.6 emit
(R10.1.2), so figures land after their demos; R10.2.8–.10 land last (drift
lock with the README, gallery CLI with the gallery module). R10.3 rules bind
from the first commit.

**Next capability in line (D6): the substrate axis.** No live demonstration
exists for the S-axis today — the memristive factory is unimplemented
(Register B) and the quantum/neuromorphic substrates are covered only by
property locks. The substrate story is the library's most distinctive pitch;
the memristive-factory pull with a visible IR-drop/noise demo is the
highest-value next row of this table.

---

## 🖼️ R10.1 — The Gallery (figures from live runs, not from files)

Figures are rendered **from the demo suite's own deterministic outputs at
HEAD** — fixed seeds, CPU, current code. The figure is what the test shows,
drawn. Nothing frozen, nothing to re-verify.

- [ ] **R10.1.1** `computronium/visualization/gallery.py` (new package;
  matplotlib is already a core dependency) — figure factory, one pure function
  per figure, each consuming the run records the demo suite emits
  (`run_records`: metrics per step/arm, feasibility verdicts, θ hashes,
  retention curves): `fig_compose_train` (D1 learning curve),
  `fig_credit_swap` (D2, three rules one plot), `fig_plasticity_swap` (D3,
  retention slopes null vs routing), `fig_memory_wall` (D4, feasibility grid),
  `fig_frozen_theta` (D5, per-stage accuracy + θ-hash badge). Each returns
  `(Figure, FigureMeta)`; `FigureMeta` is a frozen slotted dataclass carrying
  the capability id, the demo-test path, the run-record provenance (git
  commit, config hash), and the scope label. One shared style module; no
  per-figure copy-paste.
- [ ] **R10.1.2** Demo runners emit run records deterministically: same seed →
  same record → same figure. The suite already pins seeds and CPU; emission is
  one JSON dump per test (assert-light, recorded even on assertion paths).
- [ ] **R10.1.3** Output to `docs/figures/*.png` + `docs/figures/manifest.json`
  (per-figure: capability id, demo-test path, run-record provenance, scope
  label).
- [ ] **R10.1.4** **Figure lock with teeth.** A lock test regenerates every
  figure and compares **data-layer** checksums (recorded metric values, not
  pixels). A mismatch means one of two things, both caught: the code changed
  what it demonstrates (review the diff, re-pin deliberately) or the demo
  became nondeterministic (a bug — fix it). Either way the gallery cannot
  silently drift from what the code actually does.
- [ ] **R10.1.5** **Registered-scale figures are out of the gallery.** Where a
  preregistered study produced a figure worth keeping (e.g., the depth-50
  cliff — unreachable at demo scale by design), it lives in RESULTS.md's
  corroboration appendix, clearly labeled historical, provenance annotated
  where known, "provenance unknown" where not. The front page carries only
  live demonstrations.
- [ ] **Acceptance:** `comp gallery` on a clean checkout renders all five
  figures from a single suite run; the lock test is green; deleting any demo
  test removes its figure from the gallery (no orphaned claims).

## 🧪 R10.2 — Tests Are the Evidence (the demonstration layer)

Demonstrative integration tests in `tests/integration/`. Named `test_demo_*`
so pytest collects them in the default gate (addopts exclude only
`slow`/`benchmark`/`llm` — demo tests carry no marker). Each ≤50 lines, fixed
seeds, quick-mode task scale, ≤60 s on CPU with `device="cpu"` pinned
explicitly (imp-70: GPU-first defaults silently moved CI locks onto CUDA and
blew their timeouts). Module docstrings carry the narrative (house style:
Google-style, behavior-focused); assertions explain themselves. **They assert
what the runner can see** — and they fail if the API breaks, the property
regresses, *or the demonstration stops being visible*.

- [ ] **R10.2.0** **Demonstration design protocol.**
  - **Two tiers.** `compose_6axis`, `swap_credit`, `swap_plasticity` are the
    simple tier (≤50 lines, straightforward). `memory_budget` and
    `z3_frozen_theta` are the **hard tier** — multi-arm, multi-phase — with
    in-file `_`-prefixed helpers that stay inside the test file so it remains
    self-contained as documentation. **The line budget is a sprawl guard, not
    a cap:** target ≤80 lines, but if a hard-tier test needs ~120 lines to
    stay readable, let it be 120 lines. Readability beats the budget; clever-
    but-comprehensible-never is a defect in a file whose job is to be read.
  - **Design for visibility before calibrating for noise.** First choose the
    demo regime (task, budget, switch structure) that makes the effect large —
    sweep it live during development, pick the regime where the contrast is
    unmistakable, record that choice in the docstring. Guard-band calibration
    (≥10 fixed seeds, pin direction AND a floor near the 5th-percentile
    contrast) is the fallback for effects that resist a visible regime — not
    the default.
  - **Demotion is of claims, not just assertions.** If no visible regime
    exists, the claim leaves the Demonstration Table and the front page; the
    registered study keeps it in the research track. A demo test never asserts
    something a reader cannot watch.
  - **Determinism first.** Seeds fixed; `device="cpu"` explicit; L5
    determinism-lock semantics; no wall-clock-dependent assertions (imp-37
    discipline at test scale).
  - **Positive control rides in the gate.** The instrument self-check
    (`tests/property/test_positive_control.py` — plants an obvious effect,
    requires detection) stays a standing fast-gate member: the microscope
    proves it still sees on every commit. If it ever fails, every demo claim
    is silently suspended by construction — that is the point.
- [ ] **R10.2.1** **API surface audit.** `computronium/__init__.py` exposes the
  compositional API via `__all__` plus the `_LAZY` map (both must be edited in
  lockstep — that is the file's export mechanism, per AGENTS.md): `System`/
  `compose_system`/`compose_joint_system`, ontology primitives,
  `SystemTrainer`, `SystemTrainerConfig`, the factory set, `create_task`. The
  demo tests import **only from the package root** — the tests are the audit.
  (The z3 demo additionally imports its experiment module — the benchmark
  surface is public; the root-only rule scopes to the compositional API.)
  Findings from the 2026-09-02 pre-flight, to fix at pull time:
  (a) `create_task` is missing from root (README's quickstart reaches into
  `computronium.domains.factory`) — add to `_LAZY` + `__all__`;
  (b) `JointSystemTrainer` — the README's "single mathematical center" —
  **does not exist anywhere in the tree**; do NOT export it. The real joint
  training surface is the `JointSystem` protocol (`train_step`/`forward`,
  duck-typed by `SystemTrainer`); R10.2.7 corrects the README table;
  (c) root `NullPlasticity` points at `computronium.state`'s twin class,
  which `compose_joint_system` does **not** special-case (its isinstance
  tuple accepts only `core.plasticity.NullPlasticity` ≡
  `core.joint.transition.NullPlasticity`) — re-point the export so the
  composition hits the delegating `_NullJointSystem` path;
  (d) README's factory table advertises `create_memristive_mlp`, which has
  no implementation in the tree — either it lands as a Register-B pull with
  a demo test, or the R10.2.7 README restructure marks the row Planned.
  Root exports are the contract; backwards compatibility is explicitly none.
- [ ] **R10.2.2** `test_demo_compose_6axis.py` — the flagship. Full six-axis
  composition via `compose_joint_system` (Digital × Recurrent ×
  EnergyMinimization × Null × ThermodynamicContrast × Euclidean; canonical
  home is `computronium/core/system_trainer/` — `joint.py` for composition,
  `factory.py`/`spec.py` for the round-trip pair — always imported via the
  package root), quick-mode task, short `SystemTrainer.fit`. Shows: valid
  metrics (loss ≥ 0, accuracy ∈ [0,1] above chance), config round-trip
  identity, J1 zero-extension (null-plasticity behavior ≡ 5-D dynamics —
  the property lock already lives at
  `tests/property/joint/test_null_equivalence.py`; the demo runs it at
  *train* scale, end-to-end). Emits the D1 run record. *A stranger reads
  this and knows how to build anything.* Pre-flight (2026-09-02) verified
  the regime and call shapes, so implementation starts from facts:
  MNIST quick-mode, 1 epoch, batch 64, hidden `(32,)`,
  `EnergyMinimization(max_steps=5, β=0.5)`, Euclidean step 0.05 → **~19 s
  wall, train_acc ≈ 0.91** — the above-chance assertion is safe with huge
  margin, and MNIST images need a flatten wrapper (mirroring
  `test_quickstart.py`). Round-trip call shape:
  `compose_system_from_configs(**extract_config(system))` — the dict keys
  are exactly the parameter names. J1 demo form: two independent builds
  seeded identically, one 5-D, one 6-axis with Null → identical metric
  dicts (abs_tol 1e-7) and `torch.equal` θ afterwards. The joint
  `to_spec`/`from_spec` round-trip is broken (Register C) — the demo uses
  the 5-D L6 pair, not the joint spec.
- [ ] **R10.2.3** `test_demo_swap_credit.py` — one trainer, three credit rules.
  Same coordinate trained three times with the C-axis swapped (gradient /
  ThermodynamicContrast / RandomProjections) through identical `SystemTrainer`
  wiring. Shows: all three learn; the wiring code is byte-identical across
  arms except the constructor argument. *The comparison is one line.* Emits
  the D2 run record.
- [ ] **R10.2.4** `test_demo_swap_plasticity.py` — the M-axis swap, seen. Null
  vs RoutingPlasticity on the segmented switching stream (R8.3 machinery:
  `create_switching_task` in `experiments/joint/adaptation_efficiency.py`,
  segment-keyed stationary teachers via `evaluate_episode(segment=…)` in
  `core/campaign/evaluation.py`; arm structure patterned on
  `forgetting_trial.py`, compressed). Per R10.2.0: sweep regimes live first,
  pick the one where routing's retention advantage is unmistakable at toy
  scale; calibrate a guard band only if needed. **Budget the sweep as real
  design work** — hours of live iteration, not minutes: the registered study
  needed 16 seeds and d = −1.90 to see this effect; finding a toy-scale
  regime where it is visible by eye is the hard part of this test. If no
  visible regime exists AND no robust band survives calibration, **demote
  for real**: D3's row becomes the one-call-swap capability, the retention
  claim moves to the corroboration appendix, and the test asserts only what
  it can show. Emits the D3 run record
  (per-seed retention curves — the figure's data). *You watch routing
  remember what null forgets.* **Pre-flighted 2026-09-02** (toy `run_trial`,
  3 seeds, dims 8/8, lr 0.03, ~3 s wall): **a visible regime exists** — the
  trial's own default A40/B40 separates per-seed (routing retained ≥ null at
  3/3 seeds: mean 0.278 vs 0.191, d = −1.49, registered direction). At
  A20/B20 the effect **reverses** (null retains more): below mastery the
  comparison is unreadable — the Watch item's ≈0.5 mastery floor, confirmed
  live. The demo must pin A40/B40, **assert the mastery precondition before
  reading retention** (routing masters A slower: 0.28–0.50 vs null
  0.58–0.67), and run the ≥10-seed calibration — the trial costs ~1 s/seed
  at toy dims, so calibration is cheap.
- [ ] **R10.2.5** `test_demo_memory_budget.py` — honest feasibility.
  **Hard tier — budget extra time** (multi-arm). Memory profiler's feasibility
  grid at demo scale — machinery to reuse, not rebuild:
  `experiments/joint/memory_wall.py` (`MemoryAccountedModel`,
  `EnvelopeConfig.check_envelope`, `GradientCheckpointedModel`) and
  `experiments/joint/memory_budget_trial.py`: BPTT-profiled arm walled under a
  tight budget (OOM verdict), O(1)-memory arm feasible. The demonstration is
  **categorical by construction** — `check_envelope` verdicts are
  memory-profile arithmetic; the walled arm does not run. No bands, no
  calibration. Emits the D4 run record (verdict grid). *The profiler tells
  the truth before you train — visibly.* **Pre-flighted 2026-09-02:** the
  verdict tuple is `(violated, reason)` — `(True, reason)` = walled,
  `(False, None)` = feasible — and it is deterministic across repeated
  calls (verified). The wall is a **depth phenomenon** (the registered grid
  is depth 4/16/50): a single-layer toy model fits every budget, so the
  demo composes through `memory_budget_trial`'s depth environments (its
  `_compose`/`_feasibility_grid` path) to make the walled cell actually
  wall.
- [ ] **R10.2.6** `test_demo_z3_frozen_theta.py` — the lifecycle guarantee,
  bitwise. **Hard tier — budget extra time** (multi-phase: freeze → adapt →
  switch → restore → probe). Reuse the retention-arm machinery in
  `z3_fixed_weights.py` (`_run_retention_arm`: ψ-*system* snapshot/restore —
  controller + rule state, not a ψ-vector — `Z3Model.freeze_theta`,
  `_snapshot_rng` (imp-56), fixed probe sets, `theta_sha256` emission) rather
  than re-orchestrating. The demonstration is **categorical**: bitwise θ hash
  equality across the whole run, restored ψ reproducing stage-A accuracy
  *exactly*, above-chance probe accuracy post-switch at the visible regime
  chosen per R10.2.0. Emits the D5 run record. *J2, demonstrated, not just
  locked.*
- [ ] **R10.2.7** **README restructure.** Opens with the ≤10-line composition →
  train → report block **derived from `test_demo_compose_6axis.py`**, then
  factory one-liners (only factories the root actually exports), then the
  gallery figures as *what the suite currently shows*. Ontology/capability/
  architecture tables move below the fold. The "Three Perspectives" table
  inverts: ML Library first and load-bearing; research framed as the program
  that validates the library at scale. The stale `core/system_trainer.py`
  module path is corrected to the `core/system_trainer/` package. No README
  line cites a stored number; claims link tests and figures.
- [ ] **R10.2.8** **Drift lock.** README python blocks are checked verbatim
  against their source test files (small extraction script under `scripts/` +
  lock test under `tests/`). A README block that no longer matches its test
  fails CI — no doc/test drift, by construction.
- [ ] **R10.2.9** **`docs/RESULTS.md` — capabilities first, history second.**
  Front section: per axis (S/G/D/M/C/U), what the abstraction is, the one-line
  swap, the live demonstration, the figure. Back section — clearly labeled
  *Historical corroboration*: the registered studies, their preregistered
  scope, their numbers, their provenance status (including "unknown" where
  archaeology fails), and what they do **not** mean. The front never cites the
  back.
- [ ] **R10.2.10** **`comp gallery`** (`computronium/cli/gallery.py` with
  `main()`, registered as one entry in the `_SUBCOMMANDS` map of
  `computronium/cli/__main__.py` — the dispatcher's lazy-resolution pattern,
  so gallery imports stay out of the CLI's import graph): runs the demo suite
  (or reads its emitted run records from the most recent gate run), renders
  figures + manifest, exits nonzero on any missing/nondeterministic record.
- [ ] **Acceptance:** a stranger copies the README's first block and runs it;
  runs `pytest tests/integration/ -k demo` and watches all five capabilities
  demonstrate themselves; `comp gallery` renders from those same runs;
  **three consecutive green gate runs with zero demo-test flakes** before the
  round closes; every claim on the front page is one they just watched happen.

## 🔒 R10.3 — The Standing Rules

- [ ] **R10.3.1** **No test, no feature.** Every feature ships with an
  integration test that demonstrates it working end-to-end. The test is the
  example.
- [ ] **R10.3.2** **No claim without a live demonstration.** The front page
  (README, gallery, Demonstration Table) carries only claims the suite
  re-shows at HEAD. When a test is removed, flaky, or failing, its claim
  *disappears from the front page automatically* — the system degrades to
  silence, never to stale assertions. Fix the demonstration; the claim
  resumes.
- [ ] **R10.3.3** **Corroboration never carries.** Registered numbers are
  history: labeled, scoped, provenance-annotated, and confined to RESULTS.md's
  back section and the research track. A figure caption or README line that
  leans on a stored number is a defect.
- [ ] **R10.3.4** **Scope honesty.** Demo-scale demonstrations speak for demo
  scale; registered claims live in the research track where preregistration
  governs. Neither borrows the other's clothes.
- [ ] **R10.3.5** **Refutations ship with the same pipeline** — same figure
  factory, same docs, same terms. A demonstration that shows the library
  failing somewhere is as welcome as one that shows it succeeding (the
  constraint sweep's EqProp collapse is the standing example — corroboration
  class until a visible live regime exists, then a demo like any other).
- [ ] **R10.3.6** **Pull rule:** a backlog item is pulled only if it ends in a
  live demonstration, a gallery figure, or a RESULTS.md capability paragraph.
  Infrastructure is justified by the capability it lets the suite show, never
  by itself and never by a stored result that needs it.
- [ ] **CI:** `ruff format --check` → `ruff check` → `pyright` → `pytest --cov`
  → `pip-audit`, plus: demo tests, positive control, drift lock (R10.2.8),
  figure lock (R10.1.4). Demo tests join the fast gate. New modules
  (`visualization/`, `cli/gallery.py`) meet the same strict-typing bar as
  everything else. **No new verification rounds are commissioned this round**
  — R10 spends the trust R6–R9 built; it does not compound it.

## 📦 Registers (pull only under R10.3.6)

### A. Research continuation (RESEARCH3 track; outputs feed papers and the corroboration appendix, never the front page)

| Item | Status & pull condition |
|------|--------------------------|
| **Walled-regime boundary commission** | Pilot-mapped 2026-09-02 (`benchmark_results/boundary_map_pilot.json`): competence is shallow-only — boundary depth **4**, transition between depth 4 and 6; depth-4 result (0.396, d=+4.80) replicates the registered 0.406 on independent seeds. Pull when the research track needs the boundary location as a preregistered claim. |
| **Task-family generalization** | The R9 open surface: the linear-teacher boundary (nobody learns within the wall at depth ≥ 6; "gradient wins at depth" holds only on the stationary synthetic-teacher family) raises *which task families behave differently*. Pull when a research paragraph needs a second task family — design inherits the R8 gates + R9 method rules wholesale. |
| **CL prior-art revival (Split-MNIST)** | `cl_backward_transfer_matched_memory.json` + `cl_retest_discriminating_probe.json` through the R8 gates — pull when the research track wants a real-data retention study. |
| **AutoScientist boundary mapping** | Switch rate where routing retention dies; IR-drop level where the Pareto frontier shifts — pull when a research frontier figure is wanted; map only after the effect exists (R9 method rules). |
| **Z3 flagship registered commission** | RESEARCH3 CP-A hypothesis (≥95% on all three tasks within ≤20% of fine-tuning steps at exact Δθ=0; ≥5 seeds; baselines a–d incl. ICL bridge). The optimizer-hygiene defect (PR-1) is fixed in current code (fresh Adam at every boundary, documented in-module) — the commission runs on the fixed instrument and re-earns the Δθ audit at registered scale. Pull per critical path. |
| **PR-5 / PR-9 (RESEARCH3)** | Guard calibration from PR-7's harvested configs; campaign commissioning (iterate → interrupt → resume). Pull before any unattended campaign — nothing consumes the campaign stack until PR-9 passes. |
| **Registered-artifact provenance backfill** | 4 of 5 `benchmark_results/*.json` lack git-commit records (only `z3_fixed_weights/manifest.json` carries `config_sha256`/`git_commit`). Archaeology (timestamps vs git log) when the corroboration appendix is written; "provenance unknown" is an acceptable, honest label. Never blocks anything on the front page. |

### B. Library completeness (deferred ontology functionality; carried from TODO8/TODO9)

**Rule: every pulled item lands with its demo test (`tests/integration/test_demo_*`) — no test, no feature.** These are the axes' missing primitives; the demo suite can only demonstrate what exists.

| Item | Contents | Pull condition |
|------|----------|----------------|
| **Memristive factory** | README advertises `create_memristive_mlp` (Memristive × Feedforward, IR-drop/conductance preset) with no implementation in the tree; implement via `core/presets.py` over the existing `MemristiveSubstrate`, or the R10.2.7 restructure downgrades the row to Planned | R10.2.1 audit resolution, or next substrate-axis pull |
| **Geometries** | `ConvGeometry` / `GraphGeometry` / `AttentionGeometry` / 3D `SpatialLattice3D` — geometry-DEFERRED skips stay skips | Science runs on Feedforward/Recurrent/Tile at MLP scale today. Pull when a demo test or campaign manifest needs the geometry (e.g., a vision demo test wants Conv; a graph domain task wants GraphGeometry). |
| **Substrate fidelity (R3.7)** | Neuromorphic: real spike dropout or drop the cosmetic `sparsity` field; Memristive: conductance-range semantics | Pairs with RESEARCH3 substrate work; pull when a substrate-axis claim or demo test needs the fidelity to be real. |
| **Tile × dynamics matrix (R3.4)** | tile_ep/pc/gnn/snn device-dynamics incompatibilities; tile_fa/tp/hebbian — fix or document as permanent xfail with precise reasons | Pull on next touch of the tile family, or when a demo wants the full tile matrix. |
| **Adapter heuristics (R3.5)** | `_AdaptedSystem._infer_geometry` hardcoded (784→256,128→10) — recover heuristics from the deleted `adapter/` package | Pull when the strangler-fig adapter path is next touched. |
| **`_TaskTrainer` gaps (R3.6)** | Scheduler wiring, energy tracking, honor `tracker`/`safety_config` | Pull when hyperopt trials need them. |
| **Kernels (R4.1–R4.4)** | FA feedback projection through the Substrate operator API; `SubstrateSettleKernel` in `KernelRegistry`; MEP Triton kernels (Muon, Fisher whitening) → Substrate update operator; sparse transpose-mask handling, ternary `init_scale` (un-xfail ternary equivalence), per-step `inject_state_noise` | Pull when the acceleration/kernel path is next touched or a substrate-axis demo needs them. |
| **Nudge-unwired settle paths (imp-29)** | predictive_settling target clamp; diffusion target term | Pull when a campaign manifest needs those coordinates to be fully wired. |
| **Substrate facade merge (R2.2 residual)** | `ontology/_substrate.py` impl vs `ontology/substrate/` facade — facade is fine; consider merge + grep for other parallel legacy/new pairs (same pattern exists for `_dynamics.py` vs `dynamics/`) | Pull on next ontology-structure touch. |

### C. Code-quality backlog (imp-N + R-N items; pull on next touch of the file/system)

| Item | Pull condition |
|------|----------------|
| Joint `to_spec`→`from_spec` round-trip broken — `from_spec` calls `GeometryConfig(**spec["geometry"])` but `to_spec` embeds `params`/`recurrent_weight` keys → TypeError; found in the 2026-09-02 pre-flight | Next touch of `core/system_trainer/joint.py` (or when a demo/figure needs joint-spec round-trips) |
| imp-4 — Pyright full `strict` on ontology (131 findings; torch `Unknown` tracking; annotation work in `_dynamics`/`geometry`/`update`) | Next annotation pass on those modules |
| imp-8 — `compute_energy` duplication across Energy/Spike/Instantaneous/Diffusion → extract `_energy_from_state(state, geometry)` | Next touch of any dynamics module |
| imp-19 — `FrontierRecord.seed` legacy default 42 → required at next schema break | Schema break |
| imp-23 — `substrate_coupled` plasticity engagement-verified only; probe fixed-dim `step` assumptions | Next touch of that plasticity |
| imp-26 — params-moved learning locks for the remaining README-table factories (FA lock exists) | Next touch of each factory |
| imp-27 — rename rebuilder-style `settle` implementations whose names mislead | Next touch of each implementation |
| imp-30 — deployments' `family="tile"` registrations CLI-orphaned → fold into `family="equitile"` or drop | Next touch of deployment registry |
| imp-36 — campaign stability axis non-discriminative → cheap per-episode proxy | When a manifest needs stability contrast |
| imp-37 — latency objective is wall-clock noise → repeated-timing methodology or deterministic proxy | Before any task-scale latency claim |
| imp-41 — `demo/tests/` 28 stale failures → rewrite or delete | Next demo-test touch (or R11, where the demo gets rebuilt) |
| R3.8 — `natural_language_query` TF-IDF weighting; derive `V_nudged = free energy + β·loss` to strengthen the PC Lyapunov xfail | Next touch of the knowledge base / PC verification |
| test_scaling_invariants xpass — `deep_network_accuracy[100]` pre-existing xpass recurred in the full gate | Next touch of that file |

### D. Carried deferred (from TODO8; unchanged)

| Item | Reason |
|------|--------|
| Coverage floor (~16.8%) | opt-in `--cov`; raise after API stabilizes |
| Rocq general-case formalization | CP-B pull-based; diagonal case done with paper proof |
| `test_ontology_parity.py` decomposition | Slow-marked; split fast/slow only if gate iteration speed demands |
| Physical hardware (PR-3b / CP-D) | Latency-gated procurement per RESEARCH3 |

### E. Successor rounds

- **R11 — Live demo (compose-and-run UI).** Compose tab as primary surface
  (pick any 6-axis coordinate, hit run, watch curves), demo suite as
  pre-built presets, ψ visualizer + θ-hash badge as library features,
  one-click export. Built only when the API is stable — the demo presents the
  library; it does not design it. imp-41 resolves here (the demo gets
  rebuilt, not patched).
- **Drop-in PyTorch wrapper (RESEARCH3 CP-C)** — the adoption-friction
  multiplier; natural successor after the demo suite is green.
- **Hygiene sweep** (`demo/checkpoints/`, stray DBs at repo root — `dummy.db`,
  `execution_state.db` — ancient screenshot archives) — only when it blocks a
  figure, a test, or a fresh checkout.

### F. Watch (triggers convert to pull items; history canonical in TODO9.md)

- axis_probe `[2-0]` flake — no recurrence since 2026-08-31; still watching.
- CUDA tolerance boundaries shift xfail edges — CPU/GPU tests kept separate;
  construction seeding in place.
- R9.1 lr=0.03 is calibrated for the 40-episode budget — re-calibrate on
  schedule/budget changes; read A-mastery (~0.5 floor) before reading retention.
  Confirmed live at demo scale (2026-09-02 pre-flight): at A20/B20 the
  retention effect *reverses* — mastery precondition is load-bearing for D3.
- Control-band sizing (imp-59): preregistrate the at-chance band from the
  registered N of the control arm's scored samples.
- Smoke-scale campaign deltas are capped at chance by the non-stationary stream
  (imp-54) — accumulated-learning/retention claims run the persistent-θ chain
  only; CampaignStack rebuilds θ per episode regardless of teacher stationarity.
- The budget is a commissioning gate (R9.2): a feasible arm's walk is identical
  under every budget that admits it — never read walled arms' absence as
  "lost", or feasible arms' repeated readout as new evidence.
- `_LAZY` map and `__all__` can drift apart silently (both hand-maintained) —
  if the R10.2.1 audit finds divergence, add a one-shot consistency lock test
  rather than trusting discipline.

## Termination criterion

R10 closes when a stranger can, in one sitting: copy the README's first code
block, run it, and watch a system **they composed** train; then run the demo
suite and watch all five capabilities demonstrate themselves — compose, swap
credit, swap plasticity, hit the memory wall, freeze θ and see it hold
bitwise; then look at the gallery and recognize the same demonstrations drawn.
Nothing they read asks them to trust a file they didn't just regenerate;
nothing is claimed that they didn't just watch happen. **Read, run, change one
thing, see it matter** — with the proof re-earned on every commit. If that
session produces "oh, that's interesting, and I know exactly which line to
change" — the round did its job.
