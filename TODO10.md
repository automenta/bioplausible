# TODO10.md — Active Plan: The Library, Proven in Tests

> **Opened 2026-09-02.** Successor to [TODO9.md](TODO9.md) (R9 stress trials landed
> claim-grade; leftovers parked in the deferred register below).
> Research catalog: [RESEARCH3.md](RESEARCH3.md).
>
> **Identity decision (2026-09-02, v3):** **Computronium is an ML library.** The
> product is: import it, compose a learning system from the 6-axis ontology,
> train it, get results. The research validates that the library's abstractions
> are real. v1 treated findings as the product; v2 added a separate examples
> directory and a demo; v3 collapses both into the thing the repo already runs:
> **tests are the examples.** Demonstrative integration tests compose systems,
> train them, and assert library properties end-to-end. No examples directory,
> no examples gate, no drift — the test suite *is* the demonstration. The UI is
> deferred (R11 candidate): the demo presents the library; it must not design it.
>
> **Prime directive:** *The library is judged by its API, not by its JSON
> artifacts — and its API is judged by tests that read like documentation.*
> R8 made the instrument honest. R10 makes the library legible to a stranger
> with a terminal and a test runner.
>
> **State:** OPEN — R10.2 in design. All four evidence artifacts already exist
> under `benchmark_results/`; none has ever been rendered; the compositional API
> has no test that *demonstrates* it. No new experiments are commissioned this
> round. Termination criterion unchanged in spirit: **if it works it will be
> obvious.**
>
> **Register inheritance:** this doc supersedes TODO9.md and absorbs its open
> surface — R9's remaining items, the pull-based backlog (R2.2–R4.4,
> imp-4…imp-41), TODO8's deferred functionality (geometries, coverage floor,
> Rocq, hardware), and the Watch triggers. [TODO9.md](TODO9.md) remains the
> canonical append-only record for the Improvement Ledger (imp-42…imp-70
> lessons) and the Watch history; nothing below re-litigates a closed record.
> A Watch trigger firing converts its line into a pull item here.

---

## 🎯 The Evidence Table (what the artifacts prove about the library)

| # | Artifact | Registered numbers | What it proves about the library | Demo test |
|---|----------|--------------------|-----------------------------------|-----------|
| F1 | `benchmark_results/forgetting_registered.json` | Null: A-mastery ≈0.55 → A-retained ≈0.19; routing retains 0.315; d_retained −1.90 / d_delta −3.09, 16 seeds; Z3 restore bit-exact | **The M-axis is a real degree of freedom.** One call swaps the plasticity rule; the training loop never changes — and the swap measurably matters. | `test_demo_swap_plasticity.py` |
| F2 | `benchmark_results/memory_budget_registered.json` | Walled regime 0.015 MiB: thermo 0.406 vs frozen control 0.131, d = +2.89 (MDE 1.796); 0.45 MiB separates walled arms at depth 50 | **The memory machinery is honest.** The feasibility gate walls arms that cannot run (OOM semantics, not fake numbers); the resource axes are trustworthy enough to build a claim on. | `test_demo_memory_budget.py` |
| F3 | `benchmark_results/deep_credit_registered.json` | Depth 50: gradient 0.203, thermo 0.107, FA 0.128 (chance 0.125); d = +1.79/+1.54 (MDE 1.02) | **The C-axis comparison is a one-line change.** Same trainer, same task, three credit rules, three cliff edges. | `test_demo_swap_credit.py` |
| F4 | `benchmark_results/z3_fixed_weights/` | Engaged vs ψ-frozen gaps: parity +0.32..0.42, last_symbol +0.20..0.34, threshold +0.04..0.14; θ sha256 bitwise identical | **The joint lifecycle holds.** `JointSystemTrainer` promises θ never mutates intra-episode (J2) — the artifact records exactly that, bitwise, at registered scale. | `test_demo_z3_frozen_theta.py` |
| F5 | `benchmark_results/constraint_pilot.json` | Analog-noise sweep: EqProp 0.65→0.16 (collapses hardest) while Backprop 0.79→0.33 | **The harness doesn't flatter the library.** The same pipeline that proves an axis real publishes the arms that lose. | — (benchmark-only) |

**Rule of the round:** every figure caption speaks *library* — the abstraction,
the one-line swap, the evidence — and links to both its artifact and its demo
test. Scope labels (`retention`, `resource-efficiency`,
`credit assignment at depth`, `psi_engaged`) ride along verbatim from the R8.4
gates. A figure may never show more than its artifact supports.

---

## 🖼️ R10.1 — The Gallery (figures from artifacts; captions link tests)

Five figures, each generated **from the artifact JSON, not from a re-run** —
the artifact is ground truth; the figure is a lens on it.

- [ ] **R10.1.1** `computronium/visualization/gallery.py` — figure factory, one
  pure function per figure (`fig_forgetting_cliff`, `fig_memory_wall`,
  `fig_depth_cliff`, `fig_z3_tape`, `fig_refutations`). Each takes the artifact
  path, returns `(Figure, FigureMeta)`; `FigureMeta` carries claim scope, effect
  sizes, artifact sha256, and the **linked demo-test path**. matplotlib (already
  a dependency); one shared style module; no per-figure copy-paste.
- [ ] **R10.1.2** **F1 — The plasticity swap.** Panel A: per-seed slope chart,
  A-mastery → A-retained, null vs routing (16 paired lines per arm). Panel B:
  retention-delta bars with d annotation. Panel C: Z3 retention arm (mastery →
  post-switch floor → bit-exact restore). Caption: *"Swap the M-axis in one
  call; keep the training loop. The swap is worth d = −1.90."*
- [ ] **R10.1.3** **F2 — The memory wall.** Feasibility grid (budget MiB × depth
  regime), never-commissionable cells hatched (`gradient in` — OOM semantics).
  Overlaid bars for the walled cell (thermo vs frozen control, d = +2.89).
  Inset: O(depth) vs O(1) activation-memory profile. Caption: *"The library's
  memory profiler tells the truth before you train."*
- [ ] **R10.1.4** **F3 — One trainer, three credit rules.** Accuracy vs depth
  (4/16/50) for gradient / thermo / FA, chance line, 16-seed CIs; second axis:
  activation memory. Caption: *"Change one constructor argument; the comparison
  controls everything else."*
- [ ] **R10.1.5** **F4 — The lifecycle invariant, seen.** Engaged vs ψ-frozen
  accuracy per task with gap annotations; θ sha256 equality rendered as a
  first-class badge, not a footnote; ψ-trajectory panel if the artifact carries
  per-task ψ vectors. Caption: *"J2, verified bitwise — the frozen-hardware
  property is a library guarantee."*
- [ ] **R10.1.6** **F5 — The arms that lose.** Noise-sweep collapse curves
  (EqProp vs Backprop under analog noise) + one-line captions for imp-54
  (degenerate stream) and imp-55 (underpower). Caption: *"Same harness, losing
  arms, no selection."*
- [ ] **R10.1.7** Output to `docs/figures/*.png` (300 dpi) + `docs/figures/manifest.json`
  (per-figure: artifact path, artifact sha256, claim scope, effect sizes, demo-test link).
- [ ] **Acceptance:** all five figures regenerate from artifacts alone on a clean
  checkout; each caption names the library abstraction it evidences and links
  the test that demonstrates it at toy scale.

## 🧪 R10.2 — Tests Are the Examples (the demonstration layer)

Demonstrative integration tests in `tests/integration/`. Named `test_demo_*`
so pytest collects them in the default gate. Each ≤50 lines, fixed seeds,
quick-mode task scale, CPU ≤60 s, GPU-if-available. Module docstrings carry the
narrative (house style: Google-style, behavior-focused); assertions explain
themselves. **They assert library properties, not "it ran"** — the test is the
evidence, and it fails if the API breaks or the property regresses.

- [ ] **R10.2.1** **API surface audit.** `computronium/__init__.py` exposes the
  compositional API via `__all__` (per AGENTS.md): `System`/`compose_joint_system`,
  ontology primitives, `SystemTrainer`/`JointSystemTrainer`, `SystemTrainerConfig`,
  the 13 factories, `create_task`. The demo tests import **only from the package
  root** — the tests are the audit. Missing root exports get added; backwards
  compatibility is explicitly none.
- [ ] **R10.2.2** `test_demo_compose_6axis.py` — the flagship. Full six-axis
  composition via `compose_joint_system` (Digital × Recurrent ×
  EnergyMinimization × Null × ThermodynamicContrast × Euclidean), quick-mode
  task, short `SystemTrainer.fit`. Asserts: valid metrics (loss ≥ 0, accuracy ∈
  [0,1] above chance), config round-trip identity (`extract_config` →
  `compose_system_from_configs`), J1 zero-extension (null-plasticity forward ≡
  5-D dynamics). *A stranger reads this and knows how to build anything.*
- [ ] **R10.2.3** `test_demo_swap_credit.py` — one trainer, three credit rules.
  Same coordinate trained three times with the C-axis swapped (gradient /
  ThermodynamicContrast / RandomProjections) through identical `SystemTrainer`
  wiring. Asserts: all three produce valid learning metrics; the wiring code is
  byte-identical across arms except the constructor argument. *The comparison
  is one line.*
- [ ] **R10.2.4** `test_demo_swap_plasticity.py` — the M-axis matters. Null vs
  RoutingPlasticity on the segmented switching stream (R8.3 machinery, toy
  scale, fixed seeds). Asserts **directionally**: routing's A-retention exceeds
  null's A-retention with a calibrated guard band (the registered d = −1.90 is
  the registered-scale evidence; the demo test pins the direction and the
  one-call API). *The test is F1's toy-scale embodiment.*
- [ ] **R10.2.5** `test_demo_memory_budget.py` — honest feasibility. Memory
  profiler's feasibility grid at demo scale: BPTT-profiled arm walled under a
  tight budget (OOM verdict), O(1)-memory arm feasible, frozen-thermo control
  at-chance verdict available in every regime. Asserts the grid semantics, plus
  the learner directional check where cheap. *The profiler tells the truth
  before you train.*
- [ ] **R10.2.6** `test_demo_z3_frozen_theta.py` — the lifecycle guarantee. θ
  frozen, ψ adapts across a task switch A→B→A. Asserts **bitwise** θ hash
  equality across the whole run, ψ-system snapshot/restore, and above-chance
  probe accuracy post-switch. *J2, demonstrated, not just locked.*
- [ ] **R10.2.7** **README restructure.** Opens with the ≤10-line composition →
  train → report block **derived from `test_demo_compose_6axis.py`**, then
  factory one-liners, then the gallery figures as *evidence the abstractions
  are real*. Ontology/capability/architecture tables move below the fold. The
  "Three Perspectives" table inverts: ML Library first and load-bearing;
  research framed as validation of the library.
- [ ] **R10.2.8** **Drift lock.** README python blocks are checked verbatim
  against their source test files (small extraction script + lock test). A
  README block that no longer matches its test fails CI — no doc/test drift,
  by construction.
- [ ] **R10.2.9** **`docs/RESULTS.md` — evidence, organized by axis.** Per axis
  (S/G/D/M/C/U): what the abstraction is, the one-line swap, the figure, the
  demo test, the scope label, what it does **not** mean. Closes with F5 as
  "why the others are believable." Not "here are our findings" — "here is what
  the library can do, and here is the proof."
- [ ] **R10.2.10** **`comp gallery`** (`computronium/cli/gallery.py`, existing
  subcommand pattern): regenerates figures + manifest from `benchmark_results/`;
  exit nonzero on missing/hash-mismatched artifacts. Figure-drift lock test:
  regenerates, asserts manifest data-layer checksums (artifact hash, panel
  means — not pixels).
- [ ] **Acceptance:** a stranger copies the README's first block and runs it;
  reads `test_demo_swap_credit.py` top-to-bottom and knows how to swap any axis;
  `pytest tests/integration/ -k demo` is green in the fast gate on CPU.

## 🔒 R10.3 — The Standing Rules

- [ ] **R10.3.1** **No test, no feature.** Every feature ships with an
  integration test that demonstrates it working end-to-end. The test is the
  example. The library is judged by its API; its API is judged by tests that
  read like documentation.
- [ ] **R10.3.2** **No naked JSON.** A commissioned trial is done when its figure
  exists, its caption speaks library, and its Evidence Table row is filled.
  Figure plans declared at commissioning: `docs/preregistration_template.md`
  gains a **Figure** section (what gets plotted, from which artifact panel,
  which demo test embodies the design).
- [ ] **R10.3.3** **Scope labels ride along.** Every figure and RESULTS.md
  paragraph carries the R8.4 claim-scope label of its source run. Demo-scale
  tests pin directions, never claim magnitudes — a toy-scale assertion may
  never wear a registered claim's clothes.
- [ ] **R10.3.4** **Refutations ship with the same pipeline** — same figure
  factory, same docs paragraph, same terms (F5 is the template).
- [ ] **R10.3.5** **Pull rule:** a backlog item is pulled only if it ends in a
  demo test, a figure, or a RESULTS.md paragraph. Infrastructure is justified
  by the API story it enables, never by itself.
- [ ] **CI:** `ruff format --check` → `ruff check` → `pyright` → `pytest --cov`
  → `pip-audit`, plus the drift lock (R10.2.8) and figure-drift lock
  (R10.2.10). Demo tests join the fast gate. **No new verification rounds are
  commissioned this round** — R10 spends the trust R6–R9 built; it does not
  compound it.

## 📦 Registers (pull only under R10.3.5)

### A. Research continuation (pulled from R9 / RESEARCH3 CP-A)

| Item | Status & pull condition |
|------|--------------------------|
| **Walled-regime boundary commission** | Pilot-mapped 2026-09-02 (`benchmark_results/boundary_map_pilot.json`): competence is shallow-only — boundary depth **4**, transition between depth 4 and 6; depth-4 result (0.396, d=+4.80) replicates the registered 0.406 on independent seeds. The registered boundary claim is **pull-based**: a preregistration derives from this pilot's variance when the next manifest needs the boundary location as a claim. A boundary *figure* (F2 companion) is the natural R10.1 pull. |
| **Task-family generalization** | The R9 open surface: the linear-teacher boundary (nobody learns within the wall at depth ≥ 6; "gradient wins at depth" holds only on the stationary synthetic-teacher family) raises *which task families behave differently*. Pull when a RESULTS.md paragraph needs a second task family — design inherits the R8 gates + R9 method rules wholesale. |
| **CL prior-art revival (Split-MNIST)** | `cl_backward_transfer_matched_memory.json` + `cl_retest_discriminating_probe.json` through the R8 gates — pull when a real-data F1 is wanted alongside the synthetic one. |
| **AutoScientist boundary mapping** | Switch rate where routing retention dies; IR-drop level where the Pareto frontier shifts — pull when a boundary frontier *figure* is wanted; map only after the effect exists (R9 method rules). |
| **Z3 flagship registered commission** | RESEARCH3 CP-A hypothesis (≥95% on all three tasks within ≤20% of fine-tuning steps at exact Δθ=0; ≥5 seeds; baselines a–d incl. ICL bridge) — unblocked by the R8 gate work; re-verifies the ψ-gate per seed at its own scale. Pull per critical path; its figures (accuracy-vs-steps, forgetting matrix) slot into the gallery. |
| **PR-5 / PR-9 (RESEARCH3)** | Guard calibration from PR-7's harvested configs; campaign commissioning (iterate → interrupt → resume). Pull before any unattended campaign — nothing consumes the campaign stack until PR-9 passes. |

### B. Library completeness (deferred ontology functionality; carried from TODO8/TODO9)

**Rule: every pulled item lands with its demo test (`tests/integration/test_demo_*`) — no test, no feature.** These are the axes' missing primitives; the gallery and demo tests can only demonstrate what exists.

| Item | Contents | Pull condition |
|------|----------|----------------|
| **Geometries** | `ConvGeometry` / `GraphGeometry` / `AttentionGeometry` / 3D `SpatialLattice3D` — geometry-DEFERRED skips stay skips | Science runs on Feedforward/Recurrent/Tile at MLP scale today. Pull when a demo test or campaign manifest needs the geometry (e.g., a vision demo test wants Conv; a graph domain task wants GraphGeometry). |
| **Substrate fidelity (R3.7)** | Neuromorphic: real spike dropout or drop the cosmetic `sparsity` field; Memristive: conductance-range semantics | Pairs with RESEARCH3 substrate work; pull when a substrate-axis claim or demo test needs the fidelity to be real. |
| **Tile × dynamics matrix (R3.4)** | tile_ep/pc/gnn/snn device-dynamics incompatibilities; tile_fa/tp/hebbian — fix or document as permanent xfail with precise reasons | Pull on next touch of the tile family, or when a gallery figure wants the full tile matrix. |
| **Adapter heuristics (R3.5)** | `_AdaptedSystem._infer_geometry` hardcoded (784→256,128→10) — recover heuristics from the deleted `adapter/` package | Pull when the strangler-fig adapter path is next touched. |
| **`_TaskTrainer` gaps (R3.6)** | Scheduler wiring, energy tracking, honor `tracker`/`safety_config` | Pull when hyperopt trials need them. |
| **Kernels (R4.1–R4.4)** | FA feedback projection through the Substrate operator API; `SubstrateSettleKernel` in `KernelRegistry`; MEP Triton kernels (Muon, Fisher whitening) → Substrate update operator; sparse transpose-mask handling, ternary `init_scale` (un-xfail ternary equivalence), per-step `inject_state_noise` | Pull when the acceleration/kernel path is next touched or a substrate-axis figure needs them. |
| **Nudge-unwired settle paths (imp-29)** | predictive_settling target clamp; diffusion target term | Pull when a campaign manifest needs those coordinates to be fully wired. |
| **Substrate facade merge (R2.2 residual)** | `ontology/_substrate.py` impl vs `ontology/substrate/` facade — facade is fine; consider merge + grep for other parallel legacy/new pairs | Pull on next ontology-structure touch. |

### C. Code-quality backlog (imp-N + R-N items; pull on next touch of the file/system)

| Item | Pull condition |
|------|----------------|
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

- **R11 — Live demo (compose-and-run UI).** v2's R10.3 preserved here, deferred
  deliberately: Compose tab as primary surface (pick any 6-axis coordinate, hit
  run, watch curves), registered trials as pre-built presets, ψ visualizer +
  θ-hash badge as library features, one-click export. Built only when the API
  is stable — the demo presents the library; it does not design it. imp-41
  resolves here (the demo gets rebuilt, not patched).
- **Drop-in PyTorch wrapper (RESEARCH3 CP-C)** — the adoption-friction
  multiplier; natural successor after the demo tests are green.
- **Hygiene sweep** (`demo/checkpoints/`, stray DBs at repo root, ancient
  screenshot archives) — only when it blocks a figure, a test, or a fresh
  checkout.

### F. Watch (triggers convert to pull items; history canonical in TODO9.md)

- axis_probe `[2-0]` flake — no recurrence since 2026-08-31; still watching.
- CUDA tolerance boundaries shift xfail edges — CPU/GPU tests kept separate;
  construction seeding in place.
- R9.1 lr=0.03 is calibrated for the 40-episode budget — re-calibrate on
  schedule/budget changes; read A-mastery (~0.5 floor) before reading retention.
- Control-band sizing (imp-59): preregistrate the at-chance band from the
  registered N of the control arm's scored samples.
- Smoke-scale campaign deltas are capped at chance by the non-stationary stream
  (imp-54) — accumulated-learning/retention claims run the persistent-θ chain
  only; CampaignStack rebuilds θ per episode regardless of teacher stationarity.
- The budget is a commissioning gate (R9.2): a feasible arm's walk is identical
  under every budget that admits it — never read walled arms' absence as
  "lost", or feasible arms' repeated readout as new evidence.

## Termination criterion

R10 closes when a stranger can, in one sitting: copy the README's first code
block, run it, and watch a system **they composed** train; then open
`tests/integration/test_demo_swap_credit.py`, read it top-to-bottom, and know
exactly which line to change to swap any axis; then find every claim on the
front page backed by a figure whose caption names the abstraction that made the
comparison possible. **Read, run, change one thing, see it matter** — with the
proof version-controlled and running in CI. If that session produces "oh,
that's interesting, and I know exactly which line to change" — the round did
its job.
