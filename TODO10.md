# TODO10.md — Active Plan: The Library, Shown Working

> **Opened 2026-09-02.** Successor to [TODO9.md](TODO9.md) (R9 stress trials landed
> claim-grade; leftovers parked in the deferred register below).
> Research catalog: [RESEARCH3.md](RESEARCH3.md).
>
> **Identity decision (2026-09-02, amended v2 same day):** **Computronium is an ML
> library.** Someone imports it, composes a learning system from the 6-axis
> ontology, trains it, and gets results — that is the product. The research
> validates that the library's abstractions are correct. The figures demonstrate
> the library producing meaningful results. The live demo lets someone interact
> with the library. v1 of this plan treated the research findings as the product;
> that was overcorrection. The forgetting cliff is not the product — it is
> evidence that the M-axis abstraction is real. The proof serves the library.
>
> **Prime directive:** *The library is judged by its API, not by its JSON
> artifacts.* R8 made the instrument honest. R10 makes the library — and the
> evidence behind it — visible to a stranger with a terminal.
>
> **State:** OPEN — R10.2 in design. All four evidence artifacts already exist
> under `benchmark_results/`; none has ever been rendered, and the compositional
> API has no copy-paste-canonical example. No new experiments are commissioned in
> this round. Termination criterion unchanged in spirit: **if it works it will be
> obvious** — and it is not obvious from the API's current front door.

---

## 🎯 The Evidence Table (what the artifacts prove about the library)

| # | Artifact | Registered numbers | What it proves about the library |
|---|----------|--------------------|-----------------------------------|
| F1 | `benchmark_results/forgetting_registered.json` | Null: A-mastery ≈0.55 → A-retained ≈0.19; routing retains 0.315; d_retained −1.90 / d_delta −3.09, 16 seeds; Z3 restore bit-exact | **The M-axis is a real degree of freedom.** One line swaps the plasticity rule; the training loop never changes — and the swap measurably matters. The abstraction is not decoration. |
| F2 | `benchmark_results/memory_budget_registered.json` | Walled regime 0.015 MiB: thermo 0.406 vs frozen control 0.131, d = +2.89 (MDE 1.796); 0.45 MiB separates walled arms at depth 50; deep tier under wall: thermo 0.172 vs chance 0.125 | **The memory machinery is honest.** The feasibility gate walls arms that cannot run (OOM semantics, not fake numbers), and the resource axes (post-imp-45) are trustworthy enough to build a claim on. |
| F3 | `benchmark_results/deep_credit_registered.json` | Depth 50: gradient 0.203, thermo 0.107, FA 0.128 (chance 0.125); d = +1.79/+1.54 (MDE 1.02); depths 4/16/50 | **The C-axis comparison is a one-line change.** Same trainer, same task, three credit rules, three cliff edges — the common-`SystemTrainer` API makes controlled comparison the default, not a project. |
| F4 | `benchmark_results/z3_fixed_weights/` | Engaged vs ψ-frozen gaps: parity +0.32..0.42, last_symbol +0.20..0.34, threshold +0.04..0.14; θ sha256 bitwise identical across arms | **The joint lifecycle holds.** `JointSystemTrainer` promises θ never mutates intra-episode (J2) and ψ carries the adaptation — the artifact records exactly that invariant, bitwise, at registered scale. |
| F5 | `benchmark_results/constraint_pilot.json` | Analog-noise sweep: EqProp 0.65→0.16 (collapses hardest) while Backprop 0.79→0.33 | **The harness doesn't flatter the library.** The same pipeline that proves an axis real publishes the arms that lose. Evidence, not marketing. |

**Rule of the round:** every figure caption speaks *library*: the abstraction,
the one-line swap, the evidence. Every figure links to the runnable example
that reproduces its design (R10.2). Scope labels (`retention`,
`resource-efficiency`, `credit assignment at depth`, `psi_engaged`) ride along
verbatim from the R8.4 gates — a figure may never show more than its artifact
supports.

---

## 🖼️ R10.1 — The Gallery (figures stay; captions re-aimed at the library)

Five figures, each generated **from the artifact JSON, not from a re-run** — the
artifact is ground truth; the figure is a lens on it. Caption template: *what
you compose in one call → what the library measures → what the artifact shows.*

- [ ] **R10.1.1** `computronium/visualization/gallery.py` — figure factory with one
  pure function per figure (`fig_forgetting_cliff`, `fig_memory_wall`,
  `fig_depth_cliff`, `fig_z3_tape`, `fig_refutations`). Each function takes the
  artifact path, returns `(Figure, FigureMeta)`; `FigureMeta` carries claim scope,
  effect sizes, artifact sha256, and the **linked example path** (R10.2.2).
  matplotlib (already a dependency); one shared style module; no per-figure
  copy-paste.
- [ ] **R10.1.2** **F1 — The plasticity swap.** Panel A: per-seed slope/dumbbell
  chart, A-mastery → A-retained, null vs routing (16 paired lines per arm).
  Panel B: retention-delta bars with d annotation. Panel C: Z3 retention arm
  (mastery → post-switch floor → bit-exact restore). Caption: *"Swap the M-axis
  in one call; keep the training loop. The swap is worth d = −1.90."*
- [ ] **R10.1.3** **F2 — The memory wall.** Feasibility grid (budget MiB × depth
  regime) with per-arm feasibility coloring; never-commissionable cells hatched
  and labeled (`gradient in` — OOM semantics). Overlaid bars for the walled cell
  (thermo vs frozen control, d = +2.89). Inset: O(depth) vs O(1) activation-memory
  profile. Caption: *"The library's memory profiler tells the truth before you
  train."*
- [ ] **R10.1.4** **F3 — One trainer, three credit rules.** Accuracy vs depth
  (4/16/50) for gradient / thermo / FA with chance line and 16-seed CIs; second
  axis: activation memory O(depth) vs O(1). Caption: *"Change one constructor
  argument; the comparison controls everything else."*
- [ ] **R10.1.5** **F4 — The lifecycle invariant, seen.** Engaged vs ψ-frozen
  accuracy per task with gap annotations; θ sha256 equality rendered as a
  first-class element (a badge, not a footnote); ψ-trajectory panel if the
  artifact carries per-task ψ vectors. Caption: *"J2, verified bitwise, at
  registered scale — the frozen-hardware demo is a library property."*
- [ ] **R10.1.6** **F5 — The arms that lose.** The noise-sweep collapse curves
  (EqProp vs Backprop under analog noise) plus one-line captions for imp-54
  (degenerate stream) and imp-55 (underpower). Caption: *"Same harness, losing
  arms, no selection."*
- [ ] **R10.1.7** Output to `docs/figures/*.png` (300 dpi) + `docs/figures/manifest.json`
  (per-figure: artifact path, artifact sha256, claim scope, effect sizes, example link).
- [ ] **Acceptance:** all five figures regenerate from artifacts alone on a clean
  checkout; each caption names the library abstraction it evidences; each links
  to a runnable example.

## 📖 R10.2 — The Story (README opens with code, not ontology)

The library's front door is a code block. The first thing a stranger sees must
run.

- [ ] **R10.2.1** **API surface audit.** `computronium/__init__.py` exposes the
  compositional API via `__all__` (per AGENTS.md): `System`/`compose_joint_system`,
  the ontology primitives, `SystemTrainer`/`JointSystemTrainer`,
  `SystemTrainerConfig`, the 13 factories, and `create_task`. The canonical
  example imports **only from the package root**. Where the root export is
  missing, add it — backwards compatibility is explicitly none.
- [ ] **R10.2.2** **`examples/` — the canonical examples.** ≤30-line scripts, each
  a complete composition → train → report story, CPU-feasible ≤60 s
  (GPU-if-available), quick-mode task scale:
  `compose_6axis.py` (the flagship: full six-axis composition),
  `train_eqprop.py`, `swap_credit.py` (one-arg C-axis swap),
  `swap_plasticity.py` (one-call M-axis swap), `memory_budget.py`
  (feasibility gate demo), `z3_switch.py` (frozen-θ task switch with θ-hash print).
  Exact constructor signatures come from `compose_joint_system` /
  `GeometryConfig` / `EnergyMinimizationDynamics` as they exist — the examples
  gate (R10.2.5) is what makes them trustworthy, not this document.
- [ ] **R10.2.3** **README restructure.** Opens with the ≤10-line compositional
  example (a condensed `compose_6axis.py`), then the factory one-liners, then
  the gallery figures as *evidence the abstractions are real*. The 6-D
  decomposition tables, capability tables, and architecture diagrams move below
  the fold. The "Three Perspectives" table is inverted: ML Library first and
  load-bearing; research and scientific program framed as *validation of the
  library*.
- [ ] **R10.2.4** **`docs/RESULTS.md` — evidence, organized by axis.** Per axis
  (S/G/D/M/C/U): what the abstraction is, the one-line swap, the figure, the
  scope label, what it does **not** mean. Closes with F5 as "why the others are
  believable." This is not "here are our findings"; it is "here is what the
  library can do, and here is the proof."
- [ ] **R10.2.5** **The examples gate.** `examples/` runs in the fast CI gate at
  quick-mode scale; README python blocks are extracted and executed
  (`scripts/run_readme_examples.py`, wired into pre-commit/CI). A code block
  that cannot run is a defect, same class as a failing test.
- [ ] **R10.2.6** **`comp gallery`** (`computronium/cli/gallery.py`, existing
  subcommand pattern): regenerates all figures + manifest from
  `benchmark_results/`; exit nonzero if any artifact is missing or
  hash-mismatched. Figure-drift lock test: regenerates and asserts manifest
  data-layer checksums (artifact hash, panel means — not pixels).
- [ ] **Acceptance:** a stranger copies the README's first block into a fresh
  `uv run python - <<EOF` and watches a system they composed train; every README
  code block runs in CI; every figure caption names an abstraction and links an
  example.

## ⚡ R10.3 — The Live Demo (compose it, run it, watch it learn)

The demo already has the right bones: `demo/runner.py` wraps trainers as a pure
telemetry consumer, `persistence.py` exports PNG/CSV, `weight_viz.py` and
`charts.py` render. R10.3 re-aims the frame from "watch a phenomenon" to
**"use the library."**

- [ ] **R10.3.1** **Compose tab is the primary surface.** Pick substrate /
  geometry / dynamics / plasticity / credit / update (the ontology mode already
  exists — promote it), hit run, watch curves stream. The point being
  demonstrated is the library's own claim: *one trainer, any coordinate.*
- [ ] **R10.3.2** **Registered trials become pre-built configs.** The R9 trials
  appear as a preset row in the Compose tab ("interesting compositions from the
  validation campaign"), each loading its arm configuration into the same
  compose-and-run flow — pre-built configs that demonstrate the library, not
  standalone experiments. Each preset links to its RESULTS.md section and
  registered artifact, labeled "small-scale demonstration of a registered
  design."
- [ ] **R10.3.3** **ψ visualizer + θ-hash badge as library features.** During any
  joint run, animate routing gates `g_k(ψ)` / ψ trajectory, with a permanently
  visible θ-hash badge that never changes intra-episode. This is the J2
  invariant made visible — the demo of a library guarantee, not a magic trick.
- [ ] **R10.3.4** **One-click export** (PNG + JSON) reusing `persistence.py`;
  exported JSON embeds the full coordinate, seed, and gate verdict — the demo
  run is itself a reproducible artifact.
- [ ] **Acceptance:** `uv run python demo/main.py` → Compose tab → pick a credit
  rule, a geometry, a dynamics → run completes ≤60 s on CPU (GPU toggle where
  available) with live curves, ψ gates, and the frozen θ-hash visible;
  preset row loads registered designs; export round-trips.

## 🔒 R10.4 — The Standing Rules

- [ ] **R10.4.1** **No example, no feature.** No feature is done until there is a
  working, copy-pasteable example in README or docs that a stranger can run.
  The examples gate (R10.2.5) enforces this mechanically. The library is judged
  by its API, not by its JSON artifacts.
- [ ] **R10.4.2** **No naked JSON.** A commissioned trial is done when its figure
  exists, its caption speaks library, and its Headline Table row is filled.
  Figure plans are declared at commissioning: `docs/preregistration_template.md`
  gains a **Figure** section.
- [ ] **R10.4.3** **Scope labels ride along.** Every figure, demo preset, and
  RESULTS.md paragraph carries the R8.4 claim-scope label of its source run.
  A demo preview may never wear a registered claim's clothes.
- [ ] **R10.4.4** **Refutations ship with the same pipeline** — same figure
  factory, same docs paragraph, same terms (F5 is the template).
- [ ] **R10.4.5** **Pull rule:** a backlog item is pulled only if it ends in an
  example, a figure, a demo mode, or a docs paragraph — examples first among
  equals. Infrastructure is justified by the API story it enables.
- [ ] **CI:** `ruff format --check` → `ruff check` → `pyright` → `pytest --cov`
  → `pip-audit`, plus the examples gate (R10.2.5) and figure-drift lock
  (R10.2.6). **No new verification rounds are commissioned this round** — R10
  spends the trust R6–R9 built; it does not compound it.

## 📦 Deferred Register (pull only under R10.4.5)

- Split-MNIST CL prior-art revival through the R8 gates
  (`cl_backward_transfer_matched_memory.json`) — pull when a real-data F1 is wanted.
- AutoScientist boundary mapping (switch rate where routing retention dies;
  IR-drop level where the Pareto frontier shifts) — pull when a boundary
  frontier *figure* is wanted; map only after the effect exists.
- **Drop-in PyTorch wrapper (RESEARCH3 CP-C)** — the adoption-friction
  multiplier; the natural successor round once the examples gate is green.
- Physical-hardware validation — the production-library gate; unchanged, deferred.
- Hygiene sweep (`demo/checkpoints/`, stray DBs at repo root, ancient screenshot
  archives) — only when it blocks a figure, an example, or first-run experience.

## Termination criterion

R10 closes when a stranger can, in one sitting: copy the README's first code
block, run it, and watch a system **they composed** train; then open the demo,
change one axis, and see the difference; then find every claim on the front
page backed by a figure whose caption names the abstraction that made the
comparison possible. Copy, run, change one thing, see it matter. If that
session produces **"oh, that's interesting — and I know exactly which line to
change"** — the round did its job.
