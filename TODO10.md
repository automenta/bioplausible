# TODO10.md — Active Plan: The Proof-of-Life Release

> **Opened 2026-09-02.** Successor to [TODO9.md](TODO9.md) (R9 stress trials landed
> claim-grade; leftovers parked in the deferred register below).
> Research catalog: [RESEARCH3.md](RESEARCH3.md).
>
> **Identity decision (2026-09-02 strategic review, adopted):** Computronium is a
> **demonstration platform with a research engine attached.** The research engine
> (R6–R9 verification machinery) is built and stays. From here on, the *product*
> is what a stranger can **see**: registered phenomena rendered as figures, walked
> as a story, and re-run live in the browser. Production-library status is
> explicitly deferred until physical-hardware validation exists (the README's own
> honesty standard). Teaching flows from demonstration; it is not a separate goal.
>
> **Prime directive:** *A claim-grade result nobody can see is, in practice,
> quarantined.* R8 made the instrument honest. R10 makes the evidence legible.
>
> **State:** OPEN — R10.1 in design. The four headline results already exist as
> artifacts under `benchmark_results/`; none has ever been rendered. No new
> experiments are commissioned in this round. Termination criterion is unchanged:
> **if it works it will be obvious** — and this round's job is to make it obvious.

---

## 🎯 The Headline Table (what we can already show)

| # | Phenomenon | Artifact | Registered numbers | The 5-second read |
|---|-----------|----------|--------------------|--------------------|
| F1 | **The forgetting cliff** (R9.1, M-axis) | `benchmark_results/forgetting_registered.json` | Null: A-mastery ≈0.55 → A-retained ≈0.19; routing retains 0.315; d_retained −1.90 / d_delta −3.09, 16 seeds; Z3 restore bit-exact, floor ≈0.5 | Red lines fall off a cliff. Blue lines don't. θ frozen throughout. |
| F2 | **The memory wall** (R9.2, S/D axes) | `benchmark_results/memory_budget_registered.json` | Walled regime 0.015 MiB: thermo 0.406 vs frozen control 0.131, d = +2.89 (MDE 1.796); 0.45 MiB separates walled arms at depth 50; deep tier under wall: thermo 0.172 vs chance 0.125 | There is a regime where backprop literally cannot exist. |
| F3 | **The depth cliff** (R9.3, C-axis) | `benchmark_results/deep_credit_registered.json` | Depth 50: gradient 0.203, thermo 0.107, FA 0.128 (chance 0.125); d = +1.79/+1.54 (MDE 1.02); depths 4/16/50 | Three algorithms, three cliff edges — and the signal is the memory profile, not vanishing gradients. |
| F4 | **Same hardware, different tape** (Z3 flagship) | `benchmark_results/z3_fixed_weights/` | Engaged vs ψ-frozen gaps: parity +0.32..0.42, last_symbol +0.20..0.34, threshold +0.04..0.14; θ sha256 bitwise identical across arms; restored ψ == stage-A mastery bit-exact | Only the tape changed. The hardware hash never moves. |
| F5 | **What we refuted** (honest negatives) | `benchmark_results/constraint_pilot.json` | Analog-noise sweep: EqProp 0.65→0.16 (collapses hardest) while Backprop 0.79→0.33 (degrades gracefully); imp-54 degenerate stream; imp-55 underpower | We show our dead ends. That is why the other four are believable. |

**Rule of the round:** every number above came through the R8 gates
(preregistration, embedded planted-effect control, power check, control arm).
The figures inherit the gates: each figure carries its claim-scope label
(`retention`, `resource-efficiency`, `credit assignment at depth`,
`psi_engaged`) and its d/MDE annotation. A figure may never show more than its
artifact supports.

---

## 🖼️ R10.1 — The Gallery (figures from registered artifacts; no new experiments)

Five figures, each generated **from the artifact JSON, not from a re-run** — the
artifact is the ground truth; the figure is a lens on it.

- [ ] **R10.1.1** `computronium/visualization/gallery.py` — figure factory with one
  pure function per figure (`fig_forgetting_cliff`, `fig_memory_wall`,
  `fig_depth_cliff`, `fig_z3_tape`, `fig_refutations`). Each function takes the
  artifact path, returns `(Figure, FigureMeta)`; `FigureMeta` carries claim scope,
  effect sizes, artifact sha256. matplotlib (already a dependency); one shared
  style module; no per-figure copy-paste.
- [ ] **R10.1.2** **F1 — The forgetting cliff.** Panel A: per-seed slope/dumbbell
  chart, A-mastery → A-retained, null vs routing (16 paired lines per arm). Panel
  B: retention-delta bars with d annotation. Panel C: Z3 retention arm
  (mastery → post-switch floor → bit-exact restore, annotated). Title badge:
  `θ frozen throughout`.
- [ ] **R10.1.3** **F2 — The memory wall.** Feasibility grid (budget MiB × depth
  regime) with per-arm feasibility coloring; cells where an arm can never
  commission are hatched and labeled (`gradient in` — OOM semantics, not a
  performance number). Overlaid bars for the walled cell (thermo vs frozen
  control, d = +2.89). Inset: O(depth) vs O(1) activation-memory profile.
- [ ] **R10.1.4** **F3 — The depth cliff.** Accuracy vs depth (4/16/50) for
  gradient / thermo / FA with chance line and per-point CIs from the 16-seed
  artifact; second axis: activation memory O(depth) vs O(1). Caption carries the
  registered meaning: the C-axis signal is the memory profile.
- [ ] **R10.1.5** **F4 — Same hardware, different tape.** Engaged vs ψ-frozen
  accuracy per task with gap annotations; θ sha256 equality rendered as a
  first-class element of the figure (a badge, not a footnote). If per-task ψ
  trajectories exist in the artifact, render a small ψ-trajectory panel.
- [ ] **R10.1.6** **F5 — What we refuted.** The noise-sweep collapse curves
  (EqProp vs Backprop under analog noise) plus one-line captions for imp-54
  (degenerate stream) and imp-55 (underpower). The refutation figure is a
  credibility feature, not a confession.
- [ ] **R10.1.7** Output to `docs/figures/*.png` (300 dpi) + `docs/figures/manifest.json`
  (per-figure: artifact path, artifact sha256, claim scope, effect sizes).
- [ ] **Acceptance:** all five figures regenerate from artifacts alone on a clean
  checkout; each carries scope label + effect size; a stranger can state the
  finding from each figure in one sentence without reading the caption.

## 📖 R10.2 — The Story (make a stranger care in one page)

- [ ] **R10.2.1** README restructure: **figures first**. New top section "What we
  can show you" — the five figures with one-sentence findings and effect sizes.
  The 6-D ontology, capability tables, and architecture diagrams move below the
  fold. The opening claim changes from "an ontology" to "phenomena you can look at."
- [ ] **R10.2.2** `docs/RESULTS.md` — the walkthrough. Per claim: hypothesis in
  one sentence → design in one sentence → figure → what it means → what it does
  **not** mean (scope labels carry through verbatim from the gates). Includes F5
  as "how we know the others are honest."
- [ ] **R10.2.3** `comp gallery` CLI command (`computronium/cli/gallery.py`,
  registered with the existing subcommand pattern): regenerates all figures +
  manifest + RESULTS.md tables from `benchmark_results/`. Exit nonzero if any
  artifact is missing or hash-mismatched — the gallery cannot silently drift
  from the artifacts.
- [ ] **R10.2.4** Figure-drift lock: a small test regenerates the gallery and
  asserts manifest stability (data-layer checksums — artifact hash, panel means —
  not pixel hashes; rendering nondeterminism is not a defect).
- [ ] **Acceptance:** `uv run comp gallery` on a clean checkout produces
  `docs/figures/` + manifest; README front page shows the five figures above the
  fold; RESULTS.md reads start-to-finish in under five minutes.

## ⚡ R10.3 — The Live Demo (watch a phenomenon happen)

The demo already has the right bones: `demo/runner.py` wraps trainers as a pure
telemetry consumer, `persistence.py` exports PNG/CSV, `weight_viz.py` and
`charts.py` render. R10.3 adds the missing mode: **the registered phenomena,
live, small-scale.**

- [ ] **R10.3.1** New demo tab **"Trials"**: pick a registered trial (forgetting /
  memory budget / deep credit / Z3 switch), run a registered-*shaped* small
  version (3 seeds, short segments — same arms, same controls, same metrics,
  scaled budget). Curves stream per-episode; the control arm renders alongside;
  the prereg gate verdict renders as a live badge (control at chance or the run
  quarantines itself, visibly).
- [ ] **R10.3.2** **ψ visualizer**: during the Z3 trial, animate the routing gate
  `g_k(ψ)` bars and the ψ trajectory across the A→B→A switch, with a
  permanently-visible θ-hash badge that never changes during the run. The demo
  tagline writes itself: *watch the algorithm change without touching the
  hardware.*
- [ ] **R10.3.3** Scale honesty: the Trials tab labels itself "small-scale
  demonstration of a registered design" and links each trial to its registered
  artifact + RESULTS.md section. Small-scale is a preview, never a claim.
- [ ] **R10.3.4** One-click export of a trial run (PNG + JSON) reusing
  `persistence.py`; exported JSON embeds the trial shape + seed + gate verdict.
- [ ] **Acceptance:** `uv run python main.py` → Trials tab → any trial completes
  in ≤60 s on CPU (GPU toggle where available, per AGENTS.md), with live curves,
  live control-arm badge, and the frozen θ-hash visible throughout; export
  round-trips.

## 🔒 R10.4 — The Standing Rules (so the payoff never gets lost again)

- [ ] **R10.4.1** **No naked JSON.** No commissioned trial is "done" until its
  figure exists and its row is filled in the Headline Table. The figure plan is
  declared at commissioning: `docs/preregistration_template.md` gains a
  **Figure** section (what will be plotted, from which panel of the artifact).
- [ ] **R10.4.2** **Scope labels ride along.** Every figure, demo mode, and
  RESULTS.md paragraph carries the R8.4 claim-scope label of its source run.
  A demo preview may never wear a registered claim's clothes.
- [ ] **R10.4.3** **Refutations ship with the same pipeline.** A refuted
  hypothesis gets a figure and a RESULTS.md paragraph on the same terms as a
  confirmed one (F5 is the template).
- [ ] **R10.4.4** Pull-based backlog rule: a backlog item is only pulled if it
  ends in a figure, a demo mode, or a RESULTS.md paragraph. Infrastructure work
  is justified by the seeing it enables, never by itself.
- [ ] **CI unchanged:** `ruff format --check` → `ruff check` → `pyright` →
  `pytest --cov` → `pip-audit`, plus the R10.2.4 figure-drift lock. **No new
  verification rounds are commissioned this round** — R10 spends the trust R6–R9
  built; it does not compound it.

## 📦 Deferred Register (parked from R9; pull only under R10.4.4)

- Split-MNIST CL prior-art revival through the R8 gates (`cl_backward_transfer_matched_memory.json`)
  — pull when a real-data forgetting figure is wanted alongside F1.
- AutoScientist boundary mapping (switch rate at which routing retention dies;
  IR-drop level where the Pareto frontier shifts) — pull when a boundary
  *frontier figure* is wanted; map only after the effect exists, per R9 rules.
- Physical-hardware validation — the production-library gate; unchanged, deferred.
- Demo checkpoint cleanup (`demo/checkpoints/`), `dummy.db` / stray DBs at repo
  root, ancient screenshot archives — one hygiene sweep, only when it blocks a
  figure or the demo first-run experience.

## Termination criterion

R10 closes when a stranger can, in one sitting: open the README, see five
figures, know what Computronium *does* (not just what it *is*), then run one
live trial and watch a frozen-weight network switch tasks without the hardware
hash moving. If that session produces the sentence **"oh, that's interesting"**
— the round did its job.
