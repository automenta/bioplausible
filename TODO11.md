# TODO11.md — Active Plan: The Library, Completed and Connected

> **Opened 2026-09-02 (draft).** Successor to [TODO10.md](TODO10.md)
> (R10 closed: D1–D7 demonstrate all six ontology axes; three consecutive
> green gate runs banked; the acceptance session — *read, run, change one
> thing, see it matter* — is available to any stranger). This plan contains
> **all remaining TODO10 work**: Register B capability pulls, the Register C
> hygiene pass, the carried registers, and the research-track spine's
> open prerequisites. Research catalog: [RESEARCH3.md](RESEARCH3.md).
>
> **Identity (reaffirmed from R10 v4):** Computronium is an ML library whose
> every claim is a live demonstration. Tests are the evidence system. A claim
> stands only while the current code re-demonstrates it, on demand, in under
> two minutes. Verification is continuous, not archival.
>
> **Prime directive:** *Nothing is claimed that the suite does not re-show at
> HEAD. The demo suite is the proof; the README quotes it; everything else is
> history or hypothesis.*
>
> **State:** OPEN — nothing landed yet. First moves: pre-flight checks below,
> then R11.2 (hygiene) in parallel with R11.1.2 (neuromorphic fidelity — the
> one Register B item whose pull condition is already arguable).

---

## 📜 Standing Directives (carried, binding)

These are session-established user directives and measured facts. They bind
every workstream below.

- **`benchmark_results/` stays untracked and gitignored — never re-add it**
  (user directive 2026-09-02, superseding earlier TODO10 language).
- **README carries no new code snippets or evidence links while the code is
  under active development.** README stays the hand-maintained two-locked-
  block index; evidence links live in `docs/RESULTS.md` and the gallery. The
  preset-factory locked block (TODO10 R10.2.7a) is added only when a demo
  test exercises a preset factory, and only after the active-development
  directive is lifted.
- **Test-execution discipline (user directive 2026-09-02):** never run tests
  without showing output and walltime (`--durations` is in addopts; pipe
  through `tail`/`grep`, never silent `head`-truncation of failures).
  Minimize redundant test executions: measure levers in throwaway scripts
  before touching tests; re-pin in lockstep, re-run once.
- **Device policy (measured 2026-09-02, RTX 3080):** the demo suite stays on
  **CPU** — the tiny Digital builds (784→32→10, batch 64, Python settle loop)
  are kernel-launch-bound, and CUDA ran *slower* (D2 hit the 60 s timeout).
  GPU-first applies where work is FLOP-bound: registered-scale studies,
  campaign fleets, large hidden dims, long horizons. Rule: *prefer GPU where
  appropriate — measured, not assumed* (AGENTS.md), with the demo-suite CPU
  verdict as the standing counter-example.
- **DataLoader workers:** `num_workers=2` measured faster at demo scale
  (13.2 s vs 20.7 s per epoch). `num_workers=0` is the *flake* mitigation
  (D7 precedent), not a speed rule.
- **GitHub CI is not yet in use** (user directive 2026-09-02): the gates that
  matter are the locally runnable invocations recorded in this plan; workflow
  edits are bookkeeping, not acceptance criteria.

---

## 🎯 The Demonstration Table (D1–D7; re-pinned 2026-09-02)

| # | Capability | Demo test | What the runner sees |
|---|------------|-----------|----------------------|
| D1 | Six-axis composition is real | `test_demo_compose_6axis.py` | Six-axis system trains (≈ 0.84 over the 600-batch cap); config round-trips (L6); 5-D build trained identically gives bitwise-equal θ (J1) |
| D2 | One trainer, every credit rule | `test_demo_swap_credit.py` | Three credit rules through byte-identical wiring except one constructor argument — all three learn (≈ 0.87 / 0.86 / 0.62) |
| D3 | The M-axis swap matters | `test_demo_swap_plasticity.py` | Routing visibly retains what null forgets across a task switch (A40/B40, mastery precondition asserted) |
| D4 | The memory profiler is honest | `test_demo_memory_budget.py` | The BPTT-profiled arm cannot run under a tight budget (walled, deterministically); the O(1)-memory arm runs |
| D5 | Frozen θ is a guarantee, bitwise | `test_demo_z3_frozen_theta.py` | θ hash identical across freeze→adapt→switch→restore; restored ψ reproduces stage-A accuracy exactly |
| D6 | The substrate axis is physical | `test_demo_substrate_swap.py` | One swapped substrate: digital 0.91 / mild IR-drop 0.78 / severe 0.12 — differential-pair conductances (int8 STE) |
| D7 | The D-axis settles in time | `test_demo_spike_settle.py` | One swapped D-axis argument: instant 0.87 / LIF 0.85; spikes counted per (layer, step); membranes bounded by threshold — the Lyapunov lock, live |

**Adding a demo row (standing recipe, from R10's Implementation Map):**
(1) demo test `tests/integration/test_demo_<name>.py` importing only from the
package root (+ its experiment module if benchmark-surface), emitting
`emit_run_record("D<N>", "<name>", data)` **before** asserting; (2) figure
factory in `computronium/visualization/gallery.py` + `_FACTORIES` entry;
(3) entry in `EXPECTED` of `tests/integration/test_gallery_lock.py`;
(4) rows in the Demonstration Table and `docs/RESULTS.md`; (5) optional
locked README block via `scripts/readme_snippets.json` + `<!-- lock: -->`
(only when the active-development directive is lifted). **Walltime budget
per new demo: ≤ 15 s at pinned scale** — prefer a batch cap + pinned floors
(D6/D7's `BATCH_CAP` pattern) over a full epoch; if the visible regime
needs more, the regime is wrong (R10.2.0 visibility rule). The gate budget
(≤ 90 s) is the sum of its parts; a new demo that busts it re-pins its own
cap before landing.

---

## ✅ Pre-flight (before any R11.1/R11.3 pull)

- [ ] Property locks green: `uv run pytest tests/property/ -q`
- [ ] Demo gate green ≤ 90 s: `uv run pytest tests/integration/ -k "demo or gallery_lock" -q` (8/8 at HEAD: 69 s)
- [ ] Drift locks green: `uv run pytest tests/unit/core/test_readme_snippet_lock.py tests/unit/core/test_root_exports.py -q`
- [ ] Gallery manifest + run records committed (figure-lock data layer tracked); worktree clean or deliberately dirty with the re-pin noted
- [ ] `comp repro` 7/8 (the `native_tile_ep` exclusion is the R11.1.3 pointer, not a regression)

---

## 🔩 R11.1 — Capability Pulls (TODO10 Register B, complete)

**Rule (R10.3.6, inherited): every pulled item lands with its demo test —
no test, no feature.** Pulls are condition-gated, not scheduled; the
sequencing below is the expected order, not a mandate.

- [ ] **R11.1.1 Neuromorphic substrate fidelity** (the open half of the
  substrate-fidelity item; pull condition already arguable). Decide by
  whichever makes a *visible* demo: (a) real spike dropout in
  `NeuromorphicSubstrate.inject_state_noise` / forward operator — the
  `sparsity` field (0.95) becomes functional, `x` is thinned to the active
  spike set per step; or (b) drop the cosmetic `sparsity` field and say so.
  The xfail at `tests/integration/test_energy_invariants.py::test_neuromorphic_substrate_sparsity`
  is the oracle: implement (a) → un-xfail it; implement (b) → delete it with
  the field. C9 passivity lock (deterministic noise cancellation) must keep
  passing either way — dropout must key off the same seeded stream. Lands
  with demo test (`test_demo_neuromorphic_swap.py` or a fourth D6 arm —
  D6 is the natural home: one more substrate argument through identical
  wiring, spike sparsity visible in the probe counts).
- [ ] **R11.1.2 Geometries.** `ConvGeometry` / `GraphGeometry` /
  `AttentionGeometry` / 3-D `SpatialLattice3D`. Pull one per demonstration
  need: Conv when a vision demo wants translation structure (CIFAR-shaped
  input), Graph when a graph-domain task wants GraphGeometry, Attention when
  an LM/sequence demo needs it, SpatialLattice3D when `neural_cube`'s
  geometry is next expressed as a coordinate. Geometry-DEFERRED skips stay
  skips until their pull. Each lands with a demo test exercising the G-axis
  swap through identical wiring (D1/D2 pattern).
- [ ] **R11.1.3 Tile × dynamics matrix (R3.4) + `native_tile_ep` repro.**
  tile_ep/pc/gnn/snn device-dynamics incompatibilities — fix or document as
  permanent xfail with precise reasons; same for tile_fa/tp/hebbian.
  Includes the `comp repro` 7/8 pointer: fix or document `native_tile_ep`
  (pre-existing; CI's reproducibility step excludes it explicitly). Pull on
  next touch of the tile family or when a demo wants the full tile matrix.
- [ ] **R11.1.4 Kernels (R4.1–R4.4).** FA feedback projection through the
  Substrate operator API; `SubstrateSettleKernel` in `KernelRegistry`; MEP
  Triton kernels (Muon, Fisher whitening) → Substrate update operator;
  sparse transpose-mask handling, ternary `init_scale` (un-xfail ternary
  equivalence), per-step `inject_state_noise`. Pull when the
  acceleration/kernel path is next touched or a substrate-axis demo needs
  them. Kernel-equivalence locks (max_diff < 1e-5) are the acceptance bar.
- [ ] **R11.1.5 Adapter heuristics (R3.5).** `_AdaptedSystem._infer_geometry`
  hardcoded (784→256,128→10) — recover heuristics from the deleted
  `adapter/` package. Pull when the strangler-fig adapter path is next
  touched (L6's Registry.to_system totality lock is the guardrail).
- [ ] **R11.1.6 `_TaskTrainer` gaps (R3.6).** Scheduler wiring, energy
  tracking, honor `tracker`/`safety_config`. Pull when hyperopt trials need
  them.
- [ ] **R11.1.7 Nudge-unwired settle paths (imp-29).** predictive_settling
  target clamp; diffusion target term. Pull when a campaign manifest needs
  those coordinates fully wired.
- [ ] **R11.1.8 Ontology facade merge (R2.2 residual).**
  `ontology/_substrate.py` impl vs `ontology/substrate/` facade; same
  pattern for `_dynamics.py` vs `dynamics/`. Merge the parallel legacy/new
  pairs on next ontology-structure touch; grep for other twins while there
  (feeds R11.2's PlasticityConfig item).
- [ ] **R11.1.9 Timing-asymmetric STDP wired to the 5-D pipeline** (from the
  `create_spiking_snn_mlp` Register-C row; the D-axis's remaining depth).
  The pipeline-facing rate-coded surrogate has no error signal (chance on
  MNIST); genuine STDP lives unwired in `core/local_learning/rules/spiking.py`.
  This is also the standing **R10.3.5 refutation candidate**: a visible
  refutation demo (Hebbian-only plateau) ships with the same pipeline as any
  success. Pull when the SNN family is next touched or a research paragraph
  needs it; a learning claim additionally needs the trace-based rule in the
  pipeline.
- [ ] **R11.1.10 LazyStateDynamics at demo scale** (the D-axis's other
  remaining depth). Research-track/register material until a visible regime
  exists — pull only when a demo regime shows on-demand activation visibly
  (e.g., settle-count contrast on a large-dim build).
- [ ] **R11.1.11 Domain extensions (README-planned entries; pull-based, do
  not open-endedly build).** `wikitext2`/`penn_treebank` (LM),
  `mountain_car`/`lunar_lander` (RL), `diabetes`/`california_housing`
  (tabular regression), `ett_h1` (time series), PDE suite
  (Heat/Wave/Burgers/Navier-Stokes). Pull conditions: the PDE suite pulls
  **with** the RESEARCH3 Physics-informed proof item (its systems ladder);
  the others pull when a benchmark, demo, or research paragraph needs the
  domain. Planned entries never count toward implemented totals until
  landed with a test.

## 🧹 R11.2 — The Hygiene Pass (TODO10 Register C, complete)

One dedicated pull; either fix forward or scope explicitly. This also
unblocks RESEARCH3 **PR-0** (verification gate: pytest/pyright/ruff green),
which gates every empirical item.

- [ ] **R11.2.1 Ruff baseline (~4.8k findings).** `ruff check .` on
  pre-existing modules (max-args=5, preview rules, S-rules on subprocess).
  Decide per class: fix forward, configure, or per-line noqa with reason.
  End state: `ruff check .` clean at HEAD.
- [ ] **R11.2.2 Pyright baseline.** pyright-basic findings in
  `core/pipeline.py` (SystemState/CompositeState confusion),
  `core/system_trainer/joint.py` (`JointSystem` import symbol, TypeVar
  shadowing, `SystemContext` scoping, `PlasticityConfig` twin-class
  assignment), `core/plasticity/routing.py` + `fast_weights.py`
  (pseudo-gradient union-type handling), `cli/parity.py`,
  `tests/property/test_axis_certifications.py`. Then decide: repo-wide
  strict, or scope the type-check gate to the packages that must hold.
- [ ] **R11.2.3 Root `PlasticityConfig` twin-class resolution** (found in
  the R10.2.1 audit): root resolves to `computronium.state`'s twin, a
  different class from `core.joint.transition.PlasticityConfig`. Fold into
  the R11.1.8 facade merge.
- [ ] **R11.2.4 Joint `to_spec`→`from_spec` round-trip broken:**
  `from_spec` calls `GeometryConfig(**spec["geometry"])` but `to_spec`
  embeds `params`/`recurrent_weight` keys → TypeError. Next touch of
  `core/system_trainer/joint.py` — or now, since R11.2.2 is already in that
  file.
- [ ] **R11.2.5 `FeedforwardGeometry._build_layers` ignores
  `GeometryConfig.init_scale`** (three init scales gave byte-identical
  results in the D7 sweep; every factory's `init_scale` argument is
  decorative on feedforward builds). Next touch of geometry construction —
  natural to bundle with R11.1.2.
- [ ] **R11.2.6 imp-4** — Pyright full strict on ontology (131 findings;
  annotation work in `_dynamics`/`geometry`/`update`).
- [ ] **R11.2.7 imp-8** — `compute_energy` duplication across
  Energy/Spike/Instantaneous/Diffusion → extract `_energy_from_state`.
- [ ] **R11.2.8 imp-19** — `FrontierRecord.seed` legacy default 42 →
  required at next schema break.
- [ ] **R11.2.9 imp-23** — `substrate_coupled` plasticity
  engagement-verified only; probe fixed-dim `step` assumptions.
- [ ] **R11.2.10 imp-26** — params-moved learning locks for the remaining
  README-table factories (FA lock exists). When a preset factory gains its
  lock/demo, add the preset block to the README sidecar map (TODO10
  R10.2.7a superseded-note's standing pointer).
- [ ] **R11.2.11 imp-27** — rename rebuilder-style `settle` implementations
  whose names mislead.
- [ ] **R11.2.12 imp-30** — deployments' `family="tile"` registrations
  CLI-orphaned → fold into `family="equitile"` or drop.
- [ ] **R11.2.13 imp-36** — campaign stability axis non-discriminative →
  cheap per-episode proxy.
- [ ] **R11.2.14 imp-37** — latency objective is wall-clock noise →
  repeated-timing methodology or deterministic proxy. Blocks any
  task-scale latency claim.
- [ ] **R11.2.15 imp-41** — `demo/tests/` 28 stale failures → rewrite or
  delete. Resolves with the R11.4 UI rebuild (the demo gets rebuilt, not
  patched), or before it if the path is touched.
- [ ] **R11.2.16 R3.8** — `natural_language_query` TF-IDF weighting; derive
  `V_nudged = free energy + β·loss` to strengthen the PC Lyapunov xfail.
- [ ] **R11.2.17 README factual correction:** `create_snn_mlp` row
  advertises SpikeIntegration × TemporalTrace × Euclidean; the factory
  builds Instantaneous × LocalGoodness for trainer compatibility. Either
  land a true-Spike SNN factory coordinate once R11.1.9 gives the pipeline
  a real error signal, or correct the row (R10.2.7 rules).
- [ ] **R11.2.18 `test_scaling_invariants` xpass** —
  `deep_network_accuracy[100]` pre-existing xpass. Next touch of that file.
- [ ] **R11.2.19 One-command re-pin** (the effort lever for the standing
  re-pin ceremony). A `scripts/repin.sh` (or `comp gallery --repin`):
  demo suite → `comp gallery` render → drift locks + snippet lock, one
  invocation, nonzero on any red. Justified by R11.5.6: it is the tool that
  lets the suite stay truthful cheaply after every demo touch. Measure
  first (it is mostly composition, no new logic).
- [ ] **R11.2.20 Timebox the pass** (E-2 analog): R11.2 as a whole gets
  three working sessions; a finding class that resists is scoped out
  explicitly (with the reason recorded) rather than stretched. Infra
  friction doesn't consume the box.

## 🔬 R11.3 — Research-Track Pulls (TODO10 Register A; RESEARCH3 spines)

**Front-page rule: outputs feed the corroboration appendix and papers —
never the front page.** Every pull inherits RESEARCH3's Execution Protocol
(E-1 three-rung ladder, E-2 timeboxes, E-3 reproducibility, E-10 control
set, E-11 decision log). Two pulls are *scheduled* (not merely condition-
gated): **R11.3.1 (PR-9) pulls immediately after R11.2.2 lands** — hygiene
green is PR-0's type/lint half, and PR-0 gates every empirical item; **R11.3.3
(PR-5) pulls immediately after its calibration harvest exists** — the
harvest is free (the demo suite's pinned configs and outcomes are exactly
PR-7's known-good/known-bad set). The rest pull when their research
consumer exists.

- [ ] **R11.3.1 PR-9 — Campaign commissioning** (Tangible Checkpoint 3;
  the gateway to every unattended result). One tiny AutoScientist campaign
  completing a full **iterate → interrupt → checkpoint → resume** cycle
  end-to-end, recorded: the artifact is the campaign directory
  (`records/episodes.json` + checkpoint + resume event in its manifest)
  plus a one-paragraph commissioning note in `DECISIONS.md` (E-11) stating
  what was interrupted and what resume replayed. Machinery is built
  (`CampaignStack`: deterministic resume, skip-not-duplicate, YAML+SQLite
  checkpoints) but the commissioned cycle is not yet a recorded run.
- [ ] **R11.3.2 PR-2 — θ-invariance audit harness.** Snapshot → freeze →
  run → re-snapshot → exact-diff as a reusable context manager with
  per-seed reports (D5 demonstrates the guarantee; the harness makes it a
  library feature for Z3 / Algorithm-Migration / continual-learning runs).
- [ ] **R11.3.2b PR-3a — Software resource instrumentation.** `ResourceUsage`
  + `core/profiling.py` wired into suite runners emitting proxy
  FLOPs/memory/latency; feeds Z3 energy metrics (proxy tier), L2
  effective-FLOPs, the 𝒞 vector.
- [ ] **R11.3.3 PR-5 — Calibrated stability guard.** ROC-calibrated kill
  thresholds (<5% false-kill on known-good, >95% kill rate, <10%
  overhead) from the demo-suite/PR-7 harvest; `_fast_proxy` vs
  full-Jacobian disagreement rate quantified. PR-0's shipped
  `PowerPreregistration`/embedded-control machinery is the substrate.
- [ ] **R11.3.4 AutoScientist M-axis frontier** (Tangible Checkpoint 5 —
  the first *finding* figure). One axis at a time: pin S/G/D/C/U, sweep
  M ∈ {Null, Routing, FastWeight, RuleState}; ResourceUsage aggregated
  post-hoc; 2-D Pareto projections annotated per knee.
- [ ] **R11.3.5 Z3 flagship registered commission** (Tangible Checkpoint 6).
  ≥95% on all three tasks at exact Δθ=0 within ≤20% of fine-tuning steps;
  ≥5 seeds; baselines a–d incl. ICL bridge. Runs on the fixed instrument
  (fresh Adam at every boundary). If it falsifies: L1 adaptation figure
  substitutes; boundary condition becomes the publication.
- [ ] **R11.3.6 Walled-regime boundary commission** — pilot-mapped
  (`benchmark_results/boundary_map_pilot.json`); pull when the research
  track needs the boundary location as a preregistered claim.
- [ ] **R11.3.7 Task-family generalization** — the linear-teacher boundary
  raises which task families behave differently; pull when a research
  paragraph needs a second task family (design inherits R8 gates + R9
  method rules wholesale).
- [ ] **R11.3.8 CL prior-art revival (Split-MNIST)** — through the R8
  gates; pull when the research track wants a real-data retention study.
- [ ] **R11.3.9 AutoScientist boundary mapping** — switch rate where
  routing retention dies; IR-drop level where the Pareto frontier shifts;
  map only after the effect exists.
- [ ] **R11.3.9b Registered-artifact provenance backfill** — archaeology
  (timestamps vs git log) when the corroboration appendix is extended;
  "provenance unknown" is an acceptable, honest label.
- [ ] **R11.3.10 PR-2/PR-4/PR-6/PR-8 companions** — θ-audit harness (=
  R11.3.2), statistics kit completion, fairness-contract draft (writing
  only, zero compute — natural waiting-period work per E-8), export-parity
  round-trip. Pull with their consumers (Z3, benchmark paper, Edge).

## 🚀 R11.4 — Adoption Surface (TODO10 Register E)

- [ ] **R11.4.1 Drop-in PyTorch wrapper (CP-C).**
  `torch.nn.ComputroniumLinear` (+ conv/embedding as needed): one-line swap
  of `nn.Linear` + optimizer for an EqProp/FF coordinate; free/nudged
  phases, settling loops, ψ bookkeeping internal; `NullPlasticity`+backprop
  falls back to native behavior bit-for-bit. Acceptance: a script written
  by someone unfamiliar with internals runs unmodified except the swapped
  line; gradients/accuracy match the hand-written loop within noise;
  `torch.compile` + LR-scheduler smoke tests. Pull after the API stabilizes
  (R11.1/R11.2); fills any waiting period (E-8).
- [ ] **R11.4.2 PR-6 fairness contract draft.** Per-rule tuning budgets
  (GPU-hours, not epochs), early-stopping policy, seeds, data splits, the
  ICL-bridge scale-matching rule — written once, consumed by four items.
  Zero compute; draft during any blocked period.
- [ ] **R11.4.3 R11 Live demo UI.** Compose tab as primary surface (pick any
  6-axis coordinate, hit run, watch curves), demo suite as pre-built
  presets, ψ visualizer + θ-hash badge as library features, one-click
  export. **Ships only when the API is stable** — the demo presents the
  library; it does not design it. imp-41 resolves here (the `demo/tests/`
  stale suite gets rebuilt, not patched).
- [ ] **R11.4.4 Hygiene sweep.** `demo/checkpoints/`, stray DBs at repo root
  (`dummy.db`, `execution_state.db`), ancient screenshot archives — only
  when it blocks a figure, a test, or a fresh checkout.

## 📋 Register D — Carried Deferred (unchanged from TODO10)

| Item | Reason |
|------|--------|
| Coverage floor (~16.8%) | opt-in `--cov`; raise after API stabilizes |
| Rocq general-case formalization | CP-B pull-based; diagonal case done with paper proof; ψ-coverage proposition is the next statement |
| `test_ontology_parity.py` decomposition | Slow-marked; split fast/slow only if gate iteration speed demands |
| Physical hardware (PR-3b / CP-D) | Latency-gated procurement per RESEARCH3; proxy tier (R11.3.2b) decouples all software-side claims |

---

## 🔒 R11.5 — The Standing Rules (R10.3 verbatim, renumbered)

- [ ] **R11.5.1 No test, no feature.** Every feature ships with an
  integration test that demonstrates it working end-to-end.
- [ ] **R11.5.2 No claim without a live demonstration.** When a test is
  removed, flaky, or failing, its claim disappears from the front page
  automatically — the system degrades to silence, never to stale assertions.
- [ ] **R11.5.3 Corroboration never carries.** Registered numbers are
  history: labeled, scoped, provenance-annotated, confined to RESULTS.md's
  back section and the research track.
- [ ] **R11.5.4 Scope honesty.** Demo-scale demonstrations speak for demo
  scale; registered claims live in the research track. Neither borrows the
  other's clothes.
- [ ] **R11.5.5 Refutations ship with the same pipeline** — same figure
  factory, same docs, same terms (R11.1.9's Hebbian plateau is the standing
  candidate).
- [ ] **R11.5.6 Pull rule.** A backlog item is pulled only if it ends in a
  live demonstration, a gallery figure, or a RESULTS.md capability
  paragraph. Infrastructure is justified by the capability it lets the
  suite show, never by itself.
- [ ] **R11.5.7 Gates (tiered, per AGENTS.md test-execution tiers).**
  Per-commit duties are **scoped to changed files** (format + lint + pyright
  + targeted tests). The standing fast gates — property suite, demo gate
  (`pytest tests/integration/ -k "demo or gallery_lock"`, ≤90 s), drift
  locks, positive control — run on their triggers (demo/gallery/lock-
  adjacent changes), never per-edit. The full CI order and repo-wide
  ruff/pyright are R11.2's deliverable and a round-close event, not a
  habit. No new verification rounds are commissioned in R11; R11 spends
  R6–R10's trust.

---

## 👁️ Watch (triggers convert to pull items; history canonical in TODO9/TODO10)

- **axis_probe `[2-0]` flake** — no recurrence since 2026-08-31.
- **CUDA tolerance boundaries** shift xfail edges — CPU/GPU tests kept
  separate; construction seeding in place.
- **R9.1 lr=0.03** calibrated for the 40-episode budget; read A-mastery
  (~0.5 floor) before reading retention; at A20/B20 the retention effect
  *reverses* — mastery precondition is load-bearing for D3.
- **Control-band sizing (imp-59):** preregistrate the at-chance band from
  the registered N of the control arm's scored samples.
- **Smoke-scale campaign deltas (imp-54):** capped at chance by the
  non-stationary stream — accumulated-learning/retention claims run the
  persistent-θ chain only.
- **Budget commissioning gate (R9.2):** a feasible arm's walk is identical
  under every budget that admits it — never read walled arms' absence as
  "lost".
- **Transient d6 hash (2026-09-02):** one early session emission hashed
  differently (6ea65…) and never reproduced (3 subsequent runs identical at
  989683…, solo and in-suite) — the figure lock caught it as designed. If it
  recurs, treat as a real determinism defect, not a re-pin.
- **Z3 demo `MetaRecipe` defaults:** pins `meta_train_epochs=4` (fresh-ψ
  floor ≈ 0.68, restored beats floor+0.1 at 1.0). If defaults or task
  generators change, re-run the 3/4/5-epoch calibration sweep before
  re-pinning.
- **D7 spike watch:** asserts `total_spikes > 100` (observed ≈ 1.8k),
  `membrane_max ≤ 1.0` (structural). If default drift silences the hidden
  layer, re-run the D7 sweep before re-pinning. Sub-threshold-at-init is
  expected.
- **D6/D7 wall dials:** `BATCH_CAP` in `test_demo_substrate_swap.py` (1000)
  and `test_demo_spike_settle.py` (300); D1/D2 caps (600) are the other
  dial. Under differential-pair semantics the D6 staircase is 0.5→0.89,
  1.5→0.78, 3.0→0.56, 4.0→0.42, 6.0→0.22, 8.0→0.12 — severe arm sits at 8.0
  with ceiling 0.4; don't move it below 6.0-class without re-sweeping.
- **`_LAZY`↔`__all__` lockstep** — resolved 2026-09-02 via
  `tests/unit/core/test_root_exports.py`; the lock holds it.
- **Evidence layer tracked** — manifest + run records + registered figure
  committed; gallery lock live on a fresh clone. `benchmark_results/`
  stays untracked (standing directive).

---

## 🎯 Tangible-Result Checkpoints (R11 edition; each with a materialization condition)

1. **Capability pulls demonstrated (R11.1):** Register B items land each
   with its demo test green in the gate — the Demonstration Table grows
   axes/geometries/substrates as visible capabilities. **Gate:** demo suite
   green, walltime ≤ 90 s.
2. **Truthful gates (R11.2):** ruff/pyright green at HEAD or explicitly
   scoped; the CI order (`format → check → pyright → pytest → pip-audit`)
   becomes enforceable rather than aspirational. **Unblocks PR-0.**
3. **Commissioned campaign stack (R11.3.1 / RESEARCH3 PR-9):** one full
   iterate → interrupt → checkpoint → resume cycle recorded — the gateway
   to every unattended result.
4. **Calibrated stability guard (R11.3.3 / PR-5):** ROC-calibrated kill
   thresholds from the demo/PR-7 harvest (<5% false-kill, >95% kill, <10%
   overhead); the failure manifesto starts accumulating as a dataset.
5. **Adoption surface (R11.4):** wrapper v1 (pip-installable, smoke suite,
   one-line swap) and/or the live demo UI — the multiplier for everything
   else.
6. **The first research-shaped result (R11.3.4):** M-axis Pareto frontier
   over 𝒞, annotated with which primitive owns each knee — the first figure
   that is a *finding*, not a demonstration.
7. **The discovery bet (R11.3.5):** Z3 flagship at registered scale; either
   outcome is tangible per the pre-registered fallback.

Sequencing: 1–2 are R11 core (parallel; hygiene unblocks everything); 3–4
pull when the research track is wanted (3 before 4); 5 after the API
stabilizes; 6–7 are RESEARCH3 CP-A's tail. No checkpoint blocks on a later
one.

---

## 📝 Notes for the Next Editor

- On the first R11 landing, mark TODO10.md's State header **CLOSED →
  superseded by TODO11.md** (one line; TODO10 stays as history and the
  Watch record's canonical archive).
- The README-freeze directive has a sunset condition to *ask* about (never
  self-lifted — it is a user directive): lift it for the wrapper/GIF work
  only when R11.2 is closed and the R11.1 core pulls (neuromorphic,
  geometries, tile) are green in the gate — i.e., the API the README would
  quote is the API that exists and demonstrates itself.

- R10 closed cleanly at 69 s gate walltime (D1/D2 loader caps pulled,
  Register C item closed); D1 ≈ 0.84, D2 ≈ 0.87/0.86/0.62 are the pinned
  numbers everywhere (docstrings, README blocks, RESULTS.md, manifest).
- Pull one Register item per landing; each lands with its demo test, figure
  or RESULTS.md paragraph (R11.5.6). Don't pull infrastructure "just in
  case".
- Keep hygiene separate from capability pulls — R11.2 is one dedicated
  pass, not a tax on every feature.
- When any demo test is touched, re-pin its docstring regime, the README
  locked blocks (if designated), RESULTS.md rows, and the gallery manifest
  **in the same change**; the figure/snippet locks fail loudly otherwise —
  that is the system working, not a nuisance.
- RESEARCH3 protocol (E-1 smoke → pilot → full; E-11 DECISIONS.md) governs
  every R11.3 pull. Infra-failures don't consume tuning rounds.

## Termination criterion

R11 closes when a stranger can, in one sitting: compose a system from *any*
geometry, substrate, and dynamics the ontology declares (not just the
Feedforward/Recurrent/Digital/Memristive/EnergyMinimization set R10
demonstrated), watch it train in the demo suite **and** in the UI, and find
the repo's own gates — ruff, pyright, property locks, demo gate, figure
lock — green at HEAD without caveats. The library is then *complete relative
to its own ontology*: every axis declares only primitives that exist and
demonstrate themselves. Research claims remain where they belong — the
corroboration appendix and RESEARCH3, pull-based, never the front page.
