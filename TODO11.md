# TODO11.md — Active Plan: The Library, Complete

> **Opened 2026-09-02.** Successor to [TODO10.md](TODO10.md) (R10 sprint closed:
> three consecutive green gate runs, D1–D7 covering all six ontology axes, 69 s
> walltime). Research catalog: [RESEARCH3.md](RESEARCH3.md).
>
> **Identity decision (reaffirmed):** Computronium is an ML library whose every
> claim is a live demonstration. The demo suite (D1–D7) is the proof; the
> README quotes it; everything else is history or hypothesis.
>
> **State:** OPEN — R10 complete. R11 begins with the remaining library-
> completeness items (Register B), a dedicated code-quality pass (Register C),
> and the first successor-round deliverables (Register E).

---

## 🎯 The Demonstration Table (unchanged — R10 re-pinned 2026-09-02)

| # | Capability | Demo test | What the runner sees | Registered corroboration (history, not claim) |
|---|------------|-----------|----------------------|------------------------------------------------|
| D1 | Six-axis composition is real | `test_demo_compose_6axis.py` | A system composed from all six axes trains; its config round-trips to an identical system; the J1 Zero-Extension invariant holds at train scale end-to-end | EqProp parity & energy proofs (L4-locked continuously) |
| D2 | One trainer, every credit rule | `test_demo_swap_credit.py` | Three credit rules through byte-identical wiring except one constructor argument — all three learn | Deep-credit registered study |
| D3 | The M-axis swap matters | `test_demo_swap_plasticity.py` | Routing visibly retains what null forgets across a task switch — designed so the gap is seen, not computed | Retention registered study (16 seeds) |
| D4 | The memory profiler is honest | `test_demo_memory_budget.py` | The backprop-profiled arm simply **cannot run** under a tight budget (walled, deterministically); the O(1)-memory arm runs | Memory-budget registered study |
| D5 | Frozen θ is a guarantee, bitwise | `test_demo_z3_frozen_theta.py` | θ's hash identical across the whole freeze→adapt→switch→restore run; restored ψ reproduces stage-A accuracy *exactly* | Z3 registered study |
| D6 | The substrate axis is physical | `test_demo_substrate_swap.py` | One swapped substrate through identical wiring: digital learns, mild IR-drop learns less, severe IR-drop walls at chance — differential-pair conductances (int8 STE) carrying the signed weights | — |
| D7 | The D-axis settles in time | `test_demo_spike_settle.py` | One swapped D-axis argument through identical wiring: the instantaneous pass and the layer-wise LIF settle both train (≈0.87 / 0.85); the trained LIF network fires visibly (every threshold crossing counted per settle step) and its membranes come back bounded by the spike threshold — the Lyapunov lock, live | — |

---

## 📦 Register B — Library Completeness (pulled under R10.3.6)

**Rule: every pulled item lands with its demo test (`tests/integration/test_demo_*`) — no test, no feature.** These are the axes' missing primitives; the demo suite can only demonstrate what exists.

| Item | Contents | Pull condition |
|------|----------|----------------|
| **Geometries** | `ConvGeometry` / `GraphGeometry` / `AttentionGeometry` / 3D `SpatialLattice3D` — geometry-DEFERRED skips stay skips | Science runs on Feedforward/Recurrent/Tile at MLP scale today. Pull when a demo test or campaign manifest needs the geometry (e.g., a vision demo test wants Conv; a graph domain task wants GraphGeometry). |
| **Substrate fidelity — neuromorphic half** | Neuromorphic: real spike dropout or drop the cosmetic `sparsity` field. Memristive half done (differential-pair conductances, int8 STE, re-pinned 2026-09-02) | Pull when a neuromorphic-axis claim or demo test needs the fidelity to be real |
| **Tile × dynamics matrix** | tile_ep/pc/gnn/snn device-dynamics incompatibilities; tile_fa/tp/hebbian — fix or document as permanent xfail with precise reasons | Pull on next touch of the tile family, or when a demo wants the full tile matrix. |
| **Adapter heuristics** | `_AdaptedSystem._infer_geometry` hardcoded (784→256,128→10) — recover heuristics from the deleted `adapter/` package | Pull when the strangler-fig adapter path is next touched. |
| **`_TaskTrainer` gaps** | Scheduler wiring, energy tracking, honor `tracker`/`safety_config` | Pull when hyperopt trials need them. |
| **Kernels** | FA feedback projection through the Substrate operator API; `SubstrateSettleKernel` in `KernelRegistry`; MEP Triton kernels (Muon, Fisher whitening) → Substrate update operator; sparse transpose-mask handling, ternary `init_scale` (un-xfail ternary equivalence), per-step `inject_state_noise` | Pull when the acceleration/kernel path is next touched or a substrate-axis demo needs them. |
| **Nudge-unwired settle paths** | predictive_settling target clamp; diffusion target term | Pull when a campaign manifest needs those coordinates to be fully wired. |
| **Substrate facade merge** | `ontology/_substrate.py` impl vs `ontology/substrate/` facade — facade is fine; consider merge + grep for other parallel legacy/new pairs (same pattern exists for `_dynamics.py` vs `dynamics/`) | Pull on next ontology-structure touch. |

---

## 🧹 Register C — Code-Quality Backlog (dedicated hygiene pass)

| Item | Pull condition |
|------|----------------|
| Root `PlasticityConfig` resolves to `computronium.state`'s twin, which is a **different class** from `core.joint.transition.PlasticityConfig` (pyright flags the resulting confusion in `core/system_trainer/joint.py`) | Next merge of `computronium/state/` with `computronium/core/joint/` (R2.2-residual pattern); the root-exports lock test pins what exists today |
| Repo-wide ruff/pyright hygiene: `ruff check .` reports ~4.8k pre-existing findings (max-args=5, preview rules, S-rules on subprocess) and pyright-basic flags `pipeline.py` / `core/system_trainer/joint.py` / `plasticity/{routing,fast_weights}.py` / `cli/parity.py` / `tests/property/test_axis_certifications.py` — CI's ruff and pyright steps fail at HEAD independent of R10 | Dedicated hygiene pass: either fix forward or scope the CI steps to the gates that are meant to hold (property/demo/lock suites) |
| Joint `to_spec`→`from_spec` round-trip broken — `from_spec` calls `GeometryConfig(**spec["geometry"])` but `to_spec` embeds `params`/`recurrent_weight` keys → TypeError | Next touch of `core/system_trainer/joint.py` (or when a demo/figure needs joint-spec round-trips) |
| imp-4 — Pyright full `strict` on ontology (131 findings; torch `Unknown` tracking; annotation work in `_dynamics`/`geometry`/`update`) | Next annotation pass on those modules |
| imp-8 — `compute_energy` duplication across Energy/Spike/Instantaneous/Diffusion → extract `_energy_from_state(state, geometry)` | Next touch of any dynamics module |
| `FeedforwardGeometry._build_layers` ignores `GeometryConfig.init_scale` (nn.Linear defaults only; found in the D7 sweep — three init scales gave byte-identical results) | Next touch of geometry construction, or when a demo needs weight-scale control |
| imp-19 — `FrontierRecord.seed` legacy default 42 → required at next schema break | Schema break |
| imp-23 — `substrate_coupled` plasticity engagement-verified only; probe fixed-dim `step` assumptions | Next touch of that plasticity |
| imp-26 — params-moved learning locks for the remaining README-table factories (FA lock exists) | Next touch of each factory |
| imp-27 — rename rebuilder-style `settle` implementations whose names mislead | Next touch of each implementation |
| imp-30 — deployments' `family="tile"` registrations CLI-orphaned → fold into `family="equitile"` or drop | Next touch of deployment registry |
| imp-36 — campaign stability axis non-discriminative → cheap per-episode proxy | When a manifest needs stability contrast |
| imp-37 — latency objective is wall-clock noise → repeated-timing methodology or deterministic proxy | Before any task-scale latency claim |
| imp-41 — `demo/tests/` 28 stale failures → rewrite or delete | Next demo-test touch (or R11, where the demo gets rebuilt) |
| R3.8 — `natural_language_query` TF-IDF weighting; derive `V_nudged = free energy + β·loss` to strengthen the PC Lyapunov xfail | Next touch of the knowledge base / PC verification |
| README `create_snn_mlp` row advertises SpikeIntegration × TemporalTrace × Euclidean, but the factory builds Instantaneous × LocalGoodness for trainer compatibility | Next README factual-correction pass (R10.2.7 rules) |
| `create_spiking_snn_mlp` runs end-to-end post-D7-pull (it crashed at settle step 1 before) but plateaus at chance on MNIST: the pipeline-facing STDP surrogate is a pure Hebbian correlation (no error signal); timing-asymmetric STDP lives in `core/local_learning/rules/spiking.py`, unwired to the 5-D pipeline. R10.3.5 refutation candidate | When the SNN family is next touched, or a research paragraph needs it |
| test_scaling_invariants xpass — `deep_network_accuracy[100]` pre-existing xpass recurred in the full gate | Next touch of that file |

---

## 📋 Register D — Carried Deferred (from TODO8; unchanged)

| Item | Reason |
|------|--------|
| Coverage floor (~16.8%) | opt-in `--cov`; raise after API stabilizes |
| Rocq general-case formalization | CP-B pull-based; diagonal case done with paper proof |
| `test_ontology_parity.py` decomposition | Slow-marked; split fast/slow only if gate iteration speed demands |
| Physical hardware (PR-3b / CP-D) | Latency-gated procurement per RESEARCH3 |

---

## 🚀 Register E — Successor Rounds

### R11 — Live demo (compose-and-run UI)
Compose tab as primary surface (pick any 6-axis coordinate, hit run, watch curves), demo suite as pre-built presets, ψ visualizer + θ-hash badge as library features, one-click export. Built only when the API is stable — the demo presents the library; it does not design it. imp-41 resolves here (the demo gets rebuilt, not patched).

### Drop-in PyTorch wrapper (RESEARCH3 CP-C)
Remove adoption friction — users swap one line, not their training loop.
- `torch.nn.ComputroniumLinear` (+ conv/embedding): replaces `nn.Linear` + optimizer with an EqProp or Forward-Forward coordinate; free/nudged phases, settling loops, ψ bookkeeping handled internally; `NullPlasticity`+backprop coordinate falls back to native behavior bit-for-bit.
- Compatibility targets: DDP wrapping, LR schedulers, `torch.compile` smoke test, torchvision-style model zoo integration example.
- Acceptance test: training script written by someone unfamiliar with Computronium internals runs unmodified except the swapped line; gradients/accuracy match hand-written loop within noise.

### Hygiene sweep
`demo/checkpoints/`, stray DBs at repo root — `dummy.db`, `execution_state.db` — ancient screenshot archives — only when it blocks a figure, a test, or a fresh checkout.

---

## 👁️ Register F — Watch (triggers convert to pull items; history canonical in TODO9.md/TODO10.md)

- axis_probe `[2-0]` flake — no recurrence since 2026-08-31; still watching.
- CUDA tolerance boundaries shift xfail edges — CPU/GPU tests kept separate; construction seeding in place.
- R9.1 lr=0.03 is calibrated for the 40-episode budget — re-calibrate on schedule/budget changes; read A-mastery (~0.5 floor) before reading retention. Confirmed live at demo scale (2026-09-02 pre-flight): at A20/B20 the retention effect *reverses* — mastery precondition is load-bearing for D3.
- Control-band sizing (imp-59): preregistrate the at-chance band from the registered N of the control arm's scored samples.
- Smoke-scale campaign deltas are capped at chance by the non-stationary stream (imp-54) — accumulated-learning/retention claims run the persistent-θ chain only; CampaignStack rebuilds θ per episode regardless of teacher stationarity.
- The budget is a commissioning gate (R9.2): a feasible arm's walk is identical under every budget that admits it — never read walled arms' absence as "lost", or feasible arms' repeated readout as new evidence.
- `_LAZY` map and `__all__` can drift apart silently (both hand-maintained) — **resolved 2026-09-02**: the R10.2.1 audit found divergence (15 `_LAZY` names missing from `__all__`, one dead `OntologyConfig` entry) and added the standing lock `tests/unit/core/test_root_exports.py` (lockstep + every lazy entry resolves).
- Demo-suite evidence layer committed 2026-09-02: `docs/figures/manifest.json`, `docs/figures/run_records/`, and `docs/figures/registered/` are tracked — the figure lock's data layer is live on a fresh clone.

---

## 🔧 R11 — The Standing Rules (inherit R10.3 + one addition)

- [ ] **R11.1 No test, no feature.** Every feature ships with an integration test that demonstrates it working end-to-end. The test is the example.
- [ ] **R11.2 No claim without a live demonstration.** The front page (README, gallery, Demonstration Table) carries only claims the suite re-shows at HEAD. When a test is removed, flaky, or failing, its claim *disappears from the front page automatically* — the system degrades to silence, never to stale assertions. Fix the demonstration; the claim resumes.
- [ ] **R11.3 Corroboration never carries.** Registered numbers are history: labeled, scoped, provenance-annotated, and confined to RESULTS.md's back section and the research track. A figure caption or README line that leans on a stored number is a defect.
- [ ] **R11.4 Scope honesty.** Demo-scale demonstrations speak for demo scale; registered claims live in the research track where preregistration governs. Neither borrows the other's clothes.
- [ ] **R11.5 Refutations ship with the same pipeline** — same figure factory, same docs, same terms. A demonstration that shows the library failing somewhere is as welcome as one that shows it succeeding.
- [ ] **R11.6 Pull rule:** a backlog item is pulled only if it ends in a live demonstration, a gallery figure, or a RESULTS.md capability paragraph. Infrastructure is justified by the capability it lets the suite show, never by itself and never by a stored result that needs it.
- [ ] **R11.7 Demo UI is a library feature, not a sprint.** The R11 compose-and-run UI (Register E) ships only when the underlying API is stable and the demo suite is green at HEAD. The UI does not design the API; it reflects it.
- [ ] **CI:** `ruff format --check` → `ruff check` → `pyright` → `pytest --cov` → `pip-audit`, plus: demo tests, positive control, drift lock (R10.2.8), figure lock (R10.1.4). Demo tests join the fast gate. New modules meet the same strict-typing bar as everything else. **Repo-wide ruff/pyright hygiene is a separate pass (Register C); the fast gate holds on property/demo/lock suites only.**

---

## 🎯 Tangible-Result Checkpoints (what the investment returns)

R11 is an investment; these are the returns, each with a materialization condition.

1. **Library completeness (R11 sprints 1–2):** Register B items pulled under R11.6, each with its demo test green in the suite — the Demonstration Table grows axes/geometries/substrates as visible capabilities, not promised ones. **Gate:** `pytest tests/integration/ -k demo` green, walltime ≤90 s.
2. **Hygiene pass (R11 sprint 3):** Register C items resolved — repo-wide `ruff check` and `pyright` clean on the full tree (or CI steps scoped to the gates that matter with explicit allow-lists). No pre-existing findings block unrelated work.
3. **Drop-in PyTorch wrapper (R11 sprint 2–3, parallel):** CP-C deliverable — pip-installable module, smoke-test suite, one-line-swap README GIF. Adoption multiplier for every other deliverable.
4. **Live demo UI (R11 sprint 3–4):** R11 compose-and-run UI (Register E) — a stranger composes any 6-axis coordinate, hits run, watches curves in real time. Ships only when the API is stable.
5. **First research-shaped result (RESEARCH3 CP-A):** Pareto frontier over the resource vector 𝒞, one M-axis primitive swept at a time, annotated with which primitive owns each knee — the first figure that is a *finding*, not a demonstration. Depends on RESEARCH3 PR-9 campaign commissioning passing.
6. **Z3 flagship registered commission (RESEARCH3 CP-A):** ≥95% on all three tasks at exact Δθ=0 within ≤20% of fine-tuning steps — accuracy-vs-steps curves, forgetting matrix, per-seed Δθ audit. If it falsifies: the L1 adaptation figure substitutes as the campaign seed and the boundary condition becomes the publication.

Sequencing rule: 1–2 are R11 core; 3 pulls immediately after (wrapper depends on API stability from 1); 4 ships when 1 is stable; 5–6 are RESEARCH3 CP-A's spine. No checkpoint blocks on a later one.

---

## 📐 R11 Sprint Plan (proposed sequencing)

| Sprint | Focus | Deliverables |
|--------|-------|--------------|
| **1** | Geometry axis completion | `ConvGeometry` + demo test (`test_demo_conv.py`); `GraphGeometry` + demo test if graph domain task needs it; README factory rows updated; gallery re-pin. |
| **2** | Substrate fidelity + tile matrix | Neuromorphic spike dropout (or sparsity field removal) + demo test; tile_ep/tile_pc device-dynamics fixes or documented xfails; gallery re-pin. |
| **3** | Hygiene pass + kernels | Register C dedicated pass (ruff/pyright clean or CI scoped); FA feedback projection through Substrate operator; `SubstrateSettleKernel` registered; MEP Triton kernels for Muon/Fisher. |
| **4** | Adapter + _TaskTrainer + wrapper | Strangler-fig heuristics recovered; scheduler/energy/tracker wired; Drop-in PyTorch wrapper v1 + smoke test; README factory row for wrapper. |
| **5** | R11 Live demo UI | Compose tab (6-axis picker), preset gallery, ψ visualizer, θ-hash badge, one-click export. Built on stable API from sprints 1–3. |
| **6** | RESEARCH3 CP-A integration | Campaign stack commissioned (PR-9), AutoScientist M-axis frontier campaign run, Z3 flagship commissioned. |

---

## 🔍 Pre-flight Checks (before sprint 1)

- [ ] Property locks green: `uv run pytest tests/property/ -q`
- [ ] Demo gate green: `uv run pytest tests/integration/ -k "demo or gallery_lock" -q` (8/8, ≤90 s)
- [ ] Drift locks green: `uv run pytest tests/unit/core/test_readme_snippet_lock.py tests/unit/core/test_root_exports.py -q`
- [ ] Manifest + run records committed (gallery lock data layer tracked)
- [ ] RESEARCH3 PR-0 verification gate green (TIER-0/digits campaign)

---

## 📝 Notes for the Next Editor

- R10 closed cleanly: the demo suite (D1–D7) covers all six axes at 69 s walltime; the gallery manifest + run records are committed; the figure lock and snippet lock are green.
- Register B items are the *only* library-completeness work — they don't ship without a demo test. Don't pull infrastructure "just in case."
- Register C is a separate hygiene pass; don't mix it with capability pulls. Either fix the ~4.8k findings or explicitly scope CI to the gates that hold.
- RESEARCH3 items (Register A) live on the research track; they feed the corroboration appendix and papers, never the front page.
- The Z3 optimizer-hygiene defect (RESEARCH3 PR-1) is fixed in current code (fresh Adam at every boundary, documented in-module) — the commission runs on the fixed instrument.