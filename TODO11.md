# TODO11.md — Active Plan: The Library, Completed and Connected

> **Opened 2026-09-02 (draft).** Successor to [TODO10.md](TODO10.md)
> (R10 closed: D1–D7 demonstrate all six ontology axes; three consecutive
> green gate runs banked; the acceptance session — *read, run, change one
> thing, see it matter* — is available to any stranger). This plan contains
> **all remaining TODO10 work**: Register B capability pulls, the Register C
> hygiene pass, the carried registers, and the research-track spine's
> open prerequisites. Research catalog: [RESEARCH3.md](RESEARCH3.md).
> Landing-cost / wiring-hygiene work moved to [TODO.md](TODO.md) (2026-09-04).
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
> **State: CORE COMPLETE (2026-09-04).** All planned R11 capability pulls and
> hygiene items landed at HEAD; R11.3.12 (ePC, D12) pulled 2026-09-04. R11.2.24
> (resumable trainer, fold_in RNG) pulled 2026-09-04. User-directed general-
> improvements session (2026-09-04): R11.2.23 (sample-weighted metrics +
> `val_ppl`) and R11.4.1 v1 (`SystemModule` facade) landed; `GradientCredit`
> fail-loud resolved (Watch). CP-6 opened 2026-09-04: R11.3.13 (depth
> metrics) + R11.3.11 μPC init landed; depth-frontier E-1 pilot run —
> boundary ≈ depth 8 confirmed, μPC lift unconfirmed pending multi-seed
> pilot (see Remaining Items). The
> library demonstrates all 12 capabilities (D1–D12) at demo scale;
> `comp repro` 8/8; property suite 670 passed; demo gate 13/13; gallery lock
> green; `comp gallery` renders all figures. Session 2026-09-04 (continued):
> R11.3.14 (deep Hebbian chain) pulled with a live subspace-collapse
> boundary; R11.1.10 (LazyStateDynamics) pulled with a measured
> settle-count refutation. R11.3.11 audit-driven revision: μPC verdict
> downgraded from "refuted" to OPEN after an instrument audit found two
> regime mismatches (plain-MLP vs paper's residual architecture; Euclidean
> SGD vs paper's Adam/β-grid); residual geometry landed as a capability
> and the in-regime re-test still shows no lift under our trainer —
> the jpc-faithful port is the remaining gap. Remaining items are
> explicitly **pull-based** or **deprioritized** — they land only when a
> demo, campaign, or research paragraph needs them.

---

## 📜 Standing Directives (carried, binding)

These are session-established user directives and measured facts. They bind
every workstream below.

- **`benchmark_results/` stays untracked and gitignored — never re-add it**
  (user directive 2026-09-02, superseding earlier TODO10 language).
- **README: never edit it** (user directive 2026-09-03). The README/snippet
  drift-lock machinery is retired: `test_readme_snippet_lock` stays red at
  HEAD by directive and is not a gate. Evidence lives in `docs/RESULTS.md`
  and the gallery.
- **Test-execution discipline (2026-09-02):** never run tests without showing
  output and walltime (`--durations` in addopts; pipe through `tail`/`grep`,
  never silent `head`-truncation). Minimize redundant test executions:
  measure levers in throwaway scripts before touching tests.
- **Lint/type debt is deprioritized (2026-09-03):** ruff sits clean and stays
  clean passively (per-line markers self-flag on touch); pyright runs only
  on genuinely new modules when it adds signal. R11.2.2 and remaining
  lint-adjacent items are as-touch work, never a workstream. Real
  development progress is the priority.
- **Device policy (measured 2026-09-02, RTX 3080):** the demo suite stays on
  **CPU** — tiny Digital builds (784→32→10, batch 64, Python settle loop)
  are kernel-launch-bound, and CUDA ran *slower* (D2 hit 60 s timeout).
  GPU-first applies where work is FLOP-bound: registered-scale studies,
  campaign fleets, large hidden dims, long horizons. Rule: *prefer GPU where
  appropriate — measured, not assumed* (AGENTS.md), with the demo-suite CPU
  verdict as the standing counter-example.
- **DataLoader workers:** `num_workers=2` measured faster at demo scale
  (13.2 s vs 20.7 s per epoch). `num_workers=0` is the *flake* mitigation
  (D7 precedent), not a speed rule.
- **GitHub CI is not yet in use** (2026-09-02): the gates that matter are the
  locally runnable invocations recorded in this plan; workflow edits are
  bookkeeping, not acceptance criteria.

---

## 🎯 The Demonstration Table (D1–D12)

| #  | Capability                                                          | Demo test                                      |
|----|---------------------------------------------------------------------|------------------------------------------------|
| D1 | Six-axis composition is real                                        | `test_demo_compose_6axis.py`                   |
| D2 | One trainer, every credit rule                                      | `test_demo_swap_credit.py`                     |
| D3 | The P-axis swap matters                                             | `test_demo_swap_plasticity.py`                 |
| D4 | The memory profiler is honest                                       | `test_demo_memory_budget.py`                   |
| D5 | Frozen θ is a guarantee, bitwise                                    | `test_demo_z3_frozen_theta.py`                 |
| D6 | The substrate axis is physical (memristive IR-drop + neuromorphic spike dropout, five arms) | `test_demo_substrate_swap.py` |
| D7 | The D-axis settles in time                                          | `test_demo_spike_settle.py`                    |
| D8 | The G-axis is a swap (capacity-matched conv vs flat)                | `test_demo_geometry_swap.py`                   |
| D9 | The G-axis is a swap (capacity-matched graph vs flat, structural generalization) | `test_demo_graph_geometry_swap.py` |
| D10| The G-axis is a swap (capacity-matched attention vs flat, permutation sensitivity) | `test_demo_attention_geometry_swap.py` |
| D11| The G-axis is a swap (capacity-matched 3D lattice vs flat, spatial noise robustness) | `test_demo_spatial_lattice_geometry_swap.py` |
| D12| The D-axis settles without signal decay (ePC: free equilibrium = feedforward bitwise, nudged signal reaches every layer, 1/3 settle budget) | `test_demo_epc_fast_settle.py` |

---

## ✅ Completed This Session (2026-09-04)

### R11.1 — Capability Pulls (Register B)

| Item | Description | Key Evidence |
|------|-------------|--------------|
| **R11.1.1** | Neuromorphic spike dropout (D6 five-arm) | `SubstrateConfig.neuromorphic(sparsity=...)`, `create_neuromorphic_mlp` |
| **R11.1.2a** | ConvGeometry (D8) | im2col via substrate forward; capacity-matched (3,940 vs 3,818 params) |
| **R11.1.2b** | GraphGeometry (D9) | Adjacency message passing; capacity-matched (1.3k vs 1.7k params) |
| **R11.1.2c** | AttentionGeometry (D10) | Multi-head self-attention + FFN; capacity-matched (~100k params) |
| **R11.1.2d** | SpatialLattice3DGeometry (D11) | 3D neural cube; capacity-matched (~200k params) |
| **R11.1.3** | Tile × dynamics matrix documented as strict xfails | 7 tile pairings with mechanism-level reasons; `comp repro` 8/8 |
| **R11.1.4** | Tile-mesh settle kernel | Flips 7 tile xfails → xpass; `test_tile_settle_kernel.py` lock |
| **R11.1.5** | Adapter shape-probing, fail-loud | `_probe_linear_dims` walks `nn.Linear` chain; raises `TypeError` on failure |
| **R11.1.6** | _TaskTrainer scheduler/tracker/safety | Cosine/step/linear/cosine_warmup; `SafetyConfig`/`SafetyWrapper`; GPU-verified |
| **R11.1.7** | Diffusion target term (nudged-Langevin) | `compute_energy_from_state(target, beta)`; fidelity probe passes |
| **R11.1.8** | Ontology facade merge | `_dynamics.py`→`dynamics/_dynamics.py`, `_substrate.py`→`substrate/_substrate.py` |
| **R11.1.9** | Timing-asymmetric STDP wired to 5-D pipeline | Spike rasters, eligibility traces, configurable threshold |
| **R11.3.11b** | Residual feedforward geometry + in-regime μPC re-test (pulled 2026-09-04, audit-driven) | **Capability landed:** `GeometryConfig.residual` (skip between equal-width hidden layers, `a_ℓ = a_{ℓ−1} + φ(W_ℓ a_{ℓ−1} + b_ℓ)`; input/output projections unscaled) — forward, `route`, settle kernel (`SubstrateSettleKernel` + compiled `_eqprop_settle_loop`), and spec round-trip all carry it. Lock: `tests/integration/test_residual_geometry.py` (5 tests): manual-trace bitwise match, eager≡compiled parity (bitwise; the initial parity "failure" was a test-side RNG-order bug — built both systems from one seed), spec round-trip, fail-loud on non-feedforward. **In-regime re-test** (`scripts/probes/mupc_residual_regime.py`): residual depth-8/width-128 MNIST, seeds 0–2, μPC 0.137 vs default 0.139 — **still no lift under our trainer**. Verdict downgraded from "refuted" to OPEN: architecture family now matches the paper, but the trainer regime still does not (paper: Adam weights, activity step β ∈ {1e3..1e-2} tuned per run, inference steps = H, width 512; ours: Euclidean SGD, β=0.5, 60 fixed settle steps). Next pull for a clean answer: jpc-faithful port (Adam on weights, large-β activity GD, steps=H) |
| **R11.1.10** | LazyStateDynamics (pulled 2026-09-04) | Rewritten as a real sequential (Gauss–Seidel) EqProp settle: per-layer in-place updates reading freshest neighbors, substrate forward-operator bottom-up, per-sweep activation cache, fail-loud on non-layered/recurrent. Wired per the primitive checklist: registry `"lazy"`, `StateDynamicsConfig.lazy()`, root `__all__`+`_LAZY`+TYPE_CHECKING, thermo-contrast validate whitelist. Lock: `tests/integration/test_lazy_dynamics.py` (5 tests, ~5 s): monotone per-sweep Hopfield energy, nudge pulls output toward target, MNIST quick-mode 150-batch training > 2.5× chance, fail-loud non-layered/recurrent, sweep-count observable. **Measured refutation:** the plan's "settle-count contrast" expected Gauss–Seidel to win in sweeps — it does NOT at demo scale (34 sweeps vs Jacobi 21 at 256→64×6→10, τ=1e-2, step 0.05); no dominance claimed |

### R11.2 — Hygiene Pass (Register C)

| Item | Description | Key Evidence |
|------|-------------|--------------|
| **R11.2.1** | Ruff baseline clean | `ruff check .` clean at HEAD; E501 disabled forever |
| **R11.2.3** | Root `PlasticityConfig` twin-class resolution | Single source in `computronium.state.transitions`; ~140 lines deleted |
| **R11.2.4** | Joint `to_spec`→`from_spec` round-trip | `TestJointSystemSpecRoundTrip` locked (recurrent + FF bitwise) |
| **R11.2.5** | `init_scale` functional | Shared `_linear_stack` helper; default≡0.1 bitwise; 0.2≡0.1×2 |
| **R11.2.7** | Energy dedup: `_state_energy_vector` | PredictiveSettling / SpikeIntegration / Diffusion share it |
| **R11.2.8** | `FrontierRecord.seed` required | Campaign record `seed: int` no default; clean break (no compat) |
| **R11.2.10** | Params-moved locks (10/10 factories) | `test_params_moved.py`; fidelity expanded 48→60 coords |
| **R11.2.11** | imp-27 resolved-by-contract | `settle` protocol docstring + AST census lock (`test_settle_caller_census.py`) |
| **R11.2.12** | Tile family fold: `equitile`→`tile` | 7 deployments, CLI `FAMILY_MAP`, metamodel, tolerances all canonical `tile` |
| **R11.2.18** | `test_scaling_invariants` xpass removed | Marker removed; now asserts `acc > 0.3` live |
| **R11.2.20** | Timebox closed | All scoped items landed; no finding class stretched past its box |
| **R11.2.21** | Zoo Registry deleted | 6 files + ~30 consumers stripped; all surfaces resolve native 5-D factories |
| **R11.2.22** | Fidelity-gate determinism | `check_coordinate_fidelity(seed, fork_rng)`; verdicts deterministic |
| **R11.2.24** | Resumable trainer (`fold_in` RNG) | `TrainerSnapshot` + `from_snapshot`; interrupted == uninterrupted **bitwise** (`tests/integration/test_trainer_resume.py`); pure `fold_in` locked by hypothesis (`tests/property/test_fold_in_rng.py`) |
| **R11.2.25** | `torch.compile` settle fast paths | `compiled=True` now covers **both** energy families: sPC layered settle (2.0× train_step, bitwise parity) and `EnergyMinimizationDynamics`/`SubstrateSettleKernel` loop (1.75× settle, parity 9.5e-7, autograd-graph parity for thermo credit locked). Eager path byte-identical when off. Locks in `test_compiled_settle.py`; probes `torch_compile_settle.py` / `torch_compile_eqprop_kernel.py` |
| **R11.2.23** | Metric aggregation contract (pulled 2026-09-04, user-directed "general improvements" session) | Trainer epoch metrics are **sample-weighted** sums (ragged final batch no longer over-weights); `validate()` reports `val_ppl = exp(mean CE)` from the same per-sample normalization. Lock: `tests/unit/core/test_trainer_metric_aggregation.py` (weighted-mean identity via delegating spy, ragged batches, ppl identity) |

### R11.3 — Research Track (RESEARCH3 Spines)

| Item | Description | Key Evidence |
|------|-------------|--------------|
| **R11.3.1** | PR-9 Campaign commissioning | Smoke kill→resume cycle at HEAD; unbuffered pre-kill trail; `records/episodes.json` |
| **R11.3.2** | PR-2 θ-audit harness | `theta_audit()` context manager; SHA-256 over name+device+dtype+bytes |
| **R11.3.3** | PR-5 Calibrated stability guard | Demo-harvest ROC within 0.005% of deployed τ; `fast_proxy` calibration-only; artifact + live lock |
| **R11.3.12** | ePC fast-settling solver (D12) | `ErrorPredictiveCodingDynamics` — error reparameterization per Goemaere et al. (arXiv:2505.20137, ICML 2026); free equilibrium = feedforward bitwise, nudged signal reaches every hidden layer (sPC's reaches none), trains at 1/3 budget; demo + gallery figure + round-trip |
| **R11.3.11** | Multi-seed depth-frontier pilot (E-1, pulled 2026-09-04; **superseded by the in-regime re-test, see below**) | `scripts/probes/mupc_multiseed_frontier.py`: depths 4/8 × seeds 0–3 × spc/default vs spc/mupc, compiled settle, 477 s. μPC lift at depth 8 absent (0.135 vs 0.133). **Instrument audit finding:** the pilot applied μPC init to a plain MLP — outside the paper's tested domain (arXiv:2505.13124 Table 1 is specified and tested on residual networks; skip connections are load-bearing for the (N·L)^{-1/2} hidden scale) |
| **R11.3.14** | Deep Hebbian chain with per-layer activity normalization (pulled 2026-09-04, user-directed plan-and-fix) | `computronium/models/native/deep_hebbian_native.py`: `DeepHebbianChain` — spectral renorm (unit gain at init) + tanh + batch Oja decay + unit-RMS activity renorm per layer; plain-torch local learning (no backprop, no nudging). Per-layer pre-renorm signal norms O(1) at depth 10/50/100 (the tile-chain runaway-gain/NaN pathology is structurally fixed); unnormalized control decays to ~1e-14. Dominant-direction 2-class readout 1.000 at every depth. **Honest boundary (R11.5.5):** 10 direction-coded classes → L1 1.00 / L10 0.52 / L100 0.20 (> 0.1 chance): activity covariance effective rank collapses 5.1→1.5 through the chain under compounding tanh distortion + renorm + Oja spectral sharpening — Sanger, gain scaling, and per-step spectral renorm do NOT rescue it. Lock: `tests/integration/test_deep_hebbian_chain.py` (8 tests, ~7 s) |

### R11.4 — Adoption Surface

| Item | Description | Key Evidence |
|------|-------------|--------------|
| **R11.4.2** | PR-6 Fairness contract draft | `docs/FAIRNESS_CONTRACT.md` v0.1 (F-1..F-6, consumers table) |
| **R11.4.1 (v1)** | `SystemModule` drop-in nn.Module facade (pulled 2026-09-04, user-directed "general improvements" session) | `computronium/nn/system_module.py`, root export. Plain-PyTorch inference (`forward` under `no_grad`/`eval`), `fit_step` for internal credit assignment (no optimizer), `parameters()`/`train()` delegate to geometry, `to(device/dtype)` moves geometry (mirrors SystemTrainer's `geometry.to(device)` convention; pyright-strict clean). Lock: `tests/unit/nn/test_system_module.py`. Scope-honest: this is the wrapper *surface*, not pip packaging |

---

## 📋 Remaining Items (Pull-Based — No Schedule)

These land **only when a demo, campaign, or research paragraph needs them**.

| Item | Trigger | Category |
|------|---------|----------|
| **R11.3.14** Deep Hebbian fix: per-layer activity normalization | **LANDED 2026-09-04** (see Completed below) | ~~Capability~~ ✅ |
| **R11.3.11** Multi-seed depth-frontier pilot | **Pulled 2026-09-04; verdict downgraded** (see R11.3.11b in Completed). No μPC lift under our trainer even in-regime (residual); the clean answer needs a jpc-faithful trainer port — **pull-based** | Research (architecture ✅, trainer regime OPEN) |
| **R11.1.10** LazyStateDynamics | **Landed 2026-09-04** (see Completed) | ~~Capability~~ ✅ |
| **R11.1.11** Domain extensions | Benchmark/demo/research needs: `wikitext2`/`penn_treebank` (LM), `mountain_car`/`lunar_lander` (RL), `diabetes`/`california_housing` (tabular), `ett_h1` (time series), PDE suite (Heat/Wave/Burgers/Navier-Stokes) | Capability |
| **R11.2.14** Latency proxy | **Landed 2026-09-04** — `estimate_train_step_flops` (`core/profiling.py`): deterministic structure-derived FLOPs per train_step (matmul rounds per weight matrix × settle structure from `dynamics_type`, incl. the spike-substrate one-matmul-per-layer subtlety); intended as a *relative* comparator — absolute latency stays with the repeated-timing path in `analyze_joint_system`. Lock: `tests/unit/core/test_latency_proxy.py` (determinism, depth/settle-step scaling, **proxy ordering matches measured walltime**, non-layered rejection) | ~~Hygiene~~ ✅ |
| **R11.2.9** `substrate_coupled` plasticity engagement | Campaign manifest needs it; probe fixed-dim `step` assumptions; now also the home of any future latent-graph ternary learning path (see Notes) | Hygiene |
| **R11.2.13** Campaign stability proxy | Cheap per-episode proxy for stability axis | Hygiene |
| **R11.2.15** `demo/tests/` 28 stale failures | Rebuild with R11.4 UI, or before if path touched | Hygiene |
| **R11.2.16** TF-IDF weighting / `V_nudged` | Research track wants strengthened PC Lyapunov xfail | Hygiene |
| **R11.3.4** AutoScientist P-axis frontier | Tangible Checkpoint 5 — first *finding* figure (Pareto over 𝒞) | Research |
| **R11.3.11** μPC depth scaling | **Init landed 2026-09-04** — `GeometryConfig.init_scheme="mupc"` (N(0,1), hidden 1/√(N·L), output 1/N; arXiv:2505.13124) native on feedforward/recurrent/graph geometry. E-1 probe `scripts/probes/mupc_depth_init.py`; E-1 pilot `scripts/probes/mupc_depth_frontier.py` (read both docstrings): boundary at depth ≈ 8 confirmed, BP decays through it too (0.808→0.345→0.110); **μPC lift at depth 8 UNCONFIRMED** (0.131 vs 0.127 — contradicts the earlier 2× claim; confounds: compiled fixed-budget settle, seeds). Multi-seed pilot is the next pull (~2 h CPU, 55 ms/step compiled). | Research (init ✅, frontier pilot-only) |
| **R11.3.13** Depth-metric classes | **Landed 2026-09-04** | `computronium/ontology/depth.py`: `DepthMetric` Protocol, `FixedDepth`, `ShortestPathDepth` (BFS from sources, edge direction row←col matching `GraphGeometry._aggregate`), `LongestPathDepth` (DAG Kahn; fail-loud on cycles), `max_depth`. `GraphGeometry.num_nodes` + `node_depths(metric)`. Root + ontology exports. Lock: `tests/unit/core/test_depth_mupc.py` (12 tests, incl. default-init bitwise lock) |
| **R11.2.23** Energy-framed metric contract | **Pulled 2026-09-04** (see R11.2.23 in Completed) — live trainer sample-weighted metrics + `val_ppl`. FabricPC's legacy `EvalMetric` design informed the contract; FabricPC itself is archived | ~~Hygiene~~ ✅ |
| **R11.3.5** Z3 flagship registered commission | Tangible Checkpoint 6 — ≥95% on 3 tasks, exact Δθ=0, ≤20% fine-tuning steps, ≥5 seeds | Research |
| **R11.3.6–3.10** Boundary mapping, CL, task-family, provenance, companions | Pull when research paragraph needs them | Research |
| **R11.4.1** Drop-in PyTorch wrapper | **v1 pulled 2026-09-04** (see R11.4.1 in Completed) — remaining: pip packaging + acceptance test per RESEARCH3 PR sequence | Adoption |
| **R11.4.3** Live demo UI | API stable — ships only when library is stable; rebuilds `demo/tests/` | Adoption |
| **R11.4.4** Hygiene sweep | Only when blocks a figure, test, or fresh checkout | Adoption |

### Deprioritized (As-Touch Only)

| Item | Reason |
|------|--------|
| **R11.2.2** Pyright baseline | User directive: as-touch on legacy modules; new modules stay strict. Not a workstream. |

---

## 📋 Register D — Carried Deferred (unchanged from TODO10)

| Item | Reason |
|------|--------|
| Coverage floor (~16.8%) | opt-in `--cov`; raise after API stabilizes |
| Rocq general-case formalization | CP-B pull-based; diagonal case done with paper proof; ψ-coverage proposition is the next statement |
| `test_ontology_parity.py` decomposition | Slow-marked; split fast/slow only if gate iteration speed demands |
| Physical hardware (PR-3b / CP-D) | Latency-gated procurement per RESEARCH3; proxy tier (R11.3.2b) decouples all software-side claims |

---

## 🔒 R11.5 — The Standing Rules (R10.3 verbatim, renumbered)

- **R11.5.1 No test, no feature.** Every feature ships with an integration
  test that demonstrates it working end-to-end.
- **R11.5.2 No claim without a live demonstration.** When a test is removed,
  flaky, or failing, its claim disappears from the front page automatically —
  the system degrades to silence, never to stale assertions.
- **R11.5.3 Corroboration never carries.** Registered numbers are history:
  labeled, scoped, provenance-annotated, confined to RESULTS.md's back
  section and the research track.
- **R11.5.4 Scope honesty.** Demo-scale demonstrations speak for demo scale;
  registered claims live in the research track. Neither borrows the other's
  clothes.
- **R11.5.5 Refutations ship with the same pipeline** — same figure factory,
  same docs, same terms. Standing candidate: the **spiking family's learning
  claim**. Status at HEAD (2026-09-04): R11.1.9 wired timing-asymmetric STDP
  (rasters, eligibility traces, threshold) into the 5-D pipeline, but no demo
  measures whether it *learns* — the pre-wiring Hebbian-plateau result
  (TODO10: spiking at chance on MNIST) is history, not a live refutation.
  First spiking pull must show one or the other: plateau (refutation figure,
  same pipeline) or learning (capability claim). Until then the library's
  "honest failure" slot is vacant.
- **R11.5.6 Pull rule.** A backlog item is pulled only if it ends in a live
  demonstration, a gallery figure, or a RESULTS.md capability paragraph.
  Infrastructure is justified by the capability it lets the suite show,
  never by itself.
- **R11.5.7 Gates (tiered, per AGENTS.md test-execution tiers).** Per-commit
  duties are **scoped to changed files** (format + lint + pyright + targeted
  tests). The standing fast gates — property suite, demo gate
  (`pytest tests/integration/ -k "demo or gallery_lock"`, ≤90 s), drift locks,
  positive control — run on their triggers (demo/gallery/lock-adjacent
  changes), never per-edit. The full CI order and repo-wide ruff/pyright are
  R11.2's deliverable and a round-close event, not a habit. No new
  verification rounds are commissioned in R11; R11 spends R6–R10's trust.

---

## 🔒 Gate Commands (Quick Reference)

```bash
# Property locks (fast CI gate) — 670 passed
uv run pytest tests/property/ -q

# Demo gate (D1–D12) — 13/13 passed (12 demos + gallery lock), ~145s
# NOTE: invoke as `python -m pytest` — see Watch (user-site pytest drift)
uv run python -m pytest tests/integration/ -k "demo or gallery_lock" -q

# Gallery lock — figure data checksums match manifest
uv run python -m pytest tests/integration/test_gallery_lock.py -q

# Reproducibility — 8/8 native families bitwise identical
uv run comp repro --seed 42 --device cpu

# Gallery re-render from on-disk records (deliberate re-pin; `--run` re-runs
# the demo suite first and needs >2min)
uv run comp gallery

# Root exports
uv run python -m pytest tests/unit/core/test_root_exports.py -q
```

---

## 👁️ Watch (Live Items Only)

- **Settle-loop cost (measured 2026-09-04):** both energy-family settle loops
  now have `torch.compile` fast paths behind `StateDynamicsConfig.compiled=True`:
  sPC layered settle (2.0× train_step, bitwise) and EqProp kernel loop
  (1.75× settle; autograd-through-compiled verified for thermo credit).
  Guards keep the compiled path on the common case (digital, no recurrent,
  momentum=0, no tracking/checkpointing); the compiled EqProp path runs a
  fixed step budget (skips the eager convergence early-exit). Remaining
  headroom: extend to SpikeIntegration (D7) with the same recipe when the
  spiking demo is pulled; batch-per-step 4–8× stacks for free. The R11.3.11
  depth frontier is affordable — run it with `compiled=True`.
- **`GradientCredit` fail-loud — RESOLVED (2026-09-04):** `allow_unused=True`
  zero-fill replaced with a `RuntimeError` naming the detached weights
  (`credit.py`, `GradientCredit.compute_pseudo_gradient`; `BackpropCredit`
  is the same class). A future dynamics that detaches activations now fails
  loudly instead of silently degrading to last-layer-only learning.
  `LocalGoodnessCredit` keeps its zero-fill (surplus recurrent self-connection
  weights legitimately receive `None`).
- **`uv run pytest` resolves the USER-site pytest (2026-09-04):** the launcher
  at `~/.local/bin/pytest` (shebang `/usr/bin/python`) shadows the venv's,
  importing protobuf 6.33.6 from user site against gencode 7.35.1 stubs →
  `VersionError` at gRPC test collection. Invoke tests as
  **`uv run python -m pytest`** (guaranteed venv python, protobuf 7.36.1).
  Gate commands above updated accordingly.
- **Plain `uv sync` strips dev extras (2026-09-04):** a bare `uv sync
  --upgrade` re-syncs the venv to main-deps-only, silently removing
  optuna/scipy/torchvision (they live in the `dev` extra group, not main
  deps). Symptom: sudden ModuleNotFoundError at import. Durable fix:
  `uv sync --dev --all-extras`. pyproject was briefly double-listing
  optuna/scipy in main deps during triage — reverted; groups are canonical.
- **D1/D8 record drift absorbed 2026-09-04:** `comp gallery` (render-only
  re-pin) flagged compose_6axis/geometry_swap data changed vs the old
  manifest — same class as the 2026-09-03 sweep-regime note. Manifest
  re-pinned from current on-disk records; demo gate + gallery lock green
  after. If the lock fires again: check test asserts first, then re-render.
- **Demo-gate budget drifted past R11.5.7's ≤90 s (now ~145 s at 12 demos +
  lock):** D8–D12 additions grew the gate before this note; not a failure —
  re-baseline the standing rule when the suite next gets a dedicated
  fast/slow split.

- **axis_probe `[2-0]` flake** — no recurrence since 2026-08-31.
- **CUDA tolerance boundaries** shift xfail edges — CPU/GPU tests kept separate.
- **R11 sweep regime note (2026-09-03):** repo-wide ruff autofix shifted import/init order, moving D2/D7 record data. Tests pass asserts; manifest re-rendered. If figure lock fires again, check test asserts first, then re-render.
- **D8 record determinism:** seed must precede loader draw (`torch.manual_seed(42)` before `_materialize`); DEVICE must be `"cpu"`.
- **Conv = GPU pointer (measured 2026-09-03):** conv-family is first FLOP-bound path (15× CUDA speedup); CUDA nondeterministic run-to-run. Registered-scale conv studies go GPU-first; committed demo records stay CPU.
- **`benchmark_results/` stays untracked** (standing directive).
- **Fidelity probes RNG-order sensitivity — RESOLVED (R11.2.22):** `check_coordinate_fidelity` is seeded + fork-rng'd.
- **Stale eager-default metric lookups:** `d.get("free_accuracy", d["accuracy"])` evaluates default eagerly — safe idiom is nested `get`.
- **Registry-era removals:** transfer-weight loading, proposer objective ranking deleted with zoo — re-home onto native factories.
- **`equitile` deprecated:** family registrations, CLI maps, tolerances, metamodel branches key on `"tile"`. Residual mentions cosmetic — rename on next touch.

---

## 🎯 Tangible-Result Checkpoints (R11 Edition)

| # | Checkpoint | Condition |
|---|------------|-----------|
| 1 | Capability pulls demonstrated (R11.1) | Demo suite green, walltime ≤ 90s ✅ |
| 2 | Truthful gates (R11.2) | ruff/pyright green at HEAD or explicitly scoped; CI order enforceable ✅ |
| 3 | Commissioned campaign stack (R11.3.1) | Iterate → interrupt → checkpoint → resume cycle recorded ✅ |
| 4 | Calibrated stability guard (R11.3.3) | ROC-calibrated kill thresholds (<5% false-kill, >95% kill, <10% overhead) ✅ |
| 5 | Adoption surface (R11.4) | Wrapper v1 (pip-installable, smoke suite) and/or live demo UI — **pull-based** |
| 6 | First research-shaped result (R11.3.4) | P-axis Pareto frontier over 𝒞, annotated per knee — **pull-based** |
| 7 | Discovery bet (R11.3.5) | Z3 flagship at registered scale; either outcome tangible per pre-registered fallback — **pull-based** |

Sequencing: 1–4 complete; 5 after API stabilizes (done); 6–7 are RESEARCH3 CP-A's tail. No checkpoint blocks on a later one.

---

## 📝 Notes for the Next Editor (2026-09-04)

- **All core R11 items complete; R11.3.12 (ePC/D12) pulled 2026-09-04** —
  D1–D12 at demo scale; `comp repro` 8/8; property 670 passed; demo gate
  13/13; gallery lock green.
- **Registry is gone — never re-add it** (2026-09-03). Ontology is the composition surface; models resolve through native factories and `compose_*`. `KernelRegistry` (acceleration/) is unrelated and stays.
- **README is never edited** (2026-09-03). No sunset condition.
- **Ontology package layout (R11.1.8):** implementations in `_`-prefixed modules (`dynamics/_dynamics.py`, `substrate/_substrate.py`); `__init__.py` = docstrings + re-exports only.
- **PlasticityConfig single source:** `computronium.state.transitions` owns it; `core/joint/transition.py` re-exports. Never redefine — import.
- **Geometry dispatch single source (R11.1.2a):** `computronium.ontology.geometry_from_config` is the one topology_type→implementation dispatcher. Never re-inline — add a branch. New `GeometryConfig` tuple fields must be added to `_geometry_spec_parts`'s JSON tuple-restore list.
- **Tile × dynamics matrix (R11.1.3 + R11.1.4):** 7 tile strict xfails flipped xpass, promoted to live locks in `test_native_smoke.py` and `test_validation_all.py`. Single unlock: target-responsive TileMesh settle kernel. `native_tile_ep` re-added to REPRO_MODELS. `comp repro` 8/8. New lock `test_tile_settle_kernel.py`. **User directive: Tile geometry potential realized later.**
- **Diffusion target term (R11.1.7):** `DiffusionDynamics.compute_energy_from_state` accepts optional `target`, `beta` (nudged-Langevin). Fidelity probe passes. PredictiveSettlingDynamics fallback remains target-unwired (no geometry uses it).
- **PR-5 instrument (R11.3.3):** `calibrate_demo_harvest` (stability/calibration.py) single calibration surface; artifact `docs/figures/registered/stability_guard_pr5.json`. Known-bad = manufactured explosive family — re-calibrate against real diverged runs when failure manifesto accumulates. Deploying kill switch = wiring `probe_interval_for_overhead` (102 episodes) into AutoScientist loop.
- **Demo-test record determinism (D8):** seed *before* materializing loader batches; workers spawn per loader *iteration*, so materialize once and share. Match parameter counts and assert parity for fairness.
- **`equitile` deprecated (2026-09-03):** canonical key is `"tile"`. Residual mentions cosmetic (test names, model names, benchmark variables, docstrings) — rename on next touch, never as sweep.
- **ePC single source (R11.3.12 / D12, 2026-09-04):** `ErrorPredictiveCodingDynamics`
  lives in `dynamics/_dynamics.py` next to `PredictiveSettlingDynamics`.
  Attribution: Goemaere et al., "ePC: Fast and Deep Predictive Coding in
  Digital Simulation", arXiv:2505.20137 (ICML 2026) — class + config
  docstrings carry it. Surfaces wired: `dynamics/__init__`, ontology
  `__init__`, root `_LAZY` + `__all__`, `StateDynamicsConfig.error_predictive_coding()`,
  `from_spec` branch (factory.py), `SystemConfig.validate()` predictive-settling
  credit branch (accepts both PC dynamics). Free-phase equilibrium =
  feedforward pass bitwise (zero-init errors are the fixed point); nudged
  phase = β·CE driven through full-graph reverse-mode AD (requires
  `torch.enable_grad()` inside settle — pipeline runs no_grad for
  ThermodynamicContrast). Out-of-place adds only in the error forward —
  in-place adds pin the autograd graph (same CUDA-leak rule as geometry.py).
  Demo claims are structural (equilibrium, propagation, budget) — accuracy
  parity is NOT claimed: ePC ≈ 0.44 vs sPC ≈ 0.55 at (32,32)/150 batches;
  the ÷β contrastive credit caps ePC's learning signal on deeper stacks
  (candidates if revisited: PC-native weight gradient (∂ŝ/∂θ)ᵀε or a
  contrast-β decoupled from the loss weight).
- **μPC init + depth metrics (R11.3.11 init + R11.3.13, landed 2026-09-04):**
  `GeometryConfig.init_scheme` (`Literal["default","mupc"]`) is the single
  init lever — "default" is byte-identical to legacy (locked bitwise in
  `test_depth_mupc.py`); "mupc" replaces fan-in init with N(0,1) × depth
  scaling and supersedes `init_scale`. Wired in `_linear_stack` (feedforward
  + recurrent) and GraphGeometry (layers + head). `GraphGeometry.node_depths(metric)`
  is the R11.3.13 seam: per-node effective depth replaces layer-counting on
  graph topologies. `asdict` round-trip carries the new field (no
  `_geometry_spec_parts` change needed — str, not tuple).
  **Trap learned the hard way:** when rescaling weights in place
  (`Parameter.data.mul_`), initialize from `torch.randn`, never `torch.empty`
  — garbage × scale is still garbage, and the D9 graph demo silently learned
  garbage for 5 epochs before recovering (caught by the demo gate, exactly
  its job). Frontier probes: `scripts/probes/mupc_depth_frontier.py`
  (boundary + μPC-unconfirmed) and `scripts/probes/mupc_compiled_device.py`
  (device verdict: compiled CUDA 80 vs CPU 55 ms/step at width 32 — CPU
  still wins; compile 2.6× CPU). Next research step: multi-seed pilot.
- **R11.3.14 deep Hebbian chain (landed 2026-09-04):** implementation home
  is `computronium/models/native/deep_hebbian_native.py` — a plain-torch
  chain, deliberately NOT the tile graph (its per-edge σ caps interact badly
  with full connectivity). Recipe: spectral renorm at init + tanh + batch
  Oja (`w += lr·(yᵀa/n − E[y²]·w)`) + unit-RMS activity renorm per layer.
  Key measured findings (module docstring carries them):
  (1) the primary pathology was runaway per-layer gain (1.2–1.5×/layer
  compounding → inf/NaN); activity renorm fixes it structurally;
  (2) the trained chain transmits its *dominant direction* indefinitely
  (2-class readout 1.000 at depth 100);
  (3) a rank-10 class subspace decays (L1 1.00 → L100 0.20 > 0.1 chance):
  activity covariance effective rank collapses ~0.5/layer. Sanger (GHA),
  gain scaling, and per-step spectral renorm do not rescue it — renorm
  amplifies whatever the spectrum favors each layer. This is the third
  failure mode of the depth-boundary triad (error rules: telescoping
  decay; unnormalized local: runaway gain; normalized Oja: subspace
  collapse) — a candidate CP-6 finding figure.
  **Determinism trap (D8-class):** class means must be drawn ONCE per
  trial (seeded generator passed to both train and eval draws) — the
  first draft regenerated means per call, silently mismatching
  train/eval geometries. Same lesson as seed-before-loader.
  Readout convention: nearest-centroid (linear scores), NOT one-hot ridge
  — ridge without a bias term cannot represent ordered/interval class
  structure along a 1-D code and silently reports chance.
- **ruff 0.16 selector migration (2026-09-04, env drift fixed):** the venv
  ruff upgraded to 0.16.6, which dropped long-form rule names — pyproject's
  `ignore`/`per-file-ignores` no longer parsed and EVERY ruff invocation
  failed (`line-too-long`→E501, `magic-value-comparison`→PLR2004,
  `no-self-use`→PLR6301, `invalid-argument-name`→N803,
  `non-lowercase-variable-in-function`→N806, `raise-vanilla-args`→RSE102,
  `unused-function-argument`→ARG001, `unused-method-argument`→ARG002,
  ambiguous-unicode→RUF001/2/3, `float-equality-comparison`→PLR0133,
  `undefined-export`→F822, `non-empty-init-module`→INP001,
  `non-augmented-assignment`→PLR6104, subprocess/random S-codes, `assert`→
  S101). Selector strings are canonical codes only from here on. Repo-wide
  `ruff check` now reports ~580 findings under the (renamed but wider)
  effective set — Register C scope, not per-commit blockers.
- **`DeepHebbianChain` is local-only by design:** weights are
  `requires_grad=False` nn.Parameters (in-place Oja under `no_grad`);
  do not wire it into SystemTrainer/autograd credit — it is the local
  feature-learning arm, evaluated via readout.
- **LazyStateDynamics landing (2026-09-04):** the pre-existing class was a
  stub (single-tensor routing, `(acts**2).mean()` energy, unregistered);
  rewritten in place as a sequential Gauss–Seidel settle. Key measured
  facts: (1) per-sweep Hopfield energy is monotone non-increasing;
  (2) the nudged phase works like the Jacobi kernel (output nudge each
  sweep); (3) Gauss–Seidel does NOT converge in fewer sweeps than Jacobi
  at demo scale — measured 34 vs 21 at (256→64×6→10, τ=1e-2, step 0.05).
  The sequential sweep's value is the on-demand/memory strategy, not
  speed — scope-honest claim only. ReLU nets have multiple fixed points:
  Jacobi and Gauss–Seidel legitimately land on different ones (both are
  fixed points of the same map) — never assert fixed-point equality
  between the two settles.
- **Pyright ignore-comment convention (2026-09-04):** `# type: ignore`
  comments do NOT suppress pyright errors in this repo's config
  (pyrightconfig.json, basic mode); use `# pyright: ignore[<rule>]` placed
  on the exact line pyright reports (for multiline calls, on the offending
  argument's line, not the call opener). Ruff's PGH003 forbids bare
  `# pyright: ignore` — always name the rule.
- **Repo's `# ruff: ignore[...]` comments are invalid (2026-09-04):** ruff
  only honors `# noqa`; the `# ruff: ignore[x]` idiom is flagged RUF103
  under the current rule set and its suppressions are NOT applied. This
  is why repo-wide ruff reports ~580 findings despite "per-line markers
  self-flag on touch" — the markers never worked. Register C scope: a
  one-time sweep converting `# ruff: ignore[` → `# noqa: ` (codes must be
  translated to ruff names) would restore the intended suppression
  system; do it in the hygiene pass, never per-commit.
- **Multi-seed pilot verdict (R11.3.11, 2026-09-04 — DOWNGRADED after the
  audit):** the initial "μPC lift refuted" was premature on two counts.
  (1) Domain mismatch: the pilot ran plain MLPs, but the paper's Table 1
  parameterization is specified and tested on residual networks — the
  (N·L)^{-1/2} hidden scale assumes a skip path; without one the scaled
  branch has nothing to correct. `GeometryConfig.residual` now makes the
  paper's architecture family expressible (locked in
  `test_residual_geometry.py`). (2) Trainer mismatch: the paper uses Adam
  on weights, activity GD with β up to 100 (grid-searched), and inference
  steps = H (not convergence); ours uses Euclidean SGD, β=0.5, fixed 60
  settle steps. In-regime re-test (residual, width 128, seeds 0–2): μPC
  0.137 vs default 0.139 — no lift under our trainer, but the paper's
  optimizer/β regime is still untested. Status: OPEN, not refuted. Do not
  quote "μPC refuted" anywhere; the honest statement is "no lift under the
  computronium trainer; jpc-faithful port (Adam, β grid, steps=H) is the
  remaining instrument gap."
- **Audit lesson (2026-09-04, user-prompted):** before publishing a
  negative verdict, check the instrument against the source paper's
  stated regime. Two systematic mismatches (architecture family, optimizer
  regime) hid inside a plausible "refuted" conclusion. Refutations ship
  with the same pipeline — and that pipeline must demonstrably implement
  the claim's own terms (R11.5.5 applied to refutations themselves).
- **Ternary × gradient credit = strict-mechanism xfail (2026-09-04):** the
  property certification `test_substrate_with_backprop_credit[ternary]` was a
  silent casualty of the GradientCredit fail-loud landing — never gated after
  it. Mechanism (recon, not regression): `TernarySubstrate.quantize_weights`
  STE-quantizes *substrate-owned latent* weights (`detach().clone()`), so the
  forward graph is severed from the geometry parameters by design — no
  autograd gradient can reach them, and the pairing only ever produced
  silent zeros (no learning) under the old zero-fill. Ternary learning
  routes through the substrate update operator (`ternary_update` writes
  latent + re-quantizes); pairing ternary with gradient credit needs a
  latent-graph path — fold into R11.2.9 (`substrate_coupled` engagement) if
  a research paragraph ever wants learned ternary through the 5-D pipeline.
  Marked dynamic `pytest.xfail` with the mechanism string in
  `tests/property/test_axis_certifications.py` (R11.1.3 precedent). Lesson:
  **fail-loud changes gate the property suite before landing** — this one
  shipped and the first property run caught it a session later.
- **Test acceleration (2026-09-04, user directive "tests take too damn long"):**
  property suite now runs `pytest -n auto` (pytest-xdist, already a dev dep):
  **105 s → 55 s**, verified stable ×3. Demo gate stays **serial** — parallel
  demo runs re-emit records with float drift and trip the gallery lock
  (mechanism below). Record the canonical gates:
  `uv run python -m pytest tests/property/ -q -n auto` and the serial demo
  gate. Two RNG-order-fragile property tests were the only parallel
  failures: `test_deep_network_accuracy[100]` and
  `test_eqprop_vs_backprop_accuracy` built models from unseeded global RNG
  (pass/fail depended on which tests ran earlier in the process) — seeded;
  the former's claim was then refuted (see next bullet). Known intermittent
  `XPASS`: `test_backprop_memory_grows_with_depth[25]` (non-strict, CUDA
  memory measurement noise — pre-existing, harmless).
- **Deep-settle EqProp refutation (R11.5.5 slot filled, 2026-09-04):**
  `test_deep_network_accuracy[100]`'s claim ("100-settle-step EqProp >
  30% acc after 3 steps") is false at every seed — and MORE training decays
  accuracy further (3→10→30 steps: 0.22→0.03→0.0). It previously passed
  only via an unseeded draw. Converted to a strict-mechanism xfail citing
  the R11.3.11 boundary; this is the EqProp instance of the depth/settle
  signal-loss boundary. Candidate for a live failure figure (same pipeline)
  if the multi-seed pilot's levers rescue it.
- **Float-reduction record drift (D2/D7 absorbed, 2026-09-04):** demo
  records can drift at the 1e-7 level run-to-run — multithreaded CPU
  reduction order in some kernels varies with scheduling, especially under
  parallel test workers. Asserts are tolerance-based (green); the gallery
  lock's sha check is not. Manifest re-pinned after mechanism review via
  `render_gallery` directly. `comp gallery` itself was broken at HEAD —
  its CLI imported `_FACTORIES`, which the gallery refactor renamed to the
  `DEMOS` registry; fixed (`computronium/cli/gallery.py`). Per retro (e):
  when the lock fires, diagnose (isolation re-runs, byte diffs) before
  re-pinning — both drift classes this session were diagnosed, not pinned
  blind.
- **P-axis standardization (user directive, 2026-09-04):** "M-axis" is
  retired — the plasticity axis is the **P-axis** everywhere (code
  docstrings/comments, demo D3 wording, gallery figure titles, TODO/docs;
  archives untouched). README's uncommitted M→P edits are the user's own
  and stay. No `M`-prefixed identifiers exist in code, so this was a
  documentation-level sweep (19 files).
- **#2–#5 session (2026-09-04):**
  - **Deep Hebbian probe** (`scripts/probes/deep_hebbian_chain.py` — read
    its docstring): the hebbian tile chain *explodes* at depth, not
    decays — per-layer gain ~1.2–1.5× at init compounds to inf by depth
    500, and one Hebbian local_update NaNs the activities (positive
    feedback, no gain control). Track 54's "maintains signal through 50
    layers" is unverifiable at HEAD — its
    `measure_signal_propagation` method no longer exists. Sharpened CP-6
    thesis: the depth bottleneck is **structural gain control** —
    error-based rules die by telescoping decay, unnormalized local chains
    by runaway gain, μPC's parameterization IS the normalization. Next
    lever: unit-layer-gain init or homeostatic scaling on the tile chain.
  - **Compiled LIF settle (R11.2.25 extension):** `spike_integration(..., compiled=True)`
    runs the per-layer LIF loop as one graph — bitwise parity on
    membranes, spike counts, and rasters (`test_compiled_settle.py`, now
    7 locks). **Demos stay eager**: flipping D7 to compiled busts its
    60 s timeout (compile warmup ~60 s vs 0.3 ms/step saved at demo
    scale) — compiled settle paths are registered-scale levers only.
  - **Wheel acceptance (R11.4.1/CP-5):** pyproject's flat `packages`
    list shipped only top-level modules — fixed with
    `packages.find` (subpackages verified in the wheel). New
    `tests/integration/test_wheel_acceptance.py`: builds the wheel,
    installs into a fresh venv (torch via `--system-site-packages`),
    runs the stranger's first minute (import → compose → `SystemModule`
    forward → `fit_step`). CP-5's pip-packaging door is now demonstrably
    open.
- **Deep Hebbian lead — SUPERSEDED by the probe (2026-09-04):** the
  "hundreds of layers" recollection is not realized by the current
  `DeepHebbianChain` implementation (see probe findings above). The lead
  survives as the gain-control thesis; track 54's evidence string is
  orphaned history — retire or re-home it on next touch of
  `nebc_tracks.py`.
- **Metric aggregation contract (R11.2.23, pulled 2026-09-04):**
  `SystemTrainer.train_epoch`/`validate` now accumulate **sample-weighted**
  sums (`trainer.py`) — a ragged final batch no longer counts as a full
  batch-weight — and `validate()` adds `val_ppl = exp(mean CE)`. Epoch
  numbers shift microscopically vs old records (only the ragged batch
  differs); demo gate + gallery lock re-verified green at the landing. The
  lock (`tests/unit/core/test_trainer_metric_aggregation.py`) checks the
  weighted-mean identity through a delegating spy (`_SpySystem`) because
  `_ComposedSystem` attributes are read-only — reuse that pattern for
  trainer instrumentation instead of mocks.
- **GradientCredit fail-loud (2026-09-04):** detached-weight zero-fill is
  gone; `BackpropCredit is GradientCredit` (alias). Anything that relied on
  silent zeros now raises. `LocalGoodnessCredit` intentionally keeps
  `allow_unused` (surplus recurrent self-connections).
- **SystemModule (R11.4.1 v1, 2026-09-04):** `computronium/nn/system_module.py`,
  exported root + `computronium.nn`. Training stays credit-internal —
  `fit_step`, never `loss.backward()`. Remaining for CP-5: pip packaging,
  acceptance test (RESEARCH3 PR sequence), `to(device)` passthrough if a
  consumer needs it.
- **Remaining pull-based items:** R11.1.10, R11.1.11, R11.2.9/13/14/16, R11.3.4–3.11, R11.3.13, R11.4.1/4.3/4.4. Land only when demo/campaign/research needs them.
- **Resumable trainer (R11.2.24, landed 2026-09-04):** `fold_in(base, epoch, batch, *, domain)`
  (SplitMix64, `computronium/core/system_trainer/_resume.py`) + `TrainerSnapshot`
  (epoch, global_step, history, theta, opt_state). `SystemTrainer(resumable=True)`
  reseeds the global torch RNG per epoch (domain `DOMAIN_EPOCH`, fixes the
  DataLoader shuffle draw) and per batch, so *every* downstream draw — shuffle,
  substrate noise, projection masks — is a pure function of coordinates.
  Resume: `snap = trainer.snapshot()` → `SystemTrainer.from_snapshot(system=…,
  config=…, train_data=…, snapshot=snap)`; `max_epochs` counts **total** epochs.
  Opt-in flag: `resumable=False` (default) leaves legacy trajectories byte-for-byte
  unchanged — do not flip the default without re-pinning all demo records.
  Restores into `EuclideanUpdate._momentum_buffers`; updates without optimizer
  state restore an empty opt_state (fail-loud `TypeError` if snapshot has state
  but the update object has nowhere to put it).
- **Follow-ups unlocked by R11.2.24 (as-touch / pull-based):**
  campaign episodes (R11.3.1) can set `resumable=True` to make kill→resume
  bitwise rather than statistically equal; `CheckpointManager`'s global
  RNG-state capture (`checkpoint.py`) becomes redundant per-episode once
  trainers run resumable — retire it only when no consumer needs stream-position
  resume; `fold_in` is the canonical seed derivation for any future per-batch
  keyed randomness (probes, campaign shard seeds).
- **Lint/type debt deprioritized:** ruff clean passively; pyright on new modules only. Legacy findings carry per-line noqa markers that self-flag on touch.

### New Improvement Opportunities (opened 2026-09-04, pull-based)

- **R11.5.5 failure-manifesto paragraph (CP-6 candidate, recommended next
  pull):** the depth boundary now has three *measured* failure modes —
  error-based rules die by telescoping decay (R11.3.11 frontier), 
  unnormalized local Hebbian chains by runaway gain (deep_hebbian_chain
  probe), activity-normalized Oja chains by subspace collapse
  (R11.3.14) — plus the μPC no-lift-under-our-trainer result (multi-seed
  pilot; OPEN pending the jpc-faithful port). One
  figure consolidating all four arms vs depth (same pipeline, same
  terms) is the library's first *finding* figure. Deliverable: gallery
  figure + RESULTS.md paragraph + failure-manifesto entry.
- **Suppression-system repair (Register C):** convert the repo's invalid
  `# ruff: ignore[x]` comments to working `# noqa` codes (they currently
  suppress nothing). One-time sweep; unlocks the "self-flag on touch"
  mechanism the lint directive depends on.
- **jpc-faithful μPC trainer port (R11.3.11 tail):** the remaining gap
  for a clean μPC verdict — Adam (or tuned-η) weight optimizer, activity
  GD with a β grid (paper grid 1e3→1e-2), inference steps = H, width 512.
  Reference: github.com/thebuckleylab/jpc. Also verify how jpc applies the
  output premultiplier a_L = N^{-1} before a CE-softmax readout (the
  paper clamps z_L to y with MSE; naive 1/N logits may underflow CE —
  check before porting the scale to our output layer).
- **Single-seed audit: every accuracy number quoted in probe docstrings
  and RESULTS.md back-section is single-seed. The μPC refutation shows
  seed noise can fake 2× effects. On next touch of any registered-scale
  claim, add a second seed or mark it explicitly unverified.
- **`settle`/`compute_energy` type skew:** the StateDynamics Protocol
  declares `CompositeState` but the pipeline (and every demo) passes
  `SystemState` — every new consumer needs pyright-ignore noise. Fix the
  Protocol annotation to `SystemState` (or a union) on next touch of
  `_dynamics.py`.
- **`DeepHebbianChain` readout helper:** the deleted one-hot ridge helper
  was the wrong evaluation for 1-D-coded classes; if a future demo needs
  ridge readouts, add a bias term (this is why nearest-centroid is the
  convention now).

### Sprint Retro (2026-09-03, binding for future sessions)

- (a) Tests run **once at close** — mid-session gates are ruff + pyright on
  changed files only (seconds); behavioral questions get throwaway probe
  scripts; file moves/renames get grep + pyright, no test runs at all.
- (b) Any signature/config break (required fields, renamed identifiers):
  AST-walk the *entire* repo including `tests/` for call sites before
  finishing the item — eyeballing three test files missed
  `test_power_preregistration.py` this sprint.
- (c) Behavior inherited from deleted code is not automatically correct —
  when the recon shows the old implementation was itself the debt, land the
  fail-loud upgrade, don't preserve a silent fallback.
- (d) When a plan item is phrased as either/or ("fold into X or drop"),
  cross-check the repo's current naming conventions before picking a
  direction; plan phrasing can lag the codebase (the equitile→tile case).
- (e) **Diagnose nondeterminism at the data layer first** — a figure-lock
  drift that survives re-pins is an unseeded draw (D8's loader shuffle),
  not a manifest problem; three re-pin loops were spent before the seed
  order was checked. When the same sha keeps changing: seed the stream,
  run the demo twice, compare bytes — never re-pin between.
- (f) Fairness/capacity requirements arrive mid-item and reshape the demo
  (conv ≈1/10 params) — apply them by re-balancing the *weaker* arm, not by
  asserting superiority of the over-provisioned one.
- Work lean: one Register item per landing, each with a test that
  demonstrates it. Don't pull infrastructure "just in case".
- RESEARCH3 protocol (E-1 smoke → pilot → full; E-11 DECISIONS.md) governs
  every R11.3 pull. Infra-failures don't consume tuning rounds.

---

## 🚪 The R12 Fork (2026-09-04 — decision point, not work)

R11 is **core complete**. D1–D12 demonstrate every axis; `comp repro` 8/8;
property suite, demo gate, gallery lock green at HEAD. What remains is not
cleanup — it is choosing which future the completed library serves. The
remaining checkpoints are three different futures:

| Future | Checkpoint | What it produces |
|--------|-----------|------------------|
| Make it visible | CP-5 Adoption | Wrapper / UI — someone who isn't you can use it |
| Make it say something | CP-6 First finding | P-axis Pareto frontier over 𝒞 — first figure that is a *finding*, not a demonstration |
| Big swing | CP-7 Discovery bet | Z3 flagship, pre-registered with fallback |

**Standing recommendation: CP-6 before CP-5.** The instrument was made
honest at real cost; the payoff of honesty is a finding, and adoption
follows what the instrument shows, not its surface. The backlog already
points there — μPC depth scaling (R11.3.11), depth-metric classes
(R11.3.13), ePC's deep-stack credit limitation (Notes: contrastive ÷β)
are all deep-EqProp-boundary territory, and the PR-5 stability guard is
calibrated and idle. Demo-scale machinery + registered-scale GPU (conv
speedup measured) is exactly the sweep CP-6 needs. CP-7 rides CP-6's
findings; CP-5 stays pull-based until a finding gives people a reason to
adopt.

**Decision (2026-09-04, user): proceed with CP-6 first — and all three
options will eventually be built.** Sequencing commitment: CP-6 → CP-5 →
CP-7, with each door pulled when its predecessor lands a reason. First
concrete step: R11.3.11 + R11.3.13 as one landing (μPC init + depth
metrics), probe-first per RESEARCH3 E-1.

---

## 🔬 CP-6 Execution Doctrine (2026-09-04 — external strategy review, applied)

R11 core-complete means the posture shifts: not building a library —
operating a completed, verified, honest instrument. Three priorities,
everything else ruthlessly ignored while CP-6 runs.

### 1. Prime objective — interrogate the deep-EqProp boundary (R11.3.11 + R11.3.13)

The library treats locality, energy, and physical constraints as
first-class; plain BP and plain PC both decay through deep local-learning
regimes (~10 layers). The question CP-6 answers: do **μPC** (depth-scaled
init) and **ePC** (error reparameterization) actually solve this on
non-trivial topologies, or merely shift the failure mode?

- E-1 smoke probe **done** (`scripts/probes/mupc_depth_init.py` — read its
  docstring before re-deriving: boundary at depth ≥ 8 is not PC-specific;
  μPC ≈ 2× PC learning at depth 8 under a real budget).
- Implement `ShortestPathDepth`/`LongestPathDepth` (R11.3.13) so "depth" is
  measurable per-node on `GraphGeometry`/`TileMesh`, where layer-counting
  fails; then sweep μPC × ePC × substrate-noise across depth.
- Deliverable: *the exact depth and substrate-noise constraint at which
  local credit physically breaks down vs global backprop.*

### 2. Leave the CPU sandbox — registered-scale GPU campaigns

The demo suite (D1–D12) is CPU/kernel-launch-bound by measured verdict; it
proves the *ontology*, not a *finding*. Registered-scale work runs GPU with
the `torch.compile` settle fast paths (R11.2.25) already landed; conv-family
is the measured 15× FLOP-bound path.

- Commission AutoScientist on a registered-scale sweep of the
  **stability–plasticity frontier**: map the 𝒞-vector Pareto frontier
  (compute, memory, energy, latency, plastic-state capacity) across
  plasticity primitives (Routing vs FastWeight vs Null).
- Deliverable: one figure showing the trade-off between settling time
  (dynamical latency) and basin stability per plasticity primitive — the
  first *finding* (checkpoint 6), not a demonstration.
- PR-5 stability guard is calibrated and idle — wire
  `probe_interval_for_overhead` into the campaign loop when it starts.

### 3. Hunt for refutations (standing rule R11.5.5)

The goal is the *physics of learning*, not benchmark wins over PyTorch —
accuracy horse-races are solved and boring. A negative result from the deep
μPC sweep **is the finding**: publish it in the failure manifesto with the
same pipeline, same figures, same terms (*"local energy-based learning
collapses at depth L > 12 due to X; μPC scaling fails to rescue it because
of Y"*). A rigorously documented failure boundary for local, asynchronous,
energy-minimizing systems is a real contribution to neuromorphic and
biologically-plausible ML. The instrument stays honest even when the
hypothesis dies — and the spiking-family learning slot (R11.5.5) remains
open for exactly this kind of live refutation.

---

## ⚡ Performance Proposals — Evaluated (2026-09-04 external review)

Assessed against measured regime facts (demo suite is CPU and
kernel-launch-bound, not FLOP-bound; conv-family is the first GPU-bound
path at 15×; `KernelRegistry` already hosts Triton-family kernels in
`acceleration/`). All are **pull-based per R11.5.6** — perf work lands
when a registered-scale study or campaign needs it, never speculatively.

| Proposal | Verdict | Note |
|----------|---------|------|
| `torch.compile` on settle/credit paths | **Viable, pull with a registered-scale GPU study** |compile helps FLOP-bound paths; the demo suite's Python settle loop is launch-bound and CPU-pinned — compile would not move the gate. Caution: settle loops have data-dependent convergence breaks and `no_grad`/`enable_grad` context switches; wrap whole-settle, not per-step. |
| `torch.vmap` over settle steps | **Rejected (category error)** | Settling is sequentially dependent; `vmap` maps over batch axes, not time. Batch parallelism already exists via the loader. |
| Triton kernels for substrate ops | **Viable, as-touch** | `acceleration/` kernel infrastructure exists (snn/contrastive kernels); extend when a commissioned study's profile shows the operator is the bottleneck. Not TODO.md R4.3 (that is `UV_LINK_MODE`). |
| `system.compile()` static-graph export | **Pull with CP-6/CP-7** | Only worth building when campaign fleets hit abstraction overhead at registered scale; measure first (profiling infra exists). |

---

## Termination Criterion

R11 closes when a stranger can, in one sitting: compose a system from *any*
geometry, substrate, and dynamics the ontology declares (not just the
Feedforward/Recurrent/Digital/Memristive/EnergyMinimization set R10
demonstrated), watch it train in the demo suite (the live UI is a separate
adoption round, R11.4.3 — presentation, not a library-completeness gate),
and find
the repo's own gates — ruff, pyright, property locks, demo gate, figure
lock — green at HEAD without caveats. The library is then *complete relative
to its own ontology*: every axis declares only primitives that exist and
demonstrate themselves. Research claims remain where they belong — the
corroboration appendix and RESEARCH3, pull-based, never the front page.