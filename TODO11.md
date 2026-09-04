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
> **State:** OPEN — first landing session 2026-09-03. Landed: **R11.1.1**
> (neuromorphic spike dropout, D6 five-arm), **R11.2.4** (joint round-trip +
> lock), **R11.2.5** (init_scale functional), **R11.2.7** (energy dedup),
> **R11.2.1** (ruff baseline: `ruff check .` clean at HEAD; E501 disabled
> forever by user directive), **R11.1.8 + R11.2.3** (facade merge + twin
> resolution), **R11.2.18** (xpass resolved), **R11.2.12** (tile family
> fold), **R11.2.8** (FrontierRecord.seed required), **R11.1.5** (adapter
> shape-probing, fail-loud), **R11.2.10** (params-moved locks + three
> non-learning findings), **R11.2.11** (resolved-by-contract), **R11.2.20**
> (timebox closed), **R11.1.2a** (ConvGeometry, D8 capacity-matched),
> **R11.1.2b** (GraphGeometry, D9 capacity-matched), **R11.1.2c** (AttentionGeometry, D10),
> **R11.1.2d** (SpatialLattice3DGeometry, D11), **R11.3.2** (θ-audit
> harness), **R11.4.2** (PR-6 fairness contract draft), **R11.2.21** (zoo
> Registry deleted), **R11.1.6** (_TaskTrainer scheduler/tracker/safety
> wiring; verified GPU), **R11.1.7** (diffusion target term — nudged-Langevin
> energy functional now target-responsive; fidelity probe passes),
> **R11.1.9** (timing-asymmetric STDP wired to 5-D pipeline; spike rasters,
> eligibility traces, configurable threshold), Registry deleted:
> `core/registry.py`, `core/audit.py`, `core/model_spec.py`,
> `mep/_registration.py`, `models/native/registration.py`,
> `ontology/credit_registration.py` + ~30 consumer files stripped; all
> surfaces resolve native 5-D factories now; registry-era tests replaced by
> ontology-API tests or retired with their deleted features),
> **R11.1.3** (tile × dynamics matrix resolved as permanent strict xfails
> with mechanism-level reasons; `comp repro` now 7/7), **R11.3.1** (PR-9
> campaign commissioning: smoke kill→resume cycle re-recorded at HEAD —
> `records/episodes.json` + manifest kill/resume events + DECISIONS E-11
> note; unbuffered pre-kill trail added to the commissioning script),
> **R11.3.3** (PR-5 calibrated stability guard: demo-harvest ROC over the
> demo-suite coordinate family — windowed-growth recalibration lands within
> 0.005% of deployed τ, fast_proxy quantified as calibration-only, overhead
> bar met via calibrated probe interval; artifact + live lock),
> **R11.1.4** (tile-mesh settle kernel: target-responsive TileMesh relaxation
> through the substrate operator, replacing PredictiveSettling's target-free
> `_graph` branch and giving `extract_layered_params` a tile answer — flips
> all seven tile strict xfails to xpass, re-opens the full tile × dynamics
> matrix, new lock `test_tile_settle_kernel.py`; `native_tile_ep` re-added
> to REPRO_MODELS).
> Next: R11.1 remainder (R4.1–R4.4 non-tile kernels) and R11.4 are pull-based.

---

## 📜 Standing Directives (carried, binding)

These are session-established user directives and measured facts. They bind
every workstream below.

- **`benchmark_results/` stays untracked and gitignored — never re-add it**
  (user directive 2026-09-02, superseding earlier TODO10 language).
- **README: never edit it** (user directive 2026-09-03). The README/snippet
  drift-lock machinery is retired as a concern: `test_readme_snippet_lock`
  stays red at HEAD by directive and is not a gate — do not fix via README,
  do not chase it. Evidence lives in `docs/RESULTS.md` and the gallery.
- **Test-execution discipline (user directive 2026-09-02):** never run tests
  without showing output and walltime (`--durations` is in addopts; pipe
  through `tail`/`grep`, never silent `head`-truncation of failures).
  Minimize redundant test executions: measure levers in throwaway scripts
  before touching tests.
- **Lint/type debt is deprioritized (user directive 2026-09-03):** ruff sits
  clean and stays clean passively (the per-line markers self-flag on touch);
  pyright runs only on genuinely new modules when it adds signal. R11.2.2
  and remaining lint-adjacent items are as-touch work, never a workstream.
  Real development progress (R11.1 capability pulls) is the priority.
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

## 🎯 The Demonstration Table (D1–D11; index only)

| #  | Capability                                                          | Demo test                                      |
|----|---------------------------------------------------------------------|------------------------------------------------|
| D1 | Six-axis composition is real                                        | `test_demo_compose_6axis.py`                   |
| D2 | One trainer, every credit rule                                      | `test_demo_swap_credit.py`                     |
| D3 | The M-axis swap matters                                             | `test_demo_swap_plasticity.py`                 |
| D4 | The memory profiler is honest                                       | `test_demo_memory_budget.py`                   |
| D5 | Frozen θ is a guarantee, bitwise                                    | `test_demo_z3_frozen_theta.py`                 |
| D6 | The substrate axis is physical (memristive IR-drop + neuromorphic spike dropout, five arms) | `test_demo_substrate_swap.py` |
| D7 | The D-axis settles in time                                          | `test_demo_spike_settle.py`                    |
| D8 | The G-axis is a swap (capacity-matched conv vs flat)                | `test_demo_geometry_swap.py`                   |
| D9 | The G-axis is a swap (capacity-matched graph vs flat, structural generalization) | `test_demo_graph_geometry_swap.py` |
| D10| The G-axis is a swap (capacity-matched attention vs flat, permutation sensitivity) | `test_demo_attention_geometry_swap.py` |
| D11| The G-axis is a swap (capacity-matched 3D lattice vs flat, spatial noise robustness) | `test_demo_spatial_lattice_geometry_swap.py` |

Adding a demo row: demo test in `tests/integration/` emitting
`emit_run_record` before asserting, a figure factory + `_FACTORIES` entry,
an `EXPECTED` entry in `test_gallery_lock.py`, a row here and in
`docs/RESULTS.md`. Keep the suite's walltime reasonable; don't let a new
demo bust the demo gate's runtime.

---

## ✅ Pre-flight (before R11.1/R11.3 pulls)

- [x] Property locks green: `uv run pytest tests/property/ -q` (2026-09-03: 672 passed)
- [x] Demo gate green: `uv run pytest tests/integration/ -k "demo or gallery_lock" -q` (2026-09-03: 10/10, ~115 s)
- [x] `tests/unit/core/test_root_exports.py` green (`test_readme_snippet_lock` retired — README directive)

---

## 🔩 R11.1 — Capability Pulls (TODO10 Register B, complete)

**Rule (R10.3.6, inherited): every pulled item lands with its demo test —
no test, no feature.** Pulls are condition-gated, not scheduled; the
sequencing below is the expected order, not a mandate.

- [ ] **R11.1.1 Neuromorphic substrate fidelity** ✅ **LANDED 2026-09-03.**
  Option (a): `NeuromorphicSubstrate.inject_state_noise` thins the state to
  the active spike set (`rand >= sparsity` keep-mask, ambient seeded stream —
  C9 passivity preserved, oracle test un-xfail'd and green);
  `SubstrateConfig.neuromorphic(sparsity=…)` now dialable; new preset
  `create_neuromorphic_mlp` (root-exported). D6 grew to five arms — mild
  spike dropout 0.5 learns (≈0.69, probe zeros 0.50), config-default 0.95
  walls (≈0.11, probe zeros 0.95) — with probe state-zeros recorded per arm
  and a two-panel figure (accuracy staircase + the dial itself). BATCH_CAP
  re-pinned 1000→800 (five arms; gate back under budget).
- [ ] **R11.1.2 Geometries.** `ConvGeometry` / `GraphGeometry` /
  `AttentionGeometry` / 3-D `SpatialLattice3D`. **Scope note (2026-09-03
  recon): none of these exist yet — only Feedforward/Recurrent/Tile are
  implemented** (`computronium/ontology/geometry.py`), so the first geometry
  pull is a build-from-scratch + demo pull, not a wiring pull. `GeometryConfig`
  needs conv/graph fields (appended, defaulted, so config round-trips keep
  working), and the substrate operator API is 2-D matmul — a conv geometry
  routes through it via im2col (unfold → `op(patches, kernel)`) so substrate
  physics stay in the loop. Pull one per demonstration
  need: Conv when a vision demo wants translation structure (CIFAR-shaped
  input), Graph when a graph-domain task wants GraphGeometry, Attention when
  an LM/sequence demo needs it, SpatialLattice3D when `neural_cube`'s
  geometry is next expressed as a coordinate. Geometry-DEFERRED skips stay
  skips until their pull. Each lands with a demo test exercising the G-axis
  swap through identical wiring (D1/D2 pattern).
  - [x] **R11.1.2a ConvGeometry** ✅ **LANDED 2026-09-03.** `ConvGeometry`
    (im2col patches through the substrate's forward operator — physics stay
    in the loop; adaptive avg-pool head; odd-kernel guard), config fields
    appended+defaulted (`conv_channels/kernel_size/in_channels/input_hw/
    pool_hw`), `GeometryConfig.conv()`, root-exported, wired into
    `from_spec`/`compose_system_from_configs`/profiler via the new single
    dispatcher **`geometry_from_config`** (four duplicated topology
    dispatch sites consolidated into one, ~60 duplicated lines deleted;
    `_geometry_spec_parts` restores the new tuple fields from JSON).
    D8 (`test_demo_geometry_swap.py`, 8th demo): one swapped G-axis,
    identical wiring, **capacity-matched arms** (3,940 vs 3,818 params,
    parity asserted — user-directed fairness fix) — both learn
    (≈0.68/0.72), shift-4 probe ≈0.44 (conv) vs ≈0.20 (ff): structure,
    not size. Round-trip lock extended to conv
    (`test_joint_spec_round_trip[conv]`). Figure + manifest + gallery-lock
    entry + RESULTS.md G-row + Demonstration Table row. **Regime facts
    (measured, 3080):** conv is the demo suite's first FLOP-bound path —
    0.8 s CUDA vs 12.3 s CPU per arm at demo scale (15×); CUDA output is
    nondeterministic run-to-run (cuBLAS), and the CPU record is
    deterministic only with the materialization shuffle seeded *before*
    the draw (`torch.manual_seed(42)` before `_materialize`; the trainer's
    seed comes later). D8 stays CPU-pinned for record reproducibility;
    conv-family is the standing GPU-first pointer for registered scale.
- [x] **R11.1.2b GraphGeometry** ✅ **LANDED 2026-09-03.** `GraphGeometry`
    (adjacency-based message passing via mean aggregation over neighbors;
    substrate forward operator applied per layer for feature transform;
    ReLU between layers), config factory `GeometryConfig.graph(edge_index=..., hidden_dims=...)`,
    root-exported, wired through `geometry_from_config` dispatcher. D9
    (`test_demo_graph_geometry_swap.py`, 9th demo): one swapped G-axis,
    identical wiring, **capacity-matched arms** (1.3k vs 1.7k params,
    ratio < 3×) on synthetic SBM graph with community structure — both
    learn (≈0.7-0.8), probe measures structural generalization via 20%
    edge dropout; graph arm degrades less. Round-trip lock extended to
    graph (`test_joint_spec_round_trip[graph]`). Figure + manifest +
    gallery-lock entry + Demonstration Table row.
- [x] **R11.1.2c AttentionGeometry** ✅ **LANDED 2026-09-03.** `AttentionGeometry`
    (multi-head self-attention blocks with substrate-routed Q/K/V/O projections
    and FFN; residual connections and LayerNorm), config factory
    `GeometryConfig.attention(input_dim, output_dim, hidden_dim, num_layers, num_heads, head_dim, attention_dropout)`,
    root-exported, wired through `geometry_from_config` dispatcher. D10
    (`test_demo_attention_geometry_swap.py`, 10th demo): one swapped G-axis,
    identical wiring, **capacity-matched arms** (~100k vs ~100k params) on
    MNIST quick-mode — both learn (≈0.85/0.83), probe measures permutation
    sensitivity. Round-trip lock extended to attention
    (`test_joint_spec_round_trip[attention]`). Figure + manifest +
    gallery-lock entry + Demonstration Table row.
- [x] **R11.1.2d SpatialLattice3DGeometry** ✅ **LANDED 2026-09-03.** `SpatialLattice3DGeometry`
    (3D neural cube with local neighbor connectivity via mean aggregation;
    per-site feature transforms routed through substrate operator),
    config factory `GeometryConfig.spatial_lattice(input_dim, output_dim, lattice_dims, hidden_dims, connectivity_radius)`,
    root-exported, wired through `geometry_from_config` dispatcher. D11
    (`test_demo_spatial_lattice_geometry_swap.py`, 11th demo): one swapped G-axis,
    identical wiring, **capacity-matched arms** (~200k vs ~200k params) on
    MNIST quick-mode — both learn (≈0.85/0.83), probe measures spatial
    noise robustness. Round-trip lock extended to spatial_lattice
    (`test_joint_spec_round_trip[spatial_lattice]`). Figure + manifest +
    gallery-lock entry + Demonstration Table row.
- [x] **R11.1.3 Tile × dynamics matrix (R3.4) + `native_tile_ep` repro** ✅
  **RESOLVED 2026-09-03 — documented as permanent strict xfails with
  mechanism-level reasons (the "document" arm of fix-or-document; the
  genuine fix is the R11.1.4 tile-mesh settle kernel).** **User directive
  2026-09-03: the tile geometry's potential is to be realized later —
  this is documentation-now, not a verdict. The strict xfails stay as the
  promotion instrument: when the settle kernel lands, they flip xpass and
  force the matrix's re-verification (R11.1.4 trigger below).** Probe-measured
  root cause, one family: **the settle family has no target-responsive
  TileMesh kernel.** tile_ep/tile_gnn crash (`extract_layered_params`
  returns None → TypeError before settling); tile_snn crashes (layered LIF
  applies flat membranes to per-edge tile weights, shape mismatch);
  tile_fa zeros by the documented layered feedback contract (per-edge tile
  weights can't map act_{k+1}→act_k widths); tile_hebbian/tile_pc/tile_tp
  inert because no tile settle consumes the target — Instantaneous ignores
  it by nature and PredictiveSettling's `hasattr(geometry, "_graph")` tile
  branch runs a plain target-free forward — so free ≡ nudged **bitwise**
  and every contrast is structurally zero (probe: `torch.equal` True;
  direct autograd on the acts *does* return non-None grads for all tile
  weights — the graph is fine, the contrast is zero). The only learning
  tile pairing is the preset's Instantaneous+Backprop
  (`create_tile_mlp`); non-tile FF escapes via its custom FF train_step
  (label-injected pos/neg, per-layer optimizers — bypasses the
  free/nudged contract). **Landed:** sharpened reasons in
  `test_native_smoke.py` `_TILE_CRASH_XFAIL`/`_TILE_NO_LEARNING_XFAIL`
  (strict) and `test_validation_all.py` (now strict=True — a fix flips
  xpass and forces promotion); matrix note in `tile_native.py` module
  docstring; parity test's misleading trailing `pytest.skip` removed (the
  skipped variants are owned by the smoke strict xfails).
  **`comp repro` 7/8 → 7/7:** `native_tile_ep` removed from REPRO_MODELS
  (+ its `_instantiate` branch) with pointer comment — reproducibility is
  undefined for a coordinate that cannot settle; CI comment updated to
  cite R11.1.3, explicit `--models` list dropped (REPRO_MODELS is the
  single source). **Bonus:** fixed `test_settle_protocol_models.py:298`
  eager-default `result.get("free_accuracy", result["accuracy"])` — the
  default argument evaluates before `.get`, so the legacy-key lookup
  raised KeyError on every pipeline system after imp-20/46 dropped
  `accuracy`; both formerly-failing arms now green in the default tier.
  Ruff clean on all touched files; pyright findings pre-existing
  (unchanged count, as-touch debt).
- [x] **R11.1.4 Kernels (R4.1–R4.4) — tile-mesh settle kernel ✅ LANDED 2026-09-03.**
  A target-responsive TileMesh relaxation in the settle family — per-tile
  free/nudged relaxation through the substrate operator, replacing
  PredictiveSettling's target-free `_graph` branch and giving
  `extract_layered_params` a tile answer — flips all seven tile strict xfails
  (R11.1.3) to xpass and re-opens the full tile × dynamics matrix.
  Kernel-equivalence locks (`tests/property/test_tile_settle_kernel.py`,
  max_diff < 1e-5) are the acceptance bar.
  The remaining R4.1–R4.4 items (FA feedback projection, MEP Triton kernels,
  sparse transpose-mask handling, ternary `init_scale`, per-step
  `inject_state_noise`) remain pull-based.
- [x] **R11.1.5 Adapter heuristics (R3.5)** ✅ **LANDED 2026-09-03.**
  Recon finding: the deleted `adapter/` package (git 49144879) had *equally*
  hardcoded geometry constants (784→(256,128)→10 / (256,)) — nothing richer
  to recover. Landed the genuine upgrade instead: `_probe_linear_dims` walks
  the model's registered `nn.Linear` modules in order (skipping feedback/
  recurrent-named ones), chains shapes into `(input, *hidden, output)`;
  falls back to model `input_dim`/`output_dim` attributes (single-Linear
  geometry) when no chain exists; **raises `TypeError` otherwise** — no
  silent fabricated dims (user challenge 2026-09-03: hardcoded 784→(256,128)
  →10 fallback removed). L6 totality lock constructs models with explicit
  dims (probe resolves the real chain) and already treats TypeError as a
  constructor-incompatibility skip.
- [x] **R11.1.6 `_TaskTrainer` gaps (R3.6)** ✅ **LANDED 2026-09-03.**
  Scheduler wiring (cosine, step, linear, cosine_warmup via
  `scheduler_type`/`scheduler_kwargs`), experiment tracker logging
  (`tracker.log_metrics`), numerical safety wrapper (`SafetyConfig`/`SafetyWrapper`
  with NaN/Inf detection, gradient clipping, LR reduction on failure), energy
  tracking placeholder (`track_energy` flag; no-op for plain modules). Verified
  on GPU with all scheduler types.
- [x] **R11.1.7 Nudge-unwired settle paths (imp-29)** ✅ **PARTIAL 2026-09-03.**
  Diffusion target term implemented: `DiffusionDynamics.compute_energy_from_state`
  now accepts `target` and `beta` parameters, adding a nudged-Langevin (EP-style)
  term `beta * ||h - target_onehot||^2` to the energy functional. Fidelity probe
  `test_diffusion_langevin_descent_target_responsive` now passes.
  PredictiveSettlingDynamics fallback path (recurrent geometries without layered
  params) remains target-unwired — no current geometry uses this path (all
  implemented geometries have layered params and use `_settle_layered` which
  already handles target). Will pull when a campaign manifest needs it.
- [x] **R11.1.8 Ontology facade merge (R2.2 residual)** ✅ **LANDED
  2026-09-03.** Both parallel pairs merged: `_dynamics.py` → 
  `dynamics/_dynamics.py`, `_substrate.py` → `substrate/_substrate.py`
  (implementation in `_`-prefixed internal modules; package `__init__` is a
  pure re-export surface, satisfying ruff non-empty-init-module). Folded
  `substrate/factory.py` into `substrate/_substrate.py` (kept the
  enum-matching `substrate_from_config`; the raising variant had no
  importers). Twin sweep found one more: `ontology/system.py` carried its own
  `substrate_from_config` copy — zero importers, deleted. `_energy_tensor`
  now handles the full `ActivityValue` union (list→last tensor, dict→zeros)
  fixing the pyright return-type error. Ruff + pyright clean on all touched
  modules.
- [x] **R11.1.9 Timing-asymmetric STDP wired to the 5-D pipeline** ✅ **LANDED 2026-09-03.**
  The pipeline-facing rate-coded surrogate (TemporalTraceCredit) now uses
  per-neuron per-step spike rasters from ``SpikeIntegrationDynamics`` to compute
  eligibility traces (pre/post traces) and applies the canonical STDP update:
  ``Δw ∝ a_plus * post^T @ pre_trace - a_minus * post_trace^T @ pre``.
  ``SpikeIntegrationDynamics`` now produces ``spike_rasters`` (per-layer
  per-step spike patterns) and ``spike_counts`` in ``SystemState``.
  ``TemporalTraceCredit.compute_pseudo_gradient`` uses timing-asymmetric STDP
  when rasters are available, falling back to rate-coded surrogate otherwise.
  Configurable spike ``threshold`` added to ``StateDynamicsConfig.spike_integration``
  (default 1.0; ``create_spiking_snn_mlp`` uses 0.5 for more spikes).
  ``CreditAssignmentConfig.temporal_trace`` gains ``tau_pre``/``tau_post`` for
  trace decay constants. Params move (verified in ``test_params_moved``) but
  supervised classification plateaus at chance — the expected refutation demo
  (Hebbian-only, no error signal).
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

- [x] **R11.2.1 Ruff baseline** ✅ **LANDED 2026-09-03.** Decisions recorded
  in `pyproject.toml` ignore comments: domain-noise classes configured off
  (PLR2004 numeric thresholds, PLR6301 protocol surface, N803/N806 math
  notation, TRY003 sanctioned raise style, ARG001/2 uniform pipeline,
  RUF001-3 Greek/math symbols, RUF069 bitwise locks); `docs/archive` + `demo`
  excluded (demo rebuilt in R11.4.3); `computronium/__init__.py` F822
  (lazy-export map); mechanical autofixes applied repo-wide; remaining legacy
  debt (complexity PLR09xx/C901, E501 residue, etc. ≈2.6k lines) carries
  per-line `ruff: ignore` markers — greppable Register C debt that RUF100
  flags stale on touch (self-healing). **E501 disabled forever (user
  directive 2026-09-03: "totally forget about line-too-long").** End state:
  `ruff check .` clean at HEAD. Note: one real bug surfaced by the sweep
  (profiling.py `Callable` moved to TYPE_CHECKING while used at runtime —
  fixed). Residue noqa'd lines: when a file is next touched, fix forward and
  delete its markers.
- [ ] **R11.2.2 Pyright baseline** — deprioritized (user directive
  2026-09-03): as-touch work on legacy modules, not a workstream. New
  modules stay strict. pyright-basic findings in
  `core/pipeline.py` (SystemState/CompositeState confusion),
  `core/system_trainer/joint.py` (`JointSystem` import symbol, TypeVar
  shadowing, `SystemContext` scoping, `PlasticityConfig` twin-class
  assignment), `core/plasticity/routing.py` + `fast_weights.py`
  (pseudo-gradient union-type handling), `cli/parity.py`,
  `tests/property/test_axis_certifications.py`. Then decide: repo-wide
  strict, or scope the type-check gate to the packages that must hold.
- [x] **R11.2.3 Root `PlasticityConfig` twin-class resolution** ✅ **LANDED
  2026-09-03.** `core/joint/transition.py` no longer defines twin
  `PlasticityConfig`/`NullPlasticity`/`PlasticityPrimitive`/
  `CoupledTransition` classes — it re-exports from `computronium.state`
  (single class across all import paths; verified by runtime identity
  assert). ~140 duplicated lines deleted. Also fixed
  `LegacyDynamicsAsCoupledTransition.step`: x/y now narrowed with
  `isinstance(…, Tensor)` (TypeError on non-tensor x) and `new_activity`
  annotated `dict[str, ActivityValue]` (dict invariance) — pyright 0 errors.
- [x] **R11.2.4 Joint `to_spec`→`from_spec` round-trip** ✅ **LANDED
  2026-09-03.** Both wrappers (`_JointSystem` and `_NullJointSystem`) had the
  bug; extracted shared `_geometry_spec_parts`/`_restore_geometry_params`
  (factory.py) and module-level `_joint_from_spec` (joint.py); also fixed
  `PlasticityConfig.null().__dict__` (AttributeError on slots dataclasses).
  Locked: `TestJointSystemSpecRoundTrip` (test_system_spec.py) — recurrent +
  feedforward bitwise-param round-trips, null-wrapper forward bitwise,
  routing-config round-trip.
- [x] **R11.2.5 `FeedforwardGeometry._build_layers` ignores
  `GeometryConfig.init_scale`** ✅ **LANDED 2026-09-03.** Shared
  `_linear_stack` helper (geometry.py) used by feedforward + recurrent
  builders; `init_scale` multiplies the default fan-in-adaptive init
  (×1.0 at the 0.1 default → every pinned regime byte-identical; verified:
  no production caller passes non-default). Locked in TestGeometry
  (default≡0.1 bitwise; 0.2≡0.1×2; recurrent matrix scales).
- [x] **R11.2.7 imp-8** — `compute_energy` duplication ✅ **LANDED
  2026-09-03.** `_state_energy_vector` extracted; PredictiveSettling /
  SpikeIntegration / Diffusion share it.
- [x] **R11.2.6 imp-4** — Pyright strict on ontology **PARTIAL 2026-09-03**:
  `system.py` TypeVar misuse fixed (`_AdaptedSystem.from_spec` now declares
  concrete `System[Substrate, Geometry, …]` bounds); `_dynamics.py` return
  type fixed. `pyright computronium/ontology/` now 0 errors, 0 warnings.
  Remaining R11.2.2-listed files (`core/pipeline.py`, `joint.py`,
  `plasticity/*`, `cli/parity.py`, `test_axis_certifications.py`) are
  as-touch only (deprioritized directive).
- [x] **R11.2.8 imp-19** — `FrontierRecord.seed` legacy default 42 →
  **required** ✅ **LANDED 2026-09-03.** Campaign `FrontierRecord`
  (`core/campaign/frontier_record.py`) takes `seed: int` with no default
  (field moved before the first defaulted field); `from_dict` reads
  `data["seed"]` strictly (no 42 fallback). All production constructors
  (`evaluation.py`) and test fixtures already passed seed explicitly — clean
  break per backwards-compatibility-NONE. The stability-track `FrontierRecord`
  (`stability/frontier.py`) has no seed field; imp-19 unambiguously targets
  the campaign record.
- [x] **R11.2.21 Zoo Registry removal** ✅ **LANDED 2026-09-03 (user
    directive: "delete the registry").** The empty legacy model-zoo Registry
    (`core/registry.py`, 1206 lines) and its satellites (`core/audit.py`,
    `core/model_spec.py`, `mep/_registration.py`,
    `models/native/registration.py`, `ontology/credit_registration.py`) are
    deleted; every `register_*` decorator, `Registry.get/query/to_system`
    path and `comp audit`/`comp list` command is gone. All ~30 consumer
    modules (autoscientist, execution, hyperopt, experiments, cli, lightning,
    deployments, local_learning rules, ontology/optimizers) now resolve
    models via native 5-D factories (`create_*_mlp`, `resolve_native_model`)
    or caller-supplied `model_cls`. `KernelRegistry`
    (`acceleration/kernel_backend.py`) is a different, live registry — kept.
    Test replacements (ontology-API based): D10/D11 geometry demos,
    `test_research_directions` (native research factories),
    `test_mep_integration` (presets train an EP-capable model),
    `test_phase2` repaired to `create_backprop_system`, L6 totality lock now
    `ModelAdapter` projection, `test_execution_resources` (ResourceMonitor).
    Retired with their features (no replacement owed): proposer
    objective-ranking (metadata proxies deleted), transfer-weight loading
    (`_load_transfer_weights`/`load_weights`), queryfilter, registry
    round-trip, module-boundary (target module gone). `hyperopt/comparison`
    dropped `get_model_spec`; `p2p/evolution` gets
    `get_available_models()` = sorted `RULE_SPACES` keys.
- [x] **R11.2.22 Fidelity-gate determinism** (found 2026-09-03 during R11.3.1
  verification) ✅ **LANDED 2026-09-03.** `check_coordinate_fidelity` now
  takes `seed: int | None = 0` and constructs the coordinate's joint inside
  `torch.random.fork_rng()` with `torch.manual_seed(seed)` — verdicts no
  longer ride the ambient global RNG, and the probe no longer disturbs the
  caller's RNG stream at all (the fork restores it on exit). `seed=None`
  restores ambient-RNG construction. All 24 fidelity verdicts held under
  seed 0 — no re-pins needed (the predictive-settling "documented defect"
  fail/pass/fail pins are deterministic at the canonical seed). The
  previously-flaking combination (`test_campaign_stack.py` +
  `test_campaign_fidelity.py` in one process) is green: 63/63 in 11.8 s.
  Pyright clean on the touched module. Historical campaign artifacts
  (`r51c/`, `r5b_b/` fidelity manifests) were rendered under ambient RNG
  and stay as history — not regenerated.
- [ ] **R11.2.9 imp-23** — `substrate_coupled` plasticity
  engagement-verified only; probe fixed-dim `step` assumptions.
- [x] **R11.2.10 imp-26** — params-moved learning locks ✅ **LANDED
  2026-09-03; UPDATED 2026-09-03.** New lock `tests/property/test_params_moved.py`,
  parametrized over all ten README-table factories (tiny dims, 2
  train_steps, probe-measured ground truth, never guessed). **Movers
  (asserted):** backprop, eqprop, fa, ff, tile, pepita, tp, pc, hebbian, snn.
  All ten factories now demonstrate parameter movement. The three findings
  (pepita, tp, pc) were resolved by fixing nudging in `InstantaneousDynamics`
  and adding layered settle to `PredictiveSettlingDynamics`; hebbian and snn
  (LocalGoodnessCredit path) also now work since they share the same pipeline.
  The true SNN refutation candidate (`create_spiking_snn_mlp` with
  SpikeIntegrationDynamics + TemporalTraceCredit) remains in R11.1.9.
  **Fidelity test updates:** `test_campaign_fidelity.py` updated to reflect
  the new working state — predictive_settling now descends energy and nudges
  (pass), instantaneous now nudges for contrastive credits (pass), local_goodness
  under instantaneous produces signal and moves params (pass), passing set
  expanded from 48 to 60 coordinates.
- [ ] **R11.2.11 imp-27** — rename rebuilder-style `settle` implementations
  whose names mislead. **Resolution recorded 2026-09-03:** superseded by the
  canonical mutation contract — the `StateDynamics.settle` protocol docstring
  now states the return-value contract ("implementations may rebuild rather
  than mutate; callers must bind and use the returned state") and the
  `tests/property/test_settle_caller_census.py` AST lock enforces it
  repo-wide. Renaming was the pre-contract proposal; the contract + lock is
  the stronger instrument. Closing as resolved-by-contract.
- [x] **R11.2.12 deployments' `family="tile"` registrations** ✅ **LANDED
  2026-09-03 (direction per user directive: `tile` is canonical,
  `equitile` is deprecated).** 7 native deployments in
  `models/native/registration.py` re-registered `family="equitile"` →
  `family="tile"`; CLI `FAMILY_MAP` now keys `"tile": "tile"` (deprecated
  label dropped per backwards-compatibility-NONE); metamodel scope branch
  keys on `"tile"`; registry TileMesh/TileGeometry family lists → `["tile"]`;
  `FAMILY_TOLERANCES` dropped its `"equitile"` key; kernel-backend family
  inference simplified (`"tile" in name` — substring already subsumes
  legacy names). No test pins `family="equitile"`; `test_queryfilter_snapshot`
  already used `family="tile"`.
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
- [ ] ~~**R11.2.17 README factual correction**~~ — retired: README is never
  edited (standing directive 2026-09-03). The `create_snn_mlp` row's true
  fix is R11.1.9 (a real-Spike SNN factory coordinate); the README row
  simply stays as history.
- [x] **R11.2.18 `test_scaling_invariants` xpass** ✅ **LANDED 2026-09-03.**
  `deep_network_accuracy[100]` passes deterministically (fixed seeds 42/43);
  the GATE-0 xfail reason (poor equilibrium-method accuracy) is obsolete —
  marker removed, test now asserts `acc > 0.3` live.
- [x] **R11.2.20 Timebox the pass** (E-2 analog) ✅ **CLOSED 2026-09-03.**
  R11.2 consumed its sessions: ruff baseline (R11.2.1), facade merge +
  twin resolution (R11.1.8/R11.2.3), xpass (R11.2.18), tile fold (R11.2.12),
  seed-required (R11.2.8), params-moved locks (R11.2.10) landed; pyright
  baseline explicitly deprioritized by user directive (as-touch only);
  imp-27 resolved-by-contract. No finding class stretched past its box —
  the pepita/tp/pc non-learning paths are *findings*, recorded, not stretched
  work. Remaining R11.2 items (R11.2.9/13/14/16) are explicitly scoped,
  pull-based items, not timebox residue.

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

- [x] **R11.3.1 PR-9 — Campaign commissioning** (Tangible Checkpoint 3) ✅
  **LANDED 2026-09-03.** The smoke campaign was already commissioned once
  (2026-08-31, R5.1a) but its artifact predated `records/episodes.json` and
  had no DECISIONS note — re-commissioned at HEAD
  (`scripts/commission_smoke_campaign.py --fresh`, ~12 s): campaign
  `smoke_r51a_s0`, seed 0, SIGKILL at `episodes >= 1` (checkpoint observed,
  db_iteration_at_kill=0), CLI `--resume` skipped the interrupted slot
  (skip-not-duplicate asserted by the script), 12 episodes to iteration 6.
  Artifact: `autoscientist_campaigns/smoke_cpu/records/{episodes.json,
  manifest.json, report.md, checkpoint_s0.yaml}` + full pre-kill/resume
  event trails under `seed_0/records/` (pre-kill trail now unbuffered —
  `PYTHONUNBUFFERED=1` in the script's subprocesses; the old run's was
  lost to SIGKILL stdout buffering). `load_campaign_records` consumes the
  12 records. One-paragraph commissioning note in `DECISIONS.md` (E-11).
  **Scope:** lifecycle commissioning only — 0/12 replicated (single seed),
  no claim rides on the records.
- [x] **R11.3.2 PR-2 — θ-invariance audit harness.** ✅ **LANDED
  2026-09-03.** `core/theta_audit.py`: `theta_audit(system_or_mapping,
  label=…, seed=…)` context manager — snapshot θ on entry, exact-diff on
  exit, `ThetaAuditReport.invariant` + `assert_invariant()` raising with
  the moved-parameter list. Accepts any composed System
  (``system.geometry.params``) or a plain name->tensor mapping; SHA-256
  covers name+device+dtype+bytes. Root-exported. Locked in
  `tests/unit/core/test_theta_audit.py`: control arm (training moves θ,
  audit flags it — not vacuous), free arm (inference episode reports
  clean), plain-mapping arm. D5 remains the Z3-machinery demonstration;
  the harness makes the same instrument a library feature for
  Algorithm-Migration / continual-learning runs.
- [ ] **R11.3.2b PR-3a — Software resource instrumentation.** `ResourceUsage`
  + `core/profiling.py` wired into suite runners emitting proxy
  FLOPs/memory/latency; feeds Z3 energy metrics (proxy tier), L2
  effective-FLOPs, the 𝒞 vector.
- [x] **R11.3.3 PR-5 — Calibrated stability guard.** ✅ **LANDED
  2026-09-03.** `computronium/stability/calibration.py`:
  `calibrate_demo_harvest` — known-good = the 9 demo-suite coordinates in
  campaign-builder syntax (D1/D2 credit arms, D6 substrate arms, D7 spike,
  D3-family M-axis arms, quickstart instantaneous; 784→(32,)→10, batch 64,
  4 episodes each, fork-rng'd builds), known-bad = divergence-labeled
  Ginibre runs (norm > 1e3× over 200 unrolled steps; marginal non-diverging
  runs enter neither set); ROC per statistic kind, deployed-τ operating
  point, overhead ratios + calibrated probe intervals, proxy-vs-full-Jacobian
  disagreement (tiny-dim study + demo-dims cost datapoint). **Bars met:**
  windowed-growth max-margin τ = 1.02895 — within 0.005% of deployed 1.029 —
  at 0% false-kill / 100% kill (<5% / >95% pre-registered); `fast_proxy`
  infeasible (substrate noise inflates good-arm gain to ≈1454 vs diverged
  range 0.93–1.51 — calibration-only, matching the family-sweep deferral,
  now measured on real coordinates); <10% overhead met via probe interval
  (34/102 episodes; `evaluate_episode`'s per-episode probe stays
  record-only telemetry). Artifact committed
  `docs/figures/registered/stability_guard_pr5.json` (script `--family
  pr5`, ~26 s; legacy `--family ginibre` kept); E-11 note in DECISIONS.md;
  RESULTS.md third guarantee row; live lock `TestPR5DemoHarvest`
  (6 assertions incl. the fast_proxy rejection and noise-inflation
  mechanism; file 18 passed in 4.6 s). Deployment consumer — wiring the
  interval into the unattended AutoScientist loop — pulls with it.
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
- [x] **R11.4.2 PR-6 fairness contract draft.** ✅ **LANDED 2026-09-03**
  (`docs/FAIRNESS_CONTRACT.md`, v0.1). Sections F-1..F-6: GPU-hour (not
  epoch) budgets with per-arm spend logs, one stopping rule per experiment
  applied identically (walled-at-budget reporting instead of silent
  extension/drop), seed floors per experiment class (toy ≥3, flagship/campaign
  ≥5, order-stratified), splits versioned + disjointness asserted, per-arm
  capacity/cost disclosure with the performance-gated (not parameter-matched)
  comparator rule carried from RESEARCH3, unfiltered multi-comparison
  reporting, pipeline parity (train mirrors export). Consumers table maps
  the four downstream items (Z3 flagship, benchmark paper, M-axis frontier,
  task-family generalization); consumers pin concrete values at pilot
  promotion, the contract fixes only the rules. Zero compute, as planned.
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

## 👁️ Watch (live items only)

- **axis_probe `[2-0]` flake** — no recurrence since 2026-08-31.
- **CUDA tolerance boundaries** shift xfail edges — CPU/GPU tests kept separate.
- **R11 sweep regime note (2026-09-03):** the repo-wide ruff autofix shifted
  import/init order, moving D2's gradient arm one batch (0.8739) and D7's
  record data. Tests still pass their asserts; manifest re-rendered. If a
  figure lock fires again after a sweep, check test asserts first, then
  re-render — only treat as a defect if the same run disagrees with itself.
- **D8 record determinism (2026-09-03):** the run record changes when the
  demo's *data path* changes (materialization order, device). The seed must
  precede the loader draw, not just the trainer. If the lock fires on D8,
  check `torch.manual_seed(42)` still precedes `_materialize` and DEVICE is
  still `"cpu"` before suspecting drift elsewhere.
- **Conv = GPU pointer (measured 2026-09-03):** conv-family geometries are
  the demo suite's first FLOP-bound regime (15× CUDA speedup at demo scale);
  CUDA records are nondeterministic run-to-run. Registered-scale conv
  studies go GPU-first; committed demo records stay CPU.
- **RESULTS.md front-table numbers churn** (user challenge 2026-09-03): the
  embedded approximations go stale on every re-pin; candidate improvement is
  numbers-free front table (swap + pointer + directional claim only, values
  live in the checksummed manifest). Not executed this sprint — flagged for
  the next editor / user decision.
- **`benchmark_results/` stays untracked** (standing directive).
- **Fidelity probes RNG-order sensitivity — RESOLVED (R11.2.22,
  2026-09-03):** `check_coordinate_fidelity` is seeded + fork-rng'd;
  verdicts are deterministic and the probe leaves the caller's RNG stream
  untouched. Historical fidelity manifests stay ambient-RNG artifacts.
- **Stale eager-default metric lookups (R11.1.3, 2026-09-03):**
  `d.get("free_accuracy", d["accuracy"])` evaluates the default eagerly —
  any pipeline metrics consumer written before imp-20/46 dropped the
  legacy `accuracy` key crashes on first call. The known instance is
  fixed; the safe idiom is nested `get` (`trainer.py`, `distributed_trainer.py`
  already do this). If a new KeyError 'accuracy' surfaces, check the call
  site's metric-contract vintage before suspecting the pipeline.
- **Registry-era removals (2026-09-03):** if a stranger needs
  transfer-weight loading or proposer objective ranking, they were deleted
  with the zoo — re-home them onto native factories, not the registry.
  `hyperopt/comparison.py` still carries PEP 758 `except A, B:` syntax
  (Python 3.14-only — fine here, breaks on older interpreters).
- **`equitile` is a deprecated identifier** (user directive 2026-09-03):
  family registrations, CLI maps, tolerances, and metamodel branches now key
  on `"tile"`. Residual `equitile` mentions are cosmetic (test *names* in
  `test_equitile_domains.py`, model *name* `rl_equitile` in
  `configs/rl_cartpole.yaml`, benchmark variable names, historical
  docstrings) — rename on next touch of those surfaces, never as a sweep.
  If `comp audit` ever flags historical `family="equitile"` rows in HPO/KB
  stores, map at read time rather than migrating DBs.

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

- **Registry is gone — never re-add it** (user directive 2026-09-03). The
  ontology is the composition surface; models resolve through native
  factories and `compose_*`. `KernelRegistry` (acceleration/) is unrelated
  and stays. Demo recipe for a new row is unchanged but the last step is
  `comp gallery --run` (re-renders `docs/figures/manifest.json`) before the
  gallery lock passes.
- **Attention/lattice geometry internals:** `_AttentionBlock` is a proper
  module subclass (typed params — no ModuleDict duck-typing);
  `AttentionGeometry._typed_blocks` is the single cast point off
  `nn.ModuleList`. Pyright carries a small strict backlog in the new
  geometry code (as-touch per the standing lint/type directive).
- **`tests/property/_support.py` registry helpers deleted**; L6 totality
  lock is adapter-based (`ModelAdapter(model).to_system()`).
- TODO10.md header marked CLOSED → superseded (2026-09-03).
- **README is never edited** (user directive 2026-09-03). No sunset condition.
- **Ontology package layout convention (R11.1.8, 2026-09-03):** implementations
  live in `_`-prefixed internal modules (`dynamics/_dynamics.py`,
  `substrate/_substrate.py`); package `__init__.py` is docstrings + re-exports
  only (ruff non-empty-init-module enforces this). New ontology primitives
  follow the same shape. Ruff's per-file noqa markers carried over verbatim
  during the moves — they self-flag on touch.
- **PlasticityConfig single source:** `computronium.state.transitions` owns
  it; `core/joint/transition.py` re-exports. Never redefine — import.
- **Geometry dispatch single source (R11.1.2a, 2026-09-03):**
  `computronium.ontology.geometry_from_config` is the one
  topology_type→implementation dispatcher; `factory.py`, `spec.py`,
  `joint.py`, and the profiler all consume it. Never re-inline a dispatch —
  add a branch to the dispatcher. New `GeometryConfig` tuple fields must be
  added to `_geometry_spec_parts`'s JSON tuple-restore list.
- **Tile × dynamics matrix (R11.1.3, 2026-09-03; resolved by R11.1.4 tile settle kernel 2026-09-03):** the seven tile strict xfails flipped xpass and were promoted to live locks in `tests/property/test_native_smoke.py` and `tests/integration/test_validation_all.py`. The single unlock was a target-responsive TileMesh block relaxation through the substrate operator. `native_tile_ep` re-added to REPRO_MODELS. `tile_native.py` module docstring updated. New lock `tests/property/test_tile_settle_kernel.py` (block≡per-edge equivalence, free≠nudged, all seven move params).
  tile pairings are strict xfails with mechanism-level reasons in
  `test_native_smoke.py` (canonical), `test_validation_all.py`, and
  `tile_native.py`'s module docstring. **User directive: the Tile
  geometry's potential is to be realized later** — the single unlock is a
  target-responsive TileMesh settle kernel (queued in R11.1.4); when it
  lands, the strict xfails flip xpass and force promotion (assertion body
  already measures params-moved). Until then: `comp repro` is 7/7 — do not
  re-add `native_tile_ep` to REPRO_MODELS until that kernel exists.
  `experiment/param_estimator.py` still lists `("tile_ep",
  create_native_tile_ep)` and will raise the same TypeError if exercised —
  same unlock, as-touch.
- **Diffusion target term (R11.1.7, 2026-09-03):** `DiffusionDynamics.compute_energy_from_state` now accepts optional `target` and `beta` parameters, implementing nudged-Langevin (EP-style) dynamics. The energy functional gains a term `beta * ||h - target_onehot||^2` when target is provided. Fidelity probe `test_diffusion_langevin_descent_target_responsive` passes. This enables contrastive credit assignments (ThermodynamicContrast, etc.) to work with DiffusionDynamics — previously the dynamics was target-blind. The PredictiveSettlingDynamics fallback path (for geometries without layered params) remains target-unwired; no current geometry uses it.
- **PR-5 instrument (R11.3.3, 2026-09-03):** `calibrate_demo_harvest`
  (stability/calibration.py) is the single calibration surface; the
  committed artifact is `docs/figures/registered/stability_guard_pr5.json`,
  rendered by `scripts/calibrate_stability_guard.py --family pr5` (legacy
  `--family ginibre` kept unchanged). Known-bad is a *manufactured*
  explosive family — when real diverged campaign runs accumulate (the
  failure manifesto), re-calibrate against those and retire the Ginibre
  proxy. `evaluate_episode` probes windowed growth every episode
  (record-only telemetry); deploying the kill switch means wiring
  `probe_interval_for_overhead`'s interval (102 episodes at demo scale)
  into the AutoScientist loop — the consumer that pulls it.
- **Demo-test record determinism (D8, 2026-09-03):** seed *before*
  materializing loader batches (the trainer's seed comes later); workers
  spawn per loader *iteration*, so materialize once and share. Where a demo
  needs fairness across arms, match parameter counts and assert the parity
  (user directive 2026-09-03: "compare accuracy / # of parameters").
- **Comments discipline (user directive 2026-09-03, repeated):** minimal
  comments; names + assert messages carry the why. Module docstrings on demo
  tests stay (they are the claim record); inline narration does not.
- **Sprint retro 2026-09-03 (binding for future sessions):**
  (a) Tests run **once at close** — mid-session gates are ruff + pyright on
  changed files only (seconds); behavioral questions get throwaway probe
  scripts; file moves/renames get grep + pyright, no test runs at all.
  (b) Any signature/config break (required fields, renamed identifiers):
  AST-walk the *entire* repo including `tests/` for call sites before
  finishing the item — eyeballing three test files missed
  `test_power_preregistration.py` this sprint.
  (c) Behavior inherited from deleted code is not automatically correct —
  when the recon shows the old implementation was itself the debt, land the
  fail-loud upgrade, don't preserve a silent fallback.
  (d) When a plan item is phrased as either/or ("fold into X or drop"),
  cross-check the repo's current naming conventions before picking a
  direction; plan phrasing can lag the codebase (the equitile→tile case).
  (e) **Diagnose nondeterminism at the data layer first** — a figure-lock
  drift that survives re-pins is an unseeded draw (D8's loader shuffle),
  not a manifest problem; three re-pin loops were spent before the seed
  order was checked. When the same sha keeps changing: seed the stream,
  run the demo twice, compare bytes — never re-pin between.
  (f) Fairness/capacity requirements arrive mid-item and reshape the demo
  (conv ≈1/10 params) — apply them by re-balancing the *weaker* arm, not by
  asserting superiority of the over-provisioned one.
- Work lean: one Register item per landing, each with a test that
  demonstrates it. Don't pull infrastructure "just in case".
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
