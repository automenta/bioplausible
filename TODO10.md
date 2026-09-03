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
> **State:** OPEN — R10.2 implemented 2026-09-02; **R10.1 closed 2026-09-02**
> (R10.1.5 landed: the depth-50 cliff now has its registered-scale figure in
> RESULTS.md's corroboration appendix, rendered from
> `benchmark_results/deep_credit_registered.json` by
> `scripts/render_registered_figures.py` with a provenance sidecar). All
> R10.2.0–R10.2.10 and R10.1.1–R10.1.4 are landed. The demo suite re-ran
> clean at HEAD — **7/7 demo tests (D1–D7), zero flakes** — and the gallery
> manifest is re-pinned at HEAD. All three CI additions
> (demo gate, figure lock, drift locks) are wired into `.github/workflows/ci.yml`.
>
> **D-axis pull landed 2026-09-02 (the round's last open capability).**
> `SpikeIntegrationDynamics.settle` now settles **layer-wise** for
> layer-structured geometries (`_settle_layered` in
> `ontology/_dynamics.py`, extracting the Linear stack through the existing
> `extract_layered_params`): each Linear transition integrates its constant
> drive (once per layer, through the substrate's forward operator) into LIF
> membranes for `max_steps` steps — spike at threshold 1.0, reset — and the
> settled membrane carries activity to the next layer. Dim-preserving
> geometries (recurrent attractors) keep the single-membrane `route` loop;
> Tile still raises the documented RuntimeError (parity xfail preserved).
> Before this pull the D-axis could not compose with any input ≠ output
> feedforward build (shape mismatch at settle step 1 — verified live). Two
> stacked defects beneath it were found and fixed en route: (a)
> `TemporalTraceCredit`'s rate-coded pseudo-gradient is **identically zero**
> when `a_plus == a_minus` (the surrogate correlates one (pre, post) pair in
> both temporal orders — the same matrix both ways); defaults are now
> `a_plus=1.0, a_minus=0.5` (potentiation-weighted, documented in the
> classmethod; the antisymmetry lock now pins symmetric weights
> explicitly). (b) The D-axis locks' "spike variance non-increasing" was an
> artifact of the old membrane-seeded trajectory, not a Lyapunov property —
> re-pinned to what the layered settle guarantees (membrane ≤ threshold by
> reset; per-step spike totals bounded by the neuron count; finite
> variance): B1 + `TestDAxisSpikeIntegration` updated, property suite
> green (1433 passed). D7 follows the standard recipe end-to-end: demo
> `test_demo_spike_settle.py` (one swapped D-axis argument — instant pass
> 0.87 / LIF settle 0.85, ≈1.8k counted spikes, membrane max 0.92 ≤ 1.0),
> gallery figure `_fig_spike_settle` + lock EXPECTED entry, RESULTS.md
> D-row, Demonstration Table row. **Third consecutive green gate run
> banked 2026-09-02: 8/8 (7 demos + gallery lock), 104 s, zero flakes**
> (99.4 s → 102 s → 104.2 s) — the R10.2 acceptance's flake criterion is
> met.
>
> **Evidence layer committed 2026-09-02** (`docs/figures/manifest.json`,
> `docs/figures/run_records/`, `docs/figures/registered/`,
> `docs/RESULTS.md`) — the figure lock's pinned data layer is tracked, so
> the lock is live on a fresh clone. **Deviation (user directive
> 2026-09-02): `benchmark_results/` stays untracked and gitignored** — the
> plan's earlier "next commit must add benchmark_results/" line is
> superseded; do not re-add it.
>
> **Substrate-fidelity pull landed 2026-09-02 (Register B, pairs with D6):**
> `MemristiveSubstrate` now realizes signed weights as **differential-pair
> conductances** — per-device range [0, 1] (the configured
> `weight_bounds`), int8 straight-through quantization (`_CONDUCTANCE_LEVELS`
> map; STE because a bare `torch.round` zeroes the autograd path and walls
> learning at chance — caught live by D6's mild arm), pair-difference
> forward. `quantize_weights` still returns in-range conductances (the
> bounded-conductance energy-invariant lock is untouched). The physics
> change moved D6's regime: the signed pair doubles the signal path, so the
> old wall at noise 3.0 became 0.56; re-swept (0.5→0.89, 1.5→0.78, 3.0→0.56,
> 4.0→0.42, 6.0→0.22, 8.0→0.12) and re-pinned at mild 1.5 / severe 8.0
> (0.91 digital / 0.78 mild / 0.12 severe). Gallery manifest re-pinned;
> RESULTS.md S-row updated. The neuromorphic half of the register item
> (real spike dropout or drop the cosmetic `sparsity` field) remains open.
>
> **D6 (substrate axis) pulled 2026-09-02.** The memristive factory landed
> (`create_memristive_mlp` in `core/presets.py`, deduped through
> `_instant_backprop_system`, exported from the root in `_LAZY`+`__all__`)
> with demo `test_demo_substrate_swap.py` (D6, one swapped substrate, three
> arms: digital 0.91 / mild IR-drop 0.84 / severe IR-drop 0.14 at noise 3.0
> — walled), gallery figure `_fig_substrate_swap` + lock EXPECTED entry, and
> `configs/presets/memristive_mnist.yaml`. *(Numbers superseded same day by
> the differential-pair fidelity pull — see the State block; D6 now pins
> 0.91 / 0.78 / 0.12 at noise 1.5 / 8.0.)*
>
> **D6 README/RESULTS updates 2026-09-02 — rescoped by user direction.**
> The only README change is the factual factory-row label
> (`create_memristive_mlp`: **Planned** → Framework implementation). The
> S-axis evidence links and a proposed third locked README block were
> pulled back: **the README carries no new code snippets or inline
> evidence links while the code is under active development** (user
> directive 2026-09-02; README stays the hand-maintained index). D6's
> evidence lives where it belongs — the Demonstration Table,
> `docs/RESULTS.md`'s live S-axis row, and the gallery figure. The
> R10.2.7a deviation note stands; RESULTS.md got the S-row update and the
> registered depth-cliff figure.
>
> **from-config CLI path fixed end-to-end 2026-09-02.** Three stacked
> defects: `task.train_loader` → `train_dataloader` (property rename);
> `SystemTrainer(system, train_loader, val_loader, trainer_config)` bound
> positionally into the wrong fields; and the inline system build used
> `geometry_config.geometry_type` (no such field) and a nonexistent
> `System.from_configs`. Replaced by the extracted
> `_build_system_from_flat_config` in `cli/commands/train.py`, which maps
> each preset section onto its ontology config via the classmethod named by
> the section's `type` tag (overlaying explicit keys; `**kwargs`-accepting
> plasticity classmethods absorb their extras) and delegates to the
> canonical `compose_system_from_configs` / `compose_joint_system_from_configs`.
> **All 16 presets build** (5-D → System, 6-D → JointSystem), and
> `comp run from-config` trains end-to-end (XOR smoke: 100% in 2 epochs).
> `configs/presets/tile_mnist.yaml` gained the `neurons_per_tile`/
> `tiles_per_layer` keys it had been missing since the path never worked.
>
> **CI wiring fixed 2026-09-02:** the preset-validation step now imports the
> real builder (`computronium.cli.commands.train._build_system_from_flat_config`,
> was a phantom `cli.run` symbol); the reproducibility step excludes
> `native_tile_ep` with a Register-C pointer (pre-existing 7/8); new steps:
> demo gate (`-k "demo or gallery_lock"`), drift locks
> (readme-snippet + root-exports). The pyright per-env `typeCheckingMode`
> removal is in place; note the repo-wide `ruff check`/pyright-basic
> baselines still carry thousands of pre-existing findings (pipeline.py,
> joint.py, plasticity/*, parity.py, …) — those gates are aspirational at
> HEAD and are deliberately not chased this round (Register C).
>
> What remains open: **nothing on the R10 critical path** — the acceptance
> rule's three consecutive green gate runs are banked (99.4 s, 102 s, and
> 104.2 s, the last at 8/8 with D7 in the gate). The demo table now covers
> all six ontology axes (D1–D7); further capability pulls come from
> Register B's pull conditions (geometries, substrate fidelity, tile
> matrix), each landing with its demo test per R10.3.6. Known-weak spots
> discovered en route, parked in the registers: `FeedforwardGeometry`
> silently ignores `GeometryConfig.init_scale` (`_build_layers` never
> applies it); `create_snn_mlp`'s README-table row advertises
> SpikeIntegration × TemporalTrace while the factory builds Instantaneous ×
> LocalGoodness for trainer compatibility; `create_spiking_snn_mlp` now
> runs end-to-end (it crashed at settle step 1 before the pull) but the
> Hebbian STDP surrogate stays at chance on MNIST — the pipeline-facing
> rate-coded surrogate has no error signal; genuine timing-asymmetric STDP
> lives in `core/local_learning/rules/spiking.py` and is not wired to the
> 5-D pipeline (research-track item, R10.3.5 refutation candidate).

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
| D6 | The substrate axis is physical | `test_demo_substrate_swap.py` | One swapped substrate through identical wiring: digital learns, mild IR-drop learns less, severe IR-drop walls at chance — differential-pair conductances (int8 STE) carrying the signed weights | — |
| D7 | The D-axis settles in time | `test_demo_spike_settle.py` | One swapped D-axis argument through identical wiring: the instantaneous pass and the layer-wise LIF settle both train (≈0.87 / 0.85); the trained LIF network fires visibly (every threshold crossing counted per settle step) and its membranes come back bounded by the spike threshold — the Lyapunov lock, live | — |

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

**Capability coverage after D7 (2026-09-02).** The Demonstration Table now
spans all six axes: S (D6), G (D1), D (D1 energy settling + D7 LIF
settling), M (D3), C (D2), U (D1/D5), plus the two categorical guarantees
(D4 profiler, D5 frozen θ). No next demo is queued: further pulls follow
Register B's pull conditions, each landing with its demo test (R10.3.6).
The D-axis's remaining depth — genuine timing-asymmetric STDP at pipeline
level, LazyStateDynamics at demo scale — is research-track/register
material, not a front-page claim until a visible regime exists.

---

## 🖼️ R10.1 — The Gallery (figures from live runs, not from files)

Figures are rendered **from the demo suite's own deterministic outputs at
HEAD** — fixed seeds, CPU, current code. The figure is what the test shows,
drawn. Nothing frozen, nothing to re-verify.

- [x] **R10.1.1** `computronium/visualization/gallery.py` (new package;
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
- [x] **R10.1.2** Demo runners emit run records deterministically: same seed →
  same record → same figure. The suite already pins seeds and CPU; emission is
  one JSON dump per test (assert-light, recorded even on assertion paths).
- [x] **R10.1.3** Output to `docs/figures/*.png` + `docs/figures/manifest.json`
  (per-figure: capability id, demo-test path, run-record provenance, scope
  label).
- [x] **R10.1.4** **Figure lock with teeth.** A lock test regenerates every
  figure and compares **data-layer** checksums (recorded metric values, not
  pixels). A mismatch means one of two things, both caught: the code changed
  what it demonstrates (review the diff, re-pin deliberately) or the demo
  became nondeterministic (a bug — fix it). Either way the gallery cannot
  silently drift from what the code actually does.
- [x] **R10.1.5** **Registered-scale figures are out of the gallery.** Where a
  preregistered study produced a figure worth keeping (e.g., the depth-50
  cliff — unreachable at demo scale by design), it lives in RESULTS.md's
  corroboration appendix, clearly labeled historical, provenance annotated
  where known, "provenance unknown" where not. The front page carries only
  live demonstrations. **Landed 2026-09-02:**
  `scripts/render_registered_figures.py` renders
  `docs/figures/registered/deep_credit_cliff.png` +
  `deep_credit_cliff.json` sidecar (source-artifact sha256, rendered-at
  commit, "provenance unknown" for the source runs); RESULTS.md's back
  section embeds it with the historical label and links it from the
  deep-credit row. The registered PNG is committed (the `docs/figures/*.png`
  gitignore covers only the live gallery's top-level pixels).
- [x] **Acceptance:** `comp gallery` on a clean checkout renders all five
  figures from a single suite run; the lock test is green; deleting any demo
  test removes its figure from the gallery (no orphaned claims).
  **Progress 2026-09-02:** `comp gallery --run` re-rendered all **six**
  figures from one suite run and the lock test is green; the orphan-skip behavior is structural
  (`render_gallery` drops records whose demo test no longer exists).
  **Clean-checkout rendering holds as of 2026-09-02**: manifest + run
  records + registered figure are committed (the lock's data layer is
  tracked; only the top-level PNGs regenerate). **Closed 2026-09-02:** the
  gallery now renders **seven** figures (D1–D7) from one suite run; the
  lock test is green at HEAD after the D7 re-pin.

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

- [x] **R10.2.0** **Demonstration design protocol.**
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
- [x] **R10.2.1** **API surface audit.** `computronium/__init__.py` exposes the
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
- [x] **R10.2.2** `test_demo_compose_6axis.py` — the flagship. Full six-axis
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
- [x] **R10.2.3** `test_demo_swap_credit.py` — one trainer, three credit rules.
  Same coordinate trained three times with the C-axis swapped (gradient /
  ThermodynamicContrast / RandomProjections) through identical `SystemTrainer`
  wiring. Shows: all three learn; the wiring code is byte-identical across
  arms except the constructor argument. *The comparison is one line.* Emits
  the D2 run record.
- [x] **R10.2.4** `test_demo_swap_plasticity.py` — the M-axis swap, seen. Null
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
- [x] **R10.2.5** `test_demo_memory_budget.py` — honest feasibility.
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
- [x] **R10.2.6** `test_demo_z3_frozen_theta.py` — the lifecycle guarantee,
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
- [x] **R10.2.7** **README refresh — the index stays complete.** README.md
  remains the cheat-sheet: a complete index to the entire scope of
  functionality and capability — the ontology-axis table, the 13-factory
  table, the CLI reference, the capability matrix, the substrate table, the
  verification-status markers all stay as first-class content. **Do not
  restructure it into a narrative funnel; do not move tables below any
  fold.** Changes are surgical: (a) only a few designated code snippets
  (the opening composition block, one factory one-liner block) — minimal
  precisely because snippets are volatility-prone — and each is derived from
  a demo test and locked by R10.2.8; (b) the identity inversion happens in
  *labeling, not layout*: research framed as the program validating the
  library's abstractions; (c) factual corrections only: the stale
  `core/system_trainer.py` module path, the `JointSystemTrainer` fiction
  (the class does not exist — describe the real joint training surface:
  `compose_joint_system` + `SystemTrainer` duck-typing), and the
  `create_memristive_mlp` row (implement-as-pull or mark Planned);
   (d) where a gallery figure exists for a capability, link it as evidence
   beside that capability's table row — **superseded 2026-09-02 (user
   directive): no new snippets or evidence links in README while the code
   is under active development; evidence links live in `docs/RESULTS.md`
   and the gallery.**
- [x] **R10.2.8** **Drift lock (designated snippets only).** The few
  designated README code blocks are checked verbatim against their source
  demo tests (small extraction script under `scripts/` + lock test under
  `tests/`; each locked block declares its source test in a sidecar map).
  Tables and prose are the hand-maintained index — deliberately **not**
  locked. A locked snippet that no longer matches its test fails CI — no
  snippet/doc drift, by construction, with minimal lock surface.
- [x] **R10.2.9** **`docs/RESULTS.md` — capabilities first, history second.**
  Front section: per axis (S/G/D/M/C/U), what the abstraction is, the one-line
  swap, the live demonstration, the figure. Back section — clearly labeled
  *Historical corroboration*: the registered studies, their preregistered
  scope, their numbers, their provenance status (including "unknown" where
  archaeology fails), and what they do **not** mean. The front never cites the
  back.
- [x] **R10.2.10** **`comp gallery`** (`computronium/cli/gallery.py` with
  `main()`, registered as one entry in the `_SUBCOMMANDS` map of
  `computronium/cli/__main__.py` — the dispatcher's lazy-resolution pattern,
  so gallery imports stay out of the CLI's import graph): runs the demo suite
  (or reads its emitted run records from the most recent gate run), renders
  figures + manifest, exits nonzero on any missing/nondeterministic record.
- [x] **Acceptance:** a stranger copies the README's first block and runs it;
  runs `pytest tests/integration/ -k demo` and watches all five capabilities
  demonstrate themselves; `comp gallery` renders from those same runs;
  **three consecutive green gate runs with zero demo-test flakes** before the
  round closes; every claim on the front page is one they just watched happen.
  **Progress 2026-09-02:** full demo gate re-run green — **6/6 (D1–D6),
  zero flakes** — and `comp gallery` rendered from those same runs;
  **two of the three consecutive green runs are banked** (99.4 s and 102 s).
  **Closed 2026-09-02:** third green gate run banked — **8/8 (7 demos +
  gallery lock), 104.2 s, zero flakes** (99.4 s → 102 s → 104.2 s), with
  D7 in the gate.

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
  **Wired 2026-09-02:** ci.yml now runs the demo gate
  (`pytest tests/integration/ -k "demo or gallery_lock"`), the drift locks
  (`test_readme_snippet_lock.py` + `test_root_exports.py`), and the preset
  gate through the real builder; the positive control rides in
  `tests/property/`. Remaining caveat: the repo-wide `ruff check`/pyright
  baselines carry thousands of pre-existing findings outside the gates this
  round added — treat those two steps as aspirational until a dedicated
  hygiene pull (Register C).

## 🎯 Tangible-Result Checkpoints (what the investment returns)

R10 is an investment; these are the returns, each with a materialization
condition. A sprint that ships none of these is a sprint to question.

1. **Working proofs (R10, sprints 1–2):** `pytest -k demo` green — the
   demonstrated capabilities (now D1–D7, all six axes) in under two
   minutes, calibration recorded in docstrings. This is also the *feedstock*
   for checkpoint 3: the demo configs and their outcomes are PR-5's known-good/known-bad calibration
   harvest.
2. **Truthful front door (R10, sprint 3):** README index corrected (real
   joint-training surface, memristive row resolved), few locked snippets,
   `comp gallery` rendering live evidence. A stranger can verify every claim
   in one sitting.
3. **Commissioned campaign stack (PR-9, first research-track pull):** one
   full iterate → interrupt → checkpoint → resume cycle on
   `autoscientist_campaigns/` — built today but with **zero completed
   runs**; nothing consumes the campaign stack until this passes. This is
   the gateway to every unattended result.
4. **Calibrated stability guard (PR-5):** ROC-calibrated kill thresholds
   from checkpoint 1's harvest (<5% false-kill on known-good, >95% kill
   rate, <10% overhead) — unattended campaigns become safe to launch, and
   the failure manifesto starts accumulating as a dataset.
5. **The first research-shaped result (AutoScientist M-axis frontier):** a
   Pareto frontier over the resource vector 𝒞, one axis swept at a time,
   annotated with which M primitive owns each knee — the first figure that
   is a *finding*, not a demonstration.
6. **The discovery bet (Z3 flagship registered commission):** ≥95% on all
   three tasks at exact Δθ=0 within ≤20% of fine-tuning steps — accuracy-
   vs-steps curves, forgetting matrix, per-seed Δθ audit. If it falsifies:
   the L1 adaptation figure substitutes as the campaign seed and the
   boundary condition becomes the publication. Either way, tangible.

Sequencing rule: 1–2 are R10; 3–4 pull immediately after (the demo harvest
is their calibration input); 5–6 are RESEARCH3 CP-A's spine. No checkpoint
blocks on a later one.

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
| **Memristive factory** | ~~README advertises `create_memristive_mlp`~~ **Pulled 2026-09-02**: implemented via `core/presets.py` over `MemristiveSubstrate`, root-exported, demo `test_demo_substrate_swap.py` (D6), preset `memristive_mnist.yaml` | ✅ Done |
| **Geometries** | `ConvGeometry` / `GraphGeometry` / `AttentionGeometry` / 3D `SpatialLattice3D` — geometry-DEFERRED skips stay skips | Science runs on Feedforward/Recurrent/Tile at MLP scale today. Pull when a demo test or campaign manifest needs the geometry (e.g., a vision demo test wants Conv; a graph domain task wants GraphGeometry). |
| **Substrate fidelity (R3.7)** | Neuromorphic: real spike dropout or drop the cosmetic `sparsity` field. ~~Memristive: conductance-range semantics~~ **Memristive half pulled 2026-09-02**: differential-pair conductances (per-device [0,1] range, int8 straight-through quantization, pair-difference forward) in `ontology/_substrate.py`; D6 re-swept and re-pinned; bounded-conductance lock and substrate certifications untouched | Neuromorphic half: pull when a neuromorphic-axis claim or demo test needs the fidelity to be real |
| **Tile × dynamics matrix (R3.4)** | tile_ep/pc/gnn/snn device-dynamics incompatibilities; tile_fa/tp/hebbian — fix or document as permanent xfail with precise reasons | Pull on next touch of the tile family, or when a demo wants the full tile matrix. |
| **Adapter heuristics (R3.5)** | `_AdaptedSystem._infer_geometry` hardcoded (784→256,128→10) — recover heuristics from the deleted `adapter/` package | Pull when the strangler-fig adapter path is next touched. |
| **`_TaskTrainer` gaps (R3.6)** | Scheduler wiring, energy tracking, honor `tracker`/`safety_config` | Pull when hyperopt trials need them. |
| **Kernels (R4.1–R4.4)** | FA feedback projection through the Substrate operator API; `SubstrateSettleKernel` in `KernelRegistry`; MEP Triton kernels (Muon, Fisher whitening) → Substrate update operator; sparse transpose-mask handling, ternary `init_scale` (un-xfail ternary equivalence), per-step `inject_state_noise` | Pull when the acceleration/kernel path is next touched or a substrate-axis demo needs them. |
| **Nudge-unwired settle paths (imp-29)** | predictive_settling target clamp; diffusion target term | Pull when a campaign manifest needs those coordinates to be fully wired. |
| **Substrate facade merge (R2.2 residual)** | `ontology/_substrate.py` impl vs `ontology/substrate/` facade — facade is fine; consider merge + grep for other parallel legacy/new pairs (same pattern exists for `_dynamics.py` vs `dynamics/`) | Pull on next ontology-structure touch. |

### C. Code-quality backlog (imp-N + R-N items; pull on next touch of the file/system)

| Item | Pull condition |
|------|----------------|
| Root `PlasticityConfig` still resolves to `computronium.state`'s twin, which is a **different class** from `core.joint.transition.PlasticityConfig` (found in the R10.2.1 audit; pyright flags the resulting confusion in `core/system_trainer/joint.py`). Same parallel legacy/new pair as the `state/` vs `core/joint/` split itself | Next merge of `computronium/state/` with `computronium/core/joint/` (R2.2-residual pattern); the root-exports lock test pins what exists today |
| R10.2 demo flake watch (2026-09-02): the first `pytest -k demo` invocation after the tests landed reported 3 failures in 40 s that never reproduced across four subsequent full runs (75–79 s each). Suspected transient MNIST DataLoader worker crash (`num_workers=2`); no failure signature captured. If it recurs, capture the traceback and consider `create_task(..., num_workers=0)` inside the demo tests. **2026-09-02 re-run: 6/6 green, 99.4 s — still no recurrence.** D7 pins `num_workers=0` explicitly — the standing mitigation precedent for new demos | Any recurrence |
| Repo-wide ruff/pyright hygiene: `ruff check .` reports ~4.8k pre-existing findings (max-args=5, preview rules, S-rules on subprocess) and pyright-basic flags pipeline.py / core/system_trainer/joint.py / plasticity/{routing,fast_weights}.py / cli/parity.py / tests/property/test_axis_certifications.py — CI's ruff and pyright steps fail at HEAD independent of R10. Pull on a dedicated hygiene pass: either fix forward or scope the CI steps to the gates that are meant to hold (property/demo/lock suites) | Next dedicated hygiene pass |
| D1/D2 dominate the demo-suite wall clock (~70 s of ~80 s; MNIST quick-mode, 1 epoch each). If slower CI machines push the suite past the 2-minute checkpoint, cap the D1/D2 train loaders (e.g. a `_take(loader, 800)` wrapper) and re-pin the regime assertions (FA floor 0.25 was calibrated at the full epoch) | First slow-CI gate failure, or any D1/D2 regime change |
| Joint `to_spec`→`from_spec` round-trip broken — `from_spec` calls `GeometryConfig(**spec["geometry"])` but `to_spec` embeds `params`/`recurrent_weight` keys → TypeError; found in the 2026-09-02 pre-flight | Next touch of `core/system_trainer/joint.py` (or when a demo/figure needs joint-spec round-trips) |
| imp-4 — Pyright full `strict` on ontology (131 findings; torch `Unknown` tracking; annotation work in `_dynamics`/`geometry`/`update`) | Next annotation pass on those modules |
| imp-8 — `compute_energy` duplication across Energy/Spike/Instantaneous/Diffusion → extract `_energy_from_state(state, geometry)` | Next touch of any dynamics module |
| `FeedforwardGeometry._build_layers` ignores `GeometryConfig.init_scale` (nn.Linear defaults only; found in the D7 sweep — three init scales gave byte-identical results). Preset `_mlp_geometry`/`create_*_mlp` pass it through, so every factory's `init_scale` argument is currently decorative on feedforward builds | Next touch of geometry construction, or when a demo needs weight-scale control |
| imp-19 — `FrontierRecord.seed` legacy default 42 → required at next schema break | Schema break |
| imp-23 — `substrate_coupled` plasticity engagement-verified only; probe fixed-dim `step` assumptions | Next touch of that plasticity |
| imp-26 — params-moved learning locks for the remaining README-table factories (FA lock exists) | Next touch of each factory |
| imp-27 — rename rebuilder-style `settle` implementations whose names mislead | Next touch of each implementation |
| imp-30 — deployments' `family="tile"` registrations CLI-orphaned → fold into `family="equitile"` or drop | Next touch of deployment registry |
| imp-36 — campaign stability axis non-discriminative → cheap per-episode proxy | When a manifest needs stability contrast |
| imp-37 — latency objective is wall-clock noise → repeated-timing methodology or deterministic proxy | Before any task-scale latency claim |
| imp-41 — `demo/tests/` 28 stale failures → rewrite or delete | Next demo-test touch (or R11, where the demo gets rebuilt) |
| R3.8 — `natural_language_query` TF-IDF weighting; derive `V_nudged = free energy + β·loss` to strengthen the PC Lyapunov xfail | Next touch of the knowledge base / PC verification |
| README `create_snn_mlp` row advertises SpikeIntegration × TemporalTrace × Euclidean, but the factory builds Instantaneous × LocalGoodness for trainer compatibility (pre-dates R10; the docstring is honest about it). Either land a true-Spike SNN factory coordinate once STDP carries a real error signal, or correct the row | Next README factual-correction pass (R10.2.7 rules) |
| `create_spiking_snn_mlp` runs end-to-end post-D7-pull (it crashed at settle step 1 before) but plateaus at chance on MNIST: the pipeline-facing STDP surrogate is a pure Hebbian correlation (no error signal); timing-asymmetric STDP lives in `core/local_learning/rules/spiking.py`, unwired to the 5-D pipeline. R10.3.5 refutation candidate — a visible refutation demo is welcome; a learning claim needs the trace-based rule in the pipeline first | When the SNN family is next touched, or a research paragraph needs it |
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
  **resolved 2026-09-02**: the R10.2.1 audit found divergence (15 `_LAZY`
  names missing from `__all__`, one dead `OntologyConfig` entry) and added
  the standing lock `tests/unit/core/test_root_exports.py` (lockstep +
  every lazy entry resolves).
- **Demo-suite evidence layer committed 2026-09-02** (resolved):
  `docs/figures/manifest.json`, `docs/figures/run_records/`, and
  `docs/figures/registered/` are tracked — the figure lock's data layer is
  live on a fresh clone. **Deviation (user directive 2026-09-02):
  `benchmark_results/` stays untracked and gitignored — never add it**;
  the plan's earlier "must add benchmark_results/" line is superseded
  (registered artifacts were briefly committed and pulled back same day).
- The z3 demo pins `meta_train_epochs=4` for `MetaRecipe()` defaults
  (fresh-ψ floor ≈ 0.68, restored beats floor+0.1 at 1.0). If `MetaRecipe`
  defaults or the task generators change, the gate items will move —
  re-run the calibration sweep (3/4/5 epochs) before re-pinning.
- **D7 spike watch:** D7 asserts `total_spikes > 100` (observed ≈ 1.8k) and
  `membrane_max <= 1.0` (structural: the reset rule guarantees ≤ threshold)
  at seeds build-0/trainer-42. If `MetaRecipe`-style default drift ever
  moves the trained weights enough to silence the hidden layer, the spike
  floor moves — re-run the D7 sweep (the probe in the demo prints the
  staircase) before re-pinning. Sub-threshold-at-init is expected (spikes
  emerge from training at the pinned lr).
- **D6 wall watch:** D6 adds ~24 s to the demo suite. Full-suite re-run at
  HEAD 2026-09-02 (now including D7): **104 s for all seven demos** — under
  the 2-minute checkpoint with margin. If a slow CI machine pushes past it,
  `BATCH_CAP` in `test_demo_substrate_swap.py` and `BATCH_CAP` in
  `test_demo_spike_settle.py` are the dials (floors were calibrated at 1000
  and 300 batches respectively), and the D1/D2 loader caps are the other dial. Under the
  differential-pair semantics (2026-09-02) the staircase moved: 0.5→0.89,
  1.5→0.78, 3.0→0.56, 4.0→0.42, 6.0→0.22, 8.0→0.12 — the severe arm sits at
  noise 8.0 with ceiling 0.4 (the 4.0-class value 0.42 grazes that ceiling
  from above; don't move the severe dial below 6.0-class without re-sweeping).
- **CI preset validation doubly broken — resolved 2026-09-02:** the workflow
  now imports `_build_system_from_flat_config` from
  `computronium.cli.commands.train` (extracted from `run_from_yaml`, which
  also gained `train_dataloader` + correct `SystemTrainer` keyword wiring);
  all 16 presets build and a synthetic `from-config` run trains end-to-end.
- `comp repro` is 7/8 (`native_tile_ep` fails) — pre-existing; CI's
  reproducibility step now excludes it explicitly with a Register-C pointer.
  Fix or document `native_tile_ep` on next touch of the tile-native family.
- Repo-wide `ruff check` (~4.8k findings) and pyright-basic fail at HEAD on
  pre-existing modules (pipeline.py, core/system_trainer/joint.py,
  plasticity/routing.py, plasticity/fast_weights.py, cli/parity.py,
  tests/property/test_axis_certifications.py) — CI's ruff/pyright steps are
  aspirational until a dedicated hygiene pull (Register C). R10 added no new
  findings beyond the noise floor.

## 🗺️ Implementation Map (what landed 2026-09-02; how to extend)

**Landed files.**

| Piece | Path |
|-------|------|
| Root export mechanism fixes + `create_task`/`create_tile_mlp`/`create_spiking_snn_mlp` exports, `NullPlasticity` re-point to `computronium.core.plasticity` | `computronium/__init__.py` |
| `_LAZY`↔`__all__` consistency lock | `tests/unit/core/test_root_exports.py` |
| Demo tests D1–D5 (regimes pinned in module docstrings; each emits its run record **before** asserting) | `tests/integration/test_demo_compose_6axis.py`, `test_demo_swap_credit.py`, `test_demo_swap_plasticity.py`, `test_demo_memory_budget.py`, `test_demo_z3_frozen_theta.py` |
| Demo test D6 (substrate swap; factory-level S-axis swap, regime + staircase sweep pinned in docstring) | `tests/integration/test_demo_substrate_swap.py` |
| Demo test D7 (D-axis swap: instant vs layer-wise LIF settle, one swapped dynamics argument; regime + spike observables pinned in docstring) | `tests/integration/test_demo_spike_settle.py` |
| Layer-wise LIF settle (`_settle_layered`: per-Linear constant drive through the substrate operator, LIF membrane + threshold/reset, membrane carry; dim-preserving `route` loop kept for recurrent geometries) | `computronium/ontology/_dynamics.py` |
| TemporalTraceCredit degeneracy fix (rate-coded surrogate is identically zero at `a_plus == a_minus`; defaults now `a_plus=1.0, a_minus=0.5`, documented at the classmethod and class) | `computronium/ontology/credit.py` |
| D-axis lock re-pin (B1 + `TestDAxisSpikeIntegration`: membrane ≤ threshold, per-step spike totals bounded by neuron count, finite variance; STDP antisymmetry lock pins symmetric weights explicitly) | `tests/property/test_ontology_locks.py`, `tests/property/test_axis_certifications.py` |
| D7 figure factory (`_fig_spike_settle`, entry in `_FACTORIES`) | `computronium/visualization/gallery.py` |
| Memristive factory (`create_memristive_mlp`; `_instant_backprop_system` dedupes the Digital/Memristive backprop-MLP body) | `computronium/core/presets.py` |
| Memristive preset YAML | `configs/presets/memristive_mnist.yaml` |
| D6 figure factory (`_fig_substrate_swap`, entry in `_FACTORIES`) | `computronium/visualization/gallery.py` |
| Run-record emitter fixture (deterministic payload; git commit + config sha256 provenance; no timestamps) | `tests/integration/conftest.py` → `emit_run_record(capability_id, capability_name, data)` |
| Figure factories + `FigureMeta` + `render_gallery` (one pure function per figure; `_FACTORIES` map keyed by capability name) | `computronium/visualization/gallery.py`, `_style.py`, `__init__.py` |
| Figure lock | `tests/integration/test_gallery_lock.py` |
| README drift lock (extraction script + sidecar map + test; blocks marked `<!-- lock: <id> -->`) | `scripts/readme_snippet_lock.py`, `scripts/readme_snippets.json`, `tests/unit/core/test_readme_snippet_lock.py` |
| `comp gallery` (lazy `_SUBCOMMANDS` entry; `--run` executes the demo suite first; exits nonzero on missing/drifted records) | `computronium/cli/gallery.py`, registered in `computronium/cli/__main__.py` |
| Capabilities-first results doc with labeled historical corroboration | `docs/RESULTS.md` |
| Registered-scale figure factory (R10.1.5; depth-cliff figure + provenance sidecar under `docs/figures/registered/`) | `scripts/render_registered_figures.py` |
| from-config rebuild: `_build_system_from_flat_config` (tag→classmethod section mapping; delegates to `compose_system_from_configs` / `compose_joint_system_from_configs`), fixed `train_dataloader` wiring, keyword `SystemTrainer` call | `computronium/cli/commands/train.py` |
| CI gate additions: demo gate, drift locks, real preset builder import, `native_tile_ep` repro exclusion | `.github/workflows/ci.yml` |
| Substrate-fidelity pull: differential-pair conductances (`_quantize_conductance` int8 STE, `conductance_pair`, pair-difference forward operator) | `computronium/ontology/_substrate.py` |
| Evidence layer committed (manifest + run records + registered figure + RESULTS.md); `benchmark_results/` gitignored by user directive | `.gitignore`, `docs/figures/`, `docs/RESULTS.md` |
| Factory-row label correction (`create_memristive_mlp` **Planned** → Framework implementation); no snippet/link additions per user rule | `README.md` |

**Pinned regimes (assertion floors chosen from live calibration, recorded
in each docstring).** D1: MNIST quick, 1 epoch, hidden (32,), EM
max_steps=5 β=0.5 → acc ≈ 0.89, J1 metrics equal 1e-7 + θ bitwise, L6
round-trip on configs. D2: EM max_steps=3, Euclidean step 0.1 momentum 0 →
0.96 / 0.92 / 0.38, floor 0.25. D3: forgetting-trial A40/B40, 10 seeds,
mastery precondition (null ≥ 0.45/seed, routing slower in every seed), gap
≥ 0.05 (observed 0.107), routing ≥ null in ≥ 8/10 (observed 9/10). D4:
depths 4/16, budget 0.015 MiB → gradient/RP never commissionable, thermo
0 bytes, feasible walk probe ≥ 0.225 (observed 0.305 at 50 episodes). D5:
meta 4 epochs, adapt 4, probe batches 4 → θ sha256 bitwise-equal, restored
== stage A exactly, gate fully passed (floor 0.68). D6 (re-pinned
2026-09-02 under differential-pair conductance semantics): MNIST quick,
1000 batches, hidden (32,), step 0.05 → 0.91 digital / 0.78 mild (noise
1.5, floor 0.5, < digital) / 0.12 severe (noise 8.0, ceiling 0.4). D7
(pinned 2026-09-02, live sweep): MNIST quick, 300 batches, hidden (32,),
step 0.05, LIF max_steps=10 → 0.87 instant / 0.85 spike (floors 0.5; the
gap is not the claim), probe settle: 20 (layer, step) counts, total spikes
≈ 1.8k (floor 100), membrane max ≈ 0.92 ≤ threshold 1.0.

**Adding a demo row (e.g. D6, substrate):** (1) demo test
`tests/integration/test_demo_<name>.py` importing only from the package
root (+ its experiment module if benchmark-surface), emitting
`emit_run_record("D6", "<name>", data)`; (2) figure factory in
`computronium/visualization/gallery.py` + entry in `_FACTORIES`; (3) entry
in `EXPECTED` of `tests/integration/test_gallery_lock.py`; (4) row in the
Demonstration Table and `docs/RESULTS.md`; (5) optional locked README block
via `scripts/readme_snippets.json` + `<!-- lock: -->` marker.

**Deviation note (R10.2.7a).** The designated "factory one-liner block" was
realized as the D2 swap block: no demo test exercises the `create_*_mlp`
preset factories yet, so locking a preset one-liner had no source test.
When a demo (or imp-26's params-moved locks) touches a preset factory, add
the preset block to the sidecar map. **Superseded 2026-09-02 (user
directive): while the code is under active development, README carries no
new source snippets, locked or otherwise — the preset-factory swap is
demonstrated by `test_demo_substrate_swap.py` and documented in
RESULTS.md; README stays a two-locked-block index until the API stabilizes.**

## Termination criterion



R10 closes when a stranger can, in one sitting: copy the README's first code
block, run it, and watch a system **they composed** train; then run the demo
suite and watch the capabilities demonstrate themselves — compose, swap
credit, swap plasticity, hit the memory wall, freeze θ and see it hold
bitwise, swap substrate physics, watch the LIF settle spike and stay
bounded; then look at the gallery and recognize the same demonstrations drawn.
Nothing they read asks them to trust a file they didn't just regenerate;
nothing is claimed that they didn't just watch happen. **Read, run, change one
thing, see it matter** — with the proof re-earned on every commit. If that
session produces "oh, that's interesting, and I know exactly which line to
change" — the round did its job.
