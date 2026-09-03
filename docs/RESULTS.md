# RESULTS — Capabilities First, History Second

> **TODO10 R10.2.9.** The front section states what each ontology axis *is*,
> the one-line swap that exercises it, and the live demonstration that
> re-shows it at HEAD. The back section is **historical corroboration**:
> registered studies kept for context, never load-bearing. The front never
> cites the back. A claim stands only while its demo test re-demonstrates it
> — `pytest tests/integration/ -k demo`.

## Front: the axes, live

| Axis | What the abstraction is | The one-line swap | Live demonstration | Figure |
|------|------------------------|-------------------|--------------------|--------|
| **G** (Geometry) | Topology and routing of computational units — `RecurrentGeometry` here: hidden state is recurrently connected | `geometry=RecurrentGeometry(GeometryConfig.recurrent(...))` in `compose_joint_system` | D1 — a fully composed six-axis system trains on MNIST to ≈ 0.84 in one epoch over the capped train stream (`test_demo_compose_6axis.py`); config round-trips (L6); the 5-D build trained identically gives bitwise-equal θ (J1) | [d1](figures/d1_compose_6axis.png) |
| **D** (StateDynamics) | Forward evolution and settling — `EnergyMinimizationDynamics` relaxes free/nudged phases; `SpikeIntegrationDynamics` settles layer-wise LIF membranes (spike at threshold, reset), the settled membrane carrying activity between layers | `dynamics=InstantaneousDynamics()` ↔ `SpikeIntegrationDynamics(StateDynamicsConfig.spike_integration(max_steps=10))` | D1 — energy-based settling trains end-to-end through the same trainer; D7 — one swapped D-axis argument, identical wiring: both arms learn (≈0.87 / 0.85), the trained LIF network fires visibly (≈1.8k counted spikes per settle) and its membranes come back ≤ threshold 1.0 (`test_demo_spike_settle.py`) | [d1](figures/d1_compose_6axis.png), [d7](figures/d7_spike_settle.png) |
| **C** (CreditAssignment) | Error routing / pseudo-gradient computation; every rule is one constructor argument on the same coordinate | `credit=BackpropCredit()` ↔ `ThermodynamicContrast()` ↔ `RandomProjectionsCredit()` — everything else byte-identical | D2 — all three rules learn the same task through one trainer (`test_demo_swap_credit.py`): ≈ 0.87 / 0.86 / 0.62 vs 0.10 chance (600-batch cap, re-pinned 2026-09-02) | [d2](figures/d2_swap_credit.png) |
| **M** (Plasticity) | Mechanism making the computational rule itself a dynamical variable; `NullPlasticity` is the zero-extension slice | `plasticity=NullPlasticity()` ↔ `RoutingPlasticity()` on the segmented A→B stream | D3 — across a task switch, routing visibly retains segment-A competence that null forgets (10-seed calibration, mastery precondition asserted first) (`test_demo_swap_plasticity.py`) | [d3](figures/d3_swap_plasticity.png) |
| **S** (Substrate) | Physical state space — precision, noise, sparsity constraints; `MemristiveSubstrate` realizes signed weights as differential-pair conductances (per-device range [0, 1], int8 straight-through quantization) with IR-drop state noise | `create_backprop_mlp(...)` ↔ `create_memristive_mlp(..., noise_level=8.0)` (D1–D5 all run on the digital substrate through the same `Substrate` API — no special-casing anywhere in the trainer) | D6 — one swapped substrate argument, identical wiring: digital learns (≈0.91), mild IR-drop learns less (≈0.78), severe IR-drop walls at chance (≈0.12); monotone staircase re-pinned 2026-09-02 under the differential-pair semantics (`test_demo_substrate_swap.py`) | [d6](figures/d6_substrate_swap.png) |
| **U** (ParameterUpdate) | Slow, persistent parameter consolidation Δθ | `update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=...))` | D1 — the only path θ ever changes; D5 proves θ does *not* change when frozen (bitwise) | [d5](figures/d5_z3_frozen_theta.png) |

### The two categorical guarantees

| Guarantee | What the runner watches | Demonstration | Figure |
|-----------|------------------------|---------------|--------|
| **The memory profiler is honest** | Under a 0.015 MiB budget the BPTT-profiled arm is walled at every depth *before training*; the O(1)-memory arm runs and learns | `test_demo_memory_budget.py` — verdicts are profile arithmetic, walled cells produce no walk | [d4](figures/d4_memory_budget.png) |
| **Frozen θ is bitwise** | θ's SHA-256 identical across freeze → adapt ψ on A → switch to B → restore; the restored ψ-system reproduces stage-A accuracy *exactly* | `test_demo_z3_frozen_theta.py` (J2, demonstrated, not just locked) | [d5](figures/d5_z3_frozen_theta.png) |

## Back: historical corroboration

> **Everything below is history.** These are registered research-track runs
> (preregistered scope, fixed seeds, stored artifacts under
> `benchmark_results/`). They show the same effects as the front section at
> larger scale *as of the date they ran* — they are not library claims, and
> no front-page statement leans on them. Provenance status is stated per
> artifact; "provenance unknown" means the file records no git commit.

| Study | Artifact | Preregistered scope | Headline numbers | Provenance |
|-------|----------|--------------------|------------------|------------|
| Retention (forgetting) trial | `benchmark_results/forgetting_registered.json` | 16 seeds, A40/B40 segmented stream, null vs fast_weights vs routing, planted lr=0 control | Routing retains segment A where null collapses; d = −1.90 on retained accuracy at the registered scale | git commit **unknown** |
| Memory-budget trial | `benchmark_results/memory_budget_registered.json` | 5 seeds × depths 4/16/50 × budgets 0.015/0.25/0.45 MiB; walled arms never walked | Feasibility grid reproduces the O(1)-vs-O(depth) wall; walled-regime competence is shallow-only | git commit **unknown** |
| Walled-regime boundary map | `benchmark_results/boundary_map_pilot.json` | Depths 4–50, independent seeds | Competence boundary between depth 4 (probe 0.396, d = +4.80) and depth 6; depth-4 replicates the registered 0.406 | git commit **unknown** |
| Deep-credit trial | `benchmark_results/deep_credit_registered.json` | 16 seeds × depths 4/16/50 | The depth-50 cliff: BPTT-profiled arms lose competence below the wall; O(1)-memory arms degrade gracefully ([figure](figures/registered/deep_credit_cliff.png)) | git commit **unknown** |
| Z3 fixed-weights manifest | `benchmark_results/z3_fixed_weights/manifest.json` | Meta-train 50 epochs, eval 20/task, 3 seeds, CUDA | ψ-mediated task switching at exact Δθ = 0 with θ-hash audit | `config_sha256` + `git_commit` **recorded** |

**Registered-scale figure (historical):**

![The depth cliff — mean probe accuracy vs network depth per credit arm, 16 seeds](figures/registered/deep_credit_cliff.png)

*Drawn from `benchmark_results/deep_credit_registered.json` (16 seeds × depths
4/16/50): the BPTT-profiled arms hold competence at depth 4 and collapse past
the boundary; the O(1)-memory arm and the lr=0 control sit near chance
throughout. Rendered at HEAD by `scripts/render_registered_figures.py`
(provenance sidecar: `figures/registered/deep_credit_cliff.json`); the source
runs record no git commit — provenance unknown.*

**What the back section does not mean:** none of these numbers is a claim
about the library's current behavior. The front section's demo tests are the
only live evidence; this section exists because the registered studies'
*designs* (preregistration, controls, seed counts) remain the right
instruments at research scale, where registered-scale figures — e.g. the
depth-50 cliff, unreachable at demo scale by design — live.
