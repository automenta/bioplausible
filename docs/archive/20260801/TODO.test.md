# TODO.test — What the Current Unit/Property Test Suite Actually Guarantees

**Scope**: `tests/unit/` (141 files; ~763 passing in ~23 s) + `tests/property/` (4 files). Integration (`tests/integration/`, 42 files including the *real* parity/memory/equilibrium gradient work) is explicitly **out of scope** — it is neither CI-gated nor <60 s CPU.

**Verdict up front**: passing this block proves the **plumbing is consistent and mechanically dispatches correctly**, and that one specific optimizer post-condition (spectral-norm projection) actually holds. It does **not** guarantee any of the six biological axioms Bioplausible is designed to validate: contraction, fixed-point reliability, energy descent along relaxation, locality of credit assignment, weight-transport-freeness, or depth-independent memory. The promised "viability proof = passing test suite" is, today, rhetorical for the algorithmic content.

---

## 1. What the suite, as constituted, actually guarantees

| Guarantee | Where | Strength |
|---|---|---|
| Every registered `ModelSpec` exposes 13 metadata slots (citation, family, credit_assignment_type, …) | `test_refactor2_bugfixes.py::test_model_spec_slot_readable` | Real — iterates **all** registered model names |
| The Registry query engine implements a dispatch-table `_QueryFilter` whose 9 predicate axes fire in a deterministic order and short-circuit AND | `test_queryfilter_snapshot.py` (12 tests), `tests/property/test_registry.py` (4 hypothesis laws) | Real — golden order + monotone-filter laws |
| `EquiTile._relax` and `_apply_hebbian_updates` produce byte-identical tensors (atol=1e-7) for seed=7 | `test_helpers_snapshot.py` (12 tests, golden weight tensors) | Real — strongest refactor-regression net in the suite |
| `train_step` mode dispatch (backprop / pc / ep) returns the right `stats["mode"]` and golden `loss`/`accuracy`/`beta` | `test_helpers_snapshot.py::test_train_step_*_dispatch` | Real — pins dispatch + numerical output |
| Energy scalars are non-negative and zero at exact match; `contrastive_energy == (nudged−free)/β`; `hybrid_energy == PE + w·CE` | `tests/property/test_energies.py` (6 hypothesis laws), `tests/unit/core/test_energies.py` | Real *algebraic identity* — necessary preconditions for descent, **not descent itself** |
| `SpectralConstraint.enforce` post-conditions `σ_max(W) ≤ γ·(1+1e-4)` for 2D/4D/embedding params | `test_mep_strategies.py`, `test_optimizer_stubs.py`, `test_spectral_optimizer.py` | **Hardest mathematical claim anywhere in the block** — real SVD-bound post-condition |
| ErrorFeedback with `β=0` degenerates to `NoFeedback` (identity law) | `test_mep_strategies.py` | Real degeneracy law (optimizer plug-in) |
| Scheduler thresholds are pinned: SMOKE=0.12, digits SMOKE=0.50, cifar100 SMOKE=0.05, STANDARD=0.61, DEEP=0.79, evolution trigger ≥100 trials & ≥3 models, decay ratios 0.8³ and 0.9² | `test_strategy_progression.py` (30 tests), `test_strategy_coverage.py`, `test_strategy_diversity.py` | Real *behavioural pins* — but of the scheduler, not any model |
| Checkpoint save/load round-trips,供热 Config defaults populated/duplicated/overwrote warned | `test_checkpoint.py`, `test_config_defaults.py`, `test_config_schema.py` | Plumbing |
| `KnowledgeBase` SQLite CRUD survives edge cases, `from_dict(to_dict(e))==e`, keyword search falls back | `test_knowledge.py` (37 tests) | Plumbing — the "EP achieves O(1) memory" string is stored-and-searched, **not verified** |

That is the totality of meaningful guarantees.

---

## 2. What the suite *does not* guarantee (the bioplausibility axioms)

The README and RESEARCH.md name six pillars of bio-plausible learning. **None** are asserted by any test in scope:

| Pillar | Closest test | Gap |
|---|---|---|
| **Energy descent along relaxation** (Lyapunov) `E₀ ≥ E₁ ≥ … ≥ Eₙ` | `test_equitile_modes.py::test_pc_learning` asserts `pc_losses[-1] < pc_losses[0]` (single endpoint, no monotonicity, 3 epochs, random data) | No test anywhere asserts monotone energy decrease along relax steps for any EP/PC/EquiTile model. The algebraic contrastive-sign identity is asserted; the *dynamics* that uses it is not. |
| **Contraction mapping** ‖T(h)‖ Lipschitz < 1 | `test_zoo_utils.py::estimate_lipschitz` measures weight-norm via power iteration; `test_eqprop_base::test_compute_lipschitz` asserts `L ≥ 0` (tautology) | No test bounds the **relaxation operator's** Lipschitz constant; the stability regime of `step_size` is unguarded |
| **Fixed-point reliability** (attractor from arbitrary `h₀`) | `test_helpers_snapshot.py::test_full_relax_convergence_snapshot` checks *deterministic equality* to seed=7 golden tensor | No test that re-running relax from the endpoint stays put, that residual decays below a real threshold from any init, or that fixed point is unique |
| **Locality of credit assignment** | `credit_assignment_type` tag is read by `test_registry.py::test_query_by_*` | No test asserts layer-`i`'s update excludes signal from layer `j>i`. The EquiTile golden-update snapshot would *silently accept* a refactor that leaks inter-tile signal if it kept the same numeric answer |
| **Weight-transport freeness** | `feedback_alignment` model is *instantiated* (smoke) in `test_model_registry_instantiation.py::test_hybrid_models` | No test asserts backward weights ≠ forward weights, or that updates use the stationary random `B` rather than `W.T`. The defining property of FA family is unverified |
| **Memory independence of depth** | `test_registry.py` asserts `meta.memory_complexity == "O(N)"` (default string); `test_knowledge.py` stores the sentence `"EP achieves O(1) memory"` | No test in-scope verifies actual memory scaling. The real `test_memory_o1.py` lives in `tests/integration/` (excluded) |

---

## 3. The "almost-correctness" tests — what they *were meant* to do and why they don't

Two tests in `tests/unit/models/` were clearly written to be the strongest correctness checks, but were left disabled:

1. **`test_deq.py::test_gradients_match_bptt`** — computes BPTT and equilibrium gradients, computes their **cosine similarity**, then assigns the result to `_ =` and asserts **nothing**. The intended gradient-equivalence claim (EP/DEQ ≈ BPTT) is *unasserted*. This is the single test that, if wired up, would give the strongest algorithmic guarantee.
2. **`test_deq.py::test_memory_usage`** — CUDA-only; the `assertLess(mem_deq, mem_bptt)` line is **commented out**. CPU CI never runs it.

In `tests/unit/execution/`:

3. **`test_robustness.py`** — mocks the model's `inject_noise_and_relax` to return 85 % damping, then asserts the score ∈ [0,1]. The actual noise-and-relax behaviour — the most biologically-sounding test in the repo — is **fully mocked**.

And in `tests/unit/models/`:

4. **`test_oracle.py`** — *designed* to assert `steps_noisy > steps_clean` for the convergence-time-vs-noise metric; the assertion was softened to `len(deltas) > 0`.
5. **`test_equitile_modes.py::test_ep_contrastive_property`** and **`test_pc_local_hebbian_property`** — assert only `weights_changed = True`. A **random weight perturbation** would pass both. The contrastive-direction and locality-of-update claims are unverified.

These five tests are scaffolding for exactly the pillars the project exists to prove. They are disabled, softened, or mocked.

---

## 4. Property tests — what hypothesis is and isn't doing

`tests/property/` (4 files, 22 hypothesis laws total):

| File | Laws encoded | Real invariant? |
|---|---|---|
| `test_base.py` (4) | `len(compute_hidden_dims) == min(num_layers, max_layers)`; `hidden_dim=None → []`; all-equal; zero-layers → [] | **Plumbing only** — config-list length |
| `test_energies.py` (6) | Non-negativity + zero-at-exact-match for `prediction_error`, `mse`, `node` energies; `contrastive_energy == (n−f)/β`; `hybrid == pe + w·ce` | **Mathematical identities** — necessary preconditions for descent, not descent itself |
| `test_registry.py` (4) | Metadata passes own filter; monotone-by-domain; monotone-by-bio_score; empty for disjoint | **Plumbing** — real query-monotone laws, zero biology |
| `test_settling.py` (5) | `_inf_norm_converged` flips correctly; trajectory length contract; settled-count preserved; dynamics-dict keys | **Plumbing around the stop-check** — and unusually passes a `torch.tanh(h + x_in)` test map, **not** any real eqprop `forward_step` |

**Verdict**: of 22 hypothesis laws, **zero** encode a bioplausibility axiom. The settling tests don't even exercise a real model's dynamics. The energy tests prove the energy formula, not that iterating the dynamics decreases it.

---

## 5. Snapshot tests — what regressions they actually catch

### `test_queryfilter_snapshot.py` (24 tests)
Guards the `_QueryFilter.__post_init__` dispatch table — exact-predicate-count (9), ordered-zip matching, AND-short-circuit semantics, every individual predicate's truth-table on a baseline `_META`. Catches **plumbing refactor regressions in the registry query engine**. No scientific content.

### `test_helpers_snapshot.py` (12 tests, golden tensors rtol=1e-5 atol=1e-7)
Guards byte-determinism of `EquiTile._relax` (`_step_with_tolerance`/`_measure_change`/`_check_convergence`) and `_apply_hebbian_updates` (`_propagate_errors_backward`/`_compute_weight_updates`/`_apply_weight_updates`) and `train_step` mode-dispatch. Catches:
- changes to floating-point accumulation order
- changes to RNG usage anywhere in the relax/update path
- sign-flips in error propagation
- mode-routing regressions

It does **not** catch:
- a relaxation that *increases* energy (a sign-flipped `_step_with_tolerance` that produces consistent *wrong* numbers from one seed would change the snapshot — so identity is *indirectly* protected, but not correctness)
- a mathematically *correct but biologically bogus* update rule (e.g. one that sneaks non-local signal but preserves seed-is-deterministic)

**Identity is protected; correctness is not.**

---

## 6. Registry audit — exhaustive or smoke?

**Smoke.** Despite the rhetoric "80+ components registered/instantiable":

- `test_registry.py` tests the *API*. `test_all_models_have_transition_modules_or_override` **explicitly skips 7 model families** by name (pepita, forward_forward, diff_target_prop, contrastive_hebbian, three_factor_hebbian, predictive_coding_hybrid, fabricpc_graph_pcn) and `try/except: pass`es 4 eqprop classes.
- `test_refactor2_bugfixes.py::test_model_spec_slot_readable` — the **broadest** registry test — iterates every registered model name and **reads** 13 ModelSpec slots. It does **not** instantiate.
- `test_model_registry_instantiation.py` — instantiates **only ~13 of 80+** models with `assertIsNotNone(out)`.
- `test_refactor.py` — only 3 eqprop models.

The skip-list is itself an admission: the BioModel/`transition_modules` contract is **not satisfiable** by 7 families without bespoke fixtures. Sprint 2 task 2.7-2.10 ("Registry Audit Unit Test") is exactly the work to fix this — it is **not yet done**.

---

## 7. Execution / scientist tests — strategy-correctness or branch coverage?

**Mostly branch coverage**; two real pin tests:

- `test_strategy_progression.py` (30 tests) and `test_strategy_coverage.py` pin exact thresholds (SMOKE=0.12, digits SMOKE=0.50, cifar100 SMOKE=0.05, STANDARD=0.61, DEEP=0.79, evolution ≥100 trials & ≥3 models, saturation 0.995/20,ASCADE mnist→digits/usps).
- `test_strategy_diversity.py` pins exact recency-decay ratios (`0.8³` and `0.9²` to 2 dp).

Everything else (`test_robustness.py` mocked, `test_monitoring.py` mocked, `test_dashboard_logic.py`, `test_scientist_refactor.py`, `test_algorithm_constraints.py`, `test_strategy_fragility.py`, `test_strategy_transfer.py`) asserts that functions were called and returned. **None test that the scientist identifies a correct algorithm.** The meta-scientific claim of the project — that the auto-scientist pipeline *discovers faithful bioplasible algorithms* — is unguarded at unit level.

---

## 8. Do any tests promise a bio-plausible rule *learns a nontrivial task* or *matches backprop*?

**No, to both.**

- **Nontrivial learning**: no test in scope uses real data, a held-out split, a generalization assertion, or a multi-epoch regime *where overfitting matters*. Every "loss decreases" assertion either (a) trains and evaluates on the **same random batch**, or (b) is loosened to `losses[-1] <= losses[0] + 0.1` (i.e. allows increase). `test_nebc_base::test_losses_decreasing` (15 epochs, same batch, random labels) is essentially noise.
- **Backprop parity**: the single attempt — `test_deq.py::test_gradients_match_bptt` — **computes** cosine similarity and discards it. Every other test that *could* compare to backprop (EP contrastive estimator, FA alignment, target-prop targets, PC inference dynamics) is a no-crash smoke test. The single tight-tolerance assertion in the whole block (`atol=1e-6` on `FiniteNudgeEp::test_beta_magnifies_gradient`) only verifies that an existing *backprop* path scales by β=3 — a property of an optimizer scaling factor, not of any bio-plausible estimator.

---

## 9. Glaring absences — concrete tests that would convert "plumbing reliable" → "algorithmic claims testable"

These would each be tiny additions (≈30–60 LOC) and most are already half-written (disabled):

1. **EP gradient-equivalence check.** Compute `(∂E_free/∂W vs nudged-update-direction)` and assert `cos(grad_ep, grad_bptt) ≥ 0.9` on a 1-hidden-unit MLP at finite β. `test_deq.py::test_gradients_match_bptt` already constructs both gradients — wire up the assertion. This is the **single most important test** for the EP family and is currently a no-op.
2. **Lyapunov energy-descent along relaxation.** For each EP/PC/EquiTile model, run `N=20` relax steps, log `Eₜ` at each step, assert `Eₙ < E₀` AND monotone non-increase (with small `+ ε` slack for numerical). Without this, none of the contrastive-sign algebraic identities buy us the *dynamics*.
3. **Contraction-mapping invariant.** Randomly sample two `h₀`, run `T` once, assert `‖T(h₀)−T(h₀')‖ ≤ L·‖h₀−h₀'‖` for `L < 1`. Use `estimate_lipschitz`'s power iteration to read off `L`. Parameterize on `step_size ∈ {0.1, 0.3, 0.5}`.
4. **Fixed-point reliability.** Run relax from 5 random `h₀` seeds, assert all converge within `rtol=1e-3` of each other. Uniqueness + determinism = attractor.
5. **Weight-transport-freeness.** For each FA-family model, assert `not torch.allclose(forward_W, feedback_B)` at init AND assert `.grad` paths through forward vs backward are *separate* tensors (backward path does not read `forward_W.T`).
6. **Locality-of-credit invariant.** Swap tile `j+i`'s activity with noise while leaving tile `i`'s pre/post unchanged, assert tile-`(i,j)` edge update is **unchanged** modulo machine-eps. The snapshot test would today accept a non-local refactor that happened to preserve seed-7 numbers.
7. **Memory-independence-of-depth smoke.** Allocate models at `depth ∈ {5, 20, 50, 100}` under the DEQ `equilibrium` mode, assert `peak_memory` is flat to within `r_tol=2x`. Either CPU memory accounting (`tracemalloc`) or skipped on CPU and CI-only on GPU. This is the project's headline deliverable.
8. **Adaptive-FA alignment improvement.** After `K=50` training steps, assert `cos(B, W.T)` strictly increases from the initial random value. The current test asserts only that `B` *changes*.
9. **Registry audit Sprint 2.7–2.10 as specified**: instantiate every registered model, run `forward()` on a dummy tensor, assert metadata fields match implementation, assert deterministic output for fixed seed. The current closest test covers ~13 of 80+; the skip-list in `test_all_models_have_transition_modules_or_override` is the list of families needing bespoke fixtures.
10. **Reproducibility (Sprint 2.11-2.13)**: fixed seed → identical weights, identical loss trajectory (5 steps); environment capture (git commit, torch version, deps hash) round-trips. This is literally the next sprint item in `TODO.md` and is not done.

---

## 10. Bottom line — does the suite hold any promise for Bioplausible's stated discoveries?

**Not yet.** The suite says: **the project's plumbing is wired-up correctly and would survive most refactors without silent regressions** — the snapshot tests and the broad MetadataSlot-read test in particular are a strong refactor safety net. The hypothesised Energy formulas and the Spectral-projection post-condition genuinely hold.

But the suite does not yet say: **any bio-plausible learning rule in this repo actually learns even a small task, or matches backprop even at finite β, or contracts, or uses local credit, or runs O(1) memory in depth.** Every test that was designed to make those claims — `test_deq::test_gradients_match_bptt`, `test_deq::test_memory_usage`, `test_oracle`, `test_equitile_modes::test_ep_contrastive_property`, `test_robustness` — is disabled, mocked, or softened to near-tautology.

The TODO.md gate — "passing unit tests *are* the viability proof" — will only become true once the five disabled tests are wired up and the eight missing tests above exist. Until then the suite proves **viability of the framework's plumbing**, not of biological learning.

Sprint 2's parity suite (`test_backprop_parity.py`, `test_reproducibility.py`, `test_registry_audit.py`) is precisely the right next work. The disabled gradient-equivalence test in `tests/unit/models/test_deq.py` is the single fastest path from "plumbing green" to "first real algorithmic claim asserted".
