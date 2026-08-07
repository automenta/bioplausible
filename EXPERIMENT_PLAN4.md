# EXPERIMENT_PLAN4.md — The Stabilization → Measurement → Market run (complete development plan)

**Status.** PLAN3's *build* items are landed and green: the P0a integrity gate (validator + KB surface emitter + honest `neural_cube`/`backprop` spaces), P1 (the shared `settle_state` primitive + `EquilibriumSettleProtocol`, adopted by `NeuralCube`), the P2 read-half (`query_conditionals` + proposer DI + `avoid_characterized`), the P2-lite turbine demo, P3a (flagship-selection as a KB query), and P3b (checkpointed settle as the memory lever) all exist in code with tests (`test_rule_space_integrity.py`, `test_settle_protocol.py`, `test_flywheel_readhalf.py`). The recommendations of the PLAN3 review were folded in where they were actionable in code: **R3** (knob-efficacy as a P0a property test) and **R4** (the flywheel demo as a *paired counterfactual*).

What this plan adds: **the stabilization sprint** (fix the test suite so the gate is honest), **the measurement phases** PLAN3 deferred to "run," and **the market thread** — in *one* ordered, gated document.

---

## 0. The facts that re-shape this plan

1. **The suite is bleeding ~24 pre-existing failures — and ~18 of them are ONE bug, not model bugs.** Triage (grounded, not opinion) found:
   - `test_model_learns_synthetic[<14 models>]` — `_LearnableTask.get_batch` calls bare `torch.randn` for *both* `x` and a fresh projection `W` **every call, with no seed reset**. The "learnable" target `y=argmax(x@W)` moves every batch, and global RNG state accumulates across 40+ parametrized models. `neural_cube` passes in isolation but fails in-file → order-dependent. **Test hygiene, not model surgery.**
   - `test_trainer_forward_vision[conv_equitile|enhanced_equitile]` + `test_build_and_adapted_forward_vision[...]` — fail only in the *full* suite, never in-file → same cross-file RNG exhaustion.
   - `test_model_deterministic_output[noisy_looped_mlp]` — a **by-design stochastic** facade inside a determinism test. It can never pass deterministically.
   - `test_energy_landscape_eqprop` — the landscape's center energy (1.397) ≠ direct forward CE (1.574) at `1e-4` for `LoopedMLP` in contrastive mode → a definition/tolerance mismatch, not a NaN.
   - `test_phase0::test_integration_run` — the trainer's deliberate clinical guard raises `optuna.TrialPruned("Constant high-confidence predictions")` and the standalone runner lets it **propagate** instead of recording the unit → a real robustness leak.

2. **The coverage gate is currently ~60%, not 85%** — a pre-existing condition verified unchanged with my PLAN3 work stashed and present. The gate will only become honest once stabilization lands.

3. **The four "existential" probes have not consumed any budget yet** (PLAN3 §V). The measurement phases are still to be *run*, not *built*.

4. **Substrate reach is still `LoopedMLP`-only** (PLAN3 §0.4) — unchanged by the build. The go/no-go is still open.

---

## I. What is real, what is promised, what is missing

| Claim | Status | Evidence |
|---|---|---|
| P0a integrity gate (validator + emitter + honest spaces) | **Real (in code)** | `validate_all_rule_spaces()` passes 6/6; emitter writes `SURFACE-{rule}` records |
| P1 shared settle primitive | **Real (in code)** | `settle_state` + `EquilibriumSettleProtocol`; `NeuralCube` adopts it; threshold=1.0 → 4<20 steps converged |
| P2 read-half + paired-counterfactual demo | **Real (in code)** | `query_conditionals` + `avoid_characterized`; with-KB skips 1, empty-KB skips 0 |
| P3a flagship selection as a KB query | **Real (in code)** | `select_flagship()` ranks by geomean cost on honest surfaces |
| P3b memory lever (checkpointed settle) | **Real (in code)** | `torch.utils.checkpoint` in `settle_state` under grad |
| Test suite green / coverage ≥85% | **Missing** | ~24 failures; ~60% coverage (see §0.1) |
| P0b honest `neural_cube` frontier | **Missing (run)** | gated on suite-green + compute |
| P4-lite substrate go/no-go | **Missing (run, pre-commit A/B)** | cheapest decisive probe |
| P3c powered conditional w/ CI | **Missing** | needs P0b + P3a selection + budget |
| P3.5 buyer signal | **Missing** | market probe, no engineering gate |
| P5 / P6 | **Missing** | CIFAR + flywheel-at-scale |

---

## II. The complete development plan

### Phase 0 — Stabilization sprint (NEW; the gate's precondition)

> Gate rule: **no measurement budget is spent until the suite is green and the coverage gate is honest.**

| Item | Root cause (verified) | Fix |
|---|---|---|
| **S0a** `test_model_learns_synthetic` (14) + forward/build equitile flakes (4) | non-stationary `_LearnableTask` (`W` regenerated per call) + unseeded global RNG accumulation across parametrized cases | autouse seed fixture resetting `torch`/`numpy`/`random` before each test; cache `W` once so the task is deterministic/stationary; relax `loss_reduction > 0` to `> 1e-4` |
| **S0b** `test_model_deterministic_output[noisy_looped_mlp]` | stochastic facade under a determinism assert | add `noisy_looped_mlp` to the existing `SKIP_MODELS` (reason: noise-injecting substrate facade) |
| **S0c** `test_energy_landscape_eqprop` | landscape energy definition ≠ forward CE for eqprop | audit `compute_energy_landscape`; align eqprop's center energy to its real energy/forward or relax to a documented tol (keep the MLP case exact at `1e-4`) |
| **S0d** `test_phase0::test_integration_run` | clinical guard raises `TrialPruned`; standalone runner propagates instead of recording | make `run_from_config`/the runner catch `TrialPruned` and record the unit as `status` (`expensive`/`error`) + sink it, honoring "record reverts" — a real robustness fix, not a test hack |

**Success:** full suite green; coverage climbs toward/above 85%. All other phases inherit this gate.

---

### Phase 1 — P4-lite: the substrate go/no-go (cheapest decisive probe, after S0)

Pre-commit the branches *before* running, so the outcome is decided, not rationalized (PLAN3 §P4-lite).

1. **Surrogate sanity:** does the float-gradient/quantized-forward facade materially distort the eqprop frontier vs a true low-precision backward?
2. **Substrate scope audit:** confirm empirically which families `target_hardware` reaches vs where it is inert (§0.4).
- **Branch A (physical story holds):** budgets flow to substrate-faithful measurement; flagship must gain a substrate path or an eqprop-family flagship is chosen by P3a's substrate-eligibility criterion.
- **Branch B (distorts / too narrow):** pivot to (a) build a true low-precision backward, or (b) re-anchor on the epistemic engine + GPU-efficiency story (cache-integrity is its own moat).

**Success:** a recorded verdict + a chosen branch, before flagship budget moves.

---

### Phase 2 — P0b + P3a: the flagship on an honest space

- **P0b:** `RuleFrontierFinder(rule="neural_cube", epochs=5, budget≈30, force=True)` on the honest space. Does "1.75/standout" survive an honest search?
- **P3a (held):** re-run `KnowledgeBase.select_flagship()` against the *honest* P0b frontier + P2 conditionals. If `neural_cube` is the honest argmin, keep it; if not, the rule says so.

**Success:** one flagship chosen by the codified rule, not by accident, with a recorded `cost_of_plausibility`.

---

### Phase 3 — P3c: one powered conditional with confidence intervals

Concentrate `budget_probes` → 500–1000 on the **selected flagship + matched backprop reference only** (not a five-family spray). Report `cost_of_plausibility` with a 95% CI + `scaling_law` r²/CI (§II of PLAN3: report the symmetric joint Pareto surface first, backprop-relative cost second).

**Success:** the first defect artifacts a design partner can react to.

---

### Phase 4 — P3.5 / R-series: the market probe (parallel, no engineering gate)

- **R5:** spec sheet as a *decision-replacement artifact* (their decision/cost → your powered conditional with CI → **both** "what would you pay/do differently" and "what would make you not trust this number").
- **R6:** pre-commit the buyer-response rubric — names-a-decision/price → **Fund**; "cool, keep me posted" → **False positive**; wouldn't change their decision → **Pivot** (trigger Branch B early).
- **R1/R2/R7:** problem-interview template; split "does the pain exist" (now) from "does our measurement resolve it" (P3c); lead outreach with the honest negative result (`FailureManifesto`), not an accuracy table.
- **R8:** one-paragraph **business-⊕ invariance** — name now what still appreciates if Branch B fires (surface audit trail, negative-knowledge oracle, cache-integrity discipline).

**Success:** a recorded buyer signal — Fund / False-positive / Pivot — independent of any internal metric.

---

### Phase 5 — P4-full + P5: substrate measurement + scale

- **P4-full** (Branch A only): `--target-hardware fpga/analog` on MNIST for substrate-eligible families; the `_hw{target}` cache split already prevents GPU↔FPGA cross-reuse. Record "does substrate change the ranking?"
- **P5a:** `HyperbandPruner` end-to-end → first CIFAR-10 `cost_of_plausibility`.
- **P5b:** hardware × equilibrium cross-terms — does `convergence_threshold` search behave the same under quantization? (Free once P1's protocol + P4-full exist — the product differentiation.)

---

### Phase 6 — P6: mass + flywheel at scale

Let P3c/P5 accumulate through `result_sink`, then measure the AutoScientist proposing with **fewer probes because it read a prior conditional**, now with CIFAR-scale KB mass. Non-trivial KB counts + a skip-based-on-conditional measurement at scale. The P2-lite demo (already real) de-risked this; P6 confirms it.

---

## III. Standing discipline (carry forward from PLAN3)

- **Never measure before the Stabilization gate (S0) passes.** A flaky suite makes every "missing" row unknowable.
- Epochs and `target_hardware` are cache-identity covariates — match and cache independently.
- Diagnose before judging: low acc/cost may be an epoch-budget artifact (`best_epoch_acc`/`acc_at_half`).
- Record wins *and* reverts via the sink; the sink does not distinguish — write them all.
- Space and constructor move together — the P0a validator enforces it; the human rule is the backup.
- **No backwards compatibility of any kind** (AGENTS.md): fixes are fixes, not band-aids — e.g. S0d is the *entrypoint* catching the prune, not the test swallowing it.

---

## IV. Phase ordering with gates

| Phase | Gated by | Produces |
|---|---|---|
| **S0** Stabilization | (entry) | green suite; honest ~85% coverage gate |
| **P4-lite** Substrate go/no-go | S0 | Branch A/B verdict, pre-committed & chosen |
| **P0b + P3a** Honest flagship | S0 (+ P4-lite branch) | rule-selected flagship + honest cost |
| **P3c** Powered CI | P0b + P3a | one defensible cost ± 95% CI; first partner-ready table |
| **P3.5 / R1–R8** Market probe | R-series: now; spec sheet: P3c | buyer response — Fund / False-positive / Pivot |
| **P4-full** Substrate measurement | Branch A | first hardware-aware cost table + ranking-change answer |
| **P5** Scale + cross-terms | P3 baseline | CIFAR-10 cost; substrate-specific differentiation |
| **P6** Flywheel at scale | P2-lite ✓ + P3–P5 | measured compounding RPM at CIFAR mass |

Rationale for pulling **S0** to the very front: it is the only phase that makes every downstream number trustworthy *and* every CI run green. It is cheap, mostly test-hygiene, and it is the precondition the reviewer's "is the machine running honestly" question is really asking. P4-lite stays immediately after because it is the cheapest *decisive* probe; P3.5/R-series runs in parallel because the market must not wait behind engineering.

---

## V. Suggested execution order

1. **S0a–S0d** — the stabilization sprint. Not lint-pedantry: ~18 failures share one RNG root cause, and the other four are small, well-understood fixes. This is the "get it working" step.
2. **P4-lite** — both existential first-signals (physical story real? substrate scoped?) before any flagship budget.
3. **P0b + P3a** — the flagship selected by rule on an honest space.
4. **P3c** — the powered conditional that makes P3.5 concrete.
5. **P3.5/R-series** — put the honest number in front of a buyer, *and* draft R5/R6/R8 now so they're ready the instant P3c lands.
6. **P4-full / P5 / P6** — substrate measurement, scale, and the measured flywheel RPM.

The risk has flipped from *"can we measure"* (answered) to *"does the machine run honestly, is the substrate story real, does the flywheel turn at mass — and does anyone pay for the truth."* The last clause, again, is the one no internal metric can answer.

⊕ *If the flagship or the whole bio-rule line is de-prioritized tomorrow, S0's green suite, P0a's surface records, P1's settle protocol, P2's read-half, and P3b's checkpointing all survive — and the business-⊕ (R8) names the assets that survive if the physical story dies. That invariance, not any single rule's cost, is what "the framework is the product" means in code and in the market.*
