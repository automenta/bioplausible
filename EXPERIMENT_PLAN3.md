# EXPERIMENT_PLAN3.md — The Construction Phase (flagship-led, framework-first, thesis-validating)

**Status.** The thesis has crossed from *argument* to *construction*: the fair-comparison pipeline runs end-to-end on five bio families (PLAN2 §13/§16.4), the training loop is at its floor (~1.9 s/epoch, PLAN2 §5), `peak_memory_mb` is verified (PLAN2 §6), the equilibrium early-stop lever is found for `eqprop` (PLAN2 §7), and the knowledge layer is *physically wired* — every probe, execution-engine trial, and hardware track compounds into `KnowledgeBase`/`FailureTracker` via `result_sink` (PLAN2 §17/§18). The integrity risk of the **digital-GPU benchmarking fallacy** has a mitigation in code (`target_hardware` → substrate facades, hardware-keyed cache identity, PLAN2 §18).

This plan is the forward path for **Session 6 onward**. Two organizing principles:

> **1. The framework is the product; any one algorithm is a fixture.** Every hour is judged by whether it makes the *next* experiment cheaper, more honest, or more decision-relevant — not by whether it improves `neural_cube`.
>
> **2. Internal epistemic integrity is necessary but not sufficient.** The thesis is "sell *verified physical truth* to people making expensive, irreversible physical decisions." A plan that only validates its own measurements never asks whether anyone will pay for the output. This plan carries a thin, parallel external-validation thread (§V) so the *market*, the *physical story*, and the *compounding engine* — not just the flagship's third decimal — are what gates funding.

---

## 0. The facts that re-shape this plan (learned the hard way)

These are not opinions; they are verifiable facts from the current tree.

1. **The flagship's search space is partly fictional (the §12 bug, on the flagship itself).** `RULE_SPACES["neural_cube"]` advertises `damping`, `hidden_dim`, `tol` — but `NeuralCube.__init__` accepts only `cube_size, input_dim, output_dim, max_steps` with no `**kwargs`, so `build_model_kwargs` *silently drops* them. TPE has been sweeping `hidden_dim ∈ [32,1024]` to zero effect. A registry-vs-signature audit confirms **only `neural_cube` has this drift** (`eqprop`/`feedback_alignment` are safe via `**kwargs` absorption). The "1.75 / standout" number rests on a partly-honest search.

2. **The flagship's headline lever does not exist in the model.** The §7 win was measured on `StandardEqProp` (settle through `settle_activations_list`, `zoo/_settling.py:344`). `NeuralCube.forward` hand-rolls a fixed `for _ in range(steps)` (`neural_cube.py:175`) with no convergence check. Adding the knobs to its space would sample values the constructor drops. The lever is missing code, not a config toggle — and it is shareable by construction (§P1).

3. **The flagship target is arithmetically unreachable via the offered axis.** `cost_of_plausibility` is the **geometric mean** of (FLOPs× × mem× × time×): `(0.52·6.38·2.7)^(1/3) = 2.08 ≈ 2.09` (matches PLAN2 §13). A full 1.3× *time* cut → **1.90**, not 1.4; from the 1.75 baseline → **1.60**; reaching 1.5 by time alone needs **1.59×**. And the dominant axis is **memory (6.4×)**, larger than time (2.7×). The flagship's binding constraint is memory, and the plan has no memory lever for it yet.

4. **The substrate facade is `LoopedMLP`-only — and the flagship is not a `LoopedMLP`.** Verified: `backprop_mlp`, `neural_cube`, `pepita`, `forward_forward`, `feedback_alignment` are all non-`LoopedMLP`; only the eqprop family (`eqprop_mlp`, …) is swappable via `target_hardware` (the swap gates on `isinstance(model, LoopedMLP)`). Consequences, stated plainly:
   - `--target-hardware fpga/analog` is **currently inert for the flagship `neural_cube` and for the backprop reference** — only the eqprop family can be substrate-measured as built.
   - The "physical truth" story is today *only shippable on the eqprop family*. The flagship — the thing being driven to a viability target — **cannot even enter the substrate test**.
   - This is a sharper version of the surrogate-assumption problem (§P4): it is not just "float backward," it is "most rules have *no* substrate path at all."

These four facts are why the plan: leads with **framework integrity (P0)**, builds a **shared settle primitive (P1)**, codifies a **repeatable flagship-selection rule (P3)**, and elevates **substrate feasibility to a go/no-go (P4)** rather than a footnote.

---

## I. What is real, what is promised, what is missing

| Claim | Status | Evidence |
|---|---|---|
| Fair-comparison pipeline runs end-to-end | **Real** | PLAN2 §13/§16.4, five families |
| Training loop at floor | **Real** | ~1.9 s/epoch, PLAN2 §5 |
| `peak_memory_mb` verified | **Real** | 0.96× `max_memory_allocated()`, PLAN2 §6 |
| Early-stop lever found | **Real for `eqprop`; absent for others** | PLAN2 §7 on `StandardEqProp`; `NeuralCube` has none (§0.2) |
| Knowledge layer *fed* | **Real** | `result_sink` → KB→FailureTracker, PLAN2 §17/§18 |
| Knowledge layer *read* (flywheel) | **Missing** | `AutoScientist.proposer` reads the registry, not conditionals (P2) |
| Substrate-faithful measurement | **Mechanism built, unexercised, and scoped** | `target_hardware` + facades + `_hw{t}` cache key; **`LoopedMLP`-only today** (§0.4) |
| Flagship `≤ 1.5` | **Not yet meaningful** | fictional space + absent lever + memory-bound (§0.1–0.3) |
| Cache-identity discipline | **Real (the crown jewel)** | epochs (§16.3) + `target_hardware` (§18) are cache-identity covariates |
| **A full flagship** | **Unselected by rule** | no repeatable criterion; `neural_cube` was chosen by accident (§P3) |

Every "P" item below names which Status row it converts to **Real**.

---

## P0 — Framework Integrity Gate (mandatory, blocks all measurement)

> **Gate rule: no probe budget is spent on a family whose `RULE_SPACES` entry does not match its constructor signature, and whose settle path does not route through the shared early-stop utility.**

### P0a — Enforce the `RULE_SPACES` ↔ constructor contract, and *emit the surface to the KB*

`build_model_kwargs` silently drops dimensions a model doesn't accept. This must fail loudly **and** feed the framework.

**Action — the validator (pure: no `Any`, no module-level I/O, AGENTS.md §21/§39):**
- Signature `validate_rule_space(rule: str) -> None` / `validate_all_rule_spaces() -> None` in `hyperopt/search_space.py`; reflect `Registry.get(ComponentCategory.MODEL, name).__init__` via `inspect.signature`; raise a custom `SpaceSignatureMismatchError(rule, phantoms: frozenset[str])` (domain exception hierarchy, chained from source, AGENTS.md §44) on any advertised key the constructor neither accepts nor absorbs via `**kwargs`. Call in `RuleFrontierFinder.find()` and `IdealBackpropFinder` **before** the first probe.
- Test with `hypothesis` property: "every advertised key is accepted or `**kwargs`-absorbed" (AGENTS.md §49).

**Action — the emitter (this is what turns P0 from a toll booth into a compounding asset):**
- Extend the validator to also **write a machine-readable constructor-surface record per family into the `KnowledgeBase`**: which knobs are real, which are absorbed (and by what `**kwargs`), the signature snapshot, and the validated-space hash — as of this commit.
- Two benefits (the reviewer's point, adopted):
  1. **P2's flywheel can reason about which dimensions are real** when proposing — the integrity gate becomes part of the compounding asset, not just a one-time gate.
  2. A permanent, queryable audit trail that **retroactively prices every historical number** ("this space was honest as of commit X"), itself a trust artifact for a design partner.

**Fix `neural_cube` immediately:** drop `hidden_dim`, `damping`, `tol` from its space until each is implemented; keep the honest `{lr, weight_decay, cube_size, max_steps}`. Re-add each dimension in the same PR that implements it.

**Success:** `validate_all_rule_spaces()` exits 0; the emitter populates the KB surface records; a future contributor adding a phantom knob gets a CI failure *and* a KB audit entry, not a quiet probe waste.

### P0b — Re-derive the `neural_cube` 5-epoch frontier on the *honest* space

Run `RuleFrontierFinder(rule="neural_cube", epochs=5, budget=~30)` `force=True`. Small budget; the question is whether "1.75 / standout" survives an honest search. Both branches:
- **Holds** → feed into P3's selection rule as a candidate.
- **Collapses** → the selection rule (§P3) picks the real flagship; no budget lost to a wrong bet.

---

## P1 — One settle primitive, shared by every equilibrium rule (composition over inheritance)

The directive: *don't over-invest in any one algorithm; share early-stop across the family.* The machinery already exists in one place (`settle_activations_list`, `zoo/_settling.py:344`) and `EquilibriumFunction` already duck-types `convergence_start`/`convergence_threshold` off the model (`_settling.py:468-469`). `NeuralCube` just bypasses it.

**Design — `EquilibriumSettleProtocol` + a pure helper (AGENTS.md §28/§15/§29):**
- `EquilibriumSettleProtocol(Protocol)` stating the structural requirement (`convergence_threshold`, `convergence_start`, `max_steps`, `W_in`, `W_rec`, `_transform_input`, `_forward_step_impl`).
- Pure `settle_state(model: EquilibriumSettleProtocol, x, *, steps=None, return_trajectory=False) -> tuple[torch.Tensor, int, bool]` routing through `settle_activations_list`, returning `(h_final, steps_taken, converged)`. No mutation, no `Any` (AGENTS.md §21).
- `NeuralCube` adopts the protocol: add `convergence_threshold=1e-3`/`convergence_start=5` to `__init__`, replace the hand-rolled `for _ in range(steps)` (`neural_cube.py:175`) with a one-line `settle_state(...)`, expose `steps_taken`/`converged` as probe metrics.
- Searchability: `convergence_threshold ∈ [1e-4,1e-2]` (log), `convergence_start ∈ [2,10]` (int) for **any** protocol-adopting rule — added to `RULE_SPACES["neural_cube"]` in the same PR as the model change (P0a discipline).

**Success:** every protocol-adopting model early-stops via one primitive; a unit test (pytest + hypothesis) asserts a `convergence_threshold=1.0` model terminates in `< max_steps` with `converged=True`. The §7 win becomes a framework property, not an `eqprop` quirk.

---

## P2 — Wire the flywheel's read-half, then *immediately* prove the turbine turns

**The build (the read-half):** add a `ConditionalQuery` surface to the AutoScientist: `KnowledgeBase.query_conditionals(query) -> list[ConditionalResult]` given `(task, accuracy_target, memory_cap, FLOPs_cap, substrate)` where the query is a `TypedDict` and the result a `@dataclass(frozen=True, slots=True)` (AGENTS.md §20); Pydantic v2 at the method boundary, custom `ConditionalQueryError` with chaining (§44), **t-strings** for logging (§42), context-managed DB (§45), **dependency injection** of the query service into the proposer (§50), `hypothesis` property tests (§49). Gains access to the constructor-surface records emitted by P0a.

**The minimal demo (the reviewer's pull-forward — the "turbine turns" signal, not deferred to P6):** immediately after the read-half lands, run **one engineered demonstration on toy conditionals** — no CIFAR mass required: *"the proposer read a prior conditional and skipped an already-characterized probe."* That single signal answers the existential question "does it compound?" before the flagship budget is spent.

**Success:** one reproducible skip-based-on-conditional demo at P2-time, plus the read-half in place. This is the compounding claim made *measurable by construction* — and it is used to decide how confidently P3–P5 are funded.

---

## P3 — The flagship: *selected by rule*, memory-bound, CI-backed

### P3a — A repeatable flagship-selection rule (codified, not narrative)

The plan's own warning (P0b: "the flagship changes") exposes the gap: **there is no rule for choosing the new flagship.** `neural_cube` became flagship by accident. Codify it now, against the honest outputs of P0/P1:

> `flagship = argmin over validated families of cost_of_plausibility` subject to
> ↳ **non-phantom space** (P0a validation passed — dims are real),
> ↳ **≥ 1 wired lever on the binding axis** (a mechanism that exists *in code*, e.g. the P1 settle primitive or the P3b checkpointing),
> ↳ **substrate-eligible if the physical story is the pitch** (§0.4 — today that excludes `neural_cube`/backprop unless they gain a substrate path),
> ↳ **minimal accuracy-to-backprop gap**.

Implement it as a **query over the `KnowledgeBase`** (using the P0a surface records + P2's conditional query) rather than a judgment call — making flagship-selection a repeatable framework operation, consistent with the thesis. If `neural_cube` is the honest argmin, keep it; if not, the rule says so before the budget moves.

### P3b — A real memory lever (the binding axis) for the selected flagship

The cost axis analysis (§0.3) says time alone cannot cross 1.5 (1.75→1.60). The binding axis is **memory (6.4×)**. Candidates, each a **controlled experiment** (PLAN2 §7 discipline):
- `cube_size` search at fixed capacity (smaller cubes cost less: `n_neurons = cube_size**3`).
- **Activation checkpointing on the settle loop** (PLAN2 §2): ~2× compute, ~O(√steps) memory — a *framework* optimization for any protocol-adopting rule, not a `neural_cube` patch.
- Substrate precision/sparsity (once P4 unlocks a substrate path for the flagship).

**Honest target:** memory 6.4×→~3× *combined* with P1's time 1.3× lands `(0.52·3.0·2.08)^(1/3) ≈ 1.35` — viable *if* both levers hold at negligible accuracy cost. Not asserted; measured.

### P3c — Powered CIs, concentrated (not a spray)

Raise `budget_probes` toward 500–1000 (PLAN2 §15.4) **concentrated on the selected flagship + matched backprop reference**, not across five families. Report `cost_of_plausibility` with 95% CI + `scaling_law` r²/CI. This powered table is the first artifact a design partner can react to (§V).

### P3d — The flagship is secondary to the framework, by construction

P0a/P1/P2/P3b all serve the *next* algorithm too. The ⊕ invariance holds: if the selected flagship is de-prioritized tomorrow, nothing in P0–P3 is wasted.

---

## P3.5 — External-validation thread (the market probe) — *thin, parallel, cheap*

Every other item validates the *measurements*. This validates the *market*, because P0–P6 can all pass while the output is still trivia. The moment P3c produces **one powered conditional** on an honest space, translate it into a **buyer-facing spec sheet** — *not* "cost_of_plausibility 1.6," but *"rule on substrate X, accuracy Y, within memory Z and power W."* Put it in front of one design-partner candidate from the business plan.

Their response — *"this de-risks my decision"* vs. *"this is trivia"* — is the signal no internal metric can give. It costs almost nothing. It does not wait for work it can't use; it runs the instant a defensible conditional exists.

---

## P4 — The substrate is a go/no-go, not a sanity check — and it is currently scoped

The first-principles pitch is *physical truth*. The reviewer is right: if the backward pass stays in float (surrogate accumulation), substrate numbers are **digital truth wearing a physical costume**, and "the facade is a visualization, not a measurement" collapses the moat to *GPU-optimization consulting* — crowded and weak. This is not a footnote; it is the **go/no-go for the physical-truth thesis**. And §0.4 sharpens it: today **most rules (incl. the flagship) have no substrate path at all.**

### P4-lite — pre-commit the go/no-go, early and cheap (after P0)

Run two cheap probes immediately after P0 (see §VI reordering), **before** the flagship budget:
1. **Surrogate sanity:** does the float-gradient/quantized-forward facade materially distort the frontier vs. a true low-precision backward? Measure on one eqprop-family rule.
2. **Substrate scope (the §0.4 corollary):** confirm empirically which families `target_hardware` currently reaches vs. where it is inert. Pre-commit the branches **before** running this, so the outcome is decided, not rationalized:
   - **Branch A — physical story holds:** budgets flow to substrate-faithful measurement; the flagship must gain a substrate path (or an eqprop-family flagship is chosen by P3a's substrate-eligibility criterion).
   - **Branch B — surrogate materially distorts / substrate is too narrow:** pivot the pitch to (a) build a true low-precision backward (real cost), or (b) re-anchor the business on the **epistemic engine + GPU-efficiency** story (weaker but real — and roughly its own moat: cache-integrity discipline IS the product, PLAN2 §18).

### P4-full — executed substrate measurement (after the go/no-go is green)

- `--target-hardware fpga` and `--target-hardware analog` on MNIST for whatever families have a substrate path (currently eqprop `LoopedMLP`-based only; expand after P4-lite per the chosen branch). The `_hw{target}` cache split is already keyed, so GPU vs FPGA runs can't cross-reuse.
- **Success:** a real fpga/analog `cost_of_plausibility` table on substrate-eligible families, and a recorded "does substrate change the ranking" answer. If a chosen flagship has no substrate path, that fact alone — surfaced at P4-lite — is the decision.

---

## P5 — Scale (CIFAR-10) + hardware × equilibrium cross-terms

- **P5a — CIFAR-10 via multi-fidelity:** exercise `HyperbandPruner` end-to-end (PLAN2 §15.6). Fidelity = epochs / dataset fraction. Success: first CIFAR-10 `cost_of_plausibility`; the multi-fidelity path proven, not just wired.
- **P5b — Hardware × equilibrium cross-terms:** on substrate, does `convergence_threshold` search behave the same under quantization? Record the interaction via the P1 protocol (free once the protocol lands). This is the product differentiation: substrate-specific optimization, not GPU-parity.

---

## P6 — Mass, and the demonstrated flywheel at scale

P2 proved the turbine turns on toy conditionals. **P6 scales it to real mass**: let P3–P5 accumulate through `result_sink`, then measure that the AutoScientist proposes with fewer probes *because it read a prior conditional*, now with CIFAR-scale data in the KB. The P2-lite demo de-risked this early; P6 confirms it with mass. Non-trivial KB counts **and** a skip-based-on-conditional measurement at scale.

---

## II. Metric honesty — the parity trap, fixed at the report layer

`cost_of_plausibility` is defined *relative to backprop* (parity framing, PLAN2 §1). Fix at the report layer (no re-measurement):
- **Primary:** the **symmetric joint Pareto surface** — "is rule R on the Pareto frontier at all, at budget B?" Backprop is one point on the joint frontier, not the fixed reference. Report each rule's *fraction of the joint frontier* and the *crossover budgets* where domination switches.
- **Secondary:** `cost_of_plausibility` as a backprop-relative summary for the viability-threshold call (≤1.5 / ≥5).

Carried by existing `pareto_frontier`/`compare_frontiers` infrastructure — cheap, and it stops the project from publishing, as its flagship number, the metric its own §1 called the wrong question.

---

## III. Standing discipline (carry forward from PLAN2)

- **Epochs and `target_hardware` are covariates:** match and cache independently (§16.3, §18).
- **Diagnose before judging:** low acc/cost may be an epoch-budget artifact — use `best_epoch_acc`/`acc_at_half` (§12).
- **Record wins *and* reverts** via the sink; the sink does not distinguish — write them all.
- **Never measure before P0 passes.**
- **Space and constructor move together** — the P0a validator enforces this; the human rule is the backup.

---

## IV. Engineering guardrails (AGENTS.md, applied to every build item)

| Item | Guardrail (AGENTS.md) |
|---|---|
| P0a validator/emitter | pure function (§39); no `Any` (§21); custom `SpaceSignatureMismatchError` chained (§44); hypothesis property tests (§49) |
| P0a KB surface record | `TypedDict`/frozen dataclass (§20); Pydantic v2 at boundary (§20); idempotent write via `result_sink` |
| P1 protocol + helper | `Protocol` over ABCs (§15); composition over inheritance (§28); immutable/pure settle helper (§29); no `Any` (§21) |
| P2 conditional query | `TypedDict` query + frozen-dataclass result (§20); Pydantic v2 boundary (§20); custom error chaining (§44); t-strings for logging (§42); context-managed DB (§45); DI over mocking (§50); hypothesis (§49) |
| P3b checkpointing | isolated side effects; controlled experiment per PLAN2 §7 |
| Tests throughout | pytest + `--cov` ≥85% (§48); fixtures/`@pytest.mark.parametrize` (§51); hypothesis for pure logic (§49) |

---

## V. Existential risks & their cheap first signals (the thesis can't be argued into existence)

The plan now treats these three — market, physical story, and compounding — as crucially-decisive *probes*, each cheap, each run before the expensive engineering locks in:

| Existential risk | Cheap probe | Where it lands | Decides |
|---|---|---|---|
| **Does anyone care?** | buyer-facing spec sheet on one powered conditional | P3.5 | fund vs. pivot branching; the whole thesis |
| **Is the physical story real?** | substrate surrogate sanity + scope audit | P4-lite (after P0) | Branch A vs B (§P4) |
| **Does it compound?** | proposer skip-based-on-conditional on toy data | P2-lite (immediately after P2) | how confidently to fund P3–P5 |

These three are more decisive for the business than the flagship's third decimal. They must not wait behind it.

---

## VI. Phase ordering with explicit gates (the reordering)

| Phase | Gated by | Produces |
|---|---|---|
| **P0** Integrity (+ KB surface emitter) | (entry) | honest search spaces; honest `neural_cube` re-check; **`validate_rule_space` materialized in the KB** |
| **P4-lite** Substrate go/no-go | P0 | surrogate verdict + substrate-scope fact; Branch A/B pre-committed & chosen |
| **P2 + P2-lite** Flywheel read-half + minimal demo | P0 | `ConditionalQuery`; **turbine-turns signal** on toy data |
| **P1** Shared settle primitive | P0 | the §7 win as a framework property |
| **P3** Flagship (rule-selected, memory-bound, CIs) | P0; P1 (time lever); P3b (memory lever); P2/P4-lite signals | one defensible `cost ≤ ~1.5-1.6` ± CI, **or** an honest "one lever from viable" |
| **P3.5** External-validation thread | P3c (one powered conditional) | buyer reaction — the market probe |
| **P4-full** Substrate measurement | Branch A green | first hardware-aware cost table, substrate-eligible families |
| **P5** Scale + cross-terms | P3 baseline | CIFAR-10 cost; substrate-specific differentiation |
| **P6** Mass + flywheel at scale | P2-lite + P3–P5 data | the *measured* compounding RPM at CIFAR mass |

Rationale for pulling P4-lite and P2-lite forward to run right after P0: after P0, the two remaining existential risks are *"does substrate matter / is the physical story real"* and *"does it compound."* Both are cheap, both are more decisive for the business than the flagship's exact cost, and neither needs the flagship to exist. The flagship is an internal artifact; these two are thesis-validation. We keep all integrity discipline (P0 still gates everything) while **stopping the business case from waiting behind the engineering case.**

---

## Bottom line

The thesis is now a machine to run — *and to check* — not a plan to argue. The order of operations is decisive:

1. **P0 — make the search spaces real, and emit them to the KB.** Highest-leverage hours in the whole plan. It retroactively prices historical numbers, feeds P2, and is itself a trust artifact.
2. **P4-lite + P2-lite — get the two existential first-signals** (is the physical story real; does it compound) *before* the flagship budget. Pre-commit the go/no-go branches first so the outcome is decided, not rationalized.
3. **P1 — the shared settle primitive** as a framework property.
4. **P3 — the flagship, selected by a codified rule, memory-bound, CI-backed**, and reported as a symmetric Pareto result first and `cost_of_plausibility` second.
5. **P3.5 — put the powered conditional in front of a buyer**, because internal epistemic integrity is necessary but not sufficient.

The risk has flipped from *"can we measure"* to *"are the dimensions real, is the lever wired, does the flywheel turn — and does anyone pay for the truth we're selling."* That last clause is new to this revision and it is the correct one: a machine that validates only its own outputs is a machine that has stopped asking whether it matters.

⊕ *If the selected flagship or even the whole bio-rule line is de-prioritized tomorrow, P0a's surface records, P1's settle protocol, P2/P2-lite's flywheel read-half, and P3b's memory lever (checkpointing) all compound: they serve any conditional in the KB, any equilibrium rule, and any future algorithm. That invariance — not the success of any one rule — is what "the framework is the product" means in code, and the external-validation thread (§V) is what keeps the framework pointed at a market instead of a mirror.*
