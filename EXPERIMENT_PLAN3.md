# EXPERIMENT_PLAN3.md — The Construction Phase (flagship-led, framework-first)

**Status.** The thesis has crossed from *argument* to *construction*: the fair-comparison pipeline runs end-to-end on five bio families (PLAN2 §13/§16.4), the training loop is at its floor (~1.9 s/epoch, PLAN2 §5), `peak_memory_mb` is verified (PLAN2 §6), the equilibrium early-stop lever is found and searchable for `eqprop` (PLAN2 §7), and the knowledge layer is *physically wired* — every probe, execution-engine trial, and hardware track now compounds into the `KnowledgeBase`/`FailureTracker` via `result_sink` (PLAN2 §17/§18). The integrity risk identified as the single biggest correctness threat — the **digital-GPU benchmarking fallacy** — has a mitigation in code (`target_hardware` → substrate-faithful facades, hardware-keyed cache identity, PLAN2 §18).

This plan is the forward path for **Session 6 onward**. Its single organizing principle:

> **The framework is the product; any one algorithm is a fixture.** Every hour is judged by whether it makes the *next* experiment cheaper, more honest, or more decision-relevant — not by whether it improves `neural_cube`.

---

## 0. The three corrections that re-shape this plan (learned the hard way)

These are not opinions; they are verifiable facts from the current tree and they change what "drive the flagship" even means.

1. **The flagship's search space is partly fictional (the §12 bug, on the flagship itself).** `RULE_SPACES["neural_cube"]` advertises `damping`, `hidden_dim`, `tol` — but `NeuralCube.__init__` accepts only `cube_size, input_dim, output_dim, max_steps` and has no `**kwargs` to absorb the rest, so `build_model_kwargs` *silently drops* them. TPE has been spending probes sweeping `hidden_dim ∈ [32,1024]` to zero effect. A registry-vs-signature audit confirms **only `neural_cube` has this drift** — every other family's space matches its constructor (`eqprop`/`feedback_alignment` are safe via `**kwargs` absorption). The "1.75 / standout" number is therefore from a partly-honest search; it may be *better* than the true optimum (wasted budget) or the standout status may be a sampling artifact of two real knobs. **Before betting on it, make the search real.** (PLAN2 §12 caught this class of bug for `forward_forward`/`feedback_alignment` and was never checked for `neural_cube`.)

2. **The flagship's headline lever does not exist in the model.** The §7 adaptive-early-stop win was measured on `StandardEqProp`, whose settle routes through `settle_activations_list` (`zoo/_settling.py:344`) with `convergence_threshold`/`convergence_start`. `NeuralCube.forward` hand-rolls a fixed `for _ in range(steps)` loop (`neural_cube.py:175`) — **no convergence check, no `tol`, no early-stop pathway.** Adding the knobs to `RULE_SPACES["neural_cube"]` would sample values that the constructor drops. The lever is not a config toggle; it is missing code. The good news: the win is already shareable by construction — see **§3**.

3. **The flagship target is arithmetically unreachable via the offered axis.** `cost_of_plausibility` is the **geometric mean** of (FLOPs× × mem× × time×) — verified: `(0.52·6.38·2.7)^(1/3) = 2.08 ≈ 2.09` (matches PLAN2 §13). On that profile:
   - A full 1.3× *time* cut → **1.90**, not 1.4.
   - From the §16.4 baseline of 1.75, a 1.3× time-only cut → **1.60**.
   - Reaching **1.5 by time alone needs 1.59×** — more than the §7 win delivers.
   - And the dominant axis is **memory (6.4×)**, larger than time (2.7×). A time-only lever *cannot* cross 1.5. Crossing it requires also cutting the memory axis (6.4× → ~3×). **The flagship's binding constraint is memory, not time, and the plan currently has no memory lever for `neural_cube`.**

These three corrections are the reason this plan leads with **framework integrity (§1)** and **a shared early-stop mechanism (§3)**, and reframes the flagship (§5) around the *memory* axis, not the time axis.

---

## I. The Post-Session-5 State (what is real, what is promised, what is missing)

| Claim | Status | Evidence |
|---|---|---|
| Fair-comparison pipeline runs end-to-end | **Real** | PLAN2 §13/§16.4: `IdealBackpropFinder → RuleFrontierFinder → compare_frontiers → cost_of_plausibility → scaling_laws`, five families |
| Training loop at floor | **Real** | ~1.9 s/epoch, PLAN2 §5 |
| `peak_memory_mb` verified | **Real** | 0.96× `max_memory_allocated()`, PLAN2 §6 |
| Early-stop lever found + searchable | **Real for `eqprop`; non-existent for `neural_cube`** | PLAN2 §7 win measured on `StandardEqProp`; `NeuralCube` has no early-stop code (§0.2) |
| Knowledge layer *fed* | **Real** | `result_sink` writes probes/engine/tracks → KB→FailureTracker, PLAN2 §17/§18 |
| Knowledge layer *read* (the flywheel) | **Missing** | `AutoScientist.proposer` reads the component *registry*, not frontiers/`cost_of_plausibility`; the read-half is unbuilt (§4) |
| Substrate-faithful measurement | **Mechanism built, not exercised** | `target_hardware` + facades + `_hw{t}` cache identity (PLAN2 §18); no real substrate probe run |
| Flagship `neural_cube ≤ 1.5` | **Not yet meaningful** | rests on (a) a fictional space, (b) a non-existent lever, (c) arithmetic that needs a *memory* lever |
| Cache-identity discipline | **Real (the crown jewel)** | epochs (§16.3) *and* `target_hardware` (§18) are cache-identity covariates; a GPU frontier can never be silently reused as an FPGA reference |

The plan below is built on this honest table. Every "P" item names which Status row it converts to **Real**.

---

## P0 — Framework Integrity Gate (mandatory, blocks all measurement)

> **Gate rule: no probe budget is spent on a family whose `RULE_SPACES` entry does not match its constructor signature, and whose settle path does not route through the shared early-stop utility.** Running P1..P3 before P0 passes is exactly the failure mode that produced the fictional `neural_cube` numbers — optimizing no-op dimensions and reporting a lever that doesn't exist.

### P0a — Enforce the `RULE_SPACES` ↔ constructor contract (the §12 lesson, automated)

Right now `build_model_kwargs` *silently drops* dimensions a model doesn't accept (no `**kwargs`) — so a space can advertise phantom knobs and the search never knows. This must fail loudly.

- **Action:** add a `validate_rule_space(rule)` (and `validate_all_rule_spaces()`) in `hyperopt/search_space.py` that, for each `RULE_SPACES` entry, reflects the registered model's `__init__` signature and raises on any advertised key that the constructor neither accepts directly nor absorbs via `**kwargs`. Call it in `RuleFrontierFinder.find()` (and `IdealBackpropFinder` via the backprop space) *before* the first probe, so a mis-sized space aborts the search instead of silently degrading it.

**Implementation guardrails (per AGENTS.md):**
- Signature: `def validate_rule_space(rule: str) -> None` / `def validate_all_rule_spaces() -> None` — no `Any`, use `object` or generics.
- Use `inspect.signature` + `Registry.get(ComponentCategory.MODEL, name)` — reflection is localized, not at module level.
- Raise a custom exception `SpaceSignatureMismatchError(rule: str, phantoms: frozenset[str])` — define a small domain hierarchy per AGENTS.md §44; chain from the original `ValueError`.
- Type hints: `RULE_SPACES: dict[str, dict[str, tuple[float, float, str] | list[object]]]` — built-in generics (`dict`, `list`, `tuple`), no `Dict`/`List`/`Tuple` imports.
- Validator is a **pure function** — no I/O, no module-level side effects (AGENTS.md §39). Called explicitly in `RuleFrontierFinder.find()` before the first `study.optimize()`.
- Unit test with `hypothesis` (AGENTS.md §49): property "every advertised key is accepted by the model's `__init__` or absorbed by `**kwargs`" — generate arbitrary space dicts, assert validator passes iff the property holds.

- **Fix `neural_cube` immediately:** drop `hidden_dim`, `damping`, `tol` from `RULE_SPACES["neural_cube"]` until each is implemented in the model; keep `{lr, weight_decay, cube_size, max_steps}` (the honest space). Re-add each dimension in the same PR that implements it in the model.

- **Success:** `validate_all_rule_spaces()` exits 0; the `neural_cube` effective search space equals its constructor surface. A future contributor adding a phantom knob gets a CI failure (`ruff` + `pyright` + `pytest --cov`), not a quiet probe waste.

This is the single highest-leverage engineering step: it is cheap, it retroactively tells you which prior numbers are trustworthy, and it *prevents the §12 bug from ever recurring on any future family*. The framework protects the next 50 algorithms, not just this one.

### P0b — Re-derive the `neural_cube` 5-epoch frontier on the *honest* space

- **Action:** with P0a's corrected space, run `RuleFrontierFinder(rule="neural_cube", epochs=5, budget=~30)` `force=True`. Small budget — the question is *not* the cost number, it is **whether "standout at 1.75" survives once the search is honest.** Record via `result_sink`.

- **Read-out, both branches:**
  - If 1.75 (or better) **holds** → the flagship is real; proceed to P3 with confidence.
  - If it **collapses** → the flagship changes *before* spending the big budget. Either way the answer is cheaper than the wrong bet.

---

## P1 — One settle primitive, shared by every equilibrium rule (the framework win)

The user's directive: *don't over-invest in any one algorithm; share early-stop functionality across the family via a mixin or other OOP strategy.* This item is the framework embodiment of that directive, and it is also what makes the §7 lever real for `neural_cube` *and* every settle-based sibling (`DeepEP`, `SparseEquilibrium`, `LazyEqProp`, `HolomorphicEP`, …) at once.

### The situation (already half-built)

The early-stop machinery already exists in **one** place: `zoo/_settling.py:344 settle_activations_list(..., convergence_threshold=1e-3, convergence_start=5)`, and the implicit backward `EquilibriumFunction` already does `getattr(model, "convergence_start", 5)` / `getattr(model, "convergence_threshold", 2e-4)` (`_settling.py:468-469`). So the convergence knobs are **duck-typed off the model today.** The only reason `neural_cube` doesn't get the §7 win is that `NeuralCube.forward` bypasses `settle_activations_list` and hand-rolls `for _ in range(steps)`. The lever isn't missing — it's *un-routed*.

### The design — `EquilibriumSettleProtocol` + `settle_state` helper (composition over inheritance)

Per AGENTS.md §28: *Composition over inheritance. Favor small pure functions. Isolate side effects so core logic stays testable.* A mixin (inheritance) is the wrong default; we export a **Protocol + a pure helper** that any model can compose.

- **Action:** in `zoo/_settling.py`, define:
  1. `EquilibriumSettleProtocol` (AGENTS.md §15: `Protocol` over ABCs) — the structural requirement:
     ```python
     class EquilibriumSettleProtocol(Protocol):
         convergence_threshold: float
         convergence_start: int
         max_steps: int
         W_in: nn.Module
         W_rec: nn.Module
         def _transform_input(self, x: torch.Tensor) -> torch.Tensor: ...
         def _forward_step_impl(self, h: torch.Tensor, x_proj: torch.Tensor) -> torch.Tensor: ...
     ```
  2. A **pure function** `settle_state(model: EquilibriumSettleProtocol, x: torch.Tensor, *, steps: int | None = None, return_trajectory: bool = False) -> tuple[torch.Tensor, int, bool]` that routes through `settle_activations_list`, returns `(h_final, steps_taken, converged)`. No mutation, no side effects (AGENTS.md §29: *Immutability — default to immutable structures*).
  3. The convergence knobs are **data attributes** on the model (`convergence_threshold: float = 1e-3`, `convergence_start: int = 5`) — defined in the model's `__init__` and forwarded via `**kwargs` by siblings that already absorb unknowns.

- **`NeuralCube`** adopts the protocol by:
  1. Adding `convergence_threshold: float = 1e-3` and `convergence_start: int = 5` to its `__init__` (forwarded via `**kwargs` if needed — but `NeuralCube` has no `**kwargs`, so explicit).
  2. Replacing its hand-rolled `for _ in range(steps)` (`neural_cube.py:175`) with a call to `settle_state(self, x_proj)` — **one line change**.
  3. Exposing `steps_taken`/`converged` as metrics so the probe driver surfaces "did early-stop fire".

- **Searchability:** `convergence_threshold ∈ [1e-4, 1e-2]` (log) and `convergence_start ∈ [2, 10]` (int) become searchable for *any* protocol-adopting rule — added to `RULE_SPACES["neural_cube"]` in the same PR that lands the model change (P0a discipline: space and constructor move together, never one ahead of the other).

### Why this is the framework win, not a `neural_cube` patch

- It **captures the §7 win once, for the whole family** — the exact opposite of a vertical `neural_cube` investment.
- It makes "settle cost" a **measured, comparable quantity across all equilibrium rules** (`steps_taken`), turning a per-algorithm fight into a framework-level observation.
- It is the only way the §7 lever ever reaches `neural_cube` *correctly* — the alternative (hand-porting the loop into `NeuralCube.forward`) would re-entrench the duplication that produced the §0.1/§0.2 bugs in the first place.
- **No `Any`** (AGENTS.md §21): the protocol and helper are fully typed; `settle_state` takes `EquilibriumSettleProtocol`, not `object` or `nn.Module`.

**Success:** `NeuralCube` settles via the shared primitive; `convergence_threshold`/`convergence_start` are real constructor args and searchable; a unit test (pytest + hypothesis) asserts every protocol-adopting model's `forward` *actually* early-stops (a `convergence_threshold=1.0` model terminates in `< max_steps` with `converged=True`). The §7 win is no longer an `eqprop`-only result.

---

## P2 — Connect the flywheel's read-half (the build, not the verify)

> *Honest framing: Session-5 wired the **write** half (`experiments → KnowledgeBase`); the **read** half (`KnowledgeBase → better AutoScientist hypotheses`) is unbuilt. The "moat compounds" claim is true for storage, unproven for retrieval. P2 is the connect step — without it, P6 of the prior draft ("test the flywheel") is a verify of a thing that doesn't exist.*

- **Fact:** `autoscientist/proposer.py::ExperimentProposer` reads the component *registry* (`Registry.query(MODEL/PROPAGATOR)`) and generates hypotheses from model/propagator names — it never reads a `cost_of_plausibility`, a `pareto_frontier`, or a `scaling_law`. The frontier pipeline and the AutoScientist are parallel rails today.

- **Action:** add a `ConditionalQuery` surface to the AutoScientist that, given `(task, accuracy_target, memory_cap, FLOPs_cap, substrate)`, returns prior verified operating points from the KB matching those constraints, and a proposer branch that **seeds its next batch from frontier gaps** — "no verified point exists within `(±acc, ×flops)` of this candidate → it is worth a probe." This closes PLAN2 §8's loop for real: *constraints → query conditionals → targeted experiment → write back*.

**Implementation guardrails (per AGENTS.md):**
- New type `ConditionalQuery` as `TypedDict` (AGENTS.md §20: *TypedDict for unvalidated structured dicts*) for the query shape; `ConditionalResult` as `@dataclass(frozen=True, slots=True)` for the response.
- `KnowledgeBase.query_conditionals(query: ConditionalQuery) -> list[ConditionalResult]` — the new public method. Uses Pydantic v2 (AGENTS.md §20) for runtime validation at the I/O boundary (the method boundary).
- Errors: custom `ConditionalQueryError` hierarchy (AGENTS.md §44) — always chain (`raise ... from exc`). No bare `except:`.
- Logging: use **t-strings** (PEP 750, AGENTS.md §42) for safe deferred interpolation — never f-strings for DB queries or untrusted inputs.
- Resources: DB connections via context managers (`with sqlite3.connect(...) as conn:`) — AGENTS.md §45.
- Proposer branch: dependency injection (AGENTS.md §50: *Prefer Dependency Injection over `unittest.mock`*) — pass the `ConditionalQuery` service to the proposer constructor, not a global.
- Tests: pytest + fixtures (AGENTS.md §51) + `hypothesis` for property "given a known KB state, proposer seeds from the frontier gap" — AGENTS.md §49.

- **Sequencing:** after P0/P1 (so the conditionals it queries are honest) and **before** the heavy P3 hardening (so the hardening benefits from, and demonstrably feeds, the loop).

- **Success (the real P6 test, made possible by this):** a demonstration that the AutoScientist *reads* a prior conditional and skips a probe that an earlier run already characterized — the first measured RPM of the flywheel, not just a claim that data went in.

This is the item the prior draft under-rated. The sink is a reservoir; P2 installs the turbine.

---

## P3 — The flagship, honestly scoped (memory-bound, CI-backed)

> *Reframed from the prior draft. Crossing `cost ≤ 1.5` is **not** reachable from 1.75 via a 1.3× time lever (§0.3). The binding axis is **memory (6.4×)**. So P3 leads with a memory lever, then the time lever from P1, then powered CIs — and the target is held honestly: it may land at ~1.6, which is "one lever from viable," still a flagship.*

### P3a — A real memory lever for `neural_cube` (the binding axis)

- **Reality:** `NeuralCube`'s activation memory scales as `cube_size**3` per step (`n_neurons = cube_size**3`). The 6.4× memory is structural to the architecture, not a training-loop overhead. Optimization candidates (each a **controlled experiment**, PLAN2 §7 discipline — implement/measure/compare/keep-or-revert):
  - **`cube_size` search** at fixed `hidden`-equivalent capacity: smaller cubes cost less memory; does the frontier already find this once the search is honest (P0)?
  - **Activation checkpointing on the settle loop** (PLAN2 §2): ~2× compute, ~O(√steps) memory — directly attacks the per-step activation footprint. This is a *framework* optimization (applies to any protocol-adopting rule), not a `neural_cube` patch.
  - **Cube precision / sparsity**: the substrate-faithful quantization from P5 doubles as a memory lever on substrate.

- **Honest target:** a memory cut of 6.4× → ~3× *combined* with P1's time 1.3× lands `(0.52·3.0·2.08)^(1/3) ≈ 1.35` on the §13 profile — i.e., **viable** — *if* both levers hold at negligible accuracy cost. The plan does not assert they will; it asserts this is the only axis-combination that can reach the threshold, so it is where the budget goes.

### P3b — Powered CIs, concentrated (not a spray)

- **Action:** raise `budget_probes` toward 500–1000 (PLAN2 §15.4), but **concentrated on `neural_cube` + the matched backprop reference first** — not spread across five families (that guarantees thin CIs everywhere). The other four families stay at the exploratory budget until the flagship is defensible.

- **Reported, finally:** `cost_of_plausibility` with 95% CI; `scaling_law` r² + CI for FLOPs-to-accuracy. A powered flagship table is the first artifact that could go in front of a design partner.

### P3c — The flagship is *secondary to the framework* — by construction

- The point of P3 is to produce *one defensible conditional*, but it is run on infrastructure P0–P2 made honest. If `neural_cube` is later de-prioritized, **none of P0/P1/P2/P3a's work is wasted** — the integrity gate, the shared settle primitive, the flywheel read-half, and the memory levers all serve the *next* algorithm too. That is the framework-first posture the plan requires.⊕

---

## P4 — The substrate test (de-risk #2, executed)

> *The digital-GPU fallacy is the biggest correctness threat (PLAN2 §18); the mitigation is built but unexercised. P4 runs it — and the test is not "does the knob work" (tested) but "does the substrate change the ranking."*

- **Action:** `scripts/preliminary_run.py --target-hardware fpga` and `--target-hardware analog` on MNIST, for backprop + `neural_cube` (+ `eqprop` as the stress case), small budget. The cache split (`_hw{target}`) is already keyed, so GPU vs FPGA runs can't cross-reuse (PLAN2 §18).

- **The honest caveat made explicit (not buried):** the facades keep **float gradients** (a *surrogate-accumulation assumption*) while quantizing/noising only the forward. Before any substrate-faithfulness *claim*, sanity-check that this assumption does not materially distort the frontier vs. a true low-precision backward. If it does, the facade is a *visualization*, not a *measurement* — and the claim collapses to "we have a knob." Report this regardless of outcome; a negative result here is moat (PLAN2 §7 reverted-as-evidence precedent).

- **Success:** a real fpga/analog `cost_of_plausibility` table, and a recorded answer to "does substrate change the ranking" (if "no change on MNIST," that's the expected toy result — the question matters at scale, P5).

---

## P5 — Scale (CIFAR-10) + hardware × equilibrium cross-terms

> *MNIST is a toy where memory/compute never bind — which is *why* the binding axis was misread in §0.3. At CIFAR-10 scale both bind, and the answer to §1's real question ("which rule dominates *under which constraint*") becomes non-trivial.*

- **P5a — CIFAR-10 via multi-fidelity:** exercise `HyperbandPruner` end-to-end (PLAN2 §15.6; infra wired, never run). Fidelity = epochs / dataset fraction. Success: first CIFAR-10 `cost_of_plausibility`; the multi-fidelity path is proven, not just wired.

- **P5b — Hardware × equilibrium cross-terms (PLAN2 §18 "not-yet"):** on a substrate (`fpga`), does `convergence_threshold` search behave the same, or does quantization change where early-stop pays? Record the interaction as a searchable config (via the P1 protocol — the cross-term is *free* to expose once the protocol lands). This is the actual product differentiation: substrate-specific optimization, not a GPU-parity number.

---

## P6 — Mass, and the demonstrated flywheel

- **Action:** let P3–P5 accumulate through `result_sink`; with P2's read-half in place, run the loop and **measure** that the AutoScientist proposes with fewer probes because it read a prior conditional.

- **Success:** non-trivial KB counts **and** one engineered demonstration: "the proposer read a prior verified conditional and skipped an already-characterized probe." That is the compounding asset made *measurable*, not asserted.

---

## II. Metric honesty — the parity trap, fixed at the report layer

PLAN2 §1 explicitly rejects the parity framing, yet `cost_of_plausibility` is *defined relative to backprop* ("how many more FLOPs×mem×time vs backprop at matched accuracy"). That re-centers backprop as the gold standard — the parity mindset baked into the metric. The fix is at the **report** layer (no re-measurement):

- **Primary output** (new): the **symmetric joint Pareto surface** — "is rule R on the Pareto frontier at all, at budget B?" Backprop is one point on the joint frontier, not the fixed reference. Report the *fraction of the joint frontier held by each rule* and the *crossover budgets* where domination switches.

- **Secondary output** (keep): `cost_of_plausibility` as a backprop-relative summary for the viability-threshold call (≤1.5 / ≥5). It's useful, it's just not the headline.

This is a documentation/reporting change carried by existing `pareto_frontier` / `compare_frontiers` infrastructure — cheap, high-signal, and it stops the project from publishing, as its flagship number, the metric its own §1 declared the wrong question.

---

## III. Standing discipline (carry forward from PLAN2)

- **Epochs and `target_hardware` are covariates:** match and cache independently (§16.3, §18).
- **Diagnose before judging:** low acc/cost may be an epoch-budget artifact — use `best_epoch_acc`/`acc_at_half` (§12).
- **Record wins *and* reverts** via the sink: the warm-start negative (§7.2) is moat; the adaptive-early-stop win (§7.3) is moat; a P3a failure is moat. The sink does not distinguish — write them all.
- **Never measure before P0 passes** — the fictional-space numbers are the cautionary tale.
- **Space and constructor move together** — never add a knob to `RULE_SPACES` without the model accepting it. The P0a validator enforces this; the human rule is the backup.

---

## IV. Phase ordering with explicit gates

| Phase | Gated by | Produces |
|---|---|---|
| **P0** Integrity | (entry) | trustworthy search spaces; honest `neural_cube` frontier |
| **P1** Shared settle | P0 | the §7 win reaches every equilibrium rule, once |
| **P2** Flywheel read-half | P0 | the AutoScientist reads conditionals (the turbine) |
| **P3** Flagship + CIs | P0; P1 for the time lever; P3a for the memory lever | one defensible `cost ≤ ~1.5-1.6` with CI, or an honest "~1.6, one memory lever from viable" |
| **P4** Substrate | P0 | first hardware-aware cost table + surrogate-assumption verdict |
| **P5** Scale + cross-terms | P3 (baseline) | CIFAR-10 cost; the substrate-specific product differentiation |
| **P6** Mass | P2 + P3-P5 data | the *measured* flywheel RPM |

The gates are not bureaucracy. They are the difference between "the flagship is at 1.75" (resting on a fictional space and a non-existent lever) and "the flagship is at 1.6 ± CI, via mechanisms that exist and that serve the next fifty algorithms too."

---

## Bottom line

The thesis is now a machine to run, not a plan to argue — *provided* the machine's first move is to prove its own inputs are honest. The next session's job is narrow and decisive, and it is **not** "drive `neural_cube` under 1.5":

1. **P0 — make the search spaces real.** (Hours of engineering, not a probe budget.) This single gate retroactively tells you which prior numbers to trust and protects every future family.
2. **P1 — install one shared settle primitive** so the §7 win is a framework property, not an `eqprop` quirk — and so `neural_cube` receives it *correctly*, by construction.
3. **P2 — wire the read-half of the flywheel** so the compounding claim becomes measurable.
4. *Then* P3 — the flagship, honestly memory-bound, CI-backed, and reported as a symmetric Pareto result first and a `cost_of_plausibility` second.

The infrastructure is sufficient. The risk has flipped from "can we measure" to "are the dimensions we measure real, is the lever we report wired, and does the flywheel actually turn." That is a much better problem to have — and it is solved by the gate order above, not by another layer of plumbing.

⊕ *If `neural_cube` is de-prioritized tomorrow, P0/P1/P2/P3a still compound: the integrity gate protects the next algorithm, the shared settle protocol carries the §7 win to every equilibrium rule, the read-half serves any conditional in the KB, and the memory levers (checkpointing especially) are framework-level. That invariance — not the success of any one rule — is what "the framework is the product" means in code.*