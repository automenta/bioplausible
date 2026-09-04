# TODO.md — Active Plan: Landing-Cost Reduction (Wiring, Contracts, Environment)

> **Opened 2026-09-04.** Derived from the R11.3.12 (ePC/D12) landing
> retrospective. That landing succeeded — demo, figure, locks all green —
> but the process exposed structural friction that taxes *every* future
> ontology-primitive and demo addition. This plan removes the friction at
> its root: hand-maintained parallel lists, undocumented contracts,
> environment decay, and lint/type noise that hides real signal.
>
> **Status 2026-09-04:** R1.1–R1.3, R2.1–R2.2, R3.1, R4.1–R4.3, R5.1–R5.4
> **landed** (wiring lockstep lock + deep-linear transitions lock + demo
> gate + property suite all green; the lock caught a real pre-existing
> gap — `DiffusionDynamics` missing from root `__all__`/`_LAZY`). R6
> items remain as-touch/one-liners.
>
> Relationship to [TODO11.md](TODO11.md): TODO11's remaining pull-based
> items (R11.1.10, R11.1.11, R11.2.9/13/14/16, R11.3.4–3.11, R11.3.13,
> R11.4.1/4.3/4.4) stay pull-based under TODO11's rules. This plan is the
> **enablement track**: it makes those pulls — and any research-track pull
> that adds a primitive — cheaper and safer.

---

## 📜 Standing Directives (carried, binding)

- Test invocation is **`uv run python -m pytest`** everywhere (user-site
  pytest shadowing, protobuf gencode skew — TODO11 Watch 2026-09-04).
- `benchmark_results/` stays untracked. **README is never edited.**
- Test-execution discipline: gates once at close, probes before tests,
  output + walltime always visible.
- R11.5.6 pull rule governs what may land; everything here ends in either
  a working lock, a working demo path, or a restored verification signal —
  infrastructure justified by the landings it enables.

---

## 🎯 R1 — Single-Source Registries (kills the parallel-lists problem)

Evidence: adding one dynamics class required hand-edits to **five** export
surfaces plus a closed if/elif chain, each discovered by the next
ImportError; the missing `from_spec` branch was latent (found only by a
manual probe); a new demo requires three parallel dicts (demo test,
`_FACTORIES`, gallery-lock `EXPECTED`).

| Item | Description | Acceptance |
|------|-------------|------------|
| **R1.1** `DYNAMICS_REGISTRY` | `dict[str, type[StateDynamics]]` in `dynamics/__init__.py` — the one `dynamics_type` → class map. `factory.from_spec` becomes a single lookup. Root `_LAZY`/`__all__` and `ontology/__init__` derive from or reference the registry. | `grep` shows the class name in exactly two places (class def + registry entry) |
| **R1.2** Wiring lockstep lock | New property lock: every `StateDynamicsConfig` classmethod's `dynamics_type` ∈ registry; every registry class round-trips `to_spec`→`from_spec` (params bitwise); every registry class ∈ root `__all__`/`_LAZY`. | Lock fails if a future primitive skips any surface — the ePC failure mode becomes impossible |
| **R1.3** `DEMOS` registry | One table (name → demo test path + figure factory), consumed by `render_gallery` **and** the gallery lock's `EXPECTED`. Lives beside the lock that enforces it. | Adding demo D13 = demo test + one registry row + re-pin |
| **R1.4** `FACTORY` credit registry (optional, as-touch) | Same treatment for credit_type → class in `factory.from_spec`, only if touched again — do not pull speculatively. | — |

Gates: R1.2 lock + targeted round-trip tests + root exports; full demo gate
once at close (wiring touches every surface).

## 📐 R2 — Contracts Written Down (kills the archaeology)

Evidence: before writing `settle`, six files had to be reverse-read to learn
the activation layout `[input, hidden…, output]`, the phase/energy call
order, the `no_grad`-by-default settle context (internal `torch.enable_grad()`
pattern), and target semantics. The imp-27 mutation contract is documented;
everything else is folklore.

| Item | Description | Acceptance |
|------|-------------|------------|
| **R2.1** StateDynamics contract block | Canonical contract in the `StateDynamics` Protocol docstring: activation layout invariant (and who consumes it), phase loop + `compute_energy` timing, autograd context rules, input flattening expectation, free/nudged target semantics, metrics schema pointer (imp-46). | A new dynamics author reads one docstring, not six files |
| **R2.2** Wiring checklist | "Adding a new ontology primitive" checklist (registry entries, export surfaces, contract invariants, demo/gallery steps, probe conventions) — TODO11 Notes section or AGENTS.md, whichever the user prefers. Compiled from the ePC retro; maintained as R1 lands. | Next primitive landing follows the checklist verbatim |

## 🔧 R3 — Shared Settle Machinery (kills per-dynamics re-derivation)

Evidence: `extract_layered_params` returns weights/biases/activations as
separate tuples, losing interleaving — ePC had to re-walk modules privately
(`_transitions`), and the first version *silently degenerated on
deep-linear stacks* (the paper's own testbed). Depth-structure correctness
should be shared, not re-derived per dynamics.

| Item | Description | Acceptance |
|------|-------------|------------|
| **R3.1** `LayeredParams.transitions` | Expose the interleaved `(weight, bias, activations)` schedule from `extract_layered_params`; ePC's `_transitions` moves there. | Property lock: a deep-linear geometry (no activation modules) yields a correct transition schedule with error-injection positions |
| **R3.2** Migrate callers as-touch | Existing dynamics keep their paths; new code uses the schedule. No sweep. | — |

## 🌱 R4 — Environment Guardrails (kills silent decay)

Evidence: plain `uv sync --upgrade` stripped dev extras (three
ModuleNotFoundErrors mid-landing); user-site pytest shadow broke gRPC
collection on protobuf skew; `UV_LINK_MODE` warning noise.

| Item | Description | Acceptance |
|------|-------------|------------|
| **R4.1** AGENTS.md invocation + sync rules | Add to AGENTS.md: tests via `uv run python -m pytest`; env restore is `uv sync --dev --all-extras` (bare `uv sync` drops extras). | Next session cannot relearn this the hard way |
| **R4.2** Dev-env smoke | One-liner import check (`optuna, scipy, torchvision, pytest` from venv) added to the per-commit checklist in AGENTS.md; a stripped env fails in seconds, not mid-gate. | — |
| **R4.3** `UV_LINK_MODE=copy` | Set in `.env`/docs note to silence the hardlink warning on this filesystem layout. Cosmetic; as-touch. | — |

## 🧹 R5 — Verification-Signal Restoration (Register C adjacent, but signal-bearing)

Evidence: pyright reports ~17–27 lazy-map false positives per test file —
real errors are undistinguishable without sibling-baseline diffing;
`SystemConfig.validate()` carries a dormant contradiction (thermo credit
requires `energy_minimization` at one branch, permits `predictive_settling`
at another); the `non-augmented-assignment` rule fires on the out-of-place
adds that settle graphs *require* (5 inline markers this landing).

| Item | Description | Acceptance |
|------|-------------|------------|
| **R5.1** Typed lazy exports | Under `if TYPE_CHECKING:` in root `__init__.py`, import all `__all__` names explicitly — pyright then types every consumer correctly, zero runtime cost. R1.2's lock keeps the block in sync with `__all__`. | `pyright tests/integration/test_demo_spike_settle.py` drops to 0; new-file pyright gives signal again |
| **R5.2** Reconcile `SystemConfig.validate()` | Whitelist the PC family (`predictive_settling`, `error_predictive_coding`) consistently in the thermodynamic-credit branch. Behavior change on the `comp joint-validate` path only. | `comp joint-validate` accepts sPC/ePC+thermo coordinates; existing validation tests updated |
| **R5.3** Dynamics ruff allowance | Per-file-ignores `non-augmented-assignment` for `ontology/dynamics/_dynamics.py` with a comment stating *why* (out-of-place adds are the graph-safety idiom). Strip the now-redundant inline markers on next touch — never as a sweep. | Rule intent preserved via config; markers stop regrowing |
| **R5.4** `num_workers=0` default for quick-mode | `create_task(quick_mode=True)` defaults to 0 workers (forkserver flake mitigation, D7 precedent) — overridable. | Demo/probe paths stop crashing on the forkserver race |

## 🗂️ R6 — Small Items (as-touch or one-liners)

| Item | Trigger | Description |
|------|---------|-------------|
| **R6.1** `scripts/probes/` home | Next tuning landing | Keep throwaway probe scripts with their measured-regime numbers; docstring cites the demo they informed |
| **R6.2** `comp gallery --render-only` | Next gallery touch | Explicit flag pair instead of "omit `--run`"; the run path now exceeds a 2-min tool budget at 12 demos |
| **R6.3** Dead-settle-steps diagnostic | If a demo needs measured steps | Surface steps-to-convergence as a recorded diagnostic (the ePC `_steps_used` was cut as dead — revive only on demand) |
| **R6.4** Demo test-pattern template | Next demo landing | The ePC landing converged on a reusable shape — static `_ARMS` table at module scope, `_train_arm`/`_probe_arm` extracted against the too-many-locals rule, walltime printed but never recorded. One sentence in R2.2's checklist so the next demo starts from the pattern instead of re-deriving it |
| **R6.5** Cross-reference research findings | Done in TODO11 | The ePC tuning finding (contrastive ÷β credit caps error-parameterized dynamics on deeper stacks; revisit path: PC-native weight gradient) is a research note, not process debt — it lives in TODO11's Notes, and this plan intentionally does not duplicate it |

---

## 🔒 Sequencing & Gates

1. **R4.1/R4.2 first** (minutes) — protects every subsequent gate run.
2. **R1.1 + R1.2** — the registry and its lock; R5.1 rides the same
   lockstep mechanism (do together: one lock, one close).
3. **R1.3** (demo registry) — independent of R1.1; its own close.
4. **R2.1 + R2.2** — docs; no behavioral gates, pyright/doc review only.
5. **R3.1** — machinery; deep-linear property lock is the new evidence.
6. **R5.2** — validation reconciliation; joint-validate smoke + updated
   validation tests.
7. R5.3, R5.4, R6 — as-touch or one-liners.

Per-commit duties stay scoped (ruff + pyright on changed files + targeted
tests); the full demo gate runs at each item's close, not per edit.

## ✅ Termination Criterion

The next ontology primitive lands with: **two** wiring edits (class +
registry row), the contract read from one docstring, the wiring lockstep
lock proving every surface — and if it ships a demo, one registry row plus
a re-pin. Pyright on the new files reports signal, not lazy-map noise. No
ModuleNotFoundError appears mid-landing. Until a stranger (or the next
session) can do that, this plan is not closed.
