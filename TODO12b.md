# TODO12b.md — The Defect Hunt: Challenging the Pessimistic Conclusions

> **Opened 2026-09-06.** Pre-registered before any probe ran. Purpose:
> every pessimistic conclusion in TODO12 (misses, walls, "noise floors")
> rested on measurements that could hide implementation defects. This
> plan hunts the defects FIRST; after execution we return to TODO12.md
> with corrected conclusions or change strategies. Prime directive
> unchanged: nothing is claimed that the suite does not re-show at HEAD
> — and nothing is *dismissed* that a defect could still explain.
>
> **Stance (user directive):** "I bet you can find some implementation
> defects that will overturn some of the pessimistic conclusions."
> Treat each suspicion below as guilty until proven innocent. Every test
> is CPU-cheap (seconds–minutes) unless explicitly gated.
>
> **The pessimistic verdicts on trial:** (V1) unit_rms has a
> "convergence noise floor" on easy regimes (D16); (V2) Muon explodes at
> its registered lr on ePC width cells — the crutch is dead (D18/A6);
> (V3) ψ-only adaptation is impossible at HEAD (D22); (V4) the physical
> advantage is not realized (F5); (V5) closed-form FF cannot learn
> (f5b audit); (V6) PEPITA diverges structurally — realization retired.

---

## 🔬 The Hunt (ordered by expected overturn value)

| # | Suspicion (defect hypothesis) | Why it could overturn a verdict | Pre-registered falsification test | Cost |
|---|---|---|---|---|
| **H1** | **`UnitRMSUpdate` is defective.** D16's bp×unit_rms is CHANCE at every lr 0.002–0.1 on MNIST-quick while bp×euclid (lr 0.1) reaches 0.851 — and the "noise floor" explanation does not survive scrutiny: 150 batches is *pre-convergence*, where a fixed-RMS step should behave like well-scaled SGD, not a random walk. Suspect paths: momentum-EMA warmup, per-tensor normalization discarding mean direction, interaction with the EMA init | If unit_rms has a defect, then **A1/A2/D18/A6's positive conclusions were measured with a broken rule** — ePC's width rescue might be *understated*, the "noise floor" claim dies, and the momentum-EMA family's canonical status re-baselines | (a) Analytic: unit_rms on a deterministic quadratic must converge to the minimum (batch, no noise) — a correct normalize-the-momentum rule converges; (b) log per-step cosine(step, true gradient) on the quadratic — cosine must → 1; (c) MNIST-quick bp×unit_rms with grad-norm/cos logging across the lr grid; (d) diff against a hand-written reference implementation on 3 random steps (bitwise) | CPU, minutes |
| **H2** | **Biases never train — anywhere.** `_learnable_weight_names` filters to 2-D weights (`ontology/utils/params.py:18`); credits emit pseudo-gradients for weights only. If `GradientCredit` inherits the contract, then *every* arm in the repo — bp/adam included — trains with frozen biases | "Backprop" baselines aren't full backprop; every comparison margin shifts. Possibly explains stubborn gaps (ePC LM val 21 vs bp 9.9) or chance-level cells | (a) Read `GradientCredit.compute_pseudo_gradient`'s param list; (b) unit lock: bias grads must flow under bp; (c) one D2-style MNIST cell, biases-live vs frozen, accuracy delta | CPU, minutes |
| **H3** | **Credit × dynamics zero-term matrix.** The f5b audit proved hidden nudged acts == free hidden acts under `InstantaneousDynamics` — so ANY credit consuming hidden free-vs-nudged differences is computing exact zeros when paired with it. `HomeostaticCredit` (FREE+NUDGED, no autograd) is the prime suspect; check every landed credit | A registered cell could be a **zombie** — its "learning" from an unrelated term (or from nothing). Any negative verdict measured on a zombie cell ("credit X trains poorly/walls") is invalid — the credit was never actually receiving its designed signal | Pure-arithmetic probe: for each landed credit × {instantaneous, ePC, energy-minimization} dynamics, run one train step on a 2-layer net and report per-term nonzeroness (hidden ε, goodness diffs, error buses). Cross-check `SystemConfig.validate()`'s whitelist against the matrix | CPU, seconds |
| **H4** | **D18's "Muon explodes at registered lr" is an lr confound** (the F3 lesson, unresolved for this cell). Muon is scale-invariant — a 400×-small ePC gradient cannot cause divergence; only step SIZE can. If Muon×0.1 lr is stable, "crutch dead" becomes "registered lr wrong" and the A6 map's depth/width rows re-baseline | D18's headline ("ePC trains under unit_rms while Muon explodes") would survive only as an lr-tuning statement; the crutch narrative weakens | Reproduce ONE D18 cell (ePC w64 LM, CPU demo scale) at Muon lr ×{1, 0.3, 0.1, 0.03} + ‖Δθ‖/step logging (the F3 instrument) | CPU, ~2 min/arm |
| **H5** | **F5's instrument sanity.** `measure_saved_activation_bytes` counts pack events where `requires_grad`; if optimizer/state tensors or non-leaf intermediates pack differently across arms, the ratios skew | F5's pinned miss is instrument-bound; a skewed counter would mislabel the memory story | Unit lock: packed-tensor COUNT equals the analytic expectation (depth-scaled) for bp on a flat MLP; assert local-ff count == bp count shape; assert thermo count == 0 with the pack hook active during the UPDATE too | CPU, seconds |
| **H6** | **PEPITA's weight-trajectory channel** (the parked cheap probe — promoted here). ‖W_out‖ grows ~1e4 in ~600 steps even with unit-normalized steps and lr-bounded displacement — arithmetically impossible through the update path (600 × 3e-4 ≈ 0.18). The growth must enter through a non-step path: B-matrix EMA (feedback_lr 0.5 — 1600× the training lr!), settle-internal state, or a misrouted gradient (name↔acts index alignment) | If the growth is the learned-B EMA at feedback_lr 0.5, PEPITA's "structural divergence" was partly a **hyperparameter defect** (feedback_lr 1600× train lr) — the five-cause audit missed the sixth knob. The realization could be un-retired with sane feedback_lr | Probe: track ‖W_out‖, ‖B‖, per-name grad norms, and grad-routing alignment (autograd reference vs emitted pseudo-grads, cosine per name) per step for 200 steps at feedback_lr ∈ {0.5, 0.01} | CPU, ~5 min |
| **H7** | **D22's routing-mode flip.** With θ frozen, `RoutingPlasticity.step` took the EVAL branch (hard top-k) for the entire ψ-only "training" — the Gumbel-softmax training path never ran. modulate reads gate_logits only, so the verdict *should* be unaffected — verify, don't assume | If the flip mattered (it shouldn't), D22's miss re-opens | 10-line probe: re-run 200 stage-B episodes with the training branch forced (monkeypatch `is_training=True`); compare gate trajectories + B accuracy | CPU, seconds |
| **H8** | **`GradientCredit`'s loss is the CLAMPED-output CE.** The pipeline computes NUDGED loss on `out + β(onehot − out)` (β=0.5 default) — bp's gradient carries a (1−β) scale AND the loss surface is target-blended. Every "bp" baseline is *β-distorted backprop* | All bp-vs-local margins carry a shared distortion; probably benign but unquantified | Read + one probe: CE(clamped) vs CE(free) gradient cosine and scale at β ∈ {0.1, 0.5, 1.0}; report whether bp's effective loss is materially different from plain CE | CPU, seconds |

---

## 📋 What gets re-evaluated if each lands CONFIRMED

- **H1 confirmed →** V1 dies; re-measure D16's unit_rms row + A2/D18 ePC cells with the fixed rule (expect: unit_rms better than pinned); A6 map's canonical-family claim re-baselines.
- **H2 confirmed →** every margin in TODO12/D-tables re-quoted with the bias caveat; bp baselines re-run once at demo scale to size the shift.
- **H3 confirmed (zombie cell found) →** that cell's negative verdict is voided and re-run under a compatible dynamics; the compatibility matrix becomes a standing unit lock.
- **H4 confirmed →** D18/A6 re-pinned with matched-step Muon columns (P3 protocol); "crutch dead" scoped to lr-matched footing.
- **H5 confirmed →** F5 record re-measured (cheap); miss verdict survives or flips on the fixed instrument.
- **H6 confirmed →** PEPITA un-parking review with feedback_lr sane; the five-cause audit gains a sixth entry; possibly a live PEPITA arm returns to the reel.
- **H7 confirmed (matters) →** D22 re-run with forced training branch (still expected: no loss signal — the contract is the finding).
- **H8 confirmed →** one-line caveat on all bp baselines; no re-runs unless the cosine shows material distortion.

## 📋 What survives no matter what

- The f5b structural fact (nudged==free hidden acts under instantaneous
  settle) — code-read + measured, not defect-dependent.
- F5's Claim A scope audit wording (ff_hybrid = output-pseudo-loss
  backprop at HEAD) — follows from the same fact.
- D14's faithful-regime result, F4's mechanism cells, D5's bitwise θ.
- D22's contract finding (no ψ law *receives* a loss term — visible in
  the `step` signature itself).

---

## 🎯 Execution Order & Gates

1. H1 (unit_rms) — highest overturn value, pure CPU.
2. H2 (biases) + H3 (zombie matrix) + H8 (clamped CE) — one session,
   reading + seconds-scale probes.
3. H5 (instrument lock) + H7 (routing flip) — trivial locks.
4. H6 (PEPITA tracker) — the parked probe, promoted.
5. H4 (Muon lr) — CPU at demo scale, last (needs the D18 harness).

**Gate after each:** probe output + walltime visible → ruff format/check
→ findings appended to the verdict column → CONFIRMED defects get a
unit lock so they can never silently regress.

## 🏁 Completion Criteria for TODO12b

1. Every H# has a recorded verdict (confirmed/refuted/inconclusive)
   with measured numbers in the probe docstring.
2. CONFIRMED defects: unit lock landed + affected TODO12 claims listed
   for re-evaluation (re-runs only where the verdict actually flips).
3. REFUTED suspicions: the pessimistic conclusion upgrades to
   "defect-audited" status — no longer merely asserted.
4. TODO12 gets one consolidated revision with the corrected picture
   (or the strategy pivot, if the corrections are big enough).

**Budget:** everything CPU and minutes-scale; nothing here needs the
GPU or long runs (H4's D18 cell reproduction is CPU demo scale by the
device policy).
