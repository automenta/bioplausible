# Deep EqProp Limitation

**Plan 8 §D5 / §B4.** Records the boundary conditions of deep Equilibrium
Propagation as of the current diagnostic evidence. This document exists to
state *what is known* — including negative results — so the codebase does not
silently re-litigate an answered question.

---

## 1. Summary of Findings

- **1-layer EqProp works**, especially the O(1)-memory implicit-equilibrium
  path. This remains a viable memory-advantage family.
- **Multi-layer vanilla EqProp does not learn** on digits/MNIST under the
  current contrastive rule. Early hidden layers receive a near-zero
  nudged/free state difference — the **vanishing contrastive signal**,
  confirmed by the B2 autopsy (slope = -1.19, R² = 0.80 at β=0.01).
- **FA / DFA, Target Propagation, and Predictive Coding do learn deep
  architectures** and are the recommended families for deep credit assignment.
- **DirectedEP (explicit output→hidden feedback) restores the deep-layer
  contrastive signal** in diagnostics (slope = +0.16, R² = 0.85 at β=0.01),
  BUT the restored signal does NOT translate to learning accuracy.
  Gate G2 FAILED: directed_ep at depth ≥ 3 achieves only 9.2% accuracy
  (random chance for 10-class digits) after 5 epochs with 3 seeds.
- **Outcome: Deep EqProp is Limited but Not Useful for deep credit
  assignment.** Both vanilla and feedback-based deep contrastive EqProp fail
  to learn. 1-layer EqProp remains viable for memory-advantage experiments.

## 2. Contrastive State Difference

EqProp's weight update is proportional to the difference between the nudged
and free phase fixed points:

```
ΔW ∝ (post_nudge · pre_nudgeᵀ − post_free · pre_freeᵀ) / β
```

For deep networks, the output nudge must propagate backward through the
settled dynamics. If the settle is contractive, that signal decays
exponentially with depth. Early hidden layers then satisfy `h⁺ ≈ h⁻`, so the
contrastive difference — the learning signal itself — vanishes:

```
h⁺ h⁺ᵀ − h⁻ h⁻ᵀ ≈ 0
```

## 3. Why "Per-Layer β" Is Not a Fix

Dividing by a smaller per-layer β changes only the *optimizer step size* of
the computed update. If the numerator (the contrastive state difference) is
zero, scaling the denominator still yields zero. A per-layer β is therefore a
**per-layer update-scale / learning-rate hack**, not a true energy-based β. It
may help empirically by amplifying whatever tiny residual signal survives, but
it does not remove the structural vanishing-signal failure.

The codebase now separates the two concepts explicitly:

| Concept | Code knob | Role |
|---|---|---|
| Global nudge | `beta` | Energy nudge in the settling dynamics |
| Per-layer update scale | `update_scale` / `update_scale_by_depth` | Optimizer multiplier, applied *after* the EqProp gradient |
| Feedback pathway | `feedback_gain` / `feedback_init_gain` | Explicit output→hidden drive in the nudged phase |
| Recurrent init | `w_rec_init` / `w_rec_gain` | Model construction |

## 4. Diagnostic Evidence

The contrastive profiler (`scripts/contrastive_profile.py`) records per-layer
nudged/free state deltas, gradient norms, and — since Plan 8 Session 3 — the
settle residual and convergence of each phase. Depth-scaling analysis
(`analyze-depths`) fits `log(early-layer delta / output delta)` vs depth: a
consistently negative slope is the vanishing-signal signature.

### B2 Autopsy Results (Session 3, pre-registered protocol)

Three-arm comparison: eqprop (vanilla), directed_ep (feedback), directed_ep
null arm (feedback_gain=0). Depths 1-4, 3 seeds, β ∈ {0.01, 0.03, 0.1},
hidden_dim=256, batch=128, lr=0.05. Slope fit on depths 2-4 only (depth-1
excluded per protocol — physically adjacent to output nudge).

| Beta | Arm | Slope | R² |
|---:|---|---:|---:|
| 0.01 | eqprop (vanilla) | **-1.19** | **0.80** |
| 0.01 | directed_ep (feedback) | +0.16 | 0.85 |
| 0.01 | null arm (fb_gain=0) | +0.24 | 0.49 |
| 0.03 | eqprop (vanilla) | -0.46 | 0.45 |
| 0.03 | directed_ep (feedback) | -0.04 | 0.20 |
| 0.03 | null arm (fb_gain=0) | -0.19 | 0.37 |
| 0.1 | eqprop (vanilla) | +0.07 | 0.05 |
| 0.1 | directed_ep (feedback) | +0.02 | 0.01 |
| 0.1 | null arm (fb_gain=0) | +0.12 | 0.25 |

**Key finding**: At β=0.01, vanilla eqprop shows strong vanishing signal
(slope=-1.19, R²=0.80). Feedback (directed_ep) retains the signal
(slope=+0.16). The null arm (fb_gain=0) does not cleanly reproduce eqprop
because DirectedEP's extra feedback layers consume RNG state, producing
different forward-layer initialization.

### Gate G2 Results (Session 3)

Pre-registered G2 command: directed_ep with feedback_gain=0.5,
w_rec_init=xavier, depth ∈ {3,4}, 3 seeds, 5 epochs, digits.

| Model | Depth | Seeds | Mean Accuracy | Threshold | Result |
|---|---:|---:|---:|---:|---|
| directed_ep (feedback) | 3-4 | 6 | **9.2%** | >50% | **FAIL** |
| eqprop (control) | 3-4 | 6 | 10.9% | — | (also random) |

**Verdict**: Gate G2 FAILED. Feedback restores the diagnostic signal (B2
confirmed) but does NOT translate to learning. The deep contrastive EqProp
path — vanilla or feedback — does not learn on digits within 5 epochs at
depth ≥ 3. This is an honest negative result.

Run the autopsy and check the fitted slope:

```bash
uv run python scripts/contrastive_profile.py --model eqprop \
  --task digits --num-layers 4 --hidden-dim 256 --epochs 1 --device cpu
uv run python scripts/contrastive_profile.py --model directed_ep \
  --task digits --num-layers 4 --hidden-dim 256 --epochs 1 --device cpu
```

## 5. Which EqProp Variants Remain Viable

| Variant | Status | Notes |
|---|---|---|
| `eqprop` / `eqprop_mlp` at depth 1 (implicit path) | viable | O(1) memory, works |
| Deep vanilla contrastive EqProp (depth ≥ 3) | **broken** | vanishing signal (slope=-1.19); no learning (G2 fail) |
| `directed_ep` (feedback) | **experimental** | signal restored (slope=+0.16) but no learning (G2 fail: 9.2%) |
| `momentum_equilibrium` / `sparse_equilibrium` | experimental | no evidence of fixing the deep credit-assignment problem |
| `finite_nudge_ep` / `lazy_eqprop` | experimental | same engine, no mechanism change |

**Gate G2 verdict recorded**: Deep EqProp salvage via feedback is unsuccessful.
The feedback pathway restores the diagnostic contrastive signal but does not
produce competitive accuracy. Deep vanilla EqProp and feedback EqProp both
remain at random chance (~10%) on digits at depth ≥ 3 within the pre-registered
5-epoch, 3-seed budget.

## 6. Recommended Families for Deep Credit Assignment

Based on Plan 7 empirical results and the same diagnostics story:

- **FA / DFA** (`feedback_alignment`, `standard_fa`, `dfa_deep`) — strong,
  up to ~94% on digits.
- **Target Propagation** (`diff_target_prop`) — ~80%+.
- **Predictive Coding** (`fabricpc_graph_pcn`) — ~80%+.
- **Backprop** (`backprop_mlp`) — the parity baseline.

Deep vanilla EqProp should not be treated as a flagship deep learning rule
until diagnostics show sustained early-layer contrastive signal AND a
compute-matched accuracy result. The `status:*` registry tags make this
quarantine operational: known-broken models are excluded from default sweeps.

---

*Maintained by Plan 8 Track D5. Update this document when the depth-scaling
slope or Gate G2 evidence changes.*