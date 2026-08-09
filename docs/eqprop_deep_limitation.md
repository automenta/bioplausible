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
  nudged/free state difference — the **vanishing contrastive signal**.
- **FA / DFA, Target Propagation, and Predictive Coding do learn deep
  architectures** and are the recommended families for deep credit assignment.
- **DirectedEP (explicit output→hidden feedback) restores the deep-layer
  contrastive signal** in diagnostics. Whether that signal translates to
  competitive accuracy under matched compute is an open question tracked by
  Plan 8 Gate G2.

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
| Deep vanilla contrastive EqProp (depth ≥ 3) | experimental/broken | vanishing contrastive signal |
| `directed_ep` (feedback) | experimental | feedback restores signal; accuracy pending Gate G2 |
| `momentum_equilibrium` / `sparse_equilibrium` | experimental | no evidence of fixing the deep credit-assignment problem |
| `finite_nudge_ep` / `lazy_eqprop` | experimental | same engine, no mechanism change |

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