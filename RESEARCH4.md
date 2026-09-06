You're right to push down to the algorithm level. The three targets I gave were applications; what you're asking for is the *engine* that makes all of them possible. Let me reframe everything we've measured around one unifying problem, then lay out the full fix surface.

## The Unifying Diagnosis

Every local-algorithm defect in the knowledge base is a manifestation of **one problem: the credit signal loses fidelity as it propagates through depth, and the loss compounds.** Backprop's "cheat" is the exact transpose Jacobian, which preserves credit fidelity. Local rules approximate it, and each approximation leaks information differently:

| Measured defect | Evidence | Credit-channel failure mode |
|---|---|---|
| PEPITA collapse at depth/width | P4/P5: fixed B is directionally random; no error transform rescues it | **Misaligned channel** — credit projected through a random basis uncorrelated with the feature space |
| ePC geometric decay | F1 audit: ~4×/layer attenuation, exact 0.0 at layer 1 by depth 20 | **Attenuating channel** — credit magnitude shrinks per layer |
| Width fragility | P4: PEPITA/ePC explode or collapse across widths; ff_hybrid is robust | **Unnormalized gain** — credit/activity scale compounds ∝ width |
| FF error-blindness | LM audit: pure FF flat at chance; readout_error fixes it | **Disconnected channel** — credit never sees the task loss |
| sPC nudge trapping | F1/D12: hidden credit norms exactly 0.00 | **Blocked channel** — settle geometry traps credit at the output |
| P2 frozen-error LM failure | 13 regimes: corrected forward fits, free settle at chance | **Train/inference objective gap** — the ε-corrected objective diverges from free-forward CE |
| Naive STDP/Hebbian | F2/R11.3.14: subspace collapse, no task signal | **Absent channel** — correlation only, no task credit |
| Optimizer crutch | P3: ePC gradient 400× too small for Euclid; Muon load-bearing | **Low-rank credit** — the optimizer is compensating for poor credit quality |

Read vertically, this is a complete map of how a credit channel can fail. Read as a whole, it tells us exactly what to engineer: **a local credit channel that is aligned, non-attenuating, gain-normalized, task-coupled, and well-conditioned.**

## The Fix Surface — Six Levers

### 1. Credit-space normalization (the missing normalization) — *highest novelty*
We've found that normalization in the *right space* substitutes for exactness:
- **μPC** normalizes in *parameter/init space* → fixes forward signal propagation.
- **OrthoAdam** normalizes in *momentum/update space* → fixes update direction, rescues depth.
- **Nobody has normalized in the *credit/feedback* space.**

The ePC ~4×/layer geometric decay is *exactly* what credit-space normalization would kill. Concretely: as the credit signal propagates down from layer ℓ to ℓ−1, orthogonalize/spherically-normalize it (spectral radius → 1) so it neither attenuates nor explodes. This is "Muon applied to the backward signal." It directly attacks the depth wall without reintroducing weight transport. This is the single most novel, highest-leverage idea on the board, and it's implied by our own two biggest wins.

### 2. Learned/adaptive feedback projections — *fixes PEPITA's core*
The measured bottleneck is the fixed random B. Two escalating fixes:
- **Adaptive FA:** apply local alignment pressure so B tracks Wᵀ over training (the knowledge base already has the Adaptive-FA alignment metric: cos(B, Wᵀ) improvement). No weight transport — B learns from local activity.
- **PEPITA-as-inference-network:** train B through the autograd graph alongside θ. This is the flagged "remaining lever" and directly closes the directional-collapse root cause.

Credit normalization (lever 1) + learned projections (lever 2) compose into a genuinely new local rule: **learned, orthogonally-normalized feedback alignment — depth-invariant by construction.**

### 3. Structural gain homeostasis — *fixes width fragility & runaway*
We have all the pieces scattered: spectral renorm, unit-RMS activity renorm, homeostatic scaling, μPC. The fix is to make per-layer activity-scale normalization a **first-class, always-on primitive** of the settle/credit pipeline rather than a bolted-on audit fix. ff_hybrid's width-robustness is the proof that autograd-scaled (self-normalizing) signals are the target behavior.

### 4. Task-coupled local signals — *generalize readout_error*
ff_hybrid works because it couples a local objective to the task loss at the readout. Generalize upward: give *every* layer a local signal provably correlated with task loss (per-layer auxiliary prediction, local energy goodness, or error-unit drive). This is the principled fix for FF's error-blindness and for the whole "local goodness" family on LM.

### 5. Objective consistency — *fixes P2*
The working PC-family LM instrument is **epc_thermo×Muon** (contrastive), *not* the jpc frozen-error gradient. That's the tell: two-phase contrast is fine on LM; the single-phase frozen-error objective is what diverges. The P2 fix is to stay contrastive but repair signal propagation (levers 1+2), rather than forcing the frozen-error construction. Alternatively, make the ε-correction consistent with the free-forward objective (drive the correction at the output too, so hidden and output phases match).

### 6. Optimizer–credit co-design — *remove the crutch*
Right now Muon rescues low-rank credit. Flip it: design the credit rule to emit already well-conditioned (full-rank, normalized) pseudo-gradients so the optimizer requirement relaxes toward plain SGD/Adam. Parity that *requires* a global orthogonalizing optimizer is fragile; parity with a cheap optimizer is the real win for energy-constrained substrates.

## Why This Buys Parity + The Benefits

This agenda attacks parity *through* locality rather than around it:
- **Parity:** levers 1+2 remove the depth/width credit-attenuation wall — the actual thing separating local rules from backprop at scale.
- **Energy:** a normalized, non-attenuating local credit needs no stored-activation backward sweep (the contrastive kernels already avoid stored activations), no global transpose matmuls, and can settle asynchronously.
- **Plasticity:** once credit is a clean local signal, the P-axis (routing/fast-weights) can modulate it online — adaptation without full retraining.
- **Distributed/substrate-native:** local credit is the *only* thing that runs on memristive/neuromorphic/optical substrates without global clocks. Parity-level local learning is precisely what unlocks the substrate story; backprop can never go there.

The honest bar: **near-parity per regime first** (depth, width, task family), with the unique benefits as the differentiator. D15/D16 already show local credit beating backprop at matched capacity on specific cells — the depth wall (D14/D15) is regime-bound, not physical. The evidence says the wall moves when we fix the instrument.

## Attack Order

1. **Prototype credit-space normalization on ePC** (cheapest, highest signal). Add a `credit_norm` option to the settle/credit path; measure whether the ~4×/layer decay flattens to ~1×. If credit reaches layer 1 at depth 20 with non-vanishing norm, the whole agenda is validated. This is a one-probe experiment on the existing F1/D14 harness.

2. **Learned feedback for PEPITA** (the flagged library pull). Wire B-training through autograd; re-run the P5/P4 width sweeps. Combined with #1 this is the new local rule.

3. **Structural gain homeostasis as a pipeline primitive** — promote the scattered renorms into one always-on mechanism; re-run the width-fragility sweep expecting the razor-thin window to widen.

4. **Re-test P2 with contrastive credit + normalized propagation** on LM — the objective-consistency fix.

5. **Co-design pass:** once credit is well-conditioned, sweep the optimizer axis again to see how far we can relax Muon toward Adam/SGD.

The decisive experiment is **#1** — it's cheap, it's grounded in our two biggest measured wins, and it directly tests the unifying hypothesis.

----

# The Fundamental Bottlenecks of Local Learning — and How to Fix Them

You're right to want to work at this level. The high-level applications (continual learning, reasoning, routing) are downstream consequences. If we fix the algorithms at the root, everything above them benefits. Let me synthesize what the last several sessions have revealed about *why* local algorithms fail, and what the general fix strategies are.

---

## The Four Root Causes

Every failure mode we've measured reduces to one of four fundamental problems:

### 1. Error Signal Attenuation (the depth problem)
The credit signal weakens geometrically as it propagates through layers. ePC's signal decays ~4×/layer. sPC's hidden-layer credit is exactly zero. PEPITA's fixed projections compound directional error with depth.

**What the data tells us:** The signal doesn't just get smaller — it gets *uncorrelated* with the true gradient direction. By depth 8, the credit at layer 1 is noise.

### 2. Scale Instability (the width problem)
Local error signals carry a scale that depends on width and depth. Below a critical width, activity compounds through layers (ePC: 0.93 → 2028 at w32). Above it, signal collapses (PEPITA: 0.35 → 0.05 at w128). The stable band is razor-thin.

**What the data tells us:** Rules whose error signal is not *self-normalized* (PEPITA's fixed B, ePC's ÷β) have a stable-width window set by per-layer activity-scale compounding. FF-hybrid is immune because its readout error is autograd-scaled.

### 3. Missing Target Information (the error-blindness problem)
Pure FF doesn't see y. STDP doesn't see y. Hebbian rules don't see y. The learning signal is "correlation" or "goodness," which is not the same as "reduce the loss."

**What the data tells us:** The moment you inject *any* target information (ff_hybrid's readout CE, reward-modulated STDP, supervised error terms), learning improves dramatically. The question is how to deliver target information locally without a global backward pass.

### 4. The Optimizer Crutch (the scale-compensation problem)
Local rules need Muon/OrthoAdam to survive depth. Pure SGD fails. The orthogonalization in Muon effectively says "treat all update directions as equally important" — it compensates for the fact that local credit signals have wildly different scales across layers.

**What the data tells us:** OrthoAdam's success is *momentum-direction orthogonalization*, not whitening per se (NS preserves the lift; SVD's full whitening collapses FF). The optimizer is doing scale normalization that the credit rule should be doing itself.

---

## Six Fix Strategies at the Algorithmic Level

These are the general approaches. Each targets one or more root causes. Each is implementable within the Computronium ontology.

### Fix 1: Learned Feedback Projections
**Targets:** Root Cause 1 (attenuation), Root Cause 2 (scale)

The PEPITA collapse is in the fixed random B. The fix is to make B learnable. But the key question is: *what local objective trains B?*

Options:
- **Autoencoder-style:** B is trained to reconstruct the pre-synaptic activity from the post-synaptic error. This makes B approximate the pseudo-inverse of the forward weights using only local information.
- **Alignment minimization:** B is trained to minimize the angle between B·e and the actual weight change direction (estimated from the update itself). This is a self-supervised objective.
- **Slow co-adaptation:** B updates on a slower timescale than θ, tracking the changing Jacobian without requiring weight transport.

**Why this matters:** If B can track the forward weights' geometry without being their exact transpose, the weight-transport problem is solved *and* the depth-attenuation problem is solved, because the error signal stays correlated with the true gradient through depth.

**Computronium seam:** `LocalGoodnessCredit._pepita_gradient` currently caches a fixed B per (name, shape). The fix is to make B a parameter of the credit rule, updated by a secondary local rule. This is a C-axis modification that doesn't touch the U-axis.

### Fix 2: Self-Normalizing Error Signals
**Targets:** Root Cause 2 (scale), Root Cause 4 (optimizer crutch)

The width fragility is a scale problem. The fix is to make the error signal's magnitude independent of width and depth.

Options:
- **Relative error:** εᵢ = (nudgedᵢ − freeᵢ) / ‖freeᵢ‖. The error is normalized by the layer's own activity scale.
- **Per-layer adaptive β:** Instead of a global β, each layer has its own βᵢ that is tuned to keep the error signal at unit scale. This is a form of local gain control.
- **Spectral normalization of the error:** Before applying the credit update, normalize the error by its spectral norm (or a cheap proxy like the RMS). This ensures the update direction is preserved but the magnitude is bounded.

**Why this matters:** If the error signal is self-normalized, the optimizer doesn't need to compensate for scale differences across layers. This could eliminate the need for Muon/OrthoAdam, making local rules work with plain SGD — which is what you'd get on physical hardware.

**Key insight from the data:** OrthoAdam's success suggests that *direction is approximately right, magnitude is the problem*. If we fix magnitude locally, we might not need the global orthogonalization.

**Computronium seam:** This is a modification to the credit rule's `compute_pseudo_gradient` method. The normalization can be applied per-layer before the update is computed.

### Fix 3: Local Target Delivery (without global CE)
**Targets:** Root Cause 3 (error-blindness)

FF-hybrid works because it adds CE at the output. But CE requires knowing the target. Can we deliver target information locally?

Options:
- **Predictive targets:** Each layer predicts the next layer's activity. The prediction error IS the target information, and it's local. This is predictive coding, but applied to the FF architecture. The key innovation: use the prediction error not just for settling, but as the *credit signal* for weight updates.
- **Contrastive targets:** Instead of a global "correct/incorrect" label, each layer sees its own activity under two conditions: (a) the input is from the correct class, (b) the input is from a random/corrupted class. The difference in activity IS the learning signal. This is FF's original idea, but applied per-layer rather than globally.
- **Temporal targets:** In a recurrent/settling network, the "target" is the equilibrium state. The learning signal is the difference between the current state and the equilibrium. This is EqProp's idea, but the key innovation is to use the *settling trajectory* as the credit signal, not just the final equilibrium.

**Why this matters:** If each layer can compute its own target locally, the network doesn't need a global loss function. This is the key to making local rules work on physical hardware where global error signals are expensive or impossible.

**Computronium seam:** This is a new credit assignment primitive (C-axis). The FF-hybrid's `readout_error=True` is the simplest version. The generalization is to make every layer's goodness function target-aware.

### Fix 4: The Direction-Magnitude Decomposition
**Targets:** Root Cause 4 (optimizer crutch), Root Cause 1 (attenuation)

The OrthoAdam finding suggests a general principle: **separate the update direction from the update magnitude.** The credit rule provides the direction; a local normalization provides the magnitude.

Options:
- **Sign-based updates:** Only the sign of the credit signal matters; the magnitude is set by a per-layer learning rate. This is extremely hardware-friendly (1-bit updates).
- **Adam-style local normalization:** Each layer maintains its own first and second moment estimates. The update is m̂ᵢ / (√v̂ᵢ + ε), computed locally. This is Adam, but per-layer rather than global.
- **Spectral step size:** The step size is set by the spectral radius of the local Jacobian (or a cheap proxy). This ensures the update doesn't overshoot the local curvature.

**Why this matters:** If the credit rule only needs to provide the *direction* (which is easier to approximate than the full gradient), and the magnitude is handled locally, then even crude credit signals (random projections, contrastive estimates) can work.

**Key prediction:** If we apply per-layer Adam normalization to PEPITA's fixed-random-projection updates, the width fragility should disappear. The direction is random but consistent; the magnitude normalization prevents compounding.

**Computronium seam:** This is a U-axis modification. `EuclideanUpdate` becomes per-layer Adam. Or: a new `LocalAdamUpdate` that maintains per-layer state.

### Fix 5: Architectural Priors for Local Learning
**Targets:** Root Causes 1, 2, 3 simultaneously

The μPC finding (residual connections are load-bearing) suggests that architecture matters enormously for local learning. The right architecture can make local credit assignment much easier.

Options:
- **Residual everywhere:** Skip connections ensure that the error signal has a direct path to every layer. This is μPC's insight.
- **Error buses:** Dedicated channels that carry error signals alongside the forward pass. Each layer reads from the error bus and writes to it. The error bus is updated by a simple local rule (e.g., accumulation of prediction errors).
- **Normalized architectures:** LayerNorm, RMSNorm, or weight normalization built into the geometry. This addresses the scale problem at the architecture level rather than the optimizer level.
- **Sparse/dynamic architectures:** If only a subset of neurons is active at any time, the credit assignment problem is easier because there are fewer active weights to update. This connects to the P-axis (routing plasticity).

**Why this matters:** The architecture is the one thing we can design freely. If we design architectures that are *friendly to local learning*, we might not need to fix the credit rules at all.

**Computronium seam:** G-axis modifications. `GeometryConfig.residual` is already implemented. The next steps are error buses and normalized geometries.

### Fix 6: Timescale Separation (Plasticity as Learning)
**Targets:** Root Causes 3, 4; connects to energy/plasticity benefits

The deepest insight from the Computronium ontology: learning doesn't have to happen by updating θ. It can happen by updating ψ (fast weights, routing, gating).

Options:
- **Fast-weight adaptation:** The base weights θ are fixed. Learning happens by updating fast weights ψ that modulate the forward pass. The credit signal for ψ is much simpler than for θ because ψ operates on a shorter timescale.
- **Routing as learning:** The network learns by changing which pathways are active, not by changing the weights within pathways. The credit signal is "which pathway led to the correct output?" — a discrete, local decision.
- **Metaplasticity:** The learning rate itself is a learned quantity. Layers that are confident update slowly; layers that are uncertain update quickly. This is a form of local gain control that adapts over time.

**Why this matters:** This is where the energy and plasticity benefits come in. If learning happens by updating a small ψ vector rather than the full θ matrix, the energy cost is dramatically lower. If the network can adapt by rerouting rather than retraining, it's more robust to damage.

**Computronium seam:** P-axis modifications. The F3 audit showed that the current routing implementation is a scalar gain (not real routing). Realizing true per-pathway routing is the first step.

---

## The Unifying Hypothesis

All six fixes point to one principle:

> **Local learning fails when the credit signal's scale, direction, and information content are not locally self-consistent. Backprop succeeds because it enforces global self-consistency. The fix is to achieve local self-consistency through normalization, prediction, and timescale separation.**

The three pillars:
1. **Normalize locally** (Fixes 2, 4): Make the error signal's magnitude independent of depth/width.
2. **Predict locally** (Fixes 1, 3): Give each layer a local estimate of what it should be doing, so it can compute its own error.
3. **Separate timescales** (Fixes 5, 6): Let fast processes (settling, routing, fast weights) handle adaptation, and slow processes (θ updates) handle consolidation.

---

## The Research Program (Ordered by Expected Impact)

### Phase 1: Kill the Optimizer Crutch (highest leverage, cheapest)
**Experiment:** Apply per-layer Adam normalization (Fix 4) to PEPITA and ePC. Does the width fragility disappear? Does depth 8 train without Muon?

**Why first:** This is a U-axis-only change. No new credit rules, no new architectures. If it works, it proves that the credit direction is approximately right and only the magnitude is broken. That changes everything.

**Prediction:** PEPITA at w128 with per-layer Adam will train. ePC at w32 with per-layer Adam will not explode. If both hold, the optimizer crutch is a scale problem, not a direction problem.

### Phase 2: Learn the Feedback (kills the PEPITA bottleneck)
**Experiment:** Implement learned B (Fix 1) with an autoencoder-style local objective. Compare against fixed random B at depths 4/8/16.

**Why second:** This is the single biggest bottleneck for PEPITA. If B can track the forward weights, the depth-attenuation problem is solved for the entire FA/PEPITA family.

### Phase 3: Self-Normalizing Credit (kills the width bottleneck)
**Experiment:** Implement relative error (Fix 2) in ePC. Does the stable-width window widen? Does the ÷β cap become irrelevant?

**Why third:** This is a C-axis modification that addresses the scale problem at the source. If it works, ePC becomes width-robust without needing architectural changes.

### Phase 4: Local Target Delivery (kills error-blindness)
**Experiment:** Implement predictive targets (Fix 3) for FF. Each layer predicts the next layer's activity; the prediction error is the credit signal. Compare against ff_hybrid.

**Why fourth:** This is the most ambitious fix. If it works, FF becomes error-aware without any global loss. But it requires rethinking the credit rule fundamentally.

### Phase 5: Architecture Co-Design (amplifies everything)
**Experiment:** Combine residual geometry (already landed) with error buses (Fix 5). Does the depth wall dissolve at depth 20+ for all credit rules?

**Why fifth:** Architecture changes amplify all the other fixes. But they're harder to isolate. Run them after the credit-rule fixes are understood.

### Phase 6: Plasticity-Native Learning (the long game)
**Experiment:** Implement true per-pathway routing (Fix 6) and measure whether ψ-only adaptation can solve the switching task without any θ updates.

**Why last:** This is the hardest to get right, but it's the one that delivers the energy/plasticity/distributed benefits. It builds on everything above.

---

## The Energy/Plasticity/Distributed Payoff

Here's why this matters beyond "local learning works":

| Benefit | Mechanism | Which Fix Enables It |
|---------|-----------|---------------------|
| **Energy savings** | No backward pass through the full stack; local updates only touch local weights | Fixes 1–4 (local credit works without global backprop) |
| **Plasticity** | Fast-weight/routing adaptation updates ψ (small) instead of θ (large) | Fix 6 (timescale separation) |
| **Dynamic networks** | Routing plasticity allows the network to reconfigure without retraining | Fix 6 + Fix 5 (sparse/dynamic architectures) |
| **Distributed operation** | Each layer computes its own update independently; no global synchronization needed | Fixes 2, 3, 4 (local self-consistency) |
| **Fault tolerance** | If one pathway fails, routing plasticity reroutes around it | Fix 6 (routing as learning) |
| **Physical substrate compatibility** | Local updates, no weight transport, bounded precision, asynchronous operation | All fixes (local self-consistency is substrate-native) |

The key insight: **the fixes that make local learning competitive with backprop are the SAME fixes that make it work on physical hardware.** Global backprop is fundamentally incompatible with memristive crossbars, optical interference, and neuromorphic spikes. Local self-consistent learning is native to those substrates.

---

## What Should We Do First?

I'd recommend starting with **Phase 1: Kill the Optimizer Crutch**. Specifically:

1. Implement a `LocalAdamUpdate` (per-layer Adam, U-axis only).
2. Run the PEPITA width sweep (w32/64/128/256) with LocalAdam instead of Muon.
3. Run the ePC width sweep with LocalAdam.
4. If the fragility disappears, the hypothesis is confirmed: **the credit direction is approximately right; only the magnitude is broken.**

This is a one-probe experiment. It doesn't require new credit rules, new architectures, or new dynamics. It's a U-axis swap. And if it works, it changes the entire research program: we stop trying to fix the credit direction and focus entirely on fixing the credit magnitude.

