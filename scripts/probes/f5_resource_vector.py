"""F5 PRE-REGISTRATION ONLY — resource-vector accounting schema.

THE SCHEMA BELOW IS WRITTEN BEFORE ANY MEASUREMENT (TODO12 rev 12
binding: no computing the ratio after seeing it). Implementation lands
on the D17 Step-1 winner; this file is the schema of record.

CLAIM STRUCTURE (canonical wording, binding):

- Claim A (credit locality), separate line: "forward-local credit with
  a single readout supervision term — no backward sweep through the
  hidden layers." Never "fully local"; Muon/OrthoAdam are global
  per-matrix operations (SVD polar) — NOT part of the locality claim,
  only of the performance configuration.
- Claim B (physical advantage), separate line: no stored activations,
  no backward sweep — true REGARDLESS of optimizer.

VARIANTS REPORTED (both): plain-update (unit_rms — clean locality) and
OrthoAdam (best performance). Targets: ~10x memory bandwidth (LEADS —
backprop stores all layer activations; local rules do not; the ratio
scales with depth and is near-definitionally true), ~5x simulated
energy (SECONDARY, footnoted; OrthoAdam's SVD charged honestly — if it
eats the win, that is a real finding: "the physical advantage survives
on the forward pass; the optimizer is the cost").

ACCOUNTING SCHEMA
=================

Per-episode byte/FLOP-proxy counts at the D17-matched scale, fixed
val-window accuracy. Everything is arithmetic on the composed system's
declared terms — no walltime enters records (measured proxies only).

1. MEMORY (leads)
   - backprop arm: stored-activation bytes for the backward sweep =
     sum over hidden layers of (batch x width_l x 4 bytes), depth-
     scaled — measured with the D4 instrument `_measure_saved_bytes`
     extended to the winner arm + bp/adam control at the D17 geometry.
   - local arms: saved bytes under `requires_autograd=False` credits
     (D4 thermo precedent: exactly 0 — nothing saved for a backward
     sweep).
   - Ratio = bp_bytes / local_bytes (inf-floor at exact 0 → report as
     "unbounded (O(depth) vs O(1))" plus the depth-scaled bp curve).
   - Optimizer STATE bytes (momentum buffers, Adam m/v, SVD workspace)
     are listed per arm in a separate column — never netted out of the
     headline ratio, reported alongside it.

2. SIMULATED ENERGY (secondary, footnoted)
   FLOP-proxy per episode, charged per the substrate models' stated
   terms:
   - forward matmuls: identical for all arms (2 x params x batch).
   - backward matmuls: bp arm only (2 x params x batch — dact and
     dweight sweeps); local arms: 0.
   - local update: elementwise per-tensor ops (credit formation +
     normalize-the-momentum) ~ c_l x params, c_l counted from the
     landed rule's actual tensor ops (unit_rms: EMA + RMS scale; FA
     per-hop error propagation counted as the elementwise chain it is).
   - optimizer bookkeeping charged honestly:
       euclidean ~ 1 x params; unit_rms ~ 3 x params (EMA, RMS, scale);
       adam ~ 4 x params; OrthoAdam ~ 4 x params + SVD polar
       (10 x d^3 per weight matrix, counted per matrix).
   - Energy ratio = bp_total / local_total per variant.

3. COMPUTE/LATENCY
   - Per-episode FLOP proxy (deterministic, recordable).
   - Wall-clock chars/s printed for color, never recorded.

4. MATCHED ACCURACY
   - Arms run at the D17-registered geometry/step counts; the D17
     per-seed val_ppl means are the accuracy anchor quoted in the
     record (no re-litigating training here).

DELIVERABLE
===========
One FrontierRecord table over C = (compute, memory, energy, latency,
plastic-state capacity), local rules vs backprop at matched accuracy +
Pareto frontier figure. Lands as `test_demo_resource_vector.py`
(static-arms convention, `DEMOS` row "resource_vector": DemoSpec("F5",
_fig_declared), manifest re-pin via the gallery lock).
"""
