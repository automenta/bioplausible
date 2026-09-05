"""Deep Hebbian chain signal-propagation probe (CP-6, user recollection).

Track 54 (`nebc_tracks.py`) claims DeepHebbianChain "maintains signal
through 50 layers" — the user recalls testing hundreds of layers. This
probe measures per-layer activation norms through the tile-graph chain at
extreme depth, for the hebbian algorithm at several depths, before and
after a short bout of Hebbian local updates.

Why it matters: the R11.3.11 frontier found every ERROR-based credit rule
(sPC, ePC, BP) dies at depth ~8. If a Hebbian chain carries activation
signal through 100+ layers, the deep-learning boundary is an
error-telescoping problem, not an activation-signal problem — sharpening
the CP-6 finding.

Findings (2026-09-04, CPU, TileAlgorithm hebbian chain, neurons/tile 16 x
tiles/layer 4, batch 4):

    depth  init rel-norms(last4 vs input)      post-1-hebbian-update
       50  5.5e+02/6.1e+02/7.0e+02/3.1e+02    NaN
      100  5.4e+05/5.6e+05/5.8e+05/2.8e+05    NaN
      200  5.3e+11/5.9e+11/6.8e+11/4.2e+11    NaN
      500  inf/inf/inf/inf                    NaN

1. The hebbian tile chain does NOT maintain signal at depth — the opposite:
   per-layer gain ~1.2-1.5x at init (unnormalized tile weights; skip
   connections make it worse, 5e7 by layer 50), compounding
   super-exponentially. At depth 500 the forward pass overflows to inf.
2. ONE hebbian local_update (lr 0.001) drives free-phase activities to NaN
   at every depth — the settle loop + hebbian strengthening is a positive
   feedback loop with no gain control.
3. Track 54's claim ("maintains signal through 50 layers") is UNVERIFIABLE
   at HEAD: its `measure_signal_propagation` method no longer exists on
   TileAlgorithm. The registered evidence string is orphaned history.
4. Sharpened CP-6 thesis: the depth bottleneck is GAIN CONTROL, and it must
   be structural. Error-based rules die by telescoping decay (R11.3.11);
   unnormalized local chains die by runaway gain; muPC's depth-scaled
   init/LR parameterization IS the normalization that lets PC survive.
   Biophysical analog: cortical gain control / homeostatic normalization.
   Next lever if this arm continues: add weight/gain normalization to the
   tile chain (unit-layer-gain init or homeostatic activity scaling) and
   re-probe — prediction: signal stays O(1) at depth 500.

The probe file is throwaway; any landing re-demonstrates claims in tests.
"""

import time

import torch
from torch import Tensor

from computronium.core.local_learning.builder import TileAlgorithm, TileAlgorithmConfig

BATCH = 4
DEPTHS = (50, 100, 200, 500)


def _layer_norms(model: TileAlgorithm, x: Tensor | None = None) -> list[float]:
    if x is None:
        x = torch.ones(BATCH, model.config.input_dim)
    acts = model.free_phase(x)
    graph = model.graph
    out: list[float] = []
    for layer_tiles in graph.layer_ids:
        stacked = torch.cat([acts[tid] for tid in layer_tiles], dim=1)
        out.append(stacked.norm(dim=1).mean().item())
    return out


def _probe(depth: int, *, use_spectral_norm: bool, use_oja: bool) -> None:
    torch.manual_seed(0)
    config = TileAlgorithmConfig(
        input_dim=16,
        output_dim=10,
        neurons_per_tile=16,
        tiles_per_layer=4,
        num_hidden_layers=depth,
        algorithm="hebbian",
        mode="hebbian",
        free_steps=10,
        nudged_steps=10,
        learning_rate=0.001,
        beta=0.1,
        step_size=0.1,
        use_spectral_norm=use_spectral_norm,
        use_oja=use_oja,
    )
    t0 = time.perf_counter()
    model = TileAlgorithm(config)
    init_norms = _layer_norms(model)
    # Structured (low-rank) input: one dominant direction + small noise.
    # Oja's rule performs PCA, so a trained chain should transmit its
    # dominant direction at gain ~1 to any depth — the legacy "signal
    # survives 100+ layers" claim, made precise.
    dir_ = torch.randn(BATCH, 16) * 3.0
    for _ in range(30):
        x = dir_ + torch.randn(BATCH, 16) * 0.1
        y = torch.randint(0, 10, (BATCH,))
        model.local_update(x, y)
    trained_norms = _layer_norms(model, dir_ + torch.randn(BATCH, 16) * 0.1)
    wall = time.perf_counter() - t0

    def rel(norms: list[float]) -> str:
        head = norms[0] or 1.0
        vals = [f"{n / head:.1e}" for n in norms[1:][-4:]]
        return "/".join(vals) + ("  [NaN!]" if any(n != n for n in norms) else "")

    tag = f"sn={int(use_spectral_norm)},oja={int(use_oja)}"
    print(
        f"depth {depth:>4} [{tag}]: init rel-norms(last4) {rel(init_norms)}  "
        f"post-hebbian {rel(trained_norms)}  ({wall:.1f}s build+probe)",
        flush=True,
    )


if __name__ == "__main__":
    for depth in DEPTHS:
        _probe(depth, use_spectral_norm=False, use_oja=False)
    for depth in DEPTHS:
        _probe(depth, use_spectral_norm=True, use_oja=True)