"""A4 × D14 — credit_norm composed with the jpc-faithful regime (TODO12).

The D14 faithful regime (residual geometry, μPC/default init, ePC error
reparameterization, inference steps = H, β from the grid) trains depth
20 where the simple regime walls. A4's credit_norm rescales the settled
errors εᵢ per layer before the frozen-error gradient sweep.

Pre-registered predictions (falsifiable, RESEARCH4 discipline):
1. Under ADAM, credit_norm is a near-no-op on learning: Adam is
   per-coordinate scale-invariant, so per-layer ε rescaling washes out.
   If it DOES move Adam arms, the scale is entering nonlinearly through
   the ε-injected forward (energy = β·CE(y_hat)) — also a finding.
2. Under a scale-SENSITIVE update (Euclid/SGD), credit_norm=spectral
   lifts the faithful-regime arms the way A4 lifted the simple regime —
   the lever composition "faithful dynamics + normalized credit +
   cheap optimizer" is the unifying hypothesis's target endpoint
   (RESEARCH4 Lever 6: relax the optimizer crutch).

Arms: D14-exact regime (w128 d20, mupc, β=10, adam 1e-3, steps=H,
residual, batch cap 150) ± credit_norm, × {adam, euclid}.

VERDICT (2026-09-06; single seed per arm — D14's probe carries the
multi-seed baseline):
- none/adam 0.828 (replicates D14's mupc+β10 test 0.69–0.83).
- none/euclid 0.528 — the faithful DYNAMICS (residual + error
  reparameterization + steps=H + μPC init) do most of the work under
  plain SGD; the optimizer crutch in this regime is soft.
- spectral/adam 0.545, spectral/euclid 0.258, rms/* ≈ chance (0.108/
  0.116): BOTH pre-registered predictions FALSIFIED. credit_norm is
  NOT neutralized by Adam and HURTS euclid — because in the
  reparameterized regime ε is injected into the FORWARD (energy is
  computed from the ε-carrying y_hat): ε is dynamics, not merely
  credit. Rescaling it (spectral: radius→1; rms: unit-RMS — boosting
  hidden ε by ~1000×) breaks the μPC/β-calibrated scale structure.
- Composition verdict for the A6 co-design map: THE LEVERS DO NOT
  NAIVELY COMPOSE. credit_norm belongs to the simple-regime channel
  (where ε is only credit — A4's verified depth-8 lift) and to
  PEPITA-hop rescaling; the faithful regime is self-sufficient and
  credit-rescaling is actively harmful there. "Faithful dynamics +
  normalized credit + cheap optimizer" is NOT the endpoint; "faithful
  dynamics alone, SGD-capable" is closer to it.
"""

import time
from itertools import islice

import torch
from torch import optim

from computronium import (
    DigitalSubstrate,
    ErrorPredictiveCodingDynamics,
    FeedforwardGeometry,
    GeometryConfig,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemState,
    create_task,
)
from computronium.ontology._settle_kernel import extract_layered_params
from computronium.ontology.credit import _apply_credit_norm

WIDTH = 128
DEPTH = 20
BATCH_CAP = 150
EVAL_CAP = 20
ADAM_LR = 1e-3
EUCLID_LR = 0.02
ACTIVITY_STEP = 0.1
BETA = 10.0
CHANCE = 0.1


def _flatten(loader, cap):
    for x, y in islice(loader, cap):
        yield x.view(x.size(0), -1), y


def _run_arm(norm: str, optimizer: str, train_data) -> dict:
    torch.manual_seed(1)
    geometry = FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=784,
            output_dim=10,
            hidden_dims=(WIDTH,) * DEPTH,
            init_scheme="mupc",
            residual=True,
        )
    )
    substrate = DigitalSubstrate(SubstrateConfig.digital(device="cpu"))
    dynamics = ErrorPredictiveCodingDynamics(
        StateDynamicsConfig.error_predictive_coding(
            max_steps=DEPTH,
            step_size=ACTIVITY_STEP,
            beta=BETA,
            convergence_threshold=0.0,
            convergence_start=DEPTH + 1,
        )
    )
    layered = extract_layered_params(geometry)
    weights = [t[0] for t in layered.transitions]
    adam = optim.Adam(weights, lr=ADAM_LR)
    euclid_lr = EUCLID_LR

    for x, y in train_data:
        dynamics.settle(SystemState(x=x), geometry, substrate, None)
        dynamics.settle(SystemState(x=x), geometry, substrate, y)
        eps = [e.detach() for e in dynamics._last_errors]
        if norm != "none":
            eps = _apply_credit_norm(eps, norm)
        else:
            eps = [e.clone() for e in eps]

        with torch.enable_grad():
            _, y_hat = dynamics._build_forward_with_errors(
                x, layered.transitions, substrate, eps, residual=True
            )
            energy = BETA * torch.nn.functional.cross_entropy(y_hat, y)
            grads = torch.autograd.grad(energy, weights)

        with torch.no_grad():
            if optimizer == "adam":
                adam.zero_grad()
                for w, g in zip(weights, grads, strict=True):
                    w.grad = g
                adam.step()
            else:
                for w, g in zip(weights, grads, strict=True):
                    w.add_(g, alpha=-euclid_lr)
        del y_hat, grads

    correct = total = 0
    task = create_task("mnist", device="cpu", quick_mode=True, num_workers=0)
    task.setup()  # type: ignore[attr-defined]
    for x, y in _flatten(task.get_dataloader("test"), EVAL_CAP):
        state = dynamics.settle(SystemState(x=x), geometry, substrate, None)
        correct += (state.activations[-1].argmax(1) == y).sum().item()
        total += y.shape[0]
    n_train = len(train_data)
    return {
        "test": correct / total,
        "optimizer": optimizer,
        "norm": norm,
        "_n_train_batches": n_train,
    }


def main() -> None:
    torch.manual_seed(0)
    task = create_task("mnist", device="cpu", quick_mode=True, num_workers=0)
    task.setup()  # type: ignore[attr-defined]
    train_data = list(_flatten(task.get_dataloader("train"), BATCH_CAP))
    print(f"{'norm':>8} {'optimizer':>8}  test_acc")
    for norm in ("none", "spectral", "rms"):
        for optimizer in ("adam", "euclid"):
            t0 = time.time()
            r = _run_arm(norm, optimizer, train_data)
            print(
                f"{norm:>8} {optimizer:>8}  {r['test']:.3f}"
                f"  ({round(time.time() - t0, 1)} s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
