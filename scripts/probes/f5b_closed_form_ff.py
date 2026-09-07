"""F5b pre-registration probe: closed-form (autograd-free) FF credit.

Pre-registered predictions (written BEFORE the run, TODO12 rev 13):

- P1 (fidelity): the closed-form per-layer goodness gradient
  (detached inputs — Hinton-faithful FF; output-layer-only readout CE;
  ReLU trick: a⊙1{a>0} = a so ∂‖a‖²/∂W = 2·a·a_inᵀ) trains MNIST quick
  to within 0.05 test accuracy of the autograd FF realization at the
  same budget (mnist quick, 150 batches, seeds 0–2, euclid update).
  The two differ directionally by construction (the autograd sum chains
  every layer's goodness into ALL upstream weights — an implementation
  artifact, not Hinton's FF); the probe measures learning outcome.
- P2 (the F5 ratchet flip): the closed-form credit runs with
  requires_autograd=False → the pipeline's no_grad path saves EXACTLY
  0 bytes for backward (vs ff_hybrid's 45.5/177.5 KiB and bp's
  35.5/143.5 KiB at depths 4/16), and its per-step FLOPs drop below
  the autograd realization's.
- P3 (mechanism sanity): the closed-form gradient is EXACT for a
  linear network (no hidden ReLU): closed-form == autograd to float
  tolerance when the hidden activations are identity — a unit-lock
  candidate if the primitive is promoted.
- P4 (STRUCTURAL, amended BEFORE the run on code-reading evidence —
  `InstantaneousDynamics.settle` nudges ONLY the output act toward the
  one-hot; hidden nudged acts == free hidden acts): every hidden-layer
  goodness term ‖a_f‖²−‖a_n‖² is EXACTLY ZERO, so the autograd ff
  realization's ONLY nonzero term is the output pseudo-loss
  ‖out_f‖²−‖out_n‖² — and its autograd gradient backprops through the
  hidden layers. The chain IS the learning signal: the "layer-local"
  ff realization is output-pseudo-loss backprop at HEAD. Consequences
  tested here: (a) the closed-form hidden-layer gradient is exactly
  zero (no hidden learning without the chain); (b) the closed-form
  variant degenerates to readout-only training on frozen random
  features (ELM-shaped) — it may still clear the chance floor
  decisively; (c) CLAIM A WORDING AT RISK for ff_hybrid: "no backward
  sweep through the hidden layers" is FALSE of this realization —
  only the PEPITA-style (closed-form inverse routing) and
  requires_autograd=False credits satisfy it. F5's memory miss is
  thereby EXPLAINED, not an implementation quirk: the autograd chain
  is load-bearing. A TRUE layer-local ff needs per-layer nudged passes
  (B3/B4 territory: per-layer contrastive targets).

Task (amended to the D15-known-good footing before the run): mnist
quick, 300 train batches, full test set, depth 4 / width 128, ff
credit WITHOUT readout_error (D15's exact arm), euclid lr 0.2 (+
muon lr 0.02 sanity arm), seeds 0–2. Walltime printed.

VERDICT (2026-09-06, run ~30 s CPU):

- P4 CONFIRMED (the headline): closed_form_ff is at CHANCE
  (0.098/0.121/0.092, mean 0.104) — the hidden-layer closed-form
  gradient is exactly zero because nudged hidden acts == free hidden
  acts under InstantaneousDynamics (only the output is nudged). The
  autograd ff realization's ONLY nonzero goodness term is the output
  pseudo-loss, and its gradient REACHES the hidden weights only via
  the autograd chain: **the chain is the learning signal — ff_hybrid
  at HEAD is output-pseudo-loss backprop, layer-local in name only.**
- P1 FALSIFIED (per the P4 structure): no detached per-layer variant
  can learn without per-layer nudged passes (B3/B4 territory).
- P2 CONFIRMED trivially: closed_form saves EXACTLY 0 bytes
  (requires_autograd=False → pipeline no_grad path) vs autograd_ff's
  1108 KiB at this geometry — and 48.5 vs 74.5 M FLOPs.
- P3 untested (moot: the primitive is NOT promoted — the learning
  verdict fails).
- CONSEQUENCES (bind on the plan): (1) F5's memory miss is EXPLAINED —
  the autograd chain is load-bearing, not an implementation quirk.
  (2) CLAIM A WORDING AUDIT: "forward-local credit with a single
  readout supervision term — no backward sweep through the hidden
  layers" is FALSE of the ff_hybrid realization (there IS a backward
  sweep — the chain). The claim holds verbatim only for the
  requires_autograd=False credit family (thermo: measured 0 bytes)
  and PEPITA-style closed-form inverse routing. Any D17/ff_hybrid
  promotion must scope the locality claim accordingly (or land
  per-layer nudged passes first). (3) The honest memory/energy lever
  is a credit whose hidden signal is closed-form per layer
  (per-layer contrastive targets — B3/B4), not a rewrite of this one.
"""

from __future__ import annotations

import time
from itertools import islice

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile

from computronium import (
    CreditAssignmentConfig,
    GeometryConfig,
    ParameterUpdateConfig,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemState,
    compose_system_from_configs,
    create_task,
)
from computronium.core.pipeline import forward_pass, run_train_step
from computronium.core.profiling import measure_saved_activation_bytes
from computronium.ontology.credit import Phase
from computronium.ontology.utils import _learnable_weight_names

SEEDS = (0, 1, 2)
BATCH_CAP = 300
WIDTH = 128
DEPTH = 4
LR = 0.2
DEVICE = "cpu"


def _acts_list(activations) -> list[torch.Tensor]:
    if activations is None:
        return []
    return activations if isinstance(activations, list) else [activations]


def closed_form_gradient(credit, states, geometry) -> list[torch.Tensor]:  # noqa: PLR0914
    """Detached per-layer FF goodness gradient — no autograd graph.

    Layer k weight W_k (acts index k): ∂(‖a_free‖²−‖a_nudged‖²)/∂W_k =
    2·(a_free·a_free_inᵀ − a_nudged·a_nudged_inᵀ)/B under ReLU (the
    input is DETACHED — the autograd sum's cross-layer chain is the
    artifact this removes). The output layer adds the readout CE
    gradient (softmax − onehot)ᵀ·a_in/B, assigned to the output weight
    only. Zeros for recurrent/surplus weights.
    """
    free_state = states.get(Phase.FREE)
    nudged_state = states.get(Phase.NUDGED)
    free_acts = _acts_list(free_state.activations)
    nudged_acts = _acts_list(nudged_state.activations)
    names = _learnable_weight_names(geometry.params)
    if not free_acts or not nudged_acts or len(free_acts) < 2:
        return [torch.zeros_like(geometry.params[n]) for n in names]
    batch = free_acts[0].shape[0]
    y = free_state.y
    out_by_name = {}
    for name in names:
        parts = name.split("_")
        if len(parts) < 2 or not parts[1].isdigit():
            out_by_name[name] = torch.zeros_like(geometry.params[name])
            continue
        module_i = int(parts[1])
        k = module_i // 2 + 1  # acts index of this Linear's output
        is_output = k >= len(free_acts) - 1
        if is_output:
            af, an = free_acts[-1], nudged_acts[-1]
            aif, ain = free_acts[-2], nudged_acts[-2]
        else:
            af, an = free_acts[k], nudged_acts[k]
            aif, ain = free_acts[k - 1], nudged_acts[k - 1]
        # ReLU: a ⊙ 1{a>0} = a; the output layer is linear (same form).
        g = 2.0 * (af.T @ aif - an.T @ ain) / batch
        if is_output and credit.config.readout_error and y is not None:
            probs = torch.softmax(af, dim=-1)
            onehot = nn.functional.one_hot(y, af.shape[-1]).to(af.dtype)
            g += (probs - onehot).T @ aif / batch
        out_by_name[name] = g
    return [
        out_by_name.get(n)
        if out_by_name.get(n) is not None
        and out_by_name[n].shape == geometry.params[n].shape
        else torch.zeros_like(geometry.params[n])
        for n in names
    ]


class ClosedFormLocalGoodness:
    """Autograd-free FF credit: same settle contract, closed-form grads."""

    phases = (Phase.FREE, Phase.NUDGED)
    requires_autograd = False

    def __init__(self):
        self.config = CreditAssignmentConfig.local_goodness(local_objective="ff")

    def compute_pseudo_gradient(self, states, loss, geometry):
        return closed_form_gradient(self, states, geometry)


def _run_arm(credit_obj, seed: int) -> dict:
    torch.manual_seed(seed)
    task = create_task("mnist", device=DEVICE, quick_mode=True, num_workers=0)
    train_data = [
        (xb.reshape(xb.size(0), -1), yb)
        for xb, yb in islice(task.get_dataloader("train"), BATCH_CAP)
    ]
    test_batches = [
        (xb.reshape(xb.size(0), -1), yb) for xb, yb in task.get_dataloader("test")
    ]
    system = compose_system_from_configs(
        SubstrateConfig.digital(),
        GeometryConfig.feedforward(
            input_dim=784, output_dim=10, hidden_dims=(WIDTH,) * DEPTH
        ),
        StateDynamicsConfig.instantaneous(),
        CreditAssignmentConfig.local_goodness(),
        ParameterUpdateConfig.euclidean(step_size=LR),
    )
    # Inject the credit object directly (probe: both variants share the
    # ff config; the closed-form one swaps the class).
    system.credit = credit_obj
    x0, y0 = train_data[0]
    for xb, yb in train_data:
        run_train_step(
            system.substrate,
            system.geometry,
            system.dynamics,
            system.credit,
            system.update,
            xb,
            yb,
        )
    correct = total = 0
    with torch.no_grad():
        for xb, yb in test_batches:
            state = SystemState(x=xb)
            state.activations = forward_pass(system.substrate, system.geometry, xb)
            settled = system.dynamics.settle(
                state, system.geometry, system.substrate, target=None
            )
            acts = settled.activations
            out = acts[-1] if isinstance(acts, list) else acts
            correct += (out.argmax(-1) == yb).sum().item()
            total += len(yb)
    acc = correct / total
    _, saved = measure_saved_activation_bytes(
        run_train_step,
        system.substrate,
        system.geometry,
        system.dynamics,
        system.credit,
        system.update,
        x0,
        y0,
    )
    with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
        run_train_step(
            system.substrate,
            system.geometry,
            system.dynamics,
            system.credit,
            system.update,
            x0,
            y0,
        )
    return {
        "test_acc": acc,
        "saved_bytes": saved.total_bytes,
        "flops": sum(e.flops for e in prof.key_averages()),
    }


def main() -> int:
    from computronium.ontology.credit import LocalGoodnessCredit

    t0 = time.time()
    arms = {
        "autograd_ff": lambda: LocalGoodnessCredit(
            CreditAssignmentConfig.local_goodness(local_objective="ff")
        ),
        "closed_form_ff": ClosedFormLocalGoodness,
    }
    for arm, make in arms.items():
        accs = []
        for seed in SEEDS:
            r = _run_arm(make(), seed)
            accs.append(r["test_acc"])
            print(
                f"{arm} seed {seed}: acc {r['test_acc']:.3f}  "
                f"saved {r['saved_bytes'] / 1024:.1f} KiB  flops {r['flops'] / 1e6:.3f} M",
                flush=True,
            )
        mean = sum(accs) / len(accs)
        print(f"{arm}: mean acc {mean:.3f} over {SEEDS}", flush=True)
    print(f"\nwalltime {time.time() - t0:.1f}s (printed, never recorded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
