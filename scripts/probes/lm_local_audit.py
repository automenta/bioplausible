"""LM local-credit audit + PC-family smoke (user directive 2026-09-05).

Question 1 — are we doing something stupid that hurts FF/PEPITA on LM?
Question 2 — do the energy-based families (sPC, ePC) move at all?

All arms: MLP LM geometry (one-hot context window, single next-char
target), ctx 32 / w256 x4 (746k params, D13-scale), tiny Shakespeare,
1 minute each, GPU. Controls reproduce the D13-proven credit configs.

FF variants:
- ff/control      — D13-exact (feedback_scale 0.01, muon 0.02, beta 0.5)
- ff/beta2        — stronger nudge (beta 2.0): bigger free/nudged contrast
- ff/hybrid       — FF layer-local goodness + CE on the output logits
                    (the error-blindness fix: the readout finally sees y)

PEPITA variants:
- pepita/control  — D13-exact
- pepita/centered — e1 = e1 − mean(e1): kills the constant one-hot term
                    (at init softmax≈uniform so e≈onehot−1/65 is constant-
                    dominated — the D13 raw-differential pathology, resurfacing
                    in probability space because per-position targets differ)
- pepita/ortho    — orthogonal_init=True feedback projections
- pepita/fs1e-3   — feedback_scale 1e-3 (gentler feedback)

PC arms (dynamics x credit x update):
- spc/thermo/{euclid,muon}   — PredictiveSettlingDynamics(max_steps=15)
- epc/thermo/{euclid,muon}   — ErrorPredictiveCodingDynamics(max_steps=10),
                               the D12 pairing

Usage: uv run python scripts/probes/lm_local_audit.py [--minutes 1]
"""

from __future__ import annotations

import argparse
import math
import time

import torch
from torch import nn

from computronium import (
    AdamUpdate,
    BackpropCredit,
    CreditAssignmentConfig,
    DigitalSubstrate,
    EuclideanUpdate,
    FeedforwardGeometry,
    GeometryConfig,
    InstantaneousDynamics,
    LocalGoodnessCredit,
    ParameterUpdateConfig,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemState,
    ThermodynamicContrast,
    compose_system,
)
from computronium.core.pipeline import run_train_step
from computronium.ontology.credit import Phase
from computronium.ontology.dynamics import (
    ErrorPredictiveCodingDynamics,
    PredictiveSettlingDynamics,
)
from computronium.ontology.update import RiemannianOrthogonalUpdate

VOCAB = 65
CTX = 32
HIDDEN = (256,) * 4
BATCH = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VCTX = CTX


def load_tokens() -> tuple[torch.Tensor, torch.Tensor]:
    from computronium.data.lm import get_lm_dataset as _g

    train = _g("tiny_shakespeare", seq_len=64, split="train")
    val = _g("tiny_shakespeare", seq_len=64, split="validation")
    stoi = {c: i for i, c in enumerate(sorted(set(train.idx_to_char.values())))}
    return train.data.long(), torch.tensor([stoi[c] for c in val.decode(val.data)])


class CenteredPepitaCredit(LocalGoodnessCredit):
    """PEPITA with the constant one-hot term removed from the error."""

    def _pepita_gradient(
        self, free_state, free_acts, nudged_acts, weight_names, geometry
    ):
        out = free_acts[-1].detach()
        y = free_state.y
        if y is None:
            return [torch.zeros_like(geometry.params[n]) for n in weight_names]
        onehot = torch.nn.functional.one_hot(y, out.shape[-1]).to(out.dtype)
        e1 = (onehot - torch.softmax(out, dim=-1)).detach()
        e1 = torch.sub(e1, e1.mean(dim=0, keepdim=True))
        return self._pepita_from_e1(e1, free_acts, nudged_acts, weight_names, geometry)

    def _pepita_from_e1(self, e1, free_acts, nudged_acts, weight_names, geometry):
        n_trans = min(len(free_acts), len(nudged_acts)) - 1
        out_dim = e1.shape[1]
        batch = e1.shape[0]
        grads: list[torch.Tensor] = []
        for k, name in enumerate(weight_names):
            if k >= n_trans:
                grads.append(torch.zeros_like(geometry.params[name]))
                continue
            width = geometry.params[name].shape[0]
            b = self._inverse_projection(name, width, out_dim, str(e1.device), e1.dtype)
            err = e1 @ b
            grads.append(-(err.T @ nudged_acts[k].detach()) / batch)
        return grads


class HybridFFCredit(LocalGoodnessCredit):
    """FF layer-local goodness + CE on the free logits (readout error term).

    The FF objective is error-blind by construction; the hybrid adds the
    output CE so the last layer (and, through it, every hidden layer via
    the shared autograd graph) sees the target."""

    def compute_pseudo_gradient(self, states, loss, geometry):
        free_state = states.get(Phase.FREE)
        nudged_state = states.get(Phase.NUDGED)
        if free_state is None or nudged_state is None:
            return []
        fa = free_state.activations
        na = nudged_state.activations
        if fa is None or na is None:
            return []
        free_acts = fa if isinstance(fa, list) else [fa]
        nudged_acts = na if isinstance(na, list) else [na]
        weight_names = [
            n for n, p in geometry.params.items() if "weight" in n and p.ndim == 2
        ]
        n_trans = min(len(free_acts), len(nudged_acts)) - 1
        if n_trans < 1 or nudged_acts[-1] is None or not nudged_acts[-1].requires_grad:
            return [torch.zeros_like(geometry.params[n]) for n in weight_names]
        total = torch.zeros((), device=nudged_acts[-1].device)
        for i in range(1, n_trans + 1):
            total = total + free_acts[i].pow(2).mean() - nudged_acts[i].pow(2).mean()
        y = free_state.y
        if y is not None and free_acts[-1] is not None:
            logits = free_acts[-1]
            total = torch.add(total, nn.functional.cross_entropy(logits, y))
        params = [geometry.params[n] for n in weight_names]
        grads = torch.autograd.grad(
            total, params, retain_graph=False, create_graph=False, allow_unused=True
        )
        return [
            g if g is not None else torch.zeros_like(p)
            for p, g in zip(params, grads, strict=True)
        ]


def build(credit_obj, dynamics, update_factory):
    return compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device=DEVICE)),
        geometry=FeedforwardGeometry(
            GeometryConfig.feedforward(
                input_dim=CTX * VOCAB, output_dim=VOCAB, hidden_dims=HIDDEN
            )
        ),
        dynamics=dynamics,
        credit=credit_obj,
        update=update_factory(),
    )


def run(name, system, tokens, val, minutes, seed=0):
    torch.manual_seed(seed)
    system.geometry.to(DEVICE)
    gen = torch.Generator().manual_seed(seed + 1)
    curve = []
    t0 = time.time()
    step = 0
    while time.time() - t0 < minutes * 60:
        idx = torch.randint(0, len(tokens) - CTX - 1, (BATCH,), generator=gen)
        win = tokens[idx.unsqueeze(1) + torch.arange(CTX)]
        x = (
            torch.nn.functional
            .one_hot(win, VOCAB)
            .float()
            .reshape(BATCH, CTX * VOCAB)
            .to(DEVICE)
        )
        y = tokens[idx + CTX].to(DEVICE)
        m = run_train_step(
            system.substrate,
            system.geometry,
            system.dynamics,
            system.credit,
            system.update,
            x,
            y,
        )
        step += 1
        if time.time() - t0 > 20.0 * (len(curve) + 1):
            curve.append(round(m["loss"], 3))
    # final val
    tot = n = 0
    with torch.no_grad():
        for x, y in val:
            state = system.dynamics.settle(
                SystemState(x=x.to(DEVICE)), system.geometry, system.substrate, None
            )
            acts = state.activations
            logits = acts[-1] if isinstance(acts, list) else acts
            tot += nn.functional.cross_entropy(
                logits, y.to(DEVICE), reduction="sum"
            ).item()
            n += y.numel()
    avg = tot / n
    print(
        f"{name:>22}  steps {step:>5}  train {curve}  "
        f"val_loss {avg:.3f}  val_ppl {math.exp(min(avg, 20)):.2f}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=float, default=1.0)
    args = parser.parse_args(argv)

    train_t, val_t = load_tokens()
    gen = torch.Generator().manual_seed(0)
    vidx = torch.randint(0, len(val_t) - VCTX - 1, (512,), generator=gen)
    vwin = val_t[vidx.unsqueeze(1) + torch.arange(VCTX + 1)]
    eye = torch.eye(VOCAB)
    val = []
    for w in vwin.split(256):
        val.append((eye[w[:, :-1]].reshape(w.size(0), -1), w[:, -1]))

    def muon(lr=0.02):
        return RiemannianOrthogonalUpdate(
            ParameterUpdateConfig.riemannian_orthogonal(step_size=lr, momentum=0.9)
        )

    def euclid(lr=0.1):
        return EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=lr))

    def inst():
        return InstantaneousDynamics(StateDynamicsConfig.instantaneous())

    arms = [
        (
            "ff/control",
            LocalGoodnessCredit(
                CreditAssignmentConfig.local_goodness(
                    feedback_scale=0.01, local_objective="ff"
                )
            ),
            inst,
            lambda: muon(0.02),
        ),
        (
            "ff/beta2",
            LocalGoodnessCredit(
                CreditAssignmentConfig.local_goodness(
                    beta=2.0, feedback_scale=0.01, local_objective="ff"
                )
            ),
            inst,
            lambda: muon(0.02),
        ),
        (
            "ff/hybrid",
            HybridFFCredit(
                CreditAssignmentConfig.local_goodness(
                    feedback_scale=0.01, local_objective="ff"
                )
            ),
            inst,
            lambda: muon(0.02),
        ),
        (
            "pepita/control",
            LocalGoodnessCredit(
                CreditAssignmentConfig.local_goodness(
                    feedback_scale=0.01, local_objective="pepita"
                )
            ),
            inst,
            lambda: muon(0.02),
        ),
        (
            "pepita/centered",
            CenteredPepitaCredit(
                CreditAssignmentConfig.local_goodness(
                    feedback_scale=0.01, local_objective="pepita"
                )
            ),
            inst,
            lambda: muon(0.02),
        ),
        (
            "pepita/ortho",
            LocalGoodnessCredit(
                CreditAssignmentConfig.local_goodness(
                    feedback_scale=0.01, local_objective="pepita", orthogonal_init=True
                )
            ),
            inst,
            lambda: muon(0.02),
        ),
        (
            "pepita/fs1e-3",
            LocalGoodnessCredit(
                CreditAssignmentConfig.local_goodness(
                    feedback_scale=1e-3, local_objective="pepita"
                )
            ),
            inst,
            lambda: muon(0.02),
        ),
        (
            "bp/control",
            BackpropCredit(),
            inst,
            lambda: AdamUpdate(ParameterUpdateConfig.adam(step_size=1e-3)),
        ),
        (
            "spc/thermo/euclid",
            ThermodynamicContrast(),
            lambda: PredictiveSettlingDynamics(
                StateDynamicsConfig.predictive_settling(max_steps=15, step_size=0.1)
            ),
            euclid,
        ),
        (
            "spc/thermo/muon",
            ThermodynamicContrast(),
            lambda: PredictiveSettlingDynamics(
                StateDynamicsConfig.predictive_settling(max_steps=15, step_size=0.1)
            ),
            lambda: muon(0.02),
        ),
        (
            "epc/thermo/euclid",
            ThermodynamicContrast(),
            lambda: ErrorPredictiveCodingDynamics(
                StateDynamicsConfig.error_predictive_coding(max_steps=10, step_size=0.1)
            ),
            euclid,
        ),
        (
            "epc/thermo/muon",
            ThermodynamicContrast(),
            lambda: ErrorPredictiveCodingDynamics(
                StateDynamicsConfig.error_predictive_coding(max_steps=10, step_size=0.1)
            ),
            lambda: muon(0.02),
        ),
    ]

    print(f"chance = ln(65) = {math.log(65):.3f}  |  {args.minutes} min/arm, {DEVICE}")
    for name, credit, dyn, upd in arms:
        system = build(credit, dyn(), upd)
        run(name, system, train_t, val, args.minutes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
