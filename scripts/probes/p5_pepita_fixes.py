"""P5 — PEPITA theory fixes (Fundamental-Research Focus).

P4's mechanism: pepita's error signal is not self-normalized — the fixed
random feedback projections carry a width-proportional scale, and at
width >= 128 the usable signal collapses ~100x through depth (LM act stds)
while at width 32 the activity scale explodes. Candidate fixes, each a
transform of the probability-space error e = onehot(y) - softmax(out)
before the fixed projection:

  control   — the library rule (D13's realized PEPITA)
  center    — e - mean(e) over the batch (removes the constant component;
              the LM audit's "centered-e", stability-verified)
  rmsnorm   — e / rms(e) per sample (self-normalized: the P4 fix —
              signal scale no longer rides the logit scale)
  crms      — center then rmsnorm

Fronts: (1) MNIST D13 regime (depth 2, width 32 AND 128 — does the
width fragility appear on MNIST, and does any fix lift the slow
0.226-arm?); (2) the LM w128 cell where control collapses (P4 harness
shape, depth 4, 75 s).

Run: uv run python scripts/probes/p5_pepita_fixes.py [--front mnist|lm]
"""

import time

import lm_comparison as lmc
import torch

from computronium import (
    CreditAssignmentConfig,
    DigitalSubstrate,
    FeedforwardGeometry,
    GeometryConfig,
    InstantaneousDynamics,
    LocalGoodnessCredit,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemTrainer,
    SystemTrainerConfig,
    compose_system,
    create_task,
)
from computronium.core.pipeline import run_train_step
from computronium.ontology.update import RiemannianOrthogonalUpdate

SEEDS = range(3)
BATCH_CAP = 150
LR_MUON = 0.02
LM_BUDGET_S = 75.0


def _transform(kind: str, e: torch.Tensor) -> torch.Tensor:
    if kind == "control":
        return e
    if kind == "center":
        return e - e.mean(dim=0, keepdim=True)
    if kind == "rmsnorm":
        return e / (e.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-8)
    if kind == "crms":
        e -= e.mean(dim=0, keepdim=True)
        return e / (e.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-8)
    raise ValueError(kind)


class VariantPepita(LocalGoodnessCredit):
    """Pepita with a probe-side error transform (library promotion only if
    a variant wins)."""

    kind = "control"

    def _pepita_gradient(
        self, free_state, free_acts, nudged_acts, weight_names, geometry
    ):
        n_trans = min(len(free_acts), len(nudged_acts)) - 1
        out = free_acts[-1].detach()
        y = free_state.y
        if y is None:
            return [torch.zeros_like(geometry.params[n]) for n in weight_names]
        onehot = torch.nn.functional.one_hot(y, out.shape[-1]).to(out.dtype)
        e1 = _transform(self.kind, onehot - torch.softmax(out, dim=-1)).detach()
        out_dim = e1.shape[1]
        batch = e1.shape[0]
        grads = []
        for k, name in enumerate(weight_names):
            if k >= n_trans:
                grads.append(torch.zeros_like(geometry.params[name]))
                continue
            width = geometry.params[name].shape[0]
            b = self._inverse_projection(name, width, out_dim, str(e1.device), e1.dtype)
            err = e1 @ b
            grads.append(-(err.T @ nudged_acts[k].detach()) / batch)
        return grads


def _variant(kind: str):
    c = VariantPepita(
        CreditAssignmentConfig.local_goodness(
            feedback_scale=0.01, local_objective="pepita"
        )
    )
    c.kind = kind
    return c


def mnist_front() -> None:
    from itertools import islice

    task = create_task("mnist", device="cpu", quick_mode=True, num_workers=0)
    task.setup()
    torch.manual_seed(0)  # seed BEFORE the loader draw (D8 trap)
    train_data = list(
        islice(
            ((x.view(x.size(0), -1), y) for x, y in task.get_dataloader("train")),
            BATCH_CAP,
        )
    )
    config = SystemTrainerConfig(max_epochs=1, device="cpu", seed=42)
    print(
        "=== MNIST (D13 regime, depth 2), pepita x muon, mean+-stdev over 3 seeds ==="
    )
    for width in (32, 128):
        for kind in ("control", "center", "rmsnorm", "crms"):
            accs = []
            for seed in SEEDS:
                torch.manual_seed(seed)
                geometry = FeedforwardGeometry(
                    GeometryConfig.feedforward(
                        input_dim=784, output_dim=10, hidden_dims=(width, width)
                    )
                )
                system = compose_system(
                    substrate=DigitalSubstrate(SubstrateConfig.digital(device="cpu")),
                    geometry=geometry,
                    dynamics=InstantaneousDynamics(StateDynamicsConfig.instantaneous()),
                    credit=_variant(kind),
                    update=RiemannianOrthogonalUpdate(muon_update_config()),
                )
                accs.append(
                    SystemTrainer(
                        system=system, config=config, train_data=train_data
                    ).fit()[-1]["train_acc"]
                )
            m = sum(accs) / len(accs)
            sd = (sum((a - m) ** 2 for a in accs) / (len(accs) - 1)) ** 0.5
            print(
                f"width {width:>3} {kind:>8}: {m:.3f} +/- {sd:.3f}  "
                f"{[round(a, 3) for a in accs]}",
                flush=True,
            )


def muon_update_config():
    from computronium import ParameterUpdateConfig

    return ParameterUpdateConfig.riemannian_orthogonal(step_size=LR_MUON, momentum=0.9)


def lm_front() -> None:
    torch.manual_seed(0)
    train_t, val_t = lmc.load_tokens()
    _, m_val = lmc._val_sets(val_t, 32)
    print("=== LM (P4 harness shape, depth 4, ctx 32, 75 s), pepita x muon ===")
    for width in (128,):
        for kind in ("control", "center", "rmsnorm", "crms"):
            torch.manual_seed(0)
            geometry = FeedforwardGeometry(
                GeometryConfig.feedforward(
                    input_dim=32 * lmc.VOCAB,
                    output_dim=lmc.VOCAB,
                    hidden_dims=(width,) * 4,
                )
            )
            system = compose_system(
                substrate=DigitalSubstrate(SubstrateConfig.digital(device=lmc.DEVICE)),
                geometry=geometry,
                dynamics=InstantaneousDynamics(StateDynamicsConfig.instantaneous()),
                credit=_variant(kind),
                update=RiemannianOrthogonalUpdate(
                    lmc.ParameterUpdateConfig.riemannian_orthogonal(
                        step_size=5e-4, momentum=0.9
                    )
                ),
            )
            system.geometry.to(lmc.DEVICE)  # type: ignore[attr-defined]
            gen = torch.Generator().manual_seed(1)
            t0 = time.time()
            train_loss = float("nan")
            while time.time() - t0 < LM_BUDGET_S:
                idx = torch.randint(0, len(train_t) - 33, (32,), generator=gen)
                win = train_t[idx.unsqueeze(1) + torch.arange(33)]
                x = (
                    torch.nn.functional
                    .one_hot(win[:, :-1], lmc.VOCAB)
                    .float()
                    .reshape(32, 32 * lmc.VOCAB)
                    .to(lmc.DEVICE)
                )
                y = win[:, -1].to(lmc.DEVICE)
                train_loss = run_train_step(
                    system.substrate,
                    system.geometry,
                    system.dynamics,
                    system.credit,
                    system.update,
                    x,
                    y,
                )["loss"]
            val = lmc._eval(system, m_val, "mlp")
            print(
                f"width {width:>3} {kind:>8}: train {train_loss:.3f}  val {val}",
                flush=True,
            )


if __name__ == "__main__":
    import sys

    front = sys.argv[sys.argv.index("--front") + 1] if "--front" in sys.argv else "both"
    if front in {"both", "mnist"}:
        mnist_front()
    if front in {"both", "lm"}:
        lm_front()
