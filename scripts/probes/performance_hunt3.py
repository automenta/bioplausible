"""Performance hunt, round 3: scale the champion — conv geometry + budget.

Round-2 verdict: FF×Muon and BP×Muon are the champions (D15: FF 0.930 /
BP 0.911 at d4 w256, capacity-matched). Unexplored levers on the proven
high performers:
  1. CONV geometry × {bp, ff} × muon — capacity-identical within each
     pair (same geometry). The conv path is the measured FLOP-bound
     one; GPU vs CPU walltime measured first (device policy: measured,
     not assumed).
  2. BUDGET — the flat champion (ff/muon d4 w256) at 2-3 epochs.
  3. WIDER conv (16, 32).

mnist quick, seeds 0-2, TEST accuracy. Winners: D-table / RESULTS
material.
"""

import time
from itertools import islice

import numpy as np
import torch

from computronium import (
    BackpropCredit,
    ConvGeometry,
    CreditAssignmentConfig,
    DigitalSubstrate,
    FeedforwardGeometry,
    GeometryConfig,
    InstantaneousDynamics,
    LocalGoodnessCredit,
    ParameterUpdateConfig,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemTrainer,
    SystemTrainerConfig,
    compose_system,
    create_task,
)
from computronium.ontology.update import RiemannianOrthogonalUpdate

SEEDS = (0, 1, 2)
LR_MUON = 0.02


def _credit(name: str):
    if name == "bp":
        return BackpropCredit()
    objective = "ff" if name == "ff" else "pepita"
    return LocalGoodnessCredit(
        CreditAssignmentConfig.local_goodness(
            feedback_scale=0.01, local_objective=objective
        )
    )


def _geometry(name: str):
    if name == "conv":
        return ConvGeometry(GeometryConfig.conv(input_dim=784, output_dim=10))
    if name == "conv_wide":
        return ConvGeometry(
            GeometryConfig.conv(
                input_dim=784, output_dim=10, conv_channels=(16, 32)
            )
        )
    if name == "ff_d4_w256":
        return FeedforwardGeometry(
            GeometryConfig.feedforward(
                input_dim=784, output_dim=10, hidden_dims=(256,) * 4
            )
        )
    raise ValueError(name)


def _run(geometry_name: str, credit: str, seed: int, *, epochs: int = 1,
         device: str = "cpu", train_data=None, test_batches=None) -> float:
    task = create_task("mnist", device="cpu", quick_mode=True, num_workers=0)
    task.setup()
    if train_data is None:
        torch.manual_seed(0)  # seed BEFORE the loader draw (D8 trap)
        train_data = list(islice(task.get_dataloader("train"), 300))
    if test_batches is None:
        test_batches = list(task.get_dataloader("test"))
    if device != "cpu":
        # GPU: move data to device (conv is the measured FLOP-bound path;
        # CPU data batches otherwise trip the substrate device check)
        train_data = [(x.to(device), y.to(device)) for x, y in train_data]
        test_batches = [(x.to(device), y.to(device)) for x, y in test_batches]
    torch.manual_seed(seed)
    system = compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device=device)),
        geometry=_geometry(geometry_name),
        dynamics=InstantaneousDynamics(StateDynamicsConfig.instantaneous()),
        credit=_credit(credit),
        update=RiemannianOrthogonalUpdate(
            ParameterUpdateConfig.riemannian_orthogonal(
                step_size=LR_MUON, momentum=0.9
            )
        ),
    )
    SystemTrainer(
        system=system,
        config=SystemTrainerConfig(max_epochs=epochs, device=device, seed=42),
        train_data=train_data,
    ).fit()
    ok = tot = 0
    with torch.no_grad():
        for batch_x, batch_y in test_batches:
            state = system.dynamics.settle(
                __import__("computronium").SystemState(x=batch_x),
                system.geometry,
                system.substrate,
                None,
            )
            acts = state.activations
            out = acts[-1] if isinstance(acts, list) else acts
            ok += (out.argmax(1) == batch_y).sum().item()
            tot += batch_y.size(0)
    return ok / tot


def _params(name: str) -> int:
    return sum(p.numel() for p in _geometry(name).params.values())


if __name__ == "__main__":
    task = create_task("mnist", device="cpu", quick_mode=True, num_workers=0)
    task.setup()
    torch.manual_seed(0)
    train_data = list(islice(task.get_dataloader("train"), 300))
    test_batches = list(task.get_dataloader("test"))

    print(f"params: conv={_params('conv')} conv_wide={_params('conv_wide')} "
          f"ff_d4_w256={_params('ff_d4_w256')}", flush=True)

    # GPU vs CPU walltime on the conv×muon arm (device policy: measure).
    for device in ("cpu", "cuda"):
        started = time.perf_counter()
        _run("conv", "bp", 0, device=device, train_data=train_data,
             test_batches=test_batches)
        print(f"conv/bp/muon 1ep on {device}: {time.perf_counter() - started:.1f}s",
              flush=True)

    for geometry, credit in (
        ("conv", "bp"), ("conv", "ff"),
        ("conv_wide", "bp"), ("conv_wide", "ff"),
    ):
        accs = [_run(geometry, credit, s, train_data=train_data,
                     test_batches=test_batches) for s in SEEDS]
        print(f"{geometry}/{credit}/muon: {np.mean(accs):.3f} "
              f"± {np.std(accs):.3f} {accs}", flush=True)

    for epochs in (2, 3):
        accs = [_run("ff_d4_w256", "ff", s, epochs=epochs,
                     train_data=train_data, test_batches=test_batches)
                for s in SEEDS]
        print(f"ff_d4_w256/ff/muon {epochs}ep: {np.mean(accs):.3f} "
              f"± {np.std(accs):.3f} {accs}", flush=True)
