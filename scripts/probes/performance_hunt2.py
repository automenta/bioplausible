"""Performance hunt, round 2: cheap iteration on `digits`, CPU.

Round-1 findings (mnist, w128, 300 batches, seeds 0-2): Muon BREAKS the
F1 depth wall — bp/muon 0.878 @ d8, 0.813 @ d16 (Euclid BP: 0.11 @ d8);
ff/muon 0.890-0.898 @ d2-d4. This round sweeps the levers on `digits`
(8x8, ~28% faster per step than mnist on CPU; CUDA measured SLOWER —
launch-bound MLP, consistent with the standing device policy):
  1. lr grid for muon at d8 (deeper nets may want different lr)
  2. depth 16/32 at the best lr
  3. width 256 at d4/d8
Winners get mnist confirmation.
"""

from itertools import islice

import numpy as np
import torch

from computronium import (
    BackpropCredit,
    CreditAssignmentConfig,
    DigitalSubstrate,
    FeedforwardGeometry,
    GeometryConfig,
    InstantaneousDynamics,
    LocalGoodnessCredit,
    ParameterUpdateConfig,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemState,
    SystemTrainer,
    SystemTrainerConfig,
    compose_system,
    create_task,
)
from computronium.ontology.update import RiemannianOrthogonalUpdate

SEEDS = (0, 1, 2)


def _flatten(loader, cap):
    for x, y in islice(loader, cap):
        yield x.view(x.size(0), -1), y


_TASK = None


def _task():
    global _TASK
    if _TASK is None:
        _TASK = create_task("digits", device="cpu", quick_mode=True, num_workers=0)
        _TASK.setup()
    return _TASK


_INPUT_DIM = 64


def _make(credit: str, depth: int, width: int, lr: float, seed: int):
    torch.manual_seed(seed)
    geometry = FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=_INPUT_DIM, output_dim=10, hidden_dims=(width,) * depth
        )
    )
    if credit == "bp":
        credit_obj = BackpropCredit()
    else:
        objective = "ff" if credit == "ff" else "pepita"
        credit_obj = LocalGoodnessCredit(
            CreditAssignmentConfig.local_goodness(
                feedback_scale=0.01, local_objective=objective
            )
        )
    return compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device="cpu")),
        geometry=geometry,
        dynamics=InstantaneousDynamics(StateDynamicsConfig.instantaneous()),
        credit=credit_obj,
        update=RiemannianOrthogonalUpdate(
            ParameterUpdateConfig.riemannian_orthogonal(step_size=lr, momentum=0.9)
        ),
    )


def _run(credit: str, depth: int, width: int, lr: float, seed: int, cap: int = 300):
    task = _task()
    torch.manual_seed(0)  # seed BEFORE the loader draw (D8 trap)
    data = list(_flatten(task.get_dataloader("train"), cap))
    system = _make(credit, depth, width, lr, seed)
    SystemTrainer(
        system=system,
        config=SystemTrainerConfig(max_epochs=1, device="cpu", seed=42),
        train_data=data,
    ).fit()
    correct = total = 0
    with torch.no_grad():
        for x, y in task.get_dataloader("test"):
            x = x.view(x.size(0), -1)
            state = system.dynamics.settle(
                SystemState(x=x), system.geometry, system.substrate, None
            )
            acts = state.activations
            out = acts[-1] if isinstance(acts, list) else acts
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def _sweep(label: str, credit: str, depth: int, width: int, lr: float, **kw):
    accs = [_run(credit, depth, width, lr, s, **kw) for s in SEEDS]
    print(f"{label:>28}: {np.mean(accs):.3f} ± {np.std(accs):.3f} {accs}", flush=True)
    return float(np.mean(accs))


if __name__ == "__main__":
    for lr in (0.01, 0.02, 0.05):
        _sweep(f"bp d8 w128 lr{lr}", "bp", 8, 128, lr)
    for lr in (0.02, 0.05):
        _sweep(f"ff d8 w128 lr{lr}", "ff", 8, 128, lr)
    for depth in (16, 32):
        _sweep(f"bp d{depth} w128 lr0.02", "bp", depth, 128, 0.02)
    _sweep("bp d16 w128 lr0.05", "bp", 16, 128, 0.05)
    for depth in (4, 8):
        _sweep(f"bp d{depth} w256 lr0.02", "bp", depth, 256, 0.02)
