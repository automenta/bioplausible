"""Performance hunt: Muon across the ontology at a real budget (E-1, 2026-09-05).

User directive: stop studying low performers — find high performers. The
strongest measured lever in the library is the Muon-class orthogonalized
update (D13: FF×Muon 0.84 vs FF×Euclid 0.57; BP×Muon 0.84 at width 32).
Unexplored territory, probed here at width 128:

1. DEPTH SCALING — F1's depth wall (BP → 0.11 at depth 8) was measured
   with Euclidean SGD. Does Muon break the wall? depths 2/4/8/16,
   credits {bp, ff} × muon.
2. CREDIT SWEEP — Muon × every credit that composes with instantaneous
   dynamics: bp, ff, pepita.
3. THE CHAMPION SCALED — ff × muon at width 128 vs D13's 0.84 @ width 32.

MNIST quick, 300 batches, 1 epoch, test accuracy, seeds 0-2 on the
interesting arms. Probe only: promotes into a demo if something big
shows.
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
    SystemTrainer,
    SystemTrainerConfig,
    compose_system,
    create_task,
)
from computronium import SystemState
from computronium.ontology.update import RiemannianOrthogonalUpdate

WIDTH = 128
BATCH_CAP = 300
LR_MUON = 0.02
SEEDS = (0, 1, 2)


def _flatten(loader, cap):
    for x, y in islice(loader, cap):
        yield x.view(x.size(0), -1), y


def _run(credit: str, depth: int, width: int, seed: int, lr: float = LR_MUON) -> float:
    torch.manual_seed(seed)
    geometry = FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=784, output_dim=10, hidden_dims=(width,) * depth
        )
    )
    if credit == "bp":
        credit_obj = BackpropCredit()
    elif credit == "ff":
        credit_obj = LocalGoodnessCredit(
            CreditAssignmentConfig.local_goodness(
                feedback_scale=0.01, local_objective="ff"
            )
        )
    else:
        credit_obj = LocalGoodnessCredit(
            CreditAssignmentConfig.local_goodness(
                feedback_scale=0.01, local_objective="pepita"
            )
        )
    system = compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device="cpu")),
        geometry=geometry,
        dynamics=InstantaneousDynamics(StateDynamicsConfig.instantaneous()),
        credit=credit_obj,
        update=RiemannianOrthogonalUpdate(
            ParameterUpdateConfig.riemannian_orthogonal(
                step_size=lr, momentum=0.9
            )
        ),
    )
    trainer = SystemTrainer(
        system=system,
        config=SystemTrainerConfig(max_epochs=1, device="cpu", seed=42),
        train_data=_TRAIN_DATA,
    )
    trainer.fit()
    return _test_acc(trainer.system)


def _test_acc(system) -> float:
    task = create_task("mnist", device="cpu", quick_mode=True, num_workers=0)
    task.setup()
    correct = total = 0
    with torch.no_grad():
        for x, y in islice(task.get_dataloader("test"), 50):
            x = x.view(x.size(0), -1)
            state = system.dynamics.settle(
                SystemState(x=x),
                system.geometry,
                system.substrate,
                None,
            )
            acts = state.activations
            out = acts[-1] if isinstance(acts, list) else acts
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def _load():
    task = create_task("mnist", device="cpu", quick_mode=True, num_workers=0)
    task.setup()
    torch.manual_seed(0)  # seed BEFORE the loader draw (D8 trap)
    return list(_flatten(task.get_dataloader("train"), BATCH_CAP))


if __name__ == "__main__":
    _TRAIN_DATA = _load()

    results: dict[str, list[float]] = {}
    # 1. Depth scaling — the F1-wall question
    for credit in ("bp", "ff"):
        for depth in (2, 4, 8, 16):
            key = f"{credit}/muon d{depth} w{WIDTH}"
            accs = [_run(credit, depth, WIDTH, s) for s in SEEDS]
            results[key] = accs
            print(f"{key:>24}: {np.mean(accs):.3f} ± {np.std(accs):.3f} {accs}")
    # 2. Champion scaled + pepita along for the ride
    for credit in ("ff", "pepita"):
        key = f"{credit}/muon d2 w{WIDTH}"
        if f"{credit}/muon d2 w{WIDTH}" in results:
            continue
        accs = [_run(credit, 2, WIDTH, s) for s in SEEDS]
        results[key] = accs
        print(f"{key:>24}: {np.mean(accs):.3f} ± {np.std(accs):.3f} {accs}")
