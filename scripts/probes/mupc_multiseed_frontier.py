"""R11.3.11 multi-seed depth-frontier pilot (resolves the μPC lift contradiction).

mupc_depth_frontier.py (single seed 42) found NO μPC lift at depth 8
(0.131 vs 0.127), contradicting mupc_depth_init.py (0.225 vs 0.123).
This pilot repeats the critical depths with multiple seeds.

Regime: same as mupc_depth_frontier.py — MNIST quick-mode, width 32,
compiled sPC settle (60 steps), lr 0.2, thermo beta 0.5, 600 batches.
Seeds 0/1/2/3; depths 4 and 8 (the contradiction zone); arms
spc/default vs spc/mupc (bp included at depth 8 as the BP reference).

Usage: uv run python scripts/probes/mupc_multiseed_frontier.py
"""

import time
from itertools import islice

import torch

from computronium import (
    CreditAssignmentConfig,
    DigitalSubstrate,
    EuclideanUpdate,
    FeedforwardGeometry,
    GeometryConfig,
    ParameterUpdateConfig,
    PredictiveSettlingDynamics,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemTrainer,
    SystemTrainerConfig,
    ThermodynamicContrast,
    compose_system,
    create_task,
)

WIDTH = 32
DEPTHS = (4, 8)
SEEDS = (0, 1, 2, 3)
ARMS = ("spc/default", "spc/mupc")
BATCH_CAP = 600
SETTLE_STEPS = 60
LR = 0.2
BETA = 0.5


def _flatten(loader, cap):
    for x, y in islice(loader, cap):
        yield x.view(x.size(0), -1), y


def _run(depth: int, arm: str, seed: int, loader, device: str) -> float:
    torch.manual_seed(seed)
    init = arm.split("/")[1]
    geometry = FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=784,
            output_dim=10,
            hidden_dims=(WIDTH,) * depth,
            init_scheme="default" if init == "default" else "mupc",
        )
    )
    dynamics = PredictiveSettlingDynamics(
        StateDynamicsConfig.predictive_settling(max_steps=SETTLE_STEPS, compiled=True)
    )
    credit = ThermodynamicContrast(
        CreditAssignmentConfig.thermodynamic_contrast(beta=BETA)
    )
    system = compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device=device)),
        geometry=geometry,
        dynamics=dynamics,
        credit=credit,
        update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=LR)),
    )
    trainer = SystemTrainer(
        system=system,
        config=SystemTrainerConfig(max_epochs=1, device=device, seed=seed),
        train_data=list(_flatten(loader, BATCH_CAP)),
    )
    return trainer.fit()[-1]["train_acc"]


if __name__ == "__main__":
    task = create_task("mnist", device="cpu", quick_mode=True)
    task.setup()
    loader = task.get_dataloader("train")
    t_all = time.perf_counter()
    for depth in DEPTHS:
        for arm in ARMS:
            accs = []
            for seed in SEEDS:
                t0 = time.perf_counter()
                acc = _run(depth, arm, seed, loader, "cpu")
                accs.append(acc)
                print(
                    f"depth {depth:>2} {arm:>12} seed {seed}: {acc:.3f} "
                    f"({time.perf_counter() - t0:.0f}s)",
                    flush=True,
                )
            mean = sum(accs) / len(accs)
            spread = max(accs) - min(accs)
            print(
                f"depth {depth:>2} {arm:>12} MEAN {mean:.3f} (range {spread:.3f})",
                flush=True,
            )
    print(f"total {time.perf_counter() - t_all:.0f}s")
