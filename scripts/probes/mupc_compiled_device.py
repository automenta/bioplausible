"""Device/cost probe for the R11.3.11 μPC depth frontier with compiled settle.

Extends mupc_depth_init.py finding 5 (sPC layered settle is kernel-launch-
bound; CUDA measured SLOWER eager at depth 8). R11.2.25 landed a
torch.compile fast path (StateDynamicsConfig.predictive_settling(...,
compiled=True)) — this probe measures whether it flips the device verdict.

Regime: MNIST quick-mode train stream, sPC, width 32, depth 8, 60 settle
steps, batch 64, lr 0.2, thermo contrast beta 0.5, seed 42. Reports
steady-state ms/train_step (epoch 2+, compile warm after epoch 0).

Findings (2026-09-04, RTX 3080, torch 2.x): compiled settle does NOT flip
the device verdict at width 32 — CPU 55–57 ms/step vs CUDA 80–81 ms/step
(depth 8, 60 settle steps, steady-state epochs 2+). The layered settle
stays kernel-launch-bound even compiled; compile itself gave 2.6× on CPU
(142 → 55 ms/step vs eager). GPU-first applies only to FLOP-bound paths
(conv family), not the sPC settle loop.
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
DEPTH = 8
SETTLE_STEPS = 60
BATCH_CAP = 10


def _flatten(loader, cap):
    for x, y in islice(loader, cap):
        yield x.view(x.size(0), -1), y


def _epoch_ms(device: str) -> None:
    task = create_task("mnist", device=device, quick_mode=True)
    task.setup()
    loader = task.get_dataloader("train")
    geometry = FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=784, output_dim=10, hidden_dims=(WIDTH,) * DEPTH
        )
    )
    system = compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device=device)),
        geometry=geometry,
        dynamics=PredictiveSettlingDynamics(
            StateDynamicsConfig.predictive_settling(
                max_steps=SETTLE_STEPS, compiled=True
            )
        ),
        credit=ThermodynamicContrast(
            CreditAssignmentConfig.thermodynamic_contrast(beta=0.5)
        ),
        update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.2)),
    )
    trainer = SystemTrainer(
        system=system,
        config=SystemTrainerConfig(max_epochs=1, device=device, seed=42),
        train_data=list(_flatten(loader, BATCH_CAP)),
    )
    for epoch in range(3):
        if epoch == 0:
            trainer._begin_epoch()
        t0 = time.perf_counter()
        trainer.train_epoch()
        wall = time.perf_counter() - t0
        print(
            f"  {device} epoch {epoch}: {wall:.1f}s "
            f"({1000 * wall / BATCH_CAP:.0f} ms/step)"
        )


if __name__ == "__main__":
    for device in ("cpu", "cuda"):
        print(
            f"device={device} (depth {DEPTH}, {SETTLE_STEPS} settle steps, "
            "compiled=True, ms/step steady-state):"
        )
        _epoch_ms(device)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
