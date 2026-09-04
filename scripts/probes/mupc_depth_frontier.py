"""R11.3.11 μPC depth-frontier pilot (E-1, real budget).

Extends mupc_depth_init.py: the depth ≥ 8 boundary at small budgets (150
batches × 10 settle steps) dies for every credit rule including BP. Under a
real budget (600 batches × 60 settle steps, lr 0.2) μPC ≈ 2× PC learning at
depth 8. This pilot maps that real-budget frontier across depth 2–16 for
sPC/default vs sPC/mupc vs bp/default, using the R11.2.25 compiled settle
(55 ms/step CPU measured by mupc_compiled_device.py — CUDA 80 ms/step is
slower at width 32, kernel-launch-bound).

Regime: MNIST quick-mode train stream, width 32, thermo contrast beta 0.5,
Euclidean updates lr 0.2, seed 42 per arm, 600 batches × 60 settle steps.
μPC init via the native GeometryConfig init_scheme="mupc" (R11.3.11).

Findings (2026-09-04, RTX 3080 host, CPU — compiled CUDA measured SLOWER,
80 vs 55 ms/step at width 32; see mupc_compiled_device.py):

    depth  spc/default  spc/mupc  bp/default
        2        0.252     0.232       0.808
        4        0.312     0.289       0.698
        8        0.127     0.131       0.345
       12        0.119     0.127       0.110
       16        0.123     0.119       0.110

1. Boundary confirmed at depth ≈ 8 under the real budget — and BP decays
   smoothly through it (0.808 → 0.698 → 0.345 → 0.110), collapsing at 12.
   The boundary is depth-of-network, not PC-credit-specific.
2. μPC init gives NO lift at depth 8–16 here (0.131 vs 0.127 at depth 8)
   — contradicting mupc_depth_init.py finding 4 (0.225 vs 0.123). Confounds
   between the two probes: (a) this run uses the compiled fixed-budget
   settle (always 60 steps, no eager early-exit) — more compute yet worse
   final acc suggests settle-phase dynamics, not budget, dominate; (b) seed
   42 vs 0; (c) native mupc scheme vs hand-rolled. Single-seed numbers on
   both sides — treat the μPC-at-depth-8 lift as UNCONFIRMED until a
   multi-seed pilot.
3. Compiled settle locked the sPC cost at ~55 ms/step CPU (2.6× vs eager
   142 ms) — the frontier is now affordable; a full multi-seed sweep is
   ~2 h CPU.

The probe file is throwaway; any landing re-demonstrates claims in tests.
"""

import time
from itertools import islice

import torch

from computronium import (
    BackpropCredit,
    CreditAssignmentConfig,
    DigitalSubstrate,
    EuclideanUpdate,
    FeedforwardGeometry,
    GeometryConfig,
    InstantaneousDynamics,
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
DEPTHS = (2, 4, 8, 12, 16)
BATCH_CAP = 600
SETTLE_STEPS = 60
LR = 0.2
BETA = 0.5
ARMS = ("spc/default", "spc/mupc", "bp/default")


def _flatten(loader, cap):
    for x, y in islice(loader, cap):
        yield x.view(x.size(0), -1), y


def _run(depth: int, arm: str, loader, device: str) -> float:
    torch.manual_seed(42)
    dynamics_name, init = arm.split("/")
    hidden = (WIDTH,) * depth
    geometry = FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=784,
            output_dim=10,
            hidden_dims=hidden,
            init_scheme=init,
        )
    )
    if dynamics_name == "bp":
        dynamics = InstantaneousDynamics(StateDynamicsConfig.instantaneous())
        credit = BackpropCredit()
    else:
        dynamics = PredictiveSettlingDynamics(
            StateDynamicsConfig.predictive_settling(
                max_steps=SETTLE_STEPS, compiled=True
            )
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
        config=SystemTrainerConfig(max_epochs=1, device=device, seed=42),
        train_data=list(_flatten(loader, BATCH_CAP)),
    )
    return trainer.fit()[-1]["train_acc"]


if __name__ == "__main__":
    task = create_task("mnist", device="cpu", quick_mode=True)
    task.setup()
    loader = task.get_dataloader("train")
    header = "depth " + " ".join(f"{a:>12}" for a in ARMS) + "  walltime"
    print(header)
    for depth in DEPTHS:
        t0 = time.perf_counter()
        accs = [_run(depth, arm, loader, "cpu") for arm in ARMS]
        row = " ".join(f"{a:.3f}" for a in accs)
        print(f"{depth:>5} {row:>40}  {time.perf_counter() - t0:>7.1f}s", flush=True)
