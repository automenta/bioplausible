"""R11.3.11 in-regime μPC re-test (audit follow-up).

Instrument audit (2026-09-04) found the original depth-frontier pilot
applied μPC init outside the paper's tested domain: Table 1 of
arXiv:2505.13124 is specified and tested on residual networks (skip
connections, τ_ℓ=1 for hidden layers), width 512, Adam weights,
inference steps = H. The pilot used a plain MLP (no skips), width 32,
Euclidean updates — a domain mismatch, so its "refuted" verdict was
premature. This probe repeats the critical depth in-regime:

    residual feedforward (skip between hidden layers), width 128,
    depth 8, MNIST quick-mode, 600 batches, compiled sPC settle
    (60 steps), Euclidean lr 0.2, thermo beta 0.5, seeds 0-2.
    Arms: spc/default vs spc/mupc, both residual=True.

Interpretation guard: this is still our trainer/settle machinery, not a
jpc replication (their activity step β is orders of magnitude larger and
their inference parameterization differs). A μPC lift here would CONFIRM
the paper's claim transfers to our stack; no lift would leave the
question open pending a jpc-faithful port (Adam weights, β grid,
steps=H).

Usage: uv run python scripts/probes/mupc_residual_regime.py
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

WIDTH = 128
DEPTH = 8
SEEDS = (0, 1, 2)
ARMS = ("spc/default", "spc/mupc", "bp/default")
BATCH_CAP = 600
SETTLE_STEPS = 60
LR = 0.2
BETA = 0.5


def _flatten(loader, cap):
    for x, y in islice(loader, cap):
        yield x.view(x.size(0), -1), y


def _run(arm: str, seed: int, loader, device: str) -> float:
    torch.manual_seed(seed)
    dynamics_name, init = arm.split("/")
    geometry = FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=784,
            output_dim=10,
            hidden_dims=(WIDTH,) * DEPTH,
            init_scheme="default" if init == "default" else "mupc",
            residual=True,
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
        config=SystemTrainerConfig(max_epochs=1, device=device, seed=seed),
        train_data=list(_flatten(loader, BATCH_CAP)),
    )
    return trainer.fit()[-1]["train_acc"]


if __name__ == "__main__":
    task = create_task("mnist", device="cpu", quick_mode=True)
    task.setup()
    loader = task.get_dataloader("train")  # ruff-ok probe: runtime attr
    t_all = time.perf_counter()
    for arm in ARMS:
        accs = []
        for seed in SEEDS:
            t0 = time.perf_counter()
            acc = _run(arm, seed, loader, "cpu")
            accs.append(acc)
            print(
                f"{arm:>14} seed {seed}: {acc:.3f} ({time.perf_counter() - t0:.0f}s)",
                flush=True,
            )
        mean = sum(accs) / len(accs)
        print(
            f"{arm:>14} MEAN {mean:.3f} (range {max(accs) - min(accs):.3f})", flush=True
        )
    print(f"total {time.perf_counter() - t_all:.0f}s")
