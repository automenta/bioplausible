"""Failure-manifesto demo-scale probe (CP-6, R11.5.5 paragraph).

Measures the four depth-failure arms at DEMO scale before landing the
consolidating figure (`test_demo_failure_manifesto.py`). Same pipeline,
same terms: train accuracy vs depth for bp/default, spc/default,
spc/mupc (μPC no-lift under our trainer), init forward norm ratios for
the unnormalized hebbian tile chain (runaway gain), and 10-class readout
for the normalized Oja chain (subspace collapse).

Findings (2026-09-04, CPU, single seed — order-dependent loader shuffle
fixed by seeding before each depth's draw):
    depth  bp     spc     spc_mupc      | runaway ratio    | oja readout
      2   0.724  0.522   0.267          10:  1.4e0        1:  0.99
      4   0.496  0.289   0.313         50:  7.2e2        10: 0.33
      8   0.106  0.105   0.124        100:  3.2e5        50: 0.20 / 100: 0.23
All four arms reproduce at demo scale in ~19 s total.

Run: uv run python scripts/probes/failure_manifesto.py
"""

import math
import time
from itertools import islice
from typing import Literal

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
from computronium.core.local_learning.builder import TileAlgorithm, TileAlgorithmConfig
from computronium.models.native import DeepHebbianChain

WIDTH = 32
TRAIN_DEPTHS = (2, 4, 8)
BATCH_CAP = 60
SETTLE_STEPS = 15
LR = 0.2
BETA = 0.5
RUNAWAY_DEPTHS = (10, 50, 100)
OJA_DEPTHS = (1, 10, 50, 100)
BATCH = 2048
EVAL = 512

InitScheme = Literal["default", "mupc"]
ARM_NAMES = ("bp", "spc", "spc_mupc")


def _flatten(loader, cap):
    for x, y in islice(loader, cap):
        yield x.view(x.size(0), -1), y


def _train_arm(
    depth: int,
    name: str,
    substrate,
    config,
    train_data,
) -> float:
    init: InitScheme = "mupc" if name == "spc_mupc" else "default"
    t0 = time.perf_counter()
    torch.manual_seed(0)
    geometry = FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=784,
            output_dim=10,
            hidden_dims=(WIDTH,) * depth,
            init_scheme=init,
        )
    )
    dynamics, credit = _make_rule(name)
    system = compose_system(
        substrate=substrate,
        geometry=geometry,
        dynamics=dynamics,
        credit=credit,
        update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=LR)),
    )
    metrics = SystemTrainer(
        system=system,
        config=config,
        train_data=train_data,
    ).fit()[-1]
    print(f"  ({time.perf_counter() - t0:.1f}s)", flush=True)
    return metrics["train_acc"]


def _make_rule(name: str):
    if name == "bp":
        return (
            InstantaneousDynamics(StateDynamicsConfig.instantaneous()),
            BackpropCredit(),
        )
    return (
        PredictiveSettlingDynamics(
            StateDynamicsConfig.predictive_settling(max_steps=SETTLE_STEPS)
        ),
        ThermodynamicContrast(CreditAssignmentConfig.thermodynamic_contrast(beta=BETA)),
    )


def _train_arms(substrate, config, loader) -> None:
    for depth in TRAIN_DEPTHS:
        torch.manual_seed(0)  # seed BEFORE the loader draw (D8 trap)
        train_data = list(_flatten(loader, BATCH_CAP))
        for name in ARM_NAMES:
            acc = _train_arm(depth, name, substrate, config, train_data)
            print(f"depth {depth:>2} {name:>8}: acc {acc:.3f}", flush=True)


def _runaway() -> None:
    print("\n-- hebbian tile chain runaway (init forward norms) --", flush=True)
    for depth in RUNAWAY_DEPTHS:
        torch.manual_seed(0)
        model = TileAlgorithm(_tile_config(depth))
        acts = model.free_phase(torch.randn(4, 16))
        norms = [
            torch
            .cat([acts[tid] for tid in layer_tiles], dim=1)
            .norm(dim=1)
            .mean()
            .item()
            for layer_tiles in model.graph.layer_ids
        ]
        head = norms[0] or 1.0
        print(
            f"depth {depth:>3}: first {head:.3e} last {norms[-1]:.3e} "
            f"ratio {norms[-1] / head:.2e} "
            f"nan={any(math.isnan(n) for n in norms)}",
            flush=True,
        )


def _tile_config(depth: int) -> TileAlgorithmConfig:
    return TileAlgorithmConfig(
        input_dim=16,
        output_dim=10,
        neurons_per_tile=16,
        tiles_per_layer=4,
        num_hidden_layers=depth,
        algorithm="hebbian",
        mode="hebbian",
        free_steps=5,
        nudged_steps=5,
        learning_rate=0.001,
        beta=0.1,
        step_size=0.1,
    )


def _collapse() -> None:
    print("\n-- normalized Oja chain subspace collapse (10-class readout) --")
    generator = torch.Generator().manual_seed(1)
    basis = torch.linalg.qr(torch.randn(32, 10, generator=generator))[0]
    means = basis.T * 3.0
    targets = torch.randint(0, 10, (BATCH,), generator=generator)
    x_train = means[targets] + torch.randn(BATCH, 32, generator=generator) * 0.5
    eval_targets = torch.randint(0, 10, (EVAL,), generator=generator)
    x_eval = means[eval_targets] + torch.randn(EVAL, 32, generator=generator) * 0.5
    for depth in OJA_DEPTHS:
        acc = _oja_readout(depth, means, x_train, targets, x_eval, eval_targets)
        print(f"depth {depth:>3}: readout last-layer {acc:.3f}", flush=True)


def _oja_readout(depth, means, x_train, targets, x_eval, eval_targets) -> float:
    torch.manual_seed(0)
    model = DeepHebbianChain(32, 32, depth, learning_rate=1e-3)
    for i in range(0, BATCH, 64):
        model.local_update(x_train[i : i + 64])
    acts_train = model(x_train)[depth]
    acts_eval = model(x_eval)[depth]
    centroids = torch.stack([acts_train[targets == k].mean(0) for k in range(10)])
    return (
        (torch.cdist(acts_eval, centroids).argmin(1) == eval_targets)
        .float()
        .mean()
        .item()
    )


def main() -> None:
    task = create_task("mnist", device="cpu", quick_mode=True, num_workers=0)
    task.setup()
    substrate = DigitalSubstrate(SubstrateConfig.digital(device="cpu"))
    config = SystemTrainerConfig(max_epochs=1, device="cpu", seed=42)
    _train_arms(substrate, config, task.get_dataloader("train"))
    _runaway()
    _collapse()


if __name__ == "__main__":
    main()
