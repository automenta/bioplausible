"""Skeptical audit of the F1 failure-manifesto arms (instrument-first).

Before accepting the negative verdicts, probe whether they are defects of
our instrument rather than the mechanisms:

1. BP depth-8 collapse vs lr grid — is 0.106 an lr artifact?
2. sPC depth wall vs budget (settle steps × batch cap) — does the wall move?
3. sPC per-layer contrast norms — does the credit signal reach hidden
   weight matrices at all under the layered settle?

Findings (2026-09-04, CPU, single seed):
    BP lr grid (0.02/0.05/0.1/0.2): 0.108/0.108/0.109/0.112 — collapse is
    NOT an lr artifact.
    sPC budget grid: 15 steps/60 cap 0.098, 30/60 0.110, 60/60 0.212,
    15/150 0.109 — budget softens the wall (0.21 at 4× settle budget).
    sPC credit norms: exactly 0.00 for all 8 hidden weight matrices,
    2.16e-01 at the last — the layered settle's contrast trains ONLY the
    output layer at any depth. The "depth wall" for sPC is last-layer-only
    training (random-feature readout boundary), not hidden-credit decay;
    consistent with D12 (hidden nudged deviations exactly zero). Whether a
    hidden-layer contrast is achievable in a layered settle at all is OPEN.

Run: uv run python scripts/probes/failure_manifesto_audit.py
"""

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
    SystemState,
    SystemTrainer,
    SystemTrainerConfig,
    ThermodynamicContrast,
    compose_system,
    create_task,
)
from computronium.ontology.credit import Phase

WIDTH = 32
DEPTH = 8
BATCH_CAP = 60
BETA = 0.5


def _flatten(loader, cap):
    for x, y in islice(loader, cap):
        yield x.view(x.size(0), -1), y


def _train(make_dyn, credit, lr, substrate, config, train_data) -> tuple[float, object]:
    torch.manual_seed(0)
    geometry = FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=784, output_dim=10, hidden_dims=(WIDTH,) * DEPTH
        )
    )
    system = compose_system(
        substrate=substrate,
        geometry=geometry,
        dynamics=make_dyn(),
        credit=credit,
        update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=lr)),
    )
    acc = SystemTrainer(system=system, config=config, train_data=train_data).fit()[-1][
        "train_acc"
    ]
    return acc, system


def _bp_grid(substrate, config, train_data) -> None:
    print("-- BP depth-8 lr grid --", flush=True)
    for lr in (0.02, 0.05, 0.1, 0.2):
        acc, _ = _train(
            lambda: InstantaneousDynamics(StateDynamicsConfig.instantaneous()),
            BackpropCredit(),
            lr,
            substrate,
            config,
            train_data,
        )
        print(f"lr {lr:>5}: acc {acc:.3f}", flush=True)


def _spc_grid(substrate, config, loader, train_data) -> None:
    print("\n-- sPC depth-8 budget grid --", flush=True)
    for steps, cap in ((15, 60), (30, 60), (60, 60), (15, 150)):
        data = train_data if cap == BATCH_CAP else list(_flatten(loader, cap))
        acc, _ = _train(
            lambda: PredictiveSettlingDynamics(
                StateDynamicsConfig.predictive_settling(max_steps=steps)
            ),
            ThermodynamicContrast(
                CreditAssignmentConfig.thermodynamic_contrast(beta=BETA)
            ),
            0.2,
            substrate,
            config,
            data,
        )
        print(f"steps {steps:>2} cap {cap:>3}: acc {acc:.3f}", flush=True)


def _credit_norms(substrate, loader) -> None:
    print("\n-- sPC per-layer contrast norms (one settle pair) --", flush=True)
    torch.manual_seed(0)
    geometry = FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=784, output_dim=10, hidden_dims=(WIDTH,) * DEPTH
        )
    )
    dynamics = PredictiveSettlingDynamics(
        StateDynamicsConfig.predictive_settling(max_steps=15)
    )
    credit = ThermodynamicContrast(
        CreditAssignmentConfig.thermodynamic_contrast(beta=BETA)
    )
    x, y = next(iter(loader))
    x = x.view(x.size(0), -1)
    free = dynamics.settle(SystemState(x=x), geometry, substrate, target=None)
    nudged = dynamics.settle(SystemState(x=x), geometry, substrate, target=y)
    grads = credit.compute_pseudo_gradient(
        {Phase.FREE: free, Phase.NUDGED: nudged}, None, geometry
    )
    norms = " ".join(f"{g.norm().item():.2e}" for g in grads)
    print("per-layer credit norms:", norms, flush=True)


def main() -> None:
    task = create_task("mnist", device="cpu", quick_mode=True, num_workers=0)
    task.setup()
    loader = task.get_dataloader("train")
    substrate = DigitalSubstrate(SubstrateConfig.digital(device="cpu"))
    config = SystemTrainerConfig(max_epochs=1, device="cpu", seed=42)
    torch.manual_seed(0)
    train_data = list(_flatten(loader, BATCH_CAP))
    _bp_grid(substrate, config, train_data)
    _spc_grid(substrate, config, loader, train_data)
    _credit_norms(substrate, loader)


if __name__ == "__main__":
    main()
