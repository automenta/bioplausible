"""Spiking instrument audit, stage 2: why are hidden-layer STDP gradients zero?

Hypothesis: at width 32 with default init, deeper LIF layers never reach
threshold — no spikes, no rasters, zero STDP gradient, frozen readout.
Measure per-layer spike fractions across gain knobs (threshold, step_size,
init_scale) to find whether hidden-layer spiking is restorable.

Run: uv run python scripts/probes/spiking_gain_audit.py
"""

from itertools import islice

import torch

from computronium import (
    CreditAssignmentConfig,
    DigitalSubstrate,
    EuclideanUpdate,
    FeedforwardGeometry,
    GeometryConfig,
    ParameterUpdateConfig,
    SpikeIntegrationDynamics,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemState,
    TemporalTraceCredit,
    compose_system,
    create_task,
)

WIDTH = 32
DEPTH = 4
SETTLE_STEPS = 10
BATCH = 64


def _flatten(loader, cap):
    for x, y in islice(loader, cap):
        yield x.view(x.size(0), -1), y


def _build(init_scale: float | None, threshold: float, step: float):
    torch.manual_seed(0)
    kwargs: dict = {
        "input_dim": 784,
        "output_dim": 10,
        "hidden_dims": (WIDTH,) * DEPTH,
    }
    if init_scale is not None:
        kwargs["init_scale"] = init_scale
    return compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device="cpu")),
        geometry=FeedforwardGeometry(GeometryConfig.feedforward(**kwargs)),
        dynamics=SpikeIntegrationDynamics(
            StateDynamicsConfig.spike_integration(
                max_steps=SETTLE_STEPS, threshold=threshold, step_size=step
            )
        ),
        credit=TemporalTraceCredit(CreditAssignmentConfig.temporal_trace()),
        update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.2)),
    )


def _spike_profile(system, x) -> list[float]:
    state = system.dynamics.settle(
        SystemState(x=x), system.geometry, system.substrate, target=None
    )
    counts = []
    for layer_rasters in state.spike_rasters:
        total = sum(r.sum().item() for r in layer_rasters)
        neurons = layer_rasters[0].numel() * len(layer_rasters)
        counts.append(total / neurons)
    return counts


def main() -> None:
    task = create_task("mnist", device="cpu", quick_mode=True, num_workers=0)
    task.setup()
    loader = task.get_dataloader("train")
    x, _ = next(iter(loader))
    x = x.view(x.size(0), -1)[:BATCH]

    for init_scale in (None, 1.0):
        label = "default" if init_scale is None else f"{init_scale}"
        print(f"init_scale {label}:", flush=True)
        for threshold, step in ((1.0, 0.5), (0.5, 0.5), (0.2, 1.0), (0.05, 1.0)):
            system = _build(init_scale, threshold, step)
            counts = _spike_profile(system, x)
            print(
                f"  thr {threshold:<5} step {step}: per-layer spike frac "
                + " ".join(f"{c:.3f}" for c in counts),
                flush=True,
            )


if __name__ == "__main__":
    main()
