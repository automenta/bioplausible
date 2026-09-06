"""D7 — The D-axis settles in time.

The same coordinate — Feedforward × Null × Backprop × Euclidean on the
digital substrate — is trained twice on MNIST through identical
``SystemTrainer`` wiring; the only difference between arms is the state
dynamics. ``InstantaneousDynamics`` settles in a single pass;
``SpikeIntegrationDynamics`` settles layer-wise: every Linear transition
integrates constant drive into LIF membranes (spike at threshold, reset)
for ``max_steps`` steps, and the settled membrane carries activity to the
next layer. Before the layer-wise settle, this exact build crashed at
settle step 1 — ``route`` maps input width to output width, so any
input ≠ output feedforward build mismatched; the D-axis simply could not
compose with a classifier. The runner watches the physics: both arms
learn through the same wiring, the trained LIF network fires visibly
(every threshold crossing is counted per settle step), and its membranes
come back bounded by the spike threshold — the Lyapunov lock, live.

Demonstrated regime (live sweep 2026-09-02): MNIST quick-mode train
stream capped at 300 batches, 1 epoch, hidden ``(32,)``, Euclidean step
0.05, LIF ``max_steps=10`` -> accuracy ≈ 0.87 instant / 0.85 spike
(chance 0.1; the gap is not the claim — training through the settle is).
Post-training probe on one held batch: 20 (layer, step) spike-count
entries, ≈ 1.8k total spikes, membrane max ≈ 0.92 < threshold 1.0.
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
    SpikeIntegrationDynamics,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemState,
    SystemTrainer,
    SystemTrainerConfig,
    compose_system,
    create_task,
)
from computronium.visualization import bars_panel, figure_spec, lines_panel

BATCH_CAP = 300
LEARN_FLOOR = 0.5  # both arms must learn (5x chance)
LIF_STEPS = 10
THRESHOLD = 1.0  # SpikeIntegrationDynamics spike threshold
SPIKE_FLOOR = 100.0  # the trained network must fire visibly


def _flatten(loader, cap):
    for x, y in islice(loader, cap):
        yield x.view(x.size(0), -1), y


def test_demo_spike_settle(emit_run_record) -> None:
    task = create_task("mnist", device="cpu", quick_mode=True, num_workers=0)
    task.setup()
    train_loader = task.get_dataloader("train")
    config = SystemTrainerConfig(max_epochs=1, device="cpu", seed=42)

    arms = (
        ("instant", InstantaneousDynamics()),
        (
            "spike_integration",
            SpikeIntegrationDynamics(
                StateDynamicsConfig.spike_integration(max_steps=LIF_STEPS)
            ),
        ),
    )

    record: dict = {"arms": {}, "lif_steps": LIF_STEPS, "threshold": THRESHOLD}
    accs: dict[str, float] = {}
    probe: dict | None = None
    for name, dynamics in arms:
        torch.manual_seed(0)
        system = compose_system(
            substrate=DigitalSubstrate(SubstrateConfig.digital(device="cpu")),
            geometry=FeedforwardGeometry(
                GeometryConfig.feedforward(
                    input_dim=784, output_dim=10, hidden_dims=(32,)
                )
            ),
            dynamics=dynamics,  # the one swapped argument
            credit=BackpropCredit(CreditAssignmentConfig.gradient()),
            update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.05)),
        )
        metrics = SystemTrainer(
            system=system, config=config, train_data=_flatten(train_loader, BATCH_CAP)
        ).fit()[-1]
        accs[name] = metrics["train_acc"]
        print(f"{name}: {metrics['train_acc']:.1%}")
        record["arms"][name] = {"train_acc": metrics["train_acc"]}

        if not isinstance(dynamics, SpikeIntegrationDynamics):
            continue
        x, _ = next(iter(train_loader))
        state = SystemState(x=x.view(x.size(0), -1))
        settled = system.dynamics.settle(
            state, system.geometry, system.substrate, target=None
        )
        totals = [s.sum().item() for s in settled.spike_counts]
        probe = {
            "spike_totals": totals,
            "total_spikes": sum(totals),
            "membrane_max": settled.activations[-1].max().item(),
        }
        record["spike_observation"] = probe

    record["figure"] = figure_spec(
        "D7 — one wiring, one swapped D-axis",
        bars_panel(
            {
                name: {"train accuracy": arm["train_acc"]}
                for name, arm in record["arms"].items()
            },
            chance=1 / 10,
            chance_label="chance (0.1)",
            ylabel="train accuracy",
            ylim=(0, 1),
        ),
        lines_panel(
            {"spikes": probe["spike_totals"]},
            xlabel="settle step (hidden | output)",
            ylabel="spikes per step",
            title=(
                f"LIF settle: {probe['total_spikes']:.0f} spikes, "
                f"membrane max {probe['membrane_max']:.2f} ≤ {THRESHOLD}"
            ),
        ),
        figsize=[9, 4],
    )

    emit_run_record("D7", "spike_settle", record)

    assert accs["instant"] > LEARN_FLOOR, "instantaneous arm must learn"
    assert accs["spike_integration"] > LEARN_FLOOR, (
        "the same wiring must train through the LIF settle"
    )
    assert probe is not None
    assert len(probe["spike_totals"]) == LIF_STEPS * 2, (
        "one spike count per (layer, settle step)"
    )
    assert probe["total_spikes"] > SPIKE_FLOOR, (
        "the trained LIF network must fire visibly, not merely settle"
    )
    assert probe["membrane_max"] <= THRESHOLD, (
        "settled membranes must come back bounded by the spike threshold"
    )
