"""D8 — The G-axis is a swap, and it carries translation structure.

The same coordinate — Digital, Instantaneous, Null, Backprop, Euclidean —
is trained twice through byte-identical ``SystemTrainer`` wiring on the
identical materialized MNIST batch stream; the only difference between
arms is the single geometry constructor argument, and the arms are
capacity-matched (3,940 vs 3,818 parameters), so the comparison isolates
topology, not size.

Demonstrated regime: MNIST quick-mode, 1 epoch over 75 batches of 128,
hidden ``(5,)`` vs conv ``(8, 16)`` k=3 pool 4×4, Euclidean step 0.1
-> train acc ≈ 0.68 / 0.72; shifted-4 probe ≈ 0.20 (feedforward) vs
≈ 0.44 (conv). CPU-pinned for record reproducibility (standing demo-suite
device policy); the conv path is the demo suite's first FLOP-bound
regime — measured 15x faster on CUDA (0.8 s vs 12.3 s per arm, 3080).
"""

import torch

from computronium import (
    BackpropCredit,
    ConvGeometry,
    DigitalSubstrate,
    EuclideanUpdate,
    FeedforwardGeometry,
    GeometryConfig,
    InstantaneousDynamics,
    NullPlasticity,
    ParameterUpdateConfig,
    SubstrateConfig,
    SystemTrainer,
    SystemTrainerConfig,
    compose_joint_system,
    create_task,
)

DEVICE = "cpu"
BATCH_CAP = 75
PROBE_MAX_SAMPLES = 1000
PROBE_SHIFTS = (0, 2, 4)

GEOMETRY_ARMS = (
    (
        "feedforward",
        lambda: FeedforwardGeometry(
            GeometryConfig.feedforward(input_dim=784, output_dim=10, hidden_dims=(5,)),
        ),
    ),
    ("conv", lambda: ConvGeometry(GeometryConfig.conv(input_dim=784, output_dim=10))),
)


def _materialize(
    loader, *, max_batches: int | None = None, max_samples: int | None = None
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Materialize a batch stream once (worker spawn is per-iteration)."""
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    total = 0
    for x, y in loader:
        batches.append((x, y))
        total += y.numel()
        if max_batches is not None and len(batches) >= max_batches:
            break
        if max_samples is not None and total >= max_samples:
            break
    return batches


def _probe_accuracy(system, batches, shift: int) -> float:
    """Accuracy on width-rolled test digits (translation-shifted probe)."""
    device = next(iter(system.geometry.params.values())).device
    correct = total = 0
    with torch.no_grad():
        for x, y in batches:
            flat = torch.roll(x, shifts=shift, dims=-1).view(x.size(0), -1)
            pred = system.forward(flat.to(device)).argmax(-1)
            correct += (pred.cpu() == y).sum().item()
            total += y.numel()
    return correct / total


def test_demo_geometry_swap(emit_run_record) -> None:
    task = create_task("mnist", device=DEVICE, quick_mode=True, batch_size=128)
    task.setup()
    torch.manual_seed(42)
    train_batches = _materialize(task.get_dataloader("train"), max_batches=BATCH_CAP)
    probe_batches = _materialize(
        task.get_dataloader("test"), max_samples=PROBE_MAX_SAMPLES
    )
    config = SystemTrainerConfig(max_epochs=1, device=DEVICE, seed=42)

    record: dict = {"arms": {}, "probe_shifts": PROBE_SHIFTS, "device": DEVICE}
    for name, geometry in GEOMETRY_ARMS:
        torch.manual_seed(0)
        system = compose_joint_system(
            substrate=DigitalSubstrate(SubstrateConfig.digital(device=DEVICE)),
            geometry=geometry(),
            dynamics=InstantaneousDynamics(),
            plasticity=NullPlasticity(),
            credit=BackpropCredit(),
            update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.1)),
        )
        param_count = sum(p.numel() for p in system.geometry.params.values())
        metrics = SystemTrainer(
            system=system, config=config, train_data=train_batches
        ).fit()[-1]
        probes = {s: _probe_accuracy(system, probe_batches, s) for s in PROBE_SHIFTS}
        print(
            f"{name}: {metrics['train_acc']:.1%} ({param_count} params) probes={probes}"
        )
        record["arms"][name] = {
            "train_acc": metrics["train_acc"],
            "param_count": param_count,
            "probe": probes,
        }

    emit_run_record("D8", "geometry_swap", record)

    for name, arm in record["arms"].items():
        assert arm["train_acc"] > 0.25, f"{name} must learn above 2.5x chance"
        assert arm["probe"][0] > 0.4, f"{name} must classify probe digits"
    ff, conv = record["arms"]["feedforward"], record["arms"]["conv"]
    assert abs(ff["param_count"] - conv["param_count"]) < 0.1 * max(
        ff["param_count"], conv["param_count"]
    ), "arms must be capacity-matched for the comparison to be about structure"
    shift = PROBE_SHIFTS[-1]
    assert conv["probe"][shift] > ff["probe"][shift], (
        "the conv geometry must retain more accuracy under translation shift — "
        "the G-axis swap must matter"
    )
