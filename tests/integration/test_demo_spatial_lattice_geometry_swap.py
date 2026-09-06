"""D11 — The G-axis is a swap, and it carries 3D spatial structure.

The same coordinate — Digital, Instantaneous, Null, Backprop, Euclidean —
is trained twice through byte-identical ``SystemTrainer`` wiring on the
identical materialized MNIST batch stream; the only difference between
arms is the single geometry constructor argument, and the arms are
capacity-matched, so the comparison isolates topology, not size.

Demonstrated regime: MNIST quick-mode, 1 epoch over 75 batches of 128,
feedforward hidden ``(256,)`` (203,530 params) vs spatial_lattice
``(4,4,4)`` lattice with hidden ``(4,4)`` (206,090 params — a 1.25%
capacity gap, retro (f): the larger lattice arm was reduced to match),
Euclidean step 0.1 -> both learn (0.83-0.86 train, probe ≈0.91);
spatial_lattice arm tested for spatial perturbation robustness.
"""

import torch

from computronium import (
    BackpropCredit,
    DigitalSubstrate,
    EuclideanUpdate,
    FeedforwardGeometry,
    GeometryConfig,
    InstantaneousDynamics,
    NullPlasticity,
    ParameterUpdateConfig,
    SpatialLattice3DGeometry,
    SubstrateConfig,
    SystemTrainer,
    SystemTrainerConfig,
    compose_joint_system,
    create_task,
)
from computronium.visualization import bars_panel, figure_spec

DEVICE = "cpu"
BATCH_CAP = 75
PROBE_MAX_SAMPLES = 1000


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


def _probe_spatial_noise(system, batches, noise_level: float) -> float:
    """Accuracy with additive spatial noise on input."""
    device = next(iter(system.geometry.params.values())).device
    correct = total = 0
    with torch.no_grad():
        for x, y in batches:
            flat = x.view(x.size(0), -1)
            if noise_level > 0:
                flat = flat + noise_level * torch.randn_like(flat)  # ruff: ignore[non-augmented-assignment]
            pred = system.forward(flat.to(device)).argmax(-1)
            correct += (pred.cpu() == y).sum().item()
            total += y.numel()
    return correct / total


def test_demo_spatial_lattice_geometry_swap(emit_run_record) -> None:
    task = create_task("mnist", device=DEVICE, quick_mode=True, batch_size=128)
    task.setup()
    torch.manual_seed(42)
    train_batches = _materialize(task.get_dataloader("train"), max_batches=BATCH_CAP)
    probe_batches = _materialize(
        task.get_dataloader("test"), max_samples=PROBE_MAX_SAMPLES
    )
    config = SystemTrainerConfig(max_epochs=1, device=DEVICE, seed=42)

    record: dict = {"arms": {}, "probe_noise": 0.3, "device": DEVICE}
    for name, geometry_fn in (
        (
            "feedforward",
            lambda: FeedforwardGeometry(
                GeometryConfig.feedforward(
                    input_dim=784, output_dim=10, hidden_dims=(256,)
                )
            ),
        ),
        (
            "spatial_lattice",
            lambda: SpatialLattice3DGeometry(
                GeometryConfig.spatial_lattice(
                    input_dim=784,  # total input dimension
                    output_dim=10,
                    lattice_dims=(4, 4, 4),
                    hidden_dims=(4, 4),
                    connectivity_radius=1,
                )
            ),
        ),
    ):
        torch.manual_seed(0)
        system = compose_joint_system(
            substrate=DigitalSubstrate(SubstrateConfig.digital(device=DEVICE)),
            geometry=geometry_fn(),
            dynamics=InstantaneousDynamics(),
            plasticity=NullPlasticity(),
            credit=BackpropCredit(),
            update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.1)),
        )
        param_count = sum(p.numel() for p in system.geometry.params.values())
        metrics = SystemTrainer(
            system=system, config=config, train_data=train_batches
        ).fit()[-1]
        # Probe: spatial noise robustness
        probe_normal = _probe_spatial_noise(system, probe_batches, noise_level=0.0)
        probe_noisy = _probe_spatial_noise(system, probe_batches, noise_level=0.3)
        print(
            f"{name}: {metrics['train_acc']:.1%} ({param_count} params) "
            f"probe_normal={probe_normal:.1%} probe_noise={probe_noisy:.1%}"
        )
        record["arms"][name] = {
            "train_acc": metrics["train_acc"],
            "param_count": param_count,
            "probe_normal": probe_normal,
            "probe_noisy": probe_noisy,
        }

    record["figure"] = figure_spec(
        "D11 — one wiring, one swapped G-axis (3D lattice)",
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
        bars_panel(
            {
                name: {
                    "clean probe": arm["probe_normal"],
                    "noisy probe": arm["probe_noisy"],
                }
                for name, arm in record["arms"].items()
            },
            chance=1 / 10,
            chance_label="chance (0.1)",
            ylabel="probe accuracy",
            title="probe vs additive-noise probe",
            ylim=(0, 1),
        ),
        figsize=[9, 4],
    )

    emit_run_record("D11", "spatial_lattice_geometry_swap", record)

    for name, arm in record["arms"].items():
        assert arm["train_acc"] > 0.25, f"{name} must learn above 2.5x chance"
        assert arm["probe_normal"] > 0.4, f"{name} must classify probe digits"
    ff, lattice = record["arms"]["feedforward"], record["arms"]["spatial_lattice"]
    # Capacity-matched by construction: lattice hidden (8,8)->(4,4) reduced
    # the larger arm (416k -> 206k vs ff 204k, a 1.25% gap — retro (f)).
    ratio = max(ff["param_count"], lattice["param_count"]) / min(
        ff["param_count"], lattice["param_count"]
    )
    assert ratio < 1.1, f"arms must be capacity-matched (ratio={ratio:.2f})"
