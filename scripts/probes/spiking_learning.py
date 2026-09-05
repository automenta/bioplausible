"""Spiking-family learning probe (R11.5.5 — the vacant honest-failure slot).

Question: does the timing-asymmetric STDP wiring (R11.1.9) learn on MNIST
at demo scale, or plateau? Skeptical instrument audit first:

1. Do the STDP pseudo-gradients reach every weight matrix (nonzero norms)?
2. Does supervision enter anywhere? (TemporalTraceCredit declares
   phases=(FREE,) only — the nudged settle never runs; the STDP update is
   pure spike-timing correlation with no error term.)
3. Train accuracy vs chance under the standard trainer wiring.

Run: uv run python scripts/probes/spiking_learning.py
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
    SystemTrainer,
    SystemTrainerConfig,
    TemporalTraceCredit,
    compose_system,
    create_task,
)
from computronium.ontology.credit import Phase

WIDTH = 32
DEPTHS = (1, 2, 4)
BATCH_CAP = 60
SETTLE_STEPS = 10
LR = 0.2


def _flatten(loader, cap):
    for x, y in islice(loader, cap):
        yield x.view(x.size(0), -1), y


def _build(depth: int, init_scale: float | None = None):
    torch.manual_seed(0)
    kwargs: dict = {
        "input_dim": 784,
        "output_dim": 10,
        "hidden_dims": (WIDTH,) * depth,
    }
    if init_scale is not None:
        kwargs["init_scale"] = init_scale
    geometry = FeedforwardGeometry(GeometryConfig.feedforward(**kwargs))
    dynamics = SpikeIntegrationDynamics(
        StateDynamicsConfig.spike_integration(max_steps=SETTLE_STEPS)
    )
    credit = TemporalTraceCredit(CreditAssignmentConfig.temporal_trace())
    return compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device="cpu")),
        geometry=geometry,
        dynamics=dynamics,
        credit=credit,
        update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=LR)),
    )


def main() -> None:
    task = create_task("mnist", device="cpu", quick_mode=True, num_workers=0)
    task.setup()
    loader = task.get_dataloader("train")
    config = SystemTrainerConfig(max_epochs=1, device="cpu", seed=42)
    torch.manual_seed(0)
    train_data = list(_flatten(loader, BATCH_CAP))

    for init_scale in (None, 1.0):
        for depth in DEPTHS:
            system = _build(depth, init_scale)
            label = "default" if init_scale is None else "init1.0"
            # 1. mechanism audit: gradient norms on one batch
            x, _ = train_data[0]
            free = system.dynamics.settle(
                SystemState(x=x), system.geometry, system.substrate, target=None
            )
            grads = system.credit.compute_pseudo_gradient(
                {Phase.FREE: free}, None, system.geometry
            )
            norms = [f"{g.norm().item():.2e}" for g in grads]
            rasters = getattr(free, "spike_rasters", None)
            mode = (
                "timing-STDP"
                if rasters and isinstance(rasters[0], list)
                else "rate-coded"
            )
            print(
                f"{label} depth {depth} [{mode}] credit norms: {' '.join(norms)}",
                flush=True,
            )

            # 2. learning probe
            metrics = SystemTrainer(
                system=system, config=config, train_data=train_data
            ).fit()[-1]
            print(f"  train_acc {metrics['train_acc']:.3f}", flush=True)


if __name__ == "__main__":
    main()


def _feature_probe() -> None:
    """Unsupervised STDP feature quality: centroid readout on settled
    hidden membranes, STDP-trained vs random-init control."""
    import copy

    task = create_task("mnist", device="cpu", quick_mode=True, num_workers=0)
    task.setup()
    loader = task.get_dataloader("train")
    torch.manual_seed(0)
    train_data = list(_flatten(loader, 150))
    eval_data = list(_flatten(loader, 30))

    def centroid_acc(system, data):
        feats, labels = [], []
        for x, y in data:
            state = system.dynamics.settle(
                SystemState(x=x), system.geometry, system.substrate, target=None
            )
            feats.append(state.activations[-2])
            labels.append(y)
        f = torch.cat(feats)
        y = torch.cat(labels)
        centroids = torch.stack([f[y == k].mean(0) for k in range(10)])
        return centroids, f, y

    for init_scale, lr in ((1.0, 0.02), (1.0, 0.002)):
        torch.manual_seed(0)
        system = _build(1, init_scale)
        trained = copy.deepcopy(system)
        SystemTrainer(
            system=trained,
            config=SystemTrainerConfig(max_epochs=1, device="cpu", seed=42),
            train_data=train_data,
        ).fit()
        for tag, sys_ in (("random-init", system), ("stdp-trained", trained)):
            centroids, _, _ = centroid_acc(sys_, train_data)
            fe, ye = [], []
            for x, yy in eval_data:
                st = sys_.dynamics.settle(
                    SystemState(x=x), sys_.geometry, sys_.substrate, target=None
                )
                fe.append(st.activations[-2])
                ye.append(yy)
            e = torch.cat(fe)
            ye = torch.cat(ye)
            acc = (torch.cdist(e, centroids).argmin(1) == ye).float().mean().item()
            print(
                f"[features lr={lr}] {tag}: centroid readout {acc:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
    _feature_probe()
