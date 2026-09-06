"""Learning-algorithm hunt, round 3: the Muon+Adam hybrid cell.

Queued from TODO11's hunt list: "Muon+Adam hybrid (orthogonalized
Adam, e.g. Adam-momentum + NS orthogonalization)". OrthoAdam: per-
coordinate Adam moments with bias correction, then Muon's SVD polar
factor applied to the bias-corrected first moment of every MATRIX
parameter (vector params keep plain Adam). Matches Muon's recipe —
orthogonalize the momentum, then step — while inheriting Adam's
per-coordinate adaptivity for everything the polar factor can't shape.

Reference numbers from the D16 record (same regime, seeds 0-2):
mlp — adam 0.892 / muon 0.919; attention — adam 0.900 / muon 0.874;
graph — adam 0.332 / muon 0.433; lattice — adam 0.895 / muon 0.905.

Question: does orthogonalizing Adam's momentum beat BOTH parents
anywhere on the map?
"""

from itertools import islice
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np
import torch

from computronium import (
    AdamUpdate,
    AttentionGeometry,
    BackpropCredit,
    DigitalSubstrate,
    FeedforwardGeometry,
    GeometryConfig,
    GraphGeometry,
    InstantaneousDynamics,
    ParameterUpdateConfig,
    SpatialLattice3DGeometry,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemState,
    SystemTrainer,
    SystemTrainerConfig,
    compose_system,
    create_task,
)

SEEDS = (0, 1, 2)
BATCH_CAP = 150


class OrthoAdamUpdate(AdamUpdate):
    """Adam moments; matrix-shaped first-moment directions are
    orthogonalized (SVD polar factor) before the step."""

    def __init__(self, config, ortho_lr: float):
        super().__init__(config)
        self.ortho_lr = ortho_lr

    def step(self, params, pseudo_grads, geometry):
        grads = self._clip(list(pseudo_grads))
        self._t += 1
        beta1 = self.config.momentum
        beta2 = self.config.beta2
        bias1 = 1 - beta1**self._t
        bias2 = 1 - beta2**self._t

        def apply(name: str, param: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
            m = self._state(name, param, self._m)
            v = self._state(name, param, self._v)
            m.mul_(beta1).add_(grad, alpha=1 - beta1)
            v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            m_hat = m / bias1
            denom = (v / bias2).sqrt().add_(self.config.eps)
            adam_step = m_hat / denom
            if param.ndim == 2:
                U, _, Vh = torch.linalg.svd(m_hat, full_matrices=False)
                ortho = U @ Vh
                # rescale the orthogonal direction to Adam's per-tensor
                # step magnitude so lr stays comparable across arms
                ortho *= adam_step.norm() / (ortho.norm() + 1e-8)
                return param - self.ortho_lr * ortho
            return param - self.config.step_size * adam_step

        from computronium.ontology.update import apply_pseudo_gradients

        return apply_pseudo_gradients(params, grads, apply)


def _grid_edges(h: int = 8, w: int = 4) -> list[list[int]]:
    src: list[int] = []
    dst: list[int] = []
    for r in range(h):
        for c in range(w):
            i = r * w + c
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    j = nr * w + nc
                    src.extend((i, j))
                    dst.extend((j, i))
    return [src, dst]


def _geometries() -> dict[str, Callable]:
    return {
        "mlp_d2_w64": lambda: FeedforwardGeometry(
            GeometryConfig.feedforward(
                input_dim=784, output_dim=10, hidden_dims=(64, 64)
            )
        ),
        "attention": lambda: AttentionGeometry(
            GeometryConfig.attention(
                input_dim=784, output_dim=10, hidden_dim=32, num_layers=2, num_heads=4
            )
        ),
        "graph_grid8x4": lambda: GraphGeometry(
            GeometryConfig.graph(
                input_dim=784,
                output_dim=10,
                edge_index=_grid_edges(),
                hidden_dims=(56, 56),
            )
        ),
        "lattice3d": lambda: SpatialLattice3DGeometry(
            GeometryConfig.spatial_lattice(
                input_dim=784,
                output_dim=10,
                lattice_dims=(4, 3, 3),
                hidden_dims=(2,),
                connectivity_radius=1,
            )
        ),
    }


def _run(update_fn, geometry_fn, seed, train_data, test_batches) -> float:
    torch.manual_seed(seed)
    system = compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device="cpu")),
        geometry=geometry_fn(),
        dynamics=InstantaneousDynamics(StateDynamicsConfig.instantaneous()),
        credit=BackpropCredit(),
        update=update_fn(),
    )
    SystemTrainer(
        system=system,
        config=SystemTrainerConfig(max_epochs=1, device="cpu", seed=42),
        train_data=train_data,
    ).fit()
    ok = tot = 0
    with torch.no_grad():
        for batch_x, batch_y in test_batches:
            state = system.dynamics.settle(
                SystemState(x=batch_x), system.geometry, system.substrate, None
            )
            acts = state.activations
            out = acts[-1] if isinstance(acts, list) else acts
            ok += (out.argmax(1) == batch_y).sum().item()
            tot += batch_y.size(0)
    return ok / tot


if __name__ == "__main__":
    task = create_task("mnist", device="cpu", quick_mode=True, num_workers=0)
    task.setup()
    torch.manual_seed(0)  # seed BEFORE the loader draw (D8 trap)
    train_data = [
        (xb.view(xb.size(0), -1), yb)
        for xb, yb in islice(task.get_dataloader("train"), BATCH_CAP)
    ]
    test_batches = [
        (xb.view(xb.size(0), -1), yb)
        for xb, yb in task.get_dataloader("test")
        if xb.size(0) == 32
    ]

    # lr calibration on mlp, seed 0: the orthogonal direction is
    # rescaled to Adam's step magnitude, so lr 1e-3 stays comparable.
    calibration = {}
    for lr in (1e-3, 3e-3, 1e-2):
        acc = _run(
            lambda lr=lr: OrthoAdamUpdate(
                ParameterUpdateConfig.adam(step_size=1e-3), ortho_lr=lr
            ),
            _geometries()["mlp_d2_w64"],
            0,
            train_data,
            test_batches,
        )
        calibration[lr] = acc
        print(f"calib mlp ortho_adam lr {lr:g}: {acc:.3f}", flush=True)

    best_lr = max(calibration, key=calibration.get)
    for name, geometry_fn in _geometries().items():
        accs = [
            _run(
                lambda lr=best_lr: OrthoAdamUpdate(
                    ParameterUpdateConfig.adam(step_size=1e-3), ortho_lr=lr
                ),
                geometry_fn,
                s,
                train_data,
                test_batches,
            )
            for s in SEEDS
        ]
        print(
            f"{name:>12} ortho_adam(lr {best_lr:g}): {np.mean(accs):.3f} "
            f"± {np.std(accs):.3f} {accs}",
            flush=True,
        )
