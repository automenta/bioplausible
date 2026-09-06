"""Hunt cell: OrthoAdam × the D14 jpc-faithful regime.

Queued from TODO11's hunt list: "the OrthoAdam×D14 jpc regime (the
manual loop's torch.optim.Adam could become OrthoAdam — does
orthogonalized-momentum Adam lift μPC's depth-20 regime further?)"

D14 reference (same regime, jpc_faithful.py, seeds 0-2, TEST):
depth 20 / width 128 residual — mupc beta=10: test 0.686 / 0.828 /
0.831 (mean 0.78); default beta=10: 0.142 / 0.237 / 0.234 (train 0.997
— memorization).

Manual loop (the D14 reference implementation, TODO11 Notes): ePC free
settle (steps = H) → nudged settle (beta) → PC-native weight gradient
with settled errors frozen → optimizer step. The optimizer swaps:
torch.optim.Adam (D14 baseline) vs OrthoAdam directions (Adam moments,
SVD polar factor on matrix first moments, rescaled to the Adam step
magnitude, ortho_lr 3e-3 — the D15/D16 configuration of record).

Question: does orthogonalizing the momentum rescue the DEFAULT-init
memorization corner (Adam's depth-fragile second-moment normalization
is the D15 finding) and/or lift μPC beyond 0.78 mean?

Run: uv run python scripts/probes/jpc_ortho_adam.py
"""

from itertools import islice

import torch

from computronium import (
    DigitalSubstrate,
    ErrorPredictiveCodingDynamics,
    FeedforwardGeometry,
    GeometryConfig,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemState,
    create_task,
)
from computronium.ontology._settle_kernel import extract_layered_params

WIDTH = 128
DEPTH = 20
BATCH_CAP = 150
BETA = 10.0  # the D14 generalizing corner (beta 1e3 = memorization)
GAMMA = 0.1
SEEDS = (0, 1, 2)


class _OrthoAdamWeights:
    """Adam moments + SVD-polar matrix directions over a weight list —
    the OrthoAdamUpdate recipe applied to the jpc loop's plain tensor
    list (no ParameterUpdateConfig plumbing needed in the manual loop).
    State is per-arm; identical-init comparisons are safe."""

    def __init__(self, weights: list[torch.Tensor], lr: float = 3e-3):
        self.weights = weights
        self.lr = lr
        self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8
        self.t = 0
        self.m = [torch.zeros_like(w) for w in weights]
        self.v = [torch.zeros_like(w) for w in weights]

    def step(self, grads: list[torch.Tensor]) -> None:
        self.t += 1
        bias1 = 1 - self.beta1**self.t
        bias2 = 1 - self.beta2**self.t
        with torch.no_grad():
            for w, g, m, v in zip(self.weights, grads, self.m, self.v, strict=True):
                m.mul_(self.beta1).add_(g, alpha=1 - self.beta1)
                v.mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
                m_hat = m / bias1
                adam_step = m_hat / (v / bias2).sqrt().add_(self.eps)
                if w.ndim == 2:
                    U, _, Vh = torch.linalg.svd(m_hat, full_matrices=False)
                    ortho = U @ Vh
                    ortho *= adam_step.norm() / (ortho.norm() + 1e-8)
                    w.add_(ortho, alpha=-self.lr)
                else:
                    w.add_(adam_step, alpha=-self.lr)


def _flatten(loader, cap):
    for x, y in islice(loader, cap):
        yield x.view(x.size(0), -1), y


def run_arm(
    init: str,
    optimizer: str,
    train_data,
    seed: int,
) -> tuple[float, object, object, object]:
    torch.manual_seed(seed)
    geometry = FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=784,
            output_dim=10,
            hidden_dims=(WIDTH,) * DEPTH,
            init_scheme=init,
            residual=True,
        )
    )
    substrate = DigitalSubstrate(SubstrateConfig.digital(device="cpu"))
    dynamics = ErrorPredictiveCodingDynamics(
        StateDynamicsConfig.error_predictive_coding(
            max_steps=DEPTH,
            step_size=GAMMA,
            beta=BETA,
            convergence_threshold=0.0,
            convergence_start=DEPTH + 1,
        )
    )
    layered = extract_layered_params(geometry)
    weights = [t[0] for t in layered.transitions]
    if optimizer == "adam":
        opt = torch.optim.Adam(weights, lr=1e-3)
    else:
        opt = _OrthoAdamWeights(weights)

    correct = 0
    for x, y in train_data:
        free = dynamics.settle(SystemState(x=x), geometry, substrate, None)
        del free
        dynamics.settle(SystemState(x=x), geometry, substrate, y)
        eps = [e.detach() for e in dynamics._last_errors]

        with torch.enable_grad():
            _, y_hat = dynamics._build_forward_with_errors(
                x, layered.transitions, substrate, eps, residual=True
            )
            energy = BETA * torch.nn.functional.cross_entropy(y_hat, y)
            grads = torch.autograd.grad(energy, weights)

        if isinstance(opt, torch.optim.Adam):
            opt.zero_grad()
            for w, g in zip(weights, grads, strict=True):
                w.grad = g
            opt.step()
        else:
            opt.step([g.detach() for g in grads])

        correct += (y_hat.argmax(1) == y).sum().item()
    return (
        correct / (len(train_data) * train_data[0][1].shape[0]),
        dynamics,
        geometry,
        opt,
    )


def evaluate(dynamics, geometry, substrate, eval_data) -> float:
    correct, total = 0, 0
    for x, y in eval_data:
        state = dynamics.settle(SystemState(x=x), geometry, substrate, None)
        y_hat = state.activations[-1]
        correct += (y_hat.argmax(1) == y).sum().item()
        total += y.shape[0]
    return correct / total


def main() -> None:
    task = create_task("mnist", device="cpu", quick_mode=True, num_workers=0)
    task.setup()
    torch.manual_seed(0)  # seed BEFORE the loader draw (D8 trap)
    train_data = list(_flatten(task.get_dataloader("train"), BATCH_CAP))
    eval_data = list(_flatten(task.get_dataloader("test"), 20))
    substrate = DigitalSubstrate(SubstrateConfig.digital(device="cpu"))

    print(
        f"=== depth {DEPTH} width {WIDTH} residual, beta {BETA:g}, "
        f"seeds {SEEDS}: {{adam, ortho_adam}} x {{default, mupc}} ===",
        flush=True,
    )
    for init in ("default", "mupc"):
        for optimizer in ("adam", "ortho_adam"):
            for seed in SEEDS:
                train_acc, dynamics, geometry, _ = run_arm(
                    init, optimizer, train_data, seed
                )
                test_acc = evaluate(dynamics, geometry, substrate, eval_data)
                print(
                    f"{init:>7} x {optimizer:<10} seed {seed}: "
                    f"train {train_acc:.3f}  test {test_acc:.3f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
