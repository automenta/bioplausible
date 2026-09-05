"""jpc-faithful regime probe (TODO11 Path A, R11.3.11 tail): the remaining
instrument gap for BOTH open verdicts (muPC lift, F1 depth wall).

Paper regime (arXiv:2505.13124 / ePC): Adam on weights, activity GD with a
nudge-strength beta grid (1e3 -> 1e-2), inference steps = H (depth), width
512 residual networks. Our prior re-tests used Euclidean SGD, beta 0.5,
fixed 60 settle steps — both verdicts were "no lift under OUR trainer",
never refutations.

Manual loop (no SystemTrainer — trainer support is pulled only if the
regime shows lift, R11.5.6):
  1. free settle: ePC, max_steps = H (fixed; convergence off)
  2. nudged settle: ePC, beta from the grid, max_steps = H
  3. PC-native weight gradient: dE_nudged/dtheta with the settled errors
     frozen — equals the paper's Delta theta_i ~ (d s_i/d theta_i)^T eps_i
     (the reverse-mode sweep carries the propagated error to each weight)
  4. torch.optim.Adam step

Arms: residual depth 8, width 128 (the earlier in-regime re-test's
geometry, for comparability), {default, mupc} init x beta grid x gamma
(activity step) grid, one seed. E-1 smoke scale.

Run: uv run python scripts/probes/jpc_faithful.py
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
DEPTH = 8
BATCH_CAP = 150
ADAM_LR = 1e-3
BETAS = (1e3, 1e2, 10.0, 1.0, 0.1, 0.01)
GAMMAS = (0.1, 0.5)


def _flatten(loader, cap):
    for x, y in islice(loader, cap):
        yield x.view(x.size(0), -1), y


def run_arm(
    init: str,
    beta: float,
    gamma: float,
    train_data,
    depth: int = DEPTH,
    seed: int = 0,
) -> tuple[float, object, object]:
    torch.manual_seed(seed)
    geometry = FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=784,
            output_dim=10,
            hidden_dims=(WIDTH,) * depth,
            init_scheme=init,
            residual=True,
        )
    )
    substrate = DigitalSubstrate(SubstrateConfig.digital(device="cpu"))
    dynamics = ErrorPredictiveCodingDynamics(
        StateDynamicsConfig.error_predictive_coding(
            max_steps=depth,  # inference steps = H
            step_size=gamma,
            beta=beta,
            convergence_threshold=0.0,  # fixed budget, no early exit
            convergence_start=depth + 1,
        )
    )
    layered = extract_layered_params(geometry)
    weights = [t[0] for t in layered.transitions]
    adam = torch.optim.Adam(weights, lr=ADAM_LR)

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
            energy = beta * torch.nn.functional.cross_entropy(y_hat, y)
            grads = torch.autograd.grad(energy, weights)

        adam.zero_grad()
        for w, g in zip(weights, grads, strict=True):
            w.grad = g
        adam.step()

        correct += (y_hat.argmax(1) == y).sum().item()
    return correct / (len(train_data) * train_data[0][1].shape[0]), dynamics, geometry


def evaluate(dynamics, geometry, substrate, eval_data) -> float:
    """Held-out accuracy via the free (inference) settle."""
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

    print("=== E-1 smoke: depth-8 beta/gamma grid, train acc ===")
    for init in ("default", "mupc"):
        for gamma in GAMMAS:
            for beta in BETAS:
                acc, _, _ = run_arm(init, beta, gamma, train_data)
                print(
                    f"init {init:>7} gamma {gamma:<4} beta {beta:<6}: {acc:.3f}",
                    flush=True,
                )

    print("\n=== pilot: depth-20, train + held-out accuracy, seeds 0-2 ===")
    eval_data = list(_flatten(task.get_dataloader("test"), 20))
    for init in ("default", "mupc"):
        for beta in (1e3, 10.0):
            for seed in range(3):
                train_acc, dynamics, geometry = run_arm(
                    init, beta, 0.1, train_data, depth=20, seed=seed
                )
                substrate = DigitalSubstrate(SubstrateConfig.digital(device="cpu"))
                test_acc = evaluate(dynamics, geometry, substrate, eval_data)
                print(
                    f"depth 20 {init:>7} beta {beta:<6} seed {seed}: "
                    f"train {train_acc:.3f}  test {test_acc:.3f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
