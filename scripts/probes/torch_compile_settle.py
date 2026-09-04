"""torch.compile enablement probe for the sPC layered settle (CP-6 cost floor).

Question: does torch.compile over the per-step layer loop of
``PredictiveSettlingDynamics._settle_layered`` beat the launch-bound
baseline (142 ms/train_step at depth 8 / 60 steps / batch 64, CPU)?

Method: subclass with a compiled module-level step function, verify
settled-state parity (allclose) against the stock settle on identical
inputs, then time end-to-end ``train_step``. Numbers below are recorded;
the landing (if pulled) re-demonstrates in tests.
"""

import time

import torch

from computronium import (
    CreditAssignmentConfig,
    DigitalSubstrate,
    EuclideanUpdate,
    FeedforwardGeometry,
    GeometryConfig,
    ParameterUpdateConfig,
    PredictiveSettlingDynamics,
    StateDynamicsConfig,
    SubstrateConfig,
    ThermodynamicContrast,
    compose_system,
)
from computronium.ontology.dynamics._dynamics import (
    LayeredParams,
    _create_output_state,
    _one_hot,
)

DEPTH = 8
STEPS = 60
WIDTH = 32
BATCH = 64


def _compiled_settle(acts, weights, biases, step_size, n_steps):
    for _ in range(n_steps):
        new_acts = [acts[0]]
        for i in range(len(weights)):
            h_upper = acts[i + 1]
            prediction = h_upper @ weights[i]  # op(h, w.T) for digital = h @ w
            error = acts[i] - prediction
            new_acts.append(h_upper + step_size * (error @ weights[i].T))
        acts = new_acts
    return acts


_compiled_settle_c = torch.compile(_compiled_settle, dynamic=False)


class CompiledPC(PredictiveSettlingDynamics):
    """Stock PC settle with the per-step layer loop compiled."""

    def _settle_layered(
        self, state, x, geometry, layered: LayeredParams, substrate, target
    ):
        init_acts = (
            geometry.forward_with_intermediates(x, substrate)
            if hasattr(geometry, "forward_with_intermediates")
            else None
        )
        if init_acts is None or len(init_acts) != len(layered.weights) + 1:
            raise ValueError("compiled settle requires intermediates matching weights")
        acts = list(init_acts)
        step = self.config.step_size
        acts = _compiled_settle_c(
            acts, layered.weights, layered.biases, step, self.config.max_steps
        )
        if target is not None:
            acts[-1] = acts[-1] + self.config.beta * (  # ruff: ignore[non-augmented-assignment]  out-of-place add: settle graph-safety idiom (R5.3)
                _one_hot(target, acts[-1]) - acts[-1]
            )
        return _create_output_state(
            state,
            x=x,
            output=acts[-1],
            free_state=acts if target is None else None,
            nudged_state=acts if target is not None else None,
            activations=acts,
        )


def _make(dev, dynamics_cls):
    torch.manual_seed(0)
    g = FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=784, output_dim=10, hidden_dims=(WIDTH,) * DEPTH
        )
    )
    return compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device=dev)),
        geometry=g,
        dynamics=dynamics_cls(StateDynamicsConfig.predictive_settling(max_steps=STEPS)),
        credit=ThermodynamicContrast(
            CreditAssignmentConfig.thermodynamic_contrast(beta=0.5)
        ),
        update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.2)),
    )


def main() -> None:
    x = torch.randn(BATCH, 784)
    y = torch.randint(0, 10, (BATCH,))

    stock = _make("cpu", PredictiveSettlingDynamics)
    compiled = _make("cpu", CompiledPC)

    # Parity: same input, one settle each (free phase)
    from computronium import SystemState

    torch.manual_seed(7)
    state = SystemState(x=x)
    a = stock.dynamics.settle(state, stock.geometry, stock.substrate).activations
    b = compiled.dynamics.settle(
        state, compiled.geometry, compiled.substrate
    ).activations
    if a is None or b is None:
        raise ValueError("settle returned no activations")
    max_dev = max((t - u).abs().max().item() for t, u in zip(a, b, strict=True))
    print(f"parity max|Δacts| = {max_dev:.2e}")

    for name, system in (("stock", stock), ("compiled", compiled)):
        system.train_step(x, y)  # warmup + compile
        t0 = time.perf_counter()
        for _ in range(5):
            system.train_step(x, y)
        dt = (time.perf_counter() - t0) / 5
        print(f"{name:>8}: {dt * 1000:.0f} ms/train_step")


if __name__ == "__main__":
    main()
