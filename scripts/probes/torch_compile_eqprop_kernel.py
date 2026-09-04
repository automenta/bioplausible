"""torch.compile probe for SubstrateSettleKernel (EqProp/tile settle engine).

Question: does compiling the whole multi-step kernel loop beat the eager
loop, with bitwise-equivalent results? This is the EnergyMinimization
workhorse (D1/D2, EqProp family, tile meshes) — the last launch-bound
settle path.

Method: replicate ``SubstrateSettleKernel.step`` exactly as a functional
compiled loop (digital substrate, no recurrent, momentum=0 — the common
case), verify final-acts parity against the stock kernel, time both.
Recorded numbers live in the printed output and the TODO11 Watch note.
"""

import time

import torch

from computronium import (
    DigitalSubstrate,
    FeedforwardGeometry,
    GeometryConfig,
    SubstrateConfig,
)
from computronium.ontology._settle_kernel import (
    SubstrateSettleKernel,
    _one_hot,
    extract_layered_params,
)

DEPTH = 8
STEPS = 30
BATCH = 64


def _eqprop_settle(acts, weights, biases, activations, step_size, n_steps):
    """Functional replica of kernel.step loop (digital, no recurrent, no momentum)."""
    for _ in range(n_steps):
        new_acts = [acts[0]]
        num_hidden = len(acts) - 2
        for i in range(num_hidden):
            out = acts[i] @ weights[i].T
            b = biases[i]
            if b is not None:
                out = out + b
            top_down = acts[i + 2] @ weights[i + 1]
            total = out + top_down
            target_h = activations[i](total) if i < len(activations) else total
            new_acts.append(acts[i + 1] + step_size * (target_h - acts[i + 1]))
        out = new_acts[-1] @ weights[len(weights) - 1].T
        b = biases[len(weights) - 1]
        if b is not None:
            out = out + b
        new_acts.append(out)
        acts = new_acts
    return acts


_eqprop_settle_c = torch.compile(_eqprop_settle, dynamic=False)


def main() -> None:
    torch.manual_seed(0)
    geometry = FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=784, output_dim=10, hidden_dims=(32,) * DEPTH
        )
    )
    substrate = DigitalSubstrate(SubstrateConfig.digital(device="cpu"))
    params = extract_layered_params(geometry)
    assert params is not None
    kernel = SubstrateSettleKernel(
        substrate=substrate, params=params, step_size=0.5
    )
    x = torch.randn(BATCH, 784)
    acts0 = geometry.forward_with_intermediates(x, substrate)

    # Parity: stock kernel loop vs compiled functional loop
    acts = list(acts0)
    for _ in range(STEPS):
        acts, _ = kernel.step(list(acts), 0.0, None, None)
    compiled_out = _eqprop_settle_c(
        list(acts0), params.weights, params.biases, params.activations, 0.5, STEPS
    )
    max_dev = max(
        (a - b).abs().max().item() for a, b in zip(acts, compiled_out, strict=True)
    )
    print(f"parity max|Δacts| = {max_dev:.2e}")

    # Timing: eager kernel loop
    def run_eager():
        a = list(acts0)
        for _ in range(STEPS):
            a, _ = kernel.step(list(a), 0.0, None, None)
        return a

    def run_compiled():
        return _eqprop_settle_c(
            list(acts0), params.weights, params.biases, params.activations, 0.5, STEPS
        )

    for name, fn in (("eager", run_eager), ("compiled", run_compiled)):
        fn()  # warmup + compile
        t0 = time.perf_counter()
        for _ in range(5):
            fn()
        dt = (time.perf_counter() - t0) / 5
        print(f"{name:>8}: {dt * 1000:.0f} ms/settle ({STEPS} steps, depth {DEPTH})")


if __name__ == "__main__":
    main()
