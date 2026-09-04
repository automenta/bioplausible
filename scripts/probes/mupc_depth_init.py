"""μPC depth-scaling probe (R11.3.11 E-1 smoke; informed the CP-6 finding).

Question: does muPC-style depth-scaled init (Ernoult et al., arXiv:2505.13124
— weights ~ N(0,1), hidden layers scaled 1/sqrt(N*L), output scaled 1/N)
lift predictive-coding learning past the depth where the repo's default
scalar init (fan-in x 0.1) loses signal?

Regime (measured 2026-09-04, CPU, width 32, thermo contrast beta 0.5,
Euclidean updates, seed 0 per arm): MNIST quick-mode train stream, 1 epoch.

Measured numbers (first sweep, 150 batches, 10 settle steps, lr 0.05):

    depth  spc/default  spc/mupc  epc/default  epc/mupc  bp/default  bp/mupc
        2        0.522     0.415        0.441     0.197       0.797    0.761
        4        0.231     0.279        0.096     0.106       0.598    0.408
        8        0.107     0.116        0.113     0.111       0.111    0.112
       12        0.114     0.124        0.108     0.128       0.110    0.128
       16        0.105     0.100        0.104     0.121       0.121    0.121

Findings (2026-09-04):

1. All dynamics die at depth >= 8 under small budgets — including plain
   backprop, so the boundary is NOT PC-credit-specific.
2. Credit path verified healthy: per-layer autograd grad norms flow to
   every layer (~2x decay per layer, none zero) — vanilla vanishing
   gradients of a deep narrow net, no structural break. GradientCredit's
   allow_unused=True zero-fill is NOT masking a detached layer.
3. Single levers each recover a little at depth 8 (settle steps 10->60:
   0.10 -> 0.20; lr 0.05->0.2: +0.04; muPC init: +0.02), none rescues.
4. Real budget (600 batches, 60 settle steps, lr 0.2): muPC 0.225 vs
   default 0.123 at depth 8 — muPC init nearly doubles PC learning
   exactly where the boundary sits. Depth frontier above that budget was
   not completed (sweep cost, see 5).
5. Device/cost verdict: sPC layered settle is kernel-launch-bound — CUDA
   measured SLOWER than CPU (201 vs 142 ms/train_step, depth 8, 60
   steps, batch 64). Extends the demo-suite CPU verdict to registered-
   scale PC. The frontier sweep is unaffordable until the layered settle
   is vectorized/compiled (SubstrateSettleKernel or torch.compile) or
   batch-per-step is raised by 4-8x.

The probe file itself is throwaway; any landing re-demonstrates claims
in tests, never here.
"""

import time
from itertools import islice

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
    SystemTrainer,
    SystemTrainerConfig,
    ThermodynamicContrast,
    compose_system,
    create_task,
)
from computronium.ontology.dynamics import ErrorPredictiveCodingDynamics

BATCH_CAP = 150
WIDTH = 32
DEPTHS = (2, 4, 8, 12, 16)
STEP_SIZE = 0.05
LRS = (0.05, 0.2, 0.5)
BETA = 0.5
SETTLE_STEPS = 10
ARMS = (
    "spc/default",
    "spc/mupc",
    "epc/default",
    "epc/mupc",
    "bp/default",
    "bp/mupc",
)


def _flatten(loader, cap):
    for x, y in islice(loader, cap):
        yield x.view(x.size(0), -1), y


def _mupc_init(geometry, depth: int) -> None:
    """muPC scheme in place: N(0,1) weights, hidden 1/sqrt(N*L), output 1/N."""
    gen = torch.Generator().manual_seed(1234)
    for i, layer in enumerate(geometry._layers):
        if not isinstance(layer, torch.nn.Linear):
            continue
        is_output = i >= len(geometry._layers) - 2
        scale = 1.0 / WIDTH if is_output else 1.0 / (WIDTH * depth) ** 0.5
        layer.weight.data = torch.randn(layer.weight.shape, generator=gen) * scale
        layer.bias.data.zero_()


def _run(depth: int, arm: str, loader, lr: float = STEP_SIZE) -> float:
    torch.manual_seed(0)
    dynamics_name, init = arm.split("/")
    hidden = (WIDTH,) * depth
    geometry = FeedforwardGeometry(
        GeometryConfig.feedforward(input_dim=784, output_dim=10, hidden_dims=hidden)
    )
    if init == "mupc":
        _mupc_init(geometry, depth)
    if dynamics_name == "epc":
        dynamics = ErrorPredictiveCodingDynamics(
            StateDynamicsConfig.error_predictive_coding(
                max_steps=SETTLE_STEPS, step_size=0.5, beta=BETA
            )
        )
    elif dynamics_name == "bp":
        from computronium import BackpropCredit, InstantaneousDynamics

        dynamics = InstantaneousDynamics(StateDynamicsConfig.instantaneous())
    else:
        dynamics = PredictiveSettlingDynamics(
            StateDynamicsConfig.predictive_settling(max_steps=SETTLE_STEPS)
        )
    credit = (
        BackpropCredit()
        if dynamics_name == "bp"
        else ThermodynamicContrast(
            CreditAssignmentConfig.thermodynamic_contrast(beta=BETA)
        )
    )
    system = compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device="cpu")),
        geometry=geometry,
        dynamics=dynamics,
        credit=credit,
        update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=lr)),
    )
    config = SystemTrainerConfig(max_epochs=1, device="cpu", seed=42)
    metrics = SystemTrainer(
        system=system, config=config, train_data=_flatten(loader, BATCH_CAP)
    ).fit()[-1]
    return metrics["train_acc"]


def main() -> None:
    task = create_task("mnist", device="cpu", quick_mode=True)
    task.setup()
    loader = task.get_dataloader("train")
    header = "depth " + " ".join(f"{a:>12}" for a in ARMS) + "  walltime"
    print(header)
    for lr in LRS:
        t0 = time.perf_counter()
        accs = [(d, _run(d, "bp/default", loader, lr)) for d in (4, 8, 12)]
        row = " ".join(f"{d}:{a:.3f}" for d, a in accs)
        print(f"lr={lr:<4} {row}  {time.perf_counter() - t0:>7.1f}s")


if __name__ == "__main__":
    main()
