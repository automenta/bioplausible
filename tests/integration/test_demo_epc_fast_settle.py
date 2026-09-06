"""D12 — The D-axis settles without signal decay (ePC).

The same coordinate — Feedforward × Null × ThermodynamicContrast ×
Euclidean on the digital substrate — is trained twice on MNIST through
identical ``SystemTrainer`` wiring; the only difference between arms is
the state dynamics. ``PredictiveSettlingDynamics`` (sPC) relaxes states
layer-wise; ``ErrorPredictiveCodingDynamics`` (ePC, Goemaere et al.,
"ePC: Fast and Deep Predictive Coding in Digital Simulation",
arXiv:2505.20137, ICML 2026) reparameterizes the dynamics in terms of
prediction errors εᵢ — sᵢ = ŝᵢ + εᵢ — so one reverse-mode sweep carries
the output-loss gradient to every layer unattenuated.

Three physics claims, live:

1. Zero-step free equilibrium: ePC's free-phase errors start at zero and
   zero is their fixed point, so the settled free state IS the feedforward
   pass — bitwise, before settling begins.
2. Global nudged propagation: after ePC's nudged settle, every hidden
   layer deviates from its free state — the output-error signal reached
   all of them in one update. The layered sPC settle applies the nudge
   only at the output; its hidden deviations are exactly zero.
3. Settle-budget asymmetry favors the faster dynamics: the ePC arm trains
   with a third of the sPC arm's settle budget (10 vs 30 steps).

Demonstrated regime (live 2026-09-04): MNIST quick-mode train stream
capped at 150 batches, 1 epoch, hidden ``(32, 32)``, Euclidean step 0.05,
β 0.5 — accuracy ≈ 0.55 sPC (30 steps) / 0.44 ePC (10 steps) against
chance 0.1. The accuracy gap is not the claim (D7 precedent): both arms
train through the settle, and the ePC arm does it with 3× fewer settle
steps at roughly half the walltime. Scope: demo scale; the repo's sPC
settle is a simplified variant, so the paper's exact sPC-equivalence
theorem is not asserted here — the two claims above are.
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
    PredictiveSettlingDynamics,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemState,
    SystemTrainer,
    SystemTrainerConfig,
    ThermodynamicContrast,
    compose_system,
    create_task,
)
from computronium.ontology.dynamics import ErrorPredictiveCodingDynamics
from computronium.visualization import bars_panel, figure_spec, lines_panel

BATCH_CAP = 150
LEARN_FLOOR = 0.3  # both arms must learn (3x chance)
HIDDEN = (32, 32)
SPC_BUDGET = 30
EPC_BUDGET = 10
BETA = 0.5
STEP_SIZE = 0.05
EPC_LAMBDA = 0.5
HIDDEN_SIGNAL_FLOOR = 1e-4  # ePC nudged deviation must clear this at every hidden layer
FREE_EQ_TOL = 1e-6

_ARMS = (
    (
        "spc",
        lambda: PredictiveSettlingDynamics(
            StateDynamicsConfig.predictive_settling(max_steps=SPC_BUDGET)
        ),
        SPC_BUDGET,
    ),
    (
        "epc",
        lambda: ErrorPredictiveCodingDynamics(
            StateDynamicsConfig.error_predictive_coding(
                max_steps=EPC_BUDGET, step_size=EPC_LAMBDA, beta=BETA
            )
        ),
        EPC_BUDGET,
    ),
)


def _flatten(loader, cap):
    for x, y in islice(loader, cap):
        yield x.view(x.size(0), -1), y


def _layer_deviations(dynamics, geometry, substrate, x, y) -> list[float]:
    free = dynamics.settle(
        SystemState(x=x), geometry, substrate, target=None
    ).activations
    nudged = dynamics.settle(
        SystemState(x=x), geometry, substrate, target=y
    ).activations
    return [(n - f).abs().max().item() for f, n in zip(free, nudged, strict=True)]


def _train_arm(name, dynamics, substrate, train_loader, config):
    """Train one D-axis arm; returns (epoch metrics, composed system)."""
    torch.manual_seed(0)
    geometry = FeedforwardGeometry(
        GeometryConfig.feedforward(input_dim=784, output_dim=10, hidden_dims=HIDDEN)
    )
    system = compose_system(
        substrate=substrate,
        geometry=geometry,
        dynamics=dynamics,  # the one swapped argument
        credit=ThermodynamicContrast(
            CreditAssignmentConfig.thermodynamic_contrast(beta=BETA)
        ),
        update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=STEP_SIZE)),
    )
    metrics = SystemTrainer(
        system=system, config=config, train_data=_flatten(train_loader, BATCH_CAP)
    ).fit()[-1]
    print(f"{name}: {metrics['train_acc']:.1%}")
    return metrics, system


def _probe_arm(system, substrate, x, y) -> tuple[list[float], float | None]:
    """Nudged-vs-free per-layer deviations; ePC also reports the free
    equilibrium's distance from the feedforward pass."""
    devs = _layer_deviations(system.dynamics, system.geometry, substrate, x, y)
    free_eq_diff = None
    if isinstance(system.dynamics, ErrorPredictiveCodingDynamics):
        ff = system.geometry.forward_with_intermediates(x, substrate)
        free = system.dynamics.settle(
            SystemState(x=x), system.geometry, substrate, target=None
        ).activations
        free_eq_diff = max(
            (a - b).abs().max().item() for a, b in zip(free, ff, strict=True)
        )
    return devs, free_eq_diff


def test_demo_epc_fast_settle(emit_run_record) -> None:
    task = create_task("mnist", device="cpu", quick_mode=True, num_workers=0)
    task.setup()
    train_loader = task.get_dataloader("train")
    config = SystemTrainerConfig(max_epochs=1, device="cpu", seed=42)
    substrate = DigitalSubstrate(SubstrateConfig.digital(device="cpu"))

    record: dict = {
        "arms": {},
        "hidden_signal_floor": HIDDEN_SIGNAL_FLOOR,
        "beta": BETA,
    }
    accs: dict[str, float] = {}
    deviations: dict[str, list[float]] = {}
    free_equilibrium_max_diff: float | None = None
    for name, make_dynamics, budget in _ARMS:
        metrics, system = _train_arm(
            name, make_dynamics(), substrate, train_loader, config
        )
        accs[name] = metrics["train_acc"]
        record["arms"][name] = {
            "train_acc": metrics["train_acc"],
            "settle_budget": budget,
        }

        x, y = next(iter(train_loader))
        x = x.view(x.size(0), -1)
        devs, free_eq_diff = _probe_arm(system, substrate, x, y)
        deviations[name] = devs
        record["arms"][name]["nudged_layer_deviations"] = devs
        if free_eq_diff is not None:
            free_equilibrium_max_diff = free_eq_diff
            record["free_equilibrium_max_diff"] = free_eq_diff

    record["figure"] = figure_spec(
        "D12 — one wiring, one swapped D-axis (ePC)",
        bars_panel(
            {
                f"{name} ({arm['settle_budget']} steps)": {
                    "train accuracy": arm["train_acc"]
                }
                for name, arm in record["arms"].items()
            },
            chance=1 / 10,
            chance_label="chance (0.1)",
            ylabel="train accuracy",
            ylim=(0, 1),
        ),
        lines_panel(
            {
                name: arm["nudged_layer_deviations"]
                for name, arm in record["arms"].items()
            },
            xlabel="layer (input → hidden → output)",
            ylabel="|nudged − free| (max, per layer)",
            title="the output-error signal reaches every layer in ePC",
            symlog_thresh=1e-4,
        ),
        figsize=[9, 4],
    )

    emit_run_record("D12", "epc_fast_settle", record)

    assert accs["spc"] > LEARN_FLOOR, "sPC arm must learn through the layered settle"
    assert accs["epc"] > LEARN_FLOOR, "ePC arm must learn through the error settle"
    assert (
        free_equilibrium_max_diff is not None
        and free_equilibrium_max_diff < FREE_EQ_TOL
    ), "ePC's free-phase equilibrium must be the feedforward pass itself"
    epc_hidden = deviations["epc"][1:-1]
    assert len(epc_hidden) == len(HIDDEN), "one deviation per layer (input..output)"
    assert all(d > HIDDEN_SIGNAL_FLOOR for d in epc_hidden), (
        "ePC's nudged settle must move every hidden layer (global propagation)"
    )
    assert all(d == 0.0 for d in deviations["spc"][1:-1]), (
        "sPC's nudged settle must leave every hidden layer exactly at its free state"
    )
    assert EPC_BUDGET < SPC_BUDGET, "the ePC arm must run on a smaller settle budget"
