"""F1 — The depth boundary has four faces (failure manifesto).

One wiring, one figure: the four measured ways learning dies with depth,
live at demo scale, same pipeline, same terms:

1. **Depth wall (D-axis credit):** under identical ``SystemTrainer``
   wiring on MNIST quick-mode, backprop decays through depth
   (0.72 → 0.50 → 0.11; flat across an lr grid 0.02–0.2 — not an lr
   artifact) and the local contrastive rule (sPC) walls at chance by
   depth 8 at this budget. Skeptical audit
   (scripts/probes/failure_manifesto_audit.py): the wall's mechanism
   under our layered settle is last-layer-only training — per-layer
   credit norms are exactly zero for every hidden weight matrix
   (asserted live) — and budget softens it (0.21 at 60 settle steps).
   **The wall survives known fix #1** (2026-09-05 audit): ePC's error
   reparameterization DOES reach every hidden layer at depth 8 (credit
   norms > 0, asserted live) yet still walls — the contrastive signal
   decays geometrically through depth. Candidate real wall; verdict
   OPEN pending fix #2, the jpc-faithful trainer regime (Adam, β grid,
   steps=H).
2. **μPC no lift under our trainer:** depth-scaled init (``mupc``)
   rescues nothing at any depth under the computronium trainer —
   0.124 vs 0.105 at depth 8. The honest statement: no lift, OPEN
   pending the jpc-faithful port (Adam, β grid, steps=H); never quote
   "μPC refuted".
3. **Runaway gain (unnormalized local chain):** the hebbian tile
   chain's init forward gain compounds super-exponentially — norm ratio
   last/first 1.35 → 7.2e2 → 3.2e5 at depths 10/50/100; one local
   update NaNs it (probe: scripts/probes/deep_hebbian_chain.py).
4. **Subspace collapse (normalized Oja chain):** activity renorm kills
   the runaway gain, but the 10-class readout decays 0.99 → 0.23 toward
   chance (0.1) through the chain while the first layer stays ~1.0 —
   the chain transmits its dominant direction and discards the rest
   (R11.3.14).

This is the library's first *finding* figure (CP-6): not a capability
demonstration but a consolidated failure boundary for local and
energy-based learning at depth.
"""

from itertools import islice
from typing import Literal

import torch

from computronium import (
    BackpropCredit,
    CreditAssignmentConfig,
    DigitalSubstrate,
    ErrorPredictiveCodingDynamics,
    EuclideanUpdate,
    FeedforwardGeometry,
    GeometryConfig,
    InstantaneousDynamics,
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
from computronium.core.local_learning.builder import TileAlgorithm, TileAlgorithmConfig
from computronium.models.native import DeepHebbianChain
from computronium.ontology.credit import Phase
from computronium.visualization import figure_spec, lines_panel

WIDTH = 32
TRAIN_DEPTHS = (2, 4, 8)
BATCH_CAP = 60
SETTLE_STEPS = 15
LR = 0.2
BETA = 0.5
RUNAWAY_DEPTHS = (10, 50, 100)
OJA_DEPTHS = (1, 10, 50, 100)
EPC_DEPTHS = (2, 8)
EPC_SETTLE_STEPS = 5  # 1/3 of SETTLE_STEPS — the D12 budget regime
CHANCE = 0.1

InitScheme = Literal["default", "mupc"]

_ARMS: tuple[tuple[str, InitScheme], ...] = (
    ("bp", "default"),
    ("spc", "default"),
    ("spc_mupc", "mupc"),
)


def _flatten(loader, cap):
    for x, y in islice(loader, cap):
        yield x.view(x.size(0), -1), y


def _train_acc(
    depth: int, name: str, substrate, config, train_data
) -> tuple[float, object]:
    torch.manual_seed(0)
    geometry = FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=784,
            output_dim=10,
            hidden_dims=(WIDTH,) * depth,
            init_scheme=dict(_ARMS)[name],
        )
    )
    if name == "bp":
        dynamics = InstantaneousDynamics(StateDynamicsConfig.instantaneous())
        credit = BackpropCredit()
    else:
        dynamics = PredictiveSettlingDynamics(
            StateDynamicsConfig.predictive_settling(max_steps=SETTLE_STEPS)
        )
        credit = ThermodynamicContrast(
            CreditAssignmentConfig.thermodynamic_contrast(beta=BETA)
        )
    system = compose_system(
        substrate=substrate,
        geometry=geometry,
        dynamics=dynamics,
        credit=credit,
        update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=LR)),
    )
    metrics = SystemTrainer(system=system, config=config, train_data=train_data).fit()[
        -1
    ]
    return metrics["train_acc"], system


def _spc_credit_norms(system, substrate, loader) -> list[float]:
    """Per-layer thermo-contrast norms after one free/nudged settle pair.

    The mechanism probe: under the layered settle the contrast reaches only
    the LAST weight matrix — hidden norms are exactly zero at every depth
    (audit: scripts/probes/failure_manifesto_audit.py). The sPC wall is
    last-layer-only training, not hidden-credit decay.
    """
    x, y = next(iter(loader))
    x = x.view(x.size(0), -1)
    free = system.dynamics.settle(
        SystemState(x=x), system.geometry, substrate, target=None
    )
    nudged = system.dynamics.settle(
        SystemState(x=x), system.geometry, substrate, target=y
    )
    grads = system.credit.compute_pseudo_gradient(
        {Phase.FREE: free, Phase.NUDGED: nudged}, None, system.geometry
    )
    return [g.norm().item() for g in grads]


def _runaway_ratio(depth: int) -> float:
    torch.manual_seed(0)
    model = TileAlgorithm(
        TileAlgorithmConfig(
            input_dim=16,
            output_dim=10,
            neurons_per_tile=16,
            tiles_per_layer=4,
            num_hidden_layers=depth,
            algorithm="hebbian",
            mode="hebbian",
            free_steps=5,
            nudged_steps=5,
            learning_rate=0.001,
            beta=0.1,
            step_size=0.1,
        )
    )
    acts = model.free_phase(torch.randn(4, 16))
    norms = [
        torch.cat([acts[tid] for tid in layer_tiles], dim=1).norm(dim=1).mean().item()
        for layer_tiles in model.graph.layer_ids
    ]
    return norms[-1] / (norms[0] or 1.0)


def _oja_readout(depth: int, x_train, y_train, x_eval, y_eval) -> float:
    torch.manual_seed(0)
    model = DeepHebbianChain(32, 32, depth, learning_rate=1e-3)
    for i in range(0, y_train.shape[0], 64):
        model.local_update(x_train[i : i + 64])
    acts_train = model(x_train)[depth]
    acts_eval = model(x_eval)[depth]
    centroids = torch.stack([acts_train[y_train == k].mean(0) for k in range(10)])
    return (torch.cdist(acts_eval, centroids).argmin(1) == y_eval).float().mean().item()


def _direction_task(
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    basis = torch.linalg.qr(torch.randn(32, 10, generator=generator))[0]
    means = basis.T * 3.0
    targets = torch.randint(0, 10, (2048,), generator=generator)
    x_train = means[targets] + torch.randn(2048, 32, generator=generator) * 0.5
    eval_targets = torch.randint(0, 10, (512,), generator=generator)
    x_eval = means[eval_targets] + torch.randn(512, 32, generator=generator) * 0.5
    return x_train, targets, x_eval, eval_targets


def _train_all_arms(loader, substrate, config) -> tuple[dict[str, list[float]], object]:
    train_accs: dict[str, list[float]] = {}
    spc_system = None
    for depth in TRAIN_DEPTHS:
        torch.manual_seed(0)  # seed BEFORE the loader draw (D8 trap)
        train_data = list(_flatten(loader, BATCH_CAP))
        for name, _ in _ARMS:
            acc, system = _train_acc(depth, name, substrate, config, train_data)
            train_accs.setdefault(name, []).append(acc)
            if name == "spc" and depth == TRAIN_DEPTHS[-1]:
                spc_system = system
            print(f"depth {depth:>2} {name:>8}: {acc:.3f}")
    return train_accs, spc_system


def _local_arm_record(name: str, init: str, train_accs: dict[str, list[float]]) -> dict:
    return {
        "init": init,
        "depths": list(TRAIN_DEPTHS),
        "train_acc": train_accs[name],
    }


def _epc_arm(loader, substrate, config) -> dict:
    """ePC depth sweep (2026-09-05 audit, plan item 2): the wall survives
    its first known fix. ePC's error reparameterization DOES reach every
    hidden layer (min credit norm > 0 at depth 8 — the sPC pathology is
    gone) yet learning still walls at depth 8 — the contrastive credit
    decays geometrically through depth. Verdict stays OPEN pending the
    jpc-faithful trainer regime (Adam, β grid, steps=H)."""
    out: dict = {"depths": [], "train_acc": [], "min_hidden_credit_norm": []}
    for depth in EPC_DEPTHS:
        torch.manual_seed(0)
        train_data = list(_flatten(loader, BATCH_CAP))
        geometry = FeedforwardGeometry(
            GeometryConfig.feedforward(
                input_dim=784, output_dim=10, hidden_dims=(WIDTH,) * depth
            )
        )
        system = compose_system(
            substrate=substrate,
            geometry=geometry,
            dynamics=ErrorPredictiveCodingDynamics(
                StateDynamicsConfig.error_predictive_coding(
                    max_steps=EPC_SETTLE_STEPS, step_size=0.5, beta=BETA
                )
            ),
            credit=ThermodynamicContrast(
                CreditAssignmentConfig.thermodynamic_contrast(beta=BETA)
            ),
            update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=LR)),
        )
        acc = SystemTrainer(system=system, config=config, train_data=train_data).fit()[
            -1
        ]["train_acc"]
        x, y = train_data[0]
        free = system.dynamics.settle(
            SystemState(x=x), geometry, substrate, target=None
        )
        nudged = system.dynamics.settle(SystemState(x=x), geometry, substrate, target=y)
        grads = system.credit.compute_pseudo_gradient(
            {Phase.FREE: free, Phase.NUDGED: nudged}, None, geometry
        )
        norms = [g.norm().item() for g in grads]
        out["depths"].append(depth)
        out["train_acc"].append(acc)
        out["min_hidden_credit_norm"].append(min(norms[:-1]))
        print(f"depth {depth:>2} {'epc':>8}: {acc:.3f}")
    return out


def _oja_arm() -> list[float]:
    generator = torch.Generator().manual_seed(1)
    x_train, targets, x_eval, eval_targets = _direction_task(generator)
    return [_oja_readout(d, x_train, targets, x_eval, eval_targets) for d in OJA_DEPTHS]


def test_demo_failure_manifesto(emit_run_record) -> None:
    task = create_task("mnist", device="cpu", quick_mode=True, num_workers=0)
    task.setup()
    loader = task.get_dataloader("train")
    substrate = DigitalSubstrate(SubstrateConfig.digital(device="cpu"))
    config = SystemTrainerConfig(max_epochs=1, device="cpu", seed=42)

    record: dict = {"arms": {}, "chance": CHANCE, "width": WIDTH}
    train_accs, spc_system = _train_all_arms(loader, substrate, config)
    for name, init in _ARMS:
        record["arms"][name] = _local_arm_record(name, init, train_accs)

    credit_norms = _spc_credit_norms(spc_system, substrate, loader)
    record["arms"]["spc"]["credit_norms"] = credit_norms

    epc = _epc_arm(loader, substrate, config)
    record["arms"]["epc"] = epc

    ratios = [_runaway_ratio(d) for d in RUNAWAY_DEPTHS]
    record["arms"]["hebbian_runaway"] = {
        "depths": list(RUNAWAY_DEPTHS),
        "norm_ratio": ratios,
    }

    readouts = _oja_arm()
    record["arms"]["oja_collapse"] = {
        "depths": list(OJA_DEPTHS),
        "readout_acc": readouts,
    }

    record["figure"] = figure_spec(
        "F1 — the depth boundary: decay, walls, runaway gain, collapse",
        lines_panel(
            {
                name: record["arms"][name]["train_acc"]
                for name in ("bp", "spc", "spc_mupc")
            },
            x=list(TRAIN_DEPTHS),
            chance=CHANCE,
            chance_label=f"chance ({CHANCE})",
            xlabel="depth",
            ylabel="train accuracy",
            title="backprop decays, sPC walls (credit: last layer only), μPC no lift",
            legend_loc="upper right",
        ),
        lines_panel(
            {"norm ratio": record["arms"]["hebbian_runaway"]["norm_ratio"]},
            x=list(RUNAWAY_DEPTHS),
            log_y=True,
            xlabel="depth",
            ylabel="forward norm ratio (last/first)",
            title="unnormalized hebbian chain: runaway gain",
        ),
        lines_panel(
            {"readout": record["arms"]["oja_collapse"]["readout_acc"]},
            x=list(OJA_DEPTHS),
            chance=CHANCE,
            chance_label=f"chance ({CHANCE})",
            xlabel="depth",
            ylabel="10-class readout (last layer)",
            title="normalized Oja chain: subspace collapse",
        ),
        figsize=[14, 4],
    )

    emit_run_record("F1", "failure_manifesto", record)

    _assert_manifesto(train_accs, ratios, readouts, credit_norms, epc)


def _assert_manifesto(acc, ratios, readouts, credit_norms, epc) -> None:
    accmap = {
        name: dict(zip(TRAIN_DEPTHS, accs, strict=True)) for name, accs in acc.items()
    }

    assert accmap["bp"][8] < accmap["bp"][2] - 0.3, "backprop must decay with depth"
    assert accmap["bp"][8] < 0.3
    assert accmap["spc"][8] < accmap["spc"][2] - 0.2, (
        "local credit must wall with depth"
    )
    assert accmap["spc"][8] < 0.2, "sPC is at chance by depth 8"
    assert all(n == 0 for n in credit_norms[:-1]) and credit_norms[-1] > 0, (
        "the layered settle's contrast reaches only the LAST weight matrix "
        "(mechanism probe — the wall is last-layer-only training; budget "
        "softens it: 0.21 at 60 settle steps, audit probe 2026-09-04)"
    )
    for depth in TRAIN_DEPTHS:
        assert accmap["spc_mupc"][depth] <= accmap["spc"][depth] + 0.05, (
            f"μPC shows no lift at depth {depth} under our trainer (OPEN, not refuted)"
        )
    assert accmap["spc_mupc"][8] < 0.2

    # ePC audit (2026-09-05): the known fix #1 (error reparameterization)
    # repairs the MECHANISM but not the wall — the verdict is a candidate
    # real wall, pending the jpc-faithful trainer regime (fix #2).
    epcmap = dict(zip(epc["depths"], epc["train_acc"], strict=True))
    epc_norms = dict(zip(epc["depths"], epc["min_hidden_credit_norm"], strict=True))
    assert epc_norms[8] > 0, (
        "ePC's credit must reach the hidden layers at depth 8 — "
        "the sPC last-layer-only pathology is the thing ePC fixes"
    )
    assert epcmap[2] > 0.3, "ePC must learn at shallow depth (positive control)"
    assert epcmap[8] <= accmap["spc"][8] + 0.05, (
        "ePC also walls at depth 8 — the wall survives fix #1; verdict "
        "OPEN pending the jpc-faithful trainer regime (Adam, β grid, steps=H)"
    )

    assert ratios[2] > ratios[1] > ratios[0], (
        "runaway gain compounds super-exponentially"
    )
    assert ratios[2] > 100.0, ratios

    assert readouts[0] > 0.9, "depth-1 Oja readout must be near-perfect"
    assert readouts[1] < readouts[0] - 0.3, "the subspace collapse must be real"
    assert all(r > CHANCE for r in readouts), "collapse decays toward, not to, chance"
