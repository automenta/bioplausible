"""F2 — The spiking plateau, audited (R11.5.5 slot filled).

The standing rule demanded the spiking family show one or the other:
plateau (refutation figure, same pipeline) or learning (capability
claim). History claimed plateau — "spiking at chance on MNIST" — but a
skeptical instrument audit (scripts/probes/spiking_gain_audit.py,
scripts/probes/spiking_learning.py) found the historic numbers were
confounded. Two live, separated claims:

1. **The confound (instrument defect):** with default init (scale 0.1)
   and threshold 1.0, hidden LIF layers are SILENT — per-layer spike
   fraction ≈ 0 past the first layer — so every hidden weight matrix
   receives exactly zero STDP gradient and the readout is frozen at
   random init. Chance-level accuracy under this regime says nothing
   about STDP.
2. **The plateau (mechanism, after the fix):** with ``init_scale=1.0``
   every layer spikes (fraction ≈ 0.35–0.51) and STDP gradients reach
   every weight matrix. Even then:
   a. supervised accuracy stays at chance — ``TemporalTraceCredit``
      declares ``phases=(FREE,)`` and never consumes the loss; pure STDP
      has no supervision path by construction (a category fact, not a
      defect — a supervised spiking claim needs an error term, e.g.
      reward-modulated STDP);
   b. unsupervised STDP training actively DESTROYS class structure:
      nearest-centroid readout on the settled hidden membranes drops
      0.58 (random init) → 0.19 (STDP-trained) — the runaway-gain
      positive-feedback pathology (F1 face 3), spiking edition: no gain
      control in the Hebbian rule. **The collapse survives known fix #1**
      (2026-09-05 audit): homeostatic synaptic scaling holds hidden row
      norms at their init value, yet the readout still collapses — the
      STDP fixed point itself destroys class structure, not gain growth.
      OPEN pending the supervised-error-term audit (reward-modulated
      STDP); no boundary verdict before then (R11.5.5a).

This is the honest-failure slot, filled: the plateau is real but for
sharper reasons than history claimed, and the instrument now
demonstrates the difference live.
"""

import copy
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
DEPTH = 4
SETTLE_STEPS = 10
BATCH_CAP = 60
LR = 0.2
CHANCE = 0.1
SILENT_FRACTION = 0.01


def _flatten(loader, cap):
    for x, y in islice(loader, cap):
        yield x.view(x.size(0), -1), y


def _build(depth: int, init_scale: float, homeostatic_target: float | None = None):
    torch.manual_seed(0)
    return compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device="cpu")),
        geometry=FeedforwardGeometry(
            GeometryConfig.feedforward(
                input_dim=784,
                output_dim=10,
                hidden_dims=(WIDTH,) * depth,
                init_scale=init_scale,
            )
        ),
        dynamics=SpikeIntegrationDynamics(
            StateDynamicsConfig.spike_integration(max_steps=SETTLE_STEPS)
        ),
        credit=TemporalTraceCredit(
            CreditAssignmentConfig.temporal_trace(
                homeostatic_scaling=homeostatic_target is not None,
                homeostatic_target=homeostatic_target or 1.0,
            )
        ),
        update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=LR)),
    )


def _row_norms(system) -> list[float]:
    return [
        float(p.norm(dim=1).mean())
        for n, p in system.geometry.params.items()
        if "weight" in n and p.ndim == 2
    ]


def _settle(system, x):
    return system.dynamics.settle(
        SystemState(x=x), system.geometry, system.substrate, target=None
    )


def _spike_fractions(system, x) -> list[float]:
    state = _settle(system, x)
    return [
        sum(r.sum().item() for r in layer) / (layer[0].numel() * len(layer))
        for layer in state.spike_rasters
    ]


def _credit_norms(system, x) -> list[float]:
    state = _settle(system, x)
    grads = system.credit.compute_pseudo_gradient(
        {Phase.FREE: state}, None, system.geometry
    )
    return [g.norm().item() for g in grads]


def _centroid_readout(system, train_data, eval_data) -> float:
    def features(data):
        feats, labels = [], []
        for x, y in data:
            feats.append(_settle(system, x).activations[-2])
            labels.append(y)
        return torch.cat(feats), torch.cat(labels)

    f, y = features(train_data)
    centroids = torch.stack([f[y == k].mean(0) for k in range(10)])
    e, ye = features(eval_data)
    return (torch.cdist(e, centroids).argmin(1) == ye).float().mean().item()


def _homeostatic_arm(healthy, config, train_data, eval_data) -> dict:
    """Homeostatic-scaling audit (2026-09-05, plan item 3): synaptic
    scaling (known fix #1) is LIVE — row norms held at the target —
    yet the readout still collapses. The STDP fixed point itself
    destroys class structure; gain control does not rescue it. Verdict
    OPEN pending the supervised-error-term audit (reward-modulated
    STDP) — no boundary verdict before then (R11.5.5a)."""
    h_target = _row_norms(healthy)[1]  # hold hidden rows at their init norm
    h_system = _build(DEPTH, 1.0, homeostatic_target=h_target)
    SystemTrainer(system=h_system, config=config, train_data=train_data).fit()
    return {
        "target": h_target,
        "row_norms": _row_norms(h_system),
        "readout": _centroid_readout(h_system, train_data, eval_data),
    }


def _stdp_readout_arm(healthy, config, train_data, eval_data) -> dict:
    readout = {"random_init": _centroid_readout(healthy, train_data, eval_data)}
    stdp_system = _build(DEPTH, 1.0)
    SystemTrainer(system=stdp_system, config=config, train_data=train_data).fit()
    readout["stdp_trained"] = _centroid_readout(stdp_system, train_data, eval_data)
    return readout


def test_demo_spiking_plateau(emit_run_record) -> None:
    task = create_task("mnist", device="cpu", quick_mode=True, num_workers=0)
    task.setup()
    loader = task.get_dataloader("train")
    config = SystemTrainerConfig(max_epochs=1, device="cpu", seed=42)
    torch.manual_seed(0)  # seed BEFORE the loader draw (D8 trap)
    train_data = list(_flatten(loader, BATCH_CAP))
    eval_data = list(_flatten(loader, 30))
    x, _ = train_data[0]

    record: dict = {"chance": CHANCE, "width": WIDTH, "depth": DEPTH}

    silent = _build(DEPTH, 0.1)
    healthy = _build(DEPTH, 1.0)

    fractions = {
        tag: _spike_fractions(sys_, x)
        for tag, sys_ in (("default", silent), ("init1.0", healthy))
    }
    norms = {
        tag: _credit_norms(sys_, x)
        for tag, sys_ in (("default", silent), ("init1.0", healthy))
    }
    record["spike_fractions"] = fractions
    record["credit_norms"] = norms

    supervised = SystemTrainer(
        system=copy.deepcopy(healthy), config=config, train_data=train_data
    ).fit()[-1]["train_acc"]
    record["supervised_train_acc"] = supervised

    readout = _stdp_readout_arm(healthy, config, train_data, eval_data)
    record["feature_readout"] = readout

    record["homeostatic_audit"] = _homeostatic_arm(
        healthy, config, train_data, eval_data
    )

    emit_run_record("F2", "spiking_plateau", record)

    assert fractions["default"][1] < SILENT_FRACTION, (
        "default init must leave hidden layers silent (the confound)"
    )
    assert min(fractions["init1.0"]) > 0.1, "init_scale=1.0 must spike at every layer"
    assert all(
        n <= 0.0 for n in norms["default"][1:]
    ), (  # norms are non-negative: <= 0 iff exactly zero
        "silent layers must receive exactly zero STDP gradient"
    )
    assert min(norms["init1.0"]) > 0, "healthy spiking must reach every weight matrix"
    assert supervised < 0.15, (
        "pure STDP has no supervision path — supervised accuracy stays at chance"
    )
    assert readout["random_init"] > 0.3, (
        "random-init membrane features must carry class structure (>> chance)"
    )
    assert readout["stdp_trained"] < readout["random_init"] - 0.1, (
        "unsupervised STDP must degrade the readout (runaway gain)"
    )

    homeo = record["homeostatic_audit"]
    assert all(
        abs(n - homeo["target"]) < 0.5 * homeo["target"] for n in homeo["row_norms"][1:]
    ), (
        "synaptic scaling must hold hidden row norms at the target "
        f"(got {homeo['row_norms']}) — the fix must be live, not a no-op"
    )
    assert homeo["readout"] < readout["random_init"] - 0.1, (
        "collapse must persist under homeostatic scaling — the STDP fixed "
        "point itself destroys class structure (OPEN pending the "
        "reward-modulated STDP audit; no wall verdict before then)"
    )
