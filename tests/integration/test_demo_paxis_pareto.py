"""F3 — The P-axis Pareto, audited: primitives realized, confound still live.

CP-6 Path B, E-1 smoke promoted, mechanism-audited (R11.5.5a), then the
audit's registered realization gap CLOSED (2026-09-05) and re-audited.
Measured facts, locked live below:

1. RoutingPlasticity is REALIZED: the gate drive is a fixed per-gate
   input projection (gates differentiate across units AND samples) and
   modulate applies per-unit sigmoid masks (per-layer fixed gate→unit
   projection) — the old constant-0.5 scalar gain is gone. The per-unit
   mask std is 0.081 at this regime (asserted live > 0.05): real
   per-sample, per-unit routing over the network's units (the flat-MLP
   re-spec of pathway gating — a dense geometry has no distinct physical
   pathways to mask).
2. The retention advantage SURVIVES realization (routing 0.273 vs null
   0.194, ≥ null 4/5 seeds) — but the lr-MATCHED control (E-1 pilot,
   promoted) DISSOLVES it: routing's effective step is exactly half of
   null's (displacement 0.0014 vs 0.0028, matched null lr ≈ 0.0154 =
   the mask-mean-0.5 prediction) and null@matched retains 0.294. The
   ordering is effective learning rate ALONE — no routing mechanism
   claim is quotable from retention (asserted live). FastWeight's step
   size matches null's (0.032 ≈ 0.03): its retention deficit
   (0.155 vs 0.194) is REAL — the injection genuinely hurts at this
   scale (also asserted).
3. FastWeightPlasticity is REALIZED: ψ steps on the first phase's
   SETTLED activity (the pipeline passes the settled output as post —
   the old contract received the raw target, making the modulation a
   target-correlated bias). The modulation stays gradient-live: θ
   separates from null by ~3.4%/episode from identical inits (monotone,
   asserted). Retention lands below null (0.155 vs 0.194) and its
   episode latency carries the ψ-update cost the compute proxy cannot
   see (asserted live > 1.1× null).

What the Pareto itself shows (5 seeds, registered A40/B40 regime): the
scalar axes separate the arms (retention, basin width, ψ-capacity) —
the E-1 gate for the registered-scale campaign. Mechanisms realized,
and the retention axis is now CONFOUNDED-BY-CONSTRUCTION: its ordering
is effective-lr alone (lr-matched audit, locked live). The registered
campaign must therefore measure retention at matched effective lr per
arm — the campaign-design deliverable of this pilot — before any
P-axis mechanism figure is drawn.
"""

import itertools
import time

import numpy as np
import torch

from computronium.core.campaign.evaluation import (
    episode_batch,
    evaluate_episode,
    probe_episode,
)
from computronium.experiments.joint.forgetting_trial import (
    PROBE_EPISODE_BASE,
    PROBED_SEGMENT,
    TRIAL_CAMPAIGN_ID,
    TrialConfig,
    _compose,
)
from computronium.state import CompositeState
from computronium.visualization._demo_api import (
    figure_spec,
    lines_panel,
    scatter_panel,
)

ARMS = ("null", "fast_weights", "routing")
SEEDS = tuple(range(5))
SEGMENTS = (("A", 40), ("B", 40))
RADII = (0.25, 0.5, 1.0, 2.0, 4.0)
BASIN_DRAWS = 8
BASIN_EPISODES = 8
DEVICE = "cpu"
CHANCE = 1 / 8
LR = 0.03
LR_CONTROL = 0.015  # null at the routing-gain effective lr (audit control)

_CONFIG = TrialConfig(segments=SEGMENTS, seeds=SEEDS, device=DEVICE, lr=LR)


def _coordinate(arm: str) -> str:
    return f"digital/feedforward/instantaneous/{arm}/gradient/euclidean"


def _basin_curve(joint, coordinate: str, seed: int) -> list[float]:
    """Per-radius agreement of perturbed-input argmax with the unperturbed
    prediction, on held-out segment-A batches (probe episode space)."""
    curves: dict[float, list[float]] = {r: [] for r in RADII}
    device = joint.device
    with torch.no_grad():
        for i in range(BASIN_EPISODES):
            x, _ = episode_batch(
                PROBE_EPISODE_BASE + 100 + i,
                task_name="synthetic",
                batch_size=_CONFIG.batch_size,
                input_dim=_CONFIG.input_dim,
                num_classes=_CONFIG.num_classes,
                teacher_key=(TRIAL_CAMPAIGN_ID, coordinate, seed, PROBED_SEGMENT),
                teacher_noise=_CONFIG.teacher_noise,
            )
            x = x.to(device)
            base_pred = joint.forward(x).argmax(dim=-1)
            gen = torch.Generator().manual_seed(seed * 10_000 + i)
            for _ in range(BASIN_DRAWS):
                direction = torch.nn.functional.normalize(
                    torch.randn(x.shape, generator=gen).to(device), dim=-1
                )
                for radius in RADII:
                    pert = joint.forward(x + radius * direction).argmax(dim=-1)
                    curves[radius].append((pert == base_pred).float().mean().item())
    return [float(np.mean(curves[r])) for r in RADII]


def _probe(joint, coordinate: str, episode: int, seed: int) -> float:
    """Held-out segment-A probe: mean accuracy over 8 probe episodes
    (PROBE_EPISODE_BASE space — a single 16-sample batch is noise)."""
    return float(
        np.mean([
            probe_episode(
                joint,
                coordinate=coordinate,
                task_name="synthetic",
                campaign_id=TRIAL_CAMPAIGN_ID,
                episode=PROBE_EPISODE_BASE + i,
                batch_size=_CONFIG.batch_size,
                input_dim=_CONFIG.input_dim,
                num_classes=_CONFIG.num_classes,
                seed=seed,
                stationary_teacher=True,
                teacher_noise=_CONFIG.teacher_noise,
                segment=PROBED_SEGMENT,
            )
            for i in range(8)
        ])
    )


def _walk_arm(arm: str, *, lr: float = LR) -> tuple[float, dict]:
    """One arm's persistent-θ walk: resource vector + stability scalars."""
    coordinate = _coordinate(arm)
    config = (
        _CONFIG
        if lr == LR
        else TrialConfig(segments=SEGMENTS, seeds=SEEDS, device=DEVICE, lr=lr)
    )
    latencies: list[float] = []
    computes: list[float] = []
    a_mastery: list[float] = []
    a_retained: list[float] = []
    basin_curves: list[list[float]] = []
    for seed in SEEDS:
        joint = _compose(coordinate, config)
        episode = 0
        boundary: list[float] = []
        for segment, n in config.segments:
            for _ in range(n):
                started = time.perf_counter()
                record, _ = evaluate_episode(
                    joint,
                    coordinate=coordinate,
                    task_name="synthetic",
                    campaign_id=TRIAL_CAMPAIGN_ID,
                    episode=episode,
                    batch_size=config.batch_size,
                    input_dim=config.input_dim,
                    num_classes=config.num_classes,
                    guard_threshold=None,
                    seed=seed,
                    stationary_teacher=True,
                    teacher_noise=config.teacher_noise,
                    segment=segment,
                )
                latencies.append(time.perf_counter() - started)
                computes.append(record.resources.compute)
                episode += 1
            boundary.append(_probe(joint, coordinate, episode, seed))
        a_mastery.append(boundary[0])
        a_retained.append(boundary[-1])
        basin_curves.append(_basin_curve(joint, coordinate, seed))
    latency_ms = float(np.median(latencies)) * 1e3
    data = {
        "compute": float(np.mean(computes)),
        "psi_capacity": float(
            sum(
                (
                    _compose(coordinate, _CONFIG).plasticity.config.plastic_state_dims
                    or {}
                ).values()
            )
        ),
        "a_mastery": a_mastery,
        "a_retained": a_retained,
        "retention_mean": float(np.mean(a_retained)),
        "retention_sd": float(np.std(a_retained, ddof=1)),
        "basin_curve_mean": [
            float(np.mean(col)) for col in zip(*basin_curves, strict=True)
        ],
    }
    return latency_ms, data


def _gate_strength_trace() -> dict:
    """Routing's gate trace along the walk + the per-unit mask spread that
    IS the realized mechanism (deterministic: gate_logits depend only on
    the seeded inputs)."""
    coordinate = _coordinate("routing")
    joint = _compose(coordinate, _CONFIG)
    plasticity = joint.plasticity
    x, _ = episode_batch(
        0,
        task_name="synthetic",
        batch_size=_CONFIG.batch_size,
        input_dim=_CONFIG.input_dim,
        num_classes=_CONFIG.num_classes,
        teacher_key=(TRIAL_CAMPAIGN_ID, coordinate, 0, PROBED_SEGMENT),
        teacher_noise=_CONFIG.teacher_noise,
    )
    x = x.to(joint.device)
    psi = plasticity.initial_psi(joint.context, batch_size=16)
    trace = []
    mask_stds = []
    with torch.no_grad():
        for _ in range(80):
            psi = plasticity.step(
                psi,
                CompositeState(activity={"x": x, "y": x}, plastic=psi, substrate={}),
                joint.context,
            )
            trace.append(float(torch.sigmoid(psi["gate_logits"]).mean()))
            proj = plasticity._unit_projection(0, 16, psi["gate_logits"].device)
            mask = torch.sigmoid(psi["gate_logits"] @ proj)
            mask_stds.append(float(mask.std(dim=0).mean()))
    return {
        "mean": float(np.mean(trace)),
        "spread": float(np.ptp(trace)),
        "mask_unit_std": float(np.mean(mask_stds)),
    }


def _theta_divergence() -> tuple[float, bool]:
    """Relative θ gap between the fw and null trajectories over 5 paired
    episodes from IDENTICAL inits (the modulation's gradient effect).

    _compose seeds init per coordinate, so fw starts from null's exact
    init — otherwise the gap measures the init draw, not the modulation
    (the first audit draft made exactly that mistake: a flat "1.42
    divergence" that was the init draw, not the primitive).
    """
    coords = {arm: _coordinate(arm) for arm in ("null", "fast_weights")}
    joints = {arm: _compose(c, _CONFIG) for arm, c in coords.items()}
    with torch.no_grad():
        for name, param in joints["null"].geometry.params.items():
            joints["fast_weights"].geometry.params[name].data.copy_(param.data)

    def gap() -> float:
        return float(
            np.mean([
                (
                    joints["fast_weights"].geometry.params[n]
                    - joints["null"].geometry.params[n]
                )
                .norm()
                .item()
                / joints["null"].geometry.params[n].norm().item()
                for n in joints["null"].geometry.params
            ])
        )

    gaps = []
    for episode in range(5):
        for arm, joint in joints.items():
            x, y = episode_batch(
                episode,
                task_name="synthetic",
                batch_size=_CONFIG.batch_size,
                input_dim=_CONFIG.input_dim,
                num_classes=_CONFIG.num_classes,
                teacher_key=(TRIAL_CAMPAIGN_ID, coords[arm], 0, PROBED_SEGMENT),
                teacher_noise=_CONFIG.teacher_noise,
            )
            joint.train_step(x, y)
        gaps.append(gap())
    return gaps[-1], all(b > a for a, b in itertools.pairwise(gaps))


def _theta_displacement(coordinate: str, lr: float) -> float:
    """Mean relative per-parameter ||Δθ|| over 3 episodes from identical
    fixed init (seed 11) — the arm's effective step size, independent of
    the walk trajectory."""
    config = TrialConfig(segments=SEGMENTS, seeds=SEEDS, device=DEVICE, lr=lr)
    displacements = []
    for episode in range(3):
        torch.manual_seed(11)
        joint = _compose(coordinate, config)
        before = {n: p.detach().clone() for n, p in joint.geometry.params.items()}
        x, y = episode_batch(
            episode,
            task_name="synthetic",
            batch_size=_CONFIG.batch_size,
            input_dim=_CONFIG.input_dim,
            num_classes=_CONFIG.num_classes,
            teacher_key=(TRIAL_CAMPAIGN_ID, coordinate, 0, PROBED_SEGMENT),
            teacher_noise=_CONFIG.teacher_noise,
        )
        joint.train_step(x, y)
        displacements.append(
            float(
                np.mean([
                    (joint.geometry.params[n] - before[n]).norm().item()
                    / before[n].norm().item()
                    for n in before
                ])
            )
        )
    return float(np.mean(displacements))


def _matched_lr(arm: str, grid: tuple[float, ...]) -> float:
    """The null lr whose single-episode θ displacement equals the arm's
    (log-linear interpolation on the displacement grid — displacement is
    monotone in lr)."""
    arm_disp = _theta_displacement(_coordinate(arm), LR)
    grid_disp = [(lr, _theta_displacement(_coordinate("null"), lr)) for lr in grid]
    for (lr0, d0), (lr1, d1) in itertools.pairwise(grid_disp):
        if d0 <= arm_disp <= d1:
            t = (np.log(arm_disp) - np.log(d0)) / (np.log(d1) - np.log(d0))
            return float(lr0 * (lr1 / lr0) ** t)
    return grid_disp[-1][0] if arm_disp > grid_disp[-1][1] else grid_disp[0][0]


def _lr_matched_audit(arms: dict[str, dict]) -> dict:
    """Per-arm lr-matched control: run the null walk at each arm's
    effective lr and compare retention. Routing's masks halve the
    effective step (matched ≈ 0.0154); fast-weights' step matches null's.
    The E-1 pilot verdict (2026-09-05): routing's retention advantage
    DISSOLVES at matched lr — the ordering is effective-lr alone."""
    audit: dict = {}
    for arm in ("fast_weights", "routing"):
        lr_m = _matched_lr(arm, (0.0075, 0.015, 0.03, 0.06))
        arm_retained = arms[arm]["retention_mean"]
        null_retained = _walk_arm("null", lr=lr_m)[1]["retention_mean"]
        audit[arm] = {
            "matched_null_lr": lr_m,
            "arm_retained": arm_retained,
            "null_retained": null_retained,
            "advantage": arm_retained - null_retained,
        }
    return audit


def _print_lr_matched(audit: dict) -> None:
    for arm, m in audit.items():
        print(
            f"lr-matched {arm:>13}: null@lr{m['matched_null_lr']:.4f} "
            f"retained {m['null_retained']:.3f} vs arm {m['arm_retained']:.3f} "
            f"(advantage {m['advantage']:+.3f})"
        )


def _assert_lr_matched(audit: dict) -> None:
    # The lr-matched audit (E-1 pilot, promoted 2026-09-05): routing's
    # masks halve its effective step (matched null lr ≈ 0.015) and the
    # retention advantage DISSOLVES against null@matched — no routing
    # mechanism claim is quotable from retention. Fast-weights' step
    # matches null's and its retention DEFICIT is real. If either flips,
    # the mechanism story changed: re-audit before quoting anything.
    routing_m = audit["routing"]
    assert routing_m["advantage"] <= 0.01, (
        f"routing retains MORE than null at matched lr "
        f"({routing_m['advantage']:+.3f}) — the effective-lr confound no "
        "longer explains the ordering: re-audit before quoting"
    )
    fw_m = audit["fast_weights"]
    assert fw_m["advantage"] <= 0, (
        f"fast-weights' matched-control deficit flipped positive "
        f"({fw_m['advantage']:+.3f}) — re-audit"
    )


def test_demo_paxis_pareto(emit_run_record) -> None:
    # Mechanism-audit instruments (R11.5.5a): the primitives are currently
    # gain control / bias injection, not their advertised mechanisms.
    gate = _gate_strength_trace()
    fw_theta_gap, fw_gap_grows = _theta_divergence()
    lr_control = _walk_arm("null", lr=LR_CONTROL)[1]
    arm_walks = {arm: _walk_arm(arm) for arm in ARMS}
    latencies = {arm: latency for arm, (latency, _) in arm_walks.items()}
    arms = {arm: data for arm, (_, data) in arm_walks.items()}
    lr_matched = _lr_matched_audit(arms)
    # Walltime is not reproducible — the gallery lock hashes record["data"]
    # at 1e-6 and even quantized latency RATIOS flip bins run-to-run when a
    # ψ cost sits at the noise floor (routing's does). So record["data"]
    # carries NO walltime: the Pareto figure's cost axis is ψ-capacity
    # (deterministic, the doctrine's own plastic-state-capacity axis), the
    # latency-ordering claims are asserted live on in-run values below, and
    # absolute medians stay on stdout.
    record: dict = {
        "seeds": list(SEEDS),
        "segments": [list(s) for s in SEGMENTS],
        "basin_radii": list(RADII),
        "device": DEVICE,
        "chance": CHANCE,
        "arms": arms,
        "mechanism_audit": {
            "gate_strength_mean": gate["mean"],
            "gate_strength_spread": gate["spread"],
            "per_unit_mask_std": gate["mask_unit_std"],
            "fw_theta_gap_episode5": fw_theta_gap,
            "fw_theta_gap_monotone": fw_gap_grows,
            "null_lr_control_retained": lr_control["retention_mean"],
            "lr_matched_audit": lr_matched,
        },
    }
    record["figure"] = figure_spec(
        "F3 — the P-axis Pareto: retention vs plastic-state capacity",
        scatter_panel(
            {
                arm: {
                    "x": [data["psi_capacity"]],
                    "y": [data["retention_mean"]],
                }
                for arm, data in arms.items()
            },
            xlabel="plastic-state capacity (ψ dims)",
            ylabel="segment-A retention",
            point_labels={
                arm: [f"{arm} (ψ={data['psi_capacity']:.0f})"]
                for arm, data in arms.items()
            },
            chance=CHANCE,
        ),
        lines_panel(
            {arm: data["basin_curve_mean"] for arm, data in arms.items()},
            x=list(RADII),
            xlabel="perturbation radius",
            ylabel="basin agreement",
        ),
        figsize=[12.0, 4.5],
    )
    emit_run_record("F3", "paxis_pareto", record)

    for arm, data in arms.items():
        print(
            f"{arm:>13}: latency {latencies[arm]:.2f} ms/ep "
            f"compute {data['compute']:.3g} psi {data['psi_capacity']:.0f} | "
            f"mastery {np.mean(data['a_mastery']):.3f} "
            f"retained {data['retention_mean']:.3f}±{data['retention_sd']:.3f} | "
            f"basin@r4 {data['basin_curve_mean'][-1]:.2f}"
        )
    print(
        f"mechanism audit: gate strength {gate['mean']:.3f} "
        f"(spread {gate['spread']:.4f}, per-unit mask std "
        f"{gate['mask_unit_std']:.3f}) | fw θ-gap {fw_theta_gap:.2f} | "
        f"null@lr{LR_CONTROL} retained {lr_control['retention_mean']:.3f}"
    )
    _print_lr_matched(lr_matched)

    null, routing = arms["null"], arms["routing"]

    # Ratchets on the REALIZED mechanisms (fire if a primitive regresses
    # to its pre-realization behavior — scalar gain / target bias).
    assert gate["mask_unit_std"] > 0.05, (
        "routing's per-unit masks must differentiate (std > 0.05) — the "
        "realized mechanism is per-unit gating; a near-zero std means the "
        "primitive regressed to constant gain control: re-audit"
    )
    assert fw_theta_gap > 0.02 and fw_gap_grows, (
        "fast-weights' modulation must be gradient-live (θ gap grows "
        "monotonically from identical inits) — if this fires, the "
        "primitive changed behavior: re-audit"
    )
    assert lr_control["retention_mean"] > null["retention_mean"], (
        "the lr brake must move retention (the effective-lr confound is live)"
    )
    _assert_lr_matched(lr_matched)

    # Mastery precondition (read retention only where the arm learned A).
    assert min(null["a_mastery"]) >= 0.44, "null must master A before the switch"

    # The retention ordering stands AT FIXED NOMINAL lr — and with the
    # primitives realized it persists (real per-unit gates, not scalar
    # gain), but the effective-lr confound survives too (null@0.015
    # retains more): recorded, never quoted as a routing-mechanism claim.
    gap = routing["retention_mean"] - null["retention_mean"]
    wins = sum(
        r >= n for r, n in zip(routing["a_retained"], null["a_retained"], strict=True)
    )
    assert gap >= 0.05, f"routing retention advantage must be visible: {gap:.3f}"
    assert wins >= 4, f"routing must retain at least as well per-seed: {wins}/5"

    # The stability ordering likewise: comparative at fixed lr, mechanism OPEN.
    assert routing["basin_curve_mean"][-1] >= null["basin_curve_mean"][-1], (
        "routing's basin agreement at r=4 must dominate null's"
    )

    # The cost axis: the ψ-update cost is visible in latency even though
    # the compute proxy is blind to it (identical MACs — the proxy gap).
    computes = {data["compute"] for data in arms.values()}
    assert len(computes) == 1, "compute proxy must be arm-invariant (documented)"
    assert latencies["fast_weights"] > latencies["null"] * 1.1, (
        "fast-weights episode latency must exceed null's (ψ-update cost)"
    )
