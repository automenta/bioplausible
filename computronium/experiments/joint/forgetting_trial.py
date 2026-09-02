"""R9.1 forgetting trial: the M-axis catastrophic-forgetting stress test.

Hypothesis: ψ-mediated plasticity (routing / fast_weights) retains segment A
while learning segment B; Null collapses toward chance on A. The walk is the
R8-compliant persistent-θ chain (R8.3 semantics note): one composed system
per (arm, seed) walks the structured task-sequence stream A→B — segment-keyed
stationary teachers (R8.3 machinery per segment) with fresh inputs every
episode — through the real ``evaluate_episode(segment=...)`` path. Retention
is measured by ``probe_episode``: target-free, no-train accuracy on held-out
segment-A batches taken at every segment boundary, so the probe curve is the
retention trajectory and A-mastery vs final A-probe gives backward transfer.

The first commission is a pilot by its own preregistration (declared rung
caps the label; imp-55): it exists to measure the effect size and variance
the registered retention claim's power preregistration needs (R8.4), with the
planted lr=0 control embedded (R8.5) and the control band sized to the
registered record count (imp-59). ψ engagement rides the pipeline path locked
by ``test_psi_engagement.py`` (imp-43/imp-22); cross-episode ψ persistence is
NOT part of the ``train_step`` contract (ψ re-initializes per episode) — the
ψ-carried-retention mechanism is the Z3 retention arm, where ψ state persists
across adaptation steps and can be snapshot/restored.

Command:
    uv run python -m computronium.experiments.joint.forgetting_trial \
        --segments A=40,B=40 --seeds 0,1,2 \
        --output benchmark_results/forgetting_pilot.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from computronium.core.campaign.evaluation import (
    CALIBRATED_TEACHER_NOISE,
    episode_seed,
    evaluate_episode,
    probe_episode,
    resolve_device,
)
from computronium.core.system_trainer import (
    JointSystem,
    compose_joint_system_from_configs,
)
from computronium.ontology import (
    CreditAssignmentConfig,
    GeometryConfig,
    ParameterUpdateConfig,
    PlasticityConfig,
    StateDynamicsConfig,
    SubstrateConfig,
)
from computronium.validation.power_preregistration import (
    EmbeddedControl,
    PowerPreregistration,
    at_chance_band,
    n_for_target_power,
    verify_embedded_control,
)
from computronium.validation.statistics import cohens_d

if TYPE_CHECKING:
    from computronium.core.campaign.frontier_record import FrontierRecord

__all__ = [
    "CONTROL_COORDINATE",
    "PROBE_EPISODE_BASE",
    "TRIAL_CAMPAIGN_ID",
    "TrialArm",
    "TrialConfig",
    "TrialResult",
    "main",
    "run_trial",
]

TRIAL_CAMPAIGN_ID = "r91_forgetting_pilot"
M_ARMS = ("null", "fast_weights", "routing")
CONTROL_COORDINATE = "digital/feedforward/instantaneous/null/gradient/frozen"
PROBE_EPISODE_BASE = 10_000  # disjoint episode-index space: held-out probe batches
PROBED_SEGMENT = "A"  # the retention target every boundary probe measures


@dataclass(frozen=True, slots=True)
class TrialConfig:
    """Trial design knobs (defaulted to the registered smoke scale).

    ``lr`` is calibrated for within-segment competence (mastery ≈ 0.69 for
    the null arm at the 40-episode segment budget under the calibrated
    teacher noise): retention is unreadable below competence, and the
    control band assumes an at-chance frozen arm on a learnable stream.
    """

    segments: tuple[tuple[str, int], ...] = (("A", 40), ("B", 40))
    probe_episodes: int = 8
    seeds: tuple[int, ...] = (0, 1, 2)
    lr: float = 0.03
    batch_size: int = 16
    input_dim: int = 8
    num_classes: int = 8
    teacher_noise: float = CALIBRATED_TEACHER_NOISE
    # Device for the composed systems (GPU-first policy): ``None`` resolves
    # to CUDA when available, else CPU. Teacher streams stay on CPU
    # generators — placement never changes the stream semantics.
    device: str | None = None

    def __post_init__(self) -> None:
        if not self.segments or self.segments[0][0] != PROBED_SEGMENT:
            msg = f"schedule must start with segment {PROBED_SEGMENT!r}"
            raise ValueError(msg)


def _arm_coordinate(m_arm: str) -> str:
    return f"digital/feedforward/instantaneous/{m_arm}/gradient/euclidean"


@dataclass(frozen=True, slots=True)
class TrialArm:
    """One arm's trial outcome: per-seed retention metrics + raw curves."""

    label: str
    coordinate: str
    a_mastery: tuple[float, ...]  # A-probe right after the last A episode
    a_retained: tuple[float, ...]  # A-probe after the final segment boundary
    retention_delta: tuple[float, ...]  # a_retained - a_mastery (backward transfer)
    curves: dict[str, tuple[float, ...]]  # seed -> per-episode training accuracy
    probes: dict[str, tuple[float, ...]]  # seed -> boundary-probe trajectory
    mean: float  # mean a_retained across seeds
    sd: float

    def to_dict(self) -> dict[str, object]:
        return {
            "coordinate": self.coordinate,
            "a_mastery": list(self.a_mastery),
            "a_retained": list(self.a_retained),
            "retention_delta": list(self.retention_delta),
            "curves": {seed: list(c) for seed, c in self.curves.items()},
            "probes": {seed: list(p) for seed, p in self.probes.items()},
            "mean_retained": self.mean,
            "sd_retained": self.sd,
        }


def _compose(coordinate: str, config: TrialConfig) -> JointSystem:
    """One persistent system per arm run; θ init seeded per (campaign, coordinate)."""
    torch.manual_seed(episode_seed(0, TRIAL_CAMPAIGN_ID, 0, coordinate))
    m_arm, update = coordinate.split("/")[3], coordinate.split("/")[5]
    return compose_joint_system_from_configs(
        SubstrateConfig.digital(),
        GeometryConfig.feedforward(
            input_dim=config.input_dim,
            output_dim=config.num_classes,
            hidden_dims=(16,),
        ),
        StateDynamicsConfig.instantaneous(),
        getattr(PlasticityConfig, m_arm)(),
        CreditAssignmentConfig.gradient(),
        ParameterUpdateConfig.euclidean(
            step_size=0.0 if update == "frozen" else config.lr
        ),
        device=resolve_device(config.device),
    )


def _probe(
    joint: JointSystem, coordinate: str, config: TrialConfig, seed: int
) -> float:
    """Held-out segment-A probe of the system's current state.

    The seed keys the probe's teacher to the same per-(arm, seed) task the
    walk trains on — a probe scored against another seed's teacher measures
    chance by construction, not retention.
    """
    accs = [
        probe_episode(
            joint,
            coordinate=coordinate,
            task_name="synthetic",
            campaign_id=TRIAL_CAMPAIGN_ID,
            episode=PROBE_EPISODE_BASE + i,
            batch_size=config.batch_size,
            input_dim=config.input_dim,
            num_classes=config.num_classes,
            seed=seed,
            stationary_teacher=True,
            teacher_noise=config.teacher_noise,
            segment=PROBED_SEGMENT,
        )
        for i in range(config.probe_episodes)
    ]
    return float(np.mean(accs))


def _walk_arm(
    label: str, coordinate: str, config: TrialConfig
) -> tuple[TrialArm, list[FrontierRecord]]:
    """One arm's persistent-θ walk of the segmented stream, per seed."""
    a_mastery: list[float] = []
    a_retained: list[float] = []
    curves: dict[str, tuple[float, ...]] = {}
    probes: dict[str, tuple[float, ...]] = {}
    control_records: list[FrontierRecord] = []
    for seed in config.seeds:
        joint = _compose(coordinate, config)
        curve: list[float] = []
        boundary_probes: list[float] = []
        episode_index = 0
        for seg_index, (segment, n_episodes) in enumerate(config.segments):
            for _ in range(n_episodes):
                record, _ = evaluate_episode(
                    joint,
                    coordinate=coordinate,
                    task_name="synthetic",
                    campaign_id=TRIAL_CAMPAIGN_ID,
                    episode=episode_index,
                    batch_size=config.batch_size,
                    input_dim=config.input_dim,
                    num_classes=config.num_classes,
                    guard_threshold=None,
                    seed=seed,
                    stationary_teacher=True,
                    teacher_noise=config.teacher_noise,
                    segment=segment,
                )
                curve.append(record.task_accuracy)
                if label == "control":
                    control_records.append(record)
                episode_index += 1
            boundary_probes.append(_probe(joint, coordinate, config, seed))
            if seg_index == 0:
                a_mastery.append(boundary_probes[-1])
        a_retained.append(boundary_probes[-1])
        curves[str(seed)] = tuple(curve)
        probes[str(seed)] = tuple(boundary_probes)
    retained = a_retained
    deltas = [r - m for r, m in zip(a_retained, a_mastery, strict=True)]
    arm = TrialArm(
        label=label,
        coordinate=coordinate,
        a_mastery=tuple(a_mastery),
        a_retained=tuple(a_retained),
        retention_delta=tuple(deltas),
        curves=curves,
        probes=probes,
        mean=float(np.mean(retained)),
        sd=float(np.std(retained, ddof=1)) if len(retained) > 1 else 0.0,
    )
    return arm, control_records


@dataclass(frozen=True, slots=True)
class TrialResult:
    """Trial outcome bundle: arms, contrasts, control verdict, prereg."""

    config: dict[str, object]
    arms: dict[str, TrialArm]
    contrasts_vs_null: dict[str, dict[str, float | int]]
    pooled_sd: float
    control_verdict: dict[str, str] | None
    quarantined: bool
    preregistration: PowerPreregistration

    def to_dict(self) -> dict[str, object]:
        return {
            "trial": TRIAL_CAMPAIGN_ID,
            "config": self.config,
            "arms": {k: a.to_dict() for k, a in self.arms.items()},
            "contrasts_vs_null": self.contrasts_vs_null,
            "pooled_sd": self.pooled_sd,
            "embedded_control_verdict": self.control_verdict,
            "quarantined": self.quarantined,
            "preregistration": self.preregistration.to_dict(),
        }


def _contrast_d(group_a: list[float], group_b: list[float]) -> float:
    """Cohen's d, or 0.0 when both samples are constant.

    A degenerate walk (identical retained accuracy across every seed) has
    undefined spread — reported as no measurable effect (n_for_target_power
    → unreachable) rather than crashing the outcome assembly.
    """
    try:
        return cohens_d(group_a, group_b)
    except ValueError:
        return 0.0


def _top_level_contrasts(
    arms: dict[str, TrialArm], config: TrialConfig
) -> tuple[dict[str, dict[str, float | int]], float]:
    """Cohen's d vs null on retained accuracy and retention delta."""
    contrasts: dict[str, dict[str, float | int]] = {}
    top_d = 0.0
    if len(config.seeds) > 1:
        for label in M_ARMS[1:]:
            d_retained = _contrast_d(
                list(arms["null"].a_retained), list(arms[label].a_retained)
            )
            d_delta = _contrast_d(
                list(arms["null"].retention_delta), list(arms[label].retention_delta)
            )
            contrasts[label] = {
                "d_retained": round(d_retained, 4),
                "d_retention_delta": round(d_delta, 4),
                "n_for_80pct_retained": n_for_target_power(abs(d_retained)),
            }
            top_d = max(top_d, abs(d_retained), abs(d_delta))
    return contrasts, top_d


def run_trial(
    config: TrialConfig,
    preregistration: PowerPreregistration | None = None,
) -> TrialResult:
    """Walk every arm through the segmented stream and return the outcome.

    Each (arm, seed) composes ONE persistent system whose θ accumulates
    across the whole walk. The planted lr=0 control must sit at chance on
    every segment (R8.5); a moving control quarantines the trial.

    ``preregistration`` (R8.4) commissions the run at a registered design:
    it must pass every claim-grade gate *before* the walk (fail loudly by
    name), must be resourced by the config's seed count, and its embedded
    control + tolerance become the authoritative post-run verdict — the
    self-labeled pilot preregistration is built only when commissioning
    without one.
    """
    coordinates = {m: _arm_coordinate(m) for m in M_ARMS} | {
        "control": CONTROL_COORDINATE
    }
    if preregistration is not None:
        preregistration.require_claim_grade()
        if preregistration.label() != "claim_grade":
            msg = (
                f"commission declared rung {preregistration.declared_rung!r} "
                "caps the label below claim-grade"
            )
            raise ValueError(msg)
        if len(config.seeds) < preregistration.n_per_group:
            msg = (
                f"commission delivers {len(config.seeds)} obs/group but the "
                f"registered design requires {preregistration.n_per_group}"
            )
            raise ValueError(msg)
    arms: dict[str, TrialArm] = {}
    control_records: list[FrontierRecord] = []
    for label, coordinate in coordinates.items():
        arm, arm_control_records = _walk_arm(label, coordinate, config)
        arms[label] = arm
        control_records.extend(arm_control_records)

    pooled_sd = (
        float(np.std([m for arm in arms.values() for m in arm.a_retained], ddof=1))
        if len(config.seeds) > 1
        else 0.0
    )
    contrasts, top_d = _top_level_contrasts(arms, config)

    chance = 1.0 / config.num_classes
    if preregistration is not None:
        control = preregistration.embedded_control
        if control is None:  # unreachable: the claim-grade gate requires the arm
            raise ValueError(  # ruff: ignore[raise-vanilla-args] - unreachable guard
                "registered preregistration carries no embedded control"
            )
        verdict = verify_embedded_control(control_records, control)
        prereg = preregistration
    else:
        control = EmbeddedControl(
            arm="null_frozen_lr0",
            coordinate=CONTROL_COORDINATE,
            chance=chance,
            tolerance=at_chance_band(chance, len(control_records) * config.batch_size),
        )
        verdict = (
            verify_embedded_control(control_records, control)
            if control_records
            else None
        )
        prereg = PowerPreregistration(
            claim=(
                "Pilot: ψ-mediated plasticity (fast_weights/routing) retains segment-A "
                "accuracy through the segment-B shift where Null collapses; registered "
                "retention claim follows once this pilot fixes variance and effect size"
            ),
            metric="task_accuracy",
            claim_scope="retention",
            task_stream="segmented",
            expected_effect=round(top_d, 4),
            variance_estimate=round(pooled_sd, 6),
            n_per_group=max(
                (int(c["n_for_80pct_retained"]) for c in contrasts.values()), default=0
            ),
            embedded_control=control,
            declared_rung="pilot",
            created=datetime.now(UTC).date().isoformat(),
        )
    return TrialResult(
        config={
            "segments": [[s, n] for s, n in config.segments],
            "probe_episodes": config.probe_episodes,
            "seeds": list(config.seeds),
            "lr": config.lr,
            "batch_size": config.batch_size,
            "input_dim": config.input_dim,
            "num_classes": config.num_classes,
            "teacher_noise": config.teacher_noise,
            "probe_episode_base": PROBE_EPISODE_BASE,
            "chance_accuracy": chance,
            "device": resolve_device(config.device),
        },
        arms=arms,
        contrasts_vs_null=contrasts,
        pooled_sd=pooled_sd,
        control_verdict=(
            {"verdict": verdict.verdict, "detail": verdict.detail} if verdict else None
        ),
        quarantined=bool(verdict and verdict.quarantines),
        preregistration=prereg,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--segments",
        default="A=40,B=40",
        help="Task-sequence schedule as segment=count pairs (e.g. A=40,B=40)",
    )
    parser.add_argument("--probe-episodes", type=int, default=8)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument(
        "--teacher-noise",
        type=float,
        default=None,
        help=f"Teacher-logit noise (default: {CALIBRATED_TEACHER_NOISE})",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Placement for composed systems (GPU-first; None = auto)",
    )
    parser.add_argument(
        "--prereg",
        type=Path,
        default=None,
        help=(
            "Registered preregistration JSON (R8.4): claim-grade gate before the "
            "walk, registered n, and the authoritative embedded control (e.g. "
            "configs/preregistrations/r91_retention_registered.json)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/forgetting_pilot.json"),
    )
    args = parser.parse_args()

    segments = tuple(
        (part.split("=")[0], int(part.split("=")[1]))
        for part in args.segments.split(",")
        if part.strip()
    )
    noise = (
        CALIBRATED_TEACHER_NOISE if args.teacher_noise is None else args.teacher_noise
    )
    config = TrialConfig(
        segments=segments,
        probe_episodes=args.probe_episodes,
        seeds=tuple(int(s) for s in args.seeds.split(",") if s.strip()),
        lr=args.lr,
        batch_size=args.batch_size,
        input_dim=args.input_dim,
        num_classes=args.num_classes,
        teacher_noise=noise,
        device=args.device,
    )
    prereg = PowerPreregistration.load(args.prereg) if args.prereg else None
    result = run_trial(config, preregistration=prereg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8"
    )

    print(f"forgetting trial -> {args.output}")
    for arm in result.arms.values():
        print(
            f"  {arm.label:<13} mastery={np.mean(arm.a_mastery):.4f} "
            f"retained={arm.mean:.4f} (sd {arm.sd:.4f}) "
            f"delta={np.mean(arm.retention_delta):+.4f}"
        )
    for label, contrast in result.contrasts_vs_null.items():
        print(
            f"  {label} vs null: d_retained={contrast['d_retained']:+.3f} "
            f"d_delta={contrast['d_retention_delta']:+.3f} "
            f"n_for_80%={contrast['n_for_80pct_retained']}"
        )
    verdict = result.control_verdict
    print(
        f"  control: {verdict['verdict'] if verdict else 'n/a'}"
        f"{' — ' + verdict['detail'] if verdict else ''}"
    )
    print(
        f"  prereg label: {result.preregistration.to_dict()['label']} "
        f"(quarantined={result.quarantined})"
    )


if __name__ == "__main__":
    main()
