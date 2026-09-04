"""R8.3 stationary pilot: variance estimate for the R8.4 power gate.

Runs the accumulation-capable claim chain at smoke scale: persistent-θ arms
(P-axis contrast null / fast_weights / routing at matched lr, plus the
planted lr=0 ``frozen`` control) walk the stationary-teacher stream — fresh
inputs each episode, one fixed teacher per (campaign, coordinate, seed) —
through the real ``evaluate_episode(stationary_teacher=True)`` path. The
outcome metric is the late-window mean of the target-free claim accuracy
(imp-46); the pilot is labeled ``pilot`` by its own preregistration (never
claim-grade, imp-55) and exists to measure the variance the registered
campaign's power preregistration needs.

Command:
    uv run python -m computronium.experiments.joint.stationary_pilot \
        --episodes 40 --seeds 0,1,2 --output benchmark_results/stationary_pilot.json
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
    n_for_target_power,
    verify_embedded_control,
)
from computronium.validation.statistics import cohens_d

if TYPE_CHECKING:
    from computronium.core.campaign.frontier_record import FrontierRecord

__all__ = [
    "CONTROL_COORDINATE",
    "PILOT_CAMPAIGN_ID",
    "PilotArm",
    "PilotConfig",
    "PilotResult",
    "main",
    "run_pilot",
]

PILOT_CAMPAIGN_ID = "r83_stationary_pilot"
M_ARMS = ("null", "fast_weights", "routing")
CONTROL_COORDINATE = "digital/feedforward/instantaneous/null/gradient/frozen"


@dataclass(frozen=True, slots=True)
class PilotConfig:
    """Pilot design knobs (all defaulted to the registered smoke scale)."""

    episodes: int = 40
    seeds: tuple[int, ...] = (0, 1, 2)
    lr: float = 0.01
    late_window: int = 10
    batch_size: int = 16
    input_dim: int = 8
    num_classes: int = 8
    teacher_noise: float = 0.0
    # Device for the composed systems (GPU-first policy): ``None`` resolves
    # to CUDA when available, else CPU. Teacher streams stay on CPU
    # generators — placement never changes the stream semantics.
    device: str | None = None


def _arm_coordinate(m_arm: str) -> str:
    return f"digital/feedforward/instantaneous/{m_arm}/gradient/euclidean"


@dataclass(frozen=True, slots=True)
class PilotArm:
    """One arm's pilot outcome: per-seed late-window means + curves."""

    label: str
    coordinate: str
    late_means: tuple[float, ...]
    curves: dict[str, tuple[float, ...]]
    mean: float
    sd: float

    def to_dict(self) -> dict[str, object]:
        return {
            "coordinate": self.coordinate,
            "late_means": list(self.late_means),
            "curves": {seed: list(curve) for seed, curve in self.curves.items()},
            "mean": self.mean,
            "sd": self.sd,
        }


def _compose(coordinate: str, config: PilotConfig) -> JointSystem:
    """One persistent system per arm run; θ init seeded per (campaign, coordinate)."""
    torch.manual_seed(episode_seed(0, PILOT_CAMPAIGN_ID, 0, coordinate))
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


def _walk_arm(
    label: str, coordinate: str, config: PilotConfig
) -> tuple[PilotArm, list[FrontierRecord]]:
    """One arm's persistent-θ walk of the stationary stream, per seed."""
    late_means: list[float] = []
    curves: dict[str, tuple[float, ...]] = {}
    control_records: list[FrontierRecord] = []
    for seed in config.seeds:
        joint = _compose(coordinate, config)
        curve: list[float] = []
        for episode in range(config.episodes):
            record, _ = evaluate_episode(
                joint,
                coordinate=coordinate,
                task_name="synthetic",
                campaign_id=PILOT_CAMPAIGN_ID,
                episode=episode,
                batch_size=config.batch_size,
                input_dim=config.input_dim,
                num_classes=config.num_classes,
                guard_threshold=None,
                seed=seed,
                stationary_teacher=True,
                teacher_noise=config.teacher_noise,
            )
            curve.append(record.task_accuracy)
            if label == "control":
                control_records.append(record)
        curves[str(seed)] = tuple(curve)
        late_means.append(float(np.mean(curve[-config.late_window :])))
    mean = float(np.mean(late_means))
    sd = float(np.std(late_means, ddof=1)) if len(late_means) > 1 else 0.0
    arm = PilotArm(
        label=label,
        coordinate=coordinate,
        late_means=tuple(late_means),
        curves=curves,
        mean=mean,
        sd=sd,
    )
    return arm, control_records


@dataclass(frozen=True, slots=True)
class PilotResult:
    """Pilot outcome bundle: arms, P-axis contrasts, control verdict, prereg."""

    config: dict[str, object]
    arms: dict[str, PilotArm]
    contrasts_vs_null: dict[str, dict[str, float | int]]
    pooled_sd: float
    control_verdict: dict[str, str] | None
    quarantined: bool
    preregistration: PowerPreregistration

    def to_dict(self) -> dict[str, object]:
        return {
            "pilot": PILOT_CAMPAIGN_ID,
            "config": self.config,
            "arms": {k: a.to_dict() for k, a in self.arms.items()},
            "contrasts_vs_null": self.contrasts_vs_null,
            "pooled_sd": self.pooled_sd,
            "embedded_control_verdict": self.control_verdict,
            "quarantined": self.quarantined,
            "preregistration": self.preregistration.to_dict(),
        }


def run_pilot(config: PilotConfig) -> PilotResult:
    """Walk every arm through the stationary stream and return the outcome.

    Each (arm, seed) composes ONE persistent system whose θ accumulates
    across episodes — the claim chain the legacy per-episode stream capped
    at chance (imp-54). The planted lr=0 control must stay at chance on the
    same stream (R8.5); a moving control quarantines the pilot.
    """
    coordinates = {m: _arm_coordinate(m) for m in M_ARMS} | {
        "control": CONTROL_COORDINATE
    }
    arms: dict[str, PilotArm] = {}
    control_records: list[FrontierRecord] = []
    for label, coordinate in coordinates.items():
        arm, arm_control_records = _walk_arm(label, coordinate, config)
        arms[label] = arm
        control_records.extend(arm_control_records)

    pooled_sd = (
        float(np.std([m for arm in arms.values() for m in arm.late_means], ddof=1))
        if len(config.seeds) > 1
        else 0.0
    )
    contrasts: dict[str, dict[str, float | int]] = {}
    top_d = 0.0
    if len(config.seeds) > 1:
        for label in ("fast_weights", "routing"):
            d = cohens_d(list(arms["null"].late_means), list(arms[label].late_means))
            contrasts[label] = {
                "cohens_d": round(d, 4),
                "n_for_80pct": n_for_target_power(abs(d)),
            }
            top_d = max(top_d, abs(d))

    control = EmbeddedControl(
        arm="null_frozen_lr0",
        coordinate=CONTROL_COORDINATE,
        chance=1.0 / config.num_classes,
    )
    verdict = (
        verify_embedded_control(control_records, control) if control_records else None
    )
    prereg = PowerPreregistration(
        claim=(
            "Pilot-scale placeholder: the registered claim lands after R8.4 "
            "power preregistration on the pilot variance estimate"
        ),
        metric="task_accuracy",
        claim_scope="accumulated_learning",
        task_stream="stationary",
        expected_effect=round(top_d, 4),
        variance_estimate=round(pooled_sd, 6),
        n_per_group=max((int(c["n_for_80pct"]) for c in contrasts.values()), default=0),
        embedded_control=control,
        declared_rung="pilot",
        created=datetime.now(UTC).date().isoformat(),
    )
    return PilotResult(
        config={
            "episodes": config.episodes,
            "seeds": list(config.seeds),
            "lr": config.lr,
            "late_window": config.late_window,
            "batch_size": config.batch_size,
            "input_dim": config.input_dim,
            "num_classes": config.num_classes,
            "teacher_noise": config.teacher_noise,
            "chance_accuracy": 1.0 / config.num_classes,
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
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--late-window", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument(
        "--teacher-noise",
        type=float,
        default=None,
        help=f"Teacher-logit noise (default: {CALIBRATED_TEACHER_NOISE}); "
        "0.0 = noiseless teacher (ceiling risk at long budgets)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Placement for composed systems (GPU-first; None = auto)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/stationary_pilot.json"),
    )
    args = parser.parse_args()

    noise = (
        CALIBRATED_TEACHER_NOISE if args.teacher_noise is None else args.teacher_noise
    )
    config = PilotConfig(
        episodes=args.episodes,
        seeds=tuple(int(s) for s in args.seeds.split(",") if s.strip()),
        lr=args.lr,
        late_window=args.late_window,
        batch_size=args.batch_size,
        input_dim=args.input_dim,
        num_classes=args.num_classes,
        teacher_noise=noise,
        device=args.device,
    )
    result = run_pilot(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8"
    )

    print(f"stationary pilot -> {args.output}")
    for arm in result.arms.values():
        print(
            f"  {arm.label:<13} mean={arm.mean:.4f} sd={arm.sd:.4f} "
            f"per-seed={[round(m, 3) for m in arm.late_means]}"
        )
    for label, contrast in result.contrasts_vs_null.items():
        print(
            f"  {label} vs null: d={contrast['cohens_d']:+.3f} "
            f"n_for_80%={contrast['n_for_80pct']}"
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
