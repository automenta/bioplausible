"""R9.2 constraint trial: the S-axis physical-constraint stress test.

Hypothesis: under severe substrate constraints, exact-global Backprop
degrades or collapses while local-credit rules (FA, EqProp) degrade
gracefully — the C-Pareto frontier (accuracy per compute/energy) shifts
toward the local arms as severity grows. The same powered design runs
twice: (a) unconstrained Digital, where Backprop is expected to win (the
honest baseline); (b) an analog-noise severity sweep — symmetric-bounds
additive state noise, the constraint family with a usable dynamic range
(the memristive conductance clamp collapses every arm at severity 0,
leaving no curve to compare — probed 2026-09-01).

Each (arm, env, seed) composes ONE persistent system that walks a
stationary-teacher stream (R8.3 accumulation-capable; competence must
exist before degradation is measurable) through the real
``evaluate_episode`` path, then scores a held-out target-free probe. The
planted lr=0 control must sit at chance in every environment (R8.5); a
moving control anywhere quarantines the trial. The first commission is a
pilot by its own preregistration (declared rung caps the label; imp-55):
it fixes the variance and effect size the registered resource-efficiency
claim needs (R8.4). O(1)-activation-memory budgeting (BPTT storage
ceilings) is a registered-design lever, not a pilot instrument — the
memory-profiled arms belong to R9.3.

Command:
    uv run python -m computronium.experiments.joint.constraint_trial \
        --episodes 160 --seeds 0,1,2 \
        --output benchmark_results/constraint_pilot.json
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
)
from computronium.core.joint.transition import PlasticityConfig
from computronium.core.system_trainer import (
    JointSystem,
    compose_joint_system_from_configs,
)
from computronium.ontology import (
    CreditAssignmentConfig,
    GeometryConfig,
    ParameterUpdateConfig,
    StateDynamicsConfig,
    SubstrateConfig,
)
from computronium.validation.power_preregistration import (
    ControlVerdict,
    EmbeddedControl,
    PowerPreregistration,
    at_chance_band,
    verify_embedded_control,
)
from computronium.validation.statistics import cohens_d

if TYPE_CHECKING:
    from computronium.core.campaign.frontier_record import FrontierRecord

__all__ = [
    "CONSTRAINT_CAMPAIGN_ID",
    "CONTROL_CREDIT",
    "PROBE_EPISODE_BASE",
    "TRIAL_ARMS",
    "ConstraintConfig",
    "main",
    "run_trial",
]

CONSTRAINT_CAMPAIGN_ID = "r92_constraint_pilot"
TRIAL_ARMS = ("gradient", "random_projections", "thermodynamic_contrast")
CONTROL_CREDIT = "gradient"
PROBE_EPISODE_BASE = 20_000  # disjoint episode-index space: held-out probe batches
BASELINE_ENV = "digital"
_MARGIN_ABOVE_CHANCE = 0.1  # collapse boundary: probe accuracy above chance + margin
_MIN_CONTRAST_SEEDS = 2  # Cohen's d needs >= 2 observations per group

_DYNAMICS_BY_CREDIT = {
    "gradient": ("instantaneous", None),
    "random_projections": ("instantaneous", None),
    "thermodynamic_contrast": (
        "energy_minimization",
        {"max_steps": 3, "step_size": 0.1},
    ),
}


@dataclass(frozen=True, slots=True)
class ConstraintConfig:
    """Trial design knobs (defaulted to the registered smoke scale).

    ``episodes`` is calibrated for within-arm competence at baseline: FA
    (random_projections) needs ≈160 episodes at lr=0.03 to clear chance
    solidly (probed 2026-09-01) — a walk too short for the slowest arm
    turns its degradation curve into noise (imp-36 class, by design).
    """

    episodes: int = 160
    late_window: int = 20
    probe_episodes: int = 8
    seeds: tuple[int, ...] = (0, 1, 2)
    severities: tuple[float, ...] = (0.0, 0.5, 1.0)
    lr: float = 0.03
    batch_size: int = 16
    input_dim: int = 8
    num_classes: int = 8
    teacher_noise: float = CALIBRATED_TEACHER_NOISE


@dataclass(frozen=True, slots=True)
class ConstraintEnv:
    """One environment of the run-twice design.

    Attributes:
        name: Environment identity (rides ``campaign_id`` so teachers are
            keyed per environment — severities never share a stream).
        substrate_axis: The S-axis value the arm coordinates declare —
            records must name the substrate they actually ran on.
        substrate: Composed substrate config.
        unconstrained: True only for the Digital baseline.
        severity: Noise severity (0.0 for the Digital baseline).
    """

    name: str
    substrate_axis: str
    substrate: SubstrateConfig
    unconstrained: bool
    severity: float


def _environments(config: ConstraintConfig) -> tuple[ConstraintEnv, ...]:
    envs = [
        ConstraintEnv(
            name=BASELINE_ENV,
            substrate_axis="digital",
            substrate=SubstrateConfig.digital(),
            unconstrained=True,
            severity=0.0,
        )
    ]
    envs.extend(
        ConstraintEnv(
            name=f"analog_{severity:g}",
            substrate_axis="analog",
            substrate=SubstrateConfig.analog(noise_level=severity),
            unconstrained=False,
            severity=severity,
        )
        for severity in config.severities
    )
    return tuple(envs)


def _arm_coordinate(credit: str, env: ConstraintEnv, *, frozen: bool = False) -> str:
    dynamics, _ = _DYNAMICS_BY_CREDIT[credit]
    update = "frozen" if frozen else "euclidean"
    return f"{env.substrate_axis}/feedforward/{dynamics}/null/{credit}/{update}"


@dataclass(frozen=True, slots=True)
class ArmOutcome:
    """One arm's outcome across environments: degradation curve + resources."""

    label: str
    coordinate_by_env: dict[str, str]
    probe_by_env: dict[str, tuple[float, ...]]  # env -> per-seed held-out probe acc
    late_by_env: dict[str, tuple[float, ...]]  # env -> per-seed late-window acc
    compute_by_env: dict[str, float]  # mean walk compute (MACs)
    energy_by_env: dict[str, float]  # mean walk consumed-energy estimate (J)
    latency_by_env: dict[str, float]  # mean walk wall-clock (s)
    collapse_severity: float | None  # max severity with probe above chance+margin

    def to_dict(self) -> dict[str, object]:
        return {
            "coordinate_by_env": self.coordinate_by_env,
            "probe_by_env": {k: list(v) for k, v in self.probe_by_env.items()},
            "late_by_env": {k: list(v) for k, v in self.late_by_env.items()},
            "compute_by_env": self.compute_by_env,
            "energy_by_env": self.energy_by_env,
            "latency_by_env": self.latency_by_env,
            "collapse_severity": self.collapse_severity,
        }


def _compose(
    credit: str, env: ConstraintEnv, config: ConstraintConfig, *, frozen: bool = False
) -> JointSystem:
    """One persistent system per (arm, env) run; θ init seeded per (campaign, arm).

    The control composes the declared ``frozen`` update value (step_size=0)
    — the coordinate string and the composed system must agree (imp-48).
    """
    torch.manual_seed(episode_seed(0, CONSTRAINT_CAMPAIGN_ID, 0, credit))
    dynamics_name, dynamics_kwargs = _DYNAMICS_BY_CREDIT[credit]
    dynamics_factory = getattr(StateDynamicsConfig, dynamics_name)
    dynamics = (
        dynamics_factory()
        if dynamics_kwargs is None
        else dynamics_factory(**dynamics_kwargs)
    )
    return compose_joint_system_from_configs(
        env.substrate,
        GeometryConfig.feedforward(
            input_dim=config.input_dim,
            output_dim=config.num_classes,
            hidden_dims=(16,),
        ),
        dynamics,
        PlasticityConfig.null(),
        getattr(CreditAssignmentConfig, credit)(),
        ParameterUpdateConfig.euclidean(step_size=0.0 if frozen else config.lr),
    )


def _collapse_severity(
    probe_by_env: dict[str, tuple[float, ...]], chance: float
) -> float | None:
    """Max swept severity whose held-out probe stays above chance + margin."""
    boundary: float | None = None
    for name, probes in probe_by_env.items():
        if name == BASELINE_ENV:
            continue
        if np.mean(probes) > chance + _MARGIN_ABOVE_CHANCE:
            boundary = float(name.rsplit("_", 1)[1])
    return boundary


@dataclass(frozen=True, slots=True)
class TrialResult:
    """Trial outcome bundle: arms, per-env control verdicts, contrasts, prereg."""

    config: dict[str, object]
    envs: list[dict[str, object]]
    arms: dict[str, ArmOutcome]
    contrasts_vs_gradient: dict[str, dict[str, float]]
    control_verdicts: dict[str, dict[str, str]]
    quarantined: bool
    preregistration: PowerPreregistration

    def to_dict(self) -> dict[str, object]:
        return {
            "trial": CONSTRAINT_CAMPAIGN_ID,
            "config": self.config,
            "envs": self.envs,
            "arms": {k: a.to_dict() for k, a in self.arms.items()},
            "contrasts_vs_gradient": self.contrasts_vs_gradient,
            "embedded_control_verdicts": self.control_verdicts,
            "quarantined": self.quarantined,
            "preregistration": self.preregistration.to_dict(),
        }


def _contrast_d(group_a: list[float], group_b: list[float]) -> float:
    """Cohen's d, or 0.0 when both samples are constant (undefined spread)."""
    try:
        return cohens_d(group_a, group_b)
    except ValueError:
        return 0.0


def _probe(
    joint: JointSystem,
    coordinate: str,
    campaign_id: str,
    seed: int,
    config: ConstraintConfig,
) -> float:
    """Held-out target-free probe of the system's current state (R9.1 readout)."""
    return float(
        np.mean([
            probe_episode(
                joint,
                coordinate=coordinate,
                campaign_id=campaign_id,
                episode=PROBE_EPISODE_BASE + i,
                batch_size=config.batch_size,
                input_dim=config.input_dim,
                num_classes=config.num_classes,
                seed=seed,
                stationary_teacher=True,
                teacher_noise=config.teacher_noise,
            )
            for i in range(config.probe_episodes)
        ])
    )


def _walk_seed(  # ruff: ignore[too-many-arguments] - walk identity tuple travels together
    credit: str,
    frozen: bool,
    env: ConstraintEnv,
    coordinate: str,
    seed: int,
    *,
    config: ConstraintConfig,
) -> tuple[float, float, list[FrontierRecord]]:
    """One (arm, env, seed) walk: (held-out probe, late-window acc, records)."""
    joint = _compose(credit, env, config, frozen=frozen)
    campaign_id = f"{CONSTRAINT_CAMPAIGN_ID}::{env.name}"
    accs: list[float] = []
    records: list[FrontierRecord] = []
    for episode in range(config.episodes):
        record, _ = evaluate_episode(
            joint,
            coordinate=coordinate,
            task_name="synthetic",
            campaign_id=campaign_id,
            episode=episode,
            batch_size=config.batch_size,
            input_dim=config.input_dim,
            num_classes=config.num_classes,
            guard_threshold=None,
            seed=seed,
            stationary_teacher=True,
            teacher_noise=config.teacher_noise,
        )
        accs.append(record.task_accuracy)
        records.append(record)
    probe = _probe(joint, coordinate, campaign_id, seed, config)
    late = float(np.mean(accs[-config.late_window :]))
    return probe, late, records


def _walk_arm(  # ruff: ignore[too-many-arguments] - arm identity tuple travels together
    credit: str,
    frozen: bool,
    envs: tuple[ConstraintEnv, ...],
    *,
    config: ConstraintConfig,
    chance: float,
    control_records_by_env: dict[str, list[FrontierRecord]],
) -> ArmOutcome:
    """One arm's walk across every environment: degradation curve + resources."""
    coordinate_by_env: dict[str, str] = {}
    probe_by_env: dict[str, tuple[float, ...]] = {}
    late_by_env: dict[str, tuple[float, ...]] = {}
    compute_by_env: dict[str, float] = {}
    energy_by_env: dict[str, float] = {}
    latency_by_env: dict[str, float] = {}
    for env in envs:
        coordinate = _arm_coordinate(credit, env, frozen=frozen)
        coordinate_by_env[env.name] = coordinate
        probes: list[float] = []
        lates: list[float] = []
        env_records: list[FrontierRecord] = []
        for seed in config.seeds:
            probe, late, records = _walk_seed(
                credit, frozen, env, coordinate, seed, config=config
            )
            probes.append(probe)
            lates.append(late)
            env_records.extend(records)
        probe_by_env[env.name] = tuple(probes)
        late_by_env[env.name] = tuple(lates)
        compute_by_env[env.name] = float(
            np.mean([r.resources.compute for r in env_records])
        )
        energy_by_env[env.name] = float(
            np.mean([r.resources.energy for r in env_records])
        )
        latency_by_env[env.name] = float(
            np.mean([r.resources.latency for r in env_records])
        )
        if frozen:
            control_records_by_env.setdefault(env.name, []).extend(env_records)
    return ArmOutcome(
        label="control" if frozen else credit,
        coordinate_by_env=coordinate_by_env,
        probe_by_env=probe_by_env,
        late_by_env=late_by_env,
        compute_by_env=compute_by_env,
        energy_by_env=energy_by_env,
        latency_by_env=latency_by_env,
        collapse_severity=_collapse_severity(probe_by_env, chance),
    )


def _verify_controls(
    envs: tuple[ConstraintEnv, ...],
    control_records_by_env: dict[str, list[FrontierRecord]],
    chance: float,
    n_samples: int,
) -> dict[str, ControlVerdict]:
    """Per-environment at-chance verdict for the planted lr=0 arm (R8.5)."""
    verdicts: dict[str, ControlVerdict] = {}
    for env in envs:
        records = control_records_by_env.get(env.name)
        if not records:
            continue
        control = EmbeddedControl(
            arm="frozen_lr0",
            coordinate=_arm_coordinate(CONTROL_CREDIT, env, frozen=True),
            chance=chance,
            tolerance=at_chance_band(chance, n_samples),
        )
        verdicts[env.name] = verify_embedded_control(records, control)
    return verdicts


def _contrasts_vs_gradient(
    arms: dict[str, ArmOutcome],
    envs: tuple[ConstraintEnv, ...],
    config: ConstraintConfig,
) -> tuple[dict[str, dict[str, float]], float]:
    """Local arms vs gradient on held-out probe accuracy, per environment.

    Sign convention: positive d means the gradient arm retains more. The
    registered claim reads the constrained-environment gaps only (the
    Digital baseline is expected to favor gradient).
    """
    contrasts: dict[str, dict[str, float]] = {}
    top_d = 0.0
    if len(config.seeds) < _MIN_CONTRAST_SEEDS:
        return contrasts, top_d
    gradient_probes = arms["gradient"].probe_by_env
    for label in ("random_projections", "thermodynamic_contrast"):
        for env in envs:
            d = _contrast_d(
                list(gradient_probes[env.name]),
                list(arms[label].probe_by_env[env.name]),
            )
            contrasts[f"{label}@{env.name}"] = {"d_probe": round(d, 4)}
            if not env.unconstrained:
                top_d = max(top_d, abs(d))
    return contrasts, top_d


def run_trial(config: ConstraintConfig) -> TrialResult:
    """Walk every arm through every environment and return the outcome.

    The planted lr=0 control walks every environment and must sit at chance
    in each (R8.5); a moving control anywhere quarantines the trial. The
    control's coordinate carries the U-axis ``frozen`` value per env, so the
    post-run verdict can never match a learning arm's records.
    """
    chance = 1.0 / config.num_classes
    envs = _environments(config)
    # (credit, frozen) pairs: the control shares gradient's credit but not
    # its update — the frozen flag, not the credit name, is what makes it the arm.
    arm_specs = (*[(c, False) for c in TRIAL_ARMS], (CONTROL_CREDIT, True))
    control_records_by_env: dict[str, list[FrontierRecord]] = {}
    arms = {
        "control" if frozen else credit: _walk_arm(
            credit,
            frozen,
            envs,
            config=config,
            chance=chance,
            control_records_by_env=control_records_by_env,
        )
        for credit, frozen in arm_specs
    }
    env_reports = [
        {
            "name": env.name,
            "severity": env.severity,
            "unconstrained": env.unconstrained,
            "substrate_axis": env.substrate_axis,
        }
        for env in envs
    ]
    n_control_samples = config.episodes * config.batch_size * len(config.seeds)
    control_verdicts = _verify_controls(
        envs, control_records_by_env, chance, n_control_samples
    )
    contrasts, top_d = _contrasts_vs_gradient(arms, envs, config)
    pilot_control = EmbeddedControl(
        arm="frozen_lr0",
        coordinate=_arm_coordinate(CONTROL_CREDIT, envs[0], frozen=True),
        chance=chance,
        tolerance=at_chance_band(chance, n_control_samples),
    )
    pooled_sd = float(
        np.std(
            [
                p
                for arm in arms.values()
                for probes in arm.probe_by_env.values()
                for p in probes
            ],
            ddof=1,
        )
    )
    prereg = PowerPreregistration(
        claim=(
            "Pilot: under analog-noise substrate constraints the local-credit arms "
            "(FA, EqProp) degrade gracefully and take over the accuracy-per-resource "
            "frontier where exact-global Backprop collapses; the Digital baseline "
            "reproduces Backprop's advantage. Registered resource-efficiency claim "
            "follows once this pilot fixes variance and effect size"
        ),
        metric="task_accuracy",
        claim_scope="resource_efficiency",
        task_stream="stationary",
        expected_effect=round(top_d, 4),
        variance_estimate=round(pooled_sd, 6),
        n_per_group=len(config.seeds),
        embedded_control=pilot_control,
        declared_rung="pilot",
        created=datetime.now(UTC).date().isoformat(),
    )
    return TrialResult(
        config={
            "episodes": config.episodes,
            "late_window": config.late_window,
            "probe_episodes": config.probe_episodes,
            "seeds": list(config.seeds),
            "severities": list(config.severities),
            "lr": config.lr,
            "batch_size": config.batch_size,
            "input_dim": config.input_dim,
            "num_classes": config.num_classes,
            "teacher_noise": config.teacher_noise,
            "probe_episode_base": PROBE_EPISODE_BASE,
            "chance_accuracy": chance,
            "margin_above_chance": _MARGIN_ABOVE_CHANCE,
        },
        envs=env_reports,
        arms=arms,
        contrasts_vs_gradient=contrasts,
        control_verdicts={
            name: {"verdict": v.verdict, "detail": v.detail}
            for name, v in control_verdicts.items()
        },
        quarantined=bool(control_verdicts)
        and any(v.quarantines for v in control_verdicts.values()),
        preregistration=prereg,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--episodes", type=int, default=160)
    parser.add_argument("--late-window", type=int, default=20)
    parser.add_argument("--probe-episodes", type=int, default=8)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--severities", default="0.0,0.5,1.0")
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/constraint_pilot.json"),
    )
    args = parser.parse_args()

    config = ConstraintConfig(
        episodes=args.episodes,
        late_window=args.late_window,
        probe_episodes=args.probe_episodes,
        seeds=tuple(int(s) for s in args.seeds.split(",") if s.strip()),
        severities=tuple(float(s) for s in args.severities.split(",") if s.strip()),
        lr=args.lr,
        batch_size=args.batch_size,
    )
    result = run_trial(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8"
    )

    print(f"constraint trial -> {args.output}")
    for arm in result.arms.values():
        probes = " ".join(
            f"{env}={np.mean(ps):.3f}" for env, ps in arm.probe_by_env.items()
        )
        print(f"  {arm.label:<22} probe {probes} collapse@{arm.collapse_severity}")
    for name, contrast in result.contrasts_vs_gradient.items():
        print(f"  {name}: d={contrast['d_probe']:+.3f}")
    for env, verdict in result.control_verdicts.items():
        print(f"  control[{env}]: {verdict['verdict']} — {verdict['detail']}")
    print(
        f"  prereg label: {result.preregistration.to_dict()['label']} "
        f"(quarantined={result.quarantined})"
    )


if __name__ == "__main__":
    main()
