"""R9.3 deep credit trial: the C-axis temporal-dependency stress test.

Hypothesis: ThermodynamicContrast (EqProp) / RandomProjections (FA) learn a
50+ step temporal dependency with O(1) activation memory where exact-global
Backprop (gradient) memory grows O(depth) and its gradients vanish. The same
powered design sweeps depth as the severity lever: shallow tiers verify
competence (all arms learn); the deep tier (≥50 credit steps) tests the
memory/vanishing boundary.

Each (arm, depth, seed) composes ONE persistent system that walks a
stationary synthetic-teacher stream (R8.3: teacher keyed per
campaign/coordinate/seed/depth — fixed across episodes so θ accumulates)
through the real ``evaluate_episode`` path, then scores a held-out
target-free probe via ``probe_episode``. The planted lr=0 control must sit
at chance in every depth environment (R8.5); a moving control anywhere
quarantines the trial. The first commission is a pilot by its own
preregistration (declared rung caps the label; imp-55): it fixes the
variance and effect size the registered credit-at-depth claim needs (R8.4).
Memory-budgeted arms (gradient arm disqualified at per-step saved-activation
budget) is the R9.2 registered design lever — the memory profile is the
primary instrument.

Command:
    uv run python -m computronium.experiments.joint.deep_credit_trial \
        --episodes 160 --seeds 0,1,2 --width 16 \
        --output benchmark_results/deep_credit_pilot.json
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
from computronium.core.profiling import (
    measure_saved_activation_bytes,
)
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
    "CONTROL_CREDIT",
    "DEEP_CREDIT_CAMPAIGN_ID",
    "PROBE_EPISODE_BASE",
    "TRIAL_ARMS",
    "DeepCreditConfig",
    "main",
    "run_trial",
]

DEEP_CREDIT_CAMPAIGN_ID = "r93_deep_credit_pilot"
TRIAL_ARMS = ("gradient", "random_projections", "thermodynamic_contrast")
CONTROL_CREDIT = "gradient"
PROBE_EPISODE_BASE = 30_000  # disjoint episode-index space: held-out probe batches
_MARGIN_ABOVE_CHANCE = 0.1  # collapse boundary: probe accuracy above chance + margin
_MIN_CONTRAST_SEEDS = 2  # Cohen's d needs >= 2 observations per group


# Dynamics per credit (respects D x C fence: thermodynamic_contrast requires settling)
_DYNAMICS_BY_CREDIT = {
    "gradient": ("instantaneous", None),
    "random_projections": ("instantaneous", None),
    "thermodynamic_contrast": (
        "energy_minimization",
        None,  # max_steps filled per depth at compose time
    ),
}


@dataclass(frozen=True, slots=True)
class DeepCreditConfig:
    """Trial design knobs (defaulted to the registered smoke scale).

    ``depths``: credit-depth tiers (number of hidden layers + 1). The registered
    deep tier must be >= 50. Shallow tiers are the competence floor.
    ``episodes`` is calibrated for within-arm competence at the shallow tier:
    FA (random_projections) needs sufficient episodes to clear chance at
    shallow depth — a walk too short for the slowest arm turns its degradation
    curve into noise (imp-36 class, by design).
    """

    episodes: int = 160
    late_window: int = 20
    probe_episodes: int = 8
    seeds: tuple[int, ...] = (0, 1, 2)
    depths: tuple[int, ...] = (4, 16, 50)  # credit steps = hidden layers + 1
    width: int = 16
    lr: float = 0.03
    batch_size: int = 16
    input_dim: int = 8
    num_classes: int = (
        8  # counting mod 8 (state-space realization); synthetic task uses this
    )
    teacher_noise: float = CALIBRATED_TEACHER_NOISE
    # Optional: per-step saved-activation-bytes budget (MiB). Arms exceeding
    # this budget at a given depth are disqualified (BPTT memory wall).
    memory_budget_mib: float | None = None
    # Control band floor for init-to-init variation (imp-59). With few seeds,
    # the frozen arm's mean accuracy has seed-level variance; widen the floor.
    control_band_floor: float = 0.15


@dataclass(frozen=True, slots=True)
class DepthEnv:
    """One depth environment of the depth-sweep design.

    Attributes:
        name: Environment identity (rides campaign_id so teachers are keyed
            per depth — depths never share a stream).
        depth: Credit depth (number of hidden layers + 1).
        hidden_dims: Tuple of hidden layer dimensions (all equal to width).
        unconstrained: True only for the shallowest depth (competence tier).
    """

    name: str
    depth: int
    hidden_dims: tuple[int, ...]
    unconstrained: bool


def _environments(config: DeepCreditConfig) -> tuple[DepthEnv, ...]:
    """Depth sweep: each depth is an independent environment."""
    envs = []
    for i, depth in enumerate(config.depths):
        hidden_dims = tuple([config.width] * (depth - 1))  # credit steps = hidden + 1
        envs.append(
            DepthEnv(
                name=f"depth_{depth}",
                depth=depth,
                hidden_dims=hidden_dims,
                unconstrained=(i == 0),  # shallowest is the competence baseline
            )
        )
    return tuple(envs)


def _arm_coordinate(credit: str, _env: DepthEnv, *, frozen: bool = False) -> str:
    dynamics_name, _ = _DYNAMICS_BY_CREDIT[credit]
    update = "frozen" if frozen else "euclidean"
    return f"digital/feedforward/{dynamics_name}/null/{credit}/{update}"


@dataclass(frozen=True, slots=True)
class ArmOutcome:
    """One arm's outcome across depths: degradation curve + memory profile."""

    label: str
    coordinate_by_env: dict[str, str]
    probe_by_env: dict[str, tuple[float, ...]]  # env -> per-seed held-out probe acc
    late_by_env: dict[str, tuple[float, ...]]  # env -> per-seed late-window acc
    saved_bytes_by_env: dict[str, float]  # mean saved-for-backward bytes per train step
    wall_time_by_env: dict[str, float]  # mean train-step wall-clock (s)
    disqualified_by_env: dict[str, bool]  # env -> disqualified (memory budget)

    def to_dict(self) -> dict[str, object]:
        return {
            "coordinate_by_env": self.coordinate_by_env,
            "probe_by_env": {k: list(v) for k, v in self.probe_by_env.items()},
            "late_by_env": {k: list(v) for k, v in self.late_by_env.items()},
            "saved_bytes_by_env": self.saved_bytes_by_env,
            "wall_time_by_env": self.wall_time_by_env,
            "disqualified_by_env": self.disqualified_by_env,
        }


def _compose(
    credit: str, env: DepthEnv, config: DeepCreditConfig, *, frozen: bool = False
) -> JointSystem:
    """One persistent system per (arm, env) run; θ init seeded per (campaign, arm)."""
    torch.manual_seed(episode_seed(0, DEEP_CREDIT_CAMPAIGN_ID, 0, credit))

    dynamics_name, _ = _DYNAMICS_BY_CREDIT[credit]
    # For thermodynamic_contrast, set max_steps >= depth so nudge propagates
    if dynamics_name == "energy_minimization":
        dynamics = StateDynamicsConfig.energy_minimization(
            max_steps=max(3, env.depth),  # nudge must reach input layer
            step_size=0.1,
            beta=0.5,
        )
    else:
        dynamics_factory = getattr(StateDynamicsConfig, dynamics_name)
        dynamics = dynamics_factory()

    return compose_joint_system_from_configs(
        SubstrateConfig.digital(),
        GeometryConfig.feedforward(
            input_dim=config.input_dim,
            output_dim=config.num_classes,
            hidden_dims=env.hidden_dims,
        ),
        dynamics,
        PlasticityConfig.null(),
        getattr(CreditAssignmentConfig, credit)(),
        ParameterUpdateConfig.euclidean(step_size=0.0 if frozen else config.lr),
    )


def _probe(
    joint: JointSystem,
    coordinate: str,
    campaign_id: str,
    seed: int,
    config: DeepCreditConfig,
) -> float:
    """Held-out target-free probe of the system's current state (R9.1 readout)."""
    return float(
        np.mean([
            probe_episode(
                joint,
                coordinate=coordinate,
                task_name="synthetic",
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
    env: DepthEnv,
    coordinate: str,
    seed: int,
    *,
    config: DeepCreditConfig,
) -> tuple[float, float, list[FrontierRecord], float]:
    """One (arm, env, seed) walk: held-out probe, late-window acc, records,
    mean saved bytes.

    Saved-activation-bytes is an architectural property (constant per step for
    a given credit/depth). Measure once on a representative batch at the start
    to avoid double-running train_step (which would alter the model state).
    """
    joint = _compose(credit, env, config, frozen=frozen)
    campaign_id = f"{DEEP_CREDIT_CAMPAIGN_ID}::{env.name}"

    # Measure saved-activation-bytes once on a representative batch (episode 0)
    from computronium.core.campaign.evaluation import episode_batch

    x_batch, y_batch = episode_batch(
        0,
        task_name="synthetic",
        batch_size=config.batch_size,
        input_dim=config.input_dim,
        num_classes=config.num_classes,
        teacher_key=(DEEP_CREDIT_CAMPAIGN_ID, env.name, credit, 0),
        teacher_noise=config.teacher_noise,
    )
    x_batch, y_batch = x_batch.to(joint.device), y_batch.to(joint.device)

    def _train_step(x, y):
        return joint.train_step(x, y)

    _, saved = measure_saved_activation_bytes(_train_step, x_batch, y_batch)
    mean_saved = float(saved.total_bytes)

    # Actual training walk via evaluate_episode
    accs: list[float] = []
    records: list[FrontierRecord] = []
    for episode in range(config.episodes):
        record, _metrics = evaluate_episode(
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
    return probe, late, records, mean_saved


def _walk_arm(  # ruff: ignore[too-many-locals] - arm identity tuple travels together
    credit: str,
    frozen: bool,
    envs: tuple[DepthEnv, ...],
    *,
    config: DeepCreditConfig,
    control_records_by_env: dict[str, list[FrontierRecord]],
) -> ArmOutcome:
    """One arm's walk across every depth environment."""
    coordinate_by_env: dict[str, str] = {}
    probe_by_env: dict[str, tuple[float, ...]] = {}
    late_by_env: dict[str, tuple[float, ...]] = {}
    saved_bytes_by_env: dict[str, float] = {}
    wall_time_by_env: dict[str, float] = {}
    disqualified_by_env: dict[str, bool] = {}
    for env in envs:
        coordinate = _arm_coordinate(credit, env, frozen=frozen)
        coordinate_by_env[env.name] = coordinate
        probes: list[float] = []
        lates: list[float] = []
        env_records: list[FrontierRecord] = []
        env_saved_bytes: list[float] = []
        for seed in config.seeds:
            probe, late, records, mean_saved = _walk_seed(
                credit, frozen, env, coordinate, seed, config=config
            )
            probes.append(probe)
            lates.append(late)
            env_records.extend(records)
            env_saved_bytes.append(mean_saved)
        probe_by_env[env.name] = tuple(probes)
        late_by_env[env.name] = tuple(lates)
        saved_bytes_by_env[env.name] = float(np.mean(env_saved_bytes))
        # Disqualification: memory budget exceeded
        if config.memory_budget_mib is not None:
            budget_bytes = config.memory_budget_mib * 1024 * 1024
            disqualified_by_env[env.name] = saved_bytes_by_env[env.name] > budget_bytes
        else:
            disqualified_by_env[env.name] = False
        if frozen:
            control_records_by_env.setdefault(env.name, []).extend(env_records)
    return ArmOutcome(
        label="control" if frozen else credit,
        coordinate_by_env=coordinate_by_env,
        probe_by_env=probe_by_env,
        late_by_env=late_by_env,
        saved_bytes_by_env=saved_bytes_by_env,
        wall_time_by_env=wall_time_by_env,
        disqualified_by_env=disqualified_by_env,
    )


def _verify_controls(  # ruff: ignore[too-many-arguments] - verdict identity travels together
    envs: tuple[DepthEnv, ...],
    control_records_by_env: dict[str, list[FrontierRecord]],
    chance: float,
    n_samples: int,
    control_band_floor: float = 0.05,
    *,
    registered_control: EmbeddedControl | None = None,
) -> dict[str, ControlVerdict]:
    """Per-environment at-chance verdict for the planted lr=0 arm (R8.5).

    ``registered_control`` (R8.4): when a preregistration commissions the
    run, its embedded control + tolerance are the authoritative instrument —
    the self-built per-env control is only the pilot fallback.
    """
    verdicts: dict[str, ControlVerdict] = {}
    for env in envs:
        records = control_records_by_env.get(env.name)
        if not records:
            continue
        control = registered_control
        if control is None:
            # Use max of statistical band and config floor (imp-59: seed-level variance)
            tolerance = max(at_chance_band(chance, n_samples), control_band_floor)
            control = EmbeddedControl(
                arm="frozen_lr0",
                coordinate=_arm_coordinate(CONTROL_CREDIT, env, frozen=True),
                chance=chance,
                tolerance=tolerance,
            )
        verdicts[env.name] = verify_embedded_control(records, control)
    return verdicts


def _contrasts_vs_gradient(
    arms: dict[str, ArmOutcome],
    envs: tuple[DepthEnv, ...],
    config: DeepCreditConfig,
) -> tuple[dict[str, dict[str, float]], float]:
    """Local arms vs gradient on held-out probe accuracy, per depth env.

    Sign convention: positive d means the gradient arm retains more.
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


def _contrast_d(group_a: list[float], group_b: list[float]) -> float:
    """Cohen's d, or 0.0 when both samples are constant (undefined spread)."""
    try:
        return cohens_d(group_a, group_b)
    except ValueError:
        return 0.0


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
            "trial": DEEP_CREDIT_CAMPAIGN_ID,
            "config": self.config,
            "envs": self.envs,
            "arms": {k: a.to_dict() for k, a in self.arms.items()},
            "contrasts_vs_gradient": self.contrasts_vs_gradient,
            "embedded_control_verdicts": self.control_verdicts,
            "quarantined": self.quarantined,
            "preregistration": self.preregistration.to_dict(),
        }


def run_trial(  # ruff: ignore[too-many-locals] - trial identity tuple travels together
    config: DeepCreditConfig,
    preregistration: PowerPreregistration | None = None,
) -> TrialResult:
    """Walk every arm through every depth and return the outcome.

    The planted lr=0 control walks every depth environment and must sit at
    chance in each (R8.5); a moving control anywhere quarantines the trial.
    The control's coordinate carries the U-axis ``frozen`` value per env, so
    the post-run verdict can never match a learning arm's records.

    ``preregistration`` (R8.4) commissions the run at a registered design:
    it must pass every claim-grade gate *before* the walk (fail loudly by
    name), must be resourced by the config's seed count, and its embedded
    control + tolerance become the authoritative post-run verdict — the
    self-labeled pilot preregistration is built only when commissioning
    without one.
    """
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
    chance = 1.0 / config.num_classes
    envs = _environments(config)
    arm_specs = (*[(c, False) for c in TRIAL_ARMS], (CONTROL_CREDIT, True))
    registered_control: EmbeddedControl | None = None
    if preregistration is not None:
        control = preregistration.embedded_control
        if control is None:  # unreachable: the claim-grade gate requires the arm
            raise ValueError(  # ruff: ignore[raise-vanilla-args] - unreachable guard
                "registered preregistration carries no embedded control"
            )
        registered_control = control
        prereg = preregistration
    control_records_by_env: dict[str, list[FrontierRecord]] = {}
    arms = {
        "control" if frozen else credit: _walk_arm(
            credit,
            frozen,
            envs,
            config=config,
            control_records_by_env=control_records_by_env,
        )
        for credit, frozen in arm_specs
    }
    env_reports = [
        {
            "name": env.name,
            "depth": env.depth,
            "unconstrained": env.unconstrained,
            "hidden_dims": list(env.hidden_dims),
        }
        for env in envs
    ]
    n_control_samples = config.episodes * config.batch_size * len(config.seeds)
    control_verdicts = _verify_controls(
        envs,
        control_records_by_env,
        chance,
        n_control_samples,
        config.control_band_floor,
        registered_control=registered_control,
    )
    contrasts, top_d = _contrasts_vs_gradient(arms, envs, config)

    if preregistration is None:
        pilot_control = EmbeddedControl(
            arm="frozen_lr0",
            coordinate=_arm_coordinate(CONTROL_CREDIT, envs[0], frozen=True),
            chance=chance,
            tolerance=max(
                at_chance_band(chance, n_control_samples), config.control_band_floor
            ),
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
                "Pilot: under depth-scaled credit assignment, ThermodynamicContrast "
                "and RandomProjections learn a 50+ step temporal dependency with "
                "O(1) saved-activation memory where exact-global Backprop's memory "
                "grows O(depth) and its gradients vanish; the shallow tier "
                "reproduces competence. Registered credit-at-depth claim follows "
                "once this pilot fixes variance and effect size."
            ),
            metric="probe_accuracy",
            claim_scope="credit_at_depth",
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
            "depths": list(config.depths),
            "width": config.width,
            "lr": config.lr,
            "batch_size": config.batch_size,
            "input_dim": config.input_dim,
            "num_classes": config.num_classes,
            "teacher_noise": config.teacher_noise,
            "probe_episode_base": PROBE_EPISODE_BASE,
            "chance_accuracy": chance,
            "margin_above_chance": _MARGIN_ABOVE_CHANCE,
            "memory_budget_mib": config.memory_budget_mib,
            "control_band_floor": config.control_band_floor,
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
    parser.add_argument("--depths", default="4,16,50")
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--memory-budget-mib", type=float, default=None)
    parser.add_argument(
        "--prereg",
        type=Path,
        default=None,
        help=(
            "Registered preregistration JSON (R8.4): claim-grade gate before the "
            "walk, registered n, and the authoritative embedded control (e.g. "
            "configs/preregistrations/r93_deep_credit_registered.json)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/deep_credit_pilot.json"),
    )
    args = parser.parse_args()

    config = DeepCreditConfig(
        episodes=args.episodes,
        late_window=args.late_window,
        probe_episodes=args.probe_episodes,
        seeds=tuple(int(s) for s in args.seeds.split(",") if s.strip()),
        depths=tuple(int(s) for s in args.depths.split(",") if s.strip()),
        width=args.width,
        lr=args.lr,
        batch_size=args.batch_size,
        input_dim=args.input_dim,
        num_classes=args.num_classes,
        memory_budget_mib=args.memory_budget_mib,
    )
    prereg = PowerPreregistration.load(args.prereg) if args.prereg else None
    result = run_trial(config, preregistration=prereg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8"
    )

    print(f"deep credit trial -> {args.output}")
    for arm in result.arms.values():
        probes = " ".join(
            f"{env}={np.mean(ps):.3f}" for env, ps in arm.probe_by_env.items()
        )
        saved = " ".join(
            f"{env}={b / 1e6:.2f}MiB" for env, b in arm.saved_bytes_by_env.items()
        )
        dq = " ".join(
            f"{env}={'DQ' if dq else 'ok'}"
            for env, dq in arm.disqualified_by_env.items()
        )
        print(f"  {arm.label:<22} probe {probes}")
        print(f"  {'':22} saved {saved}")
        print(f"  {'':22} dq    {dq}")
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
