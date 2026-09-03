"""R9.2/R9.3 memory-budget trial: the constraint family where O(1)-memory
credit is structurally immune.

Hypothesis: under a per-step saved-activation memory ceiling, exact-global
credit (gradient) and random-projection credit (FA) are disqualified once
their O(depth) saved-activation profile exceeds the budget — they cannot be
commissioned at the walled tier at all — while thermodynamic_contrast (0
saved bytes at every depth) is structurally immune. In the fully-walled
regime it is the only feasible arm and retains above-chance competence at
the shallow tier (the R9.3-registered shallow-tier signature); at the deep
tier nobody learns within the wall (the honest boundary of the linear-
teacher family). This is the severity lever the R9.2 analog-noise family
lacked (noise punishes settling arms; the memory budget cannot touch the
O(1) arm) and pairs R9.2's resource-efficiency scope with R9.3's
deterministic memory profile.

The budget is a commissioning gate, not a dynamics perturbation: a feasible
arm's walk is identical under every budget that admits it, so each
(arm, depth) cell walks once and is read under every budget through the
feasibility grid. A cell disqualified under every registered budget never
walks (the BPTT-OOM semantics: the arm cannot be commissioned at all). The
planted lr=0 control is a frozen thermodynamic_contrast arm — the only
credit feasible at every budget — so the at-chance verdict exists in every
regime (R8.5); its identity is the (credit, frozen) pair, never a name
comparison (imp-64). The first commission is a pilot by its own
preregistration (imp-55); the registered resource-efficiency claim is
gated through the R8.4 machinery with the pilot's variance (R8.4).

Command:
    uv run python -m computronium.experiments.joint.memory_budget_trial \
        --episodes 100 --seeds 0,1,2 \
        --output benchmark_results/memory_budget_pilot.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
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
from computronium.core.joint.transition import PlasticityConfig
from computronium.core.profiling import measure_saved_activation_bytes
from computronium.core.system_trainer import (
    JointSystem,
    compose_joint_system_from_configs,
)
from computronium.experiments.joint.deep_credit_trial import DepthEnv
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
    "BOUNDARY_DEPTHS",
    "BUDGETS_MIB",
    "CONTROL_CREDIT",
    "MEMORY_BUDGET_CAMPAIGN_ID",
    "PROBE_EPISODE_BASE",
    "TRIAL_ARMS",
    "ArmOutcome",
    "BoundaryMapResult",
    "MemoryBudgetConfig",
    "TrialResult",
    "main",
    "run_boundary_map",
    "run_trial",
]

MEMORY_BUDGET_CAMPAIGN_ID = "r92_memory_budget_pilot"
TRIAL_ARMS = ("gradient", "random_projections", "thermodynamic_contrast")
CONTROL_CREDIT = "thermodynamic_contrast"  # the only credit feasible at every budget
PROBE_EPISODE_BASE = 40_000  # disjoint episode-index space: held-out probe batches
_MARGIN_ABOVE_CHANCE = 0.1  # competence: probe accuracy above chance + margin
_MIN_CONTRAST_SEEDS = 2  # Cohen's d needs >= 2 observations per group

# Registered severity sweep (per-step saved-activation ceiling, MiB), sized
# against the R9.3-registered memory profile at the registered shape
# (width 16, input 8, batch 16): gradient 27,136/137,728/451,072 B and FA
# 29,728/152,704/501,136 B at depths 4/16/50, thermo 0. 0.015 MiB walls the
# O(depth) arms at every depth; 0.25 MiB walls them only at the deep tier;
# 0.45 MiB separates the two walled arms at the deep tier (gradient 451,072 B
# in, FA 501,136 B out).
BUDGETS_MIB = (0.015, 0.25, 0.45)

# Boundary-map depth sweep (R9 boundary-condition mapping): finer than the
# registered (4, 16, 50) grid, which already placed the walled-regime
# competence boundary between depth 4 (probe 0.406) and depth 16 (0.155);
# the map locates the transition and confirms the deep tier.
BOUNDARY_DEPTHS = (4, 6, 8, 12, 16, 24, 32, 50)

# Dynamics per credit (respects D x C fence: thermodynamic_contrast requires
# settling; max_steps is filled per depth at compose time so the nudge
# reaches the input layer).
_DYNAMICS_BY_CREDIT = {
    "gradient": "instantaneous",
    "random_projections": "instantaneous",
    "thermodynamic_contrast": "energy_minimization",
}


@dataclass(frozen=True, slots=True)
class MemoryBudgetConfig:
    """Trial design knobs (defaulted to the registered calibration).

    ``episodes``/``lr`` carry the R9.3 registered competence calibration
    (lr=0.05 @ 100 episodes on the stationary synthetic stream — imp-67:
    the declared stream is the enacted stream). ``budgets_mib`` is the
    severity sweep; the walk plan admits a cell iff some registered budget
    does, so a cell walled under every budget never walks.
    """

    episodes: int = 100
    late_window: int = 20
    probe_episodes: int = 8
    seeds: tuple[int, ...] = (0, 1, 2)
    depths: tuple[int, ...] = (4, 16, 50)  # credit steps = hidden layers + 1
    budgets_mib: tuple[float, ...] = BUDGETS_MIB
    width: int = 16
    lr: float = 0.05
    batch_size: int = 16
    input_dim: int = 8
    num_classes: int = 8
    teacher_noise: float = CALIBRATED_TEACHER_NOISE
    # Device for the composed systems (GPU-first policy): ``None`` resolves
    # to CUDA when available, else CPU. Teacher streams stay on CPU
    # generators — placement never changes the stream semantics.
    device: str | None = None
    # Control band floor for init-to-init variation (imp-59/imp-66).
    control_band_floor: float = 0.15


def _environments(config: MemoryBudgetConfig) -> tuple[DepthEnv, ...]:
    """Depth sweep: each depth is an independent environment (own teacher key)."""
    return tuple(
        DepthEnv(
            name=f"depth_{depth}",
            depth=depth,
            hidden_dims=tuple([config.width] * (depth - 1)),
            unconstrained=(i == 0),  # shallowest is the competence tier
        )
        for i, depth in enumerate(config.depths)
    )


def _arm_coordinate(credit: str, *, frozen: bool = False) -> str:
    """Coordinate is env-independent: depth lives in the geometry config."""
    dynamics = _DYNAMICS_BY_CREDIT[credit]
    update = "frozen" if frozen else "euclidean"
    return f"digital/feedforward/{dynamics}/null/{credit}/{update}"


def _compose(
    credit: str, env: DepthEnv, config: MemoryBudgetConfig, *, frozen: bool = False
) -> JointSystem:
    """One persistent system per (arm, env); θ init seeded per (campaign, credit)."""
    torch.manual_seed(episode_seed(0, MEMORY_BUDGET_CAMPAIGN_ID, 0, credit))

    dynamics_name = _DYNAMICS_BY_CREDIT[credit]
    if dynamics_name == "energy_minimization":
        dynamics = StateDynamicsConfig.energy_minimization(
            max_steps=max(3, env.depth),  # nudge must reach the input layer
            step_size=0.1,
            beta=0.5,
        )
    else:
        dynamics = getattr(StateDynamicsConfig, dynamics_name)()

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
        device=resolve_device(config.device),
    )


def _measure_saved_bytes(
    credit: str, env: DepthEnv, config: MemoryBudgetConfig
) -> float:
    """Saved-for-backward bytes of one train step (deterministic per cell)."""
    from computronium.core.campaign.evaluation import episode_batch

    joint = _compose(credit, env, config)
    x, y = episode_batch(
        0,
        task_name="synthetic",
        batch_size=config.batch_size,
        input_dim=config.input_dim,
        num_classes=config.num_classes,
        teacher_key=(MEMORY_BUDGET_CAMPAIGN_ID, env.name, credit, 0),
        teacher_noise=config.teacher_noise,
    )
    x, y = x.to(joint.device), y.to(joint.device)
    _, saved = measure_saved_activation_bytes(joint.train_step, x, y)
    return float(saved.total_bytes)


def _probe(
    joint: JointSystem,
    coordinate: str,
    campaign_id: str,
    seed: int,
    config: MemoryBudgetConfig,
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


def _walk_seed(
    credit: str,
    frozen: bool,
    env: DepthEnv,
    coordinate: str,
    seed: int,
    *,
    config: MemoryBudgetConfig,
) -> tuple[float, float, list[FrontierRecord]]:
    """One (arm, env, seed) walk: held-out probe, late-window acc, records."""
    joint = _compose(credit, env, config, frozen=frozen)
    campaign_id = f"{MEMORY_BUDGET_CAMPAIGN_ID}::{env.name}"
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
    return probe, late, records


def _walk_arm(
    label: str,
    credit: str,
    frozen: bool,
    envs: tuple[DepthEnv, ...],
    *,
    config: MemoryBudgetConfig,
    saved_bytes_by_cell: dict[str, float],
    walk_plan: dict[str, bool],
    control_records_by_env: dict[str, list[FrontierRecord]],
) -> ArmOutcome:
    """One arm's walk across the envs its walk plan admits.

    Saved-activation bytes were measured once per (credit, env) by the
    caller (the profile is deterministic — re-running train_step would
    double the settle cost and alter nothing).
    """
    coordinate = _arm_coordinate(credit, frozen=frozen)
    walked: list[str] = []
    probe_by_env: dict[str, tuple[float, ...]] = {}
    late_by_env: dict[str, tuple[float, ...]] = {}
    saved_bytes_by_env: dict[str, float] = {}
    for env in envs:
        cell = f"{label}@{env.name}"
        saved_bytes_by_env[env.name] = saved_bytes_by_cell[cell]
        if not walk_plan[cell]:
            continue
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
        walked.append(env.name)
        probe_by_env[env.name] = tuple(probes)
        late_by_env[env.name] = tuple(lates)
        if frozen:
            control_records_by_env.setdefault(env.name, []).extend(env_records)
    return ArmOutcome(
        label=label,
        coordinate=coordinate,
        frozen=frozen,
        saved_bytes_by_env=saved_bytes_by_env,
        walked_envs=tuple(walked),
        probe_by_env=probe_by_env,
        late_by_env=late_by_env,
    )


@dataclass(frozen=True, slots=True)
class ArmOutcome:
    """One arm's walk outcome: memory profile + per-env readouts.

    ``probe_by_env`` exists only for envs the walk plan admitted; a missing
    env means the arm could not be commissioned there under any registered
    budget (the OOM semantics), never that it was walked and failed.
    """

    label: str
    coordinate: str
    frozen: bool
    saved_bytes_by_env: dict[str, float]
    walked_envs: tuple[str, ...]
    probe_by_env: dict[str, tuple[float, ...]]
    late_by_env: dict[str, tuple[float, ...]]

    def to_dict(self) -> dict[str, object]:
        return {
            "coordinate": self.coordinate,
            "frozen": self.frozen,
            "saved_bytes_by_env": self.saved_bytes_by_env,
            "walked_envs": list(self.walked_envs),
            "probe_by_env": {k: list(v) for k, v in self.probe_by_env.items()},
            "late_by_env": {k: list(v) for k, v in self.late_by_env.items()},
        }


def _feasibility_grid(
    saved_bytes_by_cell: dict[str, float],
    envs: tuple[DepthEnv, ...],
    budgets_mib: tuple[float, ...],
) -> tuple[dict[str, dict[str, dict[str, bool]]], tuple[str, ...]]:
    """Budget x env feasibility grid over every arm label.

    A cell is feasible under a budget iff its measured saved-activation
    bytes fit the ceiling. The grid is the trial's commissioning instrument:
    a walled cell produces no walk and no records. Returns the grid plus the
    cells disqualified under every registered budget (never commissioned).
    """
    labels = (*TRIAL_ARMS, "control")
    grid: dict[str, dict[str, dict[str, bool]]] = {}
    never: list[str] = []
    for budget in budgets_mib:
        budget_bytes = budget * 1024 * 1024
        per_env: dict[str, dict[str, bool]] = {}
        for env in envs:
            per_env[env.name] = {
                label: saved_bytes_by_cell[f"{label}@{env.name}"] <= budget_bytes
                for label in labels
            }
        grid[f"{budget}"] = per_env
    for label in labels:
        for env in envs:
            if not any(grid[f"{b}"][env.name][label] for b in budgets_mib):
                never.append(f"{label}@{env.name}")
    return grid, tuple(never)


def _verify_controls(
    envs: tuple[DepthEnv, ...],
    control_records_by_env: dict[str, list[FrontierRecord]],
    chance: float,
    n_samples: int,
    control_band_floor: float,
    *,
    registered_control: EmbeddedControl | None = None,
) -> dict[str, ControlVerdict]:
    """Per-environment at-chance verdict for the planted lr=0 arm (R8.5).

    The control is a frozen ``CONTROL_CREDIT`` arm — the only credit feasible
    at every budget — so its verdict exists in every regime. ``registered_control``
    (R8.4) is authoritative when a preregistration commissions the run.
    """
    verdicts: dict[str, ControlVerdict] = {}
    for env in envs:
        records = control_records_by_env.get(env.name)
        if not records:
            continue
        control = registered_control
        if control is None:
            tolerance = max(at_chance_band(chance, n_samples), control_band_floor)
            control = EmbeddedControl(
                arm="frozen_lr0",
                coordinate=_arm_coordinate(CONTROL_CREDIT, frozen=True),
                chance=chance,
                tolerance=tolerance,
            )
        verdicts[env.name] = verify_embedded_control(records, control)
    return verdicts


def _contrast_d(group_a: list[float], group_b: list[float]) -> float:
    """Cohen's d, or 0.0 when both samples are constant (undefined spread)."""
    try:
        return float(cohens_d(group_a, group_b))
    except ValueError:
        return 0.0


@dataclass(frozen=True, slots=True)
class TrialResult:
    """Trial outcome bundle: memory profile, feasibility grid, arms, verdicts."""

    config: dict[str, object]
    envs: list[dict[str, object]]
    memory_profile_bytes: dict[str, float]
    feasibility: dict[str, dict[str, dict[str, bool]]]
    never_commissionable: tuple[str, ...]
    arms: dict[str, ArmOutcome]
    contrasts: dict[str, dict[str, float]]
    control_verdicts: dict[str, dict[str, str]]
    quarantined: bool
    preregistration: PowerPreregistration

    def to_dict(self) -> dict[str, object]:
        return {
            "trial": MEMORY_BUDGET_CAMPAIGN_ID,
            "config": self.config,
            "envs": self.envs,
            "memory_profile_bytes": self.memory_profile_bytes,
            "feasibility": self.feasibility,
            "never_commissionable": list(self.never_commissionable),
            "arms": {k: a.to_dict() for k, a in self.arms.items()},
            "contrasts": self.contrasts,
            "embedded_control_verdicts": self.control_verdicts,
            "quarantined": self.quarantined,
            "preregistration": self.preregistration.to_dict(),
        }


def run_trial(  # ruff: ignore[too-many-locals]
    config: MemoryBudgetConfig,
    preregistration: PowerPreregistration | None = None,
) -> TrialResult:
    """Measure the memory profile, walk the feasible cells, and read the grid.

    ``preregistration`` (R8.4) commissions the run at a registered design:
    it must pass every claim-grade gate *before* the walk (fail loudly by
    name), must be resourced by the config's seed count, and its embedded
    control + tolerance are the authoritative post-run verdict. Commissioning
    without one self-builds the pilot preregistration (declared rung caps the
    label; imp-55).
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

    labels = (*TRIAL_ARMS, "control")
    credit_of = dict(zip(TRIAL_ARMS, TRIAL_ARMS, strict=True)) | {
        "control": CONTROL_CREDIT
    }
    frozen_of = dict.fromkeys(TRIAL_ARMS, False) | {"control": True}
    saved_bytes_by_cell = {
        f"{label}@{env.name}": _measure_saved_bytes(credit_of[label], env, config)
        for label in labels
        for env in envs
    }
    max_budget_bytes = max(config.budgets_mib) * 1024 * 1024
    walk_plan = {
        f"{label}@{env.name}": saved_bytes_by_cell[f"{label}@{env.name}"]
        <= max_budget_bytes
        for label in labels
        for env in envs
    }

    control_records_by_env: dict[str, list[FrontierRecord]] = {}
    arms = {
        label: _walk_arm(
            label,
            credit_of[label],
            frozen_of[label],
            envs,
            config=config,
            saved_bytes_by_cell=saved_bytes_by_cell,
            walk_plan=walk_plan,
            control_records_by_env=control_records_by_env,
        )
        for label in labels
    }

    feasibility, never_commissionable = _feasibility_grid(
        saved_bytes_by_cell, envs, config.budgets_mib
    )

    n_control_samples = config.episodes * config.batch_size * len(config.seeds)
    registered_control: EmbeddedControl | None = None
    if preregistration is not None:
        control = preregistration.embedded_control
        if control is None:  # unreachable: the claim-grade gate requires the arm
            raise ValueError("registered preregistration carries no embedded control")
        registered_control = control
        prereg = preregistration
    control_verdicts = _verify_controls(
        envs,
        control_records_by_env,
        chance,
        n_control_samples,
        config.control_band_floor,
        registered_control=registered_control,
    )

    contrasts = _contrasts(arms, envs, len(config.seeds))

    if preregistration is None:
        prereg = _pilot_preregistration(arms, envs, config, chance, n_control_samples)
    return TrialResult(
        config={
            "episodes": config.episodes,
            "late_window": config.late_window,
            "probe_episodes": config.probe_episodes,
            "seeds": list(config.seeds),
            "depths": list(config.depths),
            "budgets_mib": list(config.budgets_mib),
            "width": config.width,
            "lr": config.lr,
            "batch_size": config.batch_size,
            "input_dim": config.input_dim,
            "num_classes": config.num_classes,
            "teacher_noise": config.teacher_noise,
            "device": resolve_device(config.device),
            "probe_episode_base": PROBE_EPISODE_BASE,
            "chance_accuracy": chance,
            "margin_above_chance": _MARGIN_ABOVE_CHANCE,
            "control_band_floor": config.control_band_floor,
        },
        envs=[
            {
                "name": env.name,
                "depth": env.depth,
                "unconstrained": env.unconstrained,
                "hidden_dims": list(env.hidden_dims),
            }
            for env in envs
        ],
        memory_profile_bytes=saved_bytes_by_cell,
        feasibility=feasibility,
        never_commissionable=never_commissionable,
        arms=arms,
        contrasts=contrasts,
        control_verdicts={
            name: {"verdict": v.verdict, "detail": v.detail}
            for name, v in control_verdicts.items()
        },
        quarantined=bool(control_verdicts)
        and any(v.quarantines for v in control_verdicts.values()),
        preregistration=prereg,
    )


def _contrasts(
    arms: dict[str, ArmOutcome], envs: tuple[DepthEnv, ...], n_seeds: int
) -> dict[str, dict[str, float]]:
    """Effect contrasts on held-out probe accuracy.

    ``thermo_vs_control@<env>`` is the claim contrast (walled-regime
    competence: positive d = the O(1) arm above its frozen control);
    ``gradient_vs_thermo@<env>`` is the honesty baseline where both arms
    are feasible.
    """
    contrasts: dict[str, dict[str, float]] = {}
    if n_seeds < _MIN_CONTRAST_SEEDS:
        return contrasts
    for env in envs:
        thermo = arms["thermodynamic_contrast"].probe_by_env.get(env.name)
        control = arms["control"].probe_by_env.get(env.name)
        gradient = arms["gradient"].probe_by_env.get(env.name)
        if thermo is not None and control is not None:
            d = _contrast_d(list(thermo), list(control))
            contrasts[f"thermo_vs_control@{env.name}"] = {"d_probe": round(d, 4)}
        if gradient is not None and thermo is not None:
            d = _contrast_d(list(gradient), list(thermo))
            contrasts[f"gradient_vs_thermo@{env.name}"] = {"d_probe": round(d, 4)}
    return contrasts


def _pilot_preregistration(
    arms: dict[str, ArmOutcome],
    envs: tuple[DepthEnv, ...],
    config: MemoryBudgetConfig,
    chance: float,
    n_samples: int,
    *,
    claim: str | None = None,
) -> PowerPreregistration:
    """Self-built pilot commission (declared rung caps the label; imp-55).

    Variance and effect size come from the claim contrast itself: the O(1)
    arm against its frozen control at the shallow (competence) tier — the
    walled-regime contrast the registered resource-efficiency claim needs.
    """
    shallow = envs[0].name
    thermo_probes = list(arms[CONTROL_CREDIT].probe_by_env[shallow])
    control_probes = list(arms["control"].probe_by_env[shallow])
    pooled_sd = float(np.std(thermo_probes + control_probes, ddof=1))
    return PowerPreregistration(
        claim=claim
        or (
            "Pilot: under a per-step saved-activation memory ceiling, "
            "exact-global and random-projection credit are disqualified once "
            "their O(depth) saved-activation profile exceeds the budget while "
            "thermodynamic_contrast (0 saved bytes at every depth) is "
            "structurally immune; in the fully-walled regime it is the only "
            "feasible arm and retains above-chance competence at the shallow "
            "tier, and at the deep tier no arm learns within the wall on this "
            "task family. Registered resource-efficiency claim follows once "
            "this pilot fixes variance and effect size."
        ),
        metric="probe_accuracy",
        claim_scope="resource_efficiency",
        task_stream="stationary",
        expected_effect=round(abs(_contrast_d(thermo_probes, control_probes)), 4),
        variance_estimate=round(pooled_sd, 6),
        n_per_group=len(config.seeds),
        embedded_control=EmbeddedControl(
            arm="frozen_lr0",
            coordinate=_arm_coordinate(CONTROL_CREDIT, frozen=True),
            chance=chance,
            tolerance=max(at_chance_band(chance, n_samples), config.control_band_floor),
        ),
        declared_rung="pilot",
        created=datetime.now(UTC).date().isoformat(),
    )


@dataclass(frozen=True, slots=True)
class BoundaryMapResult:
    """Walled-regime competence boundary (R9 boundary-condition mapping).

    ``arms`` contains only the arms the fully-walled budget admits
    (``thermodynamic_contrast`` + its frozen control); the O(depth) arms'
    measured saved-byte profiles ride ``saved_bytes_by_cell`` so the budget
    that would admit them at each depth is a measured number, never a
    quoted figure (imp-68). ``boundary_depth`` is the deepest swept depth
    whose mean held-out probe clears chance + margin (``None`` when no
    swept depth is competent).
    """

    config: dict[str, object]
    envs: list[dict[str, object]]
    walled_budget_mib: float
    saved_bytes_by_cell: dict[str, float]
    arms: dict[str, ArmOutcome]
    competence_by_env: dict[str, bool]
    boundary_depth: int | None
    contrasts: dict[str, dict[str, float]]
    control_verdicts: dict[str, dict[str, str]]
    quarantined: bool
    preregistration: PowerPreregistration

    def to_dict(self) -> dict[str, object]:
        return {
            "trial": f"{MEMORY_BUDGET_CAMPAIGN_ID}_boundary_map",
            "config": self.config,
            "envs": self.envs,
            "walled_budget_mib": self.walled_budget_mib,
            "saved_bytes_by_cell": self.saved_bytes_by_cell,
            "arms": {k: a.to_dict() for k, a in self.arms.items()},
            "competence_by_env": self.competence_by_env,
            "boundary_depth": self.boundary_depth,
            "contrasts": self.contrasts,
            "embedded_control_verdicts": self.control_verdicts,
            "quarantined": self.quarantined,
            "preregistration": self.preregistration.to_dict(),
        }


def _verify_walled_premise(
    saved_bytes_by_cell: dict[str, float],
    envs: tuple[DepthEnv, ...],
    budget_bytes: float,
    walled_credits: tuple[str, ...],
) -> None:
    """Fail loudly by name if the fully-walled regime is not actually walled.

    The map's premise: the O(1) control credit is commissionable at every
    swept depth (the R8.5 verdict must exist in every regime) and every
    O(depth) credit is walled at every swept depth — otherwise the sweep is
    not a walled-regime map and its readout would silently mix regimes.
    """
    for env in envs:
        thermo_bytes = saved_bytes_by_cell[f"{CONTROL_CREDIT}@{env.name}"]
        if thermo_bytes > budget_bytes:
            msg = (
                f"walled premise broken: {CONTROL_CREDIT}@{env.name} needs "
                f"{thermo_bytes:.0f} B > budget {budget_bytes:.0f} B — the "
                "frozen control would be uncommissionable"
            )
            raise ValueError(msg)
    leaked = [
        f"{credit}@{env.name}"
        for credit in walled_credits
        for env in envs
        if saved_bytes_by_cell[f"{credit}@{env.name}"] <= budget_bytes
    ]
    if leaked:
        msg = (
            f"budget {budget_bytes:.0f} B admits walled cells {leaked} — "
            "raise the premise: this is not the fully-walled regime"
        )
        raise ValueError(msg)


def run_boundary_map(  # ruff: ignore[too-many-locals]
    config: MemoryBudgetConfig,
    *,
    depths: tuple[int, ...] = BOUNDARY_DEPTHS,
    walled_budget_mib: float | None = None,
) -> BoundaryMapResult:
    """Map where the O(1) arm's walled-regime competence tier ends in depth.

    Walks only the arms the fully-walled budget (``min`` of the registered
    budgets unless overridden) admits — ``thermodynamic_contrast`` plus its
    frozen control (imp-64 identity: the (credit, frozen) pair) — across a
    finer depth sweep than the registered grid. Every swept depth's
    saved-activation profile is measured for all three credits, so the
    artifact carries the budget threshold that would admit each walled arm.
    The walled premise is verified against the measured bytes *before* the
    walk (fail loudly by name). Self-labels its commission ``pilot``
    (imp-55): the registered boundary claim follows once this pilot fixes
    variance and effect size.
    """
    budget_mib = (
        min(config.budgets_mib) if walled_budget_mib is None else walled_budget_mib
    )
    budget_bytes = budget_mib * 1024 * 1024
    map_config = replace(config, depths=depths)
    envs = _environments(map_config)
    chance = 1.0 / config.num_classes

    saved_bytes_by_cell = {
        f"{credit}@{env.name}": _measure_saved_bytes(credit, env, map_config)
        for credit in TRIAL_ARMS
        for env in envs
    }
    _verify_walled_premise(
        saved_bytes_by_cell,
        envs,
        budget_bytes,
        walled_credits=tuple(c for c in TRIAL_ARMS if c != CONTROL_CREDIT),
    )

    labels = (CONTROL_CREDIT, "control")
    thermo_bytes_by_env = {
        env.name: saved_bytes_by_cell[f"{CONTROL_CREDIT}@{env.name}"] for env in envs
    }
    walk_plan = {f"{label}@{env.name}": True for label in labels for env in envs}
    control_records_by_env: dict[str, list[FrontierRecord]] = {}
    arms: dict[str, ArmOutcome] = {}
    for label, frozen in ((CONTROL_CREDIT, False), ("control", True)):
        arms[label] = _walk_arm(
            label,
            CONTROL_CREDIT,
            frozen,
            envs,
            config=map_config,
            saved_bytes_by_cell={
                f"{label}@{env.name}": thermo_bytes_by_env[env.name] for env in envs
            },
            walk_plan=walk_plan,
            control_records_by_env=control_records_by_env,
        )

    competence_by_env = {
        env.name: bool(
            env.name in arms[CONTROL_CREDIT].probe_by_env
            and np.mean(arms[CONTROL_CREDIT].probe_by_env[env.name])
            > chance + _MARGIN_ABOVE_CHANCE
        )
        for env in envs
    }
    competent_depths = [env.depth for env in envs if competence_by_env[env.name]]
    boundary_depth = max(competent_depths) if competent_depths else None

    contrasts: dict[str, dict[str, float]] = {}
    if len(config.seeds) >= _MIN_CONTRAST_SEEDS:
        for env in envs:
            thermo = arms[CONTROL_CREDIT].probe_by_env.get(env.name)
            control = arms["control"].probe_by_env.get(env.name)
            if thermo is not None and control is not None:
                d = _contrast_d(list(thermo), list(control))
                contrasts[f"thermo_vs_control@{env.name}"] = {"d_probe": round(d, 4)}

    n_control_samples = config.episodes * config.batch_size * len(config.seeds)
    control_verdicts = _verify_controls(
        envs,
        control_records_by_env,
        chance,
        n_control_samples,
        config.control_band_floor,
    )
    prereg = _pilot_preregistration(
        arms,
        envs,
        map_config,
        chance,
        n_control_samples,
        claim=(
            "Pilot boundary map: under the fully-walled budget only "
            "thermodynamic_contrast is commissionable; the sweep locates the "
            "deepest depth whose held-out probe clears chance + margin. "
            "Registered boundary claim follows once this pilot fixes "
            "variance and effect size."
        ),
    )
    return BoundaryMapResult(
        config={
            "episodes": config.episodes,
            "probe_episodes": config.probe_episodes,
            "seeds": list(config.seeds),
            "boundary_depths": list(depths),
            "walled_budget_mib": budget_mib,
            "width": config.width,
            "lr": config.lr,
            "batch_size": config.batch_size,
            "input_dim": config.input_dim,
            "num_classes": config.num_classes,
            "teacher_noise": config.teacher_noise,
            "device": resolve_device(config.device),
            "probe_episode_base": PROBE_EPISODE_BASE,
            "chance_accuracy": chance,
            "margin_above_chance": _MARGIN_ABOVE_CHANCE,
            "control_band_floor": config.control_band_floor,
        },
        envs=[
            {
                "name": env.name,
                "depth": env.depth,
                "unconstrained": env.unconstrained,
                "hidden_dims": list(env.hidden_dims),
            }
            for env in envs
        ],
        walled_budget_mib=budget_mib,
        saved_bytes_by_cell=saved_bytes_by_cell,
        arms=arms,
        competence_by_env=competence_by_env,
        boundary_depth=boundary_depth,
        contrasts=contrasts,
        control_verdicts={
            name: {"verdict": v.verdict, "detail": v.detail}
            for name, v in control_verdicts.items()
        },
        quarantined=bool(control_verdicts)
        and any(v.quarantines for v in control_verdicts.values()),
        preregistration=prereg,
    )


def _report_boundary_map(result: BoundaryMapResult, output: Path) -> None:
    """Write and print the boundary-map artifact."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(f"memory budget boundary map -> {output}")
    for env in result.envs:
        name = str(env["name"])
        probes = result.arms[CONTROL_CREDIT].probe_by_env.get(name, ())
        verdict = (
            f"competent={result.competence_by_env[name]}"
            if probes
            else "(never walked)"
        )
        mean = np.mean(probes) if probes else float("nan")
        print(f"  {name:<10} thermo probe {mean:.3f} {verdict}")
    print(f"  boundary depth: {result.boundary_depth}")
    for name, contrast in result.contrasts.items():
        print(f"  {name}: d={contrast['d_probe']:+.3f}")
    for env, verdict in result.control_verdicts.items():
        print(f"  control[{env}]: {verdict['verdict']} — {verdict['detail']}")
    print(
        f"  prereg label: {result.preregistration.to_dict()['label']} "
        f"(quarantined={result.quarantined})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--late-window", type=int, default=20)
    parser.add_argument("--probe-episodes", type=int, default=8)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--depths", default="4,16,50")
    parser.add_argument("--budgets-mib", default="0.015,0.25,0.45")
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--num-classes", type=int, default=8)
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
            "walk, registered n, and the authoritative embedded control"
        ),
    )
    parser.add_argument(
        "--boundary-map",
        action="store_true",
        help=(
            "Map the walled-regime competence boundary: walk only the arms "
            "the fully-walled budget admits across a finer depth sweep "
            "(pilot-scoped; self-labels its commission)"
        ),
    )
    parser.add_argument(
        "--boundary-depths",
        default=",".join(map(str, BOUNDARY_DEPTHS)),
        help=f"Depth sweep for --boundary-map (default: {BOUNDARY_DEPTHS[0]}"
        f"..{BOUNDARY_DEPTHS[-1]})",
    )
    parser.add_argument(
        "--walled-budget-mib",
        type=float,
        default=None,
        help="Fully-walled budget override (default: min of --budgets-mib)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/memory_budget_pilot.json"),
    )
    args = parser.parse_args()

    config = MemoryBudgetConfig(
        episodes=args.episodes,
        late_window=args.late_window,
        probe_episodes=args.probe_episodes,
        seeds=tuple(int(s) for s in args.seeds.split(",") if s.strip()),
        depths=tuple(int(s) for s in args.depths.split(",") if s.strip()),
        budgets_mib=tuple(float(s) for s in args.budgets_mib.split(",") if s.strip()),
        width=args.width,
        lr=args.lr,
        batch_size=args.batch_size,
        input_dim=args.input_dim,
        num_classes=args.num_classes,
        device=args.device,
    )
    if args.boundary_map:
        if args.prereg:
            msg = (
                "the boundary map self-labels its commission pilot; a "
                "registered boundary commission needs its own preregistration"
            )
            raise ValueError(msg)
        result = run_boundary_map(
            config,
            depths=tuple(int(s) for s in args.boundary_depths.split(",") if s.strip()),
            walled_budget_mib=args.walled_budget_mib,
        )
        _report_boundary_map(result, args.output)
        return

    prereg = PowerPreregistration.load(args.prereg) if args.prereg else None
    result = run_trial(config, preregistration=prereg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8"
    )

    print(f"memory budget trial -> {args.output}")
    for cell, bytes_ in sorted(result.memory_profile_bytes.items()):
        print(f"  profile {cell:<36} {bytes_ / 1024:8.1f} KiB")
    for budget, per_env in result.feasibility.items():
        for env, verdicts in per_env.items():
            walled = [a for a, ok in verdicts.items() if not ok]
            print(f"  budget {budget} MiB @ {env}: walled {walled or 'none'}")
    if result.never_commissionable:
        print(f"  never commissionable: {list(result.never_commissionable)}")
    for arm in result.arms.values():
        probes = " ".join(
            f"{env}={np.mean(ps):.3f}" for env, ps in arm.probe_by_env.items()
        )
        print(f"  {arm.label:<24} probe {probes or '(never walked)'}")
    for name, contrast in result.contrasts.items():
        print(f"  {name}: d={contrast['d_probe']:+.3f}")
    for env, verdict in result.control_verdicts.items():
        print(f"  control[{env}]: {verdict['verdict']} — {verdict['detail']}")
    print(
        f"  prereg label: {result.preregistration.to_dict()['label']} "
        f"(quarantined={result.quarantined})"
    )


if __name__ == "__main__":
    main()
