"""Real campaign episode machinery shared by commissioning and the CLI.

Builds composed 6-D joint systems from coordinate strings and runs
fault-tolerant episodes: deterministic per-episode batches, one real
``train_step`` per episode, a windowed-growth guard probe feeding the
stability fields of ``FrontierRecord``, and ENTERING-episode checkpoints.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from computronium.core.campaign.campaign_store import (
    compute_composite_state_shape,
    compute_registry_signature,
)
from computronium.core.campaign.frontier_record import FrontierRecord
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
from computronium.resources import MAC_ENERGY_J, ResourceUsage
from computronium.stability import StabilityGuard
from computronium.state import CompositeState

if TYPE_CHECKING:
    from collections.abc import Callable

    from computronium.state import SystemContext

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 16
DEFAULT_INPUT_DIM = 8
DEFAULT_NUM_CLASSES = 8
COORDINATE_AXES = 6

# R8.3 difficulty calibration: teacher-logit noise scale whose oracle
# accuracy (noiseless-teacher predictor against noisy labels) sits ≈0.86 at
# the registered shape — far above chance, with real headroom below the 1.0
# ceiling so accumulated-performance axes cannot saturate (imp-36 class).
CALIBRATED_TEACHER_NOISE = 0.5

# PR-5 calibration: windowed_growth ROC operating point (FKR=0%, KR=100%) on
# the Ginibre gain sweep. Real composed systems read exactly 1.000 when stable,
# so the margin holds there too; divergent substrates (ternary unit-alpha init,
# optical overflow) exceed it by orders of magnitude.
DEFAULT_GUARD_TAU = 1.029


class UnsupportedCoordinateError(ValueError):
    """Coordinate names an axis value with no configured implementation."""

    def __init__(self, axis: str, value: str) -> None:
        super().__init__(f"{axis}={value!r}")
        self.axis = axis


class IncompatibleCoordinateError(UnsupportedCoordinateError):
    """Coordinate pairs axis values that are conceptually incompatible (R3.9).

    Distinct from an implementation gap: no repair inside the paired axis
    values can make the coordinate meaningful, so it is rejected at
    composition rather than quarantined at attribution.
    """

    def __init__(self, pairing: str, reason: str) -> None:
        super().__init__("pairing", pairing)
        self.pairing = pairing
        self.reason = reason

    def __str__(self) -> str:
        return f"{self.pairing}: {self.reason}"


class GuardKillError(RuntimeError):
    """Guard statistic exceeded the calibrated threshold mid-episode.

    Raised after the episode's train_step completes so partial metrics are
    logged; runners catch this to skip the coordinate like an unsupported one.
    """

    def __init__(self, coordinate: str, statistic: float, threshold: float) -> None:
        super().__init__(
            f"guard kill on {coordinate}: growth={statistic:.3g} > τ={threshold:.3f}"
        )
        self.coordinate = coordinate
        self.statistic = statistic
        self.threshold = threshold


def _stable_seed(*parts: object) -> int:
    """Process-stable 64-bit digest of its parts (never Python's hash())."""
    payload = ":".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def episode_seed(
    base_seed: int, campaign_id: str, iteration: int, coordinate: str
) -> int:
    """Deterministic construction seed for one episode's θ initialization.

    Parameter init draws otherwise ride the ambient RNG stream, so a resumed
    campaign re-drew different initializations for repeated coordinates.
    Seeding construction from (base_seed, campaign_id, iteration, coordinate)
    makes episode construction replay-safe across crash/resume.
    """
    return _stable_seed(base_seed, campaign_id, iteration, coordinate)


def resolve_device(device: str | None) -> str:
    """GPU-first placement (imp-69): explicit value wins, ``None`` rides CUDA.

    Teacher streams live on CPU generators, so placement never changes the
    stream semantics — only parameter/batch residency.
    """
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _episode_targets(  # ruff: ignore[too-many-arguments] - noise knob completes the calibration contract
    task_name: str,
    x: Tensor,
    generator: torch.Generator,
    num_classes: int,
    *,
    teacher_generator: torch.Generator | None = None,
    teacher_noise: float = 0.0,
) -> Tensor:
    """Labels for one task family; unknown families are rejected, not faked.

    ``teacher_generator`` (R8.3): when set, the synthetic teacher is drawn
    from the stationarity-keyed stream instead of the per-episode stream.
    ``teacher_noise`` (R8.3 difficulty calibration): Gaussian noise on the
    teacher logits, lowering the task's achievable accuracy below 1.0 —
    the Bayes ceiling for a noiseless linear teacher. Drawn after the
    weights from the same generator, so each stream's first draws (and the
    legacy stream byte-for-byte) are untouched at the 0.0 default.
    """
    match task_name:
        case "synthetic":
            weights = torch.randn(
                x.shape[1],
                num_classes,
                generator=generator if teacher_generator is None else teacher_generator,
            )
            logits = x @ weights
            if teacher_noise > 0.0:
                source = generator if teacher_generator is None else teacher_generator
                logits += teacher_noise * torch.randn(
                    x.shape[0], num_classes, generator=source
                )
            return logits.argmax(dim=-1)
        case "parity":
            return (x > 0).sum(dim=-1) % num_classes
        case other:
            raise ValueError(  # ruff: ignore[raise-vanilla-args] - one-off validation message
                f"unsupported task family {other!r} (supported: synthetic, parity)"
            )


def episode_batch(  # ruff: ignore[too-many-arguments] - stationarity key completes the contract
    episode: int,
    *,
    task_name: str = "synthetic",
    batch_size: int = DEFAULT_BATCH_SIZE,
    input_dim: int = DEFAULT_INPUT_DIM,
    num_classes: int = DEFAULT_NUM_CLASSES,
    teacher_key: tuple[object, ...] | None = None,
    teacher_noise: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Deterministic synthetic batch for one episode of the named task family.

    Uses a local generator so batch draws never shift the global RNG stream
    (checkpoint redo equality depends on this). The "synthetic" stream keeps
    its original ``1000 + episode`` seeding so commissioned R5.1a/b campaign
    artifacts stay reproducible; other families derive an independent stream.

    ``teacher_key`` (R8.3 stationary design): when given, the synthetic
    teacher is derived from it alone — identical across episodes, so θ can
    accumulate learning — while inputs keep varying per episode. ``None``
    keeps the legacy per-episode teacher redraw (imp-54: non-stationary by
    design; per-episode-adaptation claim scope only).

    ``teacher_noise`` (R8.3 difficulty calibration): see
    ``_episode_targets``; 0.0 preserves every pre-R8.3 stream byte-for-byte.
    """
    if teacher_key is not None and task_name != "synthetic":
        msg = f"stationary teachers apply to the synthetic family, not {task_name!r}"
        raise ValueError(msg)
    seed = (
        1000 + episode if task_name == "synthetic" else _stable_seed(task_name, episode)
    )
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch_size, input_dim, generator=generator)
    teacher_generator = (
        None
        if teacher_key is None
        else torch.Generator().manual_seed(
            _stable_seed(task_name, "teacher", *teacher_key)
        )
    )
    return x, _episode_targets(
        task_name,
        x,
        generator,
        num_classes,
        teacher_generator=teacher_generator,
        teacher_noise=teacher_noise,
    )


def activity_transition(
    joint: JointSystem,
) -> Callable[[CompositeState, SystemContext], CompositeState]:
    """Whole-system activity transition F_θ(z) for stability probes.

    Returns an endomorphic map (input_dim -> input_dim) by padding/truncating
    the output logits to the input dimension. This enables the windowed-growth
    guard probe on nonsquare geometries (imp-60). Square systems are unchanged.
    """

    def transition(z: CompositeState, _context: SystemContext) -> CompositeState:
        out = joint.geometry.forward(z.activity["x"], joint.substrate)
        logits = out[-1] if isinstance(out, list) else out
        x = z.activity["x"]
        in_dim = x.shape[-1]
        if logits.shape[-1] != in_dim:
            # imp-60: dimension-preserving feedback via deterministic zero-pad.
            # No RNG side effects; the pad map is a fixed linear projection.
            padded = torch.zeros_like(x)
            n = min(logits.shape[-1], in_dim)
            padded[..., :n] = logits[..., :n]
            logits = padded
        return CompositeState(
            activity={"x": logits}, plastic=z.plastic, substrate=z.substrate
        )

    return transition


def _thunk[**P, T](
    factory: Callable[P, T], *args: P.args, **kwargs: P.kwargs
) -> Callable[[], T]:
    """Bind a config factory to fixed arguments as a true zero-arg callable."""
    return lambda: factory(*args, **kwargs)


_SUBSTRATE_FACTORIES = {
    name: _thunk(getattr(SubstrateConfig, name))
    for name in (
        "digital",
        "analog",
        "memristive",
        "neuromorphic",
        "sparse",
        "ternary",
        "optical",
        "quantum",
    )
}
_DYNAMICS_FACTORIES = {
    "energy_minimization": _thunk(
        StateDynamicsConfig.energy_minimization, max_steps=3, step_size=0.1
    ),
    "predictive_settling": _thunk(StateDynamicsConfig.predictive_settling, max_steps=3),
    "spike_integration": _thunk(StateDynamicsConfig.spike_integration, max_steps=3),
    "instantaneous": _thunk(StateDynamicsConfig.instantaneous),
    "diffusion": _thunk(StateDynamicsConfig.diffusion, max_steps=3),
}
_PLASTICITY_FACTORIES = {
    name: _thunk(getattr(PlasticityConfig, name))
    for name in ("null", "routing", "fast_weights", "substrate_coupled", "rule_state")
}
_CREDIT_FACTORIES = {
    "thermodynamic_contrast": _thunk(CreditAssignmentConfig.thermodynamic_contrast),
    "random_projections": _thunk(CreditAssignmentConfig.random_projections),
    "local_goodness": _thunk(CreditAssignmentConfig.local_goodness),
    "temporal_trace": _thunk(CreditAssignmentConfig.temporal_trace),
    "target_inversion": _thunk(CreditAssignmentConfig.target_inversion),
    "homeostatic": _thunk(CreditAssignmentConfig.homeostatic),
    "gradient": _thunk(CreditAssignmentConfig.gradient),
}
_UPDATE_FACTORIES = {
    "euclidean": _thunk(ParameterUpdateConfig.euclidean, step_size=0.01),
    # Planted-effect control arm (R8.5): a declared embedded control composes
    # this explicit value as its lr=0 coordinate — θ never consolidates, so
    # the arm must sit at chance on any learnable stream.
    "frozen": _thunk(ParameterUpdateConfig.euclidean, step_size=0.0),
    "riemannian_orthogonal": _thunk(
        ParameterUpdateConfig.riemannian_orthogonal, step_size=0.01
    ),
    "spectral_constrained": _thunk(
        ParameterUpdateConfig.spectral_constrained, step_size=0.01
    ),
    "natural_gradient": _thunk(ParameterUpdateConfig.natural_gradient, step_size=0.01),
    "elastic_consolidation": _thunk(
        ParameterUpdateConfig.elastic_consolidation, step_size=0.01
    ),
}

# Axis values whose composition is NOT yet faithful or does not yet run under
# ``compose_joint_system_from_configs``. These are composition gaps to fix,
# NOT judgments on the methods themselves; probed empirically (TODO4 session
# 5). Revisit each after its underlying issue closes.
#
# Empty since Phase 9: substrate classes are selected by the explicit
# ``substrate_type`` tag, gradient credit runs on the autograd-capable path
# (``requires_autograd``), and all update rules pair gradients via
# ``apply_pseudo_gradients`` (bias-safe).
_EXCLUDED_AXES: dict[tuple[str, str], str] = {}


_CREDIT_ALIASES = {
    "thermo": "thermodynamic_contrast",
    "backprop": "gradient",
}

# Settling dynamics iterate layered activations; tile-mesh routing exposes no
# layer sequence, so these pairs are structurally incompatible.
_LAYERED_ONLY_DYNAMICS = frozenset({
    "energy_minimization",
    "predictive_settling",
    "spike_integration",
})

# R3.9 validity matrix, D x C: ThermodynamicContrast's pseudo-gradient is the
# (nudged - free) settling contrast, defined only over dynamics with
# genuinely distinct settling phases. A single target-blind pass has no
# contrastive structure, so the pairing is conceptually dead (pinned by the
# R5b-0 gate: free = nudged implies structural zero). Fixable nudge gaps
# (predictive_settling) are NOT fenced — the dynamics probe quarantines
# them with a repair-shaped verdict instead.
_CONTRASTIVE_CREDITS = frozenset({"thermodynamic_contrast"})
_TARGET_BLIND_INSTANTANEOUS = "instantaneous"


def _check_pairwise(parts: list[str]) -> None:
    if parts[1] == "tile_mesh" and parts[2] in _LAYERED_ONLY_DYNAMICS:
        raise UnsupportedCoordinateError(
            "dynamics",
            f"{parts[2]} (requires layered geometry; incompatible with tile_mesh)",
        )
    if parts[2] == _TARGET_BLIND_INSTANTANEOUS and parts[4] in _CONTRASTIVE_CREDITS:
        raise IncompatibleCoordinateError(
            f"dynamics={parts[2]} x credit={parts[4]}",
            "contrastive settling credit requires target-responsive "
            "settlement; a single target-blind pass leaves free equal to "
            "nudged and the pseudo-gradient structurally zero",
        )


def _dispatch[T](table: dict[str, Callable[[], T]], axis: str, value: str) -> T:
    canonical = _CREDIT_ALIASES.get(value, value)
    factory = table.get(canonical)
    if factory is None or (axis, canonical) in _EXCLUDED_AXES:
        raise UnsupportedCoordinateError(axis, value)
    return factory()


def _geometry_config(
    geometry_type: str, input_dim: int, output_dim: int, hidden_dims: tuple[int, ...]
):
    match geometry_type:
        case "feedforward" | "recurrent":
            factory = getattr(GeometryConfig, geometry_type)
            return factory(
                input_dim=input_dim, output_dim=output_dim, hidden_dims=hidden_dims
            )
        case "tile_mesh":
            return GeometryConfig.tile_mesh(
                input_dim=input_dim,
                output_dim=output_dim,
                num_layers=max(len(hidden_dims), 1),
                neurons_per_tile=8,
                tiles_per_layer=2,
            )
        case other:
            raise UnsupportedCoordinateError("geometry", other)


def build_coordinate_system(
    coordinate: str,
    *,
    input_dim: int = DEFAULT_INPUT_DIM,
    output_dim: int = DEFAULT_NUM_CLASSES,
    hidden_dims: tuple[int, ...] = (16,),
    device: str | torch.device | None = None,
) -> JointSystem:
    """Compose a JointSystem from a 6-D coordinate string.

    Args:
        device: Optional target device (``None`` = build in place, typically
            CPU; ``"auto"`` = best available backend).
    """
    parts = coordinate.split("/")
    if len(parts) != COORDINATE_AXES:
        raise UnsupportedCoordinateError("parts", coordinate)
    _check_pairwise(parts)
    substrate_cfg = _dispatch(_SUBSTRATE_FACTORIES, "substrate", parts[0])
    geometry_cfg = _geometry_config(parts[1], input_dim, output_dim, hidden_dims)
    dynamics_cfg = _dispatch(_DYNAMICS_FACTORIES, "dynamics", parts[2])
    plasticity_cfg = _dispatch(_PLASTICITY_FACTORIES, "plasticity", parts[3])
    credit_cfg = _dispatch(_CREDIT_FACTORIES, "credit", parts[4])
    update_cfg = _dispatch(_UPDATE_FACTORIES, "update", parts[5])

    return compose_joint_system_from_configs(
        substrate_cfg,
        geometry_cfg,
        dynamics_cfg,
        plasticity_cfg,
        credit_cfg,
        update_cfg,
        device=device,
    )


def _episode_resources(
    joint: JointSystem,
    metrics: dict[str, float],
    *,
    batch_size: int,
    device: torch.device,
    latency: float,
) -> ResourceUsage:
    """Deterministic per-episode resource accounting (imp-17, R7 imp-45).

    The campaign Pareto needs resource axes that actually vary across the
    grid: recording wall-clock latency alone left compute/memory/energy
    constant at zero and collapsed the frontier to a single loss minimizer.
    Compute is a deterministic train-step MAC proxy: forward settle/phase
    work (params x batch x phases x settle steps) plus a documented backward
    estimate (backward ≈ 2x forward MACs — the R7 first pass found the
    forward-only version understated learning cost by the whole backward
    phase). Energy splits cleanly: the consumption axis is a work-derived
    estimate monotone in MACs, never the state's free energy — that is a
    state variable (may be negative) recorded separately as
    ``state_energy_j``. ψ-capacity comes from the plasticity config (no RNG
    side effects — never re-derive ``initial_psi`` here).
    """
    param_count = sum(p.numel() for p in joint.geometry.params.values())
    phases = max(len(getattr(joint.credit, "phases", ()) or ()), 1)
    settle_steps = max(int(getattr(joint.dynamics.config, "max_steps", 1) or 1), 1)
    forward_flops = 2 * batch_size * param_count
    settle_macs = forward_flops * phases * settle_steps
    backward_flops = 2 * settle_macs
    consumed_energy = (settle_macs + backward_flops) * MAC_ENERGY_J
    psi_dims = joint.plasticity.config.plastic_state_dims or {}
    return ResourceUsage(
        compute=float(settle_macs + backward_flops),
        memory=param_count * 4 / 1e6,
        energy=consumed_energy,
        latency=latency,
        plastic_state_capacity=float(sum(psi_dims.values())),
        device=str(device),
        batch_size=batch_size,
        forward_flops=forward_flops,
        backward_flops=backward_flops,
        param_count=param_count,
        wall_time_ms=latency * 1e3,
        state_energy_j=float(metrics["free_energy"]),
    )


def _teacher_key(
    campaign_id: str,
    coordinate: str,
    seed: int,
    *,
    stationary: bool,
    segment: str | None,
) -> tuple[object, ...] | None:
    """Stationarity key for one stream: per (campaign, coordinate, seed) or
    per (campaign, coordinate, seed, segment) when a task-sequence segment
    is declared (R9.1). ``None`` keeps the legacy per-episode teacher redraw
    (imp-54 stream). A segment without a stationary stream raises — a
    segmented legacy stream would silently re-open the imp-54
    non-stationarity inside each segment."""
    if segment is not None and not stationary:
        raise ValueError(  # ruff: ignore[raise-vanilla-args] - one-off validation message
            "segment-keyed teachers require stationary_teacher=True"
        )
    if not stationary:
        return None
    return (campaign_id, coordinate, seed) + ((segment,) if segment else ())


def evaluate_episode(  # ruff: ignore[too-many-arguments] - shape triple always defaults
    joint: JointSystem,
    *,
    coordinate: str,
    task_name: str,
    campaign_id: str,
    episode: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    input_dim: int = DEFAULT_INPUT_DIM,
    num_classes: int = DEFAULT_NUM_CLASSES,
    guard_threshold: float | None = DEFAULT_GUARD_TAU,
    seed: int = 0,
    stationary_teacher: bool = False,
    teacher_noise: float = 0.0,
    segment: str | None = None,
) -> tuple[FrontierRecord, dict[str, float]]:
    """Run one real training episode and record its frontier metrics.

    The shape triple travels together and always defaults; explicit keywords
    beat inventing a container type for two call sites. ``guard_threshold``
    gates kill decisions on the windowed-growth probe (``None`` records the
    statistic without deciding — harness/capability-probe mode). ``seed`` is
    stamped into the record so the replication gate can count seeds across
    campaigns sharing one store.

    ``stationary_teacher`` (R8.3): derives the synthetic teacher from
    (campaign_id, coordinate, seed) alone — identical across episodes, so θ
    can accumulate learning (claim scope: accumulated learning). The default
    ``False`` keeps the legacy per-episode teacher redraw (imp-54 stream;
    claim scope: per-episode adaptation only). ``teacher_noise`` calibrates
    task difficulty (see ``CALIBRATED_TEACHER_NOISE``).

    ``segment`` (R9.1): names one segment of a structured task-sequence
    stream (A→B); the teacher is stationary within a segment and re-keyed
    across segments, so forgetting/retention is measurable (claim scope:
    retention). Requires ``stationary_teacher`` — a segmented legacy stream
    would silently re-open the imp-54 non-stationarity inside each segment.

    Both design choices are stamped into the record metadata for
    artifact-level provenance.

    Batches are placed on the joint system's parameter device — the episode
    always executes where the system lives (no silent CPU fallback).
    """
    x, y = episode_batch(
        episode,
        task_name=task_name,
        batch_size=batch_size,
        input_dim=input_dim,
        num_classes=num_classes,
        teacher_key=_teacher_key(
            campaign_id,
            coordinate,
            seed,
            stationary=stationary_teacher,
            segment=segment,
        ),
        teacher_noise=teacher_noise,
    )
    device = joint.device
    x, y = x.to(device), y.to(device)
    started = time.perf_counter()
    metrics = joint.train_step(x, y)
    latency = time.perf_counter() - started

    z = CompositeState(activity={"x": x}, plastic={}, substrate={})
    guard = StabilityGuard(
        threshold=guard_threshold if guard_threshold is not None else float("inf"),
        statistic="windowed_growth",
    )
    growth = guard.probe(activity_transition(joint), z, joint.context)
    decision = guard.decide(growth)

    # task_accuracy = post-update target-free forward accuracy (honest learning metric)
    # nudged-settle fit stored in metadata for comparison only (imp-46 quarantine).
    # Strict free_* reads: the pipeline schema is closed (imp-46) — no fallback
    # to output-phase diagnostics, which would re-open the leakage channel.
    task_accuracy = metrics["free_accuracy"]
    nudged_fit_accuracy = metrics["nudged_fit_accuracy"]

    record = FrontierRecord(
        coordinate=coordinate,
        task_name=task_name,
        task_loss=metrics["free_loss"],
        task_accuracy=task_accuracy,
        adaptation_time=1,
        rho_jacobian=growth,
        lyapunov_local=0.0,
        settling_time=float(guard.window),
        basin_stability=min(1.0, 1.0 / growth),
        resources=_episode_resources(
            joint,
            metrics,
            batch_size=batch_size,
            device=device,
            latency=latency,
        ),
        plasticity_primitive=coordinate.split("/")[3],
        registry_signature=compute_registry_signature(joint.context.registry),
        composite_state_shape=compute_composite_state_shape(joint.context),
        metadata={
            "guard_kill": float(decision.kill),
            "nudged_fit_accuracy": nudged_fit_accuracy,
            "teacher_stationary": float(stationary_teacher),
            "teacher_noise": teacher_noise,
            **({"segment": segment} if segment else {}),
        },
        seed=seed,
        campaign_id=campaign_id,
        episode_index=episode,
    )
    logger.info(
        "episode %d [%s]: loss=%.4f free_acc=%.4f nudged_acc=%.4f growth=%.3f kill=%s",
        episode,
        coordinate,
        metrics["free_loss"],
        task_accuracy,
        nudged_fit_accuracy,
        growth,
        decision.kill,
    )
    if decision.kill:
        raise GuardKillError(coordinate, growth, decision.threshold)
    return record, metrics


def probe_episode(  # ruff: ignore[too-many-arguments] - shape triple always defaults
    joint: JointSystem,
    *,
    coordinate: str,
    task_name: str = "synthetic",
    campaign_id: str,
    episode: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    input_dim: int = DEFAULT_INPUT_DIM,
    num_classes: int = DEFAULT_NUM_CLASSES,
    seed: int = 0,
    stationary_teacher: bool = False,
    teacher_noise: float = 0.0,
    segment: str | None = None,
) -> float:
    """Target-free, no-train accuracy of the system's *current* state.

    The retention-side instrument (R9.1): scores the composed system on one
    episode batch without a train step — θ and ψ are untouched, so probing
    segment A mid-walk measures what the walk has retained. The accuracy
    definition is identical to the pipeline's post-update ``free_accuracy``
    (argmax of the settled output vs labels; labels score, never train).
    Teacher-key semantics match :func:`evaluate_episode` exactly, so a probe
    with the same (segment, key, episode-index space) sees the same stream
    the training episodes saw — use a disjoint episode-index space for
    held-out probe batches.
    """
    x, y = episode_batch(
        episode,
        task_name=task_name,
        batch_size=batch_size,
        input_dim=input_dim,
        num_classes=num_classes,
        teacher_key=_teacher_key(
            campaign_id,
            coordinate,
            seed,
            stationary=stationary_teacher,
            segment=segment,
        ),
        teacher_noise=teacher_noise,
    )
    device = joint.device
    with torch.no_grad():
        logits = joint.forward(x.to(device))
    return float((logits.argmax(dim=-1) == y.to(device)).float().mean())
