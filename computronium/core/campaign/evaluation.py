"""Real campaign episode machinery shared by commissioning and the CLI.

Builds composed 6-D joint systems from coordinate strings and runs
fault-tolerant episodes: deterministic per-episode batches, one real
``train_step`` per episode, a windowed-growth guard probe feeding the
stability fields of ``FrontierRecord``, and ENTERING-episode checkpoints.
"""

from __future__ import annotations

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
from computronium.core.joint.state import CompositeState
from computronium.core.ontology import (
    CreditAssignmentConfig,
    GeometryConfig,
    ParameterUpdateConfig,
    PlasticityConfig,
    StateDynamicsConfig,
    SubstrateConfig,
)
from computronium.core.profiling import ResourceUsage
from computronium.core.stability import StabilityGuard
from computronium.core.system_trainer import (
    JointSystem,
    compose_joint_system_from_configs,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from computronium.core.joint.context import SystemContext

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 16
DEFAULT_INPUT_DIM = 8
DEFAULT_NUM_CLASSES = 8
COORDINATE_AXES = 6


class UnsupportedCoordinateError(ValueError):
    """Coordinate names an axis value with no configured implementation."""

    def __init__(self, axis: str, value: str) -> None:
        super().__init__(f"{axis}={value!r}")
        self.axis = axis


def episode_batch(
    episode: int,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    input_dim: int = DEFAULT_INPUT_DIM,
    num_classes: int = DEFAULT_NUM_CLASSES,
) -> tuple[Tensor, Tensor]:
    """Deterministic synthetic classification batch for one episode.

    Uses a local generator so batch draws never shift the global RNG stream
    (checkpoint redo equality depends on this).
    """
    generator = torch.Generator().manual_seed(1000 + episode)
    x = torch.randn(batch_size, input_dim, generator=generator)
    weights = torch.randn(input_dim, num_classes, generator=generator)
    y = (x @ weights).argmax(dim=-1)
    return x, y


def activity_transition(
    joint: JointSystem,
) -> Callable[[CompositeState, SystemContext], CompositeState]:
    """Whole-system activity transition F_θ(z) for stability probes."""

    def transition(z: CompositeState, _context: SystemContext) -> CompositeState:
        out = joint.geometry.forward(z.activity["x"], joint.substrate)
        logits = out[-1] if isinstance(out, list) else out
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
    name: _thunk(getattr(SubstrateConfig, name)) for name in ("digital",)
}
# Non-digital substrates are intentionally NOT accepted yet (see
# _EXCLUDED_AXES below for the reason).
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
}
_UPDATE_FACTORIES = {
    "euclidean": _thunk(ParameterUpdateConfig.euclidean, step_size=0.01)
}

# Axis values whose composition is NOT yet faithful or does not yet run under
# ``compose_joint_system_from_configs``. These are composition gaps to fix,
# NOT judgments on the methods themselves; probed empirically (TODO4 session
# 5). Revisit each after its underlying issue closes.
_EXCLUDED_AXES: dict[tuple[str, str], str] = {
    # Substrate class is selected by config.precision, which cannot
    # distinguish e.g. analog ("float32") from digital and silently falls
    # back to DigitalSubstrate — dropping behavioral overrides (noise
    # models, precision bounds). Needs an explicit substrate-type tag.
    **{
        ("substrate", s): "substrate class not selected faithfully by compose"
        for s in ("analog", "memristive", "neuromorphic", "sparse", "ternary")
    },
    (
        "credit",
        "gradient",
    ): "requires autograd through settle; composed train_step runs detached",
    (
        "credit",
        "backprop",
    ): "requires autograd through settle; composed train_step runs detached",
    **{
        ("update", name): (
            f"{name} step crashes on composed params "
            "(shape [hidden] vs [batch, hidden]; likely bias handling)"
        )
        for name in (
            "riemannian_orthogonal",
            "spectral_constrained",
            "natural_gradient",
            "elastic_consolidation",
        )
    },
}


_CREDIT_ALIASES = {
    "thermo": "thermodynamic_contrast",
    "backprop": "gradient",
}


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
) -> JointSystem:
    """Compose a JointSystem from a 6-D coordinate string."""
    parts = coordinate.split("/")
    if len(parts) != COORDINATE_AXES:
        raise UnsupportedCoordinateError("parts", coordinate)
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
    )


def evaluate_episode(  # noqa: PLR0913 - shape triple always defaults
    joint: JointSystem,
    *,
    coordinate: str,
    task_name: str,
    campaign_id: str,
    episode: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    input_dim: int = DEFAULT_INPUT_DIM,
    num_classes: int = DEFAULT_NUM_CLASSES,
) -> tuple[FrontierRecord, dict[str, float]]:
    """Run one real training episode and record its frontier metrics.

    The shape triple travels together and always defaults; explicit keywords
    beat inventing a container type for two call sites.
    """
    x, y = episode_batch(
        episode, batch_size=batch_size, input_dim=input_dim, num_classes=num_classes
    )
    started = time.perf_counter()
    metrics = joint.train_step(x, y)
    latency = time.perf_counter() - started

    z = CompositeState(activity={"x": x}, plastic={}, substrate={})
    guard = StabilityGuard(threshold=float("inf"), statistic="windowed_growth")
    growth = guard.probe(activity_transition(joint), z, joint.context)

    record = FrontierRecord(
        coordinate=coordinate,
        task_name=task_name,
        task_loss=metrics["loss"],
        task_accuracy=metrics["accuracy"],
        adaptation_time=1,
        rho_jacobian=growth,
        lyapunov_local=0.0,
        settling_time=float(guard.window),
        basin_stability=min(1.0, 1.0 / growth),
        resources=ResourceUsage(latency=latency),
        plasticity_primitive=coordinate.split("/")[3],
        registry_signature=compute_registry_signature(joint.context.registry),
        composite_state_shape=compute_composite_state_shape(joint.context),
        campaign_id=campaign_id,
        episode_index=episode,
    )
    logger.info(
        "episode %d [%s]: loss=%.4f acc=%.4f growth=%.3f",
        episode,
        coordinate,
        metrics["loss"],
        metrics["accuracy"],
        growth,
    )
    return record, metrics
