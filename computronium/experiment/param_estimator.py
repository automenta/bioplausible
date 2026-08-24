"""Per-model parameter counting, built on the single construction layer.

The heavy lifting — reflection-driven knob routing, ``ModelConfig`` building,
and the canonical :func:`construct_model` — lives in
:mod:`computronium.core.construction`. This module keeps the stable public
parameter-estimation API that the campaign runner, scheduler and parity CLI
depend on, and re-exports the construction primitives so callers have one import
surface.

Do not add construction logic here: new construction behavior goes in
``core/construction.py`` so the trainer, the estimator, the finders and the
probe all share one path and can never disagree about a model's parameters or
knobs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, cast

from computronium.core.construction import (
    KNOBS,
)
from computronium.core.construction import (
    construct_model as _construct_model,
)
from computronium.core.construction import (
    model_kwargs as _model_kwargs,
)
from computronium.core.construction import (
    phantom_knobs as _phantom_knobs,
)
from computronium.core.registry import ComponentCategory, Registry
from computronium.utils import count_parameters

if TYPE_CHECKING:
    import torch

__all__ = [
    "KNOBS",
    "InstantiateEstimator",
    "ParamEstimateError",
    "ParamEstimator",
    "bound_estimator",
    "build_model_kwargs",
    "estimate_param_count",
    "phantom_knobs",
]


class ParamEstimateError(RuntimeError):
    """Raised when a model cannot be constructed for parameter counting."""


class ModuleFactory(Protocol):
    """A registered model constructor: builds an ``nn.Module`` from kwargs."""

    def __call__(self, **kwargs: object) -> torch.nn.Module: ...


class ParamEstimator(Protocol):
    """Static parameter-count protocol used by the campaign runner."""

    def estimate(
        self,
        model_name: str,
        config: dict[str, object],
        *,
        input_dim: int,
        output_dim: int,
    ) -> int: ...


#: Public alias: ``model_kwargs`` returns plain serializable scalars (the
#: ``TrainerConfig``/OmegaConf-safe view). ``construct_model`` turns them into a
#: live model with knobs applied.
build_model_kwargs = _model_kwargs
phantom_knobs = _phantom_knobs


class InstantiateEstimator:
    """Count parameters by constructing the model (exact, no training)."""

    @staticmethod
    def estimate(
        model_name: str,
        config: dict[str, object],
        *,
        input_dim: int,
        output_dim: int,
    ) -> int:
        model_cls = cast(
            "ModuleFactory", Registry.get(ComponentCategory.MODEL, model_name)
        )
        try:
            # Build via the canonical construction layer so the counted model is
            # bit-for-bit the one the trainer builds (same knob routing).
            model = _construct_model(
                model_cls,
                config,
                input_dim=input_dim,
                output_dim=output_dim,
                model_name=model_name,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ParamEstimateError(  # descriptive message is the public API
                f"Could not construct {model_name!r} for param counting: {exc}"
            ) from exc
        return count_parameters(model, trainable_only=False)  # type: ignore[arg-type]


def estimate_param_count(
    model_name: str,
    config: dict[str, object],
    *,
    input_dim: int,
    output_dim: int,
) -> int:
    """Estimate the parameter count for ``model_name`` under ``config``.

    This is a pure static analysis: the model is built via
    :func:`construct_model <computronium.core.construction.construct_model>`
    and its named parameters are summed. Used for the pre-training ``max_params``
    budget filter (§5.3) and for constraint expressions.
    """

    key = (model_name, input_dim, output_dim, _freeze_config(config))
    cached = _PARAM_COUNT_CACHE.get(key)
    if cached is not None:
        return cached
    # An exception here (ParamEstimateError) naturally returns before the cache
    # write below, so a failed construction is never cached and a transient
    # failure can recover on retry.
    count = InstantiateEstimator.estimate(
        model_name,
        config,
        input_dim=input_dim,
        output_dim=output_dim,
    )
    _PARAM_COUNT_CACHE[key] = count
    return count


def _freeze_config(
    config: dict[str, object],
) -> tuple[tuple[str, object], ...]:
    """Turn a config dict into a hashable, order-independent key fragment.

    Values are ``repr``-it if unhashable (e.g. a list choice); the original
    ``config`` is still passed to the estimator, so the cache is keyed by a
    faithful serialisation without ever mutating the caller's data.
    """
    frozen: list[tuple[str, object]] = []
    for key, value in config.items():
        try:
            hash(value)
        except TypeError:
            frozen.append((key, repr(value)))
        else:
            frozen.append((key, value))
    return tuple(sorted(frozen))


#: ``(model, dims, config) -> param count`` memo. Bounded by the number of
#: distinct (model, config, dims) triples a single process schedules, so it is
#: safely small for a campaign run.
_PARAM_COUNT_CACHE: dict[tuple[object, ...], int] = {}


def bound_estimator(
    model_name: str, input_dim: int, output_dim: int
) -> Callable[[dict[str, object]], int]:
    """Bind a model + dims into ``estimate(config) -> int``.

    The returned callable is the shape consumed by constraint expressions and
    :meth:`SearchSpace.sample_feasible`.
    """

    def _estimate(config: dict[str, object]) -> int:
        return estimate_param_count(
            model_name, config, input_dim=input_dim, output_dim=output_dim
        )

    return _estimate
