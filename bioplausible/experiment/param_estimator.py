"""Static per-model parameter counting (FIX2a §4.2, §13 step 3).

The estimator builds the model exactly the way :class:`CoreTrainer` does —
``Registry.get(MODEL, name)(**kwargs)`` — and sums ``numel()`` over its
parameters. Building a model is cheap and involves **no training**, which is
what the pre-training budget filter (§5.3) needs: configs that exceed an
arm's ``max_params`` are rejected before any compute is spent.

Constructor kwargs are filtered against the model's actual signature so that
over-riding search spaces (e.g. ``beta`` on a model that does not accept it)
never raise.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Protocol, cast

from bioplausible.core.registry import ComponentCategory, Registry

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

__all__ = [
    "InstantiateEstimator",
    "ParamEstimateError",
    "ParamEstimator",
    "bound_estimator",
    "build_model_kwargs",
    "estimate_param_count",
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


# Keys that may be present in a sampled config and map directly onto model
# constructor arguments. ``steps`` is aliased to ``max_steps`` because that is
# the name the EqProp/equilibrium models expose.
_KNOWN_KWARGS: frozenset[str] = frozenset({
    "beta",
    "alpha",
    "cube_size",
    "gradient_method",
    "feedback_init",
    "threshold",
    "use_spectral_norm",
    "spectral_norm_power_iterations",
    "hebbian_lr",
    "layer_lr",
    "classifier_lr",
})
_KWARG_ALIASES: dict[str, str] = {"steps": "max_steps"}


def _signature_params(model_cls: object) -> frozenset[str]:
    """Return the set of named parameters accepted by ``model_cls.__init__``.

    A ``**kwargs`` catch-all is represented by the sentinel string ``**``.
    """
    try:
        sig = inspect.signature(model_cls.__init__)
    except TypeError, ValueError:
        return frozenset()
    params = set(sig.parameters)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        params.add("**")
    params.discard("self")
    return frozenset(params)


def _derive_cube_size(config: dict[str, object]) -> dict[str, object]:
    """Map a neural_cube ``hidden_dim`` onto its ``cube_size``.

    Mirrors ``NeuralCube.build``: ``cube_size = max(3, round(hidden_dim ** (1/3)))``,
    used so a shared MLP-style search space can budget the cube model fairly.
    """
    if "cube_size" in config or "hidden_dim" not in config:
        return config
    cfg = dict(config)
    hidden = int(cast("int", cfg.pop("hidden_dim")))
    cfg["cube_size"] = max(3, round(hidden ** (1 / 3)))
    return cfg


def build_model_kwargs(
    model_cls: object,
    config: dict[str, object],
    *,
    input_dim: int,
    output_dim: int,
    model_name: str | None = None,
) -> dict[str, object]:
    """Build constructor kwargs from a sampled config, filtered to the signature.

    ``input_dim``/``output_dim``/``hidden_dim``/``num_layers`` are passed
    through; recognised tuning keys (``beta``, ``steps`` → ``max_steps``, …)
    are forwarded only when the constructor accepts them (directly or via
    ``**kwargs``). Other keys are ignored so an oversized search space never
    breaks construction. ``model_name`` enables per-model derivation such as
    the neural_cube's ``cube_size``.
    """
    if model_name == "neural_cube":
        config = _derive_cube_size(config)

    accepted = _signature_params(model_cls)
    has_catch_all = "**" in accepted

    kwargs: dict[str, object] = {
        "input_dim": config.get("input_dim", input_dim),
        "output_dim": config.get("output_dim", output_dim),
    }
    if not (has_catch_all or "input_dim" in accepted):
        kwargs.pop("input_dim", None)
    if not (has_catch_all or "output_dim" in accepted):
        kwargs.pop("output_dim", None)
    for key in ("hidden_dim", "num_layers"):
        if key in config and (has_catch_all or key in accepted):
            kwargs[key] = config[key]

    for key in _KNOWN_KWARGS:
        if key not in config:
            continue
        target = _KWARG_ALIASES.get(key, key)
        if has_catch_all or target in accepted:
            kwargs[target] = config[key]
    return kwargs


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
        kwargs = build_model_kwargs(
            model_cls,
            config,
            input_dim=input_dim,
            output_dim=output_dim,
            model_name=model_name,
        )
        try:
            model = model_cls(**kwargs)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ParamEstimateError(  # descriptive message is the public API
                f"Could not construct {model_name!r} for param counting: {exc}"
            ) from exc
        return sum(p.numel() for p in model.parameters())


def estimate_param_count(
    model_name: str,
    config: dict[str, object],
    *,
    input_dim: int,
    output_dim: int,
) -> int:
    """Estimate the parameter count for ``model_name`` under ``config``.

    This is a pure static analysis: the model is built (never trained) and its
    named parameters are summed. Used for the pre-training ``max_params``
    budget filter (§5.3) and for constraint expressions.
    """
    import bioplausible.zoo  # ruff: ignore[unused-import]  (ensure the model registry is populated before construction)

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
#: safely small for a campaign run. Reuses the exact estimator so `plan`, each
#: `run` stage, and `report` agree on a budget without rebuilding models.
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
