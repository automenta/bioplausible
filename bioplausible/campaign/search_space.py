"""Declarative, composable hyperparameter search spaces (FIX2a §4.1).

A :class:`SearchSpace` is the single source of truth for tunable
hyperparameters: a ``base`` set shared by every model, per-model
``overrides``, optional fixed ``defaults`` (immutable values such as
``gradient_method="equilibrium"``), and constraint expressions evaluated
at sample time.

The module is deliberately decoupled from the YAML schema — distributions
are plain value objects that Optuna can sample from directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable

    import optuna
    from optuna.distributions import CategoricalChoiceType

_PAIR_LEN = 2
_TRIPLE_LEN = 3

__all__ = [
    "Choice",
    "FloatRange",
    "IntRange",
    "ParamDistribution",
    "SearchSpace",
    "parse_distribution",
]


@runtime_checkable
class ParamDistribution(Protocol):
    """A single hyperparameter's sampling distribution for an Optuna trial."""

    def sample(self, trial: optuna.Trial, name: str) -> object:
        """Draw a concrete value from the distribution under ``trial``."""

    def describe(self) -> str:
        """Render a human-readable summary for ``--dry-run`` output."""
        ...


@dataclass(frozen=True, slots=True)
class FloatRange:
    """Continuous range sampled on a linear or log scale."""

    low: float
    high: float
    log: bool = False

    def sample(self, trial: optuna.Trial, name: str) -> float:
        return trial.suggest_float(name, self.low, self.high, log=self.log)

    def describe(self) -> str:
        scale = "log" if self.log else "linear"
        return f"float[{self.low}, {self.high}] {scale}"


@dataclass(frozen=True, slots=True)
class IntRange:
    """Discrete integer range."""

    low: int
    high: int

    def sample(self, trial: optuna.Trial, name: str) -> int:
        return trial.suggest_int(name, self.low, self.high)

    def describe(self) -> str:
        return f"int[{self.low}, {self.high}]"


@dataclass(frozen=True, slots=True)
class Choice:
    """Categorical choice over a fixed set of values."""

    values: tuple[object, ...]

    def sample(self, trial: optuna.Trial, name: str) -> object:
        choices = cast("list[CategoricalChoiceType]", list(self.values))
        return trial.suggest_categorical(name, choices)

    def describe(self) -> str:
        return f"choice{list(self.values)}"


def parse_distribution(value: object) -> ParamDistribution:
    """Normalize a YAML-shaped value into a :class:`ParamDistribution`.

    Accepted shapes (mirroring the FIX2a campaign schema):

    * ``[lo, hi]`` — continuous linear range of two numeric scalars.
    * ``[lo, hi, scale]`` — range where ``scale`` is ``"log"``/``"int"``
      (3-element form is used when a log scale is required).
    * any other list/tuple — categorical :class:`Choice`.
    * a :class:`ParamDistribution` instance is returned unchanged.
    """
    if isinstance(value, ParamDistribution):
        return value
    if isinstance(value, (list, tuple)):
        items = list(value)
        if len(items) == _PAIR_LEN and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in items
        ):
            low, high = float(items[0]), float(items[1])
            if float(low).is_integer() and float(high).is_integer():
                return IntRange(int(low), int(high))
            return FloatRange(low, high)
        if (
            len(items) == _TRIPLE_LEN
            and isinstance(items[2], str)
            and items[2] in {"log", "linear", "int"}
        ):
            scale = items[2]
            low, high = float(items[0]), float(items[1])
            if scale == "int":
                return IntRange(int(low), int(high))
            return FloatRange(low, high, log=scale == "log")
        return Choice(tuple(items))
    raise TypeError(  # ruff: ignore[raise-vanilla-args]  # descriptive message is the public API for callers
        f"Cannot parse distribution from {value!r}; expected [lo, hi], "
        "[lo, hi, scale], a list of choices, or a ParamDistribution"
    )


@dataclass(frozen=True, slots=True)
class SearchSpace:
    """Declarative, composable, Optuna-ready hyperparameter search space.

    Attributes:
        defaults: Fixed values injected into every sampled config. Used for
            immutable flags such as ``gradient_method="equilibrium"``.
        constants: Per-model fixed values (e.g. a model pinned to a specific
            non-tunable hyperparameter). Keys are registry model names.
        constraints: Python expressions over ``config`` and ``estimate()``
            that must hold for a config to be feasible (evaluated at sample
            time). ``estimate()`` is bound to the campaign's param estimator.
    """

    base: dict[str, ParamDistribution]
    overrides: dict[str, dict[str, ParamDistribution]] = field(default_factory=dict)
    defaults: dict[str, object] = field(default_factory=dict)
    constants: dict[str, dict[str, object]] = field(default_factory=dict)
    constraints: tuple[str, ...] = ()

    def for_model(self, model_name: str) -> dict[str, ParamDistribution]:
        """Return the merged distribution map for ``model_name``.

        Overrides replace base entries with the same name; overrides that are
        not present in the base are appended. Model-level defaults never appear
        here (they are constants, not tunable).
        """
        merged = dict(self.base)
        for name, dist in self.overrides.get(model_name, {}).items():
            merged[name] = dist
        return merged

    def sample(self, trial: optuna.Trial, model_name: str) -> dict[str, object]:
        """Draw a concrete config for ``model_name`` into ``trial``."""
        config = dict(self.defaults)
        for name, dist in self.for_model(model_name).items():
            config[name] = dist.sample(trial, name)
        for name, value in self.constants.get(model_name, {}).items():
            config[name] = value
        return config

    def render(self, model_name: str) -> dict[str, str]:
        """Render a dry-run description of the merged space for ``model_name``."""
        return {
            name: dist.describe() for name, dist in self.for_model(model_name).items()
        }

    def sample_feasible(
        self,
        trial: optuna.Trial,
        model_name: str,
        estimator: Callable[[dict[str, object]], int] | None = None,
        max_params: int | None = None,
    ) -> dict[str, object] | None:
        """Sample a config and reject it when constraints or budget fail.

        Returns ``None`` when the sampled config is infeasible so callers can
        mark the trial ``TrialPruned`` without spending any training compute.
        """
        config = self.sample(trial, model_name)
        if not self._constraints_hold(config, estimator):
            return None
        if (
            max_params is not None
            and estimator is not None
            and estimator(config) > max_params
        ):
            return None
        return config

    def _constraints_hold(
        self,
        config: dict[str, object],
        estimator: Callable[[dict[str, object]], int] | None,
    ) -> bool:
        if not self.constraints:
            return True
        if estimator is None:
            return True
        # The campaign YAML is trusted input; the expressions are authored by
        # the researcher, not end users. They run against a closed namespace.
        namespace: dict[str, object] = {
            "config": config,
            "estimate": estimator,
            **config,
        }
        for expr in self.constraints:
            try:
                # The campaign YAML is trusted, researcher-authored, and runs
                # against a closed namespace: only config keys and `estimate`.
                result = eval(expr, {"__builtins__": {}}, namespace)  # ruff: ignore[suspicious-eval-usage]
            except (NameError, TypeError, ValueError) as exc:
                raise ValueError(  # ruff: ignore[raise-vanilla-args]  # descriptive message is the public API
                    f"Constraint {expr!r} failed to evaluate: {exc}"
                ) from exc
            if not bool(result):
                return False
        return True
