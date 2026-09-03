"""Generic per-rule Pareto-frontier search over continuous ``RULE_SPACES`` (§4D.4).

Plan §4D: *"Run each bio rule's Bayesian search in its own rule-specific space
(including equilibrium params: damping, iterations, convergence threshold,
predictor-corrector config)."* This is the generic engine: it samples a rule's
continuous :data:`~computronium.hyperopt.search_space.RULE_SPACES` entry via TPE,
trains every probe through an injected driver, and returns that rule's Pareto
frontier (accuracy x FLOPs x memory x time) — directly comparable against the
ideal-backprop frontier via :func:`~computronium.hyperopt.comparator.compare_frontiers`.

The search + cache lifecycle is inherited from :class:`_FrontierFinder`;
:class:`~computronium.hyperopt.ideal_backprop.IdealBackpropFinder` is the
backprop-rule specialization of this shared engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from computronium.hyperopt._finder import (
    FrontierDriver,
    _dict_to_point,
    _FrontierFinder,
    _point_to_dict,
)
from computronium.hyperopt.frontier import RulePoint, pareto_frontier
from computronium.hyperopt.search_space import get_rule_space, validate_rule_space

if TYPE_CHECKING:
    import optuna

__all__ = [
    "RuleFrontierDecision",
    "RuleFrontierFinder",
    "find_rule_frontier",
    "sample_config_for_rule",
]

_DEFAULT_TASK: str = "mnist"
_DEFAULT_EPOCHS: int = 5

# Params sampled continuously (log/linear) that must be integral when handed to
# a model builder (dimensions, tile counts, etc.). Sampled as float then cast.
_INT_CAST_PARAMS: frozenset[str] = frozenset({"hidden_dim", "cube_size"})


@dataclass(frozen=True, slots=True)
class RuleFrontierDecision:
    """The Pareto-frontier result of one rule's Bayesian search on a task."""

    rule: str
    task: str
    budget_probes: int
    epochs: int
    target_hardware: str | None = None
    points: tuple[RulePoint, ...] = ()
    frontier: tuple[RulePoint, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible dict."""
        return {
            "rule": self.rule,
            "task": self.task,
            "budget_probes": self.budget_probes,
            "epochs": self.epochs,
            "target_hardware": self.target_hardware,
            "frontier": [_point_to_dict(p) for p in self.frontier],
        }


def sample_config_for_rule(trial: optuna.Trial, rule: str) -> dict[str, object]:
    """Sample one config from a rule's continuous ``RULE_SPACES`` entry.

    Each spec is a ``(min, max, scale)`` range or a discrete choice list:
    ``"log"``/``"linear"`` map to continuous (float, optionally log-scaled)
    suggestions, ``"int"`` maps to a discrete integer suggestion, and a list is
    suggested as a categorical. Params that must be integral (in
    :data:`_INT_CAST_PARAMS`) are cast after sampling.

    Args:
        trial: The Optuna trial to suggest on.
        rule: Rule key into ``RULE_SPACES`` (e.g. ``"eqprop"``).

    Returns:
        A config dict mirroring the rule's search space.

    Raises:
        ValueError: If the rule has no defined space.
    """
    space = get_rule_space(rule)
    config: dict[str, object] = {}
    for name, spec in space.items():
        if isinstance(spec, list):
            config[name] = trial.suggest_categorical(name, list(spec))
            continue
        min_v, max_v, scale = spec
        if scale == "int":
            config[name] = trial.suggest_int(name, int(min_v), int(max_v))
        elif scale == "log":
            config[name] = trial.suggest_float(name, min_v, max_v, log=True)
        else:  # "linear"
            config[name] = trial.suggest_float(name, min_v, max_v)
        if name in _INT_CAST_PARAMS:
            config[name] = int(config[name])
    return config


class RuleFrontierFinder(_FrontierFinder[RuleFrontierDecision]):
    """Runs (or loads) one rule's Pareto-frontier search for a task.

    Args:
        driver: Probe driver that executes one training run.
        rule: Rule key into ``RULE_SPACES`` (e.g. ``"eqprop"``, ``"neural_cube"``).
        model: Registered model name to train (defaults to the rule key).
        task: Task name (e.g. ``"mnist"``, ``"cifar10"``).
        budget_probes: Number of TPE probes.
        epochs: Training epochs per probe.
        seed: Master seed for reproducibility.
        device: Target device string.
        cache_dir: Directory to store the JSON frontier cache.
        target_hardware: Substrate facade for the probes (plan §17). Part of the
            cache identity.
    """

    _cache_prefix: str = "rule_frontier"
    _default_task: str = _DEFAULT_TASK
    _default_budget: int = 100
    _default_epochs: int = _DEFAULT_EPOCHS

    def __init__(  # ruff: ignore[too-many-arguments]
        self,
        driver: FrontierDriver,
        *,
        rule: str,
        model: str | None = None,
        task: str | None = None,
        budget_probes: int | None = None,
        epochs: int | None = None,
        seed: int = 42,
        device: str = "cpu",
        cache_dir: str = "logs",
        target_hardware: str | None = None,
    ) -> None:
        self.rule = rule
        self.model = model or rule
        super().__init__(
            driver,
            task=task,
            budget_probes=budget_probes,
            epochs=epochs,
            seed=seed,
            device=device,
            cache_dir=cache_dir,
            target_hardware=target_hardware,
        )

    @property
    def _rule_key(self) -> str:
        return self.rule

    @property
    def _train_model(self) -> str:
        return self.model

    @property
    def _cache_identity(
        self,
    ) -> dict[str, object]:
        return {**super()._cache_identity, "rule": self.rule}

    def _cache_name(self) -> str:
        # Rule-first segment order (``rule_frontier_{rule}_{task}``) is
        # preserved from the pre-§17 finder so existing on-disk caches stay
        # valid; epochs (§16.3) and target_hardware (§17) are part of identity.
        hw = "" if not self.ctx.target_hardware else f"_hw{self.ctx.target_hardware}"
        return (
            f"rule_frontier_{self.rule}_{self.ctx.task}"
            f"_epochs{self.ctx.epochs}_budget{self.ctx.budget_probes}{hw}.json"
        )

    def _sample_config(self, trial: optuna.Trial) -> dict[str, object]:
        return sample_config_for_rule(trial, self.rule)

    def _validate_before_search(self) -> None:
        """P0a gate: refuse to probe ``neural_cube``-style phantom spaces."""
        validate_rule_space(self.rule)

    def _build_decision(self, points: list[RulePoint]) -> RuleFrontierDecision:
        return RuleFrontierDecision(
            rule=self.rule,
            task=self.ctx.task,
            budget_probes=self.ctx.budget_probes,
            epochs=self.ctx.epochs,
            target_hardware=self.ctx.target_hardware,
            points=tuple(points),
            frontier=tuple(pareto_frontier(points)),
        )

    def _from_payload(self, payload: dict[str, object]) -> RuleFrontierDecision:
        points = tuple(_dict_to_point(self.rule, d) for d in payload.get("points", []))
        return RuleFrontierDecision(
            rule=self.rule,
            task=self.ctx.task,
            budget_probes=int(payload.get("budget_probes", self.ctx.budget_probes)),
            epochs=self.ctx.epochs,
            target_hardware=self.ctx.target_hardware,
            points=points,
            frontier=tuple(pareto_frontier(list(points))),
        )


def find_rule_frontier(  # ruff: ignore[too-many-arguments]
    driver: FrontierDriver,
    *,
    rule: str,
    model: str | None = None,
    task: str = _DEFAULT_TASK,
    budget_probes: int = 100,
    epochs: int = _DEFAULT_EPOCHS,
    seed: int = 42,
    device: str = "cpu",
    cache_dir: str = "logs",
    target_hardware: str | None = None,
    force: bool = False,
) -> RuleFrontierDecision:
    """Convenience wrapper to run :class:`RuleFrontierFinder.find`."""
    return RuleFrontierFinder(
        driver,
        rule=rule,
        model=model,
        task=task,
        budget_probes=budget_probes,
        epochs=epochs,
        seed=seed,
        device=device,
        cache_dir=cache_dir,
        target_hardware=target_hardware,
    ).find(force=force)
