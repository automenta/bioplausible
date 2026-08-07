"""Generic per-rule Pareto-frontier search over continuous ``RULE_SPACES`` (§4D.4).

Plan §4D: *"Run each bio rule's Bayesian search in its own rule-specific space
(including equilibrium params: damping, iterations, convergence threshold,
predictor-corrector config)."* This is the generic engine: it samples a rule's
continuous :data:`~bioplausible.hyperopt.search_space.RULE_SPACES` entry via TPE,
trains every probe through an injected driver, and returns that rule's Pareto
frontier (accuracy x FLOPs x memory x time) — directly comparable against the
ideal-backprop frontier via :func:`~bioplausible.hyperopt.comparator.compare_frontiers`.

:class:`~bioplausible.hyperopt.ideal_backprop.IdealBackpropFinder` is the
backprop-rule specialization of this engine. Training the rule-specific
hyperparameters is delegated to the injected probe driver, mirroring how the
ideal-backprop finder trains backprop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import optuna

from bioplausible.hyperopt.frontier import RulePoint, pareto_frontier
from bioplausible.hyperopt.search_space import get_rule_space

__all__ = [
    "RuleFrontierDecision",
    "RuleFrontierFinder",
    "find_rule_frontier",
    "sample_config_for_rule",
]

_DEFAULT_TASK: str = "mnist"
_DEFAULT_EPOCHS: int = 5
_OPTUNA_N_STARTUP: int = 10

# Params sampled continuously (log/linear) that must be integral when handed to
# a model builder (dimensions, tile counts, etc.). Sampled as float then cast.
_INT_CAST_PARAMS: frozenset[str] = frozenset({"hidden_dim", "cube_size"})


class RuleDriver(Protocol):
    """Minimal training surface the finder depends on."""

    def train(  # ruff: ignore[too-many-arguments]  (mirrors the CoreTrainerDriver contract)
        self,
        *,
        model: str,
        task: str,
        config: dict[str, object],
        seed: int,
        epochs: int,
        device: str,
    ) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class RuleFrontierDecision:
    """The Pareto-frontier result of one rule's Bayesian search on a task."""

    rule: str
    task: str
    budget_probes: int
    points: tuple[RulePoint, ...]
    frontier: tuple[RulePoint, ...]

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible dict."""
        return {
            "rule": self.rule,
            "task": self.task,
            "budget_probes": self.budget_probes,
            "frontier": [_point_to_dict(p) for p in self.frontier],
        }


def _point_to_dict(p: RulePoint) -> dict[str, object]:
    return {
        "accuracy": p.accuracy,
        "total_flops": p.total_flops,
        "peak_memory_mb": p.peak_memory_mb,
        "wall_time_s": p.wall_time_s,
        "config": dict(p.config),
    }


def _dict_to_point(rule: str, d: dict[str, object]) -> RulePoint:
    config = d.get("config") or {}
    if not isinstance(config, dict):
        config = {}
    return RulePoint(
        rule=rule,
        accuracy=float(d.get("accuracy", 0.0)),
        total_flops=float(d.get("total_flops", 0.0)),
        peak_memory_mb=float(d.get("peak_memory_mb", 0.0)),
        wall_time_s=float(d.get("wall_time_s", 0.0)),
        config=tuple(sorted(config.items())),
    )


def sample_config_for_rule(trial: optuna.Trial, rule: str) -> dict[str, object]:
    """Sample one config from a rule's continuous ``RULE_SPACES`` entry.

    Each spec is a ``(min, max, scale)`` range or a discrete choice list:
    ``"log"``/``"linear"`` map to continuous (float, optionally log-scaled)
    suggestions, ``"int"`` maps to a discrete integer suggestion, and a list is
    suggested as a categorical.

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


class RuleFrontierFinder:
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
    """

    def __init__(  # ruff: ignore[too-many-arguments]  (finder bundles all search+context config at once)
        self,
        driver: RuleDriver,
        *,
        rule: str,
        model: str | None = None,
        task: str = _DEFAULT_TASK,
        budget_probes: int = 100,
        epochs: int = _DEFAULT_EPOCHS,
        seed: int = 42,
        device: str = "cpu",
        cache_dir: str = "logs",
    ) -> None:
        self.driver = driver
        self.rule = rule
        self.model = model or rule
        self.task = task
        self.budget_probes = budget_probes
        self.epochs = epochs
        self.seed = seed
        self.device = device
        self.cache_path = Path(cache_dir) / self._cache_name()

    def _cache_name(self) -> str:
        return f"rule_frontier_{self.rule}_{self.task}_budget{self.budget_probes}.json"

    def load_cache(self) -> RuleFrontierDecision | None:
        """Load the cached frontier for this rule/task, or ``None`` if absent."""
        if not self.cache_path.exists():
            return None
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except ValueError:
            return None
        if payload.get("rule") != self.rule or payload.get("task") != self.task:
            return None
        points = tuple(_dict_to_point(self.rule, d) for d in payload.get("points", []))
        return RuleFrontierDecision(
            rule=self.rule,
            task=self.task,
            budget_probes=int(payload.get("budget_probes", self.budget_probes)),
            points=points,
            frontier=tuple(pareto_frontier(list(points))),
        )

    def find(self, force: bool = False) -> RuleFrontierDecision:
        """Return the rule's frontier, training if not cached.

        Args:
            force: If True, re-run the search even when a cache exists.

        Returns:
            The cached or freshly-computed :class:`RuleFrontierDecision`.
        """
        get_rule_space(self.rule)  # validate rule early (raises ValueError)
        if not force:
            cached = self.load_cache()
            if cached is not None:
                return cached
        decision = self._search()
        self._save_cache(decision)
        return decision

    def _search(self) -> RuleFrontierDecision:
        """Run the TPE search and compute this rule's Pareto frontier."""
        sampler = optuna.samplers.TPESampler(
            seed=self.seed, n_startup_trials=_OPTUNA_N_STARTUP, multivariate=True
        )
        study = optuna.create_study(direction="maximize", sampler=sampler)
        points: list[RulePoint] = []

        def _objective(trial: optuna.Trial) -> float:
            config = sample_config_for_rule(trial, self.rule)
            metrics = self.driver.train(
                model=self.model,
                task=self.task,
                config=config,
                seed=self.seed,
                epochs=self.epochs,
                device=self.device,
            )
            accuracy = float(metrics.get("final_acc", 0.0))
            total_flops = float(metrics.get("forward_flops", 0) or 0) + float(
                metrics.get("backward_flops", 0) or 0
            )
            points.append(
                RulePoint(
                    rule=self.rule,
                    accuracy=accuracy,
                    total_flops=total_flops,
                    peak_memory_mb=float(metrics.get("peak_memory_mb", 0.0)),
                    wall_time_s=float(metrics.get("wall_time_s", 0.0)),
                    config=tuple(sorted(config.items())),
                )
            )
            return accuracy

        study.optimize(_objective, n_trials=self.budget_probes)
        return RuleFrontierDecision(
            rule=self.rule,
            task=self.task,
            budget_probes=self.budget_probes,
            points=tuple(points),
            frontier=tuple(pareto_frontier(points)),
        )

    def _save_cache(self, decision: RuleFrontierDecision) -> None:
        """Persist the frontier + raw points JSON for later reuse."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(decision.to_dict())
        payload["points"] = [_point_to_dict(p) for p in decision.points]
        self.cache_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )


def find_rule_frontier(  # ruff: ignore[too-many-arguments]  (convenience wrapper mirrors finder constructor)
    driver: RuleDriver,
    *,
    rule: str,
    model: str | None = None,
    task: str = _DEFAULT_TASK,
    budget_probes: int = 100,
    epochs: int = _DEFAULT_EPOCHS,
    seed: int = 42,
    device: str = "cpu",
    cache_dir: str = "logs",
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
    ).find(force=force)
