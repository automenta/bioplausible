"""``IdealBackpropFinder`` — the ideal-backprop reference frontier for a task (§9).

Plan §9/§4D: the "ideal backprop" is *not* one point but the **Pareto frontier**
of backprop over ``(accuracy, total_flops, memory, time)`` found by a full
Bayesian search over backprop's own continuous space. Every bio rule is then
compared against **this** frontier (at its own optimum), never against a single
coarse-grid baseline.

This is infrastructure, not an experiment: it runs the search once per task,
caches the resulting frontier (plus every measured point) to JSON, and serves
as the reference for all subsequent bio-rule experiments on that task.

Search space: :data:`~bioplausible.hyperopt.search_space.RULE_SPACES["backprop"]`
(the continuous, log-sampled ranges from §4A/§10).

Training is delegated to an injected probe driver (:class:`CoreTrainerDriver`),
so the component can be unit-tested with a fake driver and wired to the real
one in production.
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
    "IdealBackpropDecision",
    "IdealBackpropFinder",
    "find_ideal_backprop",
]

_DEFAULT_TASK: str = "mnist"
_DEFAULT_BUDGET: int = 2000
_DEFAULT_EPOCHS: int = 5
_OPTUNA_N_STARTUP: int = 10


class ProbeDriverProto(Protocol):
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
class IdealBackpropDecision:
    """The cached result of a full backprop search for one task.

    ``frontier`` is the Pareto-optimal backprop operating points (§11); the
    raw ``points`` are kept too, so callers can re-fit scaling laws or re-derive
    a frontier under a different dominance rule without retraining.
    """

    task: str
    budget_probes: int
    backprop: str
    points: tuple[RulePoint, ...]
    frontier: tuple[RulePoint, ...]

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible dict."""
        return {
            "task": self.task,
            "budget_probes": self.budget_probes,
            "backprop": self.backprop,
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


def _sample_backprop_config(
    trial: optuna.Trial, space: dict[str, object]
) -> dict[str, object]:
    """Sample one backprop config from the continuous rule space."""
    config: dict[str, object] = {}
    for name, spec in space.items():
        if name in {"lr", "weight_decay", "hidden_dim"}:
            config[name] = trial.suggest_float(name, spec[0], spec[1], log=True)
        elif name == "num_layers":
            config[name] = trial.suggest_int(name, int(spec[0]), int(spec[1]))
        elif name == "dropout":
            config[name] = trial.suggest_float(name, spec[0], spec[1])
        if name == "hidden_dim":
            config[name] = int(config[name])
    return config


class IdealBackpropFinder:
    """Runs (or loads) the full backprop frontier search for a task.

    Args:
        driver: Probe driver that executes one training run.
        task: Task name (e.g. ``"mnist"``, ``"cifar10"``).
        backprop: Registered backprop model name.
        budget_probes: Number of TPE probes (§4A: 500-1000+ for a true frontier).
        epochs: Training epochs per probe.
        seed: Master seed for reproducibility.
        device: Target device string.
        cache_dir: Directory to store the JSON frontier cache (default ``logs``).
    """

    def __init__(  # ruff: ignore[too-many-arguments]  (finder bundles all search+context config at once)
        self,
        driver: ProbeDriverProto,
        *,
        task: str = _DEFAULT_TASK,
        backprop: str = "backprop_mlp",
        budget_probes: int = _DEFAULT_BUDGET,
        epochs: int = _DEFAULT_EPOCHS,
        seed: int = 42,
        device: str = "cpu",
        cache_dir: str = "logs",
    ) -> None:
        self.driver = driver
        self.task = task
        self.backprop = backprop
        self.budget_probes = budget_probes
        self.epochs = epochs
        self.seed = seed
        self.device = device
        self.cache_path = Path(cache_dir) / self._cache_name()

    def _cache_name(self) -> str:
        return (
            f"ideal_backprop_{self.task}_{self.backprop}"
            f"_budget{self.budget_probes}.json"
        )

    def load_cache(self) -> IdealBackpropDecision | None:
        """Load the cached frontier for this task, or ``None`` if absent."""
        if not self.cache_path.exists():
            return None
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except ValueError:
            return None
        if payload.get("task") != self.task or payload.get("backprop") != self.backprop:
            return None
        frontier = tuple(
            _dict_to_point(self.backprop, d) for d in payload.get("frontier", [])
        )
        return IdealBackpropDecision(
            task=self.task,
            budget_probes=int(payload.get("budget_probes", self.budget_probes)),
            backprop=self.backprop,
            points=frontier,
            frontier=frontier,
        )

    def find(self, force: bool = False) -> IdealBackpropDecision:
        """Return the ideal-backprop frontier, training if not cached.

        Args:
            force: If True, re-run the search even when a cache exists.

        Returns:
            The cached or freshly-computed :class:`IdealBackpropDecision`.
        """
        if not force:
            cached = self.load_cache()
            if cached is not None:
                return cached

        decision = self._search()
        self._save_cache(decision)
        return decision

    def _search(self) -> IdealBackpropDecision:
        """Run the TPE search and compute the backprop Pareto frontier."""
        space = get_rule_space("backprop")
        sampler = optuna.samplers.TPESampler(
            seed=self.seed, n_startup_trials=_OPTUNA_N_STARTUP, multivariate=True
        )
        study = optuna.create_study(direction="maximize", sampler=sampler)

        points: list[RulePoint] = []

        def _objective(trial: optuna.Trial) -> float:
            config = _sample_backprop_config(trial, space)
            metrics = self.driver.train(
                model=self.backprop,
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
                    rule=self.backprop,
                    accuracy=accuracy,
                    total_flops=total_flops,
                    peak_memory_mb=float(metrics.get("peak_memory_mb", 0.0)),
                    wall_time_s=float(metrics.get("wall_time_s", 0.0)),
                    config=tuple(sorted(config.items())),
                )
            )
            return accuracy

        study.optimize(_objective, n_trials=self.budget_probes)

        return IdealBackpropDecision(
            task=self.task,
            budget_probes=self.budget_probes,
            backprop=self.backprop,
            points=tuple(points),
            frontier=tuple(pareto_frontier(points)),
        )

    def _save_cache(self, decision: IdealBackpropDecision) -> None:
        """Persist the frontier JSON.

        The cache is a pure function of the search inputs
        (task x model x budget x space), so overwriting it with an identical
        search is harmless.
        """
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(decision.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )


def find_ideal_backprop(  # ruff: ignore[too-many-arguments]  (convenience wrapper mirrors finder constructor)
    driver: ProbeDriverProto,
    *,
    task: str = _DEFAULT_TASK,
    backprop: str = "backprop_mlp",
    budget_probes: int = _DEFAULT_BUDGET,
    epochs: int = _DEFAULT_EPOCHS,
    seed: int = 42,
    device: str = "cpu",
    cache_dir: str = "logs",
    force: bool = False,
) -> IdealBackpropDecision:
    """Convenience wrapper to run :class:`IdealBackpropFinder.find`."""
    return IdealBackpropFinder(
        driver,
        task=task,
        backprop=backprop,
        budget_probes=budget_probes,
        epochs=epochs,
        seed=seed,
        device=device,
        cache_dir=cache_dir,
    ).find(force=force)
