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

The search + cache lifecycle is inherited from :class:`_FrontierFinder`; this
class only supplies the backprop-specialised space sampler and the
:class:`IdealBackpropDecision` (see §16.3/§17 for the cache identity).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from bioplausible.hyperopt._finder import (
    FrontierDriver,
    _dict_to_point,
    _FrontierFinder,
    _point_to_dict,
)
from bioplausible.hyperopt.frontier import RulePoint, pareto_frontier
from bioplausible.hyperopt.search_space import get_rule_space, validate_rule_space

if TYPE_CHECKING:
    import optuna

__all__ = [
    "IdealBackpropDecision",
    "IdealBackpropFinder",
    "find_ideal_backprop",
]

_DEFAULT_TASK: str = "mnist"
_DEFAULT_BUDGET: int = 2000
_DEFAULT_EPOCHS: int = 5
_DEFAULT_BACKPROP: str = "backprop_mlp"


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
    epochs: int
    target_hardware: str | None = None
    points: tuple[RulePoint, ...] = ()
    frontier: tuple[RulePoint, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible dict."""
        return {
            "task": self.task,
            "budget_probes": self.budget_probes,
            "backprop": self.backprop,
            "epochs": self.epochs,
            "target_hardware": self.target_hardware,
            "frontier": [_point_to_dict(p) for p in self.frontier],
        }


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


class IdealBackpropFinder(_FrontierFinder[IdealBackpropDecision]):
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
        target_hardware: Substrate facade for the probes (plan §17). Part of the
            cache identity — a GPU-derived reference is never reused for an
            FPGA/analog comparison.
    """

    _cache_prefix: str = "ideal_backprop"
    _default_task: str = _DEFAULT_TASK
    _default_budget: int = _DEFAULT_BUDGET
    _default_epochs: int = _DEFAULT_EPOCHS

    def __init__(  # ruff: ignore[too-many-arguments]  (subclass ctor re-exposes the base knobs + own model)
        self,
        driver: FrontierDriver,
        *,
        task: str | None = None,
        backprop: str = _DEFAULT_BACKPROP,
        budget_probes: int | None = None,
        epochs: int | None = None,
        seed: int = 42,
        device: str = "cpu",
        cache_dir: str = "logs",
        target_hardware: str | None = None,
    ) -> None:
        self.backprop = backprop
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
        return self.backprop

    @property
    def _cache_identity(
        self,
    ) -> dict[str, object]:
        return {**super()._cache_identity, "backprop": self.backprop}

    def _sample_config(self, trial: optuna.Trial) -> dict[str, object]:  # ruff: ignore[no-self-use]  # polymorphic template hook; backprop's space is fixed
        return _sample_backprop_config(trial, get_rule_space("backprop"))

    def _validate_before_search(self) -> None:
        """P0a gate: the backprop reference space must be honest too."""
        validate_rule_space("backprop")

    def _build_decision(self, points: list[RulePoint]) -> IdealBackpropDecision:
        return IdealBackpropDecision(
            task=self.ctx.task,
            budget_probes=self.ctx.budget_probes,
            backprop=self.backprop,
            epochs=self.ctx.epochs,
            target_hardware=self.ctx.target_hardware,
            points=tuple(points),
            frontier=tuple(pareto_frontier(points)),
        )

    def _from_payload(self, payload: dict[str, object]) -> IdealBackpropDecision:
        frontier = tuple(
            _dict_to_point(self.backprop, d) for d in payload.get("frontier", [])
        )
        return IdealBackpropDecision(
            task=self.ctx.task,
            budget_probes=int(payload.get("budget_probes", self.ctx.budget_probes)),
            backprop=self.backprop,
            epochs=self.ctx.epochs,
            target_hardware=self.ctx.target_hardware,
            points=frontier,
            frontier=frontier,
        )


def find_ideal_backprop(  # ruff: ignore[too-many-arguments]  (convenience wrapper mirrors finder constructor)
    driver: FrontierDriver,
    *,
    task: str = _DEFAULT_TASK,
    backprop: str = _DEFAULT_BACKPROP,
    budget_probes: int = _DEFAULT_BUDGET,
    epochs: int = _DEFAULT_EPOCHS,
    seed: int = 42,
    device: str = "cpu",
    cache_dir: str = "logs",
    target_hardware: str | None = None,
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
        target_hardware=target_hardware,
    ).find(force=force)
