"""Shared Optuna+cache machinery for the per-rule frontier finders (plan §4D/§9).

``IdealBackpropFinder`` and ``RuleFrontierFinder`` both run an Optuna TPE search
over a rule's continuous space, cache the resulting Pareto frontier to JSON, and
serve it back on later calls. They differ only in (a) which space they sample,
(b) the model identity that stamps the cache (``backprop`` vs ``rule``), and (c)
the decision dataclass they return. The search + cache + find lifecycle is
therefore shared here.

Cache identity (plan §16.3, §17): the cache filename and the payload-validity
check include ``epochs`` and ``target_hardware`` (plus task/model), so a frontier
measured at one epoch budget or substrate is never silently reused for another.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import optuna

from bioplausible.hyperopt.frontier import RulePoint

__all__ = [
    "FrontierDriver",
    "_FrontierFinder",
    "_SearchContext",
    "_dict_to_point",
    "_point_to_dict",
]

_OPTUNA_N_STARTUP: int = 10


@runtime_checkable
class FrontierDriver(Protocol):
    """Minimal training surface a finder depends on."""

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
class _SearchContext:
    """Knobs common to every frontier search (plan §4A/§8/§17)."""

    task: str
    budget_probes: int
    epochs: int
    seed: int = 42
    device: str = "cpu"
    cache_dir: str = "logs"
    target_hardware: str | None = None


def _point_to_dict(p: RulePoint) -> dict[str, object]:
    """Serialize a :class:`RulePoint` to a JSON-compatible dict."""
    return {
        "accuracy": p.accuracy,
        "total_flops": p.total_flops,
        "peak_memory_mb": p.peak_memory_mb,
        "wall_time_s": p.wall_time_s,
        "config": dict(p.config),
    }


def _dict_to_point(rule: str, d: dict[str, object]) -> RulePoint:
    """Deserialize a cached point dict back into a :class:`RulePoint`."""
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


class _FrontierFinder[D]:
    """Template-method base: cache-keyed Optuna search -> Pareto frontier.

    Subclasses supply the varying parts:
    - ``_cache_prefix``: filename prefix (``"ideal_backprop"`` / ``"rule_frontier"``).
    - ``_rule_key``: identity stamped on cache filename + ``RulePoint.rule``
      (the rule tag), and ``_train_model``: the registered model actually trained.
    - ``_cache_identity``: extra payload keys to validate on a cache hit.
    - ``_sample_config(trial)``: the rule's continuous search space.
    - ``_build_decision(points)`` / ``_from_payload(payload)``: construct the
      rule's decision dataclass fresh or from a cache hit.

    The default search knobs are class attributes so each subclass sizes its own
    probe budget without re-declaring the constructor.
    """

    _cache_prefix: str = ""
    _default_task: str = "mnist"
    _default_budget: int = 100
    _default_epochs: int = 5

    def __init__(  # ruff: ignore[too-many-arguments]  (finder bundles all search+context config at once)
        self,
        driver: FrontierDriver,
        *,
        task: str | None = None,
        budget_probes: int | None = None,
        epochs: int | None = None,
        seed: int = 42,
        device: str = "cpu",
        cache_dir: str = "logs",
        target_hardware: str | None = None,
    ) -> None:
        self.driver = driver
        self.ctx = _SearchContext(
            task=task if task is not None else self._default_task,
            budget_probes=(
                budget_probes if budget_probes is not None else self._default_budget
            ),
            epochs=epochs if epochs is not None else self._default_epochs,
            seed=seed,
            device=device,
            cache_dir=cache_dir,
            target_hardware=target_hardware,
        )
        self.cache_path = Path(self.ctx.cache_dir) / self._cache_name()

    # --- subclass hooks -------------------------------------------------

    @property
    def _rule_key(self) -> str:
        """Identity tag for the cache filename and :class:`RulePoint.rule`."""
        raise NotImplementedError

    @property
    def _train_model(self) -> str:
        """Registered model name handed to the probe driver."""
        return self._rule_key

    @property
    def _cache_identity(self) -> dict[str, object]:
        """Payload keys -> values validated on a cache hit (epochs & substrate)."""
        return {
            "task": self.ctx.task,
            "epochs": self.ctx.epochs,
            "target_hardware": self.ctx.target_hardware,
        }

    def _sample_config(self, trial: optuna.Trial) -> dict[str, object]:
        """Sample one probe config from this rule's search space."""
        raise NotImplementedError

    def _build_decision(self, points: list[RulePoint]) -> D:
        """Wrap a fresh search's points (incl. Pareto frontier) as this rule's decision."""
        raise NotImplementedError

    def _from_payload(self, payload: dict[str, object]) -> D:
        """Reconstruct this rule's decision from a validated cache payload."""
        raise NotImplementedError

    def _validate_before_search(self) -> None:
        """P0a gate hook: assert this rule's search space is honest before probing.

        Default is a no-op; rule finders override to run
        :func:`~bioplausible.hyperopt.search_space.validate_rule_space`. Called at
        the top of :meth:`find`, so a cached frontier for a now-phantom space is
        refused rather than silently reused.
        """

    # --- shared machinery ----------------------------------------------

    def _cache_name(self) -> str:
        # ``epochs`` (§16.3) and ``target_hardware`` (§17) are part of the
        # identity: a frontier derived at one epoch budget / substrate must not
        # be reused for another.
        hw = "" if not self.ctx.target_hardware else f"_hw{self.ctx.target_hardware}"
        return (
            f"{self._cache_prefix}_{self.ctx.task}_{self._rule_key}"
            f"_epochs{self.ctx.epochs}_budget{self.ctx.budget_probes}{hw}.json"
        )

    def load_cache(self) -> D | None:
        """Load this finder's cached frontier, or ``None`` if absent/invalid."""
        if not self.cache_path.exists():
            return None
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except ValueError:
            return None
        if not isinstance(payload, dict):
            return None
        if any(payload.get(k) != v for k, v in self._cache_identity.items()):
            return None
        return self._from_payload(payload)

    def find(self, force: bool = False) -> D:
        """Return this rule's frontier, training if not cached.

        Args:
            force: If True, re-run the search even when a cache exists.
        """
        self._validate_before_search()
        if not force:
            cached = self.load_cache()
            if cached is not None:
                return cached

        decision = self._search()
        self._save_cache(decision)
        return decision

    def _search(self) -> D:
        """Run the TPE search and compute this rule's Pareto frontier."""
        sampler = optuna.samplers.TPESampler(
            seed=self.ctx.seed,
            n_startup_trials=_OPTUNA_N_STARTUP,
            multivariate=True,
        )
        study = optuna.create_study(direction="maximize", sampler=sampler)
        points: list[RulePoint] = []
        rule_key = self._rule_key
        train_model = self._train_model
        ctx = self.ctx

        def _objective(trial: optuna.Trial) -> float:
            config = self._sample_config(trial)
            metrics = self.driver.train(
                model=train_model,
                task=ctx.task,
                config=config,
                seed=ctx.seed,
                epochs=ctx.epochs,
                device=ctx.device,
            )
            accuracy = float(metrics.get("final_acc", 0.0))
            total_flops = float(metrics.get("forward_flops", 0) or 0) + float(
                metrics.get("backward_flops", 0) or 0
            )
            points.append(
                RulePoint(
                    rule=rule_key,
                    accuracy=accuracy,
                    total_flops=total_flops,
                    peak_memory_mb=float(metrics.get("peak_memory_mb", 0.0)),
                    wall_time_s=float(metrics.get("wall_time_s", 0.0)),
                    config=tuple(sorted(config.items())),
                )
            )
            return accuracy

        study.optimize(_objective, n_trials=ctx.budget_probes)
        return self._build_decision(points)

    def _save_cache(self, decision: D) -> None:
        """Persist the frontier JSON for parameter-identical reuse."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(decision.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
