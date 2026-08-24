"""Config production — the grid as an Optuna sampler (architecture §6.5).

The experiment layer schedules probes through :class:`ConfigProducer`, whose
default implementation (:class:`HyperoptGridProducer`) treats a stage's config
grid as an Optuna ``GridSampler`` search space. This reuses ``hyperopt``'s study
+ resume machinery: the probe count is enumerable from the grid cardinality
(exact ``plan``). The producer exposes a single method,
:meth:`ConfigProducer.configs_for`, which enumerates a stage's grid once;
scheduling (per-model probe work) and resume-skip live in the callers
(``StaircaseRunner`` / ``cli``), not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import optuna

if TYPE_CHECKING:
    from computronium.experiment.schema import Stage
__all__ = [
    "ConfigProducer",
    "HyperoptGridProducer",
    "OptunaBayesProducer",
    "ProbeWork",
    "grid_cardinality",
]


@dataclass(frozen=True, slots=True)
class ProbeWork:
    """One scheduled unit of work: a model under one config."""

    model: str
    config: dict[str, object]
    config_key: str


def grid_cardinality(configs: dict[str, list[object]]) -> int:
    """Return the exact number of unique config combinations in a grid."""
    if not configs:
        return 1
    product = 1
    for choices in configs.values():
        product *= len(choices)
    return product


@runtime_checkable
class ConfigProducer(Protocol):
    """Produces the ordered configs for a stage's grid.

    The single scheduling seam: `configs_for` yields the deterministic,
    cardinality-exact grid enumeration that both ``plan`` and ``run`` consume.
    """

    def configs_for(self, stage: Stage) -> list[dict[str, object]]: ...


class HyperoptGridProducer:
    """Grid-as-sampler :class:`ConfigProducer` over Optuna's ``GridSampler``.

    Deterministic (fixed grid order via ``GridSampler``), with the exact probe
    count following from the grid cardinality. The grid is enumerated **once
    per stage** and shared across every surviving model (no per-model study).
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def configs_for(self, stage: Stage) -> list[dict[str, object]]:
        """Enumerate the stage's grid into an ordered list of configs.

        Uses Optuna's ``GridSampler`` so the enumeration is deterministic and
        the exact probe count follows directly from the grid cardinality.
        Enumerating is cheap (pure, no training) and stage-scoped: call once
        and reuse across every surviving model.
        """
        space: dict[str, list[object]] = {
            name: list(choices) for name, choices in stage.configs.items()
        }
        sampler = optuna.samplers.GridSampler(space, seed=self.seed)
        study = optuna.create_study(sampler=sampler)

        # Objective must call suggest_categorical for each grid column
        # so GridSampler stores the enumerated params (not empty {})
        def _objective(trial: optuna.Trial) -> float:
            for name, choices in space.items():
                trial.suggest_categorical(name, choices)
            return 0.0

        study.optimize(_objective, n_trials=grid_cardinality(stage.configs))
        return [dict(t.params) for t in study.trials]


class OptunaBayesProducer:
    """TPESampler-backed :class:`ConfigProducer` (architecture §6.5, plan §B.1).

    Swaps :class:`HyperoptGridProducer`'s exhaustive ``GridSampler`` for Optuna's
    tree-structured Parzen estimator while keeping the same single-interface
    contract: ``configs_for`` returns a deterministic, cardinality-exact list of
    configs that both ``plan`` and ``run`` consume. The sampler draws
    ``n_candidates`` distinct points from the stage's grid; the exact probe
    count therefore follows from ``n_candidates`` rather than the grid
    cardinality. ``n_trials`` is capped below the grid space size so TPE cannot
    exhaust the search space into a plain grid enumeration.
    """

    def __init__(
        self,
        n_candidates: int = 50,
        seed: int = 42,
        pruner: optuna.pruners.BasePruner | None = None,
    ) -> None:
        self.n_candidates = n_candidates
        self.seed = seed
        self.pruner = pruner
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def configs_for(self, stage: Stage) -> list[dict[str, object]]:
        """Enumerate ``n_candidates`` TPE-sampled configs from the stage grid.

        The grid's each column is presented to the sampler as a categorical
        over its declared choices (the same suggestion shape the grid producer
        uses), so a stage's config space and its budget constraints are
        unchanged — only the sampling strategy differs.
        """
        space: dict[str, list[object]] = {
            name: list(choices) for name, choices in stage.configs.items()
        }
        n_trials = max(1, min(self.n_candidates, grid_cardinality(stage.configs)))
        sampler = optuna.samplers.TPESampler(seed=self.seed, n_startup_trials=1)
        study = optuna.create_study(sampler=sampler, pruner=self.pruner)

        def _objective(trial: optuna.Trial) -> float:
            for name, choices in space.items():
                trial.suggest_categorical(name, choices)
            return 0.0

        seen: set[str] = set()
        configs: list[dict[str, object]] = []
        while len(configs) < n_trials and len(seen) < grid_cardinality(stage.configs):
            study.optimize(_objective, n_trials=1)
            trial = study.trials[-1]
            key = ",".join(f"{k}={v!r}" for k, v in sorted(trial.params.items()))
            if key in seen:
                continue
            seen.add(key)
            configs.append(dict(trial.params))
        return configs
