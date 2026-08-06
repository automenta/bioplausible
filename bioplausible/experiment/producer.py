"""Config production — the grid as an Optuna sampler (architecture §6.5).

The experiment layer schedules probes through :class:`ConfigProducer`, whose
default implementation (:class:`HyperoptGridProducer`) treats a stage's config
grid as an Optuna ``GridSampler`` search space. This reuses ``hyperopt``'s study
+ resume machinery: the probe count is enumerable from the grid cardinality
(exact ``plan``), and ``schedule`` yields :class:`ProbeWork` per (model, config)
for the surviving models, skipping already-finished ``config_key``s when a
Report is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import optuna

from bioplausible.experiment.probe import config_key

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from bioplausible.experiment.schema import Stage

__all__ = [
    "ConfigProducer",
    "HyperoptGridProducer",
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
    """Produces :class:`ProbeWork` for a stage's surviving models."""

    def schedule(
        self,
        stage: Stage,
        survivors: Sequence[str],
        finished: set[str] | None = None,
    ) -> Iterator[ProbeWork]: ...


class HyperoptGridProducer:
    """Grid-as-sampler :class:`ConfigProducer` over Optuna's ``GridSampler``.

    Deterministic (fixed grid order via ``GridSampler``), with the probe count
    enumerable from the grid cardinality. When ``finished`` (a set of completed
    ``config_key``s from the Report) is supplied, already-finished configs are
    skipped for resumability.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def _grid_for(self, stage: Stage) -> list[dict[str, object]]:
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

    def schedule(
        self,
        stage: Stage,
        survivors: Sequence[str],
        finished: set[str] | None = None,
    ) -> Iterator[ProbeWork]:
        """Yield :class:`ProbeWork` for every (model, grid config) pair.

        Args:
            stage: The stage whose grid is enumerated.
            survivors: Models that survived the preceding stages.
            finished: Completed ``config_key``s to skip (resume).

        Yields:
            One :class:`ProbeWork` per (model, config), skipping finished keys.
        """
        done = finished or set()
        for model in survivors:
            for config in self._grid_for(stage):
                key = config_key(config)
                probe_key = f"{model}:{key}"
                if probe_key in done:
                    continue
                yield ProbeWork(model=model, config=config, config_key=probe_key)
