"""R7 probe #4 (imp-46): metric-honesty locks on the campaign claim chain.

Pins, over the commissioned ``joint_grid`` credit families:
- the closed metric schema of ``run_train_step`` — parity keys present, bare
  ``accuracy`` extinct (renamed ``nudged_fit_accuracy``, quarantined),
- ``evaluate_episode`` claim fields derived exclusively from the post-update
  target-free settle: ``task_loss``/``task_accuracy`` ==
  ``free_loss``/``free_accuracy``; the nudged fit lives in metadata only,
- state energy vs consumption split: ``state_energy_j`` tracks the free
  settle energy, never the work-derived consumption estimate.

The human-readable census lives in ``docs/metric_provenance.md``; a schema
change must update both.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from computronium.cli.campaign import _get_search_space
from computronium.core.campaign import space_grid
from computronium.core.campaign.evaluation import (
    build_coordinate_system,
    evaluate_episode,
)
from computronium.core.pipeline import METRIC_SCHEMA

GRID = space_grid(_get_search_space("joint_grid"))
CREDITS = sorted({c.split("/")[4] for c in GRID})
PARITY_KEYS = frozenset(METRIC_SCHEMA)


def _batch(batch: int = 16, dim: int = 8) -> tuple[Tensor, Tensor]:
    return torch.randn(batch, dim), torch.randint(0, 8, (batch,))


@pytest.mark.parametrize("credit", CREDITS)
def test_pipeline_schema_is_closed(credit: str) -> None:
    """Every credit family emits the parity schema; bare ``accuracy`` is extinct."""
    system = build_coordinate_system(
        f"digital/feedforward/energy_minimization/null/{credit}/euclidean"
    )
    x, y = _batch()
    metrics = system.train_step(x, y)
    assert set(metrics) >= PARITY_KEYS, (
        f"{credit}: missing {PARITY_KEYS - set(metrics)}"
    )
    assert "accuracy" not in metrics, (
        f"{credit}: bare 'accuracy' key resurrected — leak channel re-opened"
    )


class TestEvaluateEpisodeProvenance:
    coordinate = "digital/feedforward/instantaneous/null/random_projections/euclidean"

    @pytest.fixture(scope="class")
    def episode(
        self,
    ) -> tuple[object, object, dict[str, float]]:
        joint = build_coordinate_system(self.coordinate)
        record, metrics = evaluate_episode(
            joint,
            coordinate=self.coordinate,
            task_name="parity",
            campaign_id="imp46-lock",
            episode=0,
            guard_threshold=None,
        )
        return joint, record, metrics

    def test_claim_fields_are_free_settle_reads(self, episode) -> None:
        _, record, metrics = episode
        assert record.task_loss == metrics["free_loss"]
        assert record.task_accuracy == metrics["free_accuracy"]

    def test_nudged_fit_is_metadata_only(self, episode) -> None:
        _, record, metrics = episode
        assert record.metadata["nudged_fit_accuracy"] == metrics["nudged_fit_accuracy"]
        assert "nudged_fit_accuracy" not in {"task_loss", "task_accuracy"}

    def test_state_energy_is_free_energy_not_consumption(self, episode) -> None:
        _, record, metrics = episode
        assert record.resources.state_energy_j == pytest.approx(metrics["free_energy"])
        assert record.resources.energy >= 0.0
        assert record.resources.compute > 0.0
