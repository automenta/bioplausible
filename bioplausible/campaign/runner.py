"""Campaign orchestration: resolve, gate, sample, and log (FIX2a §2, §8).

A :class:`CampaignRunner` turns a validated :class:`Campaign` into an ordered
execution plan: resolve arm models, expose the ``--dry-run`` view, and drive
the TIER 0 / 0.5 gates that triage broken models before any parity compute is
spent. Gate outcomes are collected into a :class:`CampaignRunResult` ready for
the JSONL logger and markdown reporter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from bioplausible.campaign import tiers

if TYPE_CHECKING:
    from bioplausible.campaign.schema import Campaign

logger = logging.getLogger(__name__)

__all__ = [
    "ArmPlan",
    "CampaignResult",
    "CampaignRunner",
    "run_gates",
]


@dataclass(frozen=True, slots=True)
class ArmPlan:
    """A resolved arm: its models, input/output geometry, and budget."""

    name: str
    models: tuple[str, ...]
    input_dim: int
    output_dim: int
    max_params: int
    flatten: bool
    protocol: str


@dataclass(frozen=True, slots=True)
class CampaignResult:
    """Aggregate result of running the gate tiers for a campaign."""

    campaign_name: str
    tiers: dict[str, list[tiers.TierOutcome]] = field(default_factory=dict)
    excluded: list[tiers.TierOutcome] = field(default_factory=list)


class CampaignRunner:
    """Resolve a validated campaign into executable plans and reports."""

    def __init__(self, campaign: Campaign, output: Path | None = None) -> None:
        self.campaign = campaign
        self.output = output or Path(campaign.output.artifacts_dir)

    def plan(self) -> list[ArmPlan]:
        """Resolve every arm of the campaign into an :class:`ArmPlan`."""
        plans: list[ArmPlan] = []
        for name, arm in self.campaign.arms.items():
            plans.append(
                ArmPlan(
                    name=name,
                    models=tuple(arm.models),
                    input_dim=self.campaign.arm_input_dim(name),
                    output_dim=self.campaign.arm_output_dim(name),
                    max_params=arm.max_params,
                    flatten=arm.flatten,
                    protocol=self.campaign.protocols.resolve(name),
                )
            )
        return plans

    def dry_run(self) -> str:
        """Render a human-readable description of what the campaign would run."""
        lines = [f"Campaign: {self.campaign.meta.name}"]
        if self.campaign.meta.description:
            lines.append(f"  {self.campaign.meta.description}")
        lines.append(f"Seed: {self.campaign.reproducibility.seed}")
        lines.append(f"Output: {self.output}")
        lines.append("")
        for arm in self.plan():
            lines.append(
                f"[{arm.name}] ({arm.protocol}, max_params={arm.max_params:,})"
            )
            for model in arm.models:
                lines.append(f"  - {model}")
        return "\n".join(lines)


def run_gates(
    campaign: Campaign,
    *,
    device: str = "cpu",
    n_seeds: int = 3,
    min_accuracy: float = 0.95,
) -> CampaignResult:
    """Run the TIER 0 then TIER 0.5 gates for every arm model.

    Models that fail TIER 0 are excluded from TIER 0.5 and every parity tier;
    models that fail TIER 0.5 are labelled ``digits-fail``. Per-arm budgets
    are enforced at sampling time, not here (that is the HPO tier's job).

    Args:
        campaign: Validated campaign definition.
        device: Training device for the gate runs.
        n_seeds: Number of digits seeds per model (TIER 0.5).
        min_accuracy: TIER 0.5 accuracy gate (default 95%).
    """
    result = CampaignResult(campaign_name=campaign.meta.name)

    for arm in CampaignRunner(campaign).plan():
        plan0 = tiers.run_tier0(
            list(arm.models),
            tiers.GateSettings(
                input_dim=arm.input_dim,
                output_dim=arm.output_dim,
                device=device,
                seed=campaign.reproducibility.seed,
            ),
            config={"hidden_dim": 32, "num_layers": 2},
        )
        result.tiers.setdefault("tier0", []).extend(plan0)

        survivors = [o.model for o in plan0 if o.passed]
        plan05 = tiers.run_tier05(
            survivors,
            tiers.GateSettings(
                input_dim=arm.input_dim,
                output_dim=arm.output_dim,
                device=device,
                seed=campaign.reproducibility.seed,
                epochs=tiers.TIER05_EPOCHS,
                n_seeds=n_seeds,
                min_accuracy=min_accuracy,
            ),
            config={"hidden_dim": 64, "num_layers": 1},
        )
        result.tiers.setdefault("tier0.5", []).extend(plan05)
        result.excluded.extend(o for o in plan05 if not o.passed)
    return result
