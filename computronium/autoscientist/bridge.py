"""
Bridge that translates AutoScientist experiment proposals into
ExperimentTask objects consumable by the Scientist execution engine.

AutoScientist proposes experiments; the Bridge packages them as
``propsal_to_task`` configs that the Scientist can execute.
"""

from dataclasses import dataclass, field

from computronium.core.logging import get_logger
from computronium.core.registry import ComponentCategory, Domain, Registry

__all__ = [
    "AutoScientistBridge",
    "ExperimentProposal",
    "logger",
]
logger = get_logger()


@dataclass(frozen=True, slots=True)
class ExperimentProposal:
    """A proposed experiment from AutoScientist."""

    hypothesis: str
    model: str
    task: str
    propagator: str | None = None
    optimizer: str = "adam"
    hyperparams: dict[str, object] = field(default_factory=dict)
    justification: str = ""
    expected_outcome: str = ""
    priority: float = 0.5
    tags: list[str] = field(default_factory=list)


class AutoScientistBridge:
    """
    Translates between AutoScientist proposals and Scientist execution tasks.
    """

    def __init__(self):
        self._proposals: list[ExperimentProposal] = []

    def proposal_to_task(self, proposal: ExperimentProposal) -> dict[str, object]:
        """Convert an ExperimentProposal to a config dict for CoreTrainer."""
        config = {
            "model": proposal.model,
            "task": proposal.task,
            "optimizer": proposal.optimizer,
            "tags": {
                "hypothesis": proposal.hypothesis,
                "justification": proposal.justification,
                "autoscientist_priority": proposal.priority,
                **{f"tag_{i}": t for i, t in enumerate(proposal.tags)},
            },
        }
        config.update(proposal.hyperparams)
        if proposal.propagator:
            config["propagator"] = proposal.propagator
        return config

    def discover_viable_combinations(
        self,
        domain: Domain | None = None,
        min_bio_score: float = 0.0,
    ) -> list[dict[str, object]]:
        """
        Discover all viable model+propagator+optimizer combinations.

        Used by AutoScientist to generate search space.
        """
        models = Registry.query(
            category=ComponentCategory.MODEL,
            domain=domain,
            min_bio_score=min_bio_score,
        )
        propagators = Registry.query(
            category=ComponentCategory.CREDIT_ASSIGNMENT,
            domain=domain,
            min_bio_score=min_bio_score,
        )
        optimizers = Registry.query(
            category=ComponentCategory.PARAM_UPDATE,
            domain=domain,
        )

        combinations = []
        for m in models:
            for p in propagators:
                for o in optimizers:
                    combinations.append({
                        "model": m["name"],
                        "model_meta": m["metadata"],
                        "propagator": p["name"],
                        "propagator_meta": p["metadata"],
                        "optimizer": o["name"],
                        "optimizer_meta": o["metadata"],
                    })

        logger.info(
            "Discovered %d viable combinations "
            "(models=%d, propagators=%d, optimizers=%d)",
            len(combinations),
            len(models),
            len(propagators),
            len(optimizers),
        )
        return combinations

    def submit_proposal(self, proposal: ExperimentProposal) -> None:
        """Submit a proposal for execution."""
        self._proposals.append(proposal)
        logger.info(
            f"Proposal submitted: {proposal.model}/{proposal.task} "
            f"({proposal.hypothesis[:60]})"
        )

    def pending_proposals(self) -> list[ExperimentProposal]:
        """Get all pending proposals."""
        return list(self._proposals)

    def clear_executed(self, proposal_ids: list[int]) -> None:
        """Remove executed proposals (by index)."""
        for idx in sorted(proposal_ids, reverse=True):
            if 0 <= idx < len(self._proposals):
                self._proposals.pop(idx)
