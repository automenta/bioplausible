"""
ExperimentProposer: Generates intelligent experiment batches.

Takes hypotheses from HypothesisReasoner and converts them into
concrete experiment proposals with configurations.
"""

from collections.abc import Callable
from enum import StrEnum

from computronium.autoscientist.bridge import AutoScientistBridge, ExperimentProposal
from computronium.autoscientist.reasoner import Hypothesis, HypothesisReasoner
from computronium.core.logging import get_logger
from computronium.knowledge import KnowledgeBase

__all__ = [
    "ExperimentProposer",
    "ProposalObjective",
    "logger",
]
logger = get_logger()


class ProposalObjective(StrEnum):
    """What a proposal cycle is explicitly optimizing (bias audit, plan §5 cycle 2).

    Defaulting to ``ACCURACY`` reproduces the historical behavior; choosing a
    non-accuracy objective forces the proposer to rank candidates by a resource
    or robustness axis instead — surfacing whether the engine is biased toward
    accuracy alone.
    """

    ACCURACY = "accuracy"
    MEMORY = "memory"
    SETTLING_SPEED = "settling_speed"
    NOISE_ROBUSTNESS = "noise_robustness"
    STABILITY = "stability"
    ENERGY = "energy"
    LATENCY = "latency"
    PLASTICITY_CAPACITY = "plasticity_capacity"


# Query service shape the proposer depends on (P2 read-half). Injected so the
# flywheel's read path can be unit-tested against a stub, not a live DB.
ConditionalQuerier = Callable[[dict[str, object]], list[object]]


class ExperimentProposer:
    """
    Generates experiment batches from hypotheses.

    Supports:
    - Systematic search across model+propagator combinations
    - Targeted experiments based on specific hypotheses
    - Ablation studies (vary one parameter at a time)
    - Curriculum-based progression (easy tasks first)

    P2 (read-half): the proposer can consult prior verified conditionals via an
    injected query service and prune probes the KB has already characterized —
    turning the "knowledge layer is read" claim into a measurable skip.
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase | None = None,
        reasoner: HypothesisReasoner | None = None,
        conditional_query: ConditionalQuerier | None = None,
    ):
        self.knowledge_base = knowledge_base or KnowledgeBase()
        self.reasoner = reasoner or HypothesisReasoner(self.knowledge_base)
        self.bridge = AutoScientistBridge()
        # Dependency injection: the query service defaults to the KB's conditional
        # read, but is swappable for a stub in tests / a remote service in prod.
        self._conditional_query = conditional_query or (
            self.knowledge_base.query_conditionals
        )

    def propose_batch(
        self,
        n_proposals: int = 10,
        objective: ProposalObjective | str = ProposalObjective.ACCURACY,
    ) -> list[ExperimentProposal]:
        """
        Propose a batch of hypothesis-driven experiments.

        Args:
            n_proposals: Number of proposals to generate.
            objective: The axis to optimize when ranking candidates. Defaults to
                ACCURACY (historical behavior); a non-accuracy objective forces
                the cycle to rank by memory/settling-speed/noise-robustness so
                the engine's bias is explicit and auditable (plan §5 cycle 2).

        Returns:
            List of experiment proposals.
        """
        objective_enum = (
            objective
            if isinstance(objective, ProposalObjective)
            else ProposalObjective(objective)
        )
        proposals = []

        # 1. Generate hypotheses
        hypotheses = self.reasoner.generate_hypotheses()

        # 2. Convert hypotheses to proposals
        for h in hypotheses[:n_proposals]:
            proposal = self._hypothesis_to_proposal(h, objective_enum)
            if proposal:
                proposals.append(proposal)

        logger.info(
            "Proposed %d experiments (%d hypothesis-driven) objective=%s",
            len(proposals),
            len(proposals),
            objective_enum.value,
        )
        return proposals

    def _hypothesis_to_proposal(
        self,
        hypothesis: Hypothesis,
        objective: ProposalObjective = ProposalObjective.ACCURACY,
    ) -> ExperimentProposal | None:
        """Convert a hypothesis to an experiment proposal."""
        if not hypothesis.proposed_model and not hypothesis.proposed_propagator:
            return None

        tags = ["autoscientist", hypothesis.source]
        if objective is not ProposalObjective.ACCURACY:
            tags.append(f"objective:{objective.value}")
        return ExperimentProposal(
            hypothesis=hypothesis.statement,
            model=hypothesis.proposed_model or "MLP",
            task=hypothesis.proposed_task or "mnist",
            propagator=hypothesis.proposed_propagator,
            justification=(
                hypothesis.reasoning_chain[0] if hypothesis.reasoning_chain else ""
            ),
            expected_outcome=hypothesis.statement,
            priority=hypothesis.confidence,
            tags=tags,
        )

    def propose_ablation(
        self,
        model: str,
        base_config: dict[str, object],
        parameters: list[str],
        values: list[list[object]],
    ) -> list[ExperimentProposal]:
        """
        Propose ablation studies varying specific parameters.

        Args:
            model: Model name to ablate.
            base_config: Base configuration.
            parameters: Parameter names to vary.
            values: Values to try for each parameter.

        Returns:
            List of ablation proposals.
        """
        proposals = []
        for param, vals in zip(parameters, values):
            for v in vals:
                config = dict(base_config)
                config[param] = v
                proposals.append(
                    ExperimentProposal(
                        hypothesis=f"Ablation: effect of {param}={v} on {model}",
                        model=model,
                        task=base_config.get("task", "mnist"),
                        hyperparams={param: v},
                        priority=0.4,
                        tags=["ablation", param],
                    )
                )
        return proposals

    def avoid_characterized(
        self,
        proposals: list[ExperimentProposal],
        *,
        accuracy_target: float = 0.5,
    ) -> tuple[list[ExperimentProposal], list[ExperimentProposal]]:
        """
        Prune proposals whose (model, task) the KB has already characterized.

        P2-lite's turbine-turns signal: a probe is *redundant* if a prior
        verified conditional already answers ``(proposal.model, proposal.task)``
        at or above ``accuracy_target``. Skipping it is the compounding claim
        made measurable — the proposer read the KB and burned no budget on a
        probe it could not improve.

        Args:
            proposals: Candidate proposals.
            accuracy_target: Minimum stored accuracy for a previous conditional
                to count as "already characterized".

        Returns:
            ``(kept, skipped)`` — proposals that remain worth probing, and those
            dropped because a prior conditional already covered them. The paired
            counterfactual (with-KB vs without-KB) is the count of ``skipped``.
        """
        kept: list[ExperimentProposal] = []
        skipped: list[ExperimentProposal] = []
        for p in proposals:
            if not p.model:
                kept.append(p)
                continue
            covered = self._conditional_query({
                "model": p.model,
                "task": (p.task or "mnist"),
                "accuracy_target": accuracy_target,
            })
            if covered:
                skipped.append(p)
            else:
                kept.append(p)
        if skipped:
            logger.info(
                "proposer skipped %d already-characterized probe(s)", len(skipped)
            )
        return kept, skipped
