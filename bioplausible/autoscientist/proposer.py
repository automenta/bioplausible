"""
ExperimentProposer: Generates intelligent experiment batches.

Takes hypotheses from HypothesisReasoner and converts them into
concrete experiment proposals with configurations.
"""

from collections.abc import Callable
from enum import StrEnum

from bioplausible.autoscientist.bridge import AutoScientistBridge, ExperimentProposal
from bioplausible.autoscientist.reasoner import Hypothesis, HypothesisReasoner
from bioplausible.core.logging import get_logger
from bioplausible.core.registry import (
    ComponentCategory,
    ComponentMetadata,
    ComputeProfile,
    Domain,
    LocalityLevel,
    Registry,
)
from bioplausible.knowledge import KnowledgeBase

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


# Proxies for ranking candidates on non-accuracy axes. These are *declared*
# approximations over the registry's static metadata (there is no runtime cost
# signal at proposal time), documented so the bias is auditable, not implicit.
_MEMORY_ORDER: dict[str, int] = {
    "O(1)": 0,
    "O(log N)": 1,
    "O(N)": 2,
    "O(N log N)": 3,
    "O(N^2)": 4,
}
_ROBUST_PROFILES: frozenset[ComputeProfile] = frozenset({
    ComputeProfile.ANALOG,
    ComputeProfile.NEUROMORPHIC,
    ComputeProfile.MEMRISTOR,
})
_ROBUST_LOCALITY: frozenset[LocalityLevel] = frozenset({
    LocalityLevel.LOCAL,
    LocalityLevel.FORWARD_ONLY,
})


def _objective_rank(
    objective: ProposalObjective, meta: ComponentMetadata
) -> tuple[float, ...]:
    """Return a sort key ranking a candidate on a non-accuracy objective.

    Args:
        objective: The active proposal objective.
        meta: The candidate's registry metadata.

    Returns:
        A tuple ordering candidates from best to worst on the objective; the
        ACCURACY objective returns an empty key (no bias, original ordering).
    """
    match objective:
        case ProposalObjective.ACCURACY:
            return ()
        case ProposalObjective.MEMORY:
            # Fewer resources (as declared by memory_complexity) rank first.
            return (_MEMORY_ORDER.get(meta.memory_complexity, 5),)
        case ProposalObjective.SETTLING_SPEED:
            # Proxy: lower-complexity rules settle faster than equilibrium ones.
            return (_MEMORY_ORDER.get(meta.memory_complexity, 5),)
        case ProposalObjective.NOISE_ROBUSTNESS:
            # Prefer analog/neuromorphic substrates and local credit assignment,
            # then cheaper memory within the robust set.
            robust = (
                meta.compute_profile in _ROBUST_PROFILES
                or meta.locality_level in _ROBUST_LOCALITY
            )
            return (
                0.0 if robust else 1.0,
                -_MEMORY_ORDER.get(meta.memory_complexity, 5),
            )


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
        domain: str | None = None,
        n_proposals: int = 10,
        min_bio_score: float = 0.0,
        objective: ProposalObjective | str = ProposalObjective.ACCURACY,
    ) -> list[ExperimentProposal]:
        """
        Propose a batch of experiments.

        Combines systematic exploration with hypothesis-driven targeting.

        Args:
            domain: Optional domain filter.
            n_proposals: Number of proposals to generate.
            min_bio_score: Minimum bio-plausibility score.
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
        for h in hypotheses[: n_proposals // 2]:
            proposal = self._hypothesis_to_proposal(h, objective_enum)
            if proposal:
                proposals.append(proposal)

        # 3. Fill remaining slots with systematic combinations
        remaining = n_proposals - len(proposals)
        if remaining > 0:
            systematic = self._systematic_proposals(
                domain, remaining, min_bio_score, objective_enum
            )
            proposals.extend(systematic)

        h_count = len(hypotheses)
        s_count = len(proposals) - h_count
        logger.info(
            "Proposed %d experiments (%d hypothesis-driven, %d systematic) "
            "objective=%s",
            len(proposals),
            h_count,
            s_count,
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

    def _systematic_proposals(
        self,
        domain: str | None = None,
        n: int = 5,
        min_bio_score: float = 0.0,
        objective: ProposalObjective = ProposalObjective.ACCURACY,
    ) -> list[ExperimentProposal]:
        """Generate systematic exploration proposals."""
        # Get models and propagators
        models = Registry.query(
            category=ComponentCategory.MODEL,
            min_bio_score=min_bio_score,
        )
        propagators = Registry.query(
            category=ComponentCategory.CREDIT_ASSIGNMENT,
            min_bio_score=min_bio_score,
        )

        # Bias audit (plan §5 cycle 2): when optimizing a non-accuracy axis,
        # rank candidates by that axis instead of the registry's default order.
        if objective is not ProposalObjective.ACCURACY:
            models = sorted(
                models,
                key=lambda m: _objective_rank(objective, m["metadata"]),
            )

        proposals = []
        for i in range(min(n, len(models) * len(propagators))):
            m_idx = i % len(models)
            p_idx = (i // len(models)) % len(propagators) if propagators else 0
            model = models[m_idx]
            propagator = propagators[p_idx] if propagators else None

            bias = (
                f" targeting {objective.value}"
                if objective is not ProposalObjective.ACCURACY
                else ""
            )
            proposals.append(
                ExperimentProposal(
                    hypothesis=(
                        "Systematic exploration of model-propagator combinations"
                    ),
                    model=model["name"],
                    task="mnist",
                    propagator=propagator["name"] if propagator else None,
                    justification=(
                        f"Testing {model['name']} with "
                        f"{propagator['name'] if propagator else 'default'}: "
                        f"bio_score={model['metadata'].bio_plausibility_score}"
                        f"{bias}"
                    ),
                    priority=0.3,
                    tags=[
                        "systematic",
                        "exploration",
                        f"objective:{objective.value}",
                    ],
                )
            )

        return proposals

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

    def propose_hypercube_ablation(
        self,
        fixed: dict[str, str | list[str]],
        sweep: str,
        sweep_values: list[str],
        domain: str | None = None,
        n_proposals: int = 10,
        min_bio_score: float = 0.0,
        objective: ProposalObjective | str = ProposalObjective.ACCURACY,
    ) -> list[ExperimentProposal]:
        """Propose experiments via hypercube ablation along the 5-D ontology axes.

        This enables the AutoScientist to perform rigorous ablation studies by
        holding some ontology layers constant and sweeping others.

        Args:
            fixed: Dictionary of layer -> value(s) to hold constant.
                Keys: "substrate", "geometry", "dynamics", "credit", "update"
                Values: single value or list of values
            sweep: Layer to sweep over ("substrate", "geometry", "dynamics", "credit", "update")
            sweep_values: Values to sweep for the sweep layer
            domain: Optional domain filter
            n_proposals: Number of proposals to generate
            min_bio_score: Minimum bio-plausibility score
            objective: The axis to optimize when ranking candidates

        Returns:
            List of experiment proposals for the hypercube ablation.
        """
        # Query the registry along the 5-D ontology axes
        results = Registry.query_ontology(
            fixed=fixed,
            sweep=sweep,
            sweep_values=sweep_values,
            domain=Domain(domain) if domain else None,
            min_bio_score=min_bio_score,
        )

        # Convert results to proposals
        proposals = []
        for i, r in enumerate(results[:n_proposals]):
            meta = r["metadata"]
            layers = r.get("ontology_layers", {})

            # Build hypothesis from the ablation
            fixed_str = ", ".join(f"{k}={v}" for k, v in fixed.items())
            sweep_str = (
                f"{sweep}={sweep_values}"
                if isinstance(sweep_values, list)
                else f"{sweep}={sweep_values}"
            )
            hypothesis = f"Hypercube ablation: fixed [{fixed_str}], sweep [{sweep_str}]"

            proposals.append(
                ExperimentProposal(
                    hypothesis=hypothesis,
                    model=meta.name,
                    task="mnist",
                    propagator=None,  # Will use model's default
                    justification=(
                        f"5-D ontology ablation: {hypothesis}. "
                        f"Layers: {layers}. Bio-score: {meta.bio_plausibility_score}"
                    ),
                    priority=0.5,
                    tags=[
                        "hypercube_ablation",
                        f"sweep:{sweep}",
                        f"fixed:{list(fixed.keys())}",
                        f"objective:{objective.value if isinstance(objective, ProposalObjective) else objective}",
                    ],
                )
            )

        logger.info(
            "Proposed %d hypercube ablation experiments (fixed=%s, sweep=%s=%s)",
            len(proposals),
            list(fixed.keys()),
            sweep,
            sweep_values,
        )
        return proposals
