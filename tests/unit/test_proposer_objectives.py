"""ProposalObjective expansion (P5): STABILITY / ENERGY / LATENCY /
PLASTICITY_CAPACITY ranking proxies over registry metadata."""

from __future__ import annotations

from pathlib import Path

import pytest

from computronium.autoscientist.proposer import (
    ExperimentProposer,
    ProposalObjective,
    _objective_rank,
)
from computronium.core.registry import (
    ComponentCategory,
    ComponentMetadata,
    ComputeProfile,
    LocalityLevel,
)


def meta(
    profile: ComputeProfile = ComputeProfile.GPU,
    locality: LocalityLevel = LocalityLevel.GLOBAL,
    memory: str = "O(N)",
) -> ComponentMetadata:
    return ComponentMetadata(
        name="m",
        category=ComponentCategory.MODEL,
        compute_profile=profile,
        locality_level=locality,
        memory_complexity=memory,
    )


def make_proposer(tmp_path: Path) -> ExperimentProposer:
    from computronium.knowledge import KnowledgeBase

    return ExperimentProposer(knowledge_base=KnowledgeBase(tmp_path / "kb.db"))


class TestExpandedObjectives:
    def test_all_objectives_rankable(self) -> None:
        m = meta()
        for objective in ProposalObjective:
            assert isinstance(_objective_rank(objective, m), tuple)

    def test_stability_prefers_bounded_equilibrium(self) -> None:
        stable = _objective_rank(
            ProposalObjective.STABILITY,
            meta(ComputeProfile.MEMRISTOR, LocalityLevel.LOCAL),
        )
        unstable = _objective_rank(
            ProposalObjective.STABILITY, meta(ComputeProfile.GPU, LocalityLevel.GLOBAL)
        )
        assert stable < unstable

    def test_energy_prefers_event_driven_profiles(self) -> None:
        neuromorphic = _objective_rank(
            ProposalObjective.ENERGY, meta(ComputeProfile.NEUROMORPHIC)
        )
        gpu = _objective_rank(ProposalObjective.ENERGY, meta(ComputeProfile.GPU))
        assert neuromorphic < gpu

    def test_latency_prefers_forward_only(self) -> None:
        forward_only = _objective_rank(
            ProposalObjective.LATENCY, meta(locality=LocalityLevel.FORWARD_ONLY)
        )
        backprop = _objective_rank(
            ProposalObjective.LATENCY, meta(locality=LocalityLevel.GLOBAL)
        )
        assert forward_only < backprop

    def test_plasticity_capacity_inverts_memory(self) -> None:
        rich = _objective_rank(
            ProposalObjective.PLASTICITY_CAPACITY, meta(memory="O(N^2)")
        )
        poor = _objective_rank(
            ProposalObjective.PLASTICITY_CAPACITY, meta(memory="O(1)")
        )
        assert rich < poor

    @pytest.mark.parametrize(
        "objective",
        [
            ProposalObjective.STABILITY,
            ProposalObjective.ENERGY,
            ProposalObjective.LATENCY,
            ProposalObjective.PLASTICITY_CAPACITY,
        ],
    )
    def test_proposals_tag_objective(
        self, tmp_path: Path, objective: ProposalObjective
    ) -> None:
        proposer = make_proposer(tmp_path)
        (proposal,) = proposer.propose_batch(n_proposals=1, objective=objective)
        assert f"objective:{objective.value}" in proposal.tags

    def test_string_objective_accepted(self, tmp_path: Path) -> None:
        proposer = make_proposer(tmp_path)
        (proposal,) = proposer.propose_batch(n_proposals=1, objective="energy")
        assert "objective:energy" in proposal.tags
