"""Unit tests for the proposer's objective-based bias audit (plan §5 cycle 2).

Verifies the proposer can be *forced* to optimize a non-accuracy axis (memory,
settling speed, noise robustness) instead of silently defaulting to accuracy —
the measurable claim behind "is the AutoScientist over-optimizing for
accuracy?"
"""

from __future__ import annotations

from bioplausible.autoscientist.proposer import (
    ExperimentProposer,
    ProposalObjective,
    _objective_rank,
)
from bioplausible.core.registry import (
    ComponentCategory,
    ComponentMetadata,
)


def _meta(
    memory: str, *, locality: str = "global", profile: str = "gpu"
) -> ComponentMetadata:
    from bioplausible.core.registry import ComputeProfile, LocalityLevel

    return ComponentMetadata(
        name="x",
        category=ComponentCategory.MODEL,
        memory_complexity=memory,
        locality_level=LocalityLevel(locality),
        compute_profile=ComputeProfile(profile),
    )


def test_accuracy_objective_has_no_bias_key() -> None:
    assert _objective_rank(ProposalObjective.ACCURACY, _meta("O(N)")) == ()


def test_memory_objective_ranks_cheaper_memory_first() -> None:
    cheap = _objective_rank(ProposalObjective.MEMORY, _meta("O(1)"))
    expensive = _objective_rank(ProposalObjective.MEMORY, _meta("O(N)"))
    assert cheap < expensive


def test_settling_speed_uses_memory_proxy() -> None:
    assert _objective_rank(
        ProposalObjective.SETTLING_SPEED, _meta("O(1)")
    ) < _objective_rank(ProposalObjective.SETTLING_SPEED, _meta("O(N^2)"))


def test_noise_robustness_prefers_analog_local() -> None:
    robust = _objective_rank(
        ProposalObjective.NOISE_ROBUSTNESS,
        _meta("O(N)", locality="local", profile="analog"),
    )
    fragile = _objective_rank(
        ProposalObjective.NOISE_ROBUSTNESS, _meta("O(N)", profile="gpu")
    )
    assert robust < fragile


def test_propose_batch_tags_memory_objective(tmp_path) -> None:
    from bioplausible.knowledge import KnowledgeBase

    kb = KnowledgeBase(db_path=tmp_path / "kb.db")
    proposer = ExperimentProposer(knowledge_base=kb)
    proposals = proposer.propose_batch(
        n_proposals=8, objective=ProposalObjective.MEMORY
    )
    assert proposals
    obj_tags = {
        tag for p in proposals for tag in p.tags if tag.startswith("objective:")
    }
    assert obj_tags == {"objective:memory"}


def test_propose_batch_accepts_string_objective(tmp_path) -> None:
    from bioplausible.knowledge import KnowledgeBase

    kb = KnowledgeBase(db_path=tmp_path / "kb.db")
    proposer = ExperimentProposer(knowledge_base=kb)
    proposals = proposer.propose_batch(n_proposals=4, objective="memory")
    assert all("objective:memory" in p.tags for p in proposals)
