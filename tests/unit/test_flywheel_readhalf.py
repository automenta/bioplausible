"""P2 + P3a — the flywheel read-half, its paired counterfactual demo (R4), and
the flagship-selection query.

The P2-lite "turbine turns" signal is the proposer reading a prior conditional
and skipping an already-characterized probe — measured as a paired counterfactual
(proposer-with-KB vs proposer-without-KB), not an anecdote.
"""

import tempfile

import pytest

from bioplausible.autoscientist.bridge import ExperimentProposal
from bioplausible.autoscientist.proposer import ExperimentProposer
from bioplausible.core.exceptions import ConditionalQueryError
from bioplausible.experiment.result_sink import configure as sink_configure
from bioplausible.experiment.result_sink import record_experiment_result
from bioplausible.knowledge.kb import KnowledgeBase


@pytest.fixture
def sink_paths():
    with tempfile.TemporaryDirectory() as tmpdir:
        sink_configure(kb_path=f"{tmpdir}/kb.db", failure_path=f"{tmpdir}/fail.db")
        yield tmpdir
        sink_configure()


def _proposal(model: str, task: str = "mnist") -> ExperimentProposal:
    return ExperimentProposal(
        hypothesis=f"Characterize {model} on {task}",
        model=model,
        task=task,
        priority=0.5,
    )


def _kb(sink_paths: str) -> KnowledgeBase:
    return KnowledgeBase(db_path=f"{sink_paths}/kb.db", auto_embed=False)


def test_conditional_query_returns_verified_conditionals(sink_paths):
    """P2 read-half: a prior verified result is queryable by filter."""
    record_experiment_result(
        model="neural_cube",
        task="mnist",
        config={"cube_size": 4},
        metrics={
            "final_acc": 0.94,
            "forward_flops": 300,
            "backward_flops": 100,
            "peak_memory_mb": 40,
            "wall_time_s": 2,
        },
        status="completed",
    )
    kb = _kb(sink_paths)
    res = kb.query_conditionals({
        "model": "neural_cube",
        "task": "mnist",
        "accuracy_target": 0.9,
    })
    assert len(res) == 1
    assert res[0].accuracy == pytest.approx(0.94)
    assert res[0].task == "mnist"


def test_conditional_query_respects_caps(sink_paths):
    """Caps exclude conditionals that exceed the budget."""
    record_experiment_result(
        model="neural_cube",
        task="mnist",
        config={"cube_size": 4},
        metrics={
            "final_acc": 0.94,
            "forward_flops": 300,
            "backward_flops": 100,
            "peak_memory_mb": 40,
            "wall_time_s": 2,
        },
        status="completed",
    )
    kb = _kb(sink_paths)
    assert not kb.query_conditionals({
        "model": "neural_cube",
        "task": "mnist",
        "memory_cap": 10.0,
    })
    assert kb.query_conditionals({
        "model": "neural_cube",
        "task": "mnist",
        "memory_cap": 100.0,
    })


def test_conditional_query_validates_input(sink_paths):
    """A malformed query raises the domain chained error, not a raw Pydantic error."""
    kb = _kb(sink_paths)
    with pytest.raises(ConditionalQueryError):
        kb.query_conditionals({"accuracy_target": 2.0})  # out of [0,1]


def test_turbine_turns_skip_based_on_conditional(sink_paths):
    """R4 paired counterfactual: with a prior conditional the proposer skips.

    The proposer-with-KB prunes an already-characterized probe; the
    proposer-without-KB (empty KB) runs it. ``redundant-probes-avoided`` is the
    measured flywheel signal.
    """
    record_experiment_result(
        model="neural_cube",
        task="mnist",
        config={"cube_size": 4},
        metrics={"final_acc": 0.94},
        status="completed",
    )
    candidates = [_proposal("neural_cube"), _proposal("pepita")]

    with_kb = ExperimentProposer(knowledge_base=_kb(sink_paths))
    kept, skipped = with_kb.avoid_characterized(candidates, accuracy_target=0.9)
    # neural_cube is already characterized; pepita is not.
    assert [p.model for p in skipped] == ["neural_cube"]
    assert [p.model for p in kept] == ["pepita"]
    redundant_avoided = len(skipped)

    # Counterfactual: a fresh (empty) KB knows nothing → nothing is skipped.
    empty_kb = KnowledgeBase(db_path=f"{sink_paths}/empty.db", auto_embed=False)
    without_kb = ExperimentProposer(knowledge_base=empty_kb)
    kept0, skipped0 = without_kb.avoid_characterized(candidates, accuracy_target=0.9)
    assert skipped0 == []
    assert len(kept0) == len(candidates)

    assert redundant_avoided > len(skipped0)


def test_conditional_query_injected_stub(sink_paths):
    """DI: the query service is injectable for deterministic tests."""

    def stub(query):
        return ["covered"] if query.get("model") == "neural_cube" else []

    proposer = ExperimentProposer(
        knowledge_base=_kb(sink_paths), conditional_query=stub
    )
    kept, skipped = proposer.avoid_characterized(
        [_proposal("neural_cube"), _proposal("pepita")], accuracy_target=0.5
    )
    assert [p.model for p in skipped] == ["neural_cube"]
    assert [p.model for p in kept] == ["pepita"]


def test_flagship_selection_queries_kb(sink_paths):
    """P3a: flagship is selected by a KB query, not a judgment call."""
    from bioplausible.hyperopt.search_space import emit_rule_space_surfaces

    # Honest surfaces + empirical conditionals for two validated families.
    kb = _kb(sink_paths)
    emit_rule_space_surfaces(kb)
    record_experiment_result(
        model="backprop_mlp",
        task="mnist",
        config={"hidden_dim": 64},
        metrics={
            "final_acc": 0.98,
            "forward_flops": 1000,
            "backward_flops": 500,
            "peak_memory_mb": 100,
            "wall_time_s": 5,
        },
        status="completed",
    )
    record_experiment_result(
        model="neural_cube",
        task="mnist",
        config={"cube_size": 4},
        metrics={
            "final_acc": 0.94,
            "forward_flops": 300,
            "backward_flops": 100,
            "peak_memory_mb": 40,
            "wall_time_s": 2,
        },
        status="completed",
    )
    decision = kb.select_flagship(task="mnist")
    assert decision.chosen == "neural_cube"
    assert len(decision.ranked) == 1
    assert decision.ranked[0].cost_of_plausibility < 1.0
