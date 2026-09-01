"""Order-robustness redesign of the Z3 benchmark (v4 registration riders).

Covers the registered design elements:
- per-task Adam rebuild across the switching stream (starvation fix);
- adaptation gate-entropy exploration floor;
- per-step gate-history recording across every adaptation arm.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from computronium.experiments.joint.z3_fixed_weights import (
    MetaRecipe,
    Z3Model,
    _adaptation_objective,
    _new_gate_history,
    _record_gate_step,
    _run_adaptation,
    evaluate_z3,
)

COORDINATE = (
    "digital/recurrent/energy_minimization/rule_state/thermodynamic_contrast/euclidean"
)
_NUM_OPERATORS = 8
_ADAPT_BETA = 0.1
_GATE_SUM_TOL = 1e-4
_SMOKE_EPOCHS = 3


@pytest.fixture(scope="module")
def shape():
    from computronium.experiments.joint.z3_fixed_weights import TaskShape

    return TaskShape(
        batch_size=64, seq_len=10, input_dim=32, device=torch.device("cpu")
    )


def test_adaptation_entropy_floor_changes_objective() -> None:
    torch.manual_seed(0)
    beta = _ADAPT_BETA
    model = Z3Model(num_operators=_NUM_OPERATORS, operator_dim=16)
    x = torch.randn(4, 5, 16)
    y = torch.randint(0, 2, (4,))

    model(x)  # populate gates
    loss = torch.nn.functional.cross_entropy(model(x), y)
    gated = _adaptation_objective(loss, model, beta)
    ungated = _adaptation_objective(loss, model, 0.0)

    assert torch.equal(ungated, loss)
    assert float(gated) <= float(loss) + 1e-7  # entropy term subtracts
    assert not torch.isclose(gated, loss)  # non-degenerate gates ⇒ strict drop


def test_run_adaptation_records_gate_history(shape) -> None:
    from computronium.experiments.joint.z3_fixed_weights import create_parity_task

    torch.manual_seed(0)
    model = Z3Model(num_operators=_NUM_OPERATORS, operator_dim=32)
    probe_batches = 2
    probe_samples = [shape.sample(create_parity_task) for _ in range(probe_batches)]
    probe = (
        torch.cat([x for x, _ in probe_samples]),
        torch.cat([y for _, y in probe_samples]),
    )
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    epochs = 5
    record = _run_adaptation(
        model,
        optimizer,
        nn.CrossEntropyLoss(),
        shape,
        create_parity_task,
        epochs=epochs,
        probe=probe,
        adapt_entropy_beta=_ADAPT_BETA,
    )

    assert len(record.losses) == len(record.accuracy_curve) == epochs
    assert len(record.gate_history["entropy"]) == epochs
    assert all(len(g) == _NUM_OPERATORS for g in record.gate_history["mean_gates"])
    assert all(
        abs(sum(f) - 1.0) < _GATE_SUM_TOL
        for f in record.gate_history["hard_op_fraction"]
    )
    assert all(
        math.isfinite(e[0]) and e[0] >= 0 for e in record.gate_history["entropy"]
    )


def test_gate_history_helper_handles_missing_gates() -> None:
    model = Z3Model(num_operators=_NUM_OPERATORS, operator_dim=16)
    history = _new_gate_history()
    _record_gate_step(history, model)  # no forward yet → empty record appended
    assert history["entropy"] == [[]]


@pytest.mark.slow
def test_evaluate_z3_persists_gate_histories_all_arms() -> None:
    result = evaluate_z3(
        COORDINATE,
        meta_train_epochs=_SMOKE_EPOCHS,
        eval_epochs_per_task=_SMOKE_EPOCHS,
        batch_size=32,
        seq_len=5,
        input_dim=16,
        device="cpu",
        seed=42,
        recipe=MetaRecipe(
            warmup_fraction=0.6,
            entropy_beta=0.2,
            episode_len=4,
            adapt_temp=2.0,
            adapt_entropy_beta=_ADAPT_BETA,
        ),
        task_order=("threshold", "last_symbol", "parity"),
    )

    assert result["theta_invariant"] is True
    assert result["meta_recipe"]["adapt_entropy_beta"] == pytest.approx(_ADAPT_BETA)
    for row in result["tasks"].values():
        assert set(row["gate_history"]) == {"mean_gates", "hard_op_fraction", "entropy"}
        assert len(row["gate_history"]["mean_gates"]) == _SMOKE_EPOCHS
    random_rows = result["baselines"]["random_psi"]["tasks"]
    assert all("gate_history" in r for r in random_rows.values())
    finetune_stages = result["baselines"]["finetune_forgetting"]["gate_histories"]
    assert sorted(finetune_stages) == ["last_symbol", "parity", "threshold"]
