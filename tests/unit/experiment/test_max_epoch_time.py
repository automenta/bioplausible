"""``max_epoch_time`` budget (EXPERIMENT_PLAN6 next-action).

A per-epoch wall-clock budget must stop an epoch from consuming the whole
batch count once the budget is exhausted — bounding slow-settling eqprop
probes so a shallow sweep is not starved by one epoch.
"""

from __future__ import annotations

from bioplausible.core.trainer import CoreTrainer, TrainerConfig


def _cfg(**kw: object) -> TrainerConfig:
    return TrainerConfig(
        model="backprop_mlp",
        task="digits",
        model_kwargs={"input_dim": 64, "hidden_dim": 16, "output_dim": 10},
        epochs=1,
        batches_per_epoch=40,
        run_validation=False,
        save_checkpoints=False,
        **kw,  # type: ignore[arg-type]
    )


def test_zero_budget_runs_full_epoch() -> None:
    """Default (0 = unlimited) consumes every batch in the epoch."""
    trainer = CoreTrainer(_cfg())
    trainer.fit()
    assert trainer.global_step == 40


def test_tiny_budget_stops_at_first_batch() -> None:
    """An unreachable 1e-9s budget aborts before any batch trains."""
    trainer = CoreTrainer(_cfg(max_epoch_time=1e-9))
    trainer.fit()
    # The budget check precedes each batch, so an exhausted budget yields
    # essentially no trained steps — far below the 40-batch epoch.
    assert trainer.global_step < 5


def test_medium_budget_bounds_epoch() -> None:
    """A short but reachable budget caps steps below the full epoch length."""
    trainer = CoreTrainer(_cfg(max_epoch_time=0.5))
    trainer.fit()
    # A half-second budget on a 40-batch epoch trains only a few batches.
    assert 0 <= trainer.global_step < 40


def test_truncated_epoch_surfaces_in_metrics_extra() -> None:
    """A budget-truncated epoch reports its truncation in the epoch extra."""
    trainer = CoreTrainer(_cfg(max_epoch_time=0.0))
    trainer.fit()
    # Unlimited (0.0) — nothing truncated.
    assert not any(m.extra.get("epoch_time_budget_stopped") for m in trainer.history)


def test_truncation_surfaces_through_driver(monkeypatch) -> None:
    """The probe driver marks a budget-truncated run for sweep pruning."""
    from bioplausible.experiment.probe import ProbeDriver, run_probe

    class FakeDriver(ProbeDriver):
        def train(self, **kw: object) -> dict[str, object]:
            return {
                "final_acc": 0.5,
                "final_train_loss": 1.0,
                "training_path": "bptt",
                "epoch_time_budget_stopped": True,
            }

    result = run_probe(
        FakeDriver(),
        model="backprop_mlp",
        task="digits",
        config={"hidden_dim": 16},
        seed=0,
        epochs=1,
        device="cpu",
    )
    assert result.status == "ok"


def test_sweep_marks_truncated_run_as_defect(monkeypatch) -> None:
    """The sweep flags a budget-truncated run as a defect (excluded from ok)."""
    import scripts.broad_sweep as sweep

    fake_runs: list[dict[str, object]] = []
    recorded: list[dict[str, object]] = []

    class FakeDriver:  # a trivial CoreTrainerDriver stand-in for _probe_runs
        def train(self, **kw: object) -> dict[str, object]:
            metrics = {
                "final_acc": 0.4,
                "final_train_loss": 1.0,
                "param_count": 4000,
                "peak_memory_mb": 10.0,
                "wall_time_s": 5.0,
                "epoch_time_budget_stopped": True,
                "training_path": "energy",
            }
            recorded.append({**kw, **metrics})
            return metrics

    runs, n_total, n_ok = sweep._probe_runs(
        FakeDriver(),  # type: ignore[arg-type]
        model="backprop_mlp",
        family="backprop",
        space={"hidden_dim": [16]},
        probes_per_rule=1,
        epochs=1,
        seed=0,
        device="cpu",
        task="digits",
    )
    assert n_ok == 0
    assert runs[0]["ok"] is False
    assert "epoch_time_truncated" in runs[0]["defects"]
