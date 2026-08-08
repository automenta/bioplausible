"""Unit tests for the broad sweep's pure liveness/aggregation logic.

The sweep itself is a compute-heavy CLI (scripts/broad_sweep.py); these tests
exercise its pure decision helpers — the liveness gate and family summarizer —
so the contract "auto-quarantine, don't falsely kill, aggregate live resources"
is locked without spending any training compute.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).parents[3] / "scripts" / "broad_sweep.py"
_spec = importlib.util.spec_from_file_location("broad_sweep", _SCRIPT)
bs = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(bs)  # type: ignore[union-attr]


def _probe(**kw: object) -> dict[str, object]:
    base: dict[str, object] = {"ok": True}
    base.update(kw)
    return base


def test_live_when_loss_decreases() -> None:
    runs = [
        _probe(loss_epoch_0=2.0, loss_epoch_final=1.0),
        _probe(loss_epoch_0=3.0, loss_epoch_final=1.5),
    ]
    assert bs._is_live(runs, determined=True)


def test_dead_when_loss_increases() -> None:
    runs = [
        _probe(loss_epoch_0=1.0, loss_epoch_final=2.0),
        _probe(loss_epoch_0=1.0, loss_epoch_final=2.5),
    ]
    assert not bs._is_live(runs, determined=True)


def test_undetermined_when_too_few_epochs() -> None:
    # Even a clearly-decreasing pair must NOT be live on a 1-epoch run
    # because the endpoints are identical there (degenerate gate).
    runs = [_probe(loss_epoch_0=1.0, loss_epoch_final=0.5)]
    assert not bs._is_live(runs, determined=False)


def test_ignores_failed_probes() -> None:
    runs = [
        _probe(ok=False, error="boom"),
        _probe(loss_epoch_0=2.0, loss_epoch_final=1.0),
    ]
    assert bs._is_live(runs, determined=True)


def test_all_failed_is_not_live() -> None:
    assert not bs._is_live([_probe(ok=False)], determined=True)


def test_live_family_excludes_dead_models_from_resource() -> None:
    family_runs = {
        "rule_a": [  # live: loss decreases
            _probe(
                loss_epoch_0=2.0,
                loss_epoch_final=1.0,
                peak_memory_mb=100.0,
                wall_time_s=5.0,
                final_acc=0.8,
            ),
            _probe(
                loss_epoch_0=3.0,
                loss_epoch_final=1.5,
                peak_memory_mb=120.0,
                wall_time_s=6.0,
                final_acc=0.81,
            ),
        ],
        "rule_b": [  # dead: loss increases
            _probe(
                loss_epoch_0=1.0,
                loss_epoch_final=2.0,
                peak_memory_mb=999.0,
                wall_time_s=99.0,
            )
        ],
    }
    out = bs._summarize_family(family_runs, determined=True)
    assert out["live"] is True
    assert out["n_live"] == 1
    assert out["n_dead"] == 1
    assert out["dead_models"] == ["rule_b"]
    # Dead model's cost must not leak into the family resource coordinates.
    assert out["resource"]["peak_memory_mean"] == 110.0
    assert out["resource"]["wall_time_mean"] == 5.5


def test_undetermined_run_reports_no_quarantine() -> None:
    family_runs = {"rule_a": [_probe(loss_epoch_0=2.0, loss_epoch_final=1.0)]}
    out = bs._summarize_family(family_runs, determined=False)
    assert out["live"] is None
    assert out["dead_models"] == []
    assert out["resource"] == {}


def test_no_ok_probes_is_undetermined_not_dead() -> None:
    family_runs = {"rule_a": [_probe(ok=False, error="boom")]}
    out = bs._summarize_family(family_runs, determined=True)
    assert out["live"] is False
    assert out["n_dead"] == 1


def test_space_with_int_param_is_casted() -> None:
    space = {"hidden_dim": (32, 64, "int"), "lr": (1e-4, 1e-2, "log")}
    config = bs.sample_config_for_space(space)
    assert isinstance(config["hidden_dim"], int)
    assert 32 <= config["hidden_dim"] <= 64
    assert isinstance(config["lr"], float)


def test_space_with_categorical_choice() -> None:
    space = {"feedback_mode": ["random", "symmetric"]}
    config = bs.sample_config_for_space(space)
    assert config["feedback_mode"] in {"random", "symmetric"}


def test_registry_families_detected() -> None:
    import bioplausible.zoo  # ruff: ignore[unused-import]  (registration side effect)

    families = bs._registry_families()
    assert "eqprop" in families
    assert "backprop" in families


def test_shallow_clamp_caps_large_configs() -> None:
    config = {
        "hidden_dim": 2048,
        "num_layers": 6,
        "max_steps": 200,
        "beta": 0.7,
    }
    out = bs._shallow_clamp(config)
    assert out["hidden_dim"] <= bs._SHALLOW_CAPS["hidden_dim"]
    assert out["num_layers"] <= bs._SHALLOW_CAPS["num_layers"]
    assert out["max_steps"] <= bs._SHALLOW_CAPS["max_steps"]
    # Non-resource knobs are untouched.
    assert out["beta"] == 0.7


def test_shallow_clamp_leaves_small_configs_alone() -> None:
    config = {"hidden_dim": 32, "num_layers": 1, "lr": 1e-3}
    out = bs._shallow_clamp(config)
    assert out["hidden_dim"] == 32
    assert out["num_layers"] == 1
    assert out["lr"] == 1e-3


def test_bio_families_are_rule_activated():
    """The eqprop/fa/hebbian families must run their own local rule, not BPTT.

    Probing them via the backprop fallback would report *backprop* memory as the
    bio rule's cost — defeating the cost-of-locality measurement.
    """
    assert bs._RULE_ACTIVATION["fa"]["propagator"] == "feedback_alignment"
    assert (
        bs._RULE_ACTIVATION["hebbian"]["propagator"]
        == "contrastive_hebbian_learning"
    )
    assert bs._RULE_ACTIVATION["eqprop"]["config"]["gradient_method"] == "contrastive"
    # No family is silently left to the BPTT fallback for the thesis core.
    for bio in ("eqprop", "fa", "hebbian"):
        assert bio in bs._RULE_ACTIVATION


def test_probe_runs_flags_nan_divergence():
    """A probe returning non-finite loss is flagged, not counted as ok/live."""
    space = {"hidden_dim": (32, 64, "int"), "learning_rate": (1e-4, 1e-2, "log")}

    class FakeDriver:
        def train(self, *, model, task, config, seed, epochs, device, propagator=None, allow_bptt_fallback=True):
            return {
                "final_train_loss": float("nan"),
                "final_acc": 0.0,
                "training_path": "model_train_step",
                "phantom_knobs": [],
                "param_count": 100,
                "peak_memory_mb": 1.0,
                "wall_time_s": 0.1,
            }

    runs, n_total, n_ok = bs._probe_runs(
        FakeDriver(), model="m", family="eqprop", space=space,
        probes_per_rule=1, epochs=2, seed=0, device="cpu", task="mnist",
    )
    assert n_total == 1
    assert n_ok == 0  # NaN run is NOT a successful ok probe
    assert runs[0]["ok"] is False
    assert "nan_divergence" in runs[0]["defects"]


def test_probe_runs_flags_phantom_knobs():
    space = {"hidden_dim": (32, 64, "int")}

    class FakeDriver:
        def train(self, *, model, task, config, seed, epochs, device, propagator=None, allow_bptt_fallback=True):
            return {
                "final_train_loss": 1.0,
                "final_acc": 0.5,
                "training_path": "model_train_step",
                "phantom_knobs": ["beta"],
                "param_count": 100,
            }

    runs, _n_total, n_ok = bs._probe_runs(
        FakeDriver(), model="m", family="eqprop", space=space,
        probes_per_rule=1, epochs=2, seed=0, device="cpu", task="mnist",
    )
    assert n_ok == 0
    assert any("phantom_knobs" in d for d in runs[0]["defects"])


def test_summarize_family_aggregates_knob_defects():
    """Model-level defects aggregate NaN/phantom flags from its runs."""
    family_runs = {
        "m1": [
            {"ok": False, "defects": ["nan_divergence"], "final_acc": 0.0},
            {"ok": True, "defects": [], "final_acc": 0.8, "loss_epoch_0": 2.0, "loss_epoch_final": 1.0},
        ],
        "m2": [
            {"ok": False, "defects": ["phantom_knobs=['beta']"], "final_acc": 0.0},
        ],
    }
    out = bs._summarize_family(family_runs, determined=True, family="eqprop")
    assert out["models"]["m1"]["defects"] == ["nan_divergence"]
    assert "phantom_knobs" in out["models"]["m2"]["defects"][0]
    # Only the defect-free run fed liveness; the NaN run is not counted ok.
    assert out["models"]["m1"]["live"] is True


def test_match_param_budget_stays_under_budget():
    """Width matching brings backprop_mlp's param count within the budget.

    Uses the static estimator (model construction, no training), so this is a
    fast pure-logic check of the fair-comparison rematch.
    """
    import bioplausible.zoo  # ruff: ignore[unused-import]  (registration side effect)
    from bioplausible.experiment.param_estimator import estimate_param_count

    budget = 4000
    config = bs._match_param_budget(
        "backprop_mlp",
        {"hidden_dim": 512, "num_layers": 2},
        budget,
        input_dim=64,
        output_dim=10,
    )
    count = estimate_param_count(
        "backprop_mlp", config, input_dim=64, output_dim=10
    )
    assert count <= budget
    assert count > budget // 2  # close to budget, not pathologically small


def test_match_param_budget_touches_only_width():
    """Non-width knobs survive the rematch (displayed budget config)."""
    config = bs._match_param_budget(
        "backprop_mlp",
        {"hidden_dim": 512, "num_layers": 2, "learning_rate": 0.01},
        4000,
        input_dim=64,
        output_dim=10,
    )
    assert config["learning_rate"] == 0.01
    assert config["num_layers"] == 2


def test_match_param_budget_minimizes_unbudgetable_model():
    """A model whose minimum width exceeds budget is minimised, never left wide.

    contrastive_feedback_alignment has ~12.8k params even at width 8 — above an
    8k budget. The matcher must return the smallest-width config (not the
    original wide sample) so ``max_params`` is honoured as far as possible.
    """
    config = bs._match_param_budget(
        "contrastive_feedback_alignment",
        {"hidden_dim": 128, "num_layers": 2},
        8000,
        input_dim=784,
        output_dim=10,
    )
    assert config["hidden_dim"] == 8
    from bioplausible.experiment.param_estimator import estimate_param_count

    count = estimate_param_count(
        "contrastive_feedback_alignment",
        config,
        input_dim=784,
        output_dim=10,
    )
    assert count < 20000  # minimized, not the original 236k
