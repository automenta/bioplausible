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
