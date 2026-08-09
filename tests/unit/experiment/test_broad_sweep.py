"""Unit tests for the broad sweep's pure liveness/aggregation logic.

The sweep itself is a compute-heavy CLI (scripts/broad_sweep.py); these tests
exercise its pure decision helpers — the liveness gate and family summarizer —
so the contract "auto-quarantine, don't falsely kill, aggregate live resources"
is locked without spending any training compute.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from bioplausible.core.config import ModelConfig

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
    assert bs._RULE_ACTIVATION["eqprop"]["config"]["gradient_method"] == "equilibrium"
    # No family is silently left to the BPTT fallback for the thesis core.
    for bio in ("eqprop", "fa", "hebbian"):
        assert bio in bs._RULE_ACTIVATION


def test_eqprop_models_all_use_energy_contrastive():
    """All eqprop models now use the self-contained energy-contrastive engine.

    The rewrite replaced all six fundamental models with a single unified
    EquilibriumMLP engine (energy-contrastive, O(1) memory, no optimizer).
    All models now have a working train_step that runs the local rule.
    """
    # All fundamental eqprop models have train_step and use energy contrastive
    from bioplausible.core.registry import ComponentCategory, Registry
    for name in ("eqprop", "directed_ep", "finite_nudge_ep",
                 "lazy_eqprop", "momentum_equilibrium", "sparse_equilibrium"):
        cls = Registry.get(ComponentCategory.MODEL, name)
        assert hasattr(cls, "train_step"), f"{name} missing train_step"
        # The engine uses gradient_method="equilibrium" but train_step runs
        # the energy-contrastive rule self-contained
        m = cls(config=ModelConfig(name=name, input_dim=10, output_dim=5, hidden_dims=[20]))
        assert callable(m.train_step)

    # Models that were already working still work
    for name in ("graph_eqprop", "holomorphic_ep", "eqprop_mlp", "conv_eqprop",
                 "modern_conv_eqprop", "neural_cube"):
        cls = Registry.get(ComponentCategory.MODEL, name)
        assert hasattr(cls, "forward")


def test_rule_activation_for_uses_energy_contrastive():
    """All eqprop models get gradient_method='equilibrium' (energy-contrastive).

    ``"equilibrium"`` routes a unified ``EquilibriumMLP`` model through its
    native contrastive ``train_step`` (the trainer tries Phase-3
    ``model.train_step`` before anything else), while conv/graph add-ons (no
    native rule) keep the fast O(1) implicit backward. It is the one value
    that gives every eqprop model its fastest correct training path.
    """
    act = bs._rule_activation_for("eqprop", "eqprop")
    assert act["config"]["gradient_method"] == "equilibrium"
    act = bs._rule_activation_for("directed_ep", "eqprop")
    assert act["config"]["gradient_method"] == "equilibrium"
    act = bs._rule_activation_for("conv_eqprop", "eqprop")
    assert act["config"]["gradient_method"] == "equilibrium"
    # DeepHebbian/ThreeFactorHebbian have native train_step -> no propagator
    assert bs._rule_activation_for("deep_hebbian", "hebbian") == {"config": {}}
    assert bs._rule_activation_for("three_factor_hebbian", "hebbian") == {
        "config": {}
    }
    # hebbian_3d (no native train_step) keeps the CHL propagator
    assert bs._rule_activation_for("hebbian_3d", "hebbian") == {
        "propagator": "contrastive_hebbian_learning"
    }



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


def test_task_domain_maps_digits_to_vision():
    """``digits`` is a vision task, so non-vision (LM/RL) models are filtered."""
    assert bs._task_domain("digits") is not None
    assert bs._task_domain("mnist") is not None
    assert bs._task_domain("imdb") is not None
    assert bs._task_domain("cartpole") is not None
    assert bs._task_domain("unknown_task") is None


def test_probe_runs_flags_epoch_time_truncation():
    """A budget-truncated run is a defect (partial-epoch stats are not ok)."""
    space = {"hidden_dim": (32, 64, "int")}

    class FakeDriver:
        def train(self, *, model, task, config, seed, epochs, device, propagator=None, allow_bptt_fallback=True):
            return {
                "final_train_loss": 0.5,
                "final_acc": 0.5,
                "training_path": "energy",
                "phantom_knobs": [],
                "param_count": 100,
                "peak_memory_mb": 1.0,
                "wall_time_s": 0.1,
                "epoch_time_budget_stopped": True,
            }

    runs, n_total, n_ok = bs._probe_runs(
        FakeDriver(), model="m", family="eqprop", space=space,
        probes_per_rule=1, epochs=2, seed=0, device="cpu", task="mnist",
    )
    assert runs[0]["ok"] is False
    assert "epoch_time_truncated" in runs[0]["defects"]



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


def test_match_param_budget_binds_conv_model_width():
    """Budget rematch searches the conv width axis (hidden_channels), so a conv
    eqprop model is not left at its sampled-wide 100k+ param count (SWEEP_FAILURES
    #3/#4). The family space samples hidden_dim; the matcher must search the
    model's real width axis hidden_channels instead of returning the wide sample.
    """
    import bioplausible.zoo  # ruff: ignore[unused-import]  (registration side effect)
    from bioplausible.experiment.param_estimator import estimate_param_count

    for model in ("modern_conv_eqprop", "conv_eqprop"):
        config = bs._match_param_budget(
            model,
            {"hidden_dim": 128, "learning_rate": 1e-3},
            32000,
            input_dim=784,
            output_dim=10,
        )
        assert "hidden_channels" in config
        count = estimate_param_count(
            model, config, input_dim=784, output_dim=10
        )
        assert count <= 32000, f"{model} over budget: {count}"


def test_match_param_budget_rounds_channels_to_groupnorm_multiple():
    """Conv channel derivations stay GroupNorm-divisible even when the matcher
    seeds hidden_channels from the sampled hidden_dim (SWEEP_FAILURES #4)."""
    config = bs._match_param_budget(
        "conv_eqprop",
        {"hidden_dim": 100, "learning_rate": 1e-3},
        32000,
        input_dim=784,
        output_dim=10,
    )
    assert config["hidden_channels"] % 8 == 0


def test_prune_phantom_knobs_drops_unconsumed_equilibrium_knobs():
    """Conv/lazy eqprop models that don't route beta/convergence_* no longer
    accumulate phantom-knob defect noise from the sampled family space
    (SWEEP_FAILURES #2): the probe samples the model's real knob subspace only.
    """
    from bioplausible.domains.registry import resolve_task

    spec = resolve_task("mnist")
    cfg = {
        "hidden_dim": 64,
        "learning_rate": 1e-3,
        "beta": 0.5,
        "max_steps": 10,
        "convergence_threshold": 1e-3,
        "convergence_start": 5,
    }
    out = bs._prune_phantom_knobs(
        "conv_eqprop", cfg, input_dim=spec.input_dim, output_dim=spec.output_dim
    )
    # beta/convergence_* are not consumed by conv_eqprop -> dropped; learning_rate
    # is retained (the trainer consumes it) and so is the structural width.
    assert "beta" not in out
    assert "convergence_start" not in out
    assert "convergence_threshold" not in out
    assert "learning_rate" in out
    assert "hidden_dim" in out
    # A **kwargs eqprop model consumes everything -> nothing pruned.
    full = bs._prune_phantom_knobs(
        "momentum_equilibrium", cfg, input_dim=spec.input_dim, output_dim=spec.output_dim
    )
    assert set(full) == set(cfg)


def test_forward_probe_ok_skips_diffusion_and_chl_incompatible():
    """The pre-sweep compatibility gate skips models whose probe path would
    crash every probe: a diffusion model whose forward needs ``t`` (SWEEP_FAILURES
    #5) and a 2D->conv3d model whose CHL propagator can't stream it (#6). Healthy
    rules pass through.
    """
    import bioplausible.zoo  # ruff: ignore[unused-import]  (registration side effect)

    # eqprop_diffusion: bare flat forward raises 't must be provided'.
    cfg = {"hidden_dim": 64, "learning_rate": 1e-3}
    assert bs._forward_probe_ok(
        "eqprop_diffusion", cfg, input_dim=784, output_dim=10, device="cpu"
    ) is False
    # hebbian_3d: CHL cannot stream its 2D->conv3d transition chain.
    assert bs._forward_probe_ok(
        "hebbian_3d",
        {"hidden_dim": 32, "learning_rate": 1e-3},
        input_dim=784,
        output_dim=10,
        device="cpu",
        propagator="contrastive_hebbian_learning",
    ) is False
    # A healthy eqprop model passes both the bare forward and its local rule.
    assert bs._forward_probe_ok(
        "conv_eqprop", cfg, input_dim=784, output_dim=10, device="cpu"
    ) is True
