"""Lock-in: sampled hyper-parameters must actually reach the model.

Regression for the phantom-drift bug where ``beta``/``max_steps``/
``learning_rate`` were silently dropped into ``ModelConfig.extra`` (ignored) on
the direct-init construction path, so every eqprop probe trained with identical
defaults regardless of its sampled config.

The single construction layer (:mod:`bioplausible.core.construction`) routes
sampled knobs into real ``ModelConfig`` fields for config-accepting models,
surfaces unconsumable knobs as *phantoms*, and keeps ``model_kwargs`` free of
non-serialisable nested objects (so the ``TrainerConfig`` OmegaConf round-trip
and checkpoints never see a ``Literal``-typed dataclass field).
"""

from __future__ import annotations

import bioplausible.zoo  # ruff: ignore[unused-import]  # triggers model registration
from bioplausible.core.construction import (
    KNOBS,
    build_model_config,
    construct_model,
    model_kwargs,
    phantom_knobs,
)
from bioplausible.core.registry import ComponentCategory, Registry


def test_config_accepting_model_honors_directed_ep_knobs() -> None:
    """DirectedEP must train with the *sampled* beta/lr/max_steps, not defaults.

    Before the construction-layer fix every DirectedEP probe used the default
    lr=1e-3/beta=0.2 and returned identical loss — the bug this locks in.
    """
    cls = Registry.get(ComponentCategory.MODEL, "directed_ep")
    model = construct_model(
        cls,
        {"hidden_dim": 64, "num_layers": 2, "learning_rate": 0.05, "beta": 0.5, "max_steps": 20},
        input_dim=784,
        output_dim=10,
        model_name="directed_ep",
    )
    assert model.config.learning_rate == 0.05
    assert model.config.beta == 0.5
    assert model.config.max_steps == 20
    assert model.config.hidden_dims == [64, 64]


def test_config_accepting_model_honors_learning_rate_field() -> None:
    cls = Registry.get(ComponentCategory.MODEL, "eqprop")
    model = construct_model(
        cls,
        {"hidden_dim": 32, "num_layers": 2, "learning_rate": 0.02, "beta": 1.3, "max_steps": 7},
        input_dim=784,
        output_dim=10,
        model_name="eqprop",
    )
    assert model.config.learning_rate == 0.02
    assert model.config.beta == 1.3
    assert model.config.max_steps == 7


def test_legacy_lr_alias_is_normalized() -> None:
    """``lr`` (legacy) is canonicalised to ``learning_rate`` at the boundary."""
    cls = Registry.get(ComponentCategory.MODEL, "eqprop")
    model = construct_model(
        cls,
        {"hidden_dim": 32, "num_layers": 2, "lr": 0.04, "beta": 0.6},
        input_dim=784,
        output_dim=10,
        model_name="eqprop",
    )
    assert model.config.learning_rate == 0.04


def test_unconsumable_knob_is_reported_as_phantom() -> None:
    """conv_eqprop cannot consume ``beta``; it must be surfaced, not silent."""
    cls = Registry.get(ComponentCategory.MODEL, "conv_eqprop")
    cfg = {"hidden_dim": 64, "num_layers": 2, "beta": 0.3, "max_steps": 10}
    assert "beta" in phantom_knobs(cls, cfg, input_dim=784, output_dim=10)
    # max_steps is a declared constructor param -> consumed, not phantom.
    assert "max_steps" not in phantom_knobs(cls, cfg, input_dim=784, output_dim=10)


def test_config_accepting_model_has_no_phantoms() -> None:
    cls = Registry.get(ComponentCategory.MODEL, "directed_ep")
    cfg = {"hidden_dim": 64, "num_layers": 2, "learning_rate": 0.05, "beta": 0.5}
    assert phantom_knobs(cls, cfg, input_dim=784, output_dim=10) == frozenset()


def test_model_kwargs_are_plain_scalars_not_objects() -> None:
    """model_kwargs must stay OmegaConf-safe: no nested ModelConfig/dataclass."""
    cls = Registry.get(ComponentCategory.MODEL, "directed_ep")
    kw = model_kwargs(
        cls,
        {"hidden_dim": 64, "num_layers": 2, "learning_rate": 0.05, "beta": 0.5},
        input_dim=784,
        output_dim=10,
        model_name="directed_ep",
    )
    assert "config" not in kw
    for value in kw.values():
        assert isinstance(value, (int, float, str, bool)) or value is None
    assert kw.get("learning_rate") == 0.05


def test_matching_param_count_from_construct_model() -> None:
    """The estimator's count must equal the trainer's construct path."""
    from bioplausible.experiment.param_estimator import estimate_param_count

    cls = Registry.get(ComponentCategory.MODEL, "eqprop")
    cfg = {"hidden_dim": 32, "num_layers": 2, "learning_rate": 0.02}
    model = construct_model(cls, cfg, input_dim=784, output_dim=10, model_name="eqprop")
    actual = sum(p.numel() for p in model.parameters())
    assert estimate_param_count("eqprop", cfg, input_dim=784, output_dim=10) == actual


def test_knobs_are_reflection_derived() -> None:
    """KNOBS is derived from ModelConfig fields, not a hand-maintained list."""
    for knob in ("learning_rate", "beta", "max_steps", "convergence_start"):
        assert knob in KNOBS
    for structural in ("input_dim", "output_dim", "hidden_dims", "extra", "name"):
        assert structural not in KNOBS


def test_build_model_config_surfaces_every_knob_in_extra() -> None:
    cfg = build_model_config(
        {"hidden_dim": 16, "num_layers": 1, "learning_rate": 0.01, "damping": 0.3, "tol": 1e-3},
        input_dim=784,
        output_dim=10,
        model_name="m",
    )
    # Non-field knobs are preserved for a model that reads ``extra``.
    assert cfg.extra["damping"] == 0.3
    assert cfg.extra["tol"] == 1e-3
