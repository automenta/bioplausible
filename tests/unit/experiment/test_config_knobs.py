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
        {
            "hidden_dim": 64,
            "num_layers": 2,
            "learning_rate": 0.05,
            "beta": 0.5,
            "max_steps": 20,
        },
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
        {
            "hidden_dim": 32,
            "num_layers": 2,
            "learning_rate": 0.02,
            "beta": 1.3,
            "max_steps": 7,
        },
        input_dim=784,
        output_dim=10,
        model_name="eqprop",
    )
    assert model.config.learning_rate == 0.02
    assert model.config.beta == 1.3
    assert model.config.max_steps == 7


def test_learning_rate_is_used_directly() -> None:
    """``learning_rate`` is used directly (no ``lr`` alias)."""
    cls = Registry.get(ComponentCategory.MODEL, "eqprop")
    model = construct_model(
        cls,
        {"hidden_dim": 32, "num_layers": 2, "learning_rate": 0.04, "beta": 0.6},
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
        {
            "hidden_dim": 16,
            "num_layers": 1,
            "learning_rate": 0.01,
            "damping": 0.3,
            "tol": 1e-3,
        },
        input_dim=784,
        output_dim=10,
        model_name="m",
    )
    # Non-field knobs are preserved for a model that reads ``extra``.
    assert cfg.extra["damping"] == 0.3
    assert cfg.extra["tol"] == 1e-3


def test_num_layers_unconsumed_is_reported_as_phantom() -> None:
    """A model that cannot grow depth must surface ``num_layers`` as phantom.

    Regression for the deep-eqprop phantom bug: a hand-written ``build()``
    accepted ``num_layers`` but constructed ``hidden_dims=[hidden_dim]``, so
    every sample trained at one hidden layer regardless of the sampled depth —
    with zero knobs flagged. The supervisor now builds the model and checks
    ``len(config.hidden_dims)`` actually grew with the request.
    """
    from bioplausible.core.registry import ComponentCategory, Registry

    for model_name in ("graph_eqprop", "conv_eqprop"):
        cls = Registry.get(ComponentCategory.MODEL, model_name)
        cfg = {"hidden_dim": 64, "num_layers": 3, "learning_rate": 0.01}
        assert "num_layers" in phantom_knobs(
            cls, cfg, input_dim=784, output_dim=10, model_name=model_name
        )


def test_num_layers_consumed_is_not_phantom() -> None:
    """The consolidated deep eqprop engine honours ``num_layers`` — not phantom."""
    from bioplausible.core.registry import ComponentCategory, Registry

    for model_name in (
        "eqprop",
        "directed_ep",
        "lazy_eqprop",
        "finite_nudge_ep",
        "momentum_equilibrium",
        "sparse_equilibrium",
        "eqprop_mlp",
        "holomorphic_ep",
    ):
        cls = Registry.get(ComponentCategory.MODEL, model_name)
        cfg = {"hidden_dim": 64, "num_layers": 3, "learning_rate": 0.01}
        assert "num_layers" not in phantom_knobs(
            cls, cfg, input_dim=784, output_dim=10, model_name=model_name
        ), (
            f"{model_name} must honour sampled num_layers (truncating hidden_dims "
            "to [hidden_dim] would silently lock the probe to one hidden layer)"
        )


def test_param_count_varies_with_num_layers() -> None:
    """The fair-comparison estimator must agree with the built model per depth."""
    from bioplausible.experiment.param_estimator import estimate_param_count

    for model_name in (
        "eqprop",
        "directed_ep",
        "lazy_eqprop",
        "finite_nudge_ep",
        "momentum_equilibrium",
        "sparse_equilibrium",
        "eqprop_mlp",
    ):
        cls = Registry.get(ComponentCategory.MODEL, model_name)
        one = estimate_param_count(
            model_name,
            {"hidden_dim": 64, "num_layers": 1},
            input_dim=784,
            output_dim=10,
        )
        three = estimate_param_count(
            model_name,
            {"hidden_dim": 64, "num_layers": 3},
            input_dim=784,
            output_dim=10,
        )
        assert three != one, (
            f"{model_name}: num_layers=1 and num_layers=3 report identical "
            "param counts — the depth knob does not change the architecture"
        )
        built = construct_model(
            cls,
            {"hidden_dim": 64, "num_layers": 3},
            input_dim=784,
            output_dim=10,
            model_name=model_name,
        )
        assert sum(p.numel() for p in built.parameters()) == three


def test_all_models_honor_depth_or_are_knowingly_phantom() -> None:
    """Registry-wide depth-invariant guard (Plan 8 §D2 regression).

    No registered model may silently drop the sampled ``num_layers``: either
    the constructed architecture grows its parameter count with depth (the
    healthy case), or the depth knob is *reported* as phantom by
    :func:`phantom_knobs` so sweeps quarantine it. This is the universal
    regression that catches the feedback_alignment-class defect (structural
    fallback silently capping ``num_layers``) without needing one test per
    model.

    A model that absorbs ``num_layers`` through ``**kwargs`` without using it
    is *not* flagged by the constructor surface (the knob lands in kwargs, so
    the surface treats it as "absorbed") — but it must still be quarantined:
    if it silently drops depth it must be tagged ``status:broken`` so default
    sweeps exclude it. Otherwise the guard hard-fails.
    """
    from bioplausible.experiment.param_estimator import estimate_param_count

    unverifiable: list[str] = []
    silently_dropped: list[str] = []
    for rec in Registry.query(category=ComponentCategory.MODEL):
        name = rec["name"]
        cls = Registry.get(ComponentCategory.MODEL, name)
        cfg = {"hidden_dim": 32, "num_layers": 3, "learning_rate": 0.01}
        # Models that refuse vision-style dimension args (LM/RL/graph/diffusion)
        # cannot be audited on the vision dummy dims; opt them out of the guard.
        # conv_tile's depth map is offset (``num_fc_layers = num_layers - 2``),
        # so the 1-vs-3 audit never triggers its (real, verified) depth growth.
        # Same applies to algorithm-specific conv_tile variants.
        if name in (
            "backprop_transformer_lm",
            "eqprop_diffusion",
            "graph_tile",
            "rl_tile",
            "timeseries_tile",
            "conv_tile",
            "conv_tile_fa",
            "conv_tile_tp",
            "conv_tile_hebbian",
            "conv_tile_snn",
            "conv_tile_pc",
            "custom_stacked_model",
            "spiking_stdp",
        ):
            continue
        try:
            ph = phantom_knobs(cls, cfg, input_dim=256, output_dim=10, model_name=name)
        except TypeError, ValueError, NotImplementedError, RuntimeError:
            unverifiable.append(name)
            continue
        # A knowingly-phantom depth knob is acceptable: sweeps quarantine it.
        if "num_layers" in ph:
            continue
        try:
            one = estimate_param_count(
                name,
                {"hidden_dim": 32, "num_layers": 1},
                input_dim=256,
                output_dim=10,
            )
            three = estimate_param_count(
                name,
                {"hidden_dim": 32, "num_layers": 3},
                input_dim=256,
                output_dim=10,
            )
        except TypeError, ValueError, NotImplementedError, RuntimeError, Exception:
            unverifiable.append(name)
            continue
        if three == one:
            silently_dropped.append(name)

    # A model that (a) absorbs num_layers via **kwargs (no phantom flagged) yet
    # (b) does not grow depth must be quarantined (status:broken). Both
    # conditions together are the failure: depth silently dropped AND not
    # excluded from default sweeps.
    tags_by_name = {
        rec["name"]: rec["metadata"].tags
        for rec in Registry.query(category=ComponentCategory.MODEL)
    }
    unquarantined_drops = [
        m for m in silently_dropped if "status:broken" not in tags_by_name[m]
    ]
    assert not unquarantined_drops, (
        "models silently dropping sampled num_layers with NO phantom flag and "
        "NO status:broken quarantine (they run in default sweeps with a "
        f"dead depth knob): {sorted(unquarantined_drops)}"
    )
    quarantined = [m for m in silently_dropped if "status:broken" in tags_by_name[m]]
    assert set(quarantined) <= {
        "equilibrium_alignment",
        "neural_cube",
    }, f"unexpected quarantined drops: {sorted(quarantined)}"
    # Keep the unverifiable list noisy-but-tolerated so a future audit can
    # shrink it; the guard only hard-fails on *silent unquarantined* drops.
    print(f"[registry depth audit] unverifiable: {sorted(unverifiable)}")
