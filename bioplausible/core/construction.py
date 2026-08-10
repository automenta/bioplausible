"""Single canonical model-construction layer (no phantom drift, no aliasing).

Why this module exists
----------------------
A zoo model can be constructed many ways (loose ``model_cls(**kwargs)``, a
``build()`` classmethod, a ``config=`` object), and finders, the sweep, the
param estimator and the trainer each historically built models their own way.
That divergence is what let a sampled hyper-parameter be *silently dropped*
— e.g. ``beta``/``max_steps``/``learning_rate`` landing in ``ModelConfig.extra``
(ignored) on the direct-init path, so every eqprop probe trained with identical
defaults regardless of its config.

The contract here is simple and enforced by tooling, not eyeballs:

* Every model declares what it consumes through *reflection* (:func:`inspect
  `.signature``). A ``config: ModelConfig = None`` parameter means the model
  reads its hyper-parameters from ``ModelConfig`` fields — so those knobs are
  routed into a real, fully-populated ``ModelConfig``, never ``extra``.
* Sampled knobs that nothing can consume are *phantoms*: reported by
  :func:`phantom_knobs`, never silently ignored.
  * One canonical name per knob. ``ModelConfig``'s dataclass fields are the
  schema (reflection via ``dataclasses.fields``), so adding a field to
  ``ModelConfig`` automatically extends the knob schema. A small, named set of
  legacy aliases (``steps`` → ``max_steps``, ``lr`` → ``learning_rate``) is
  rewritten at the
  boundary — there is no scattered per-key aliasing.

Serialization is kept orthogonal to construction: :func:`model_kwargs` returns
a plain dict of scalar values (safe for the ``TrainerConfig`` OmegaConf
round-trip and checkpoints), while :func:`construct_model` is the single
function that turns those scalars into a living ``nn.Module`` with every knob
applied. The trainer, the estimator, the finders and the probe all build via
:func:`construct_model`, so they can never disagree.
"""

from __future__ import annotations

import inspect
import math
import types as _types
import typing
from dataclasses import dataclass
from dataclasses import fields as _dataclass_fields

from bioplausible.config.unified import ModelConfig, compute_hidden_dims

__all__ = [
    "KNOBS",
    "Consumption",
    "build_model_config",
    "construct_model",
    "model_kwargs",
    "phantom_knobs",
    "resolve_consumption",
]

#: Legacy names accepted at the config boundary, mapped onto a canonical knob
#: before any downstream logic. A single named constant (no scattered aliasing).
_KNOB_ALIASES: dict[str, str] = {
    "steps": "max_steps",
}

#: ``ModelConfig`` fields that are identity/structure, not tuning knobs.
_STRUCTURAL_FIELDS: frozenset[str] = frozenset({
    "name",
    "input_dim",
    "output_dim",
    "hidden_dims",
    "extra",
})

#: The canonical tuning-knob schema — derived by reflection from
#: ``ModelConfig``'s own fields, so it can never drift from the real config.
KNOBS: frozenset[str] = frozenset(
    f.name for f in _dataclass_fields(ModelConfig) if f.name not in _STRUCTURAL_FIELDS
)

#: Tuning knobs that are *not* ``ModelConfig`` fields (rule-engine / arch params).
#: Forwarded to a constructor only when the constructor declares them.
_OTHER_KWARGS: frozenset[str] = frozenset({
    "alpha",
    "backend",
    "cube_size",
    "gradient_method",
    "feedback_init",
    "threshold",
    "feedback_mode",
    "input_channels",
    "hidden_channels",
    "hebbian_lr",
    "layer_lr",
    "classifier_lr",
    "damping",
    "tol",
    "weight_decay",
})


@dataclass(frozen=True, slots=True)
class Consumption:
    """The declared consumer contract of one model constructor (by reflection).

    Attributes:
        accepted: Named parameters of ``__init__`` (``"self"`` removed).
        has_catch_all: Whether ``__init__`` has a ``**kwargs`` catch-all
            (represented by the sentinel ``"**"`` in ``accepted``).
        accepts_config: Whether ``__init__`` has an explicit ``config`` param —
            i.e. the model reads tuning knobs from ``ModelConfig``.
    """

    accepted: frozenset[str]
    has_catch_all: bool
    accepts_config: bool

    def can_consume(self, knob: str) -> bool:
        """Whether ``knob`` reaches this model.

        A model that accepts ``config`` consumes every knob (they land in
        ``ModelConfig`` fields or ``extra``). Otherwise the knob must be a
        declared constructor parameter or be absorbed by ``**kwargs``.
        """
        if self.accepts_config:
            return True
        return knob in self.accepted or self.has_catch_all


def _config_param_accepts_modelconfig(model_cls: object) -> bool:
    """Check if the model's ``config`` param is annotated to accept ``ModelConfig``."""
    try:
        sig = inspect.signature(model_cls.__init__)  # type: ignore[misc]
    except TypeError, ValueError:
        return False
    param = sig.parameters.get("config")
    if param is None:
        return False
    # Use get_type_hints to evaluate string annotations (from __future__ import annotations)
    try:
        hints = typing.get_type_hints(model_cls.__init__)
    except Exception:
        hints = {}
    ann = hints.get("config", param.annotation)
    if ann is ModelConfig:
        return True
    origin = getattr(ann, "__origin__", None)
    args = getattr(ann, "__args__", ())
    if origin in {typing.Union, _types.UnionType}:
        return any(a is ModelConfig for a in args)
    return False


def resolve_consumption(model_cls: object) -> Consumption:
    """Resolve a model's consumer contract by reflecting on its ``__init__``."""
    try:
        sig = inspect.signature(model_cls.__init__)  # type: ignore[misc]
    except TypeError, ValueError:
        return Consumption(frozenset(), False, False)
    params = set(sig.parameters)
    has_catch_all = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if has_catch_all:
        params.add("**")
    params.discard("self")
    accepts_config = _config_param_accepts_modelconfig(model_cls)
    return Consumption(frozenset(params), has_catch_all, accepts_config)


def _normalize(config: dict[str, object]) -> dict[str, object]:
    """Return a copy of ``config`` with legacy knob aliases canonicalised."""
    if not any(alias in config for alias in _KNOB_ALIASES):
        return config
    out = dict(config)
    for legacy, canonical in _KNOB_ALIASES.items():
        if legacy in out and canonical not in out:
            out[canonical] = out.pop(legacy)
    return out


def _as_float(config: dict[str, object], key: str, default: float) -> float:
    value = config.get(key)
    return float(value) if isinstance(value, int | float) else default


def _as_int(config: dict[str, object], key: str, default: int) -> int:
    value = config.get(key)
    return int(value) if isinstance(value, int) else default


def _as_bool(config: dict[str, object], key: str, default: bool) -> bool:
    value = config.get(key)
    return bool(value) if isinstance(value, bool) else default


def _derive_cube_size(config: dict[str, object]) -> dict[str, object]:
    """Map a neural_cube ``hidden_dim`` onto its ``cube_size`` (NeuralCube.build)."""
    if "cube_size" in config or "hidden_dim" not in config:
        return config
    cfg = dict(config)
    hidden = int(cfg.pop("hidden_dim"))
    cfg["cube_size"] = max(3, round(hidden ** (1 / 3)))
    return cfg


def _derive_conv_channels(
    config: dict[str, object], *, input_dim: object
) -> dict[str, object]:
    """Map a shared flat space onto a conv model's channel signature."""
    cfg = dict(config)
    # Derive hidden_channels from the shared sampled hidden_dim only when not
    # already set (a param matcher sets it directly). GroupNorm (fixed group
    # count, e.g. 8) needs a multiple of 8 so an arbitrary sampled width never
    # breaks a conv probe at construction.
    if "hidden_channels" not in cfg:
        hidden_dim = cfg.get("hidden_dim")
        if isinstance(hidden_dim, int):
            cfg["hidden_channels"] = max(8, 8 * math.ceil(hidden_dim / 8))
    # input_channels is ALWAYS derived for a conv model when absent, even if
    # hidden_channels was set directly (param matcher) — otherwise the model
    # cannot be counted/built (SWEEP_FAILURES #4).
    if "input_channels" not in cfg:
        if isinstance(input_dim, tuple) and input_dim:
            cfg["input_channels"] = int(input_dim[0])
        else:
            cfg["input_channels"] = 1
    return cfg


def build_model_config(
    config: dict[str, object],
    *,
    input_dim: int,
    output_dim: int,
    model_name: str,
) -> ModelConfig:
    """Build a fully-populated :class:`ModelConfig` from a sampled config.

    This is the single place where sampled training knobs are mapped onto real
    ``ModelConfig`` **fields** (the canonical names in :data:`KNOBS`). The raw
    config is preserved under ``extra`` so a model that digs there still sees
    its non-field knobs. Nothing a config-accepting model reads is dropped.
    """
    cfg = _normalize(config)
    hidden_dim = cfg.get("hidden_dim")
    num_layers = int(cfg.get("num_layers", 1))
    return ModelConfig(
        name=model_name,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=compute_hidden_dims(hidden_dim, num_layers),
        learning_rate=_as_float(cfg, "learning_rate", 0.001),
        beta=_as_float(cfg, "beta", 0.2),
        max_steps=_as_int(cfg, "max_steps", 30),
        convergence_threshold=_as_float(cfg, "convergence_threshold", 1e-3),
        convergence_start=_as_int(cfg, "convergence_start", 5),
        use_spectral_norm=_as_bool(cfg, "use_spectral_norm", True),
        spectral_norm_power_iterations=_as_int(
            cfg, "spectral_norm_power_iterations", 5
        ),
        extra=dict(cfg),
    )


def model_kwargs(
    model_cls: object,
    config: dict[str, object],
    *,
    input_dim: int,
    output_dim: int,
    model_name: str | None = None,
) -> dict[str, object]:
    """Return a plain, serializable kwargs dict for ``model_cls``.

    Scalars only — safe for the ``TrainerConfig`` OmegaConf round-trip and
    checkpoint serialization (never embeds a ``ModelConfig`` object). This is
    the *serialized* view; :func:`construct_model` is what actually builds a
    model with the knobs applied.

    Args:
        model_cls: The registered model constructor.
        config: Sampled config (may use legacy aliases).
        input_dim: Flattened input size.
        output_dim: Output size.
        model_name: Registered name (enables per-model derivation like
            neural_cube's ``cube_size``).

    Returns:
        A dict of scalar constructor kwargs reflecting what the model consumes.
    """
    cfg = _normalize(config)
    if model_name == "neural_cube":
        cfg = _derive_cube_size(cfg)

    consumption = resolve_consumption(model_cls)
    accepted, has_catch_all = consumption.accepted, consumption.has_catch_all

    # Conv-channel derivation only fires for models that *declare* channel
    # params — never merely because a ``**kwargs`` catch-all exists (an MLP with
    # a catch-all must not be silently treated as a conv model).
    if "input_channels" in accepted or "hidden_channels" in accepted:
        cfg = _derive_conv_channels(cfg, input_dim=input_dim)

    kwargs: dict[str, object] = {
        "input_dim": cfg.get("input_dim", input_dim),
        "output_dim": cfg.get("output_dim", output_dim),
    }
    if not (has_catch_all or "input_dim" in accepted):
        kwargs.pop("input_dim", None)
    if not (has_catch_all or "output_dim" in accepted):
        kwargs.pop("output_dim", None)
    for key in ("hidden_dim", "num_layers"):
        if key in cfg and (has_catch_all or key in accepted):
            kwargs[key] = cfg[key]
    for key in _OTHER_KWARGS:
        if key in cfg and (has_catch_all or key in accepted):
            kwargs[key] = cfg[key]

    # ``ModelConfig`` tuning knobs reach config-accepting models through the
    # config object; for models without ``config`` only forward what the
    # constructor declares.
    if not consumption.accepts_config:
        for key in KNOBS:
            if key in cfg and (has_catch_all or key in accepted):
                kwargs[key] = cfg[key]

    # learning_rate is ALWAYS surfaced as a scalar: the trainer's optimizer
    # consumes it for trainer-driven (BPTT) models, whose constructors may not
    # accept it. construct_model filters it back out for such constructors.
    if "learning_rate" in cfg:
        kwargs["learning_rate"] = cfg["learning_rate"]
    return kwargs


def construct_model(
    model_cls: object,
    config: dict[str, object],
    *,
    input_dim: int,
    output_dim: int,
    model_name: str | None = None,
) -> object:
    """Build a live model from a sampled config, applying every consumed knob.

    This is the single construction entrypoint used by the trainer, the param
    estimator, the finders and the probe. A model that accepts ``config`` gets
    a fully-populated :class:`ModelConfig`. A model without ``config`` gets the
    scalars it declares. A phantom knob (nothing consumes it) is left out and is
    reported by :func:`phantom_knobs` — never silently ignored.

    Returns:
        The constructed ``nn.Module``.
    """
    consumption = resolve_consumption(model_cls)
    if consumption.accepts_config:
        cfg = build_model_config(
            config,
            input_dim=input_dim,
            output_dim=output_dim,
            model_name=model_name or getattr(model_cls, "__name__", "model"),
        )
        try:
            return model_cls(config=cfg)  # type: ignore[operator]
        except TypeError:
            # Structural fallback: the model declares required positional
            # args in addition to ``config``. Do NOT cap depth — the
            # sampled ``num_layers`` has already been threaded into
            # ``hidden_dims`` by ``build_model_config``, so re-deriving
            # ``num_layers`` from the config must preserve the full depth
            # (a min(..., 2) cap here silently truncated every depth>=3
            # architecture through this path — the phantom-num_layers bug).
            structural = {
                "input_dim": cfg.input_dim,
                "hidden_dim": cfg.hidden_dims[0] if cfg.hidden_dims else 0,
                "output_dim": cfg.output_dim,
                "num_layers": max(len(cfg.hidden_dims), 1),
            }
            return model_cls(**structural, config=cfg)  # type: ignore[operator]
    kwargs = model_kwargs(
        model_cls,
        config,
        input_dim=input_dim,
        output_dim=output_dim,
        model_name=model_name,
    )
    # A model without ``**kwargs`` must only receive parameters it declares (the
    # serialized view may carry e.g. ``learning_rate`` for the trainer).
    if not consumption.has_catch_all:
        kwargs = {k: v for k, v in kwargs.items() if k in consumption.accepted}
    return model_cls(**kwargs)  # type: ignore[operator]


def phantom_knobs(
    model_cls: object,
    config: dict[str, object],
    *,
    input_dim: int,
    output_dim: int,
    model_name: str | None = None,
) -> frozenset[str]:
    """Return tuning knobs in ``config`` that nothing on the model can consume.

    A knob is *phantom* when it would change training behaviour but neither the
    model (no ``config=``, no constructor param, no ``**kwargs``) nor the
    trainer can use it. ``learning_rate`` is excluded because the trainer
    consumes it for BPTT models. The probe/sweep surface these as a defect
    instead of silently ignoring them.

    Unlike a static reflection-only check, this also *constructs* the model for
    config-accepting models and verifies that a sampled ``num_layers`` is
    honoured by the actual architecture (``len(model.config.hidden_dims)``
    tracks the request). That closes the ``build()``/``_build_layers`` phantom
    drift the old supervisor missed: a model could accept ``config``, get a
    fully-populated ``ModelConfig``, and then silently truncate ``hidden_dims``
    to ``[hidden_dim]`` — every probe trained at one hidden layer regardless of
    the sampled ``num_layers`` with zero knobs flagged.
    """
    cfg = _normalize(config)
    consumption = resolve_consumption(model_cls)
    # Depth supervision runs for every model: a sampled ``num_layers`` must
    # grow the constructed architecture, whether the model consumes ``config``
    # (knob → ``ModelConfig.hidden_dims``) or not (knob → structural args).
    knobs = set(
        key
        for key in KNOBS
        if key != "learning_rate"
        and key in cfg
        and not (consumption.has_catch_all or key in consumption.accepted)
    )
    if not consumption.accepts_config:
        knobs |= _config_num_layers_phantoms(
            model_cls,
            cfg,
            input_dim=input_dim,
            output_dim=output_dim,
            model_name=model_name,
        )
        return frozenset(knobs)
    return _config_num_layers_phantoms(
        model_cls,
        cfg,
        input_dim=input_dim,
        output_dim=output_dim,
        model_name=model_name,
    )


def _safe_layer_count(model: object, cfg_obj: object) -> int:
    """Best-effort layer count for the phantom-depth audit (Plan 8 §15.2 #3).

    Returns the number of constructed layers, or ``-1`` to signal "audit
    oracle unavailable" — the caller skips the phantom flag in that case (see
    :func:`_config_num_layers_phantoms`). Some models (EquiTile family,
    ``fabricpc_graph_pcn``) deliberately override ``transition_modules`` to
    raise ``NotImplementedError`` because their width axis is a graph / tile
    parameter rather than a layers list. Their ``config.hidden_dims`` follows
    model-specific conventions (e.g. EquiTile stores ``num_layers - 2``
    interior layers), so it cannot be compared against the sampled
    ``num_layers`` uniformly without false positives. The registry-wide
    param-count growth guard is the authoritative check for these models;
    depth-oracle truth here would only cry wolf.

    A non-negative return is the layer count constructed — ``0`` when neither
    signal is available. ``0`` is safe because the audit only flags
    ``actual_layers < num_layers`` (zero never triggers that path).
    """
    try:
        built = getattr(model, "transition_modules", list)()
    except NotImplementedError:
        return -1
    if built:
        return len(built)
    if cfg_obj is not None:
        return len(getattr(cfg_obj, "hidden_dims", ()))
    return 0


def _config_num_layers_phantoms(
    model_cls: object,
    cfg: dict[str, object],
    *,
    input_dim: int,
    output_dim: int,
    model_name: str | None,
) -> frozenset[str]:
    """Check a config-accepting model's *depth* honours the sampled ``num_layers``.

    Returns ``{"num_layers"}`` when ``num_layers > 1`` is sampled but the
    constructed model's ``config.hidden_dims`` does not grow with it (i.e. the
    architecture/silently dropped the knob). Safe to call on any config-
    accepting model: constructs under ``try``, and models that cannot reflect
    depth (e.g. some conv/graph types) are treated as depth-honouring to avoid
    false positives when the model *deliberately* uses another width axis.
    """
    num_layers = cfg.get("num_layers")
    if not isinstance(num_layers, int) or num_layers <= 1:
        return frozenset()
    # Build with a small width so the construction is cheap and does not OOM
    # (a probe on the real config already happens elsewhere; this is a shallow
    # supervision probe). ``hidden_dim`` is bumped only if the caller did not
    # already supply one — we never want to skip the check just to win a race
    # against a huge ``hidden_dim``.
    probe_cfg = dict(cfg)
    probe_cfg.setdefault("hidden_dim", 64)
    probe_cfg["num_layers"] = num_layers
    try:
        model = construct_model(
            model_cls,
            probe_cfg,
            input_dim=input_dim,
            output_dim=output_dim,
            model_name=model_name,
        )
    except TypeError, ValueError, RuntimeError, NotImplementedError:
        # Cannot construct here (e.g. fixture-only model); cannot verify depth —
        # do not cry wolf on unverifiable models.
        return frozenset()
    cfg_obj = getattr(model, "config", None)
    # Prefer the *actually built* architecture over ``config.hidden_dims``:
    # the structural path can route ``num_layers`` separately from the config
    # (the phantom-num-layers defect), so reading hidden_dims alone cannot see
    # when the built model diverges from the sampled depth. ``transition_modules``
    # reflects what was materially constructed; fall back to hidden_dims only
    # for models whose width axis is not a layers list (conv ``hidden_channels``,
    # cube ``cube_size``).
    actual_layers = _safe_layer_count(model, cfg_obj)
    # ``-1`` from the oracle means the model deliberately opts out of the
    # standard ``transition_modules`` / ``hidden_dims`` convention (EquiTile
    # family, ``fabricpc_graph_pcn`` whose width is a tile graph not a layers
    # list). The phantom-depth audit cannot be applied to such models using the
    # standard ``hidden_dims`` axis: the audit would false-positive on
    # nonstandard conventions (EquiTile stores interior layers only, length
    # ``num_layers-2``). The registry-wide depth guard
    # (test_all_models_honor_depth_or_are_knowingly_phantom) already verifies
    # param-count growth, which is the authoritative signal in that case;
    # don't cry wolf here.
    if actual_layers < 0:
        return frozenset()
    # Some models grow a *fixed* width axis instead of ``hidden_dims`` (conv
    # ``hidden_channels``, cube ``cube_size``), and ``construct_model`` may
    # route ``num_layers`` through a structural cap. Only flag when the config
    # faithfully carries ``hidden_dims`` and the count is stuck far below the
    # request — the canonical MLP-failure signature.
    if actual_layers > 0 and actual_layers < num_layers:
        return frozenset({"num_layers"})
    return frozenset()
