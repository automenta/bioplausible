"""``broad_sweep`` - Relaxed Discovery Loop, cycle 1 (EXPERIMENT_PLAN4 sec.1-3).

Maps the resource territory across **every registered rule family** with cheap
shallow probes, applying the **liveness gate** so non-converging families are
excluded from the resource map without manual audit.

This is deliberately *not* a flagship search: each probe is 1 epoch (default),
a few configs per family are sampled, and the measured axes are **resource
costs** (memory variance, compute time, FLOPs) rather than accuracy. The
outcome is a coarse Pareto landscape showing where each *live* family sits in
resource space, plus an auto-surfaced list of *dead* (non-loss-decreasing)
families — the quarantine signal for the later rule-health audit.

Every completed probe is sunk to the KnowledgeBase via the probe driver
(``CoreTrainerDriver``), so the engine's moat compounds per run.

Usage::

    uv run python scripts/broad_sweep.py --epochs 1 --probes-per-rule 3 --families all
    uv run python scripts/broad_sweep.py --families fa,hebbian --probes-per-rule 5 --device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from computronium.hyperopt.search_space import get_rule_space, get_search_space

if TYPE_CHECKING:
    from computronium.experiment.probe import CoreTrainerDriver

logger = logging.getLogger(__name__)

__all__ = ["broad_sweep", "main", "sample_config_for_space", "space_for_family"]


def _git_sha() -> str:
    """Current git HEAD short hash, or ``"unknown"`` outside a git repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # ruff: ignore[start-process-with-partial-path]
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _env_provenance() -> dict[str, str]:
    """Environment fingerprint for the sweep record."""
    return {
        "git_sha": _git_sha(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


_DEFAULT_TASK = "mnist"
_INT_PARAMS = frozenset({"hidden_dim", "num_layers", "cube_size", "max_steps"})
_LIVE_MAJORITY = 0.5
_MIN_LIVENESS_EPOCHS = 2

# A Broad Sweep maps the territory, it does not search for a winner. Capping the
# sampled config to small model sizes keeps every shallow probe cheap — a 1000+
# wide / 100-settle-step model is fine for a fine search but wastes GPU on a
# 1-2 epoch breadth probe. These caps are deliberately modest (they only bound
# the resource-biggest axes, never the learning-rate/alpha knobs).
_SHALLOW_CAPS: dict[str, int] = {
    "hidden_dim": 128,
    "num_layers": 2,
    "max_steps": 20,
    "steps": 20,
    "cube_size": 4,
}

# Contrastive/settling eqprop models (SWEEP_FAILURES #8) pay the full
# bidirectional settle loop twice (free+nudged phases) per batch, so even a
# capped ``max_steps=20`` boards out the shallow probe's 30s epoch budget and
# Families whose registered models can silently degrade to plain BPTT when not
# explicitly configured. Probing them that way would report *backprop* memory
# as the bio rule's cost — defeating the cost-of-locality thesis. Each entry
# activates the family's actual local rule:
#   - eqprop   -> energy-contrastive via model's own train_step (gradient_method="equilibrium")
#   - fa       -> a Feedback-Alignment propagator (local random-feedback)
#   - hebbian  -> a Contrastive-Hebbian propagator (synapse-local)
_RULE_ACTIVATION: dict[str, dict[str, object]] = {
    "eqprop": {"config": {"gradient_method": "equilibrium"}},
    "fa": {"propagator": "feedback_alignment"},
    "hebbian": {"propagator": "contrastive_hebbian_learning"},
}


def _eqprop_gradient_method(model: str) -> str:
    """All eqprop models use the energy-contrastive engine with train_step.

    ``gradient_method="equilibrium"`` is the fast, O(1)-memory implicit
    differentiation path — for models with a native contrastive ``train_step``
    (the unified ``EquilibriumMLP`` family) the trainer routes through the
    local rule anyway; for models without one (conv/graph add-ons) it keeps
    the cheap implicit backward instead of the slow explicit free+nudged
    settle or a BPTT fallback. The bare ``"equilibrium"`` value therefore
    gives every eqprop model its fastest correct training path.
    """
    return "equilibrium"


def _rule_activation_for(model: str, family: str) -> dict[str, object]:
    """Resolve the per-model rule activation for a family.

    ``hebbian`` models that ship their own local ``train_step`` (e.g.
    ``three_factor_hebbian``, ``deep_hebbian``, ``hebbian_chain``) should use
    their native rule — the forced CHL propagator would otherwise override
    their bespoke update in the trainer dispatch (Phase 2 before Phase 3). The
    CHL propagator is applied only to hebbian models without a native
    ``train_step``.
    """
    activation = dict(_RULE_ACTIVATION.get(family, {}))
    cfg = dict(activation.get("config") or {})
    if family == "eqprop" and cfg.get("gradient_method") is not None:
        cfg["gradient_method"] = _eqprop_gradient_method(model)
        activation["config"] = cfg
    return activation


def _shallow_clamp(config: dict[str, object]) -> dict[str, object]:
    """Clamp resource-heavy config axes to small values for shallow probes."""
    clamped = dict(config)
    for name, cap in _SHALLOW_CAPS.items():
        if name in clamped and isinstance(clamped[name], (int, float)):
            clamped[name] = min(int(clamped[name]), int(cap))
    return clamped


# RULE_SPACES is the richer per-rule space (incl. equilibrium knobs). Prefer it
# by family name; models without a rule space fall back to their search space.
_RULE_FAMILIES = frozenset({
    "eqprop",
    "backprop",
    "fa",
    "forward_only",
    "predictive_coding",
    "spiking",
    "hebbian",
    "target_prop",
})


def _family_rule_key(family: str) -> str | None:
    """Return the RULE_SPACES key to sample for a family, if one exists."""
    aliases: dict[str, str] = {
        "fa": "feedback_alignment",
        "forward_only": "pepita",
        "predictive_coding": "pepita",
    }
    key = aliases.get(family, family)
    try:
        get_rule_space(key)
    except ValueError:
        return None
    else:
        return key


def sample_config_for_space(space: dict[str, object]) -> dict[str, object]:
    """Sample one config from a ``{name: spec}`` space via uniform draws.

    Mirrors ``SearchSpace.sample`` (list → categorical choice, ``(min,max,'int')``
    → int, ``'log'`` → log-uniform, ``'linear'`` → uniform) but accepts both a
    ``RULE_SPACES`` dict and a ``SearchSpace.params`` dict, so the sweep never
    needs an Optuna trial for a shallow breadth probe.

    Args:
        space: Parameter name → range tuple or discrete-choice list.

    Returns:
        A config dict of sampled parameter values.
    """
    import numpy as np

    config: dict[str, object] = {}
    for name, spec in space.items():
        if isinstance(spec, list):
            config[name] = np.random.choice(spec).item()
        else:
            min_v, max_v, scale = spec
            if scale == "int":
                config[name] = int(np.random.randint(int(min_v), int(max_v) + 1))
            elif scale == "log":
                log_min, log_max = math.log(min_v), math.log(max_v)
                config[name] = float(np.exp(np.random.uniform(log_min, log_max)))
            else:  # "linear"
                config[name] = float(np.random.uniform(min_v, max_v))
        if name in _INT_PARAMS:
            config[name] = int(config[name])
    return config


def space_for_family(model: str, family: str) -> dict[str, object] | None:
    """Return a search space dict for ``model``, or None if unsamplable.

    Resolution order: an exact ``RULE_SPACES`` entry for the model → a rule
    space for the family → the model's registered search space.
    """
    try:
        return dict(get_rule_space(model))
    except ValueError:
        pass
    rule_key = _family_rule_key(family)
    if rule_key is not None:
        return dict(get_rule_space(rule_key))
    try:
        return dict(get_search_space(model).params)
    except ValueError:
        return None


def _is_live(runs: list[dict[str, object]], determined: bool) -> bool:
    """Liveness gate (binary): loss decreases across the run for >= half of probes.

    The "epoch 0 vs epoch final" comparison needs a run of **two or more**
    epochs to be meaningful; with a single epoch both endpoints are identical
    and every rule would look dead. ``determined`` is ``False`` for such
    degenerate runs, so the gate reports ``False`` (undetermined) rather than a
    false "dead" verdict that would wrongly quarantine a healthy rule.
    """
    if not determined or not runs:
        return False
    relevant = [r for r in runs if r.get("ok")]
    if not relevant:
        return False
    decreases = 0
    total = 0
    for r in relevant:
        loss_0 = float(r.get("loss_epoch_0") or 0.0)
        loss_final = float(r.get("loss_epoch_final") or 0.0)
        if loss_0 > 0.0 or loss_final > 0.0:
            total += 1
            if loss_0 > loss_final:
                decreases += 1
    if total == 0:
        return False
    return decreases / total >= _LIVE_MAJORITY


def _coerce(v: object, default: float = 0.0) -> float:
    """Coerce a metric to float, returning ``default`` on non-finite values."""
    try:
        f = float(v)
    except TypeError, ValueError:
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


def _summarize(runs: list[dict[str, object]], key: str) -> dict[str, float]:
    """Mean/std over the ok probes for a metric key (empty → zeros)."""
    vals = [_coerce(r.get(key)) for r in runs if r.get("ok")]
    if not vals:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    mean = sum(vals) / len(vals)
    if len(vals) > 1:
        variance = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0
    return {"mean": mean, "std": std, "n": len(vals)}


def _probe_runs(  # ruff: ignore[too-many-arguments]
    driver: CoreTrainerDriver,
    *,
    model: str,
    family: str,
    space: dict[str, object],
    probes_per_rule: int,
    epochs: int,
    seed: int,
    device: str,
    task: str,
    max_params: int = 0,
) -> tuple[list[dict[str, object]], int, int]:
    """Run ``probes_per_rule`` shallow probes of one model.

    Applies the family's rule activation (so bio-rule probes measure their own
    local cost, not a BPTT fallback) and clamps sampled configs to small sizes.
    Bio families are also told the trainer must NOT silently fall back to BPTT
    (``allow_bptt_fallback=False``) so any degradation is loud + recorded, and
    the summary flags it as a defect.

    Args:
        driver: The probe driver.
        model: Registered model name.
        family: The model's family (for rule activation).
        space: Sampled search space.
        probes_per_rule: Number of seeds to run.
        epochs: Epochs per probe.
        seed: Master seed.
        device: Target device.
        task: Task name.

    Returns:
        A tuple of the per-probe runs, the total probe count, and the ok count.
    """
    activation = _rule_activation_for(model, family)
    propagator = activation.get("propagator")
    # Bio families may not silently fall back to BPTT: a bio probe that ends up
    # backprop-pathed is a defect, so the trainer warns loudly and path records.
    allow_bptt_fallback = family not in _RULE_ACTIVATION
    runs: list[dict[str, object]] = []
    n_total = 0
    n_ok = 0
    for probe_i in range(probes_per_rule):
        config = _shallow_clamp(sample_config_for_space(space))
        if activation.get("config"):
            config = {**config, **activation["config"]}
        probe_seed = seed + 10_000 * probe_i
        n_total += 1
        logger.info(
            "probe family=%s model=%s probe=%d cfg=%s",
            family,
            model,
            probe_i,
            config,
        )
        try:
            metrics = driver.train(
                model=model,
                task=task,
                config=config,
                seed=probe_seed,
                epochs=epochs,
                device=device,
                propagator=propagator,
                allow_bptt_fallback=allow_bptt_fallback,
            )
        except Exception as exc:  # a broken probe must not kill the sweep
            logger.warning(
                "probe family=%s model=%s probe=%d failed: %s",
                family,
                model,
                probe_i,
                exc,
            )
            runs.append({
                "ok": False,
                "error": str(exc),
                "defects": ["nan_divergence"] if "diverged" in str(exc) else [],
            })
            continue
        # NaN divergence that slipped past the trainer guard (defense in depth):
        # a non-finite loss means the run is not a real result — flag it as a
        # defect and exclude it from liveness/ok accounting.
        defects: list[str] = []
        if "final_train_loss" in metrics:
            final_loss = _coerce(metrics.get("final_train_loss"), default=float("nan"))
            if not math.isfinite(final_loss):
                defects.append("nan_divergence")
        phantom = metrics.get("phantom_knobs") or []
        if phantom:
            defects.append(f"phantom_knobs={sorted(phantom)!r}")
        if max_params > 0 and int(metrics.get("param_count") or 0) > max_params:
            defects.append(f"over_budget={int(metrics.get('param_count') or 0)}")
        # Epoch-time truncation: a run whose epoch was cut short by the
        # ``max_epoch_time`` budget carries resource metrics over a *partial*
        # epoch. Averaging those into the family map alongside full epochs would
        # distort the memory/time comparison, so prune the run (flag as defect,
        # exclude from ok/liveness) instead of reporting a fair-looking run.
        if metrics.get("epoch_time_budget_stopped"):
            defects.append("epoch_time_truncated")
        n_ok += 1 if not defects else 0
        metrics["defects"] = defects
        runs.append({"ok": not defects, **metrics, "config": config})
        logger.info(
            "probe done family=%s model=%s probe=%d ok=%s defects=%s "
            "acc=%.4f params=%s mmb=%.1f",
            family,
            model,
            probe_i,
            not defects,
            defects,
            _coerce(metrics.get("final_acc")),
            metrics.get("param_count"),
            _coerce(metrics.get("peak_memory_mb")),
        )
    return runs, n_total, n_ok


def broad_sweep(  # ruff: ignore[too-many-arguments, too-many-locals]
    *,
    families: list[str],
    probes_per_rule: int,
    epochs: int,
    task: str,
    device: str,
    seed: int,
    num_workers: int,
    target_hardware: str | None = None,
    max_params: int = 0,
    max_epoch_time: float = 0.0,
    exclude_models: list[str] | None = None,
) -> dict[str, object]:
    """Run the shallow breadth sweep and return the resource-landscape report.

    Args:
        families: Rule-family names to sweep (``"all"`` expands to every
            family with a rule space).
        probes_per_rule: Number of configs to sample per family.
        epochs: Training epochs per probe (kept shallow by design).
        task: Task name (e.g. ``"mnist"``).
        device: Target device.
        seed: Master seed for reproducibility.
        num_workers: DataLoader worker count per probe.
        target_hardware: Substrate facade for the probes (plan §17).
        max_params: Fair-comparison parameter budget (0 = breadth mode).
        max_epoch_time: Per-epoch wall-clock budget in seconds (0 = unlimited) —
            caps slow-settling eqprop epochs so a shallow probe stays bounded.

    Returns:
        The sweep report dict (families, liveness, live resource map, dead list).
    """
    import numpy as np

    np.random.seed(seed)
    from computronium.experiment.probe import CoreTrainerDriver

    driver = CoreTrainerDriver(
        num_workers=num_workers,
        batch_size=128,
        track_energy=True,
        track_flops=True,
        track_memory=True,
        target_hardware=target_hardware,
        max_epoch_time=max_epoch_time,
    )

    all_families = sorted(_RULE_FAMILIES)
    if "all" in families or not families:
        requested = all_families
    else:
        requested = [f for f in families if f in all_families]
        missing = [f for f in families if f not in all_families]
        if missing:
            logger.warning("families not registered, skipped: %s", missing)

    report: dict[str, object] = {}
    probe_total = 0
    probe_ok = 0
    skipped: dict[str, list[str]] = {}

    for family in requested:
        rule_key = _family_rule_key(family)
        models = [rule_key] if rule_key is not None else []
        if not models:
            logger.warning("family=%s: no rule space, skipped", family)
            skipped[family] = []
            report[family] = _summarize_family(
                {}, determined=epochs >= _MIN_LIVENESS_EPOCHS, family=family
            )
            continue
        if exclude_models:
            models = [m for m in models if m not in exclude_models]
        family_runs: dict[str, list[dict[str, object]]] = {}
        family_skipped: list[str] = []
        for model in models:
            space = space_for_family(model, family)
            if space is None:
                logger.info(
                    "family=%s model=%s: no searchable space, skipped", family, model
                )
                continue
            runs, n_total, n_ok = _probe_runs(
                driver,
                model=model,
                family=family,
                space=space,
                probes_per_rule=probes_per_rule,
                epochs=epochs,
                seed=seed,
                device=device,
                task=task,
                max_params=max_params,
            )
            probe_total += n_total
            probe_ok += n_ok
            family_runs[model] = runs
        skipped[family] = family_skipped
        report[family] = _summarize_family(
            family_runs, determined=epochs >= _MIN_LIVENESS_EPOCHS, family=family
        )

    report["_meta"] = {
        "epochs": epochs,
        "probes_per_rule": probes_per_rule,
        "task": task,
        "device": device,
        "seed": seed,
        "target_hardware": target_hardware,
        "max_params": max_params,
        "max_epoch_time": max_epoch_time,
        "probes_total": probe_total,
        "probes_ok": probe_ok,
        "skipped": skipped,
        "defects": {
            fam: entry["defects"]
            for fam, entry in report.items()
            if not fam.startswith("_")
        },
        "provenance": _env_provenance(),
    }
    return report


def _bio_defect(runs: list[dict[str, object]]) -> bool:
    """True if any ok probe in a bio family fell back to silent BPTT.

    A bio-rule probe whose dominant credit-assignment path is ``"bptt"`` means
    the family's own local rule did not engage (silent fallback). This is the
    plan's flagship self-diagnosis flag: surfaced in the report, never audited
    by a human.
    """
    return any(r.get("ok") and r.get("training_path") == "bptt" for r in runs)


def _summarize_family(
    family_runs: dict[str, list[dict[str, object]]],
    *,
    determined: bool,
    family: str = "",
) -> dict[str, object]:
    """Reduce a family's per-model probe runs into a landscape entry.

    Associates each model with a ``live`` verdict (liveness gate), per-model
    resource means/stds (memory, time, flops), and a family-level verdict
    (live iff any model is live). ``determined`` is False when the run had too
    few epochs for the liveness comparison to be meaningful; then no model is
    marked live or dead (both are undetermined) and the resource map is empty.
    Bio families (``family in _RULE_ACTIVATION``) additionally get a
    ``defects`` list naming models that silently fell back to BPTT.

    Args:
        family_runs: Per-model list of probe metrics dicts.
        determined: Whether the liveness gate is meaningful for this run.
        family: Rule-family name (for bio BPTT-fallback defect diagnosis).
    """
    is_bio = family in _RULE_ACTIVATION
    models_out: dict[str, object] = {}
    live_models = 0
    defect_models: list[str] = []
    for model, runs in sorted(family_runs.items()):
        live = _is_live(runs, determined)
        if live:
            live_models += 1
        # Per-model defect set: union of run-level defects (nan_divergence,
        # phantom knobs) plus the bio BPTT-fallback defect.
        model_defects: list[str] = []
        for r in runs:
            model_defects.extend(r.get("defects") or [])
        if is_bio and _bio_defect(runs):
            model_defects.append("bptt_fallback")
        model_defects = sorted(set(model_defects))
        if model_defects:
            defect_models.append(model)
        models_out[model] = {
            "live": live,
            "determined": determined,
            "defect": bool(model_defects),
            "defects": model_defects,
            "probes_run": len(runs),
            "probes_ok": sum(1 for r in runs if r.get("ok")),
            "final_acc_mean": _summarize(runs, "final_acc"),
            "peak_memory_mb": _summarize(runs, "peak_memory_mb"),
            "wall_time_s": _summarize(runs, "wall_time_s"),
            "forward_flops": _summarize(runs, "forward_flops"),
            "param_count": _summarize(runs, "param_count"),
        }
    defects = sorted(defect_models)
    living_models = [m for m, d in models_out.items() if d["live"]]
    dead_models = [m for m, d in models_out.items() if not d["live"]]
    live_aggregate = {
        "peak_memory_mean": _summarize(
            [r for m in living_models for r in family_runs[m]], "peak_memory_mb"
        )["mean"],
        "wall_time_mean": _summarize(
            [r for m in living_models for r in family_runs[m]], "wall_time_s"
        )["mean"],
        "peak_memory_std": _summarize(
            [r for m in living_models for r in family_runs[m]], "peak_memory_mb"
        )["std"],
    }
    if not determined:
        return {
            "live": None,
            "determined": False,
            "n_models": len(models_out),
            "n_live": 0,
            "n_dead": 0,
            "living_models": [],
            "dead_models": [],
            "undetermined_models": sorted(models_out),
            "defects": defects,
            "models": models_out,
            "resource": {},
            "reason": "fewer than 2 epochs: loss comparison degenerate",
        }
    return {
        "live": any(d["live"] for d in models_out.values()),
        "n_models": len(models_out),
        "n_live": live_models,
        "n_dead": len(dead_models),
        "living_models": living_models,
        "dead_models": dead_models,
        "undetermined_models": [],
        "defects": defects,
        "models": models_out,
        "resource": live_aggregate if living_models else {},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--families",
        default="all",
        help="Comma-separated families, or 'all' (default) for every registered family",
    )
    parser.add_argument("--probes-per-rule", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--task", default=_DEFAULT_TASK)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-epoch-time",
        type=float,
        default=0.0,
        help="Per-epoch wall-clock budget in seconds (0 = unlimited). Caps "
        "slow-settling eqprop epochs so shallow probes stay bounded.",
    )
    parser.add_argument(
        "--max-params",
        type=int,
        default=0,
        help="Fair-comparison budget: rematch each probe's width to ~max_params "
        "(0 = breadth mode, compare at matched depth/width)",
    )
    parser.add_argument(
        "--exclude-models",
        default="",
        help="Comma-separated model names to exclude from the sweep",
    )
    parser.add_argument("--cache-dir", default="logs")
    parser.add_argument(
        "--target-hardware",
        choices=["gpu", "fpga", "analog"],
        default=None,
        help="Substrate facade for probes (plan §17)",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    start = time.time()
    families = [f.strip() for f in args.families.split(",") if f.strip()]

    report = broad_sweep(
        families=families,
        probes_per_rule=args.probes_per_rule,
        epochs=args.epochs,
        task=args.task,
        device=args.device,
        seed=args.seed,
        num_workers=args.num_workers,
        target_hardware=args.target_hardware,
        max_params=args.max_params,
        max_epoch_time=args.max_epoch_time,
        exclude_models=[m.strip() for m in args.exclude_models.split(",") if m.strip()],
    )
    report["_meta"]["elapsed_s"] = round(time.time() - start, 1)

    out_path = Path(args.cache_dir) / f"broad_sweep_{args.task}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    meta = report["_meta"]
    logger.info(
        "broad sweep done: %d probes, %d ok, %d families in %.1fs",
        meta["probes_total"],
        meta["probes_ok"],
        len([k for k in report if not k.startswith("_")]),
        meta["elapsed_s"],
    )
    for family, entry in sorted(
        (k, v) for k, v in report.items() if not k.startswith("_")
    ):
        res = entry.get("resource") or {}
        logger.info(
            "family=%s live=%s n_live=%d n_dead=%d mem=%.1fMB(±%.1f) time=%.2fs",
            family,
            entry["live"],
            entry["n_live"],
            entry["n_dead"],
            res.get("peak_memory_mean", 0.0),
            res.get("peak_memory_std", 0.0),
            res.get("wall_time_mean", 0.0),
        )

    live = sorted(
        fam for fam, e in report.items() if not fam.startswith("_") and e["live"]
    )
    dead = sorted(
        fam
        for fam, e in report.items()
        if not fam.startswith("_") and e["live"] is False
    )
    undetermined = sorted(
        fam
        for fam, e in report.items()
        if not fam.startswith("_") and e["live"] is None
    )
    logger.info("LIVE families: %s", live)
    logger.info("DEAD families (auto-quarantined): %s", dead)
    if undetermined:
        logger.warning(
            "UNDETERMINED families (too few epochs for liveness): %s", undetermined
        )
    logger.info("report written to %s", out_path)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
