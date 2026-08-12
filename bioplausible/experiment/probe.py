"""Probe submission and normalization (architecture §6.2, §6.4).

A **probe** is one ``(model, task, config, seed)`` training run. The layer's
:class:`ProbeDriver` is a thin adapter over the existing training path
(default ``CoreTrainer`` via ``cli``); :func:`run_probe` is the single point
where a probe's per-seed record is normalized once into a
:class:`ProbeResult`.

``run_verify`` (existing in ``cli/run.py``) already emits per-seed JSONL with
CI/effect-size metadata; this module consumes that record shape rather than
re-implementing a training loop.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from bioplausible.core.trainer import CoreTrainer, TrainerConfig

# Whether probes persist results to the knowledge layer (KnowledgeBase /
# FailureTracker) by default. Environment-controllable so tests can isolate.
_DEFAULT_RECORD = os.environ.get("BIOPLAUSIBLE_RECORD_RESULTS", "1") != "0"

__all__ = [
    "CoreTrainerDriver",
    "ProbeDriver",
    "ProbeResult",
    "config_key",
    "run_probe",
]

_EXCLUDED_CONFIG_KEYS = frozenset({
    "epochs",
    "batch_size",
    "tier",
    "is_verification",
    "verified_trial_id",
    "seed",
})


def config_key(config: dict[str, object]) -> str:
    """Return a content hash of a config for idempotence.

    Excludes run-control keys (epochs/seed/batch) so the same architecture
    config across two runs maps to the same key — the resume index matches on
    it. The key is order-independent.
    """
    canonical = {k: config[k] for k in config if k not in _EXCLUDED_CONFIG_KEYS}
    blob = json.dumps(canonical, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _dominant_training_path(
    paths: object,
) -> str:
    """Return the most-frequent credit-assignment path observed, or ``""``.

    ``paths`` is the per-epoch ``training_paths`` dict recorded by
    ``CoreTrainer`` (path name → step count). The dominant path is the probe
    headline; the full map is preserved separately as ``training_paths``.
    """
    if not isinstance(paths, dict) or not paths:
        return ""
    return max(paths.items(), key=lambda kv: int(kv[1]))[0]


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Normalized per-probe metrics record (architecture §6.2)."""

    model: str
    task: str
    config: dict[str, object]
    config_key: str
    seed: int
    status: str  # "ok" | "error"
    final_acc: float = 0.0
    final_train_loss: float = 0.0
    epoch_time_s: float = 0.0
    param_count: int = 0
    forward_flops: int = 0
    backward_flops: int = 0
    peak_memory_mb: float = 0.0
    wall_time_s: float = 0.0
    training_path: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, object]:
        """Serialize to a plain dict for JSONL output."""
        return field_to_dict(self)


def field_to_dict(result: ProbeResult) -> dict[str, object]:
    """Serialize a :class:`ProbeResult` to a JSON-compatible dict."""
    config = dict(result.config)
    return {
        "model": result.model,
        "task": result.task,
        "config": config,
        "config_key": result.config_key,
        "seed": result.seed,
        "status": result.status,
        "final_acc": result.final_acc,
        "final_train_loss": result.final_train_loss,
        "epoch_time_s": result.epoch_time_s,
        "param_count": result.param_count,
        "forward_flops": result.forward_flops,
        "backward_flops": result.backward_flops,
        "peak_memory_mb": result.peak_memory_mb,
        "wall_time_s": result.wall_time_s,
        "training_path": result.training_path,
        "error": result.error,
    }


@runtime_checkable
class ProbeDriver(Protocol):
    """Narrow adapter over the existing training path (architecture §6.4)."""

    def train(  # ruff: ignore[too-many-arguments]  (probe driver signature is the public protocol contract)
        self,
        *,
        model: str,
        task: str,
        config: dict[str, object],
        seed: int,
        epochs: int,
        device: str,
        propagator: str | None = None,
    ) -> dict[str, object]: ...


class CoreTrainerDriver:
    """Drives a probe through ``CoreTrainer`` (the existing training path).

    Compute settings (worker count, tracking toggles) come from the campaign's
    ``compute`` block and are threaded into every ``TrainerConfig`` so a probe
    respects the operator's declared resource budget — e.g. ``num_workers: 0``
    on a bulk overnight run spawns no DataLoader worker processes per probe.
    """

    def __init__(  # ruff: ignore[too-many-arguments]  # driver constructor captures all campaign compute settings at once
        self,
        *,
        num_workers: int = 0,
        batch_size: int = 64,
        track_energy: bool = False,
        track_flops: bool = True,
        track_memory: bool = True,
        batches_per_epoch: int | None = None,
        record_results: bool = _DEFAULT_RECORD,
        target_hardware: str | None = None,
        allow_bptt_fallback: bool = True,
        max_epoch_time: float = 0.0,
    ) -> None:
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.track_energy = track_energy
        self.track_flops = track_flops
        self.track_memory = track_memory
        self.batches_per_epoch = batches_per_epoch
        self.record_results = record_results
        self.target_hardware = target_hardware
        self.allow_bptt_fallback = allow_bptt_fallback
        self.max_epoch_time = max_epoch_time

    def train(  # ruff: ignore[too-many-arguments]  (probe driver signature is the public protocol contract)
        self,
        *,
        model: str,
        task: str,
        config: dict[str, object],
        seed: int,
        epochs: int,
        device: str,
        propagator: str | None = None,
        allow_bptt_fallback: bool | None = None,
    ) -> dict[str, object]:
        """Train one probe and return aggregated metrics.

        Uses ``TrainerConfig`` so the run follows the exact CoreTrainer path
        (registration, data loading, tracking) used by the parity CLI. Compute
        settings captured at construction (worker count, tracking) are applied.

        Args:
            model: Registered model name.
            task: Registered task name.
            config: Architecture config (hidden_dim, num_layers, ...).
            seed: Master seed.
            epochs: Training epochs.
            device: Target device.
            propagator: Registered learning-rule propagator (e.g.
                ``"feedback_alignment"``, ``"contrastive_hebbian_learning"``).
                When set, the trainer drives this rule instead of letting the
                model degrade to plain BPTT — required so a bio-rule probe
                measures *local* cost, not backprop cost.

        Returns:
            A metrics dict with ``final_acc``, ``epoch_time_s``, flops, memory.

        Raises:
            RuntimeError: If training raises or returns no history.
        """
        import bioplausible.zoo  # ruff: ignore[unused-import]  (registration side effect; mirrors cli/parity.py)
        from bioplausible.core.exceptions import NumericalInstabilityError
        from bioplausible.core.registry import ComponentCategory, Registry
        from bioplausible.domains.registry import resolve_task
        from bioplausible.experiment.param_estimator import (
            build_model_kwargs,
            estimate_param_count,
            phantom_knobs,
        )
        from bioplausible.utils import seed_everything

        seed_everything(seed, device)
        spec = resolve_task(task)
        model_cls = Registry.get(ComponentCategory.MODEL, model)
        model_kwargs = build_model_kwargs(
            model_cls,
            config,
            input_dim=spec.input_dim,
            output_dim=spec.output_dim,
            model_name=model,
        )
        # Phantom-drift diagnosis: sampled tuning knobs the model cannot consume
        # (surfaced as a sweep defect instead of silently ignored).
        phantom = sorted(
            phantom_knobs(
                model_cls,
                config,
                input_dim=spec.input_dim,
                output_dim=spec.output_dim,
                model_name=model,
            )
        )
        # Static parameter count under this config (fair-comparison budget).
        try:
            param_count = estimate_param_count(
                model,
                config,
                input_dim=spec.input_dim,
                output_dim=spec.output_dim,
            )
        except Exception:  # defensive: counting must never break a probe
            param_count = 0
        # Thread the sampled learning rate into the trainer's optimizer so
        # trainer-driven (BPTT) models respect it — self-training models already
        # read it from their own ``config``. The scalar is carried in
        # ``model_kwargs`` (the OmegaConf-safe view), never a nested object.
        learn_rate = model_kwargs.get("learning_rate")
        opt_kwargs: dict[str, object] = (
            {"lr": float(learn_rate)} if learn_rate is not None else {}
        )
        core_train_flag = self.track_energy or self.track_flops or self.track_memory
        cfg = TrainerConfig(
            model=model,
            model_kwargs=model_kwargs,
            task=task,
            epochs=epochs,
            seed=seed,
            device=device,
            propagator=propagator,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            optimizer_kwargs=opt_kwargs,
            allow_bptt_fallback=(
                self.allow_bptt_fallback
                if allow_bptt_fallback is None
                else allow_bptt_fallback
            ),
            # Probes are disposable resource measurements: they must not write
            # resumable checkpoints to disk (that path is for the settle-state
            # memory lever / long runs, not shallow probes).
            save_checkpoints=False,
            # CoreTrainer's EnergyTracker computes flops+memory+energy under one
            # gate; enable it when the campaign asks for any of them so the
            # declared `compute.track` produces real values.
            track_energy=core_train_flag,
            track_flops=self.track_flops,
            track_memory=self.track_memory,
            batches_per_epoch=self.batches_per_epoch,
            max_epoch_time=self.max_epoch_time,
            target_hardware=self.target_hardware,
        )
        try:
            history = CoreTrainer(cfg).fit()
        except NumericalInstabilityError as exc:
            self._record(
                model=model,
                task=task,
                config=config,
                status="error",
                extra={"error": str(exc), "defect": "nan_divergence"},
                seed=seed,
                device=device,
            )
            raise RuntimeError(f"probe {model}/{task} diverged: {exc}") from exc
        except Exception as exc:  # broad: a broken model must not kill the gate
            self._record(
                model=model,
                task=task,
                config=config,
                status="error",
                extra={"error": str(exc)},
                seed=seed,
                device=device,
            )
            raise RuntimeError(  # descriptive message is the public API
                f"probe {model}/{task} failed: {exc}"
            ) from exc
        if not history:
            self._record(
                model=model,
                task=task,
                config=config,
                status="error",
                extra={"error": "no history"},
                seed=seed,
                device=device,
            )
            raise RuntimeError(  # descriptive message is the public API
                f"probe {model}/{task} returned no history"
            )

        last = history[-1]
        total_time = sum(float(m.epoch_time or 0.0) for m in history)
        last_extra = getattr(last, "extra", {}) or {}
        # Liveness-gate endpoints (plan §5 cycle 1): the broad sweep marks a
        # rule "dead" iff loss does not decrease across the run. Expose both
        # ends so the sweep (and the KB sink) can gate without the full series.
        loss_0 = float(history[0].train_loss or 0.0)
        loss_final = float(last.train_loss or 0.0)
        # Convergence diagnostic: max accuracy over the run and accuracy at the
        # halfway epoch. A rule with low `final_acc` but a rising, non-flat
        # trajectory (best_epoch_acc >> final_acc, or acc_at_half << final_acc)
        # is *mid-convergence* — a training-budget (epochs) issue — not a model
        # failure. This distinguishes "needs more epochs" from "never learns".
        accs = [float(m.train_acc or 0.0) for m in history if m.train_acc]
        half_idx = max(1, len(accs) // 2) if accs else 0
        metrics = {
            "final_acc": float(last.train_acc or last.val_acc or 0.0),
            "final_train_loss": float(last.train_loss or 0.0),
            "epoch_time_s": total_time,
            "param_count": param_count,
            "forward_flops": int(last.forward_flops or 0),
            "backward_flops": int(last.backward_flops or 0),
            "peak_memory_mb": float(last.peak_memory_mb or 0.0),
            # peak_memory_mb is CUDA-only; wall_time_s is not populated by
            # CoreTrainer on CPU, so fall back to the summed epoch time so the
            # parity contract's `matched_by.reported: [wall_time_s]` is real.
            "wall_time_s": total_time,
            "best_epoch_acc": max(accs) if accs else float(last.train_acc or 0.0),
            "acc_at_half": float(accs[half_idx - 1])
            if accs and half_idx
            else float(last.train_acc or 0.0),
            "loss_epoch_0": loss_0,
            "loss_epoch_final": loss_final,
            # Self-diagnosis (EXPERIMENT_PLAN5 §1): the credit-assignment path
            # actually used by this probe (energy | model_train_step |
            # propagator | bptt). A bio-family probe reporting "bptt" is a
            # silent-fallback defect surfaced without human audit.
            "training_paths": dict(last_extra.get("training_paths") or {}),
            "training_path": _dominant_training_path(last_extra.get("training_paths")),
            # Epoch-time truncation: if any epoch was cut short by the
            # ``max_epoch_time`` budget, the run's resource metrics are over a
            # partial epoch — not comparable to full-epoch runs. The sweep must
            # prune (flag as defect) such a run, not average partial stats in.
            "epoch_time_budget_stopped": bool(
                any(
                    bool(m.extra.get("epoch_time_budget_stopped"))
                    for m in history
                    if getattr(m, "extra", None)
                )
            ),
            # Phantom-drift diagnosis: sampled tuning knobs this probe could not
            # deliver to the model. A non-empty list is a self-diagnosis defect
            # flagged by the sweep (the config advertised knobs that had no
            # consumer — reported, never silently ignored).
            "phantom_knobs": phantom,
            # Hardware-aware fields (plan §17): present only when the trainer
            # swapped in a substrate facade via TrainerConfig.target_hardware.
            "target_hardware": last_extra.get("target_hardware"),
            "bits": last_extra.get("bits"),
            "noise_level": last_extra.get("noise_level"),
        }
        self._record(
            model=model,
            task=task,
            config=config,
            status="completed",
            metrics=metrics,
            seed=seed,
            device=device,
        )
        return metrics

    def _record(
        self,
        *,
        model: str,
        task: str,
        config: dict[str, object],
        status: str,
        metrics: dict[str, object] | None = None,
        extra: dict[str, object] | None = None,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        """Persist a probe outcome to the knowledge layer (best-effort).

        Recording must never break a probe: a DB/embedding failure is logged
        and swallowed. Gated by ``record_results`` so tests can disable it.
        """
        if not self.record_results:
            return
        try:
            from bioplausible.experiment.result_sink import record_experiment_result

            record_experiment_result(
                model=model,
                task=task,
                config=config,
                metrics=metrics,
                status=status,
                seed=seed,
                device=device,
                extra=extra,
            )
        except Exception as exc:  # pragma: no cover  # best-effort persistence
            from bioplausible.core.logging import get_logger

            get_logger().error(
                "result_sink recording failed for %s/%s: %s", model, task, exc
            )


def run_probe(  # ruff: ignore[too-many-arguments]  (one normalization entrypoint carries all probe identity + parameters)
    driver: ProbeDriver,
    *,
    model: str,
    task: str,
    config: dict[str, object],
    seed: int,
    epochs: int,
    device: str,
    param_count: int = 0,
    propagator: str | None = None,
) -> ProbeResult:
    """Run one probe and normalize the outcome into a :class:`ProbeResult`.

    This is the single normalization point for the layer: every probe —
    scheduled by the producer, executed by the driver — becomes a
    :class:`ProbeResult`. Parameter count comes from
    ``experiment.param_estimator`` (passed in by the scheduler).

    Args:
        driver: The :class:`ProbeDriver` to execute training.
        model: Registered model name.
        task: Registered task name.
        config: Architecture config.
        seed: Seed for this probe.
        epochs: Training epochs.
        device: Target device.
        param_count: Static parameter count (from the estimator).
        propagator: Registered learning-rule propagator for bio-rule probes.

    Returns:
        A normalized :class:`ProbeResult` (status ``"ok"`` or ``"error"``).
    """
    try:
        call_kwargs: dict[str, object] = {
            "model": model,
            "task": task,
            "config": config,
            "seed": seed,
            "epochs": epochs,
            "device": device,
        }
        # ``propagator`` is an optional learning-rule override: only forward it
        # when set, so drivers that represent the default (no-rule) probe path
        # do not need to declare a keyword they never use.
        if propagator is not None:
            call_kwargs["propagator"] = propagator
        metrics = driver.train(**call_kwargs)
        return ProbeResult(
            model=model,
            task=task,
            config=config,
            config_key=config_key(config),
            seed=seed,
            status="ok",
            final_acc=float(metrics.get("final_acc", 0.0)),
            final_train_loss=float(metrics.get("final_train_loss", 0.0)),
            epoch_time_s=float(metrics.get("epoch_time_s", 0.0)),
            param_count=param_count,
            forward_flops=int(metrics.get("forward_flops", 0)),
            backward_flops=int(metrics.get("backward_flops", 0)),
            peak_memory_mb=float(metrics.get("peak_memory_mb", 0.0)),
            wall_time_s=float(metrics.get("wall_time_s", 0.0)),
            training_path=str(metrics.get("training_path", "")),
        )
    except Exception as exc:  # broad: normalize any probe failure
        return ProbeResult(
            model=model,
            task=task,
            config=config,
            config_key=config_key(config),
            seed=seed,
            status="error",
            error=str(exc),
        )
