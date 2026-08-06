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
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from bioplausible.core.trainer import CoreTrainer, TrainerConfig

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
    ) -> None:
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.track_energy = track_energy
        self.track_flops = track_flops
        self.track_memory = track_memory
        self.batches_per_epoch = batches_per_epoch

    def train(  # ruff: ignore[too-many-arguments]  (probe driver signature is the public protocol contract)
        self,
        *,
        model: str,
        task: str,
        config: dict[str, object],
        seed: int,
        epochs: int,
        device: str,
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

        Returns:
            A metrics dict with ``final_acc``, ``epoch_time_s``, flops, memory.

        Raises:
            RuntimeError: If training raises or returns no history.
        """
        import bioplausible.zoo  # ruff: ignore[unused-import]  (registration side effect; mirrors cli/parity.py)
        from bioplausible.core.registry import ComponentCategory, Registry
        from bioplausible.domains.registry import resolve_task
        from bioplausible.experiment.param_estimator import build_model_kwargs
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
        core_train_flag = self.track_energy or self.track_flops or self.track_memory
        cfg = TrainerConfig(
            model=model,
            model_kwargs=model_kwargs,
            task=task,
            epochs=epochs,
            seed=seed,
            device=device,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # CoreTrainer's EnergyTracker computes flops+memory+energy under one
            # gate; enable it when the campaign asks for any of them so the
            # declared `compute.track` produces real values.
            track_energy=core_train_flag,
            track_flops=self.track_flops,
            track_memory=self.track_memory,
            batches_per_epoch=self.batches_per_epoch,
        )
        try:
            history = CoreTrainer(cfg).fit()
        except Exception as exc:  # broad: a broken model must not kill the gate
            raise RuntimeError(  # descriptive message is the public API
                f"probe {model}/{task} failed: {exc}"
            ) from exc
        if not history:
            raise RuntimeError(  # descriptive message is the public API
                f"probe {model}/{task} returned no history"
            )

        last = history[-1]
        total_time = sum(float(m.epoch_time or 0.0) for m in history)
        return {
            "final_acc": float(last.train_accuracy or last.val_accuracy or 0.0),
            "final_train_loss": float(last.train_loss or 0.0),
            "epoch_time_s": total_time,
            "forward_flops": int(last.forward_flops or 0),
            "backward_flops": int(last.backward_flops or 0),
            "peak_memory_mb": float(last.peak_memory_mb or 0.0),
            # peak_memory_mb is CUDA-only; wall_time_s is not populated by
            # CoreTrainer on CPU, so fall back to the summed epoch time so the
            # parity contract's `matched_by.reported: [wall_time_s]` is real.
            "wall_time_s": total_time,
        }


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

    Returns:
        A normalized :class:`ProbeResult` (status ``"ok"`` or ``"error"``).
    """
    try:
        metrics = driver.train(
            model=model,
            task=task,
            config=config,
            seed=seed,
            epochs=epochs,
            device=device,
        )
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
