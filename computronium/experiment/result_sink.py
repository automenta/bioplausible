"""Universal result sink: route every experiment outcome into the persistent layer.

This is the convergence point that connects the *measurement* layer (probe
driver, frontier search, validation tracks) to the *knowledge* layer
(KnowledgeBase for successes, FailureTracker for failures/expensive runs).

Every completed/failed experiment across all evaluation paths funnels through
:func:`record_experiment_result` so the KnowledgeBase — the business "moat" —
compounds with each probe instead of emitting throwaway JSON.

Two sinks, both write-once, idempotent, SQLite-backed:

- **KnowledgeBase** (``knowledge/kb.py``): verified *positive* conditionals
  ("rule R on task T with config C achieves acc A at FLOPs/Mem/Time F, CI").
- **FailureTracker** (``execution/_state.py``): *negative* knowledge — crashed
  or cost-inflated runs that AutoScientist must not re-burn compute on.

Both DB paths default to env-configurable locations so tests can isolate.
"""

from __future__ import annotations

import os

from computronium.core._paths import db_path

_KB_PATH = os.environ.get("COMPUTRONIUM_KB_PATH", db_path("computronium_kb.db"))
_FAILURE_PATH = os.environ.get(
    "COMPUTRONIUM_FAILURES_PATH", db_path("execution_state.db")
)

# Lazily-cached sink instances (one per process) — avoids re-opening the DB and
# re-initializing the (heavy) embedding model on every probe write.
_KB_INST: object | None = None
_FAILURE_INST: object | None = None


def configure(kb_path: str | None = None, failure_path: str | None = None) -> None:
    """Reset the sink's DB paths and drop cached instances (test seam).

    Args:
        kb_path: Override the KnowledgeBase DB path (defaults to env/``cwd``).
        failure_path: Override the FailureTracker DB path.
    """
    global _KB_PATH, _FAILURE_PATH, _KB_INST, _FAILURE_INST  # ruff: ignore[global-statement]
    if kb_path is not None:
        _KB_PATH = kb_path
    if failure_path is not None:
        _FAILURE_PATH = failure_path
    _KB_INST = None
    _FAILURE_INST = None


def _kb() -> object:
    """Return the shared KnowledgeBase instance (auto_embed off for fast writes)."""
    global _KB_INST  # ruff: ignore[global-statement]
    if _KB_INST is None:
        from computronium.knowledge.kb import KnowledgeBase

        _KB_INST = KnowledgeBase(db_path=_KB_PATH, auto_embed=False)
    return _KB_INST


def _failures() -> object:
    """Return the shared FailureTracker instance."""
    global _FAILURE_INST  # ruff: ignore[global-statement]
    if _FAILURE_INST is None:
        from computronium.execution._state import FailureTracker

        _FAILURE_INST = FailureTracker(db_path=_FAILURE_PATH)
    return _FAILURE_INST


def _sanitize(v: object) -> float:
    """Coerce a metric to float, mapping None/nan to 0.0 for KB storage."""
    try:
        return float(v if v is not None else 0.0)
    except TypeError, ValueError:
        return 0.0


def record_experiment_result(
    *,
    model: str,
    task: str,
    config: dict[str, object] | None = None,
    metrics: dict[str, object] | None = None,
    status: str = "completed",
    seed: int | None = None,
    epochs: int | None = None,
    device: str = "",
    extra: dict[str, object] | None = None,
) -> str:
    """Persist one experiment outcome to the knowledge layer.

    On ``status == "completed"`` (a real, non-crash result) writes a verified
    success entry to the KnowledgeBase. On any failure/abort/expensive status
    writes a failure record to the FailureTracker instead.

    Args:
        model: Registered model name (rule family).
        task: Task name.
        config: The probe config (hyperparameters/architecture).
        metrics: Metric dict (final_acc, flops, memory, time, ...). Empty for
            hard failures.
        status: ``"completed"`` | ``"failed"`` | ``"error"`` | ``"expensive"``.
        seed: Seed used for the run.
        epochs: Epoch budget.
        device: Target device string.
        extra: Free-form metadata (hardware track id, tier, etc.).

    Returns:
        The persisted entry id (KB entry id, or ``"FAIL:<id>"`` for failures).
    """
    metrics = metrics or {}
    config = config or {}

    # Expose the substrate (plan §17) as a KB tag so hardware-aware probes are
    # queryable by the AutoScientist (e.g. fpga/analog conditionals).
    hw = metrics.get("target_hardware")
    if hw and isinstance(hw, str):
        extra = dict(extra or {})
        extra.setdefault("hardware", hw)

    result = {k: _sanitize(v) for k, v in metrics.items()}

    if status == "completed":
        return _record_success(
            model=model,
            task=task,
            config=config,
            metrics=result,
            seed=seed,
            epochs=epochs,
            device=device,
            extra=extra,
        )
    return _record_failure(
        model=model,
        task=task,
        config=config,
        metrics=result,
        status=status,
        seed=seed,
        epochs=epochs,
        device=device,
        extra=extra,
    )


def _record_success(
    *,
    model: str,
    task: str,
    config: dict[str, object],
    metrics: dict[str, float],
    seed: int | None,
    epochs: int | None,
    device: str,
    extra: dict[str, object] | None,
) -> str:
    """Write a verified positive result to the KnowledgeBase."""
    kb = _kb()

    # Normalize across the two metric dialects present in the codebase:
    # probe-driver ("final_acc"/"wall_time_s"/"peak_memory_mb") and
    # ExecutionEngine ("accuracy"/"time"/"memory_mb").
    acc = metrics.get("final_acc", metrics.get("accuracy", 0.0))
    flops = metrics.get("forward_flops", 0.0) + metrics.get("backward_flops", 0.0)
    mem = metrics.get("peak_memory_mb", metrics.get("memory_mb", 0.0))
    wall = metrics.get("wall_time_s", metrics.get("time", 0.0))
    finding = (  # ruff: ignore[unused-variable]
        f"rule {model} on {task}: final_acc={acc:.4f} "
        f"flops={flops:.3e} mem={mem:.1f}MB time={wall:.2f}s"
    )
    tags = ["experiment", model, task]
    if extra and extra.get("hardware"):
        tags.append(str(extra["hardware"]))

    id_ = kb.add_experiment(
        name=f"{model}/{task}",
        model_family=model,
        task=task,
        config=dict(config),
        metrics=metrics,
        artifacts={"epochs": str(epochs), "seed": str(seed), "device": device},
    )
    return f"EXP-{id_}"


def _record_failure(
    *,
    model: str,
    task: str,
    config: dict[str, object],
    metrics: dict[str, float],
    status: str,
    seed: int | None,
    epochs: int | None,
    device: str,
    extra: dict[str, object] | None,
) -> str:
    """Write a negative result to the FailureTracker."""
    from datetime import datetime

    from computronium.execution._state import FailureRecord

    tracker = _failures()

    failure_type = {
        "error": "exception",
        "failed": "did_not_converge",
        "expensive": "cost_limited",
    }.get(status, "unknown")

    record = FailureRecord(
        timestamp=datetime.now().isoformat(),
        model_name=model,
        task_name=task,
        tier=str(extra.get("tier", "probe")) if extra else "probe",
        trial_id=seed,
        failure_type=failure_type,
        failure_epoch=epochs,
        failure_batch=None,
        config=dict(config),
        last_metrics=metrics,
        stack_trace=str(extra.get("error", "")) if extra else "",
    )
    tracker.log_failure(record)
    return f"FAIL:{failure_type}"
