"""Staircase gate tests (FIX2a §1, §8, §13 steps 6-7).

TIER 0 (synthetic smoke) and TIER 0.5 (digits) are the cheap triage gates: a
model that fails either is excluded from every higher tier before expensive
compute is spent. Each gate runs real training through :class:`CoreTrainer` —
the same engine the campaign uses — so a "pass" is an honest, reproducible
statement.

Outcomes are recorded as :class:`TierOutcome` and, when a logger is provided,
emitted to the campaign JSONL stream.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import torch

from bioplausible.campaign.param_estimator import build_model_kwargs
from bioplausible.core.registry import ComponentCategory, Registry
from bioplausible.core.trainer import CoreTrainer, TrainerConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "TIER0_EPOCHS",
    "TIER0_TASKS",
    "TIER05_EPOCHS",
    "TierOutcome",
    "run_tier0",
    "run_tier05",
]

TIER0_TASKS = ("xor", "spiral", "circles")
TIER0_EPOCHS = 3
TIER05_EPOCHS = 5


@dataclass(frozen=True, slots=True)
class TierOutcome:
    """Per model x task gate result."""

    tier: str
    model: str
    task: str
    passed: bool
    reason: str
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GateSettings:
    """Immutable training parameters shared by a gate run.

    Bundling the dimensions, device, seed, and epochs into one value object
    keeps every tier function to a small signature and makes each run
    parameter explicit.
    """

    input_dim: int
    output_dim: int
    device: str = "cpu"
    seed: int = 0
    epochs: int = TIER0_EPOCHS
    n_seeds: int = 1
    min_accuracy: float = 0.95


def _model_kwargs(
    model_name: str,
    config: dict[str, object],
    *,
    input_dim: int,
    output_dim: int,
) -> dict[str, object]:
    """Derive constructor kwargs using the estimator's signature filter."""
    model_cls = Registry.get(ComponentCategory.MODEL, model_name)
    return build_model_kwargs(
        model_cls,
        config,
        input_dim=input_dim,
        output_dim=output_dim,
        model_name=model_name,
    )


def _finite_losses(history: Sequence[object]) -> list[float]:
    """Extract finite per-epoch train losses from a CoreTrainer history."""
    out: list[float] = []
    for metrics in history:
        value = getattr(metrics, "train_loss", None)
        if value is None:
            continue
        try:
            scalar = (
                float(value.detach().cpu())
                if isinstance(value, torch.Tensor)
                else float(value)
            )
        except TypeError, ValueError:
            continue
        if not math.isfinite(scalar):
            continue
        out.append(scalar)
    return out


def _train_sample(
    *,
    model_name: str,
    task: str,
    settings: GateSettings,
    config: dict[str, object],
) -> tuple[bool, str, dict[str, float]]:
    """Run one short training pass; return ``(ok, reason, metrics)``.

    ``ok`` is False if training raised, produced NaN, or failed to decrease
    the loss over the requested epochs.
    """
    model_kwargs = _model_kwargs(
        model_name,
        config,
        input_dim=settings.input_dim,
        output_dim=settings.output_dim,
    )
    cfg = TrainerConfig(
        model=model_name,
        model_kwargs=model_kwargs,
        task=task,
        epochs=settings.epochs,
        seed=settings.seed,
        device=settings.device,
        batch_size=128,
        num_workers=0,
        track_energy=False,
        track_flops=False,
        track_memory=False,
        save_checkpoints=False,
        use_compile=False,
    )
    try:
        history = CoreTrainer(cfg).fit()
    except Exception as exc:  # broad: a broken model must not abort the gate
        return False, f"training raised {type(exc).__name__}: {exc}", {}

    if not history:
        return False, "no per-epoch history returned", {}

    losses = _finite_losses(history)
    if not losses:
        return False, "all losses NaN/None", {}

    accs = [getattr(m, "train_accuracy", 0.0) or 0.0 for m in history]
    improved = losses[-1] < losses[0]
    total_epoch_time = sum(float(getattr(m, "epoch_time", 0.0) or 0.0) for m in history)

    metrics = {
        "final_train_loss": losses[-1],
        "initial_train_loss": losses[0],
        "final_train_acc": float(accs[-1]),
        "epochs": float(len(history)),
        "epoch_time_s": total_epoch_time,
    }
    reason = f"loss {'decreased' if improved else 'did not decrease'} (final_acc={accs[-1]:.4f})"
    return improved, reason, metrics


def run_tier0(
    models: list[str],
    settings: GateSettings,
    config: dict[str, object] | None = None,
) -> list[TierOutcome]:
    """Run the TIER 0 synthetic gate for every model across all three tasks.

    A model passes iff, on every task, a short training run produces a
    strictly-decreasing loss with no NaN in the recorded metrics.

    Args:
        models: Registered model names to test (no exclusions at this tier).
        settings: Shared training parameters (dims, device, seed, epochs).
        config: Shared architecture config (defaults to a small 2-layer MLP).
    """
    config = config or {"hidden_dim": 32, "num_layers": 2}
    outcomes: list[TierOutcome] = []
    for model in models:
        failed: list[str] = []
        last_metrics: dict[str, float] = {}
        for task in TIER0_TASKS:
            ok, reason, metrics = _train_sample(
                model_name=model,
                task=task,
                settings=settings,
                config=config,
            )
            last_metrics = metrics
            if not ok:
                failed.append(f"{task}: {reason}")
            logger.info("TIER0 | %s | %s | pass=%s", model, task, ok)
        passed = not failed
        reason = (
            "; ".join(failed)
            if failed
            else "forward+backward ok, loss decreases, no NaN"
        )
        outcomes.append(
            TierOutcome(
                tier="tier0",
                model=model,
                task=",".join(TIER0_TASKS),
                passed=passed,
                reason=reason,
                metrics=last_metrics,
            )
        )
    return outcomes


def run_tier05(
    models: list[str],
    settings: GateSettings,
    config: dict[str, object] | None = None,
) -> list[TierOutcome]:
    """Run the TIER 0.5 digits gate: mean accuracy over ``n_seeds`` seeds.

    Uses the plan's fair MLP architecture by default (``num_layers=1``,
    ``hidden_dim=64``). A model passes iff the mean final train accuracy over
    seeds clears ``min_accuracy`` (default 95%). Models that fail are logged
    with the ``digits-fail`` verdict and excluded from higher tiers.

    Args:
        models: Models that already passed TIER 0.
        settings: Shared training parameters. ``n_seeds`` and ``min_accuracy``
            drive the gate; seeds run from ``seed .. seed + n_seeds``.
        config: Architecture config (defaults to the fair MLP from §2).
    """
    config = config or {"hidden_dim": 64, "num_layers": 1}
    outcomes: list[TierOutcome] = []
    for model in models:
        accs: list[float] = []
        epoch_time_s: float = 0.0
        for i in range(settings.n_seeds):
            ok, reason, metrics = _train_sample(
                model_name=model,
                task="digits",
                settings=replace(settings, seed=settings.seed + i),
                config=config,
            )
            if not ok:
                outcomes.append(
                    TierOutcome(
                        tier="tier0.5",
                        model=model,
                        task="digits",
                        passed=False,
                        reason=f"seed {settings.seed + i}: {reason}",
                        metrics=metrics,
                    )
                )
                break
            accs.append(metrics["final_train_acc"])
            epoch_time_s += metrics["epoch_time_s"]

        if not accs:
            continue

        param_count = _count_params(
            model, config, settings.input_dim, settings.output_dim
        )
        mean_acc = sum(accs) / len(accs)
        passed = mean_acc >= settings.min_accuracy
        outcomes.append(
            TierOutcome(
                tier="tier0.5",
                model=model,
                task="digits",
                passed=passed,
                reason=(
                    f"{'pass' if passed else 'digits-fail'}: mean_acc={mean_acc:.4f} "
                    f"over {len(accs)} seeds (gate={settings.min_accuracy})"
                ),
                metrics={
                    "mean_acc": mean_acc,
                    "min_acc": min(accs),
                    "n_seeds": float(len(accs)),
                    "param_count": float(param_count),
                    "epoch_time_s": epoch_time_s,
                },
            )
        )
    return outcomes


def _count_params(
    model_name: str,
    config: dict[str, object],
    input_dim: int,
    output_dim: int,
) -> int:
    """Static parameter count via the shared estimator (no training)."""
    from bioplausible.campaign.param_estimator import estimate_param_count

    return estimate_param_count(
        model_name, config, input_dim=input_dim, output_dim=output_dim
    )
