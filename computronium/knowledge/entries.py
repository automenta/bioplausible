"""
Knowledge base entry types and query/result data structures.
"""

import time
from dataclasses import asdict, dataclass, field
from typing import TypedDict

from pydantic import BaseModel, Field


@dataclass(frozen=True, slots=True)
class KnowledgeEntry:
    """A single knowledge entry with metadata and optional embedding."""

    id: str
    topic: str
    model_family: str
    finding: str
    details: str
    confidence: float
    tags: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    source: str = "manual"  # "manual", "experiment", "surrogate", "causal"
    experiment_id: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    hyperparameters: dict[str, object] = field(default_factory=dict)
    embedding: list[float] | None = None
    extra: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        d = asdict(self)
        # Don't store embedding in JSON
        d.pop("embedding", None)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> KnowledgeEntry:
        ann = cls.__annotations__
        kwargs = {}
        for k, v in d.items():
            if k in ann:
                expected = ann[k]
                # Handle type conversions for common types
                if expected == str and v is not None:
                    kwargs[k] = str(v)
                elif expected == float and v is not None:
                    kwargs[k] = float(v)
                elif expected == list[str] and isinstance(v, list):
                    kwargs[k] = [str(x) for x in v]
                elif expected == dict[str, float] and isinstance(v, dict):
                    kwargs[k] = {str(k2): float(v2) for k2, v2 in v.items()}
                elif expected == dict[str, object] and isinstance(v, dict):
                    kwargs[k] = v
                elif expected == list[float] | None and isinstance(v, list):
                    kwargs[k] = [float(x) for x in v]
                else:
                    kwargs[k] = v
        return cls(**kwargs)


class ConditionalQuery(TypedDict):
    """Read-half filter for the AutoScientist flywheel (P2).

    A request for previously-verified positive conditionals: which rules already
    achieved a target accuracy within memory/flops caps on a task (optionally on
    a specific substrate). Empty/None fields act as wildcards.
    """

    model: str | None
    task: str | None
    accuracy_target: float | None
    memory_cap: float | None
    flops_cap: float | None
    substrate: str | None


class _ConditionalQueryModel(BaseModel):
    """Pydantic v2 runtime-validated form of :class:`ConditionalQuery`."""

    model: str | None = Field(default=None)
    task: str | None = Field(default=None)
    accuracy_target: float | None = Field(default=None, ge=0.0, le=1.0)
    memory_cap: float | None = Field(default=None, ge=0.0)
    flops_cap: float | None = Field(default=None, ge=0.0)
    substrate: str | None = Field(default=None)


@dataclass(frozen=True, slots=True)
class ConditionalResult:
    """One previously-verified positive conditional satisfying a query.

    A frozen value object so the proposer can reason about *already-spent*
    budget without mutating the KB or the caller's data.
    """

    model: str
    task: str
    accuracy: float
    memory_mb: float
    flops: float
    wall_time_s: float
    config: tuple[tuple[str, object], ...] = ()
    substrate: str | None = None
    entry_id: str | None = None


@dataclass(frozen=True, slots=True)
class FlagshipCandidate:
    """One validated family's cost-of-plausibility operating point (P3a)."""

    model: str
    accuracy: float
    memory_mb: float
    flops: float
    wall_time_s: float
    cost_of_plausibility: float
    substrate: str | None = None


@dataclass(frozen=True, slots=True)
class FlagshipDecision:
    """The outcome of the P3a flagship-selection query."""

    task: str
    chosen: str | None
    ranked: tuple[FlagshipCandidate, ...]


def _entry_accuracy(metrics: dict[str, float]) -> float:
    """Normalize accuracy across the probe/engine metric dialects."""
    return float(metrics.get("final_acc", metrics.get("accuracy", 0.0)))


def _entry_memory(metrics: dict[str, float]) -> float:
    return float(metrics.get("peak_memory_mb", metrics.get("memory_mb", 0.0)))


def _entry_flops(metrics: dict[str, float]) -> float:
    return float(metrics.get("forward_flops", 0.0) + metrics.get("backward_flops", 0.0))


def _entry_substrate(entry: KnowledgeEntry) -> str | None:
    """Substrate tag, if one was recorded (plan §17 IBKB key)."""
    hw = entry.extra.get("hardware") if isinstance(entry.extra, dict) else None
    return str(hw) if hw else None


def _entry_task(entry: KnowledgeEntry) -> str:
    """Task name persisted through ``add_experiment``'s ``extra`` dict."""
    meta = entry.extra.get("task") if isinstance(entry.extra, dict) else None
    return str(meta) if meta else ""


__all__ = [
    "ConditionalQuery",
    "ConditionalResult",
    "FlagshipCandidate",
    "FlagshipDecision",
    "KnowledgeEntry",
    "_ConditionalQueryModel",
    "_entry_accuracy",
    "_entry_memory",
    "_entry_flops",
    "_entry_substrate",
    "_entry_task",
]
