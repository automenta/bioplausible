"""Experiment campaign schema (architecture §6.1).

Migrated and rewritten from ``campaign/schema.py``: the thin experiment layer's
YAML contract. A campaign is an ordered list of **Stages** (the staircase), an
arm param budget, and compute/reproducibility settings. Geometry is inherited
from the task registry (``domains.registry.resolve_task``) — never redeclared.

Validation is strict:
* unknown task in any stage -> error,
* ``baseline:`` (evidence) stages reject ``seeds < 10``,
* parity evidence stages require ``matched_by`` and dual ``energy``.

Every runner field is required-or-defaulted here so the layer never falls back
to ``getattr(field, fallback)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from bioplausible.domains.registry import SUPPORTED_TASKS, resolve_task

__all__ = [
    "Arm",
    "Campaign",
    "Compute",
    "Meta",
    "MetricRule",
    "PassRule",
    "Reproducibility",
    "Stage",
    "Track",
    "load_campaign",
    "validate_yaml",
]

MIN_EVIDENCE_SEEDS = 10


class Track(BaseModel):
    """Measurement/energy tracking settings for a campaign."""

    model_config = ConfigDict(extra="forbid")

    flops: bool = True
    memory: bool = True
    energy: bool = False


class Compute(BaseModel):
    """Compute-resource settings."""

    model_config = ConfigDict(extra="forbid")

    device: str = "auto"
    num_workers: int = 0
    track: Track = Field(default_factory=Track)


class Meta(BaseModel):
    """Campaign identity and provenance."""

    model_config = ConfigDict(extra="forbid")

    name: str
    created: str = ""
    description: str = ""


class Arm(BaseModel):
    """One comparison group: a param budget and a model set."""

    model_config = ConfigDict(extra="forbid")

    max_params: int = Field(..., gt=0)
    models: list[str] = Field(min_length=1)


class MetricRule(BaseModel):
    """A single pass criterion over one metric (no eval)."""

    model_config = ConfigDict(extra="forbid")

    metric: Literal["acc", "epoch_time_s", "loss", "flops", "memory"]
    op: Literal[">=", "<=", ">", "<"]
    value: float
    aggregate: Literal["median", "mean", "min"] = "median"


class PassRule(BaseModel):
    """Structured pass criteria plus a minimum ok-seed count."""

    model_config = ConfigDict(extra="forbid")

    min_seed_ok: int = 1
    rules: list[MetricRule] = Field(default_factory=list)


class Reproducibility(BaseModel):
    """Reproducibility settings."""

    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    capture_env: bool = True


class MatchedBy(BaseModel):
    """Compute-matched contract for a parity stage (RESEARCH §0.1)."""

    model_config = ConfigDict(extra="forbid")

    equal_budget: Literal["max_params"] = "max_params"
    reported: list[str] = Field(default_factory=list)


class Stage(BaseModel):
    """One rung of the staircase: task + grid + pass rule + seed count."""

    model_config = ConfigDict(extra="forbid")

    name: str
    task: str
    epochs: int = Field(..., gt=0)
    seeds: int = Field(1, ge=1)
    configs: dict[str, list[object]] = Field(default_factory=dict)
    pass_rule: PassRule | None = None
    baseline: str | None = None
    matched_by: MatchedBy | None = None
    energy: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_task_and_evidence(
        self,
    ) -> Stage:  # descriptive validator messages are the public API
        if self.task not in SUPPORTED_TASKS:
            raise ValueError(
                f"stage {self.name!r}: unknown task {self.task!r} "
                f"(available: {sorted(SUPPORTED_TASKS)})"
            )
        if self.baseline is not None:
            if self.seeds < MIN_EVIDENCE_SEEDS:
                raise ValueError(
                    f"stage {self.name!r}: baseline evidence requires "
                    f"seeds >= {MIN_EVIDENCE_SEEDS} (got {self.seeds})"
                )
            if self.matched_by is None:
                raise ValueError(
                    f"stage {self.name!r}: baseline evidence requires matched_by"
                )
            if not self.energy:
                raise ValueError(
                    f"stage {self.name!r}: baseline evidence requires dual energy "
                    "(e.g. [gpu_tdp_x_util, op_count])"
                )
        return self


class Campaign(BaseModel):
    """Top-level validated experiment definition (ordered staircase)."""

    model_config = ConfigDict(extra="forbid")

    meta: Meta
    compute: Compute = Field(default_factory=Compute)
    arms: dict[str, Arm]
    stages: list[Stage] = Field(min_length=1)
    reproducibility: Reproducibility = Field(default_factory=Reproducibility)

    def geometry(self, task: str) -> tuple[int, int]:  # ruff: ignore[no-self-use]  (kept as a method for pydantic-model ergonomics)
        """Return the resolved (input_dim, output_dim) for ``task``."""
        spec = resolve_task(task)
        return spec.input_dim, spec.output_dim


def validate_yaml(text: str) -> Campaign:
    """Parse and validate a campaign YAML string into a :class:`Campaign`.

    Raises:
        yaml.YAMLError: on malformed YAML.
        ValueError: when the document is empty.
        ValidationError: when required fields are missing or invalid.
    """
    data = yaml.safe_load(text)
    if data is None:
        raise ValueError(  # descriptive message is the public API
            "Campaign YAML is empty; expected a mapping of settings"
        )
    return Campaign.model_validate(data)


def load_campaign(path: str | Path) -> Campaign:
    """Load and validate a campaign from a YAML file path."""
    return validate_yaml(Path(path).read_text(encoding="utf-8"))
