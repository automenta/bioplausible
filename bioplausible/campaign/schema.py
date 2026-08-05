"""Pydantic v2 YAML campaign schema (FIX2a §3, §13 step 2).

One validated YAML file per campaign is the single source of truth for every
experiment setting — "If it's not in the YAML, it's not configurable." This
module parses and validates that file and exposes a typed, immutable model
hierarchy.

List-form shorthand for distributions:

* ``[lo, hi]`` — continuous (integer if both bounds are integral) range.
* ``[lo, hi, "log"|"linear"|"int"]`` — explicit scale.
* any other list — categorical choice.
* any scalar — a fixed constant (e.g. ``gradient_method: equilibrium``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from bioplausible.campaign.search_space import SearchSpace, parse_distribution

__all__ = [
    "HPO",
    "Arm",
    "Campaign",
    "Compute",
    "Meta",
    "Output",
    "Pareto",
    "Protocols",
    "Reproducibility",
    "Resources",
    "SearchSpaceConfig",
    "Task",
    "load_campaign",
    "validate_yaml",
]


class Meta(BaseModel):
    """Campaign identity and provenance metadata."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    created: str = "2026-01-01"
    git_commit: str | None = None


class Compute(BaseModel):
    """Compute-resource settings for the campaign."""

    model_config = ConfigDict(extra="forbid")

    device: str = "auto"
    max_parallel: int = Field(1, ge=1)
    max_wall_hours: float | None = Field(None, gt=0)


class Arm(BaseModel):
    """One architecture-separated comparison group (MLP vs Conv).

    Each arm carries its own param budget and input geometry, so a conv arm
    can spend more than an MLP arm without conflating credit assignment with
    capacity (FIX2a §5.2).
    """

    model_config = ConfigDict(extra="forbid")

    input_dim: int | None = None
    input_shape: list[int] | None = None
    num_classes: int | None = None
    flatten: bool = True
    max_params: int = Field(..., gt=0)
    models: list[str] = Field(min_length=1)


class SearchSpaceConfig(BaseModel):
    """Shared tunable hyperparameter bounds (distributions or constants)."""

    model_config = ConfigDict(extra="forbid")

    base: dict[str, Any] = Field(default_factory=dict)


class Protocols(BaseModel):
    """Training protocol per model (end2end / layerwise / spiking_surrogate).

    Each model x protocol combo spawns a sub-study; results are reported
    side-by-side (FIX2a §5.4).
    """

    model_config = ConfigDict(extra="forbid")

    default: str = "end2end"
    overrides: dict[str, str] = Field(default_factory=dict)

    def resolve(self, model_name: str) -> str:
        """Return the protocol for ``model_name`` (override or default)."""
        return self.overrides.get(model_name, self.default)


class Task(BaseModel):
    """A task specification for the staircase tiers."""

    model_config = ConfigDict(extra="forbid")

    name: str
    epochs: int = Field(..., gt=0)
    input_dim: int | None = None
    num_classes: int | None = None
    flatten: bool = True


class Pareto(BaseModel):
    """Pareto-frontier handling configuration."""

    model_config = ConfigDict(extra="forbid")

    knee_point: bool = True
    epsilon: float = Field(0.01, ge=0.0)


class HPO(BaseModel):
    """Multi-objective hyperparameter optimization settings."""

    model_config = ConfigDict(extra="forbid")

    sampler: str = "nsga2"
    objectives: list[str] = Field(
        default_factory=lambda: ["accuracy", "param_count", "epoch_time_s"]
    )
    directions: list[Literal["maximize", "minimize"]] | None = None
    n_trials: int = Field(200, ge=1)
    n_startup_trials: int = Field(10, ge=0)
    n_seeds: int = Field(5, ge=1)
    prune_worse_than_pareto: bool = True
    pareto: Pareto = Field(default_factory=Pareto)


class Resources(BaseModel):
    """Resource budgets and early-pruning knobs."""

    model_config = ConfigDict(extra="forbid")

    max_wall_hours: float | None = Field(None, gt=0)
    max_epoch_time_sec: float | None = Field(None, gt=0)
    early_stop_patience: int | None = Field(None, ge=1)


class Output(BaseModel):
    """Output artifact configuration."""

    model_config = ConfigDict(extra="forbid")

    db: str = "results/campaign.db"
    artifacts_dir: str = "artifacts/campaign"
    log_level: str = "INFO"
    emit_every: int = Field(10, ge=1)


class Reproducibility(BaseModel):
    """Reproducibility settings."""

    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    capture_env: bool = True
    artifact_hash: bool = True


class Campaign(BaseModel):
    """Top-level validated campaign definition."""

    model_config = ConfigDict(extra="forbid")

    meta: Meta
    compute: Compute = Field(default_factory=Compute)
    search_space: SearchSpaceConfig = Field(default_factory=SearchSpaceConfig)
    model_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)
    arms: dict[str, Arm]
    protocols: Protocols = Field(default_factory=Protocols)
    tasks: list[Task] = Field(default_factory=list)
    hpo: HPO = Field(default_factory=HPO)
    resources: Resources = Field(default_factory=Resources)
    output: Output = Field(default_factory=Output)
    reproducibility: Reproducibility = Field(default_factory=Reproducibility)

    def build_search_space(self) -> SearchSpace:
        """Assemble the executable :class:`SearchSpace` from the YAML values.

        Scalars in ``search_space.base``/``model_overrides`` become fixed
        constants; lists become tunable distributions.
        """
        base_dists: dict[str, Any] = {}
        base_constants: dict[str, object] = {}
        for key, value in self.search_space.base.items():
            if isinstance(value, list):
                base_dists[key] = parse_distribution(value)
            else:
                base_constants[key] = value

        overrides: dict[str, dict[str, Any]] = {}
        constants: dict[str, dict[str, object]] = {}
        for model_name, mapping in self.model_overrides.items():
            dists: dict[str, Any] = {}
            fixed: dict[str, object] = {}
            for key, value in mapping.items():
                if isinstance(value, list):
                    dists[key] = parse_distribution(value)
                else:
                    fixed[key] = value
            if dists:
                overrides[model_name] = dists
            if fixed:
                constants[model_name] = fixed

        return SearchSpace(
            base=base_dists,
            overrides=overrides,
            defaults=base_constants,
            constants=constants,
            constraints=tuple(self.constraints),
        )

    def arm_input_dim(self, arm_name: str) -> int:
        """Resolve the scalar input dimension for an arm.

        Uses the arm's ``input_dim``, else the product of ``input_shape``,
        else the first compatible task's ``input_dim``.
        """
        arm = self.arms[arm_name]
        if arm.input_dim is not None:
            return arm.input_dim
        if arm.input_shape:
            product = 1
            for dim in arm.input_shape:
                product *= dim
            return product
        for task in self.tasks:
            if task.input_dim is not None:
                return task.input_dim
        raise ValueError(  # ruff: ignore[raise-vanilla-args]  # descriptive message is the public API
            f"arm {arm_name!r} has no input_dim/input_shape and no task provides one"
        )

    def arm_output_dim(self, arm_name: str) -> int:
        """Resolve the number of classes for an arm."""
        arm = self.arms[arm_name]
        if arm.num_classes is not None:
            return arm.num_classes
        for task in self.tasks:
            if task.num_classes is not None:
                return task.num_classes
        raise ValueError(  # ruff: ignore[raise-vanilla-args]  # descriptive message is the public API
            f"arm {arm_name!r} has no num_classes and no task provides one"
        )


def validate_yaml(text: str) -> Campaign:
    """Parse and validate a campaign YAML string into a :class:`Campaign`.

    Raises:
        yaml.YAMLError: on malformed YAML.
        ValueError: when the document is empty.
        ValidationError: when required fields are missing or invalid.
    """
    data = yaml.safe_load(text)
    if data is None:
        raise ValueError("Campaign YAML is empty; expected a mapping of settings")  # ruff: ignore[raise-vanilla-args]  # descriptive message is the public API
    return Campaign.model_validate(data)


def load_campaign(path: str | Path) -> Campaign:
    """Load and validate a campaign from a YAML file path."""
    text = Path(path).read_text(encoding="utf-8")
    return validate_yaml(text)
