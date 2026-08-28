"""Configuration dataclasses and factories for stability estimators.

All configs are frozen, slotted dataclasses (PEP 695) with to_spec/from_spec
for YAML/JSON round-trip serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from computronium.stability.basin import BasinStabilityEstimator
from computronium.stability.guard import DEFAULT_TAU, StabilityGuard
from computronium.stability.lyapunov import LyapunovEstimator
from computronium.stability.settling import SettlingMonitor
from computronium.stability.spectral_radius import SpectralRadiusEstimator


@dataclass(frozen=True, slots=True)
class SpectralRadiusConfig:
    """Configuration for spectral radius estimation."""

    num_iterations: int = 20
    perturbation_scale: float = 1e-4
    activity_key: str = "x"
    fast_mode: bool = False

    def to_spec(self) -> dict:
        """Serialize to dictionary for YAML/JSON."""
        return {
            "num_iterations": self.num_iterations,
            "perturbation_scale": self.perturbation_scale,
            "activity_key": self.activity_key,
            "fast_mode": self.fast_mode,
        }

    @classmethod
    def from_spec(cls, spec: dict) -> SpectralRadiusConfig:
        """Deserialize from dictionary."""
        return cls(
            num_iterations=spec.get("num_iterations", 20),
            perturbation_scale=spec.get("perturbation_scale", 1e-4),
            activity_key=spec.get("activity_key", "x"),
            fast_mode=spec.get("fast_mode", False),
        )


@dataclass(frozen=True, slots=True)
class LyapunovConfig:
    """Configuration for Lyapunov exponent estimation."""

    num_steps: int = 50
    perturbation_scale: float = 1e-6
    activity_key: str = "x"
    renormalize_interval: int = 1
    fast_mode: bool = False

    def to_spec(self) -> dict:
        return {
            "num_steps": self.num_steps,
            "perturbation_scale": self.perturbation_scale,
            "activity_key": self.activity_key,
            "renormalize_interval": self.renormalize_interval,
            "fast_mode": self.fast_mode,
        }

    @classmethod
    def from_spec(cls, spec: dict) -> LyapunovConfig:
        return cls(
            num_steps=spec.get("num_steps", 50),
            perturbation_scale=spec.get("perturbation_scale", 1e-6),
            activity_key=spec.get("activity_key", "x"),
            renormalize_interval=spec.get("renormalize_interval", 1),
            fast_mode=spec.get("fast_mode", False),
        )


@dataclass(frozen=True, slots=True)
class SettlingConfig:
    """Configuration for settling time measurement."""

    tolerance: float = 1e-4
    max_steps: int = 1000
    activity_key: str = "x"
    norm_type: Literal["relative", "absolute"] = "relative"
    record_trajectory: bool = False

    def to_spec(self) -> dict:
        return {
            "tolerance": self.tolerance,
            "max_steps": self.max_steps,
            "activity_key": self.activity_key,
            "norm_type": self.norm_type,
            "record_trajectory": self.record_trajectory,
        }

    @classmethod
    def from_spec(cls, spec: dict) -> SettlingConfig:
        return cls(
            tolerance=spec.get("tolerance", 1e-4),
            max_steps=spec.get("max_steps", 1000),
            activity_key=spec.get("activity_key", "x"),
            norm_type=spec.get("norm_type", "relative"),
            record_trajectory=spec.get("record_trajectory", False),
        )


@dataclass(frozen=True, slots=True)
class BasinConfig:
    """Configuration for basin stability estimation."""

    num_samples: int = 100
    perturbation_radius: float = 1.0
    max_steps: int = 200
    tolerance: float = 1e-3
    activity_key: str = "x"
    distance_metric: Literal["euclidean", "cosine"] = "euclidean"
    fast_mode: bool = False

    def to_spec(self) -> dict:
        return {
            "num_samples": self.num_samples,
            "perturbation_radius": self.perturbation_radius,
            "max_steps": self.max_steps,
            "tolerance": self.tolerance,
            "activity_key": self.activity_key,
            "distance_metric": self.distance_metric,
            "fast_mode": self.fast_mode,
        }

    @classmethod
    def from_spec(cls, spec: dict) -> BasinConfig:
        return cls(
            num_samples=spec.get("num_samples", 100),
            perturbation_radius=spec.get("perturbation_radius", 1.0),
            max_steps=spec.get("max_steps", 200),
            tolerance=spec.get("tolerance", 1e-3),
            activity_key=spec.get("activity_key", "x"),
            distance_metric=spec.get("distance_metric", "euclidean"),
            fast_mode=spec.get("fast_mode", False),
        )


@dataclass(frozen=True, slots=True)
class GuardConfig:
    """Configuration for the stability guard."""

    threshold: float = DEFAULT_TAU
    statistic: Literal["fast_proxy", "windowed_growth"] = "windowed_growth"
    window: int = 10
    estimator_config: SpectralRadiusConfig = field(default_factory=SpectralRadiusConfig)

    def to_spec(self) -> dict:
        return {
            "threshold": self.threshold,
            "statistic": self.statistic,
            "window": self.window,
            "estimator_config": self.estimator_config.to_spec(),
        }

    @classmethod
    def from_spec(cls, spec: dict) -> GuardConfig:
        return cls(
            threshold=spec.get("threshold", DEFAULT_TAU),
            statistic=spec.get("statistic", "windowed_growth"),
            window=spec.get("window", 10),
            estimator_config=SpectralRadiusConfig.from_spec(
                spec.get("estimator_config", {})
            ),
        )


def create_spectral_radius_estimator(
    config: SpectralRadiusConfig,
) -> SpectralRadiusEstimator:
    """Factory for spectral radius estimator."""
    return SpectralRadiusEstimator(
        num_iterations=config.num_iterations,
        perturbation_scale=config.perturbation_scale,
        activity_key=config.activity_key,
        fast_mode=config.fast_mode,
    )


def create_lyapunov_estimator(config: LyapunovConfig) -> LyapunovEstimator:
    """Factory for Lyapunov estimator."""
    return LyapunovEstimator(
        num_steps=config.num_steps,
        perturbation_scale=config.perturbation_scale,
        activity_key=config.activity_key,
        renormalize_interval=config.renormalize_interval,
        fast_mode=config.fast_mode,
    )


def create_settling_monitor(config: SettlingConfig) -> SettlingMonitor:
    """Factory for settling monitor."""
    return SettlingMonitor(
        tolerance=config.tolerance,
        max_steps=config.max_steps,
        activity_key=config.activity_key,
        norm_type=config.norm_type,
        record_trajectory=config.record_trajectory,
    )


def create_basin_estimator(config: BasinConfig) -> BasinStabilityEstimator:
    """Factory for basin stability estimator."""
    return BasinStabilityEstimator(
        num_samples=config.num_samples,
        perturbation_radius=config.perturbation_radius,
        max_steps=config.max_steps,
        tolerance=config.tolerance,
        activity_key=config.activity_key,
        distance_metric=config.distance_metric,
        fast_mode=config.fast_mode,
    )


def create_guard(config: GuardConfig) -> StabilityGuard:
    """Factory for stability guard."""
    estimator = create_spectral_radius_estimator(config.estimator_config)
    return StabilityGuard(
        threshold=config.threshold,
        estimator=estimator,
        statistic=config.statistic,
        window=config.window,
    )
