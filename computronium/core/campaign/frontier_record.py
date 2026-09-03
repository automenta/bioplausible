"""
Frontier Record for Joint Campaigns.

Captures the complete evaluation result for a 6-D coordinate including
task performance, stability metrics, resource usage, and plasticity primitive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from computronium.core.profiling import (
        ResourceUsage,
    )


@dataclass(frozen=True, slots=True)
class FrontierRecord:
    """Complete evaluation record for a 6-D coordinate on a task."""

    # Coordinate identification
    coordinate: str  # S/G/D/M/C/U string (e.g., "digital/recurrent/energy_min/routing/thermo/euclidean")
    task_name: str  # Task identifier (e.g., "mnist", "cifar10")

    # Task performance
    task_loss: float  # Final loss
    task_accuracy: float  # Final accuracy (or relevant metric)
    adaptation_time: int  # Episodes/steps to adapt

    # Stability metrics (from computronium.core.stability)
    rho_jacobian: float  # Spectral radius of Jacobian
    lyapunov_local: float  # Local Lyapunov exponent
    settling_time: float  # Settling time (steps)
    basin_stability: float  # Basin stability measure

    # Resource usage
    resources: ResourceUsage  # Compute, memory, energy, latency

    # Provenance seed (required: no silent legacy default)
    seed: int

    # Plasticity identification
    plasticity_primitive: str = (
        "null"  # null, routing, fast_weights, substrate_coupled, rule_state
    )
    plasticity_config: dict = field(default_factory=dict)  # Full plasticity config

    # State Registry metadata
    registry_signature: str = ""  # Hash of StateVariable registrations
    composite_state_shape: dict[str, dict[str, tuple[int, ...]]] = field(
        default_factory=dict
    )  # Shape of z=(x,ψ,σ)

    # Episode consolidation events
    consolidation_events: list[dict] = field(
        default_factory=list
    )  # List of {episode, promoted_vars, scale}

    # Metadata (provenance: floats for numeric knobs, str for stream labels)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, float | str] = field(default_factory=dict)

    # Campaign tracking
    campaign_id: str | None = None
    episode_index: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "coordinate": self.coordinate,
            "task_name": self.task_name,
            "task_loss": self.task_loss,
            "task_accuracy": self.task_accuracy,
            "adaptation_time": self.adaptation_time,
            "rho_jacobian": self.rho_jacobian,
            "lyapunov_local": self.lyapunov_local,
            "settling_time": self.settling_time,
            "basin_stability": self.basin_stability,
            "resources": self.resources.to_dict(),
            "plasticity_primitive": self.plasticity_primitive,
            "plasticity_config": self.plasticity_config,
            "registry_signature": self.registry_signature,
            "composite_state_shape": {
                k: {name: list(shape) for name, shape in v.items()}
                for k, v in self.composite_state_shape.items()
            },
            "consolidation_events": self.consolidation_events,
            "timestamp": self.timestamp,
            "seed": self.seed,
            "metadata": self.metadata,
            "campaign_id": self.campaign_id,
            "episode_index": self.episode_index,
        }

    @classmethod
    def from_dict(cls, data: dict) -> FrontierRecord:
        """Create from dictionary."""
        from computronium.resources import ResourceUsage

        return cls(
            coordinate=data["coordinate"],
            task_name=data["task_name"],
            task_loss=data["task_loss"],
            task_accuracy=data["task_accuracy"],
            adaptation_time=data["adaptation_time"],
            rho_jacobian=data["rho_jacobian"],
            lyapunov_local=data["lyapunov_local"],
            settling_time=data["settling_time"],
            basin_stability=data["basin_stability"],
            resources=ResourceUsage.from_dict(data["resources"]),
            plasticity_primitive=data.get("plasticity_primitive", "null"),
            plasticity_config=data.get("plasticity_config", {}),
            registry_signature=data.get("registry_signature", ""),
            composite_state_shape={
                k: {name: tuple(shape) for name, shape in v.items()}
                for k, v in data.get("composite_state_shape", {}).items()
            },
            consolidation_events=data.get("consolidation_events", []),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            seed=data["seed"],
            metadata=data.get("metadata", {}),
            campaign_id=data.get("campaign_id"),
            episode_index=data.get("episode_index", 0),
        )

    def stability_score(self) -> float:
        """Composite stability score (higher = more stable)."""
        # Lower rho_jacobian, lower lyapunov, lower settling_time, higher basin = better
        return (
            (1.0 - min(self.rho_jacobian, 2.0) / 2.0) * 0.3
            + (1.0 - min(abs(self.lyapunov_local), 2.0) / 2.0) * 0.3
            + (1.0 - min(self.settling_time, 100) / 100.0) * 0.2
            + min(self.basin_stability, 1.0) * 0.2
        )

    def efficiency_score(self) -> float:
        """Composite efficiency score (higher = more efficient)."""
        # Lower compute, memory, latency = better
        # Normalize by typical values
        compute_norm = min(self.resources.compute / 1e12, 1.0)  # 1 TFLOP
        memory_norm = min(self.resources.memory / 8192, 1.0)  # 8 GB
        latency_norm = min(self.resources.latency / 3600, 1.0)  # 1 hour
        return (
            (1.0 - compute_norm) * 0.4
            + (1.0 - memory_norm) * 0.3
            + (1.0 - latency_norm) * 0.3
        )

    def pareto_key(self) -> tuple[float, float, float, float]:
        """Key for Pareto frontier: (accuracy, -loss, stability, -resources)."""
        return (
            self.task_accuracy,
            -self.task_loss,
            self.stability_score(),
            -self.efficiency_score(),
        )


# Re-export ResourceUsage for convenience
