"""Frontier Record: Aggregation of stability, resource, and performance metrics."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    """Resource consumption vector for a joint system coordinate.

    Attributes:
        compute: FLOPs or compute cycles (proxy).
        memory: Peak memory usage in MB.
        energy: Energy consumption in Joules (proxy).
        latency: Wall-clock latency in seconds.
        plastic_state_capacity: Number of plastic state parameters (ψ).
    """

    compute: float = 0.0
    memory: float = 0.0
    energy: float = 0.0
    latency: float = 0.0
    plastic_state_capacity: float = 0.0

    def __add__(self, other: ResourceUsage) -> ResourceUsage:
        """Add two resource usage vectors."""
        return ResourceUsage(
            compute=self.compute + other.compute,
            memory=max(self.memory, other.memory),
            energy=self.energy + other.energy,
            latency=self.latency + other.latency,
            plastic_state_capacity=max(
                self.plastic_state_capacity, other.plastic_state_capacity
            ),
        )

    def __truediv__(self, scalar: float) -> ResourceUsage:
        """Divide by scalar for averaging."""
        if scalar == 0:
            return ResourceUsage()
        return ResourceUsage(
            compute=self.compute / scalar,
            memory=self.memory / scalar,
            energy=self.energy / scalar,
            latency=self.latency / scalar,
            plastic_state_capacity=self.plastic_state_capacity / scalar,
        )

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            "compute": self.compute,
            "memory": self.memory,
            "energy": self.energy,
            "latency": self.latency,
            "plastic_state_capacity": self.plastic_state_capacity,
        }

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> ResourceUsage:
        """Create from dictionary."""
        return cls(
            compute=data.get("compute", 0.0),
            memory=data.get("memory", 0.0),
            energy=data.get("energy", 0.0),
            latency=data.get("latency", 0.0),
            plastic_state_capacity=data.get("plastic_state_capacity", 0.0),
        )


@dataclass(frozen=True, slots=True)
class FrontierRecord:
    """Complete frontier record for a 6-D coordinate evaluation.

    Captures the stability-plasticity tradeoff at a single coordinate
    in the joint architecture space.

    Attributes:
        coordinate: String representation of the 6-D coordinate (S/G/D/M/C/U).
        task_loss: Final task loss after adaptation/evaluation.
        adaptation_time: Number of steps/episodes to adapt.
        rho_jacobian: Spectral radius ρ(J_F) of the joint transition.
        lyapunov_local: Local Lyapunov exponent estimate.
        settling_time: Settling time in steps.
        basin_stability: Basin stability estimate [0, 1].
        resources: Resource usage vector.
        plasticity_primitive: Name of plasticity primitive used.
        metadata: Additional metadata (e.g., episode consolidation events).
    """

    coordinate: str
    task_loss: float
    adaptation_time: int
    rho_jacobian: float
    lyapunov_local: float
    settling_time: float
    basin_stability: float
    resources: ResourceUsage
    plasticity_primitive: str = "null"
    metadata: dict[str, float] = field(default_factory=dict)

    def is_stable(self, rho_threshold: float = 1.0) -> bool:
        """Check if the system is stable (ρ(J_F) < 1)."""
        return self.rho_jacobian < rho_threshold

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "coordinate": self.coordinate,
            "task_loss": self.task_loss,
            "adaptation_time": self.adaptation_time,
            "rho_jacobian": self.rho_jacobian,
            "lyapunov_local": self.lyapunov_local,
            "settling_time": self.settling_time,
            "basin_stability": self.basin_stability,
            "resources": self.resources.to_dict(),
            "plasticity_primitive": self.plasticity_primitive,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> FrontierRecord:
        """Create from dictionary."""
        return cls(
            coordinate=data["coordinate"],
            task_loss=data["task_loss"],
            adaptation_time=data["adaptation_time"],
            rho_jacobian=data["rho_jacobian"],
            lyapunov_local=data["lyapunov_local"],
            settling_time=data["settling_time"],
            basin_stability=data["basin_stability"],
            resources=ResourceUsage.from_dict(data["resources"]),
            plasticity_primitive=data.get("plasticity_primitive", "null"),
            metadata=data.get("metadata", {}),
        )


@dataclass(slots=True)
class FrontierAggregator:
    """Aggregates frontier records for Pareto analysis and reporting."""

    _records: list[FrontierRecord] = field(default_factory=list, init=False)

    def add(self, record: FrontierRecord) -> None:
        """Add a frontier record."""
        self._records.append(record)

    def get_pareto_frontier(
        self,
        objectives: list[str] | None = None,
        maximize: list[bool] | None = None,
    ) -> list[FrontierRecord]:
        """Compute Pareto frontier over records.

        Args:
            objectives: List of objective keys from record.to_dict().
            maximize: List of booleans indicating whether to maximize each objective.

        Returns:
            List of records on the Pareto frontier.
        """
        if not self._records:
            return []

        if objectives is None:
            objectives = [
                "task_loss",
                "adaptation_time",
                "rho_jacobian",
                "settling_time",
                "resources.compute",
            ]
        if maximize is None:
            maximize = [False, False, False, False, False]  # All minimize by default

        # Convert records to objective vectors
        obj_vectors = []
        for record in self._records:
            d = record.to_dict()
            vec = []
            for obj in objectives:
                if obj.startswith("resources."):
                    vec.append(d["resources"].get(obj.split(".", 1)[1], 0.0))
                else:
                    vec.append(d.get(obj, 0.0))
            obj_vectors.append(vec)

        # Simple Pareto dominance check (O(n^2), fine for small n)
        pareto = []
        for i, vec_i in enumerate(obj_vectors):
            dominated = False
            for j, vec_j in enumerate(obj_vectors):
                if i == j:
                    continue
                # Check if j dominates i
                better = False
                for k, (vi, vj) in enumerate(zip(vec_i, vec_j)):
                    if maximize[k]:
                        if vj < vi:
                            break
                        if vj > vi:
                            better = True
                    else:
                        if vj > vi:
                            break
                        if vj < vi:
                            better = True
                else:
                    if better:
                        dominated = True
                        break
            if not dominated:
                pareto.append(self._records[i])

        return pareto

    def get_best_by_objective(
        self, objective: str, maximize: bool = False
    ) -> FrontierRecord | None:
        """Get the best record by a single objective."""
        if not self._records:
            return None

        def get_val(r: FrontierRecord) -> float:
            d = r.to_dict()
            if objective.startswith("resources."):
                return d["resources"].get(objective.split(".", 1)[1], 0.0)
            return d.get(objective, float("inf") if not maximize else -float("inf"))

        if maximize:
            return max(self._records, key=get_val)
        return min(self._records, key=get_val)

    def clear(self) -> None:
        """Clear all records."""
        self._records.clear()

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self._records)
