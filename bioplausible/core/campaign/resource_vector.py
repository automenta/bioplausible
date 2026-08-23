"""
Resource Usage Tracking for Joint Campaigns.

Captures compute, memory, energy, latency, and plastic state capacity
for every coordinate evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    """Resource consumption vector for a coordinate evaluation."""

    compute: float              # FLOPs (forward + backward)
    memory: float               # Peak memory in MB
    energy: float               # Energy proxy (Joules or relative units)
    latency: float              # Wall time in seconds
    plastic_state_capacity: float  # Bytes of plastic state (ψ)

    # Additional metrics for detailed accounting
    forward_flops: float = field(default=0.0)
    backward_flops: float = field(default=0.0)
    parameter_count: int = field(default=0)
    activation_memory_mb: float = field(default=0.0)
    gradient_memory_mb: float = field(default=0.0)
    substrate_overhead: float = field(default=0.0)  # Adapter/compilations overhead

    def __add__(self, other: ResourceUsage) -> ResourceUsage:
        """Aggregate resource usage across episodes."""
        if not isinstance(other, ResourceUsage):
            return NotImplemented
        return ResourceUsage(
            compute=self.compute + other.compute,
            memory=max(self.memory, other.memory),  # Peak memory
            energy=self.energy + other.energy,
            latency=self.latency + other.latency,
            plastic_state_capacity=max(self.plastic_state_capacity, other.plastic_state_capacity),
            forward_flops=self.forward_flops + other.forward_flops,
            backward_flops=self.backward_flops + other.backward_flops,
            parameter_count=max(self.parameter_count, other.parameter_count),
            activation_memory_mb=max(self.activation_memory_mb, other.activation_memory_mb),
            gradient_memory_mb=max(self.gradient_memory_mb, other.gradient_memory_mb),
            substrate_overhead=self.substrate_overhead + other.substrate_overhead,
        )

    def __truediv__(self, divisor: float) -> ResourceUsage:
        """Average resource usage."""
        if divisor == 0:
            raise ValueError("Cannot divide by zero")
        return ResourceUsage(
            compute=self.compute / divisor,
            memory=self.memory / divisor,
            energy=self.energy / divisor,
            latency=self.latency / divisor,
            plastic_state_capacity=self.plastic_state_capacity / divisor,
            forward_flops=self.forward_flops / divisor,
            backward_flops=self.backward_flops / divisor,
            parameter_count=int(self.parameter_count / divisor),
            activation_memory_mb=self.activation_memory_mb / divisor,
            gradient_memory_mb=self.gradient_memory_mb / divisor,
            substrate_overhead=self.substrate_overhead / divisor,
        )

    @property
    def total_flops(self) -> float:
        """Total FLOPs (forward + backward)."""
        return self.forward_flops + self.backward_flops

    @property
    def efficiency_score(self) -> float:
        """Compute efficiency: accuracy proxy per FLOP (to be set externally)."""
        return 0.0

    def to_dict(self) -> dict[str, float | int]:
        """Convert to dictionary for serialization."""
        return {
            "compute": self.compute,
            "memory_mb": self.memory,
            "energy_j": self.energy,
            "latency_s": self.latency,
            "plastic_state_capacity_bytes": self.plastic_state_capacity,
            "forward_flops": self.forward_flops,
            "backward_flops": self.backward_flops,
            "parameter_count": self.parameter_count,
            "activation_memory_mb": self.activation_memory_mb,
            "gradient_memory_mb": self.gradient_memory_mb,
            "substrate_overhead": self.substrate_overhead,
        }

    @classmethod
    def from_dict(cls, data: dict[str, float | int]) -> ResourceUsage:
        """Create from dictionary."""
        return cls(
            compute=data.get("compute", 0.0),
            memory=data.get("memory_mb", 0.0),
            energy=data.get("energy_j", 0.0),
            latency=data.get("latency_s", 0.0),
            plastic_state_capacity=data.get("plastic_state_capacity_bytes", 0.0),
            forward_flops=data.get("forward_flops", 0.0),
            backward_flops=data.get("backward_flops", 0.0),
            parameter_count=int(data.get("parameter_count", 0)),
            activation_memory_mb=data.get("activation_memory_mb", 0.0),
            gradient_memory_mb=data.get("gradient_memory_mb", 0.0),
            substrate_overhead=data.get("substrate_overhead", 0.0),
        )

    @classmethod
    def measure(
        cls,
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        plastic_state: dict[str, torch.Tensor] | None = None,
        device: str = "cuda",
    ) -> ResourceUsage:
        """Measure resource usage for a forward/backward pass.

        Args:
            model: The model to measure
            input_tensor: Example input tensor
            plastic_state: Optional plastic state tensors
            device: Device to run on

        Returns:
            ResourceUsage with measured values
        """
        import time

        import torch

        model.eval()
        model.to(device)
        input_tensor = input_tensor.to(device)

        if plastic_state:
            plastic_state = {k: v.to(device) for k, v in plastic_state.items()}

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_tensor)

        torch.cuda.synchronize() if device == "cuda" else None

        # Measure forward
        start = time.perf_counter()
        output = model(input_tensor)
        torch.cuda.synchronize() if device == "cuda" else None
        forward_time = time.perf_counter() - start

        # Measure backward
        loss = output.sum()
        start = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize() if device == "cuda" else None
        backward_time = time.perf_counter() - start

        # Memory stats
        if device == "cuda":
            peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
            torch.cuda.reset_peak_memory_stats()
        else:
            peak_memory = 0.0

        # Parameter count
        param_count = sum(p.numel() for p in model.parameters())

        # Plastic state capacity
        plastic_capacity = 0
        if plastic_state:
            plastic_capacity = sum(p.numel() * p.element_size() for p in plastic_state.values())

        # Estimate FLOPs (rough approximation)
        # For a linear layer: 2 * in_features * out_features * batch_size
        forward_flops = 0
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                forward_flops += 2 * module.in_features * module.out_features * input_tensor.shape[0]
            elif isinstance(module, torch.nn.Conv2d):
                forward_flops += (
                    2
                    * module.in_channels
                    * module.out_channels
                    * module.kernel_size[0]
                    * module.kernel_size[1]
                    * input_tensor.shape[2]
                    * input_tensor.shape[3]
                    * input_tensor.shape[0]
                )

        backward_flops = forward_flops * 2  # Rough estimate

        return cls(
            compute=forward_flops + backward_flops,
            memory=peak_memory,
            energy=peak_memory * 1e-9 * (forward_time + backward_time),  # Very rough proxy
            latency=forward_time + backward_time,
            plastic_state_capacity=plastic_capacity,
            forward_flops=forward_flops,
            backward_flops=backward_flops,
            parameter_count=param_count,
        )