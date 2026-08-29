"""Resource Usage: Universal currency for stability frontier records.

Single canonical ResourceUsage dataclass serving both stability-frontier
and campaign resource vectors.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    """Canonical resource-usage record for a system coordinate (PR-3a).

    Single definition serving both former duplicates (the stability-frontier
    vector and the campaign resource vector): an aggregable five-axis
    consumption vector (compute / memory / energy / latency / ψ-capacity)
    plus detailed measurement fields for frontier records.

    Aggregation semantics: additive axes sum; peak-like axes (memory,
    ψ-capacity, param count, per-tensor memories) take the max.
    """

    # Aggregable vector core
    compute: float = 0.0
    memory: float = 0.0
    energy: float = 0.0
    latency: float = 0.0
    plastic_state_capacity: float = 0.0

    # Measurement detail
    coordinate: str = ""
    device: str = "cpu"
    batch_size: int = 0
    forward_flops: int = 0
    backward_flops: int = 0
    param_count: int = 0
    param_memory_mb: float = 0.0
    activation_memory_mb: float = 0.0
    gradient_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    peak_activation_bytes: int = 0
    activation_sparsity: float = 0.0
    weight_sparsity: float = 0.0
    wall_time_ms: float = 0.0
    energy_proxy: float = 0.0
    substrate_overhead: float = 0.0
    spectral_radius: float | None = None
    lyapunov_exponent: float | None = None
    effective_flops: float = 0.0

    @property
    def total_flops(self) -> float:
        """Total FLOPs (forward + backward)."""
        return self.forward_flops + self.backward_flops

    def __add__(self, other: ResourceUsage) -> ResourceUsage:
        """Aggregate two usage vectors."""
        return ResourceUsage(
            compute=self.compute + other.compute,
            memory=max(self.memory, other.memory),
            energy=self.energy + other.energy,
            latency=self.latency + other.latency,
            plastic_state_capacity=max(
                self.plastic_state_capacity, other.plastic_state_capacity
            ),
            coordinate=self.coordinate or other.coordinate,
            device=self.device or other.device,
            batch_size=self.batch_size + other.batch_size,
            forward_flops=self.forward_flops + other.forward_flops,
            backward_flops=self.backward_flops + other.backward_flops,
            param_count=max(self.param_count, other.param_count),
            param_memory_mb=max(self.param_memory_mb, other.param_memory_mb),
            activation_memory_mb=max(
                self.activation_memory_mb, other.activation_memory_mb
            ),
            gradient_memory_mb=max(self.gradient_memory_mb, other.gradient_memory_mb),
            peak_memory_mb=max(self.peak_memory_mb, other.peak_memory_mb),
            peak_activation_bytes=max(
                self.peak_activation_bytes, other.peak_activation_bytes
            ),
            activation_sparsity=max(
                self.activation_sparsity, other.activation_sparsity
            ),
            weight_sparsity=max(self.weight_sparsity, other.weight_sparsity),
            wall_time_ms=self.wall_time_ms + other.wall_time_ms,
            energy_proxy=self.energy_proxy + other.energy_proxy,
            substrate_overhead=self.substrate_overhead + other.substrate_overhead,
            spectral_radius=other.spectral_radius or self.spectral_radius,
            lyapunov_exponent=other.lyapunov_exponent or self.lyapunov_exponent,
            effective_flops=self.effective_flops + other.effective_flops,
        )

    def __truediv__(self, divisor: float) -> ResourceUsage:
        """Average over ``divisor`` samples."""
        if divisor == 0:
            raise ValueError("Cannot divide by zero")
        return ResourceUsage(
            compute=self.compute / divisor,
            memory=self.memory / divisor,
            energy=self.energy / divisor,
            latency=self.latency / divisor,
            plastic_state_capacity=self.plastic_state_capacity / divisor,
            coordinate=self.coordinate,
            device=self.device,
            batch_size=int(self.batch_size / divisor),
            forward_flops=int(self.forward_flops / divisor),
            backward_flops=int(self.backward_flops / divisor),
            param_count=self.param_count,
            param_memory_mb=self.param_memory_mb / divisor,
            activation_memory_mb=self.activation_memory_mb / divisor,
            gradient_memory_mb=self.gradient_memory_mb / divisor,
            peak_memory_mb=self.peak_memory_mb / divisor,
            peak_activation_bytes=int(self.peak_activation_bytes / divisor),
            activation_sparsity=self.activation_sparsity,
            weight_sparsity=self.weight_sparsity,
            wall_time_ms=self.wall_time_ms / divisor,
            energy_proxy=self.energy_proxy / divisor,
            substrate_overhead=self.substrate_overhead / divisor,
            spectral_radius=self.spectral_radius,
            lyapunov_exponent=self.lyapunov_exponent,
            effective_flops=self.effective_flops / divisor,
        )

    def to_dict(self) -> dict[str, float | int | str]:
        """Serialize (vector keys match the campaign JSONL schema)."""
        return {
            "compute": self.compute,
            "memory_mb": self.memory,
            "energy_j": self.energy,
            "latency_s": self.latency,
            "plastic_state_capacity_bytes": self.plastic_state_capacity,
            "coordinate": self.coordinate,
            "device": self.device,
            "batch_size": self.batch_size,
            "forward_flops": self.forward_flops,
            "backward_flops": self.backward_flops,
            "param_count": self.param_count,
            "peak_memory_mb": self.peak_memory_mb,
            "peak_activation_bytes": self.peak_activation_bytes,
            "wall_time_ms": self.wall_time_ms,
            "energy_proxy": self.energy_proxy,
            "substrate_overhead": self.substrate_overhead,
            "effective_flops": self.effective_flops,
        }

    @classmethod
    def from_dict(cls, data: dict[str, float | int | str]) -> ResourceUsage:
        """Deserialize a usage record."""
        return cls(
            compute=float(data.get("compute", 0.0)),
            memory=float(data.get("memory_mb", 0.0)),
            energy=float(data.get("energy_j", 0.0)),
            latency=float(data.get("latency_s", 0.0)),
            plastic_state_capacity=float(data.get("plastic_state_capacity_bytes", 0.0)),
            coordinate=str(data.get("coordinate", "")),
            device=str(data.get("device", "cpu")),
            batch_size=int(data.get("batch_size", 0)),
            forward_flops=int(data.get("forward_flops", 0)),
            backward_flops=int(data.get("backward_flops", 0)),
            # Legacy field name from pre-consolidation campaign records
            param_count=int(data.get("param_count", data.get("parameter_count", 0))),
            peak_memory_mb=float(data.get("peak_memory_mb", 0.0)),
            peak_activation_bytes=int(data.get("peak_activation_bytes", 0)),
            wall_time_ms=float(data.get("wall_time_ms", 0.0)),
            energy_proxy=float(data.get("energy_proxy", 0.0)),
            substrate_overhead=float(data.get("substrate_overhead", 0.0)),
            effective_flops=float(data.get("effective_flops", 0.0)),
        )

    @classmethod
    def measure(
        cls,
        model: nn.Module,
        input_tensor: torch.Tensor,
        plastic_state: dict[str, torch.Tensor] | None = None,
        device: str | None = None,
    ) -> ResourceUsage:
        """Measure resource usage for one forward/backward pass.

        ``device=None`` infers from the model's parameters so CPU-only callers
        are measured honestly instead of silently requiring CUDA.
        """
        import time
        import torch
        from torch import nn

        from computronium.utils import count_parameters

        device = device or next(model.parameters()).device.type
        model.eval()
        model.to(device)
        input_tensor = input_tensor.to(device)

        if plastic_state:
            plastic_state = {k: v.to(device) for k, v in plastic_state.items()}

        with torch.no_grad():
            for _ in range(3):
                _ = model(input_tensor)

        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        output = model(input_tensor)
        if device == "cuda":
            torch.cuda.synchronize()
        forward_time = time.perf_counter() - start

        loss = output.sum()
        start = time.perf_counter()
        loss.backward()
        if device == "cuda":
            torch.cuda.synchronize()
        backward_time = time.perf_counter() - start

        if device == "cuda":
            peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
            torch.cuda.reset_peak_memory_stats()
        else:
            peak_memory = 0.0

        param_count = sum(p.numel() for p in model.parameters())

        plastic_capacity = 0.0
        if plastic_state:
            plastic_capacity = sum(
                p.numel() * p.element_size() for p in plastic_state.values()
            )

        forward_flops = 0
        for module in model.modules():
            if isinstance(module, nn.Linear):
                forward_flops += (
                    2 * module.in_features * module.out_features * input_tensor.shape[0]
                )
            elif isinstance(module, nn.Conv2d):
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

        backward_flops = forward_flops * 2

        return cls(
            compute=forward_flops + backward_flops,
            memory=peak_memory,
            energy=peak_memory * 1e-9 * (forward_time + backward_time),
            latency=forward_time + backward_time,
            plastic_state_capacity=plastic_capacity,
            device=device,
            batch_size=input_tensor.shape[0],
            forward_flops=forward_flops,
            backward_flops=backward_flops,
            param_count=param_count,
        )
