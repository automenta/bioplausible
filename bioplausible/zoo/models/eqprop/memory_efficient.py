"""Equilibrium Propagation model variants."""

from typing import Any

import torch

from bioplausible.acceleration.kernels import HAS_CUPY, EqPropKernel

from ..base import EqPropModel
from .looped_mlp import LoopedMLP


class MemoryEfficientLoopedMLP(LoopedMLP):
    """
    Memory-efficient version of LoopedMLP that defaults to O(1) memory kernel backend.

    This model uses the NumPy/CuPy kernel for O(1) memory training, making it suitable
    for deep networks where PyTorch autograd would consume O(N) memory.

    Example:
        >>> model = MemoryEfficientLoopedMLP(784, 256, 10)
        >>> print(model.backend)  # 'kernel' if CUDA/CuPy available, else 'pytorch'
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        use_spectral_norm: bool = True,
        max_steps: int = 30,
        gradient_method: str = "bptt",
        use_gpu_if_available: bool = True,
    ) -> None:
        if use_gpu_if_available and HAS_CUPY and torch.cuda.is_available():
            backend = "kernel"
        else:
            backend = "pytorch"

        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            use_spectral_norm=use_spectral_norm,
            max_steps=max_steps,
            gradient_method=gradient_method,
            backend=backend,
        )

        self.is_memory_efficient = self.backend == "kernel"

    def __repr__(self) -> str:
        backend_str = f", backend={self.backend}"
        efficiency_str = (
            ", O(1) memory" if self.is_memory_efficient else ", O(N) memory"
        )
        return (
            f"MemoryEfficientLoopedMLP(input={self.input_dim}, hidden={self.hidden_dim}, "
            f"output={self.output_dim}, steps={self.max_steps}, "
            f"spectral_norm={self.use_spectral_norm}{backend_str}{efficiency_str})"
        )


class MemoryEfficientEqPropModel(EqPropModel):
    """
    Base class for memory-efficient EqProp models that can leverage kernel backend.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        max_steps: int = 30,
        gradient_method: str = "bptt",
        use_spectral_norm: bool = True,
        memory_efficient: bool = True,
        use_gpu: bool = True,
    ):
        self.memory_efficient = memory_efficient
        self.use_gpu = use_gpu and HAS_CUPY and torch.cuda.is_available()

        if memory_efficient and HAS_CUPY and self.use_gpu:
            self.backend = "kernel"
        else:
            self.backend = "pytorch"

        super().__init__(
            max_steps=max_steps,
            gradient_method=gradient_method,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            use_spectral_norm=use_spectral_norm,
        )

        if self.backend == "kernel":
            self._engine = EqPropKernel(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                max_steps=max_steps,
                use_spectral_norm=use_spectral_norm,
                use_gpu=self.use_gpu,
            )
        else:
            self._engine = None

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float] | None:
        if self.backend == "kernel" and self._engine is not None:
            if isinstance(x, torch.Tensor):
                x_np = x.detach().cpu().numpy()
            else:
                x_np = x

            if isinstance(y, torch.Tensor):
                y_np = y.detach().cpu().numpy()
            else:
                y_np = y

            metrics = self._engine.train_step(x_np, y_np)
            return metrics

        return super().train_step(x, y)

    def forward(self, x: torch.Tensor, steps: int | None = None, **kwargs):
        if self.backend == "kernel" and self._engine is not None:
            if isinstance(x, torch.Tensor):
                x_np = x.detach().cpu().numpy()
            else:
                x_np = x

            h_star, _, _ = self._engine.solve_equilibrium(x_np)
            logits_np = self._engine.compute_output(h_star)

            return torch.from_numpy(logits_np).to(x.device)

        return super().forward(x, steps, **kwargs)


def create_memory_efficient_model(
    model_type: str, input_dim: int, hidden_dim: int, output_dim: int, **kwargs
) -> Any:
    if model_type.lower() in ["loopedmlp", "memory_efficient", "o1_memory"]:
        return MemoryEfficientLoopedMLP(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, **kwargs
        )
    else:
        raise ValueError(f"Unsupported memory-efficient model type: {model_type}")
