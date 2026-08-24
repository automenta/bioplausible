"""Equilibrium Propagation model variants."""

import torch

from computronium.acceleration.kernels import HAS_CUPY, EqPropKernel
from computronium.config.unified import ModelConfig

from ..base import EqPropModel
from ..transitions import TransitionGraphMixin
from ._energy import EquilibriumMLP

__all__ = [
    "MemoryEfficientEqPropModel",
    "MemoryEfficientLoopedMLP",
    "create_memory_efficient_model",
]


def _kernel_backend_step(
    engine: object,
    x: torch.Tensor,
    y: torch.Tensor,
) -> dict[str, float] | None:
    """Run a train step via the NumPy/CuPy kernel engine.

    Returns ``None`` when the engine is unavailable (caller falls back to
    the PyTorch implementation). The kernel engine is single-hidden-state
    only; layered ``LoopedMLP`` (``num_layers > 1``) must keep ``backend``
    on ``"pytorch"`` and never reach here — the consolidated engine's
    ``train_step`` path is then used instead.
    """
    if engine is None:
        return None
    x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y
    if x_np.ndim > 2:
        x_np = x_np.reshape(x_np.shape[0], -1)
    return engine.train_step(x_np, y_np)


def _make_config(
    name: str,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    use_spectral_norm: bool = True,
    max_steps: int = 30,
    gradient_method: str = "bptt",
    **extra,
) -> ModelConfig:
    """Create a ModelConfig for memory-efficient constructors."""
    return ModelConfig(
        name=name,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=(hidden_dim,),
        num_layers=1,
        learning_rate=extra.pop("learning_rate", 0.01),
        beta=extra.pop("beta", 0.5),
        max_steps=max_steps,
        convergence_threshold=extra.pop("convergence_threshold", 1e-4),
        convergence_start=extra.pop("convergence_start", 5),
        use_spectral_norm=use_spectral_norm,
        spectral_norm_power_iterations=extra.pop("spectral_norm_power_iterations", 5),
        activation=extra.pop("activation", "tanh"),
        lipschitz_mode=extra.pop("lipschitz_mode", "power_iteration"),
        output_scaling_mode=extra.pop("output_scaling_mode", "uniform"),
        dropout=extra.pop("dropout", 0.0),
        neurons_per_tile=extra.pop("neurons_per_tile", 48),
        tiles_per_layer=extra.pop("tiles_per_layer", 4),
        algorithm=extra.pop("algorithm", "ep"),
        mode=extra.pop("mode", "ep"),
        inference_steps=extra.pop("inference_steps", 10),
        step_size=extra.pop("step_size", 0.1),
        input_channels=extra.pop("input_channels", 3),
        input_size=extra.pop("input_size", 32),
        conv_channels=extra.pop("conv_channels", (32, 64, 128)),
        kernel_sizes=extra.pop("kernel_sizes", (3, 3, 3)),
        use_pooling=extra.pop("use_pooling", True),
        pooling_size=extra.pop("pooling_size", 2),
        attention_heads=extra.pop("attention_heads", 4),
        use_positional_encoding=extra.pop("use_positional_encoding", True),
        use_temporal_attention=extra.pop("use_temporal_attention", True),
        seq_len=extra.pop("seq_len", 64),
        hidden_dim=extra.pop("hidden_dim", hidden_dim),
        obs_dim=extra.pop("obs_dim", 8),
        action_dim=extra.pop("action_dim", 4),
        action_type=extra.pop("action_type", "discrete"),
        log_std_init=extra.pop("log_std_init", 0.0),
        log_std_min=extra.pop("log_std_min", -20.0),
        log_std_max=extra.pop("log_std_max", 2.0),
        entropy_coef=extra.pop("entropy_coef", 0.01),
        value_coef=extra.pop("value_coef", 0.5),
        max_grad_norm=extra.pop("max_grad_norm", 0.5),
        node_features=extra.pop("node_features", 10),
        aggregation=extra.pop("aggregation", "mean"),
        readout=extra.pop("readout", "mean"),
        extra={**extra, "gradient_method": gradient_method, "backend": "pytorch"},
    )


class MemoryEfficientLoopedMLP(EquilibriumMLP):
    """Memory-efficiency metadata facade over the consolidated layered eqprop MLP.

    The prior implementation routed to a NumPy/CuPy single-hidden kernel for
    O(1)-memory training. That kernel was bound to a single hidden state and
    is incompatible with the layered engine that now backs ``LoopedMLP`` (the
    consolidated ``EquilibriumMLP`` keeps one state per hidden layer and
    settles jointly via ``settle_activations_list``). The kernel backend is
    no longer wired; ``backend`` is recorded for trainer-compat and the
    contrastive ``train_step`` runs on the canonical PyTorch path.

    The class survives mainly because the hardware-targeted reproducibility /
    signal-probe validation tracks instantiate it for memory-efficiency
    measurement. Once those tracks grow layered-aware kernels, this facade
    can be retired.

    Example:
        >>> model = MemoryEfficientLoopedMLP(784, 256, 10)
        >>> print(model.backend)  # 'pytorch' (kernel backend is single-state only)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        use_spectral_norm: bool = True,
        max_steps: int = 30,
        gradient_method: str = "bptt",
        use_gpu_if_available: bool = True,  # ruff: ignore[unused-method-argument]  (compat kwarg, ignored)
    ) -> None:
        config = _make_config(
            "memory_efficient_looped_mlp",
            input_dim,
            hidden_dim,
            output_dim,
            use_spectral_norm=use_spectral_norm,
            max_steps=max_steps,
            gradient_method=gradient_method,
        )
        super().__init__(config=config)
        self.is_memory_efficient = False

    def __repr__(self) -> str:
        efficiency_str = (
            ", O(N) memory" if self.is_memory_efficient else ", layered PyTorch path"
        )
        return (
            f"MemoryEfficientLoopedMLP(input={self.input_dim}, hidden={self.hidden_dim}, "
            f"output={self.output_dim}, steps={self.max_steps}, "
            f"spectral_norm={self.use_spectral_norm}{efficiency_str})"
        )


class MemoryEfficientEqPropModel(TransitionGraphMixin, EqPropModel):
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

        config = _make_config(
            "memory_efficient_eqprop_model",
            input_dim,
            hidden_dim,
            output_dim,
            max_steps=max_steps,
            gradient_method=gradient_method,
            use_spectral_norm=use_spectral_norm,
        )
        super().__init__(config=config)

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
        if self.backend == "kernel":
            metrics = _kernel_backend_step(self._engine, x, y)
            if metrics is not None:
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
) -> object:
    if model_type.lower() in ["loopedmlp", "memory_efficient"]:
        return MemoryEfficientLoopedMLP(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, **kwargs
        )
    else:
        raise ValueError(f"Unsupported memory-efficient model type: {model_type}")
