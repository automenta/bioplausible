import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
from typing import Optional, List, Tuple, Dict, Union

from .eqprop_base import EqPropModel
from ..acceleration import compile_settling_loop
from .triton_kernel import TritonEqPropOps

# =============================================================================
# LoopedMLP - Core EqProp Model
# =============================================================================


class LoopedMLP(EqPropModel):
    """
    A recurrent MLP that iterates to a fixed-point equilibrium.

    The key insight: By constraining Lipschitz constant L < 1 via spectral norm,
    the network is guaranteed to converge to a unique fixed point.

    Architecture:
        h_{t+1} = tanh(W_in @ x + W_rec @ h_t)
        output = W_out @ h*  (where h* is the fixed point)

    This model can be trained using:
    1. BPTT (Backpropagation Through Time): With EqPropTrainer(use_kernel=False)
    2. EqProp (Equilibrium Propagation): Using EqPropTrainer(use_kernel=True).
       Note: For EqProp kernel mode, the weights are managed by the kernel (NumPy/CuPy),
       not this PyTorch module. This module is primarily for BPTT or inference/visualization.

    Example:
        >>> model = LoopedMLP(784, 256, 10, use_spectral_norm=True)
        >>> x = torch.randn(32, 784)
        >>> output = model(x, steps=30)  # [32, 10]
        >>> L = model.compute_lipschitz()  # Should be < 1.0
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        use_spectral_norm: bool = True,
        max_steps: int = 30,
        gradient_method: str = "bptt",
    ) -> None:
        # EqPropModel calls NEBCBase init which builds layers via _build_layers
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            max_steps=max_steps,
            use_spectral_norm=use_spectral_norm,
            gradient_method=gradient_method,
        )
        self._init_weights()

    def __repr__(self) -> str:
        return (
            f"LoopedMLP(input={self.input_dim}, hidden={self.hidden_dim}, "
            f"output={self.output_dim}, steps={self.max_steps}, "
            f"spectral_norm={self.use_spectral_norm})"
        )

    def _build_layers(self):
        """Build layers. Called by NEBCBase init."""
        # Input projection
        self.W_in = nn.Linear(self.input_dim, self.hidden_dim)

        # Recurrent (hidden-to-hidden) connection
        self.W_rec = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Output projection
        self.W_out = nn.Linear(self.hidden_dim, self.output_dim)

        # Apply spectral normalization if enabled
        if self.use_spectral_norm:
            self.W_in = spectral_norm(self.W_in)
            self.W_rec = spectral_norm(self.W_rec)
            self.W_out = spectral_norm(self.W_out)

    def _init_weights(self) -> None:
        """Initialize weights for stable equilibrium dynamics."""
        for layer in [self.W_in, self.W_rec, self.W_out]:
            self._initialize_single_layer(layer)

    def _initialize_single_layer(self, layer: nn.Module) -> None:
        """Initialize a single layer with proper weight and bias values."""
        actual_layer = self._get_actual_layer(layer)
        if hasattr(actual_layer, "weight"):
            nn.init.xavier_uniform_(actual_layer.weight, gain=0.5)
            if actual_layer.bias is not None:
                nn.init.zeros_(actual_layer.bias)

    def _get_actual_layer(self, layer: nn.Module) -> nn.Module:
        """Get the actual layer from a potentially wrapped layer."""
        if hasattr(layer, "parametrizations") and hasattr(
            layer.parametrizations, "weight"
        ):
            return layer.parametrizations.weight.original
        return layer

    def _initialize_hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        """Initialize the hidden state tensor."""
        batch_size = x.shape[0]
        return torch.zeros(
            (batch_size, self.hidden_dim), device=x.device, dtype=x.dtype
        )

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        """Transform input: W_in @ x"""
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Input dimension mismatch: expected {self.input_dim}, got {x.shape[1]}"
            )
        return self.W_in(x)

    def _forward_step_impl(
        self, h: torch.Tensor, x_transformed: torch.Tensor
    ) -> torch.Tensor:
        """Single step implementation (uncompiled)."""
        # Use Triton kernel if available for fused update
        if TritonEqPropOps.is_available():
            # pre_act = W_rec(h) + x_transformed
            # The kernel computes (1-a)h + a*tanh(pre_act)
            # Here we want straight tanh(pre_act), so alpha=1.0
            pre_act = x_transformed + self.W_rec(h)
            return TritonEqPropOps.step(h, pre_act, alpha=1.0)

        return torch.tanh(x_transformed + self.W_rec(h))

    @compile_settling_loop
    def forward_step(
        self, h: torch.Tensor, x_transformed: torch.Tensor
    ) -> torch.Tensor:
        """Single step: h = tanh(W_in x + W_rec h)"""
        return self._forward_step_impl(h, x_transformed)

    def _output_projection(self, h: torch.Tensor) -> torch.Tensor:
        """Output: W_out @ h"""
        return self.W_out(h)

    def contrastive_update(self, h_free, h_nudged, x, y):
        """
        Implement contrastive Hebbian update for LoopedMLP.
        Delta W_rec ~ h_n @ h_n.T - h_f @ h_f.T
        Delta W_in  ~ h_n @ x.T   - h_f @ x.T
        """
        batch_size = x.size(0)
        scale = 1.0 / (self.beta * batch_size)

        # 1. W_rec Update
        # grad = h_f @ h_f.T - h_n @ h_n.T
        # But we need to handle batch dim.
        # einsum 'bi,bj->ij'
        grad_rec_free = torch.einsum('bi,bj->ij', h_free, h_free)
        grad_rec_nudged = torch.einsum('bi,bj->ij', h_nudged, h_nudged)

        grad_rec = scale * (grad_rec_free - grad_rec_nudged)

        # Helper to get parameter
        def get_param(layer):
            actual = self._get_actual_layer(layer)
            if isinstance(actual, torch.nn.Parameter):
                return actual
            return actual.weight

        # Set .grad for W_rec
        w_rec_param = get_param(self.W_rec)
        if w_rec_param.grad is None:
            w_rec_param.grad = grad_rec
        else:
            w_rec_param.grad.add_(grad_rec)

        # 2. W_in Update
        # grad = h_f @ x.T - h_n @ x.T
        grad_in_free = torch.einsum('bi,bj->ij', h_free, x)
        grad_in_nudged = torch.einsum('bi,bj->ij', h_nudged, x)

        grad_in = scale * (grad_in_free - grad_in_nudged)

        w_in_param = get_param(self.W_in)
        if w_in_param.grad is None:
            w_in_param.grad = grad_in
        else:
            w_in_param.grad.add_(grad_in)

        # 3. W_out Update (Standard Gradient)
        # We compute gradient of CE loss w.r.t W_out at h_free
        # We can use autograd for this part easily
        logits = self._output_projection(h_free)
        loss = torch.nn.functional.cross_entropy(logits, y)

        # We only want grad for W_out
        w_out_param = get_param(self.W_out)

        # Use autograd.grad
        grads_out = torch.autograd.grad(loss, w_out_param, retain_graph=False)[0]

        if w_out_param.grad is None:
            w_out_param.grad = grads_out
        else:
            w_out_param.grad.add_(grads_out)


# =============================================================================
# BackpropMLP - Baseline for Comparison
# =============================================================================


class BackpropMLP(nn.Module):
    """Standard feedforward MLP for comparison (no equilibrium dynamics)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
