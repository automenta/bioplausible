"""
EqProp-Torch Models

All neural network architectures supporting Equilibrium Propagation training.
Models use spectral normalization to guarantee Lipschitz constant L < 1 for stable dynamics.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

# =============================================================================
# Utility Functions
# =============================================================================

def spectral_linear(in_features: int, out_features: int, bias: bool = True, use_sn: bool = True) -> nn.Module:
    """Create a linear layer with optional spectral normalization."""
    layer = nn.Linear(in_features, out_features, bias=bias)
    return spectral_norm(layer) if use_sn else layer


def spectral_conv2d(in_channels: int, out_channels: int, kernel_size: int,
                   stride: int = 1, padding: int = 0, bias: bool = True, use_sn: bool = True) -> nn.Module:
    """Create a Conv2d layer with optional spectral normalization."""
    layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
    return spectral_norm(layer) if use_sn else layer


def estimate_lipschitz(layer: nn.Module, iterations: int = 3) -> float:
    """
    Estimate Lipschitz constant (spectral norm) of a layer using power iteration.
    Works for Linear and Conv2d layers.
    """
    weight = _get_layer_weight(layer)
    if weight is None:
        return 0.0

    device = weight.device
    W = _reshape_weight_for_power_iteration(weight)

    with torch.no_grad():
        u = F.normalize(torch.randn(W.shape[1], device=device), dim=0)

        for _ in range(iterations):
            v = F.normalize(torch.mv(W, u), dim=0)
            u = F.normalize(torch.mv(W.t(), v), dim=0)

        sigma = torch.dot(u, torch.mv(W.t(), v))

    return sigma.item()


def _get_layer_weight(layer: nn.Module) -> Optional[torch.Tensor]:
    """Extract weight tensor from a layer."""
    if hasattr(layer, 'parametrizations') and hasattr(layer.parametrizations, 'weight'):
        return layer.weight
    elif hasattr(layer, 'weight'):
        return layer.weight
    return None


def _has_spectral_norm(layer: nn.Module) -> bool:
    """Check if a module has spectral normalization."""
    return (hasattr(layer, 'parametrizations') and
            hasattr(layer.parametrizations, 'weight'))


def _reshape_weight_for_power_iteration(weight: torch.Tensor) -> torch.Tensor:
    """Reshape weight tensor for power iteration (conv weights to 2D matrix)."""
    return weight.reshape(weight.shape[0], -1) if weight.dim() > 2 else weight


# =============================================================================
# LoopedMLP - Core EqProp Model
# =============================================================================

class LoopedMLP(nn.Module):
    """
    A recurrent MLP that iterates to a fixed-point equilibrium.

    The key insight: By constraining Lipschitz constant L < 1 via spectral norm,
    the network is guaranteed to converge to a unique fixed point.

    Architecture:
        h_{t+1} = tanh(W_in @ x + W_rec @ h_t)
        output = W_out @ h*  (where h* is the fixed point)

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
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_steps = max_steps
        self.use_spectral_norm = use_spectral_norm

        # Input projection
        self.W_in = nn.Linear(input_dim, hidden_dim)

        # Recurrent (hidden-to-hidden) connection
        self.W_rec = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.W_out = nn.Linear(hidden_dim, output_dim)

        # Apply spectral normalization if enabled
        if use_spectral_norm:
            self.W_in = spectral_norm(self.W_in)
            self.W_rec = spectral_norm(self.W_rec)
            self.W_out = spectral_norm(self.W_out)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for stable equilibrium dynamics."""
        for layer in [self.W_in, self.W_rec, self.W_out]:
            self._initialize_single_layer(layer)

    def _initialize_single_layer(self, layer: nn.Module) -> None:
        """Initialize a single layer with proper weight and bias values."""
        actual_layer = self._get_actual_layer(layer)
        if hasattr(actual_layer, 'weight'):
            nn.init.xavier_uniform_(actual_layer.weight, gain=0.5)
            if actual_layer.bias is not None:
                nn.init.zeros_(actual_layer.bias)

    def _get_actual_layer(self, layer: nn.Module) -> nn.Module:
        """Get the actual layer from a potentially wrapped layer."""
        if hasattr(layer, 'parametrizations') and hasattr(layer.parametrizations, 'weight'):
            return layer.parametrizations.weight.original
        return layer

    def _get_layer_weight(self, layer: nn.Module) -> Optional[torch.Tensor]:
        """Extract weight tensor from a layer."""
        if hasattr(layer, 'parametrizations') and hasattr(layer.parametrizations, 'weight'):
            return layer.weight
        elif hasattr(layer, 'weight'):
            return layer.weight
        return None

    def forward(
        self,
        x: torch.Tensor,
        steps: Optional[int] = None,
        return_trajectory: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass: iterate to equilibrium.

        Args:
            x: Input tensor [batch, input_dim]
            steps: Override number of iteration steps
            return_trajectory: If True, return all hidden states

        Returns:
            Output logits [batch, output_dim]
            (optionally) trajectory of hidden states
        """
        steps = steps or self.max_steps
        batch_size = x.shape[0]

        h = self._initialize_hidden_state(batch_size, x)
        x_proj = self.W_in(x)

        trajectory = [h] if return_trajectory else None

        h = self._iterate_to_equilibrium(h, x_proj, steps, return_trajectory, trajectory)

        out = self.W_out(h)

        if return_trajectory:
            return out, trajectory
        return out

    def _initialize_hidden_state(self, batch_size: int, x: torch.Tensor) -> torch.Tensor:
        """Initialize the hidden state tensor."""
        return self._create_zeros_tensor((batch_size, self.hidden_dim), x)

    def _create_zeros_tensor(self, shape: Tuple[int, ...], reference_tensor: torch.Tensor) -> torch.Tensor:
        """Create a zeros tensor with the same device and dtype as reference tensor.

        Args:
            shape: Shape of the tensor to create
            reference_tensor: Reference tensor to get device and dtype from

        Returns:
            Zeros tensor with specified shape and same device/dtype as reference
        """
        return torch.zeros(shape, device=reference_tensor.device, dtype=reference_tensor.dtype)

    def _iterate_to_equilibrium(self, h: torch.Tensor, x_proj: torch.Tensor, steps: int,
                               return_trajectory: bool, trajectory: Optional[List[torch.Tensor]]) -> torch.Tensor:
        """Iterate the hidden state to equilibrium."""
        for _ in range(steps):
            h = torch.tanh(x_proj + self.W_rec(h))
            if return_trajectory:
                trajectory.append(h)
        return h

    def compute_lipschitz(self) -> float:
        """
        Compute the Lipschitz constant of the recurrent dynamics.

        Returns:
            Lipschitz constant (guaranteed to be <= 1 with spectral norm)
        """
        with torch.no_grad():
            W = self.W_rec.weight
            s = torch.linalg.svdvals(W)
            return s[0].item()

    def inject_noise_and_relax(
        self,
        x: torch.Tensor,
        noise_level: float = 1.0,
        injection_step: int = 15,
        total_steps: int = 30,
    ) -> Dict[str, float]:
        """
        Demonstrate self-healing: inject noise and measure damping.

        Args:
            x: Input tensor [batch, input_dim]
            noise_level: Magnitude of injected noise
            injection_step: Step at which to inject noise
            total_steps: Total number of steps to run

        Returns:
            Dictionary containing noise metrics and damping information
        """
        batch_size = x.shape[0]
        h = self._create_zeros_tensor((batch_size, self.hidden_dim), x)
        x_proj = self.W_in(x)

        # Run to injection point
        h = self._run_equilibrium_steps(h, x_proj, injection_step)

        # Inject noise
        h_clean = h.clone()
        h_noisy = h + torch.randn_like(h) * noise_level

        initial_noise_norm = self._compute_noise_norm(h, h_clean).item()

        # Run noisy and clean trajectories
        h_final, h_clean_final = self._run_trajectories(h_noisy, h_clean, x_proj, injection_step, total_steps)

        final_noise_norm = self._compute_noise_norm(h_final, h_clean_final).item()
        damping_info = self._calculate_damping(initial_noise_norm, final_noise_norm)

        return {
            'initial_noise': initial_noise_norm,
            'final_noise': final_noise_norm,
            'damping_ratio': damping_info['ratio'],
            'damping_percent': damping_info['percent'],
        }

    def _run_trajectories(self, h_noisy: torch.Tensor, h_clean: torch.Tensor, x_proj: torch.Tensor,
                         injection_step: int, total_steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run both noisy and clean trajectories for the remaining steps."""
        steps_remaining = total_steps - injection_step
        h_final = self._run_equilibrium_steps(h_noisy, x_proj, steps_remaining)
        h_clean_final = self._run_equilibrium_steps(h_clean, x_proj, steps_remaining)
        return h_final, h_clean_final

    def _run_equilibrium_steps(self, h: torch.Tensor, x_proj: torch.Tensor, steps: int) -> torch.Tensor:
        """Run equilibrium dynamics for a specified number of steps."""
        for _ in range(steps):
            h = torch.tanh(x_proj + self.W_rec(h))
        return h

    def _compute_noise_norm(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        """Compute mean noise norm between two states."""
        return (h1 - h2).norm(dim=1).mean()

    def _calculate_damping(self, initial_norm: float, final_norm: float) -> Dict[str, float]:
        """Calculate damping ratio and percentage."""
        ratio = final_norm / initial_norm if initial_norm > 0 else 0
        return {
            'ratio': ratio,
            'percent': (1 - ratio) * 100
        }


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


# =============================================================================
# ConvEqProp - Convolutional EqProp for Vision Tasks
# =============================================================================

class ConvEqProp(nn.Module):
    """
    Convolutional Equilibrium Propagation Model.

    Uses ResNet-like loop structure with spectral normalization.
    Suitable for image classification tasks (MNIST, CIFAR-10).

    Example:
        >>> model = ConvEqProp(1, 32, 10)  # MNIST
        >>> x = torch.randn(32, 1, 28, 28)
        >>> output = model(x, steps=25)  # [32, 10]
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        output_dim: int,
        gamma: float = 0.5,
        use_spectral_norm: bool = True
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gamma = gamma

        # Input embedding
        self.embed = spectral_conv2d(
            input_channels, hidden_channels, kernel_size=3, padding=1,
            use_sn=use_spectral_norm
        )

        # Recurrent weights
        self.W1 = spectral_conv2d(
            hidden_channels, hidden_channels * 2, kernel_size=3, padding=1,
            use_sn=use_spectral_norm
        )
        self.W2 = spectral_conv2d(
            hidden_channels * 2, hidden_channels, kernel_size=3, padding=1,
            use_sn=use_spectral_norm
        )

        self.norm = nn.GroupNorm(8, hidden_channels)

        # Classifier head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_channels, output_dim)
        )

        # Initialize for stability
        with torch.no_grad():
            self.W1.weight.mul_(0.5)
            self.W2.weight.mul_(0.5)

    def forward_step(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Single equilibrium iteration step.

        Args:
            h: Current hidden state
            x: Input tensor

        Returns:
            Next hidden state
        """
        h_norm = self.norm(h)
        x_emb = self.embed(x)

        pre_act = self.W1(h_norm)
        hidden = torch.tanh(pre_act)
        ffn_out = self.W2(hidden)

        h_target = ffn_out + x_emb
        # Use torch.lerp for more efficient interpolation
        h_next = torch.lerp(h, h_target, self.gamma)
        return h_next

    def forward(self, x: torch.Tensor, steps: int = 25) -> torch.Tensor:
        """
        Forward pass: iterate to equilibrium.

        Args:
            x: Input tensor [batch, channels, height, width]
            steps: Number of equilibrium steps

        Returns:
            Output logits [batch, output_dim]
        """
        B, _, H, W = x.shape
        h = self._create_hidden_state_tensor(B, H, W, x)

        for _ in range(steps):
            h = self.forward_step(h, x)

        return self.head(h)

    def _create_hidden_state_tensor(self, batch_size: int, height: int, width: int, reference_tensor: torch.Tensor) -> torch.Tensor:
        """Create the initial hidden state tensor for ConvEqProp.

        Args:
            batch_size: Size of the batch dimension
            height: Height of the spatial dimensions
            width: Width of the spatial dimensions
            reference_tensor: Reference tensor to get device from

        Returns:
            Initialized hidden state tensor
        """
        return torch.zeros(batch_size, self.hidden_channels, height, width, device=reference_tensor.device)


# =============================================================================
# TransformerEqProp - Attention with Equilibrium Dynamics
# =============================================================================

class EqPropAttention(nn.Module):
    """Self-attention that participates in equilibrium dynamics."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, use_sn: bool = True) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.W_q = spectral_linear(hidden_dim, hidden_dim, use_sn=use_sn)
        self.W_k = spectral_linear(hidden_dim, hidden_dim, use_sn=use_sn)
        self.W_v = spectral_linear(hidden_dim, hidden_dim, use_sn=use_sn)
        self.W_o = spectral_linear(hidden_dim, hidden_dim, use_sn=use_sn)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = h.shape

        Q, K, V = self._compute_qkv(h, batch_size, seq_len)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        return self.W_o(self._reshape_output(out, batch_size, seq_len))

    def _compute_qkv(self, h: torch.Tensor, batch_size: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Query, Key, and Value tensors."""
        Q = self.W_q(h).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(h).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(h).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        return Q, K, V

    def _reshape_output(self, out: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        """Reshape attention output back to the original format."""
        return out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)


class TransformerEqProp(nn.Module):
    """
    Transformer with equilibrium dynamics.

    All layers (attention + FFN) iterate together to a joint equilibrium.
    Spectral normalization ensures stable convergence.

    Example:
        >>> model = TransformerEqProp(vocab_size=1000, hidden_dim=256, output_dim=10)
        >>> x = torch.randint(0, 1000, (32, 64))  # [batch, seq_len]
        >>> output = model(x, steps=20)  # [32, 10]
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        max_seq_len: int = 128,
        alpha: float = 0.5,
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.alpha = alpha

        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)

        self.attentions = nn.ModuleList([
            EqPropAttention(hidden_dim, num_heads, use_sn=use_spectral_norm)
            for _ in range(num_layers)
        ])

        self.ffns = nn.ModuleList([
            nn.Sequential(
                spectral_linear(hidden_dim, hidden_dim * 2, use_sn=use_spectral_norm),
                nn.ReLU(),
                spectral_linear(hidden_dim * 2, hidden_dim, use_sn=use_spectral_norm)
            ) for _ in range(num_layers)
        ])

        self.norms1 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        self.head = nn.Linear(hidden_dim, output_dim)

    def forward_step(self, h: torch.Tensor, x_emb: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Single equilibrium iteration step for one layer.

        Args:
            h: Current hidden state
            x_emb: Embedded input
            layer_idx: Index of the current layer

        Returns:
            Next hidden state
        """
        h_norm = self.norms1[layer_idx](h)
        h = h + self.attentions[layer_idx](h_norm)

        h_norm = self.norms2[layer_idx](h)
        ffn_out = self.ffns[layer_idx](h_norm)

        h_target = h + ffn_out + x_emb
        # Use torch.lerp for more efficient interpolation
        return torch.lerp(h, torch.tanh(h_target), self.alpha)

    def forward(self, x: torch.Tensor, steps: int = 20) -> torch.Tensor:
        """
        Forward pass: iterate all layers to joint equilibrium.

        Args:
            x: Input tensor [batch, seq_len]
            steps: Number of equilibrium steps

        Returns:
            Output logits [batch, output_dim]
        """
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x_emb = self.token_emb(x) + self.pos_emb(positions)

        h = torch.zeros_like(x_emb)

        for _ in range(steps):
            for i in range(self.num_layers):
                h = self.forward_step(h, x_emb, i)

        return self.head(h.mean(dim=1))


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Utility functions
    'spectral_linear',
    'spectral_conv2d',
    'estimate_lipschitz',
    # Models
    'LoopedMLP',
    'BackpropMLP',
    'ConvEqProp',
    'TransformerEqProp',
    'EqPropAttention',
]
