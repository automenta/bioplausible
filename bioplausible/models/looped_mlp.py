import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
from typing import Optional, List, Tuple, Dict, Union

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

    def __repr__(self) -> str:
        return (f"LoopedMLP(input={self.input_dim}, hidden={self.hidden_dim}, "
                f"output={self.output_dim}, steps={self.max_steps}, "
                f"spectral_norm={self.use_spectral_norm})")

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
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Input dimension mismatch: expected {self.input_dim}, got {x.shape[1]}")

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
