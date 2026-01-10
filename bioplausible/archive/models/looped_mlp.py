"""
LoopedMLP - Core Equilibrium Propagation Model with Spectral Normalization

This is the foundational model demonstrating:
1. Recurrent fixed-point dynamics
2. Spectral normalization for stability (L < 1)
3. Equilibrium-based gradient computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


class LoopedMLP(nn.Module):
    """
    A recurrent MLP that iterates to a fixed-point equilibrium.
    
    The key insight: By constraining Lipschitz constant L < 1 via spectral norm,
    the network is guaranteed to converge to a unique fixed point.
    
    Architecture:
        h_{t+1} = tanh(W_in @ x + W_rec @ h_t)
        output = W_out @ h*  (where h* is the fixed point)
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int,
        use_spectral_norm: bool = True,
        max_steps: int = 30,
    ):
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
        
        # Initialize with small weights for better stability
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable equilibrium dynamics."""
        for m in [self.W_in, self.W_rec, self.W_out]:
            # Handle spectral norm wrapper
            layer = m.parametrizations.weight.original if hasattr(m, 'parametrizations') else m
            if hasattr(layer, 'weight'):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        steps: int = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
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
        
        # Initialize hidden state
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
        
        # Pre-compute input contribution (constant across iterations)
        x_proj = self.W_in(x)
        
        trajectory = [h] if return_trajectory else None
        
        # Iterate to equilibrium
        for _ in range(steps):
            h = torch.tanh(x_proj + self.W_rec(h))
            if return_trajectory:
                trajectory.append(h)
        
        # Output from equilibrium state
        out = self.W_out(h)
        
        if return_trajectory:
            return out, trajectory
        return out
    
    def compute_lipschitz(self) -> float:
        """
        Compute the Lipschitz constant of the recurrent dynamics.
        
        For tanh activation: L = max_singular_value(W_rec) * lip(tanh)
        Since lip(tanh) = 1, L = sigma_max(W_rec)
        
        With spectral norm: L is guaranteed to be <= 1
        """
        # Get the effective weight (after spectral normalization if applied)
        with torch.no_grad():
            # Forward through the layer to get the actual normalized weight
            W = self.W_rec.weight  # This returns the normalized weight
            s = torch.linalg.svdvals(W)
            return s[0].item()
    
    def inject_noise_and_relax(
        self, 
        x: torch.Tensor, 
        noise_level: float = 1.0,
        injection_step: int = 15,
        total_steps: int = 30,
    ) -> dict:
        """
        Demonstrate self-healing: inject noise and measure damping.
        
        Returns a dict with initial noise, final noise, and damping ratio.
        """
        batch_size = x.shape[0]
        
        # Phase 1: Relax to near-equilibrium
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
        x_proj = self.W_in(x)
        
        for _ in range(injection_step):
            h = torch.tanh(x_proj + self.W_rec(h))
        
        # Record pre-noise state
        h_clean = h.clone()
        
        # Inject noise
        noise = torch.randn_like(h) * noise_level
        h_noisy = h + noise
        
        initial_noise_norm = noise.norm(dim=1).mean().item()
        
        # Phase 2: Continue relaxation with noisy state
        h = h_noisy
        for _ in range(total_steps - injection_step):
            h = torch.tanh(x_proj + self.W_rec(h))
        
        # Measure residual difference from clean trajectory
        # (Clean trajectory would have continued from h_clean)
        h_clean_final = h_clean
        for _ in range(total_steps - injection_step):
            h_clean_final = torch.tanh(x_proj + self.W_rec(h_clean_final))
        
        final_noise_norm = (h - h_clean_final).norm(dim=1).mean().item()
        
        damping_ratio = final_noise_norm / initial_noise_norm if initial_noise_norm > 0 else 0
        
        return {
            'initial_noise': initial_noise_norm,
            'final_noise': final_noise_norm,
            'damping_ratio': damping_ratio,
            'damping_percent': (1 - damping_ratio) * 100,
        }


class BackpropMLP(nn.Module):
    """Standard feedforward MLP for comparison (no equilibrium dynamics)."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
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
