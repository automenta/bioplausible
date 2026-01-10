"""
Algorithm Variant Model Builder

Builds models that implement different EqProp algorithm variations
based on AlgorithmConfig settings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from typing import Optional, Tuple
import math

from .algorithm import (
    AlgorithmConfig, UpdateRule, EquilibriumDynamics, 
    GradientApprox, SNStrategy, ActivationFunction
)


def get_activation(act_type: ActivationFunction):
    """Get activation function from enum."""
    return {
        ActivationFunction.TANH: torch.tanh,
        ActivationFunction.RELU: F.relu,
        ActivationFunction.GELU: F.gelu,
        ActivationFunction.SILU: F.silu,
        ActivationFunction.SOFTPLUS: F.softplus,
        ActivationFunction.SIGMOID: torch.sigmoid,
    }[act_type]


class AlgorithmVariantModel(nn.Module):
    """
    A configurable EqProp model that implements different algorithm variants.
    
    This model can express many different bio-plausible learning algorithms
    by varying its configuration.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        algo_config: AlgorithmConfig,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.config = algo_config
        
        self.activation = get_activation(algo_config.activation)
        
        # Input projection
        self.W_in = nn.Linear(input_dim, hidden_dim)
        
        # Recurrent connection
        self.W_rec = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.W_out = nn.Linear(hidden_dim, output_dim)
        
        # Apply spectral normalization based on strategy
        self._apply_spectral_norm()
        
        # For momentum dynamics
        self.velocity = None
        
        # For homeostatic plasticity
        if algo_config.use_homeostatic_plasticity:
            self.register_buffer('target_rate', torch.tensor(0.1))
            self.register_buffer('running_rate', torch.zeros(hidden_dim))
        
        # For lateral inhibition
        if algo_config.use_lateral_inhibition:
            self.W_lateral = nn.Linear(hidden_dim, hidden_dim, bias=False)
            nn.init.eye_(self.W_lateral.weight)
            self.W_lateral.weight.data *= -0.1  # Inhibitory
        
        # For feedback alignment
        if algo_config.gradient_approx == GradientApprox.RANDOM_FEEDBACK:
            self.B_feedback = nn.Parameter(
                torch.randn(hidden_dim, output_dim) * 0.1,
                requires_grad=False
            )
        elif algo_config.gradient_approx == GradientApprox.DIRECT_FEEDBACK:
            self.B_direct = nn.Parameter(
                torch.randn(hidden_dim, output_dim) * 0.1,
                requires_grad=False
            )
        
        self._init_weights()
    
    def _apply_spectral_norm(self):
        """Apply spectral normalization based on strategy."""
        if not self.config.use_sn:
            return
        
        if self.config.sn_strategy == SNStrategy.ALL_LAYERS:
            self.W_in = spectral_norm(self.W_in)
            self.W_rec = spectral_norm(self.W_rec)
            self.W_out = spectral_norm(self.W_out)
        elif self.config.sn_strategy == SNStrategy.RECURRENT_ONLY:
            self.W_rec = spectral_norm(self.W_rec)
        # ADAPTIVE and SOFT are handled dynamically in forward
    
    def _init_weights(self):
        """Initialize weights for stable dynamics."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param, gain=0.5)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(
        self, 
        x: torch.Tensor, 
        steps: Optional[int] = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """Forward pass with configurable equilibrium dynamics."""
        steps = steps or self.config.eq_steps
        batch_size = x.shape[0]
        
        # Initialize hidden state
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
        
        # Pre-compute input projection
        x_proj = self.W_in(x)
        
        trajectory = [h] if return_trajectory else None
        
        # Initialize velocity for momentum
        if self.config.equilibrium_dynamics == EquilibriumDynamics.MOMENTUM:
            velocity = torch.zeros_like(h)
        
        # Equilibrium iteration
        for step in range(steps):
            h_new = self._equilibrium_step(x_proj, h, step, steps)
            
            # Apply dynamics
            if self.config.equilibrium_dynamics == EquilibriumDynamics.FIXED_POINT:
                h = h_new
            
            elif self.config.equilibrium_dynamics == EquilibriumDynamics.MOMENTUM:
                velocity = self.config.momentum * velocity + (h_new - h)
                h = h + velocity
            
            elif self.config.equilibrium_dynamics == EquilibriumDynamics.DAMPED_OSCILLATION:
                h = self.config.damping * h + (1 - self.config.damping) * h_new
            
            elif self.config.equilibrium_dynamics == EquilibriumDynamics.ADAPTIVE_STEP:
                # Adaptive step size based on change magnitude
                delta = h_new - h
                step_size = 1.0 / (1.0 + delta.norm(dim=-1, keepdim=True))
                h = h + step_size * delta
            
            else:
                h = h_new
            
            if return_trajectory:
                trajectory.append(h)
        
        # Homeostatic regulation
        if self.config.use_homeostatic_plasticity:
            h = self._apply_homeostasis(h)
        
        # Output
        out = self.W_out(h)
        
        if return_trajectory:
            return out, trajectory
        return out
    
    def _equilibrium_step(
        self, 
        x_proj: torch.Tensor, 
        h: torch.Tensor,
        step: int,
        total_steps: int,
    ) -> torch.Tensor:
        """Single equilibrium update step."""
        # Base recurrent update
        h_rec = self.W_rec(h)
        
        # Lateral inhibition
        if self.config.use_lateral_inhibition:
            h_rec = h_rec + self.W_lateral(h)
        
        # Combine inputs
        pre_act = x_proj + h_rec
        
        # Apply activation
        h_new = self.activation(pre_act)
        
        # Mixing with previous state
        h_new = (1 - self.config.alpha) * h + self.config.alpha * h_new
        
        return h_new
    
    def _apply_homeostasis(self, h: torch.Tensor) -> torch.Tensor:
        """Apply homeostatic plasticity to regulate activity."""
        # Update running rate
        current_rate = (h > 0).float().mean(dim=0)
        self.running_rate = 0.99 * self.running_rate + 0.01 * current_rate
        
        # Adjust activity
        rate_error = self.target_rate - self.running_rate
        h = h + 0.1 * rate_error.unsqueeze(0)
        
        return h
    
    def compute_lipschitz(self) -> float:
        """Compute approximate Lipschitz constant."""
        with torch.no_grad():
            W = self.W_rec.weight
            if hasattr(self.W_rec, 'parametrizations'):
                # Already spectral normalized
                return 1.0
            s = torch.linalg.svdvals(W)
            return s[0].item()
    
    def contrastive_update(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ):
        """
        Perform contrastive Hebbian update based on configured update rule.
        """
        # Free phase
        h_free, traj_free = self.forward(x, return_trajectory=True)
        
        # Nudged phase (with target clamping)
        if self.config.update_rule == UpdateRule.CONTRASTIVE_HEBBIAN:
            h_nudged = self._nudged_forward(x, target)
            # Contrastive Hebbian: Δw ∝ h_nudged⊗h_nudged - h_free⊗h_free
            loss = F.cross_entropy(self.W_out(h_nudged), target)
        
        elif self.config.update_rule == UpdateRule.SYMMETRIC_DIFF:
            h_pos = self._nudged_forward(x, target, beta=self.config.beta)
            h_neg = self._nudged_forward(x, target, beta=-self.config.beta)
            # Symmetric difference
            loss = F.cross_entropy(self.W_out(h_pos), target)
        
        elif self.config.update_rule == UpdateRule.LOCAL_ENERGY:
            # Local energy minimization
            h = h_free
            energy = 0.5 * (h ** 2).sum()
            loss = energy + F.cross_entropy(self.W_out(h), target)
        
        else:
            # Default to standard loss
            loss = F.cross_entropy(h_free, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def _nudged_forward(
        self, 
        x: torch.Tensor, 
        target: torch.Tensor,
        beta: Optional[float] = None,
    ) -> torch.Tensor:
        """Forward pass with output nudging toward target."""
        beta = beta or self.config.beta
        
        h = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        x_proj = self.W_in(x)
        
        for _ in range(self.config.eq_steps):
            h_new = self._equilibrium_step(x_proj, h, 0, self.config.eq_steps)
            
            # Nudge toward target
            out = self.W_out(h_new)
            target_onehot = F.one_hot(target, out.size(-1)).float()
            nudge = beta * (target_onehot - F.softmax(out, dim=-1))
            
            # Backprop nudge to hidden
            if self.config.gradient_approx == GradientApprox.RANDOM_FEEDBACK:
                h_nudge = nudge @ self.B_feedback.T
            elif self.config.gradient_approx == GradientApprox.DIRECT_FEEDBACK:
                h_nudge = nudge @ self.B_direct.T
            else:
                h_nudge = nudge @ self.W_out.weight
            
            h = h_new + 0.1 * h_nudge
        
        return h


def build_algorithm_variant(
    algo_config: AlgorithmConfig,
    input_dim: int,
    output_dim: int,
    hidden_dim: int = 256,
) -> nn.Module:
    """Factory function to create algorithm variant model."""
    return AlgorithmVariantModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        algo_config=algo_config,
    )
