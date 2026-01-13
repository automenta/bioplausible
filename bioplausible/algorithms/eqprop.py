"""
Standard Equilibrium Propagation

Reference implementation with correct top-down feedback dynamics.
"""

import torch
import torch.nn as nn
from .base import BaseAlgorithm, AlgorithmConfig
from typing import Dict, Optional, List


class StandardEqProp(BaseAlgorithm):
    """
    Standard EqProp with free/nudged phases and bidirectional relaxation.

    Implements the dynamics:
    h_i = sigma(W_i h_{i-1} + W_{i+1}^T h_{i+1} + b_i)
    """
    
    def __init__(self, config: AlgorithmConfig):
        super().__init__(config)
        self.beta = config.beta
        self.eq_steps = config.equilibrium_steps
        self.lr = config.learning_rate

        # We need to manually manage weights for transpose access
        # BaseAlgorithm creates nn.Linear layers in self.layers
        # We will use them but access .weight directly
    
    def forward_dynamics(self, activations: List[torch.Tensor], beta: float = 0.0, target: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Run one pass of relaxation dynamics over all layers.
        
        Args:
            activations: List of current state tensors [h0, h1, ..., hL]
            beta: Nudge strength
            target: Target tensor (one-hot)

        Returns:
            New list of state tensors
        """
        new_activations = [activations[0]] # Input is clamped
        
        num_layers = len(self.layers)
        
        for i in range(num_layers):
            # Layer i connects activations[i] -> activations[i+1]
            # We are updating activations[i+1] (which is hidden layer i or output)
            
            layer = self.layers[i]
            
            # Bottom-up input: W_i @ h_{i} + b_i
            # Note: activations indices are shifted by 1 relative to layers
            # activations[0] is input. activations[1] is output of layer 0.
            
            # h_{i+1} update
            h_prev = activations[i]     # h_{i}

            # Forward contribution
            a_bu = layer(h_prev) # W_i h_{i} + b_i

            # Top-down contribution
            a_td = 0.0
            if i < num_layers - 1:
                # Next layer is i+1. Weights are self.layers[i+1].weight
                # We need W_{i+1}^T @ h_{i+2}
                # h_{i+2} corresponds to activations[i+2]

                next_layer = self.layers[i+1]
                h_next = activations[i+2]

                # Check dimensions
                # next_layer.weight is [out_dim, in_dim]
                # h_next is [batch, out_dim]
                # We want [batch, in_dim] -> h_next @ W

                a_td = torch.matmul(h_next, next_layer.weight)

            # Total input
            total_input = a_bu + a_td

            # Output layer nudge
            if i == num_layers - 1 and beta > 0 and target is not None:
                # Nudge is effectively an extra term in the energy
                # For output h_L, we add -beta(h_L - y) to the gradient of Energy
                # Or simply nudge the state: h = h - beta * (h - y)
                # In discrete update: h_new = sigma(...)
                # Scellier et al usually do Euler update: dh/dt = -dH/dh - beta(h-y)
                # Here we assume the update is h = sigma(total_input) and then we nudge it?
                # Or we add the nudge to the input of the sigma?
                # Let's add it to the input (linear nudge) or post-activation.
                # Standard impl often clamps output or adds term.

                # We'll use the weak clamping formulation:
                # h = sigma(inputs) + beta * (target - h) ? No.

                # Let's stick to the formulation where the nudge is a gradient of the cost
                # added to the dynamics.
                # If loss is 1/2 ||h - y||^2, grad is (h - y).
                # Dynamics: dot{h} = -h + sigma(...) + beta(y - h)
                # Fixed point: h = sigma(...) + beta(y - h)
                pass

            # Apply activation
            if i < num_layers - 1:
                h_new = self.activation(total_input)
            else:
                # Output layer usually linear or identity for regression, but softmax/linear for class.
                # EqProp often uses identity output for simplicity in derivations or same activation.
                # BaseAlgorithm sets activation for hidden only?
                # BaseAlgorithm uses self.activation for hidden layers in its forward loop.
                # For consistency, let's assume identity for output (common in EqProp regression).
                h_new = total_input

            # Apply output nudge
            if i == num_layers - 1 and beta > 0 and target is not None:
                h_new = h_new + beta * (target - h_new)

            new_activations.append(h_new)

        return new_activations

    def forward(self, x: torch.Tensor, beta: float = 0.0, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Run equilibrium dynamics.
        """
        # Initialize states with a feedforward pass
        activations = [x]
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
            activations.append(h)

        # Relaxation iterations
        for _ in range(self.eq_steps):
            activations = self.forward_dynamics(activations, beta, target)

        # Cache for training
        self._last_activations = activations
        return activations[-1]
    
    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, float]:
        """EqProp training step with contrastive phases."""
        # One-hot encode target
        target = torch.zeros(y.size(0), self.config.output_dim, device=y.device)
        target.scatter_(1, y.unsqueeze(1), 1.0)
        
        # Free phase (beta=0)
        with torch.no_grad():
            self.forward(x, beta=0.0)
            free_activations = self._last_activations
            output_free = free_activations[-1]
        
        # Nudged phase (beta > 0)
        with torch.no_grad():
            self.forward(x, beta=self.beta, target=target)
            nudged_activations = self._last_activations
        
        # Contrastive update
        # dE/dW ~ h_post * h_prev^T
        # We want -(dE_nudged - dE_free) / beta
        # which is (h_post_nudged * h_prev_nudged^T - h_post_free * h_prev_free^T) / beta

        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                # activations index i is input to layer i
                # activations index i+1 is output of layer i

                h_prev_free = free_activations[i]
                h_post_free = free_activations[i+1] # This is post-activation
                
                h_prev_nudged = nudged_activations[i]
                h_post_nudged = nudged_activations[i+1]
                
                # Check shapes:
                # h_post: [batch, out_dim]
                # h_prev: [batch, in_dim]
                # Grad: [out_dim, in_dim] -> h_post.T @ h_prev

                # Note: For strict energy based models, the update depends on the specific energy function.
                # If E = 1/2 ||h - W u||^2, then dE/dW = -(h - W u) u^T.
                # At equilibrium, h = sigma(W u + ...).
                # The standard heuristic is simplified to the product of activities.

                # Calculate product
                prod_nudged = torch.matmul(h_post_nudged.T, h_prev_nudged)
                prod_free = torch.matmul(h_post_free.T, h_prev_free)

                dW = (prod_nudged - prod_free) / self.beta
                dW = dW / x.size(0) # Average over batch

                # Update weights (handle spectral norm)
                if hasattr(layer, 'weight_orig'):
                    # If spectral norm is active, we must update the original weight
                    # Note: dW is gradient wrt effective weight.
                    # Strictly speaking, we should backprop through SN, but for CHL/EqProp
                    # we often just update the parameters directly or assume SN handles the projection.
                    # However, simply adding to weight_orig is the standard "hack" in these implementations.
                    layer.weight_orig.data += self.lr * dW
                else:
                    layer.weight.data += self.lr * dW
                
                if layer.bias is not None:
                    db = (h_post_nudged - h_post_free).sum(0) / self.beta
                    db = db / x.size(0)
                    layer.bias.data += self.lr * db
        
        # Metrics
        pred = output_free.argmax(dim=1)
        acc = (pred == y).float().mean().item()
        loss = nn.functional.cross_entropy(output_free, y).item()
        
        return {
            'loss': loss,
            'accuracy': acc,
        }
