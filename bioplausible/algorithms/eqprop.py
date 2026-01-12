"""
Standard Equilibrium Propagation

Reference implementation for comparison.
"""

import torch
import torch.nn as nn
from .base import BaseAlgorithm, AlgorithmConfig
from typing import Dict, Optional


class StandardEqProp(BaseAlgorithm):
    """Standard EqProp with free/nudged phases."""
    
    def __init__(self, config: AlgorithmConfig):
        super().__init__(config)
        self.beta = config.beta
        self.eq_steps = config.equilibrium_steps
        self.lr = config.learning_rate
    
    def forward(self, x: torch.Tensor, beta: float = 0.0, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Run equilibrium dynamics.
        
        Args:
            x: Input
            beta: Nudge strength (0 for free phase)
            target: Target for nudging
        """
        # Initialize with feedforward
        activations = [x]
        h = x
        
        for layer in self.layers[:-1]:
            h = self.activation(layer(h))
            activations.append(h)
        
        # Output layer
        h = self.layers[-1](h)
        activations.append(h)
        
        # Equilibrium iterations
        for _ in range(self.eq_steps):
            # Compute energy gradient at each layer
            h = activations[0]
            new_activations = [h]
            
            for i, layer in enumerate(self.layers[:-1]):
                h = self.activation(layer(h))
                new_activations.append(h)
            
            # Output with nudge
            h = self.layers[-1](h)
            if beta > 0 and target is not None:
                # Nudge toward target
                h = h - beta * (h - target)
            new_activations.append(h)
            
            activations = new_activations
        
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
        
        # Free phase
        with torch.no_grad():
            self.forward(x, beta=0.0)
            free_activations = []
            h = x
            for i, layer in enumerate(self.layers[:-1]):
                h = self.activation(layer(h))
                free_activations.append(h)
            output_free = self.layers[-1](h)
            free_activations.append(output_free)
        
        # Nudged phase
        with torch.no_grad():
            output_nudged = self.forward(x, beta=self.beta, target=target)
            nudged_activations = []
            h = x
            for i, layer in enumerate(self.layers[:-1]):
                h = self.activation(layer(h))
                nudged_activations.append(h)
            nudged_activations.append(output_nudged)
        
        # Contrastive update
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                if i == 0:
                    h_prev_free = x
                    h_prev_nudged = x
                else:
                    h_prev_free = free_activations[i-1]
                    h_prev_nudged = nudged_activations[i-1]
                
                h_post_free = free_activations[i]
                h_post_nudged = nudged_activations[i]
                
                # Contrastive Hebbian
                dW = torch.mm(h_post_nudged.T, h_prev_nudged) - torch.mm(h_post_free.T, h_prev_free)
                dW = dW / x.size(0)  # Average over batch
                
                # Update
                layer.weight.data += self.lr * dW
                if layer.bias is not None:
                    db = (h_post_nudged - h_post_free).mean(0)
                    layer.bias.data += self.lr * db
        
        # Metrics
        pred = output_free.argmax(dim=1)
        acc = (pred == y).float().mean().item()
        loss = nn.functional.cross_entropy(output_free, y).item()
        
        return {
            'loss': loss,
            'accuracy': acc,
        }
