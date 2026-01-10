"""
EnergyGuidedFA - Novel Algorithm

Uses energy level to weight the Feedback Alignment updates.
High energy (unstable/bad) states get stronger updates.
"""

import torch
import torch.nn as nn
from .base import BaseAlgorithm, AlgorithmConfig
from typing import Dict


class EnergyGuidedFA(BaseAlgorithm):
    """FA with energy-weighted updates."""
    
    def __init__(self, config: AlgorithmConfig):
        super().__init__(config)
        
        self.feedback_weights = []
        dims = [config.input_dim] + config.hidden_dims + [config.output_dim]
        for i in range(len(dims) - 1):
            B = torch.randn(dims[i+1], dims[i], device=self.device) * 0.1
            self.feedback_weights.append(B)
            
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h

    def compute_energy(self, activations: list) -> torch.Tensor:
        """Compute Hopfield-like energy of the state."""
        energy = 0
        # E = -0.5 * sum(h_i * w_ij * h_j)
        for i, layer in enumerate(self.layers):
            pre = activations[i]
            post = activations[i+1]
            # Simple approximation: magnitude of activity * weights
            # (Just a heuristic metric for 'state quality')
            energy += (pre.norm(dim=1) * post.norm(dim=1)).mean()
        return -energy

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        # Forward
        activations = [x]
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
            activations.append(h)
            
        output = activations[-1]
        loss = self.criterion(output, y)
        
        # Calculate Energy
        energy = self.compute_energy(activations)
        
        # Energy modulation: Scale learning rate based on energy
        # Higher energy (worse state) -> higher LR? Or lower?
        # Hypothesis: High energy = chaos -> lower LR to stabilize
        energy_factor = torch.exp(-0.1 * torch.abs(energy)).item()
        effective_lr = self.config.learning_rate * energy_factor
        
        # FA Update
        error = output - torch.nn.functional.one_hot(y, self.config.output_dim).float()
        
        for i in reversed(range(len(self.layers))):
            h_prev = activations[i]
            
            if i == len(self.layers) - 1:
                grad_h = error
            else:
                grad_h = torch.mm(error, self.feedback_weights[i+1].to(error.device))
                h_curr = activations[i+1]
                if self.config.activation == 'relu':
                    grad_h = grad_h * (h_curr > 0).float()
            
            grad_W = torch.mm(grad_h.T, h_prev) / x.size(0)
            self.layers[i].weight.data -= effective_lr * grad_W
            if self.layers[i].bias is not None:
                self.layers[i].bias.data -= effective_lr * grad_h.mean(0)
                
            error = grad_h
            
        return {'loss': loss.item(), 'accuracy': (output.argmax(1) == y).float().mean().item()}
