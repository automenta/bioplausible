"""
EnergyMinimizingFA - Novel Algorithm

Directly minimizes the energy function using FA gradients.
Treats energy minimization as the training objective.
"""

import torch
import torch.nn as nn
from .base import BaseAlgorithm, AlgorithmConfig
from typing import Dict


class EnergyMinimizingFA(BaseAlgorithm):
    """Energy minimization via FA."""
    
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

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        self.zero_grad()
        
        activations = [x]
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
            activations.append(h)
            
        output = activations[-1]
        
        # Standard loss term
        task_loss = self.criterion(output, y)
        
        # Energy term (minimize activity magnitude)
        energy_loss = 0
        for act in activations:
            energy_loss += 0.01 * act.pow(2).mean()
            
        total_loss = task_loss + energy_loss
        
        # FA Backward pass (modulating by energy gradients)
        error = output - torch.nn.functional.one_hot(y, self.config.output_dim).float()
        
        for i in reversed(range(len(self.layers))):
            h_prev = activations[i]
            if i == len(self.layers) - 1:
                grad_h = error
            else:
                grad_h = torch.mm(error, self.feedback_weights[i+1].to(error.device))
                # Add energy gradient component directly to feedback
                grad_h += 0.01 * 2 * activations[i+1] # d/dx(x^2) = 2x
                
                h_curr = activations[i+1]
                if self.config.activation == 'relu':
                    grad_h = grad_h * (h_curr > 0).float()
            
            grad_W = torch.mm(grad_h.T, h_prev) / x.size(0)
            self.layers[i].weight.data -= self.config.learning_rate * grad_W
            if self.layers[i].bias is not None:
                self.layers[i].bias.data -= self.config.learning_rate * grad_h.mean(0)
            error = grad_h
            
        return {'loss': total_loss.item(), 'accuracy': (output.argmax(1) == y).float().mean().item()}
