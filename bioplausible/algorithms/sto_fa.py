"""
StochasticFA - Novel Algorithm

Randomly drops out feedback connections during training.
Acts as regularization for the feedback alignment signal.
"""

import torch
import torch.nn as nn
from .base import BaseAlgorithm, AlgorithmConfig
from typing import Dict


class StochasticFA(BaseAlgorithm):
    """FA with dropout on feedback signals."""
    
    def __init__(self, config: AlgorithmConfig):
        super().__init__(config)
        
        self.feedback_weights = []
        dims = [config.input_dim] + config.hidden_dims + [config.output_dim]
        for i in range(len(dims) - 1):
            B = torch.randn(dims[i+1], dims[i], device=self.device) * 0.1
            self.feedback_weights.append(B)
            
        self.criterion = nn.CrossEntropyLoss()
        self.drop_prob = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        activations = [x]
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
            activations.append(h)
            
        output = activations[-1]
        loss = self.criterion(output, y)
        error = output - torch.nn.functional.one_hot(y, self.config.output_dim).float()
        
        for i in reversed(range(len(self.layers))):
            h_prev = activations[i]
            
            if i == len(self.layers) - 1:
                grad_h = error
            else:
                # Stochastic feedback mask
                B = self.feedback_weights[i+1].to(error.device)
                mask = (torch.rand_like(B) > self.drop_prob).float()
                B_effective = B * mask * (1.0 / (1.0 - self.drop_prob))
                
                grad_h = torch.mm(error, B_effective)
                h_curr = activations[i+1]
                if self.config.activation == 'relu':
                    grad_h = grad_h * (h_curr > 0).float()
            
            grad_W = torch.mm(grad_h.T, h_prev) / x.size(0)
            self.layers[i].weight.data -= self.config.learning_rate * grad_W
            if self.layers[i].bias is not None:
                self.layers[i].bias.data -= self.config.learning_rate * grad_h.mean(0)
            
            error = grad_h
            
        return {'loss': loss.item(), 'accuracy': (output.argmax(1) == y).float().mean().item()}
