"""
EquilibriumAlignment - Novel Hybrid Algorithm

Combines EqProp's equilibrium dynamics for rich representations
with FA's random feedback for efficient training.
"""

import torch
import torch.nn as nn
from .base import BaseAlgorithm, AlgorithmConfig
from typing import Dict, List


class EquilibriumAlignment(BaseAlgorithm):
    """Equilibrium features + Feedback Alignment training."""
    
    def __init__(self, config: AlgorithmConfig):
        super().__init__(config)
        
        # Random fixed feedback weights (stored as list)
        self.feedback_weights = []
        dims = [config.input_dim] + config.hidden_dims + [config.output_dim]
        
        for i in range(len(dims) - 1):
            B = torch.randn(dims[i+1], dims[i], device=self.device) * 0.1
            self.feedback_weights.append(B)
        
        self.eq_steps = config.equilibrium_steps
        self.criterion = nn.CrossEntropyLoss()
    
    def forward_phase(self, x: torch.Tensor, steps: int = None) -> List[torch.Tensor]:
        if steps is None:
            steps = self.eq_steps
            
        activations = [x]
        h = x
        for layer in self.layers[:-1]:
            h = self.activation(layer(h))
            activations.append(h)
        h = self.layers[-1](h)
        activations.append(h)
        
        for _ in range(steps):
            h = activations[0]
            new_acts = [h]
            for i, layer in enumerate(self.layers[:-1]):
                h = self.activation(layer(h))
                new_acts.append(h)
            h = self.layers[-1](h)
            new_acts.append(h)
            activations = new_acts
            
        return activations
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Get equilibrium output."""
        return self.forward_phase(x)[-1]
    
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        activations = self.forward_phase(x)
        output = activations[-1]
        loss = self.criterion(output, y)
        
        error = output - torch.nn.functional.one_hot(y, self.config.output_dim).float()
        
        with torch.no_grad():
            for i in reversed(range(len(self.layers))):
                h_prev = activations[i]
                
                if i == len(self.layers) - 1:
                    grad_h = error
                else:
                    # Correct index: i
                    grad_h = torch.mm(error, self.feedback_weights[i+1].to(error.device))
                    h_curr = activations[i+1]
                    
                    if self.config.activation == 'relu':
                        grad_h = grad_h * (h_curr > 0).float()
                    elif self.config.activation == 'tanh':
                        grad_h = grad_h * (1 - h_curr**2)
                
                grad_W = torch.mm(grad_h.T, h_prev) / x.size(0)
                self.layers[i].weight.data -= self.config.learning_rate * grad_W
                
                if self.layers[i].bias is not None:
                    self.layers[i].bias.data -= self.config.learning_rate * grad_h.mean(0)
                
                error = grad_h
                
        return {'loss': loss.item(), 'accuracy': (output.argmax(1) == y).float().mean().item()}
