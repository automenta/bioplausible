"""
MomentumEquilibrium - Novel Algorithm

Adds momentum term to the equilibrium settling dynamics.
h_{t+1} = h_t + alpha * (target - h_t) + mu * (h_t - h_{t-1})
Accelerates convergence to fixed point.
"""

import torch
import torch.nn as nn
from .base import BaseAlgorithm, AlgorithmConfig
from typing import Dict, List, Optional


class MomentumEquilibrium(BaseAlgorithm):
    """EqProp with momentum-accelerated settling."""
    
    def __init__(self, config: AlgorithmConfig):
        super().__init__(config)
        self.momentum = 0.5
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run settling with momentum."""
        activations = [x]
        h = x
        for layer in self.layers[:-1]:
            h = self.activation(layer(h))
            activations.append(h)
        h = self.layers[-1](h)
        activations.append(h)
        
        # Momentum buffers
        velocities = [torch.zeros_like(a) for a in activations]
        
        for _ in range(self.config.equilibrium_steps):
            new_acts = [activations[0]]
            h = activations[0]
            
            for i, layer in enumerate(self.layers[:-1]):
                target = self.activation(layer(h))
                
                # Update with momentum
                # v = mu * v + (target - current)
                # current = current + v
                delta = target - activations[i+1]
                velocities[i+1] = self.momentum * velocities[i+1] + 0.5 * delta
                
                h = activations[i+1] + velocities[i+1]
                new_acts.append(h)
                
            h = self.layers[-1](h)
            new_acts.append(h)
            activations = new_acts
            
        return activations[-1]

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        # Simple backprop through the momentum dynamics
        output = self.forward(x)
        loss = self.criterion(output, y)
        loss.backward()
        
        with torch.no_grad():
            for p in self.parameters():
                if p.grad is not None:
                    p.data -= self.config.learning_rate * p.grad
                    p.grad.zero_()
                    
        return {'loss': loss.item(), 'accuracy': (output.argmax(1) == y).float().mean().item()}
