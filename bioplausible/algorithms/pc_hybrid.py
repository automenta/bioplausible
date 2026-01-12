"""
PredictiveCodingHybrid - Novel Algorithm

Combines Predictive Coding (top-down predictions) with Feedback Alignment.
Layers try to predict their inputs.
"""

import torch
import torch.nn as nn
from .base import BaseAlgorithm, AlgorithmConfig
from typing import Dict


class PredictiveCodingHybrid(BaseAlgorithm):
    """Layers predict inputs; FA propagates prediction errors."""
    
    def __init__(self, config: AlgorithmConfig):
        super().__init__(config)
        self.criterion = nn.CrossEntropyLoss()
        
        # Feedback connections are now separate parameters (top-down predictors)
        self.top_down = nn.ModuleList()
        dims = [config.input_dim] + config.hidden_dims + [config.output_dim]
        
        for i in range(len(dims) - 1):
            # Predict layer i from layer i+1
            layer = nn.Linear(dims[i+1], dims[i]) 
            self.top_down.append(layer)
            
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard bottom-up pass."""
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """PC training step."""
        self.optimizer.zero_grad()
        
        # 1. Bottom-up pass
        activations = [x]
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
            activations.append(h)
            
        output = activations[-1]
        loss_cls = self.criterion(output, y)
        
        # 2. Top-down prediction errors
        pc_loss = 0
        for i in range(len(self.layers)):
            # Predict layer i from i+1
            upper = activations[i+1].detach() # Stop gradients flowing back up immediately?
            lower_target = activations[i].detach()
            
            prediction = self.top_down[i](upper)
            pc_loss += nn.functional.mse_loss(prediction, lower_target)
            
        # Total loss optimization
        # In true PC, gradients flow differently, but this is a hybrid approximation
        total_loss = loss_cls + 0.1 * pc_loss
        total_loss.backward()
        self.optimizer.step()
        
        return {'loss': total_loss.item(), 'accuracy': (output.argmax(1) == y).float().mean().item()}
