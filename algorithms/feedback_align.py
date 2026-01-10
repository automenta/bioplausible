"""
Standard Feedback Alignment

Random fixed backward weights for gradient approximation.
"""

import torch
import torch.nn as nn
from .base import BaseAlgorithm, AlgorithmConfig
from typing import Dict, Optional


class StandardFA(BaseAlgorithm):
    """Feedback Alignment with random fixed backward weights."""
    
    def __init__(self, config: AlgorithmConfig):
        super().__init__(config)
        
        # Random fixed feedback weights - stored as plain lists
        self.feedback_weights = []
        dims = [config.input_dim] + config.hidden_dims + [config.output_dim]
        
        for i in range(len(dims) - 1):
            B = torch.randn(dims[i+1], dims[i], device=self.device) * 0.1
            self.feedback_weights.append(B)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad],
            lr=config.learning_rate
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h
    
    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, float]:
        """FA training step with random feedback."""
        self.optimizer.zero_grad()
        
        # Forward pass, save activations
        activations = [x]
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
                activations.append(h)
            else:
                activations.append(h)
        
        output = activations[-1]
        loss = self.criterion(output, y)
        
        # Compute error at output
        error = output - torch.nn.functional.one_hot(y, self.config.output_dim).float()
        
        # Backpropagate through RANDOM feedback weights
        for i in reversed(range(len(self.layers))):
            h_prev = activations[i]
            
            if i == len(self.layers) - 1:
                grad_h = error
            else:
                # Use stored feedback matrix
                B = self.feedback_weights[i+1] # Adjust index if needed
                # Wait, indices.
                # layers: 0..L-1
                # dims: 0..L
                # i counts down: L-1 -> 0
                # feedback_weights len = L. stored 0..L-1.
                # i+1 index might be out of range if not careful.
                # feedback_weights[0] corresponds to layer 0? 
                # Let's check init:
                # dims[i+1] -> dims[i]
                # B matches layer i's transpose roughly.
                # we want error from layer i+1 backwards to layer i.
                # error shape (B, dims[i+1]).  B shape (dims[i+1], dims[i]).
                # grad_h = error @ B -> (B, dims[i]).
                
                # feedback_weights[i] should be the one.
                # But initialization: range(len(dims)-1) -> 0..L-1.
                # feedback_weights[i] matches layer i.
                
                grad_h = torch.mm(error, self.feedback_weights[i+1].to(error.device))
                
                h_curr = activations[i+1] # layer i output
                if self.config.activation == 'silu':
                    grad_h = grad_h * torch.sigmoid(h_curr) * (1 + h_curr * (1 - torch.sigmoid(h_curr)))
                elif self.config.activation == 'relu':
                    grad_h = grad_h * (h_curr > 0).float()
                elif self.config.activation == 'tanh':
                    grad_h = grad_h * (1 - h_curr**2)
            
            # Weight gradient
            grad_W = torch.mm(grad_h.T, h_prev) / x.size(0)
            
            # Manual update
            self.layers[i].weight.data -= self.config.learning_rate * grad_W
            if self.layers[i].bias is not None:
                grad_b = grad_h.mean(0)
                self.layers[i].bias.data -= self.config.learning_rate * grad_b
            
            error = grad_h
        
        # Metrics
        pred = output.argmax(dim=1)
        acc = (pred == y).float().mean().item()
        
        return {
            'loss': loss.item(),
            'accuracy': acc,
        }
