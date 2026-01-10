"""
AdaptiveFeedbackAlignment - Novel Hybrid Algorithm

FA with slowly-evolving feedback matrix that adapts toward better alignment.
"""

import torch
import torch.nn as nn
from .base import BaseAlgorithm, AlgorithmConfig
from typing import Dict


class AdaptiveFeedbackAlignment(BaseAlgorithm):
    """FA with slow adaptive feedback evolution."""
    
    def __init__(self, config: AlgorithmConfig):
        super().__init__(config)
        
        # Feedback weights as ParameterList
        self.feedback_weights = nn.ParameterList()
        dims = [config.input_dim] + config.hidden_dims + [config.output_dim]
        
        for i in range(len(dims) - 1):
            B = torch.randn(dims[i+1], dims[i]) * 0.1
            self.feedback_weights.append(nn.Parameter(B, requires_grad=True))
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.w_optimizer = torch.optim.Adam(
            self.layers.parameters(),
            lr=config.learning_rate
        )
        self.b_optimizer = torch.optim.Adam(
            self.feedback_weights.parameters(),
            lr=config.learning_rate * 0.001
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h
    
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        self.w_optimizer.zero_grad()
        self.b_optimizer.zero_grad()
        
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
        
        with torch.no_grad():
            for i in reversed(range(len(self.layers))):
                h_prev = activations[i]
                
                if i == len(self.layers) - 1:
                    grad_h = error
                else:
                    grad_h = torch.mm(error, self.feedback_weights[i+1])
                    h_curr = activations[i+1]
                    
                    if self.config.activation == 'relu':
                        grad_h = grad_h * (h_curr > 0).float()
                    elif self.config.activation == 'tanh':
                        grad_h = grad_h * (1 - h_curr**2)
                
                grad_W = torch.mm(grad_h.T, h_prev) / x.size(0)
                self.layers[i].weight.data -= self.config.learning_rate * grad_W
                
                if self.layers[i].bias is not None:
                    self.layers[i].bias.data -= self.config.learning_rate * grad_h.mean(0)
                
                # Update B to match W
                if i < len(self.layers) - 1:
                    # Update B[i+1] to align with W[i+1]
                    # B[i+1] (Out, In) matches W[i+1] (Out, In)
                    # We want B to approximate W^T?
                    # No, FA matrix B replaces W^T.
                    # Forward: y = Wx.
                    # Backward: dx = B dy.
                    # Standard Backprop: dx = W^T dy.
                    # So B should approximate W^T.
                    # If W is (Out, In). W^T is (In, Out).
                    # B is (Out, In)?
                    # My backward pass: `grad_h = torch.mm(error, B)`.
                    # Error (Batch, Out). B must be (Out, In).
                    # Output (Batch, In).
                    # So B is (Out, In).
                    # Wait. (Batch, Out) x (Out, In) -> (Batch, In).
                    # W^T would be (In, Out).
                    # (Batch, Out) x (In, Out)^T -> (Batch, Out) x (Out, In).
                    # So B has same shape as W.
                    # So B approx W. NOT W^T.
                    # So target is W.
                    
                    target_B = self.layers[i+1].weight.data
                    current_B = self.feedback_weights[i+1].data
                    
                    diff = target_B - current_B
                    self.feedback_weights[i+1].data += self.config.learning_rate * 0.001 * diff
                
                error = grad_h
                
        return {'loss': loss.item(), 'accuracy': (output.argmax(1) == y).float().mean().item()}
