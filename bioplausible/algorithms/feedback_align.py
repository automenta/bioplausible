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
        
        # Random fixed feedback weights - stored as buffers to move with model
        self.feedback_weights = nn.ParameterList() # Use ParameterList with requires_grad=False for easy management
        dims = [config.input_dim] + config.hidden_dims + [config.output_dim]
        
        for i in range(len(dims) - 1):
            # B maps from layer i+1 (out) to layer i (in)
            # Shape: [dims[i], dims[i+1]] ?
            # In train_step: grad_h = error @ B.
            # error: [batch, dims[i+1]].
            # We need grad_h: [batch, dims[i]].
            # So [batch, dims[i+1]] @ [dims[i+1], dims[i]] -> [batch, dims[i]].
            # B should be [dims[i+1], dims[i]].

            B = torch.randn(dims[i+1], dims[i]) * 0.1
            # Register as parameter but freeze it
            p = nn.Parameter(B, requires_grad=False)
            self.feedback_weights.append(p)
        
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
                # Loop goes from len(layers)-1 down to 0.
                # Layer i connects activations[i] to activations[i+1].
                # We are at layer i.
                # error is at activations[i+1] (output of layer i).
                # We need to propagate error through feedback weights of layer i to get error at activations[i]?
                # No, backprop: dL/dh_{i+1} -> dL/dh_i.
                # Standard backprop: dL/dh_i = (dL/dh_{i+1} * sigma'(z)) @ W_i.
                # FA: dL/dh_i = (dL/dh_{i+1} * sigma'(z)) @ B_i.

                # feedback_weights[i] corresponds to layer i.

                # In the loop:
                # First iteration: i = L-1 (output layer). handled by 'if i == ...'. grad_h = error.
                # Next iteration: i = L-2. We have 'error' from previous iter (which was grad_h of layer L-1 input).

                # Wait, 'error' variable in loop is propagated backwards.
                # Start: error = output - target. (at output of layer L-1).
                # i = L-1: grad_h = error. (This is dL/dz at output).
                # Update layer L-1 weights using grad_h and h_{L-1}.
                # Propagate error to input of layer L-1?
                # No, 'error' variable becomes 'grad_h' at end of loop.
                # grad_h is dL/dz at OUTPUT of layer i (if linear) or pre-activation?

                # Let's trace standard backprop manually.
                # y = Wx. E = (y-t)^2. dE/dy = (y-t). dE/dW = dE/dy * x.T. dE/dx = W.T * dE/dy.
                # Here:
                # i = L-1. Output layer.
                # grad_h (dE/dz) = error (dE/dy) * sigma'(z) (if activation).
                # But BaseAlgorithm puts activation AFTER layer i for i < L-1.
                # Output layer (i=L-1) has NO activation in BaseAlgorithm (usually).
                # So grad_h = error.

                # Next layer down (i = L-2).
                # We need error at input of layer L-1 (which is output of layer L-2).
                # dE/dh_{L-2} = W_{L-1}^T * dE/dz_{L-1}.
                # FA replaces W^T with B.
                # So backprop_error = dE/dz_{L-1} @ B_{L-1}.
                # dE/dz_{L-1} is 'grad_h' from previous iteration.
                # B_{L-1} is feedback_weights[L-1].

                # So we need to use feedback_weights[i+1] where i+1 is the index of the PREVIOUS layer we processed?
                # The loop variable 'i' is the current layer we are updating.
                # The error signal comes from layer 'i+1'.
                # So we need B from layer 'i+1'.

                # Example: 2 layers. 0, 1.
                # reversed: 1, 0.
                # i=1: grad_h = error. Update layer 1. error_out = grad_h.
                # i=0: we need to backprop through layer 1 to get error for layer 0?
                # Wait, the code says:
                # grad_h = torch.mm(error, self.feedback_weights[i+1])
                # If i=0, we access feedback_weights[1]. This is B for layer 1.
                # This matches: we propagate error from layer 1 output to layer 0 output through B_1.
                # So yes, index i+1 is correct for accessing the feedback weights of the layer *above*.

                # Code check: 'error' at start of i=0 loop is 'grad_h' from i=1 loop.
                # grad_h from i=1 is dL/dz_1.
                # We want dL/dh_1 = dL/dz_1 @ B_1.
                # Then dL/dz_0 = dL/dh_1 * sigma'(z_0).
                
                # Code:
                # grad_h = torch.mm(error, self.feedback_weights[i+1])
                # Here 'error' is dL/dz_{i+1} (from previous iter).
                # self.feedback_weights[i+1] is B_{i+1}.
                # So grad_h becomes dL/dh_{i} (roughly).
                # Then we apply sigma derivative of layer i output.
                # This looks correct.
                
                grad_h = torch.mm(error, self.feedback_weights[i+1])
                
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
