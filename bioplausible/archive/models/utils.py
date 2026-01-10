"""
Shared utilities for TorEqProp models.
"""

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm
from typing import Optional

def spectral_linear(in_features: int, out_features: int, bias: bool = True, use_sn: bool = True) -> nn.Module:
    """Factory for linear layers with optional spectral normalization."""
    layer = nn.Linear(in_features, out_features, bias=bias)
    if use_sn:
        return spectral_norm(layer)
    return layer

def spectral_conv2d(in_channels: int, out_channels: int, kernel_size: int, 
                   stride: int = 1, padding: int = 0, bias: bool = True, use_sn: bool = True) -> nn.Module:
    """Factory for Conv2d layers with optional spectral normalization."""
    layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
    if use_sn:
        return spectral_norm(layer)
    return layer

def estimate_lipschitz(layer: nn.Module, iterations: int = 3) -> float:
    """
    Estimate Lipschitz constant (spectral norm) of a layer using power iteration.
    Works for Linear and Conv2d layers.
    """
    if hasattr(layer, 'parametrizations') and hasattr(layer.parametrizations, 'weight'):
        # For spectral_norm wrapped layers, the weight is already normalized
        # But to be safe/general, we can compute it manually or trust the wrapper
        # The wrapper ensures sigma_max <= 1 (usually), but let's measure actual sigma
        weight = layer.weight
    elif hasattr(layer, 'weight'):
        weight = layer.weight
    else:
        return 0.0
        
    device = weight.device
    
    # Reshape conv weights to 2D matrix [out_channels, in_channels * k * k]
    if weight.dim() > 2:
        W = weight.reshape(weight.shape[0], -1)
    else:
        W = weight
        
    with torch.no_grad():
        u = torch.randn(W.shape[1], device=device)
        u = torch.nn.functional.normalize(u, dim=0)
        
        for _ in range(iterations):
            v = torch.mv(W, u)
            v = torch.nn.functional.normalize(v, dim=0)
            u = torch.mv(W.t(), v)
            u = torch.nn.functional.normalize(u, dim=0)
            
        sigma = torch.dot(u, torch.mv(W.t(), v))
        
    return sigma.item()
