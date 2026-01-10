"""
Visualization Utilities for EqProp Trainer

Weight matrix visualization and display helpers.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
from collections import OrderedDict


def extract_weights(model: nn.Module, max_matrices: int = 6) -> Dict[str, np.ndarray]:
    """
    Extract weight matrices from a model for visualization.

    Args:
        model: PyTorch model
        max_matrices: Maximum number of matrices to extract

    Returns:
        Dict mapping layer names to weight arrays (as numpy)
    """
    return extract_weights_optimized(model, max_matrices)


def extract_weights_optimized(model: nn.Module, max_matrices: int = 6) -> Dict[str, np.ndarray]:
    """
    Optimized version of weight extraction using list comprehension and early filtering.

    Args:
        model: PyTorch model
        max_matrices: Maximum number of matrices to extract

    Returns:
        Dict mapping layer names to weight arrays (as numpy)
    """
    weights = OrderedDict()

    # Filter parameters first to avoid unnecessary operations
    weight_params = [
        (name, param)
        for name, param in model.named_parameters()
        if 'weight' in name and 'norm' not in name.lower()
    ][:max_matrices]

    for name, param in weight_params:
        # Copy to CPU and convert to numpy
        W = param.detach().cpu().numpy()

        # Handle different shapes
        if W.ndim == 4:
            # Conv weights: [out_channels, in_channels, kernel_h, kernel_w]
            # Flatten spatial dimensions
            W = W.reshape(W.shape[0], -1)
        elif W.ndim > 2:
            # Other high-dim weights: flatten to 2D
            W = W.reshape(W.shape[0], -1)

        # Transpose if needed to make it more square-ish for better display
        if W.shape[0] > W.shape[1] * 3:
            W = W.T

        weights[name] = W

    return weights


def format_weight_for_display(
    W: np.ndarray,
    max_size: int = 512
) -> np.ndarray:
    """
    Format weight matrix for heatmap display.

    Args:
        W: Weight matrix (2D numpy array)
        max_size: Maximum dimension for display (downsample if larger)

    Returns:
        Formatted weight array
    """
    # Downsample if too large
    if W.shape[0] > max_size or W.shape[1] > max_size:
        # Simple downsampling by taking every Nth element
        stride_r = max(1, W.shape[0] // max_size)
        stride_c = max(1, W.shape[1] // max_size)
        W = W[::stride_r, ::stride_c]

    return W


def create_colormap_for_weights() -> np.ndarray:
    """
    Create diverging colormap for weights.
    Red (negative) -> White (zero) -> Blue (positive)

    Returns:
        Colormap as (256, 3) RGB array
    """
    # Simple diverging colormap
    colormap = np.zeros((256, 3), dtype=np.uint8)

    # Red for negative
    colormap[:128, 0] = np.linspace(255, 255, 128).astype(np.uint8)  # R
    colormap[:128, 1] = np.linspace(0, 255, 128).astype(np.uint8)     # G
    colormap[:128, 2] = np.linspace(0, 255, 128).astype(np.uint8)     # B

    # Blue for positive
    colormap[128:, 0] = np.linspace(255, 0, 128).astype(np.uint8)     # R
    colormap[128:, 1] = np.linspace(255, 128, 128).astype(np.uint8)   # G
    colormap[128:, 2] = np.linspace(255, 255, 128).astype(np.uint8)   # B

    return colormap


def normalize_weights_for_display(
    W: np.ndarray,
    percentile: float = 95.0
) -> np.ndarray:
    """
    Normalize weights to [0, 1] range for visualization.

    Args:
        W: Weight matrix
        percentile: Clip outliers at this percentile

    Returns:
        Normalized weights in [0, 1]
    """
    # Use more efficient percentile calculation
    neg_percentile = 100 - percentile
    vmin, vmax = np.percentile(W, [neg_percentile, percentile])

    # Make symmetric around zero
    vabs = max(abs(vmin), abs(vmax))
    W_clipped = np.clip(W, -vabs, vabs)

    # Normalize to [0, 1] - avoid division by zero
    if vabs == 0:
        return np.zeros_like(W_clipped)

    W_normalized = (W_clipped + vabs) / (2 * vabs)

    return W_normalized


def get_layer_description(name: str) -> str:
    """
    Get human-readable description for a layer name.

    Args:
        name: Full parameter name (e.g., 'layers.0.weight')

    Returns:
        Short description (e.g., 'Layer 0')
    """
    # Remove common prefixes
    name = name.replace('module.', '')
    name = name.replace('_orig', '')

    # Extract meaningful parts
    if 'W_in' in name:
        return 'Input Weights'
    elif 'W_rec' in name:
        return 'Recurrent Weights'
    elif 'W_out' in name or 'lm_head' in name:
        return 'Output Weights'
    elif 'embed' in name.lower():
        return 'Embedding'
    elif 'attention' in name.lower():
        if 'W_q' in name:
            return 'Attn Query'
        elif 'W_k' in name:
            return 'Attn Key'
        elif 'W_v' in name:
            return 'Attn Value'
        elif 'W_o' in name:
            return 'Attn Output'
        else:
            return 'Attention'
    elif 'ffn' in name.lower() or 'mlp' in name.lower():
        return 'FFN'
    elif 'conv' in name.lower():
        return 'Conv Layer'
    elif 'layer' in name.lower():
        # Extract layer number
        parts = name.split('.')
        for i, part in enumerate(parts):
            if part.startswith('layer') and i + 1 < len(parts):
                try:
                    return f'Layer {parts[i+1]}'
                except:
                    pass

    # Fallback: just return cleaned name
    return name.replace('.weight', '').replace('.', ' ').title()[:20]
