"""
Modern Convolutional EqProp for CIFAR-10 (Track 34)

Multi-stage convolutional architecture with equilibrium settling.
Target: 75%+ accuracy on CIFAR-10 (vs 44.5% baseline with LoopedMLP).

Architecture inspired by ResNet with spectral normalization for stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from .utils import spectral_conv2d

class ModernConvEqProp(nn.Module):
    """
    Multi-stage ConvEqProp with equilibrium settling.
    
    Architecture:
        Input: 3×32×32 (CIFAR-10)
        Stage 1: Conv 3→64, no pooling (32×32)
        Stage 2: Conv 64→128, stride 2 (16×16)
        Stage 3: Conv 128→256, stride 2 (8×8)
        Equilibrium: Recurrent conv at 256 channels
        Output: Global pool → Linear(256, 10)
    
    Key Features:
    - All convolutions use spectral normalization
    - GroupNorm instead of BatchNorm (better for small batches)
    - Equilibrium settling only in deepest stage (efficient)
    """
    
    def __init__(
        self, 
        eq_steps: int = 15, 
        gamma: float = 0.5,
        hidden_channels: int = 64,
        use_spectral_norm: bool = True
    ):
        super().__init__()
        self.eq_steps = eq_steps
        self.gamma = gamma
        self.hidden_channels = hidden_channels
        
        # Stage 1: Initial feature extraction (32×32)
        self.stage1 = nn.Sequential(
            spectral_conv2d(3, hidden_channels, 3, padding=1, use_sn=use_spectral_norm),
            nn.GroupNorm(8, hidden_channels),
            nn.Tanh()
        )
        
        # Stage 2: Downsample to 16×16
        self.stage2 = nn.Sequential(
            spectral_conv2d(hidden_channels, hidden_channels*2, 3, stride=2, padding=1, use_sn=use_spectral_norm),
            nn.GroupNorm(8, hidden_channels*2),
            nn.Tanh()
        )
        
        # Stage 3: Downsample to 8×8
        self.stage3 = nn.Sequential(
            spectral_conv2d(hidden_channels*2, hidden_channels*4, 3, stride=2, padding=1, use_sn=use_spectral_norm),
            nn.GroupNorm(8, hidden_channels*4),
            nn.Tanh()
        )
        
        # Equilibrium recurrent block (operates at 8×8 spatial resolution)
        self.eq_conv = spectral_conv2d(hidden_channels*4, hidden_channels*4, 3, padding=1, use_sn=use_spectral_norm)
        self.eq_norm = nn.GroupNorm(8, hidden_channels*4)
        
        # Output classification head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden_channels*4, 10)
        
        # Initialize weights for stable equilibrium
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with small weights for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Handle spectral norm wrapper
                if hasattr(m, 'parametrizations'):
                    weight = m.parametrizations.weight.original
                else:
                    weight = m.weight
                nn.init.kaiming_normal_(weight, mode='fan_out', nonlinearity='tanh')
                # Scale down for stability
                weight.data.mul_(0.5)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, steps: int = None) -> torch.Tensor:
        """
        Forward pass with equilibrium settling.
        
        Args:
            x: Input images [batch, 3, 32, 32]
            steps: Number of equilibrium iterations (default: self.eq_steps)
        
        Returns:
            Logits [batch, 10]
        """
        steps = steps or self.eq_steps
        
        # Non-recurrent feature extraction
        h = self.stage1(x)   # [batch, 64, 32, 32]
        h = self.stage2(h)   # [batch, 128, 16, 16]
        h = self.stage3(h)   # [batch, 256, 8, 8]
        
        # Equilibrium settling at deepest layer
        for _ in range(steps):
            h_norm = self.eq_norm(h)
            h_next = torch.tanh(self.eq_conv(h_norm))
            # Exponential moving average update
            h = (1 - self.gamma) * h + self.gamma * h_next
        
        # Classification from equilibrium state
        features = self.pool(h).flatten(1)  # [batch, 256]
        logits = self.fc(features)          # [batch, 10]
        
        return logits
    
    def compute_lipschitz(self) -> float:
        """
        Estimate Lipschitz constant of equilibrium block.
        
        Returns approximate upper bound on L.
        """
        with torch.no_grad():
            # Get weight from equilibrium conv (potentially spectral normed)
            W = self.eq_conv.weight  # [out_ch, in_ch, k, k]
            
            # Reshape to 2D matrix
            W_2d = W.reshape(W.size(0), -1)
            
            # Compute max singular value
            s = torch.linalg.svdvals(W_2d)
            
            # Account for Tanh Lipschitz constant (=1)
            # and gamma blending
            L = self.gamma * s[0].item()
            
            return L


class SimpleConvEqProp(nn.Module):
    """
    Simplified single-stage ConvEqProp for comparison.
    
    This is a baseline to demonstrate that multi-stage architecture
    provides significant improvement.
    """
    
    def __init__(
        self, 
        hidden_channels: int = 128,
        eq_steps: int = 20,
        gamma: float = 0.5,
        use_spectral_norm: bool = True
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.eq_steps = eq_steps
        self.gamma = gamma
        
        # Single-stage embedding
        self.embed = spectral_conv2d(3, hidden_channels, 3, padding=1, use_sn=use_spectral_norm)
        
        # Recurrent block
        self.W_rec = spectral_conv2d(hidden_channels, hidden_channels, 3, padding=1, use_sn=use_spectral_norm)
        self.norm = nn.GroupNorm(8, hidden_channels)
        
        # Classifier
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, 10)
        )
    
    def forward(self, x: torch.Tensor, steps: int = None) -> torch.Tensor:
        steps = steps or self.eq_steps
        
        # Embed input
        x_emb = self.embed(x)
        
        # Initialize hidden state
        h = torch.zeros_like(x_emb)
        
        # Equilibrium settling
        for _ in range(steps):
            h_norm = self.norm(h)
            h_next = torch.tanh(self.W_rec(h_norm) + x_emb)
            h = (1 - self.gamma) * h + self.gamma * h_next
        
        # Classify
        return self.head(h)
