"""
Convolutional EqProp (Track 13)

Extends Equilibrium Propagation to convolutional architectures for image tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from .utils import spectral_conv2d

class ConvEqProp(nn.Module):
    """
    Convolutional Equilibrium Propagation Model.
    Structure: Single-Block ResNet-like Loop.
    """
    
    def __init__(self, input_channels, hidden_channels, output_dim, gamma=0.5, 
                 use_spectral_norm=True):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gamma = gamma
        
        # Input embedding
        self.embed = spectral_conv2d(input_channels, hidden_channels, kernel_size=3, padding=1, 
                                    use_sn=use_spectral_norm)
        
        # Recurrent Weights
        # W1: Expansion
        self.W1 = spectral_conv2d(hidden_channels, hidden_channels*2, kernel_size=3, padding=1,
                                 use_sn=use_spectral_norm)
        # W2: Contraction
        self.W2 = spectral_conv2d(hidden_channels*2, hidden_channels, kernel_size=3, padding=1,
                                 use_sn=use_spectral_norm)
            
        self.norm = nn.GroupNorm(8, hidden_channels)
        
        # Classifier Head
        self.Head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_channels, output_dim)
        )
        
        # Init
        with torch.no_grad():
            self.W1.weight.mul_(0.5)
            self.W2.weight.mul_(0.5)

    def forward_step(self, h, x):
        h_norm = self.norm(h)
        x_emb = self.embed(x)
        
        pre_act = self.W1(h_norm)
        hidden = torch.tanh(pre_act)
        ffn_out = self.W2(hidden)
        
        h_target = ffn_out + x_emb
        h_next = (1 - self.gamma) * h + self.gamma * h_target
        return h_next

    def forward(self, x, steps=25):
        # Initialize h to 0
        B, _, H, W = x.shape
        # Note: simplistic assumption that H, W match after embed (true for padding=1, stride=1)
        h = torch.zeros(B, self.hidden_channels, H, W, device=x.device)
        
        for _ in range(steps):
             h = self.forward_step(h, x)
             
        return self.Head(h)
