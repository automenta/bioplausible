"""
Novel Hybrid Learning Algorithms

Base classes and interfaces for all algorithm implementations.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class AlgorithmConfig:
    """Configuration for a learning algorithm."""
    name: str
    
    # Model architecture
    input_dim: int
    hidden_dims: list
    output_dim: int
    
    # Training hyperparameters
    learning_rate: float = 0.001
    beta: float = 0.2  # For EqProp
    equilibrium_steps: int = 20
    
    # Algorithm-specific
    use_spectral_norm: bool = True
    activation: str = 'silu'
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.input_dim > 0
        assert self.output_dim > 0
        assert len(self.hidden_dims) > 0


class BaseAlgorithm(nn.Module, ABC):
    """Base class for all learning algorithms."""
    
    def __init__(self, config: AlgorithmConfig):
        super().__init__()
        self.config = config
        self.device = 'cpu'  # Force CPU for proper testing/compatibility
        
        # Build layers
        self.layers = nn.ModuleList()
        dims = [config.input_dim] + config.hidden_dims + [config.output_dim]
        
        for i in range(len(dims) - 1):
            layer = nn.Linear(dims[i], dims[i+1])
            if config.use_spectral_norm:
                layer = nn.utils.spectral_norm(layer)
            self.layers.append(layer)
        
        # Activation
        if config.activation == 'silu':
            self.activation = nn.SiLU()
        elif config.activation == 'relu':
            self.activation = nn.ReLU()
        elif config.activation == 'tanh':
            self.activation = nn.Tanh()
        elif config.activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
        
        self.to(self.device)
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        pass
    
    @abstractmethod
    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Returns:
            Dictionary with loss and other metrics
        """
        pass
    
    def get_num_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BackpropBaseline(BaseAlgorithm):
    """Standard backpropagation for comparison."""
    
    def __init__(self, config: AlgorithmConfig):
        super().__init__(config)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:  # No activation on output
                h = self.activation(h)
        return h
    
    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, float]:
        """Standard backprop training step."""
        self.optimizer.zero_grad()
        
        # Forward
        output = self.forward(x)
        loss = self.criterion(output, y)
        
        # Backward
        loss.backward()
        self.optimizer.step()
        
        # Metrics
        pred = output.argmax(dim=1)
        acc = (pred == y).float().mean().item()
        
        return {
            'loss': loss.item(),
            'accuracy': acc,
        }


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(
    algorithm_name: str,
    input_dim: int,
    hidden_dims: list,
    output_dim: int,
    **kwargs
) -> BaseAlgorithm:
    """
    Factory function to create algorithm instances.
    
    Args:
        algorithm_name: Name of algorithm
        input_dim: Input dimension
        hidden_dims: List of hidden layer sizes
        output_dim: Output dimension
        **kwargs: Additional config parameters
        
    Returns:
        Algorithm instance
    """
    config = AlgorithmConfig(
        name=algorithm_name,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        **kwargs
    )
    
    # Import here to avoid circular dependency
    if algorithm_name == 'backprop':
        return BackpropBaseline(config)
    elif algorithm_name == 'eqprop':
        from .eqprop import StandardEqProp
        return StandardEqProp(config)
    elif algorithm_name == 'feedback_alignment':
        from .feedback_align import StandardFA
        return StandardFA(config)
    elif algorithm_name == 'eq_align':
        from .eq_align import EquilibriumAlignment
        return EquilibriumAlignment(config)
    elif algorithm_name == 'ada_fa':
        from .ada_fa import AdaptiveFeedbackAlignment
        return AdaptiveFeedbackAlignment(config)
    elif algorithm_name == 'cf_align':
        from .cf_align import ContrastiveFeedbackAlignment
        return ContrastiveFeedbackAlignment(config)
    elif algorithm_name == 'leq_fa':
        from .leq_fa import LayerwiseEquilibriumFA
        return LayerwiseEquilibriumFA(config)
    elif algorithm_name == 'eg_fa':
        from .eg_fa import EnergyGuidedFA
        return EnergyGuidedFA(config)
    elif algorithm_name == 'pc_hybrid':
        from .pc_hybrid import PredictiveCodingHybrid
        return PredictiveCodingHybrid(config)
    elif algorithm_name == 'sparse_eq':
        from .sparse_eq import SparseEquilibrium
        return SparseEquilibrium(config)
    elif algorithm_name == 'mom_eq':
        from .mom_eq import MomentumEquilibrium
        return MomentumEquilibrium(config)
    elif algorithm_name == 'sto_fa':
        from .sto_fa import StochasticFA
        return StochasticFA(config)
    elif algorithm_name == 'em_fa':
        from .em_fa import EnergyMinimizingFA
        return EnergyMinimizingFA(config)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
