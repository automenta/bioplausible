"""
Bio-Plausible Model Base Classes

Unified foundation for all biologically plausible learning algorithms and models.
Combines functionality for:
- Spectral Normalization (Stability)
- Lipschitz Constant Tracking
- Custom Training Steps (Heuristic/Contrastive updates)
- Configuration Management
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, Optional, Optional, Tuple, List, Union, Any
from dataclasses import dataclass, field
from torch.nn.utils.parametrizations import spectral_norm


@dataclass
class ModelConfig:
    """Configuration for a bio-plausible model."""
    name: str
    input_dim: int
    output_dim: int
    hidden_dims: List[int] = field(default_factory=list)

    # Training hyperparameters
    learning_rate: float = 0.001
    beta: float = 0.2  # For EqProp
    equilibrium_steps: int = 20

    # Architecture
    use_spectral_norm: bool = True
    activation: str = 'silu'

    # Additional kwargs
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration."""
        # input_dim can be 0 for Conv models (placeholder)
        assert self.input_dim >= 0
        assert self.output_dim > 0


class BioModel(nn.Module, ABC):
    """
    Abstract base class for all bio-plausible models/algorithms.

    Unifies:
    - NEBCBase (Spectral Norm, Lipschitz)
    - BaseAlgorithm (train_step, config)
    """

    algorithm_name: str = "BioModel"

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        # Legacy/Direct init support
        input_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        use_spectral_norm: bool = True,
        **kwargs
    ):
        super().__init__()

        # Handle config vs direct args
        if config is None:
            if input_dim is None or output_dim is None:
                raise ValueError("Must provide either config or input_dim/output_dim")

            # Legacy/Direct init
            self.config = ModelConfig(
                name=self.algorithm_name,
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=[hidden_dim] if hidden_dim else [],
                use_spectral_norm=use_spectral_norm,
                extra=kwargs
            )
        else:
            self.config = config

        # Shortcuts for convenience
        self.input_dim = self.config.input_dim
        self.output_dim = self.config.output_dim
        self.use_spectral_norm = self.config.use_spectral_norm

        # Helper for activation
        self.activation = self._get_activation(self.config.activation)

    def _get_activation(self, name: str) -> nn.Module:
        if name == 'silu': return nn.SiLU()
        if name == 'relu': return nn.ReLU()
        if name == 'tanh': return nn.Tanh()
        if name == 'gelu': return nn.GELU()
        return nn.ReLU()

    def apply_spectral_norm(self, layer: nn.Module) -> nn.Module:
        """Apply spectral normalization to a layer if enabled."""
        if self.use_spectral_norm and isinstance(layer, (nn.Linear, nn.Conv2d)):
            return spectral_norm(layer, n_power_iterations=5)
        return layer

    def compute_lipschitz(self) -> float:
        """Compute the maximum Lipschitz constant across all layers."""
        max_L = 0.0
        with torch.no_grad():
            for module in self.modules():
                # Access .weight property if available (handles spectral_norm)
                if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
                    w = module.weight
                    if w.dim() >= 2:
                        w_mat = w.view(w.size(0), -1)
                        s = torch.linalg.svdvals(w_mat)
                        if s.numel() > 0:
                            max_L = max(max_L, s[0].item())
        return max_L

    def get_stats(self) -> Dict[str, float]:
        """Get algorithm-specific statistics for reporting."""
        return {
            'lipschitz': self.compute_lipschitz(),
            'num_params': sum(p.numel() for p in self.parameters()),
            'spectral_norm': self.use_spectral_norm,
        }

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass."""
        pass

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """
        Custom training step.
        Override this for algorithms that don't use standard autograd (e.g. EqProp, FA).
        If not overridden, EqPropTrainer will assume standard BPTT/Autograd can be used
        if this returns None or raises NotImplementedError, OR EqPropTrainer handles BPTT itself.

        However, for compatibility with BaseAlgorithm, we allow this to be abstract or default to BPTT.
        """
        raise NotImplementedError("Model does not implement custom train_step. Use BPTT.")


class ModelRegistry:
    """Registry for BioModels."""
    _models: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(model_cls: type):
            cls._models[name] = model_cls
            model_cls.algorithm_name = name
            return model_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> type:
        if name not in cls._models:
            raise ValueError(f"Unknown model: {name}. Available: {list(cls._models.keys())}")
        return cls._models[name]

# Convenience
register_model = ModelRegistry.register
