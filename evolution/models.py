"""
Model registry and builder implementation.

Provides a centralized registry for building models from configurations,
with proper validation and error handling.
"""

import torch.nn as nn
from typing import Dict, Type, Callable, Optional
import logging

from .base import ModelBuilder
from .breeder import ArchConfig
from .config import TASK_CONFIGS, MODEL_CONSTRAINTS


logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for model builders."""
    
    _builders: Dict[str, Callable] = {}
    
    @classmethod
    def register(cls, model_type: str):
        """Decorator to register a model builder function."""
        def decorator(builder_fn: Callable) -> Callable:
            cls._builders[model_type] = builder_fn
            return builder_fn
        return decorator
    
    @classmethod
    def get_builder(cls, model_type: str) -> Optional[Callable]:
        """Get builder function for a model type."""
        return cls._builders.get(model_type)
    
    @classmethod
    def list_models(cls) -> list:
        """List all registered model types."""
        return list(cls._builders.keys())


class DefaultModelBuilder(ModelBuilder):
    """Default implementation of ModelBuilder using the registry."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def build(self, config: ArchConfig, task: str) -> nn.Module:
        """Build model from configuration."""
        # Validate first
        if not self.validate_config(config, task):
            raise ValueError(f"Invalid config for task {task}")
        
        # Get task info
        task_info = TASK_CONFIGS.get(task, {})
        input_dim = task_info.get('input_dim', 784)
        output_dim = task_info.get('output_dim', 10)
        
        # Get builder
        builder = ModelRegistry.get_builder(config.model_type)
        if builder is None:
            # Fallback to looped_mlp
            self.logger.warning(f"Unknown model type {config.model_type}, using looped_mlp")
            builder = ModelRegistry.get_builder('looped_mlp')
        
        # Build
        try:
            model = builder(config, input_dim, output_dim)
            return model
        except Exception as e:
            self.logger.error(f"Failed to build {config.model_type}: {e}")
            raise
    
    def validate_config(self, config: ArchConfig, task: str) -> bool:
        """Validate configuration."""
        constraints = MODEL_CONSTRAINTS.get(config.model_type, {})
        
        # Check depth
        if 'max_depth' in constraints:
            if config.depth > constraints['max_depth']:
                self.logger.warning(
                    f"Depth {config.depth} exceeds max {constraints['max_depth']}"
                )
                return False
        
        # Check SN requirement
        if constraints.get('requires_sn', False) and not config.use_sn:
            self.logger.warning(f"{config.model_type} requires spectral normalization")
            return False
        
        return True


# ============================================================================
# Register model builders
# ============================================================================

@ModelRegistry.register('looped_mlp')
def build_looped_mlp(config: ArchConfig, input_dim: int, output_dim: int) -> nn.Module:
    """Build LoopedMLP from config."""
    from models import LoopedMLP
    return LoopedMLP(
        input_dim=input_dim,
        hidden_dim=config.width,
        output_dim=output_dim,
        use_spectral_norm=config.use_sn,
        max_steps=config.eq_steps,
    )


@ModelRegistry.register('transformer')
def build_transformer(config: ArchConfig, input_dim: int, output_dim: int) -> nn.Module:
    """Build Transformer from config."""
    from models import CausalTransformerEqProp
    
    # Ensure num_heads divides hidden_dim
    num_heads = config.num_heads
    while config.width % num_heads != 0 and num_heads > 1:
        num_heads -= 1
    
    return CausalTransformerEqProp(
        vocab_size=output_dim,
        hidden_dim=config.width,
        num_layers=min(config.depth, 6),
        num_heads=num_heads,
        eq_steps=config.eq_steps,
    )


@ModelRegistry.register('conv')
def build_conv(config: ArchConfig, input_dim: int, output_dim: int) -> nn.Module:
    """Build ConvNet from config."""
    from models import ModernConvEqProp
    return ModernConvEqProp(eq_steps=config.eq_steps)


@ModelRegistry.register('hebbian')
def build_hebbian(config: ArchConfig, input_dim: int, output_dim: int) -> nn.Module:
    """Build Hebbian chain from config."""
    from models import DeepHebbianChain
    return DeepHebbianChain(
        input_dim=input_dim,
        hidden_dim=config.width,
        output_dim=output_dim,
        num_layers=min(config.depth, 500),  # Limit depth
        use_spectral_norm=config.use_sn,
    )


@ModelRegistry.register('feedback_alignment')
def build_feedback_alignment(config: ArchConfig, input_dim: int, output_dim: int) -> nn.Module:
    """Build Feedback Alignment model from config."""
    from models import FeedbackAlignmentEqProp
    return FeedbackAlignmentEqProp(
        input_dim=input_dim,
        hidden_dim=config.width,
        output_dim=output_dim,
        num_layers=min(config.depth, 20),  # Limit depth
        use_spectral_norm=config.use_sn,
    )
