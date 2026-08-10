"""Unified base configuration and factory for EquiTile deployments.

Consolidates the configuration hierarchies and factory patterns shared across
the vision, time-series, RL, and graph deployment modules. Shared NN modules
(feature extractors, graph layers, scatter utilities) live in the private
``_feature_extractors`` module and are re-exported by the public deployment
submodules for backward-compatible imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

from bioplausible.core.model import BioModel
from bioplausible.equitile.core import EquiTile
from bioplausible.equitile.core.config import EquiTileConfig

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "ConvDeploymentConfig",
    "DeploymentConfig",
    "GraphDeploymentConfig",
    "RLDeploymentConfig",
    "TemporalDeploymentConfig",
    "create_deployment_model",
]


# =============================================================================
# Base Deployment Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class DeploymentConfig:
    """Base configuration shared by all EquiTile deployments.

    Attributes:
        neurons_per_tile: Number of neurons in each tile.
        tiles_per_layer: Number of tiles per layer.
        num_fc_layers: Number of fully-connected layers in the head.
        learning_rate: Base learning rate.
        dropout: Dropout probability.
        weight_decay: Weight decay coefficient.
        mode: Learning mode (pc, ep, backprop).
        inference_steps: Number of inference/relaxation steps.
        step_size: Step size for relaxation dynamics.
        beta: Beta parameter for EP nudging.
        activation: Activation function.
        task_type: Type of task (classification, regression, etc.).
        equitile_kwargs: Additional kwargs passed to EquiTileConfig.
    """

    neurons_per_tile: int = 64
    tiles_per_layer: int = 4
    num_fc_layers: int = 2
    learning_rate: float = 1e-3
    dropout: float = 0.1
    weight_decay: float = 1e-4
    mode: Literal["pc", "ep", "backprop"] = "pc"
    inference_steps: int = 10
    step_size: float = 0.1
    beta: float = 0.1
    activation: Literal["tanh", "relu", "gelu", "silu"] = "gelu"
    task_type: Literal["classification", "regression", "binary", "multilabel"] = (
        "classification"
    )
    equitile_kwargs: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ConvDeploymentConfig(DeploymentConfig):
    """Configuration for convolutional (vision) deployments."""

    input_channels: int = 3
    input_size: int = 32
    num_classes: int = 10
    conv_channels: list[int] = field(default_factory=lambda: [32, 64, 128])
    kernel_sizes: list[int] = field(default_factory=lambda: [3, 3, 3])
    use_pooling: bool = True
    pooling_size: int = 2


@dataclass(frozen=True, slots=True)
class TemporalDeploymentConfig(DeploymentConfig):
    """Configuration for temporal (time series) deployments."""

    seq_len: int = 100
    pred_len: int = 10
    input_dim: int = 10
    output_dim: int = 1
    model_type: Literal["forecasting", "classification", "anomaly_detection"] = (
        "forecasting"
    )
    hidden_dim: int = 64
    num_layers: int = 3
    attention_heads: int = 4
    use_positional_encoding: bool = True
    use_temporal_attention: bool = True


@dataclass(frozen=True, slots=True)
class RLDeploymentConfig(DeploymentConfig):
    """Configuration for RL deployments."""

    obs_dim: int = 8
    action_dim: int = 4
    action_type: Literal["discrete", "continuous"] = "discrete"
    hidden_dim: int = 128
    num_layers: int = 2
    log_std_init: float = 0.0
    log_std_min: float = -20
    log_std_max: float = 2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5


@dataclass(frozen=True, slots=True)
class GraphDeploymentConfig(DeploymentConfig):
    """Configuration for graph deployments."""

    node_features: int = 10
    hidden_dim: int = 64
    num_classes: int = 2
    num_layers: int = 3
    attention_heads: int = 4
    aggregation: Literal["mean", "sum", "max", "attention"] = "mean"
    readout: Literal["mean", "sum", "max", "attention"] = "mean"


# =============================================================================
# Generic Factory Function
# =============================================================================


def create_deployment_model(
    config: DeploymentConfig,
    feature_extractor: nn.Module,
    head_input_dim: int,
    head_output_dim: int,
    **kwargs,
) -> BioModel:
    """Create a deployment model with a feature extractor and EquiTile head.

    Args:
        config: DeploymentConfig subclass instance.
        feature_extractor: Module that extracts features from raw input.
        head_input_dim: Input dimension for the EquiTile head (feature_extractor output).
        head_output_dim: Output dimension for the EquiTile head.
        **kwargs: Additional arguments passed to EquiTileConfig.

    Returns:
        A BioModel with feature_extractor and head attributes.
    """
    head_kwargs = config.equitile_kwargs.copy()
    head_kwargs.update({
        "neurons_per_tile": config.neurons_per_tile,
        "tiles_per_layer": config.tiles_per_layer,
        "num_layers": config.num_fc_layers + 2,  # input + fc + output
        "learning_rate": config.learning_rate,
        "dropout": config.dropout,
        "weight_decay": config.weight_decay,
        "mode": config.mode,
        "inference_steps": config.inference_steps,
        "step_size": config.step_size,
        "beta": config.beta,
        "activation": config.activation,
        "task_type": config.task_type,
    })
    head_kwargs.update(kwargs)

    head_config = EquiTileConfig(**head_kwargs)

    head = EquiTile(
        config=head_config,
        input_dim=head_input_dim,
        output_dim=head_output_dim,
    )

    class DeploymentModel(BioModel):
        def __init__(self) -> None:
            from bioplausible.core.config import ModelConfig

            super().__init__(
                ModelConfig(
                    name=config.__class__.__name__.replace("Config", "").lower(),
                    input_dim=head_input_dim,
                    output_dim=head_output_dim,
                )
            )
            self.config = config
            self.feature_extractor = feature_extractor
            self.head = head
            self._step_count = 0

            self._optim_feature = torch.optim.Adam(
                self.feature_extractor.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            self._optim_head = torch.optim.Adam(
                self.head.parameters(),
                lr=config.learning_rate,
            )

        def forward(self, x: Tensor, **kwargs) -> Tensor:
            features = self.feature_extractor(x)
            return self.head(features, **kwargs)

        def train_step(self, x: Tensor, y: Tensor) -> dict[str, float]:
            self._step_count += 1
            features = self.feature_extractor(x)
            if config.mode == "backprop":
                logits = self.head(features)
                loss = self.head.task_handler.compute_loss(logits, y)
                self._optim_feature.zero_grad()
                self._optim_head.zero_grad()
                loss.backward()
                self._optim_feature.step()
                self._optim_head.step()
                return {
                    "loss": loss.item(),
                    "accuracy": self.head.compute_metrics(logits, y),
                    "mode": config.mode,
                }
            else:
                return self.head.train_step(features.detach(), y)

    return DeploymentModel()
