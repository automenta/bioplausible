"""Unified base configuration and factory for TileNet deployments.

Consolidates the configuration hierarchies and factory patterns shared across
the vision, time-series, RL, and graph deployment modules. Shared NN modules
(feature extractors, graph layers, scatter utilities) live in the private
``_feature_extractors`` module and are re-exported by the public deployment
submodules for backward-compatible imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from torch import nn

from bioplausible.core.local_learning import (
    TaskHandler,
    TileAlgorithm,
    TileAlgorithmConfig,
)
from bioplausible.core.model import BioModel

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "ConvDeploymentConfig",
    "DeploymentConfig",
    "GraphDeploymentConfig",
    "RLDeploymentConfig",
    "TemporalDeploymentConfig",
    "build_tile_head",
    "create_deployment_model",
]


# =============================================================================
# Base Deployment Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class DeploymentConfig:
    """Base configuration shared by all TileNet deployments.

    Attributes:
        neurons_per_tile: Number of neurons in each tile.
        tiles_per_layer: Number of tiles per layer.
        num_fc_layers: Number of fully-connected layers in the head.
        learning_rate: Base learning rate.
        dropout: Dropout probability.
        weight_decay: Weight decay coefficient.
        algorithm: Tile algorithm (ep, fa, tp, pc, hebbian, snn).
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
    algorithm: Literal["ep", "fa", "tp", "pc", "hebbian", "snn"] = "ep"
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


def build_tile_head(
    config: DeploymentConfig,
    input_dim: int,
    output_dim: int,
    **kwargs,
) -> TileAlgorithm:
    """Build a substrate ``TileAlgorithm`` head from a deployment config.

    Canonical head construction for the deployment model classes: maps the
    shared deployment fields (topology, algorithm, optimizer knobs) onto
    ``TileAlgorithmConfig`` and injects a ``TaskHandler`` for the target task.
    ``equitile_kwargs`` and any explicit ``kwargs`` spill into the substrate
    ``extra`` bucket (separate per-algorithm knobs).

    Args:
        config: DeploymentConfig subclass instance.
        input_dim: Feature dimension the head consumes (extractor output).
        output_dim: Model output dimension.
        **kwargs: Additional substrate config field overrides.

    Returns:
        A configured TileAlgorithm head.
    """
    extra = dict(config.equitile_kwargs)
    extra.update(kwargs)
    algorithm = getattr(config, "algorithm", config.mode)
    head_config = TileAlgorithmConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        neurons_per_tile=config.neurons_per_tile,
        tiles_per_layer=config.tiles_per_layer,
        num_hidden_layers=config.num_fc_layers,
        algorithm=algorithm,
        mode=config.mode,
        learning_rate=config.learning_rate,
        beta=config.beta,
        step_size=config.step_size,
        free_steps=config.inference_steps,
        nudged_steps=config.inference_steps,
        extra=extra,
    )
    return TileAlgorithm(
        head_config,
        task_handler=TaskHandler(task_type=config.task_type, output_dim=output_dim),
    )


def create_deployment_model(
    config: DeploymentConfig,
    feature_extractor: nn.Module,
    head_input_dim: int,
    head_output_dim: int,
    **kwargs,
) -> BioModel:
    """Create a deployment model with a feature extractor and tile-substrate head.

    Args:
        config: DeploymentConfig subclass instance.
        feature_extractor: Module that extracts features from raw input.
        head_input_dim: Input dimension for the tile-substrate head (feature_extractor output).
        head_output_dim: Output dimension for the tile-substrate head.
        **kwargs: Additional arguments passed to the substrate head config.

    Returns:
        A BioModel with feature_extractor and head attributes.
    """
    head = build_tile_head(config, head_input_dim, head_output_dim, **kwargs)

    class DeploymentModel(BioModel):
        def __init__(self) -> None:
            from bioplausible.config.unified import ModelConfig

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

        def forward(self, x: Tensor, **kwargs) -> Tensor:
            features = self.feature_extractor(x)
            return self.head(features, **kwargs)

        def train_step(self, x: Tensor, y: Tensor) -> dict[str, float]:
            self._step_count += 1
            features = self.feature_extractor(x)
            return self.head.local_update(features.detach(), y)

    return DeploymentModel()
