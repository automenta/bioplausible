"""
EquiTile Time Series: Sequential Data Modeling
================================================

Extends EquiTile for time series and sequential data:
- TimeSeriesEquiTile: Recurrent and convolutional architectures
- Temporal attention mechanisms
- Support for forecasting, classification, and anomaly detection
- Multi-variate time series support

The shared temporal layers now live in the private ``_feature_extractors``
module and are re-exported; this module adds the time-series-specific model
(output projections, forecasting, anomaly detection). The time-series model
trains with standard backprop, so its config deliberately excludes the PC/EP
dynamics fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F
from torch import nn

from bioplausible.config.unified import ModelConfig
from bioplausible.core.model import BioModel
from bioplausible.core.model_status import status_tag
from bioplausible.core.registry import Domain, LocalityLevel, register_model
from bioplausible.core.utils.optimizer import OptimizerConfig, create_optimizer
from bioplausible.equitile.deployments import _feature_extractors as _fe

# Re-export shared temporal components under their historical names so
# ``from bioplausible.equitile.deployments.timeseries import ...`` keeps working.
TemporalPositionalEncoding = _fe.TemporalPositionalEncoding
TemporalAttentionLayer = _fe.TemporalAttentionLayer
TimeSeriesEquiTileLayer = _fe.TemporalEquiTileLayer

__all__ = [
    "TemporalAttentionLayer",
    "TemporalPositionalEncoding",
    "TimeSeriesConfig",
    "TimeSeriesEquiTile",
    "TimeSeriesEquiTileLayer",
    "create_anomaly_detection_model",
    "create_classification_model",
    "create_forecasting_model",
]
if TYPE_CHECKING:
    from torch import Tensor


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class TimeSeriesConfig:
    """Configuration for Time Series EquiTile.

    The time-series model trains with standard backprop through its temporal
    EquiTile layers, so it deliberately excludes the PC/EP dynamics fields
    (``mode``, ``inference_steps``, ``step_size``, ``beta``, ``task_type``)
    exposed by the vision/RL deployment configs.
    """

    # Input/Output
    input_dim: int = 10
    seq_len: int = 100
    output_dim: int = 1
    pred_len: int = 10

    # Architecture
    model_type: Literal["forecasting", "classification", "anomaly_detection"] = (
        "forecasting"
    )
    hidden_dim: int = 64
    num_layers: int = 3
    neurons_per_tile: int = 32
    tiles_per_layer: int = 4
    attention_heads: int = 4

    # Temporal settings
    use_positional_encoding: bool = True
    use_temporal_attention: bool = True

    # Learning
    learning_rate: float = 1e-3
    dropout: float = 0.1
    activation: Literal["tanh", "relu", "gelu", "silu"] = "gelu"
    equitile_kwargs: dict[str, object] = field(default_factory=dict)


# =============================================================================
# Time Series EquiTile
# =============================================================================


@register_model(
    "timeseries_equitile",
    domains=[Domain.TIMESERIES],
    locality_level=LocalityLevel.LOCAL,
    bio_plausibility_score=0.75,
    requires_backward=False,
    credit_assignment_type="hebbian",
    family="equitile",
    tags=[status_tag("experimental")],
)
class TimeSeriesEquiTile(BioModel):
    """Time Series EquiTile for sequential data.

    Combines temporal attention with EquiTile's tile-based
    processing for forecasting, classification, and anomaly detection.

    Parameters
    ----------
    config : TimeSeriesConfig, optional
        Configuration
    **kwargs
        Additional configuration parameters
    """

    algorithm_name = "TimeSeriesEquiTile"

    def __init__(
        self,
        config: TimeSeriesConfig | None = None,
        **kwargs: object,
    ) -> None:
        if config is None:
            config = TimeSeriesConfig(**kwargs)

        super().__init__(
            ModelConfig(
                name="timeseries_equitile",
                input_dim=config.input_dim,
                output_dim=config.output_dim,
            )
        )

        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)

        # Positional encoding
        if config.use_positional_encoding:
            self.pos_encoding = TemporalPositionalEncoding(
                embed_dim=config.hidden_dim,
                max_len=config.seq_len,
                dropout=config.dropout,
            )
        else:
            self.pos_encoding = None

        # Time series layers (shared layer implementation)
        self.layers = nn.ModuleList([
            TimeSeriesEquiTileLayer(config, _fe.tile_model_factory)
            for _ in range(config.num_layers)
        ])

        # Output projection based on task
        if config.model_type == "forecasting":
            self.output_proj = nn.Linear(
                config.hidden_dim, config.pred_len * config.output_dim
            )
        elif config.model_type == "classification":
            self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)
        elif config.model_type == "anomaly_detection":
            self.output_proj = nn.Linear(config.hidden_dim, config.input_dim)
        else:
            self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)

        # Optimizer
        self.optimizer = create_optimizer(
            self, OptimizerConfig(name="adam", lr=config.learning_rate)
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (batch, seq_len, input_dim)
        mask : torch.Tensor, optional
            Attention mask

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        batch_size = x.shape[0]

        # Input projection
        x = self.input_proj(x)

        # Positional encoding
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)

        # Time series layers
        for layer in self.layers:
            x = layer(x, mask)

        # Output projection based on task
        if self.config.model_type == "forecasting":
            # Use last time step for forecasting
            x = x[:, -1, :]
            x = self.output_proj(x)
            x = x.view(batch_size, self.config.pred_len, self.config.output_dim)
        elif self.config.model_type == "classification":
            # Global average pooling for classification
            x = x.mean(dim=1)
            x = self.output_proj(x)
        elif self.config.model_type == "anomaly_detection":
            # Reconstruct input
            x = self.output_proj(x)
        else:
            x = self.output_proj(x)

        return x

    def train_step(
        self,
        x: Tensor,
        y: Tensor,
        mask: Tensor | None = None,
    ) -> dict[str, float]:
        """Perform one training step."""
        predictions = self.forward(x, mask)

        if self.config.model_type == "forecasting":
            loss = F.mse_loss(predictions, y)
        elif self.config.model_type == "classification":
            loss = F.cross_entropy(predictions, y)
        elif self.config.model_type == "anomaly_detection":
            # Reconstruction loss
            loss = F.mse_loss(predictions, x)
        else:
            loss = F.mse_loss(predictions, y)

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        self.optimizer.step()

        with torch.no_grad():
            if self.config.model_type == "forecasting":
                mae = F.l1_loss(predictions, y).item()
                metrics = {"mae": mae}
            elif self.config.model_type == "classification":
                accuracy = (predictions.argmax(dim=-1) == y).float().mean().item()
                metrics = {"accuracy": accuracy}
            else:
                metrics = {}

        return {
            "loss": loss.item(),
            **metrics,
        }

    def forecast(
        self,
        x: Tensor,
        steps: int | None = None,
    ) -> Tensor:
        """Make forecasts.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence
        steps : int, optional
            Number of steps to forecast

        Returns
        -------
        torch.Tensor
            Forecasts
        """
        self.eval()
        with torch.no_grad():
            if steps is not None:
                # Auto-regressive forecasting
                predictions = []
                current_input = x.clone()

                for _ in range(steps):
                    pred = self.forward(current_input)
                    if pred.dim() > 2:
                        pred = pred[:, -1:, :]
                    predictions.append(pred)

                    # Update input
                    if current_input.shape[1] >= self.config.seq_len:
                        current_input = torch.cat(
                            [current_input[:, 1:, :], pred], dim=1
                        )
                    else:
                        current_input = torch.cat([current_input, pred], dim=1)

                return torch.cat(predictions, dim=1)
            else:
                return self.forward(x)

    def detect_anomalies(
        self,
        x: Tensor,
        threshold: float = 0.1,
    ) -> tuple[Tensor, Tensor]:
        """Detect anomalies using reconstruction error.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence
        threshold : float
            Anomaly threshold

        Returns
        -------
        tuple
            (anomaly_scores, anomaly_flags)
        """
        self.eval()
        with torch.no_grad():
            reconstruction = self.forward(x)
            error = torch.abs(x - reconstruction).mean(dim=-1)
            anomaly_flags = error > threshold
            return error, anomaly_flags


# =============================================================================
# Factory Functions
# =============================================================================


def create_forecasting_model(
    input_dim: int,
    seq_len: int,
    pred_len: int,
    **kwargs: object,
) -> TimeSeriesEquiTile:
    """Create forecasting model."""
    config = TimeSeriesConfig(
        input_dim=input_dim,
        seq_len=seq_len,
        output_dim=input_dim,
        pred_len=pred_len,
        model_type="forecasting",
        **kwargs,
    )
    return TimeSeriesEquiTile(config)


def create_classification_model(
    input_dim: int,
    seq_len: int,
    num_classes: int,
    **kwargs: object,
) -> TimeSeriesEquiTile:
    """Create classification model."""
    config = TimeSeriesConfig(
        input_dim=input_dim,
        seq_len=seq_len,
        output_dim=num_classes,
        model_type="classification",
        **kwargs,
    )
    return TimeSeriesEquiTile(config)


def create_anomaly_detection_model(
    input_dim: int,
    seq_len: int,
    **kwargs: object,
) -> TimeSeriesEquiTile:
    """Create anomaly detection model."""
    config = TimeSeriesConfig(
        input_dim=input_dim,
        seq_len=seq_len,
        output_dim=input_dim,
        model_type="anomaly_detection",
        **kwargs,
    )
    return TimeSeriesEquiTile(config)
