"""
EquiTile Time Series: Sequential Data Modeling
==============================================

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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

from bioplausible.config.unified import ModelConfig
from bioplausible.core.model import BioModel
from bioplausible.core.model_status import status_tag
from bioplausible.core.registry import Domain, LocalityLevel, register_model
from bioplausible.zoo.models.deployments import _feature_extractors as _fe
from bioplausible.zoo.models.deployments.base import (
    TemporalDeploymentConfig,
    build_tile_head,
)

# Re-export shared temporal components under their historical names so
# ``from bioplausible.zoo.models.deployments.timeseries import ...`` keeps working.
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
class TimeSeriesConfig(TemporalDeploymentConfig):
    """Configuration for Time Series EquiTile.

    Inherits the shared deployment fields from ``TemporalDeploymentConfig`` and
    keeps the same defaults the historical ``TimeSeriesConfig`` exposed.
    """

    # Historical time-series defaults
    learning_rate: float = 1e-3
    dropout: float = 0.1
    num_layers: int = 3
    neurons_per_tile: int = 32
    tiles_per_layer: int = 4
    attention_heads: int = 4
    mode: Literal["pc", "ep", "backprop"] = "backprop"


class _TimeSeriesEncoder(nn.Module):
    """Temporal encoder: input projection, positional encoding, tile layers."""

    def __init__(self, config: TimeSeriesConfig) -> None:
        super().__init__()
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)

        if config.use_positional_encoding:
            self.pos_encoding = _fe.TemporalPositionalEncoding(
                embed_dim=config.hidden_dim,
                max_len=config.seq_len,
                dropout=config.dropout,
            )
        else:
            self.pos_encoding = None

        self.layers = nn.ModuleList([
            _fe.TemporalEquiTileLayer(config, _fe.tile_model_factory)
            for _ in range(config.num_layers)
        ])

    def forward(self, x: Tensor) -> Tensor:
        """Encode ``(batch, seq, input_dim)`` -> ``(batch, seq, hidden_dim)``."""
        x = self.input_proj(x)
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x)
        return x


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

    @classmethod
    def build(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
        cls,
        spec,
        input_dim,
        output_dim,
        hidden_dim,
        num_layers,
        device,
        task_type,
        **kwargs,
    ):
        """Build TimeSeriesEquiTile from factory arguments."""
        model_type = kwargs.get("model_type", "forecasting")
        pred_len = kwargs.get("pred_len", 10)

        config_kwargs = {
            "input_dim": input_dim,
            "seq_len": kwargs.get("seq_len", 100),
            "output_dim": output_dim,
            "pred_len": pred_len,
            "model_type": model_type,
            "task_type": task_type
            or ("regression" if model_type == "forecasting" else "classification"),
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "learning_rate": kwargs.get("lr", spec.default_lr),
            "neurons_per_tile": kwargs.get("neurons_per_tile", 32),
            "tiles_per_layer": kwargs.get("tiles_per_layer", 4),
            "attention_heads": kwargs.get("attention_heads", 4),
        }

        valid_keys = TimeSeriesConfig.__annotations__.keys()
        for k, v in kwargs.items():
            if k in valid_keys:
                config_kwargs[k] = v

        for k, v in spec.custom_hyperparams.items():
            if k in valid_keys:
                config_kwargs[k] = v

        config = TimeSeriesConfig(**config_kwargs)

        model = cls(config=config)
        return model.to(device)

    @staticmethod
    def _head_output_dim(config: TimeSeriesConfig) -> int:
        """Task-dependent head output dimension."""
        if config.model_type == "forecasting":
            return config.pred_len * config.output_dim
        if config.model_type == "anomaly_detection":
            return config.input_dim
        return config.output_dim

    def __init__(
        self,
        config: TimeSeriesConfig | None = None,
        **kwargs,
    ) -> None:
        if config is None:
            config = TimeSeriesConfig(**kwargs)

        head_output_dim = self._head_output_dim(config)

        super().__init__(
            ModelConfig(
                name="timeseries_equitile",
                input_dim=config.input_dim,
                output_dim=head_output_dim,
            )
        )

        self.config = config

        self.feature_extractor = _TimeSeriesEncoder(config)

        # Tile-substrate output head
        self.head = build_tile_head(config, config.hidden_dim, head_output_dim)

        self._step_count = 0
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def _pool_features(self, features: Tensor) -> Tensor:
        """Reduce the encoded sequence into the head's 2D input.

        ``features`` is ``(batch, seq, hidden)``. Forecasting pools to the last
        timestep, classification mean-pools, anomaly detection flattens all
        positions for per-step reconstruction.
        """
        if self.config.model_type == "forecasting":
            return features[:, -1, :]
        if self.config.model_type == "anomaly_detection":
            return features.reshape(-1, features.shape[-1])
        return features.mean(dim=1)

    def _head_output(self, features: Tensor, head_ready: Tensor) -> Tensor:
        """Run the substrate head and reshape into the task's output format."""
        head_out = self.head.forward_logits(head_ready, detach_input=False)

        if self.config.model_type == "anomaly_detection":
            batch, seq = features.shape[:2]
            return head_out.view(batch, seq, self.config.input_dim)
        if self.config.model_type == "forecasting":
            batch, _, _ = features.shape
            return head_out.view(batch, self.config.pred_len, self.config.output_dim)
        return head_out

    def forward(
        self,
        x: Tensor,
        _mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (batch, seq_len, input_dim)
        _mask : torch.Tensor, optional
            Attention mask (unused; kept for API compatibility)

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        features = self.feature_extractor(x)
        head_ready = self._pool_features(features)
        return self._head_output(features, head_ready)

    def train_step(
        self,
        x: Tensor,
        y: Tensor,
        _mask: Tensor | None = None,
    ) -> dict[str, float]:
        """Perform one training step."""
        self._step_count += 1

        features = self.feature_extractor(x)
        return self.head.local_update(features.detach(), y)

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
                predictions = []
                current_input = x.clone()

                for _ in range(steps):
                    pred = self.forward(current_input)
                    # pred is (batch, pred_len, output_dim) -> 3D
                    if pred.dim() == 3:  # ruff: ignore[magic-value-comparison]
                        pred = pred[:, -1:, :]
                    predictions.append(pred)

                    if current_input.shape[1] >= self.config.seq_len:
                        current_input = torch.cat(
                            [current_input[:, 1:, :], pred], dim=1
                        )
                    else:
                        current_input = torch.cat([current_input, pred], dim=1)

                return torch.cat(predictions, dim=1)
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
    **kwargs,
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
    **kwargs,
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
    **kwargs,
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
