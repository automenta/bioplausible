"""
EquiTile Vision: Convolutional EquiTile for Image Processing
=============================================================

Extends EquiTile with convolutional capabilities for vision tasks:
- ConvEquiTile: Convolutional tile architecture
- Vision-specific tile configurations
- Image augmentation support
- Vision benchmarks (MNIST, CIFAR-10, ImageNet)

The configuration and feature extractor now inherit from the unified
``DeploymentConfig`` hierarchy in ``deployments/base``; this module only
adds the vision-specific pieces (augmentation, the registered model).
"""

from __future__ import annotations

import math
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
    ConvDeploymentConfig,
    build_tile_head,
)

# Re-export the (now shared) feature extractor under its historical name.
ConvFeatureExtractor = _fe.ConvFeatureExtractor

__all__ = [
    "ConvEquiTile",
    "ConvEquiTileConfig",
    "ConvFeatureExtractor",
    "VisionAugmentation",
    "create_cifar_model",
    "create_imagenet_model",
    "create_mnist_model",
    "create_vision_model",
]
if TYPE_CHECKING:
    from torch import Tensor


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class ConvEquiTileConfig(ConvDeploymentConfig):
    """Configuration for Convolutional EquiTile.

    Inherits the shared deployment fields from ``ConvDeploymentConfig`` and
    keeps the same defaults the historical ``ConvEquiTileConfig`` exposed.
    """

    # Historical vision default learning rate differs from the generic base.
    learning_rate: float = 0.01


# =============================================================================
# Convolutional EquiTile
# =============================================================================


@register_model(
    "conv_equitile",
    domains=[Domain.VISION],
    locality_level=LocalityLevel.LOCAL,
    bio_plausibility_score=0.8,
    requires_backward=False,
    credit_assignment_type="hebbian",
    family="equitile",
    tags=[status_tag("experimental")],
)
class ConvEquiTile(BioModel):
    """Convolutional EquiTile for vision tasks.

    Combines convolutional feature extraction with EquiTile's
    tile-based local learning for the classification head.

    Parameters
    ----------
    config : ConvEquiTileConfig, optional
        Configuration
    **kwargs
        Additional configuration parameters

    Examples
    --------
    >>> config = ConvEquiTileConfig(
    ...     input_channels=3,
    ...     input_size=32,
    ...     num_classes=10,
    ... )
    >>> model = ConvEquiTile(config)
    >>> for images, labels in dataloader:
    ...     stats = model.train_step(images, labels)
    """

    algorithm_name = "ConvEquiTile"

    @classmethod
    def build(
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
        """Build ConvEquiTile from factory arguments."""
        # Handle spatial tuple input_dim (e.g., (1, 28, 28) for MNIST)
        if isinstance(input_dim, tuple):
            input_dim = math.prod(input_dim)
        if input_dim == 784:
            channels, size = 1, 28
        elif input_dim == 3072:
            channels, size = 3, 32
        elif input_dim == 1024:
            channels, size = 1, 32
        else:
            channels = kwargs.get("input_channels", 3)
            size = kwargs.get("input_size", int((input_dim / channels) ** 0.5))

        config_kwargs = {
            "input_channels": channels,
            "input_size": size,
            "num_classes": output_dim,
            "learning_rate": kwargs.get("lr", spec.default_lr),
            "neurons_per_tile": kwargs.get("neurons_per_tile", 64),
            "tiles_per_layer": kwargs.get("tiles_per_layer", 4),
            "num_fc_layers": max(1, num_layers - 2),
        }

        valid_keys = ConvEquiTileConfig.__annotations__.keys()
        for k, v in kwargs.items():
            if k in valid_keys:
                config_kwargs[k] = v

        for k, v in spec.custom_hyperparams.items():
            if k in valid_keys:
                config_kwargs[k] = v

        config = ConvEquiTileConfig(**config_kwargs)

        model = cls(config=config)
        return model.to(device)

    def __init__(
        self,
        config: ConvEquiTileConfig | None = None,
        **kwargs: object,
    ) -> None:
        if config is None:
            config = ConvEquiTileConfig(**kwargs)

        super().__init__(
            ModelConfig(
                name="conv_equitile",
                input_dim=config.input_channels * config.input_size * config.input_size,
                output_dim=config.num_classes,
            )
        )

        self.config = config
        self.input_format = "spatial"  # Signal to CoreTrainer to preserve spatial input

        # Convolutional feature extractor (shared implementation)
        self.feature_extractor = ConvFeatureExtractor(config)

        # EquiTile classification head
        self._build_tile_head(config)

        # Regularization
        self._dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        )

        # State tracking
        self._step_count = 0

    def _build_tile_head(self, config: ConvEquiTileConfig) -> None:
        """Build the tile-substrate classification head."""
        feature_dim = self.feature_extractor.output_size
        self.head = build_tile_head(config, feature_dim, config.num_classes)

    def extract_features(self, x: Tensor) -> Tensor:
        """Extract convolutional features."""
        return self.feature_extractor(x)

    def train_step(self, x: Tensor, y: Tensor) -> dict[str, float]:
        """Perform one training step.

        Parameters
        ----------
        x : torch.Tensor
            Input images (batch, channels, height, width)
        y : torch.Tensor
            Target labels

        Returns
        -------
        dict
            Training statistics
        """
        self._step_count += 1

        features = self.extract_features(x)
        features = self._dropout(features)

        return self.head.local_update(features.detach(), y)

    def forward(
        self,
        x: Tensor,
        return_features: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input images
        return_features : bool
            If True, return features as well

        Returns
        -------
        torch.Tensor or tuple
            Logits, or (logits, features)
        """
        features = self.extract_features(x)
        logits = self.head(features)

        if return_features:
            return logits, features
        return logits


# =============================================================================
# Vision Data Augmentation
# =============================================================================


class VisionAugmentation:
    """Vision data augmentation utilities."""

    def __init__(
        self,
        random_crop: bool = False,
        crop_size: int | None = None,
        random_flip: bool = False,
        color_jitter: bool = False,
        normalize: bool = True,
        mean: tuple[float, ...] = (0.485, 0.456, 0.406),
        std: tuple[float, ...] = (0.229, 0.224, 0.225),
    ) -> None:
        self.random_crop = random_crop
        self.crop_size = crop_size
        self.random_flip = random_flip
        self.color_jitter = color_jitter
        self.normalize = normalize
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def __call__(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        """Apply augmentation."""
        if self.random_crop and self.crop_size:
            x = self._random_crop(x)

        if self.random_flip:
            x = self._random_flip(x)

        if self.color_jitter:
            x = self._color_jitter(x)

        if self.normalize:
            x = (x - self.mean.to(x.device)) / self.std.to(x.device)

        return x

    def _random_crop(self, x: Tensor) -> Tensor:
        """Random crop."""
        b, c, h, w = x.shape
        top = torch.randint(0, h - self.crop_size + 1, (1,)).item()
        left = torch.randint(0, w - self.crop_size + 1, (1,)).item()
        return x[:, :, top : top + self.crop_size, left : left + self.crop_size]

    def _random_flip(self, x: Tensor) -> Tensor:
        """Random horizontal flip."""
        if torch.rand(1) > 0.5:
            return x.flip(-1)
        return x

    def _color_jitter(self, x: Tensor) -> Tensor:
        """Simple color jitter."""
        brightness = torch.empty(1).uniform_(0.8, 1.2).item()
        x = x * brightness

        contrast = torch.empty(1).uniform_(0.8, 1.2).item()
        x = x * contrast

        return x


# =============================================================================
# Factory Functions
# =============================================================================


def create_vision_model(
    input_channels: int = 3,
    input_size: int = 32,
    num_classes: int = 10,
    conv_channels: list[int] | None = None,
    neurons_per_tile: int = 64,
    mode: Literal["pc", "ep"] = "pc",
    **kwargs: object,
) -> ConvEquiTile:
    """Create a ConvEquiTile model for vision tasks."""
    config = ConvEquiTileConfig(
        input_channels=input_channels,
        input_size=input_size,
        num_classes=num_classes,
        conv_channels=conv_channels or [32, 64, 128],
        neurons_per_tile=neurons_per_tile,
        mode=mode,
        **kwargs,
    )
    return ConvEquiTile(config)


def create_mnist_model(
    neurons_per_tile: int = 64,
    **kwargs: object,
) -> ConvEquiTile:
    """Create ConvEquiTile for MNIST."""
    return create_vision_model(
        input_channels=1,
        input_size=28,
        num_classes=10,
        conv_channels=[16, 32, 64],
        neurons_per_tile=neurons_per_tile,
        **kwargs,
    )


def create_cifar_model(
    neurons_per_tile: int = 128,
    **kwargs: object,
) -> ConvEquiTile:
    """Create ConvEquiTile for CIFAR-10/100."""
    return create_vision_model(
        input_channels=3,
        input_size=32,
        num_classes=10,  # or 100 for CIFAR-100
        conv_channels=[64, 128, 256],
        neurons_per_tile=neurons_per_tile,
        use_pooling=True,
        **kwargs,
    )


def create_imagenet_model(
    neurons_per_tile: int = 256,
    num_classes: int = 1000,
    **kwargs: object,
) -> ConvEquiTile:
    """Create ConvEquiTile for ImageNet."""
    conv_channels = [64, 128, 256, 512]
    return create_vision_model(
        input_channels=3,
        input_size=224,
        num_classes=num_classes,
        conv_channels=conv_channels,
        kernel_sizes=[3] * len(conv_channels),
        neurons_per_tile=neurons_per_tile,
        use_pooling=True,
        **kwargs,
    )
