"""Unified TileNet Deployment Models.

Consolidates vision, RL, time-series, and graph deployments into a single
factory with a shared `FeatureExtractor` protocol. Eliminates ~3000 lines
of duplicate boilerplate across the four domain-specific modules.

Usage:
    from computronium.models.deployments import create_deployment_model

    # Vision
    model = create_deployment_model("vision", input_channels=3, num_classes=10)

    # RL
    model = create_deployment_model("rl", obs_dim=8, action_dim=4)

    # Time Series
    model = create_deployment_model("timeseries", input_dim=10, seq_len=100)

    # Graph
    model = create_deployment_model("graph", node_features=10, num_classes=2)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, TypeVar

from torch import nn

from computronium.config.unified import ModelConfig
from computronium.core.local_learning import (
    TaskHandler,
    TileAlgorithm,
    TileAlgorithmConfig,
)
from computronium.core.model import BioModel
from computronium.core.tile.feature_extractors import (
    ConvFeatureExtractor,
    GraphFeatureExtractor,
    TemporalFeatureExtractor,
)
from computronium.models.deployments._feature_extractors import tile_model_factory

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

__all__ = [
    "ConvDeploymentConfig",
    "DeploymentConfig",
    "DeploymentDomain",
    "FeatureExtractor",
    "GraphDeploymentConfig",
    "RLDeploymentConfig",
    "TemporalDeploymentConfig",
    "TileDeploymentModel",
    "create_deployment_model",
    "create_graph_model",
    "create_rl_model",
    "create_timeseries_model",
    "create_vision_model",
    "register_deployment_variants",
]


# =============================================================================
# Domain Enum
# =============================================================================


class DeploymentDomain(StrEnum):
    """Supported deployment domains."""

    VISION = "vision"
    RL = "rl"
    TIMESERIES = "timeseries"
    GRAPH = "graph"


# =============================================================================
# Configuration Hierarchy
# =============================================================================


@dataclass(frozen=True, slots=True)
class DeploymentConfig:
    """Base configuration shared by all TileNet deployments."""

    neurons_per_tile: int = 64
    tiles_per_layer: int = 4
    num_fc_layers: int = 2
    learning_rate: float = 1e-3
    dropout: float = 0.1
    weight_decay: float = 1e-4
    algorithm: str = "ep"
    mode: str = "pc"
    inference_steps: int = 10
    step_size: float = 0.1
    beta: float = 0.1
    activation: str = "gelu"
    task_type: str = "classification"
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
    model_type: str = "forecasting"
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
    action_type: str = "discrete"
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
    aggregation: str = "mean"
    readout: str = "mean"


# =============================================================================
# FeatureExtractor Protocol
# =============================================================================


TConfig = TypeVar("TConfig", bound=DeploymentConfig)


class FeatureExtractor(Protocol):
    """Protocol for domain-specific feature extractors.

    All feature extractors must implement `forward` and expose `output_dim`
    so the deployment head can be sized correctly.
    """

    @property
    def output_dim(self) -> int:
        """Output feature dimension after extraction."""
        ...

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        """Extract features from raw input."""
        ...


# =============================================================================
# Domain-Specific Feature Extractors
# =============================================================================


class VisionFeatureExtractor(nn.Module):
    """Convolutional feature extractor for vision deployments."""

    def __init__(self, config: ConvDeploymentConfig) -> None:
        super().__init__()
        self.config = config
        # Use type ignore since ConvFeatureExtractor expects a protocol
        self.extractor = ConvFeatureExtractor(config)  # type: ignore[arg-type]

    @property
    def output_dim(self) -> int:
        return self.extractor.output_size

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.extractor(x)


class TimeSeriesFeatureExtractor(nn.Module):
    """Temporal feature extractor for time series deployments."""

    def __init__(self, config: TemporalDeploymentConfig) -> None:
        super().__init__()
        self.config = config
        # Use type ignore since TemporalFeatureExtractor expects a protocol
        self.extractor = TemporalFeatureExtractor(config, tile_model_factory)  # type: ignore[arg-type]

    @property
    def output_dim(self) -> int:
        return self.config.hidden_dim

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.extractor(x)


class GraphFeatureExtractorWrapper(nn.Module):
    """Graph feature extractor for graph deployments."""

    def __init__(self, config: GraphDeploymentConfig) -> None:
        super().__init__()
        self.config = config
        # Use type ignore since GraphFeatureExtractor expects a protocol
        self.extractor = GraphFeatureExtractor(config, tile_model_factory)  # type: ignore[arg-type]

    @property
    def output_dim(self) -> int:
        return self.config.hidden_dim

    def forward(self, x: Tensor, edge_index: Tensor, **kwargs) -> Tensor:
        return self.extractor(x, edge_index)


class RLFeatureExtractor(nn.Module):
    """RL feature extractor using tile substrate."""

    def __init__(self, config: RLDeploymentConfig) -> None:
        super().__init__()
        self.config = config

        tile_dim = config.neurons_per_tile * config.tiles_per_layer
        num_hidden_layers = max(0, config.num_layers - 2)

        substrate_kwargs = dict(config.equitile_kwargs)
        substrate_kwargs.update({
            "neurons_per_tile": config.neurons_per_tile,
            "tiles_per_layer": config.tiles_per_layer,
            "learning_rate": config.learning_rate,
            "importance_lr": config.learning_rate * 0.1,
            "beta": config.beta,
        })

        head_config = TileAlgorithmConfig(
            input_dim=config.obs_dim,
            output_dim=tile_dim,
            num_hidden_layers=num_hidden_layers,
            algorithm="ep",
            mode="backprop",
            extra=substrate_kwargs,
        )
        self.tile_model = TileAlgorithm(head_config)

    @property
    def output_dim(self) -> int:
        return self.config.neurons_per_tile * self.config.tiles_per_layer

    def forward(self, obs: Tensor, **kwargs) -> Tensor:
        return self.tile_model(obs)


# =============================================================================
# Feature Extractor Registry
# =============================================================================


_FEATURE_EXTRACTORS: dict[str, Callable[[DeploymentConfig], FeatureExtractor]] = {}


def register_feature_extractor(
    domain: str,
    extractor_factory: Callable[[DeploymentConfig], FeatureExtractor],
) -> None:
    """Register a custom feature extractor for a domain."""
    _FEATURE_EXTRACTORS[domain] = extractor_factory


def get_feature_extractor(domain: str, config: DeploymentConfig) -> FeatureExtractor:
    """Get the feature extractor for a domain."""
    # Default extractors
    default_extractors = {
        DeploymentDomain.VISION: VisionFeatureExtractor,
        DeploymentDomain.RL: RLFeatureExtractor,
        DeploymentDomain.TIMESERIES: TimeSeriesFeatureExtractor,
        DeploymentDomain.GRAPH: GraphFeatureExtractorWrapper,
    }

    # Check registry first, then defaults
    domain_enum = DeploymentDomain(domain.lower())
    extractor_cls = _FEATURE_EXTRACTORS.get(domain, default_extractors.get(domain_enum))
    if extractor_cls is None:
        raise ValueError(
            f"Unknown domain: {domain}. Available: {list(default_extractors.keys())}"
        )
    return extractor_cls(config)


# =============================================================================
# Head Builder
# =============================================================================


def build_tile_head(
    config: DeploymentConfig,
    input_dim: int,
    output_dim: int,
    **kwargs,
) -> TileAlgorithm:
    """Build a substrate TileAlgorithm head from a deployment config."""
    extra = dict(config.equitile_kwargs)
    extra.update(kwargs)
    algorithm = getattr(config, "algorithm", config.mode)
    # TaskHandler expects Literal["classification", "regression", "binary", "multilabel"]
    task_type = (
        config.task_type
        if config.task_type in ("classification", "regression", "binary", "multilabel")  # ruff: ignore[literal-membership]
        else "classification"
    )
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
        task_handler=TaskHandler(task_type=task_type, output_dim=output_dim),
    )


# =============================================================================
# Unified Deployment Model
# =============================================================================


class TileDeploymentModel(BioModel):
    """Unified TileNet deployment model with domain-specific feature extractor."""

    domain: str
    config: DeploymentConfig
    feature_extractor: FeatureExtractor
    head: TileAlgorithm
    _step_count: int

    def __init__(
        self,
        domain: str,
        config: DeploymentConfig,
        feature_extractor: FeatureExtractor,
        head: TileAlgorithm,
        **model_kwargs,
    ) -> None:
        # Determine input/output dims for BioModel
        input_dim = getattr(config, "input_dim", getattr(config, "obs_dim", None))
        if input_dim is None and hasattr(config, "input_channels"):
            input_dim = config.input_channels * config.input_size * config.input_size  # type: ignore[attr-defined]

        output_dim = getattr(
            config,
            "num_classes",
            getattr(config, "action_dim", getattr(config, "output_dim", None)),
        )
        if output_dim is None and hasattr(config, "pred_len"):
            output_dim = config.pred_len * config.output_dim  # type: ignore[attr-defined]

        super().__init__(
            ModelConfig(
                name=f"{domain}_tile",
                input_dim=input_dim or 0,
                output_dim=output_dim or 0,
                **model_kwargs,
            )
        )

        self.domain = domain
        self.config = config
        self.feature_extractor = feature_extractor
        self.head = head
        self._step_count = 0

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        features = self.feature_extractor.forward(x, **kwargs)
        return self.head(features)

    def train_step(self, x: Tensor, y: Tensor, **kwargs) -> dict[str, float]:
        self._step_count += 1
        features = self.feature_extractor.forward(x, **kwargs)
        return self.head.local_update(features.detach(), y)


# =============================================================================
# Factory Function
# =============================================================================


def create_deployment_model(
    domain: str,
    **config_kwargs,
) -> TileDeploymentModel:
    """Create a deployment model for the specified domain.

    Args:
        domain: One of "vision", "rl", "timeseries", "graph"
        **config_kwargs: Configuration parameters for the domain

    Returns:
        A TileDeploymentModel instance
    """
    domain_enum = DeploymentDomain(domain.lower())

    # Domain-specific config classes
    config_classes = {
        DeploymentDomain.VISION: ConvDeploymentConfig,
        DeploymentDomain.RL: RLDeploymentConfig,
        DeploymentDomain.TIMESERIES: TemporalDeploymentConfig,
        DeploymentDomain.GRAPH: GraphDeploymentConfig,
    }

    config_class = config_classes[domain_enum]
    config = config_class(**config_kwargs)

    # Get feature extractor
    feature_extractor = get_feature_extractor(domain, config)

    # Determine head output dimension
    if domain == DeploymentDomain.VISION:
        head_output_dim = config.num_classes
    elif domain == DeploymentDomain.RL:
        head_output_dim = config.action_dim
    elif domain == DeploymentDomain.TIMESERIES:
        if config.model_type == "forecasting":
            head_output_dim = config.pred_len * config.output_dim
        elif config.model_type == "anomaly_detection":
            head_output_dim = config.input_dim
        else:
            head_output_dim = config.output_dim
    elif domain == DeploymentDomain.GRAPH:
        head_output_dim = config.num_classes
    else:
        raise ValueError(f"Unknown domain: {domain}")

    # Build head
    head = build_tile_head(config, feature_extractor.output_dim, head_output_dim)

    return TileDeploymentModel(domain, config, feature_extractor, head)


# =============================================================================
# Backward-Compatible Factory Functions
# =============================================================================


def create_vision_model(
    input_channels: int = 3,
    input_size: int = 32,
    num_classes: int = 10,
    conv_channels: list[int] | None = None,
    neurons_per_tile: int = 64,
    algorithm: str = "ep",
    mode: str = "pc",
    **kwargs,
) -> TileDeploymentModel:
    """Create a vision deployment model (backward compatible)."""
    return create_deployment_model(
        "vision",
        input_channels=input_channels,
        input_size=input_size,
        num_classes=num_classes,
        conv_channels=conv_channels or [32, 64, 128],
        neurons_per_tile=neurons_per_tile,
        algorithm=algorithm,
        mode=mode,
        **kwargs,
    )


def create_rl_model(
    obs_dim: int,
    action_dim: int,
    action_type: str = "discrete",
    hidden_dim: int = 128,
    **kwargs,
) -> TileDeploymentModel:
    """Create an RL deployment model (backward compatible)."""
    return create_deployment_model(
        "rl",
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_type=action_type,
        hidden_dim=hidden_dim,
        **kwargs,
    )


def create_timeseries_model(
    input_dim: int,
    seq_len: int,
    model_type: str = "forecasting",
    pred_len: int = 10,
    **kwargs,
) -> TileDeploymentModel:
    """Create a time series deployment model (backward compatible)."""
    return create_deployment_model(
        "timeseries",
        input_dim=input_dim,
        seq_len=seq_len,
        model_type=model_type,
        pred_len=pred_len,
        **kwargs,
    )


def create_graph_model(
    node_features: int,
    num_classes: int,
    hidden_dim: int = 64,
    num_layers: int = 3,
    **kwargs,
) -> TileDeploymentModel:
    """Create a graph deployment model (backward compatible)."""
    return create_deployment_model(
        "graph",
        node_features=node_features,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        **kwargs,
    )


# =============================================================================
# Algorithm Variant Registration
# =============================================================================


def _credit_assignment_type(algorithm: str) -> str:
    """Map algorithm to credit assignment type."""
    mapping = {
        "ep": "equilibrium",
        "pc": "equilibrium",
        "fa": "target",
        "tp": "target",
        "hebbian": "hebbian",
        "snn": "spiking",
        "gnn": "equilibrium",
    }
    return mapping.get(algorithm, "equilibrium")


def register_deployment_variants(
    base_name: str,
    domain: str,
    config_class: type[DeploymentConfig],
    bio_score: float = 0.75,
) -> None:
    """Register algorithm-specific variants for a deployment model.

    Args:
        base_name: Base model name (e.g., "conv_tile", "rl_tile")
        domain: Deployment domain
        config_class: Config class for the domain
        bio_score: Base bio-plausibility score
    """
    algorithms = {
        "ep": ("equilibrium", 0.8),
        "pc": ("equilibrium", 0.8),
        "fa": ("target", 0.7),
        "tp": ("target", 0.65),
        "hebbian": ("hebbian", 0.6),
        "snn": ("spiking", 0.65),
        "gnn": ("equilibrium", 0.75),
    }

    for algorithm, (credit_type, score) in algorithms.items():

        class _DeploymentVariant(TileDeploymentModel):
            def __init__(self, config=None, **kwargs):
                if config is None:
                    kwargs.setdefault("algorithm", algorithm)
                    config = config_class(**kwargs)
                elif config.algorithm != algorithm:
                    config_dict = config.__dict__.copy()
                    config_dict["algorithm"] = algorithm
                    config = config_class(**config_dict)

                feature_extractor = get_feature_extractor(domain, config)
                head_output_dim = getattr(
                    config,
                    "num_classes",
                    getattr(
                        config,
                        "action_dim",
                        getattr(
                            config, "output_dim", getattr(config, "num_classes", 0)
                        ),
                    ),
                )
                if (
                    domain == "timeseries"
                    and getattr(config, "model_type", "") == "forecasting"
                ):
                    head_output_dim = getattr(config, "pred_len", 10) * getattr(
                        config, "output_dim", 1
                    )
                elif (
                    domain == "timeseries"
                    and getattr(config, "model_type", "") == "anomaly_detection"
                ):
                    head_output_dim = getattr(config, "input_dim", 10)

                head = build_tile_head(
                    config, feature_extractor.output_dim, head_output_dim
                )
                super().__init__(domain, config, feature_extractor, head)


# Register default variants for each domain
register_deployment_variants("conv_tile", "vision", ConvDeploymentConfig)
register_deployment_variants("rl_tile", "rl", RLDeploymentConfig)
register_deployment_variants("timeseries_tile", "timeseries", TemporalDeploymentConfig)
register_deployment_variants("graph_tile", "graph", GraphDeploymentConfig)


# =============================================================================
# Deprecated Module Imports (for backward compatibility)
# =============================================================================

# These imports maintain backward compatibility. They will emit deprecation warnings.
# Users should migrate to `create_deployment_model(domain, ...)` or the
# domain-specific factory functions above.

_DEPRECATED_ATTRS = {
    "ConvTileNet": ("vision", "ConvTileNet"),
    "ConvTileNetConfig": ("vision", "ConvTileNetConfig"),
    "VisionAugmentation": ("vision", "VisionAugmentation"),
    "create_cifar_model": ("vision", "create_cifar_model"),
    "create_imagenet_model": ("vision", "create_imagenet_model"),
    "create_mnist_model": ("vision", "create_mnist_model"),
    "create_vision_model": ("vision", "create_vision_model"),
    "RLTileNet": ("rl", "RLTileNet"),
    "RLTileNetConfig": ("rl", "RLTileNetConfig"),
    "RecurrentRLTileNet": ("rl", "RecurrentRLTileNet"),
    "RolloutBuffer": ("rl", "RolloutBuffer"),
    "compute_gae": ("rl", "compute_gae"),
    "create_atari_model": ("rl", "create_atari_model"),
    "create_mujoco_model": ("rl", "create_mujoco_model"),
    "create_recurrent_rl_model": ("rl", "create_recurrent_rl_model"),
    "create_rl_model": ("rl", "create_rl_model"),
    "TimeSeriesConfig": ("timeseries", "TimeSeriesConfig"),
    "TimeSeriesTileNet": ("timeseries", "TimeSeriesTileNet"),
    "create_anomaly_detection_model": ("timeseries", "create_anomaly_detection_model"),
    "create_classification_model": ("timeseries", "create_classification_model"),
    "create_forecasting_model": ("timeseries", "create_forecasting_model"),
    "GraphTileNet": ("graph", "GraphTileNet"),
    "GraphTileNetConfig": ("graph", "GraphTileNetConfig"),
    "create_molecule_model": ("graph", "create_molecule_model"),
    "create_social_graph_model": ("graph", "create_social_graph_model"),
    "create_graph_model": ("graph", "create_graph_model"),
}


def __getattr__(name: str):
    """Lazy imports with deprecation warnings for backward compatibility."""
    if name in _DEPRECATED_ATTRS:
        module_name, attr_name = _DEPRECATED_ATTRS[name]
        warnings.warn(
            f"Importing {name} from computronium.models.deployments is deprecated. "
            f"Use computronium.models.deployments.{module_name}.{attr_name} instead, "
            f"or use the unified create_deployment_model('{module_name}', ...) factory.",
            DeprecationWarning,
            stacklevel=2,
        )
        module = __import__(
            f"computronium.models.deployments.{module_name}",
            fromlist=[attr_name],
        )
        return getattr(module, attr_name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
