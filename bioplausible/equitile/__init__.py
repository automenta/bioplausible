"""
EquiTile: Substrate-Backed Scalable Local-Learning Architecture
================================================================

This package is the home of the substrate-backed deployments (vision, graph,
RL, timeseries) and the ``TileLM`` language model, plus the analysis tooling
that was ported onto the ``TileAlgorithm`` substrate in Sprint 2.1. The legacy
``EquiTile``/``EquiTileEP``/``EnhancedEquiTile`` model hierarchies and their
builder/training/deployment-exporter scaffolding were deleted in Sprint 2.2;
the PC-mode substrate model is ``tile_pc`` (``zoo/models/tile_models.py``),
constructible through the generic trainer via ``construct_model``.

Deployments: ConvEquiTile / GraphEquiTile / RLEquiTile / TimeSeriesEquiTile are
stacked on the generic :class:`~bioplausible.core.local_learning.TileAlgorithm`
substrate (feature extractor + ``build_tile_head`` head + split optimizers).
TileLM is the per-position substrate language model (``zoo/models/tile_lm.py``).
"""

from bioplausible.analysis.tile_dynamics import (
    DynamicTileAlgorithm,
    TileGrowthManager,
    TileMetrics,
    create_dynamic_model,
)
from bioplausible.analysis.tile_dynamics import (
    DynamicTileConfig as DynamicsConfig,
)
from bioplausible.analysis.tile_dynamics import (
    TileGrowthConfig as DynamicsTileGrowthConfig,
)
from bioplausible.analysis.tile_profiler import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkRunner,
    LearningMonitor,
    MemoryProfiler,
    ProfileResult,
    TileAlgorithmProfiler,
    TileStats,
    create_profiler,
    run_benchmark,
)
from bioplausible.analysis.tile_research import (
    AblationConfig,
    AblationStudy,
    ExperimentConfig,
    ExperimentTracker,
    MetricCollector,
    MetricEntry,
    VisualizationHelper,
    create_ablation_study,
    create_metric_collector,
    create_tracker,
    create_visualization_helper,
)
from bioplausible.core.tile import TileGraph, TileState

# Deployments
from bioplausible.equitile.deployments.graph import (
    GraphAttentionLayer,
    GraphEquiTile,
    GraphEquiTileConfig,
    GraphEquiTileLayer,
    aggregate_messages,
    create_graph_model,
    create_molecule_model,
    create_social_graph_model,
    scatter_max,
    scatter_mean,
    scatter_sum,
)
from bioplausible.equitile.deployments.rl import (
    RecurrentRLEquiTile,
    RLEquiTile,
    RLEquiTileConfig,
    RolloutBuffer,
    compute_gae,
    create_atari_model,
    create_mujoco_model,
    create_recurrent_rl_model,
    create_rl_model,
)
from bioplausible.equitile.deployments.timeseries import (
    TemporalAttentionLayer,
    TemporalPositionalEncoding,
    TimeSeriesConfig,
    TimeSeriesEquiTile,
    TimeSeriesEquiTileLayer,
    create_anomaly_detection_model,
    create_classification_model,
    create_forecasting_model,
)
from bioplausible.equitile.deployments.vision import (
    ConvEquiTile,
    ConvEquiTileConfig,
    ConvFeatureExtractor,
    VisionAugmentation,
    create_cifar_model,
    create_imagenet_model,
    create_mnist_model,
    create_vision_model,
)
from bioplausible.zoo.models.tile_lm import TileLM
from bioplausible.zoo.models.tile_models import TilePC

__all__ = [  # ruff: ignore[unsorted-dunder-all]  (intentional domain-grouped export order)
    "TileGraph",
    "TileState",
    "TileLM",
    "TilePC",
    # Analysis (substrate-native, ported Sprint 2.1)
    "TileMetrics",
    "TileGrowthManager",
    "DynamicsConfig",
    "DynamicsTileGrowthConfig",
    "DynamicTileAlgorithm",
    "create_dynamic_model",
    "TileStats",
    "ProfileResult",
    "TileAlgorithmProfiler",
    "LearningMonitor",
    "MemoryProfiler",
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkRunner",
    "create_profiler",
    "run_benchmark",
    "ExperimentConfig",
    "ExperimentTracker",
    "MetricEntry",
    "MetricCollector",
    "VisualizationHelper",
    "AblationConfig",
    "AblationStudy",
    "create_tracker",
    "create_metric_collector",
    "create_visualization_helper",
    "create_ablation_study",
    # Domain-specific: Vision
    "ConvEquiTile",
    "ConvEquiTileConfig",
    "ConvFeatureExtractor",
    "VisionAugmentation",
    "create_vision_model",
    "create_mnist_model",
    "create_cifar_model",
    "create_imagenet_model",
    # Domain-specific: RL
    "RLEquiTile",
    "RLEquiTileConfig",
    "RecurrentRLEquiTile",
    "RolloutBuffer",
    "compute_gae",
    "create_rl_model",
    "create_recurrent_rl_model",
    "create_atari_model",
    "create_mujoco_model",
    # Domain-specific: Graph
    "GraphEquiTile",
    "GraphEquiTileConfig",
    "GraphAttentionLayer",
    "GraphEquiTileLayer",
    "aggregate_messages",
    "scatter_mean",
    "scatter_sum",
    "scatter_max",
    "create_graph_model",
    "create_molecule_model",
    "create_social_graph_model",
    # Domain-specific: Time Series
    "TimeSeriesEquiTile",
    "TimeSeriesConfig",
    "TemporalPositionalEncoding",
    "TemporalAttentionLayer",
    "TimeSeriesEquiTileLayer",
    "create_forecasting_model",
    "create_classification_model",
    "create_anomaly_detection_model",
]
