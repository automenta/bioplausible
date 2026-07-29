"""
EquiTile: Scalable Local-Learning Architecture
==============================================

A production-ready, tile-based local learning framework featuring:
- Tile-based parallel architecture
- Local Hebbian weight updates (no global backprop)
- Multi-GPU support with NCCL
- Mixed precision training
- Dynamic tile growth/pruning
- Enhanced EP with LayerNorm and curriculum learning
- Async execution support
- Comprehensive profiling and benchmarking
- Research utilities for experiments

Quick Start
-----------
>>> from bioplausible.equitile import EquiTile
>>> model = EquiTile(
...     neurons_per_tile=64,
...     num_layers=4,
...     tiles_per_layer=4,
...     input_dim=784,
...     output_dim=10,
... )
>>> for X, y in dataloader:
...     stats = model.train_step(X, y)

Modules
-------
core : Core EquiTile implementation
config : Configuration classes
enhanced : Enhanced EP features
dynamics : Tile growth/pruning
async_execution : Async tile processing
distributed : Multi-GPU training (merged with distributed)
distributed : Distributed training
profiler : Performance profiling
builder : Fluent builder API
research : Research utilities
vision : Vision (ConvEquiTile)
language : Language modeling (LMEquiTile)
fast_lm : Fast visualization variant (FastLMEquiTile)
rl : Reinforcement learning (RLEquiTile)
graph : Graph neural networks (GraphEquiTile)
timeseries : Time series modeling
deployment : Model export and optimization

Examples
--------
Basic usage:
>>> from bioplausible.equitile import EquiTile, create_production_config
>>> config = create_production_config()
>>> model = EquiTile(
...     neurons_per_tile=config.neurons_per_tile,
...     num_layers=config.num_layers,
...     tiles_per_layer=config.tiles_per_layer,
...     input_dim=784,
...     output_dim=10,
... )

Builder pattern:
>>> from bioplausible.equitile.builder import EquiTileBuilder
>>> model = (
...     EquiTileBuilder
...     .production(input_dim=784, output_dim=10)
...     .with_learning_rate(0.01)
...     .build()
... )

Multi-GPU:
>>> from bioplausible.equitile import DistributedEquiTile, DistributedConfig
>>> multi_gpu = DistributedEquiTile(model, device_ids=[0, 1, 2, 3])

Async execution:
>>> from bioplausible.equitile import AsyncEquiTile, AsyncConfig
>>> async_model = AsyncEquiTile(model, config=AsyncConfig(n_workers=4))
>>> with async_model.async_context():
...     stats = async_model.train_step(X, y)

Profiling:
>>> from bioplausible.equitile import EquiTileProfiler
>>> profiler = EquiTileProfiler(model)
>>> with profiler.profile():
...     model.train_step(X, y)
>>> profiler.print_report()

Research utilities:
>>> from bioplausible.equitile.research import ExperimentTracker
>>> tracker = ExperimentTracker("my_experiment")
>>> tracker.log_params({"lr": 0.01})
>>> tracker.log_metrics({"loss": 0.5}, step=100)
"""

from bioplausible.core.registry import (
    Domain,
    LocalityLevel,
    register_model,
)

# Internal: builder + enhanced
from bioplausible.equitile._internal.builder import (
    EnhancedEquiTileBuilder,
    EquiTileBuilder,
    InferenceContext,
    TrainingContext,
    build_enhanced_model,
    build_model,
)
from bioplausible.equitile._internal.enhanced import (
    EnhancedEquiTile,
    TileLayerNorm,
    create_enhanced_model,
)

# Analysis: dynamics + profiler + research
from bioplausible.equitile.analysis.dynamics import (
    DynamicEquiTile,
    TileGrowthManager,
    TileMetrics,
    create_dynamic_model,
)
from bioplausible.equitile.analysis.dynamics import (
    DynamicEquiTileConfig as DynamicsConfig,
)
from bioplausible.equitile.analysis.dynamics import (
    TileGrowthConfig as DynamicsTileGrowthConfig,
)
from bioplausible.equitile.analysis.profiler import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkRunner,
    EquiTileProfiler,
    LearningMonitor,
    MemoryProfiler,
    ProfileResult,
    TileStats,
    create_profiler,
    run_benchmark,
)
from bioplausible.equitile.analysis.research import (
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

# Core
from bioplausible.equitile.core import EquiTile, EquiTileEP
from bioplausible.equitile.core.config import (
    AsyncConfig,
    CurriculumConfig,
    DistributedConfig,
    DynamicEquiTileConfig,
    EnhancedEquiTileConfig,
    EquiTileConfig,
    TileGrowthConfig,
    create_dynamic_config,
    create_enhanced_config,
    create_fast_config,
    create_production_config,
    create_research_config,
)
from bioplausible.equitile.core.topology import TileGraph, TileState

# Deployments
from bioplausible.equitile.deployments.deployment import (
    DeploymentChecker,
    EquiTileExporter,
    ExportConfig,
    ModelPruner,
    check_deployment,
    export_model,
    prune_model,
    quantize_model,
)
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
from bioplausible.equitile.language import (
    OptimizedEquiTileTransformerLayer,
    OptimizedLMEquiTile,
    OptimizedTileAttention,
    OptimizedTileFeedForward,
    create_optimized_lm,
    create_optimized_small_lm,
)

# Language models
from bioplausible.equitile.language.canonical import (
    EquiTileTransformerLayer,
    LMEquiTile,
    LMEquiTileConfig,
    PositionalEncoding,
    SimpleTokenizer,
    TileAttention,
    TileFeedForward,
    create_large_lm,
    create_lm_model,
    create_medium_lm,
    create_small_lm,
)
from bioplausible.equitile.language.fast import FastLMConfig, FastLMEquiTile

# Training: async + distributed
from bioplausible.equitile.training import NCCLCommunicator
from bioplausible.equitile.training.async_execution import (
    AsyncConfig as AsyncExecutionConfig,
)
from bioplausible.equitile.training.async_execution import (
    AsyncEquiTile,
    TileProcessor,
    TileResult,
    TileScheduler,
    TileTask,
    create_async_model,
)
from bioplausible.equitile.training.distributed import (
    AsyncTileExecutor,
    DeviceAssignment,
    DistributedEquiTile,
    MixedPrecisionTrainer,
    TileCommunicator,
    create_distributed_model,
    spawn_distributed_worker,
)
from bioplausible.equitile.training.distributed import (
    DistributedConfig as DistributedConfigClass,
)
from bioplausible.equitile.training.distributed import (
    TileGrowthConfig as DistributedGrowthConfig,
)

__all__ = [
    # Core
    "EquiTile",
    "EquiTileEP",
    "TileGraph",
    "TileState",
    # "EdgeParams",  # Removed
    # Config
    "EquiTileConfig",
    "create_production_config",
    "create_research_config",
    "create_fast_config",
    "create_enhanced_config",
    "create_dynamic_config",
    # Distributed configs
    "DistributedConfig",
    "NCCLConfig",
    "AsyncConfig",
    # Enhanced configs
    "EnhancedEquiTileConfig",
    "CurriculumConfig",
    # Dynamics configs
    "TileGrowthConfig",
    "DynamicEquiTileConfig",
    # Enhanced
    "TileLayerNorm",
    "EnhancedEquiTile",
    "create_enhanced_model",
    # Dynamics
    "DynamicsTileGrowthConfig",
    "TileMetrics",
    "TileGrowthManager",
    "DynamicsConfig",
    "DynamicEquiTile",
    "create_dynamic_model",
    # Async execution
    "TileTask",
    "TileResult",
    "TileProcessor",
    "TileScheduler",
    "AsyncExecutionConfig",
    "AsyncEquiTile",
    "create_async_model",
    # Distributed
    "DeviceAssignment",
    "DistributedConfigClass",
    "TileCommunicator",
    "MixedPrecisionTrainer",
    "DistributedGrowthConfig",
    "DistributedEquiTile",
    "NCCLCommunicator",
    "AsyncTileExecutor",
    "spawn_distributed_worker",
    "create_distributed_model",
    # Profiler
    "TileStats",
    "ProfileResult",
    "EquiTileProfiler",
    "LearningMonitor",
    "MemoryProfiler",
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkRunner",
    "create_profiler",
    "run_benchmark",
    # Builder
    "EquiTileBuilder",
    "EnhancedEquiTileBuilder",
    "TrainingContext",
    "InferenceContext",
    "build_model",
    "build_enhanced_model",
    # Research utilities
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
    # Domain-specific: Language
    "LMEquiTile",
    "LMEquiTileConfig",
    "PositionalEncoding",
    "TileAttention",
    "TileFeedForward",
    "EquiTileTransformerLayer",
    "SimpleTokenizer",
    "create_lm_model",
    "create_small_lm",
    "create_medium_lm",
    "create_large_lm",
    # Fast LM
    "FastLMConfig",
    "FastLMEquiTile",
    # Optimized Language
    "OptimizedLMEquiTile",
    "OptimizedTileAttention",
    "OptimizedTileFeedForward",
    "OptimizedEquiTileTransformerLayer",
    "create_optimized_lm",
    "create_optimized_small_lm",
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
    # Deployment
    "EquiTileExporter",
    "ExportConfig",
    "ModelPruner",
    "DeploymentChecker",
    "export_model",
    "quantize_model",
    "prune_model",
    "check_deployment",
    # Registry
    "register_model",
    "Domain",
    "LocalityLevel",
]

# Version managed by top-level bioplausible package.
