"""
Bioplausible: Unified Platform for Bio-Plausible Learning Research

Minimal, clean API for training and experimentation.

Quick Start:
    from bioplausible import CoreTrainer, TrainerConfig

    config = TrainerConfig(
        model="equitile",
        model_kwargs={"input_dim": 784, "hidden_dim": 256, "output_dim": 10},
        optimizer="smep",
        optimizer_kwargs={"lr": 0.01},
        task="mnist",
        epochs=10
    )
    trainer = CoreTrainer(config)
    history = trainer.fit()

Or from YAML:
    trainer = CoreTrainer.from_yaml("config.yaml")
    history = trainer.fit()

Two-Tier Propagator/Model Architecture:
--------------------------------------
The zoo provides two complementary interfaces for bio-plausible learning:

1. **Propagator side** (`bioplausible.zoo.propagators`): Learning rules implemented
   as drop-in `torch.optim.Optimizer` subclasses (`BioOptimizer` /
   `LearningRuleOptimizer`). These mutate parameters of *any* model:
   Backprop, FeedbackAlignment, EqProp, ContrastiveHebbianLearning, MEP
   presets (smep, sdmep, ...). Use via the Registry API:
   `registry.make_optimizer("eq_prop", model.parameters(), model=model)`.

2. **Model side** (`bioplausible.zoo.models`): Learning rules that require
   *model-side* control of the forward/training loop (custom dual-phase passes,
   learned inverse maps, settling dynamics with internal state). These expose
   `train_step(x, y) -> dict[str, float]` instead of `optimizer.step()`.

The propagator stubs for FF, PEPITA, TargetProp, DifferenceTargetProp, and PCN
raise `NotImplementedError` with docstrings pointing to their working
model-side implementations. These are re-exported from
`bioplausible.zoo.propagators` alongside the stubs so registry consumers can
reach them without crossing module boundaries.
"""

# AutoScientist (LLM meta-reasoner)
from bioplausible.autoscientist import (
    AutoScientistBridge,
    AutoScientistCampaign,
    ExperimentProposal,
    ExperimentProposer,
    Hypothesis,
    HypothesisReasoner,
    LLMHypothesisGenerator,
)

# Config
from bioplausible.config import (
    DEFAULT_CONFIGS,
    DatasetConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    PropagatorConfig,
    ScientistConfig,
    SparsityConfig,
    TrainingConfig,
    get_default_config,
    get_named_config,
    list_named_configs,
    register_default_config,
    validate_config,
)
from bioplausible.core.registry import (
    ComponentCategory,
    ComponentMetadata,
    ComputeProfile,
    Domain,
    LocalityLevel,
    Registry,
    list_models,
    register_metric,
    register_model,
    register_optimizer,
    register_propagator,
    register_sparsity,
)
from bioplausible.core.trainer import (
    CoreTrainer,
    TrainerConfig,
    TrainingMetrics,
)

# Data
from bioplausible.data.lm import get_lm_dataset
from bioplausible.data.vision import create_data_loaders, get_vision_dataset

# Domains
from bioplausible.domains import (
    Batch,
    DomainSpec,
    DomainTask,
    DomainType,
    GraphTask,
    LMTask,
    Metrics,
    RLTask,
    ScientificTask,
    TabularTask,
    TaskSplit,
    TimeSeriesTask,
    VisionTask,
    create_domain_task,
    list_domains,
)

# EquiTile top-level package — importing registers all variants
from bioplausible.equitile import EquiTile as _EquiTile  # noqa: F401

# Evaluation
from bioplausible.evaluation import (
    BenchmarkRegistry,
    BenchmarkResult,
    BenchmarkSuiteConfig,
    BenchmarkSuiteResult,
    CrossDomainBenchmarkSuite,
    EvaluatorBase,
    MetricSuite,
    evaluate_model_on_task,
    get_benchmark,
    list_benchmarks,
    run_cross_domain_benchmark,
)

# Scientist (execution engine) - now in execution
from bioplausible.execution.engine import ExecutionEngine
from bioplausible.execution.task import ExperimentTask

# Knowledge Base
from bioplausible.knowledge import (
    DEFAULT_KB,
    KnowledgeBase,
    KnowledgeEntry,
    create_knowledge_base,
)

# Leaderboard
from bioplausible.leaderboard.generator import LeaderboardEntry, LeaderboardGenerator

# Lightning Integration
from bioplausible.lightning_ import (
    BioLightningModule,
    BioOptunaPruner,
    BioPrecisionCallback,
    BioPrecisionMixin,
    BioPredictionWriter,
    BioRayTuneSearch,
    EnergyConvergenceCallback,
    build_trainer,
    run_nas_search,
    run_pl_trial,
    run_pl_trial_with_wandb,
)

# Utilities
from bioplausible.utils import count_parameters

# Zoo
from bioplausible.zoo import models as zoo_models
from bioplausible.zoo import optimizers as zoo_optimizers
from bioplausible.zoo import propagators as zoo_propagators
from bioplausible.zoo import sparsity as zoo_sparsity

# Re-export model-side classes from propagators package (two-tier arch)
from bioplausible.zoo.propagators import (
    DifferenceTargetProp,
    FabricPCGraphPCN,
    ForwardForwardNet,
    PEPITA,
    PredictiveCodingHybrid,
)

# Optimizers / Propagators
from bioplausible.zoo.mep.presets import muon_backprop, smep, smep_fast
from bioplausible.zoo.models.eqprop import (
    BackpropMLP,
    ConvEqProp,
    LoopedMLP,
    MemoryEfficientLoopedMLP,
    TransformerEqProp,
)
from bioplausible.zoo.propagators.eqprop import EqProp
from bioplausible.zoo.propagators.fa import DirectFA, FeedbackAlignment

__version__ = "1.0.0"

__all__ = [
    "DEFAULT_CONFIGS",
    "DEFAULT_KB",
    "AutoScientistBridge",
    "AutoScientistCampaign",
    "BackpropMLP",
    "Batch",
    "BenchmarkRegistry",
    "BenchmarkResult",
    "BenchmarkSuiteConfig",
    "BenchmarkSuiteResult",
    "BioLightningModule",
    "BioOptunaPruner",
    "BioPrecisionCallback",
    "BioPrecisionMixin",
    "BioPredictionWriter",
    "BioRayTuneSearch",
    "ComponentCategory",
    "ComponentMetadata",
    "ComputeProfile",
    "ConvEqProp",
    "CoreTrainer",
    "CrossDomainBenchmarkSuite",
    "DatasetConfig",
    "DirectFA",
    "Domain",
    "DomainSpec",
    "DomainTask",
    "DomainType",
    "EnergyConvergenceCallback",
    "EqProp",
    "EvaluatorBase",
    "ExecutionEngine",
    "ExperimentConfig",
    "ExperimentProposal",
    "ExperimentProposer",
    "ExperimentTask",
    "FeedbackAlignment",
    "GraphTask",
    "Hypothesis",
    "HypothesisReasoner",
    "KnowledgeBase",
    "KnowledgeEntry",
    "LLMHypothesisGenerator",
    "LMTask",
    "LeaderboardEntry",
    "LeaderboardGenerator",
    "LocalityLevel",
    "LoopedMLP",
    "MemoryEfficientLoopedMLP",
    "MetricSuite",
    "Metrics",
    "ModelConfig",
    "OptimizerConfig",
    "PropagatorConfig",
    "RLTask",
    "Registry",
    "ScientificTask",
    "ScientistConfig",
    "SparsityConfig",
    "TabularTask",
    "TaskSplit",
    "TimeSeriesTask",
    "TrainerConfig",
    "TrainingConfig",
    "TrainingMetrics",
    "TransformerEqProp",
    "VisionTask",
    "build_trainer",
    "count_parameters",
    "create_data_loaders",
    "create_domain_task",
    "create_knowledge_base",
    "evaluate_model_on_task",
    "get_benchmark",
    "get_default_config",
    "get_lm_dataset",
    "get_named_config",
    "get_vision_dataset",
    "list_benchmarks",
    "list_domains",
    "list_models",
    "list_named_configs",
    "muon_backprop",
    "register_default_config",
    "register_metric",
    "register_model",
    "register_optimizer",
    "register_propagator",
    "register_sparsity",
    "run_cross_domain_benchmark",
    "run_nas_search",
    "run_pl_trial",
    "run_pl_trial_with_wandb",
    "smep",
    "smep_fast",
    "validate_config",
    "zoo_models",
    "zoo_optimizers",
    "zoo_propagators",
    "zoo_sparsity",
    # Model-side re-exports (two-tier propagator/model architecture)
    "DifferenceTargetProp",
    "FabricPCGraphPCN",
    "ForwardForwardNet",
    "PEPITA",
    "PredictiveCodingHybrid",
]
