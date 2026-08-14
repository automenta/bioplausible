"""
Bioplausible: Unified Platform for Bio-Plausible Learning Research

Minimal, clean API for training and experimentation.

Quick Start:
    from bioplausible import CoreTrainer, TrainerConfig

    config = TrainerConfig(
        model="tile_pc",
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

1. **Learning rules** (``bioplausible.core.local_learning.rules``): Learning rules implemented
   as drop-in `torch.optim.Optimizer` subclasses (`BioOptimizer` /
   `LearningRuleOptimizer`). These mutate parameters of *any* model:
   Backprop, FeedbackAlignment, EqProp, ContrastiveHebbianLearning, MEP
   presets (smep, sdmep, ...). Use via the Registry API:
   `Registry.get(ComponentCategory.OPTIMIZER, "eq_prop")`.

2. **Model side** (`bioplausible.zoo.models`): Learning rules that require
   *model-side* control of the forward/training loop (custom dual-phase passes,
   learned inverse maps, settling dynamics with internal state). These expose
   `train_step(x, y) -> dict[str, float]` instead of `optimizer.step()`.

Some algorithms (FF, PEPITA, TargetProp, PCN) inherently require model-level
control and are registered as models, not propagators. Querying them via
`Registry.get(ComponentCategory.PROPAGATOR, "pepita")` resolves through the
compatibility alias map to the model-side registration
(`Registry.get(ComponentCategory.MODEL, "pepita")`) — no `ValueError` raised.
"""

# Lazy top-level API (Sprint 0.5 module-boundary hardening). `import
# bioplausible` no longer eagerly pulls the entire zoo (torchvision, lightning,
# optuna, ...): names are imported on first attribute access via __getattr__,
# so a lightweight consumer (e.g. `import bioplausible.core.registry`) stays
# fast and dependency-slim. Side-effect model registration is preserved because
# importing any zoo symbol triggers `bioplausible.zoo` (which imports all
# components); consumers that need a registered model must import it (or
# `bioplausible.zoo`) explicitly.

__version__ = "1.0.0"

# name -> (submodule_path, attr_or_None). attr None returns the submodule itself.
_LAZY: dict[str, tuple[str, str | None]] = {
    # AutoScientist (LLM meta-reasoner)
    "AutoScientistBridge": ("bioplausible.autoscientist", "AutoScientistBridge"),
    "AutoScientistCampaign": ("bioplausible.autoscientist", "AutoScientistCampaign"),
    "ExperimentProposal": ("bioplausible.autoscientist", "ExperimentProposal"),
    "ExperimentProposer": ("bioplausible.autoscientist", "ExperimentProposer"),
    "Hypothesis": ("bioplausible.autoscientist", "Hypothesis"),
    "HypothesisReasoner": ("bioplausible.autoscientist", "HypothesisReasoner"),
    # Config
    "DatasetConfig": ("bioplausible.config", "DatasetConfig"),
    "ExperimentConfig": ("bioplausible.config", "ExperimentConfig"),
    "ModelConfig": ("bioplausible.config", "ModelConfig"),
    "OptimizerConfig": ("bioplausible.config", "OptimizerConfig"),
    "PropagatorConfig": ("bioplausible.config", "PropagatorConfig"),
    "ScientistConfig": ("bioplausible.config", "ScientistConfig"),
    "SparsityConfig": ("bioplausible.config", "SparsityConfig"),
    "TrainingConfig": ("bioplausible.config", "TrainingConfig"),
    "get_default_config": ("bioplausible.config", "get_default_config"),
    "get_named_config": ("bioplausible.config", "get_named_config"),
    "list_named_configs": ("bioplausible.config", "list_named_configs"),
    "register_default_config": ("bioplausible.config", "register_default_config"),
    "validate_config": ("bioplausible.config", "validate_config"),
    # Core registry / trainer
    "ComponentCategory": ("bioplausible.core.registry", "ComponentCategory"),
    "ComponentMetadata": ("bioplausible.core.registry", "ComponentMetadata"),
    "ComputeProfile": ("bioplausible.core.registry", "ComputeProfile"),
    "Domain": ("bioplausible.core.registry", "Domain"),
    "LocalityLevel": ("bioplausible.core.registry", "LocalityLevel"),
    "Registry": ("bioplausible.core.registry", "Registry"),
    "list_models": ("bioplausible.core.registry", "list_models"),
    "register_metric": ("bioplausible.core.registry", "register_metric"),
    "register_model": ("bioplausible.core.registry", "register_model"),
    "register_optimizer": ("bioplausible.core.registry", "register_optimizer"),
    "register_propagator": ("bioplausible.core.registry", "register_propagator"),
    "register_sparsity": ("bioplausible.core.registry", "register_sparsity"),
    "CoreTrainer": ("bioplausible.core.trainer", "CoreTrainer"),
    "TrainerConfig": ("bioplausible.core.trainer", "TrainerConfig"),
    "TrainingMetrics": ("bioplausible.core.trainer", "TrainingMetrics"),
    # Data
    "get_lm_dataset": ("bioplausible.data.lm", "get_lm_dataset"),
    "create_data_loaders": ("bioplausible.data.vision", "create_data_loaders"),
    "get_vision_dataset": ("bioplausible.data.vision", "get_vision_dataset"),
    # Domains
    "Batch": ("bioplausible.domains", "Batch"),
    "DomainSpec": ("bioplausible.domains", "DomainSpec"),
    "DomainTask": ("bioplausible.domains", "DomainTask"),
    "DomainType": ("bioplausible.domains", "DomainType"),
    "GraphTask": ("bioplausible.domains", "GraphTask"),
    "LMTask": ("bioplausible.domains", "LMTask"),
    "Metrics": ("bioplausible.domains", "Metrics"),
    "RLTask": ("bioplausible.domains", "RLTask"),
    "ScientificTask": ("bioplausible.domains", "ScientificTask"),
    "TabularTask": ("bioplausible.domains", "TabularTask"),
    "TaskSplit": ("bioplausible.domains", "TaskSplit"),
    "TimeSeriesTask": ("bioplausible.domains", "TimeSeriesTask"),
    "VisionTask": ("bioplausible.domains", "VisionTask"),
    "create_domain_task": ("bioplausible.domains", "create_domain_task"),
    "list_domains": ("bioplausible.domains", "list_domains"),
    # Execution engine / callbacks
    "BaseExecutionCallback": (
        "bioplausible.execution.callbacks",
        "BaseExecutionCallback",
    ),
    "ExecutionCallback": ("bioplausible.execution.callbacks", "ExecutionCallback"),
    "ExecutionEngine": ("bioplausible.execution.engine", "ExecutionEngine"),
    "ExperimentTask": ("bioplausible.execution.task", "ExperimentTask"),
    # Knowledge Base
    "KnowledgeBase": ("bioplausible.knowledge", "KnowledgeBase"),
    "KnowledgeEntry": ("bioplausible.knowledge", "KnowledgeEntry"),
    # Leaderboard
    "LeaderboardEntry": ("bioplausible.leaderboard.generator", "LeaderboardEntry"),
    "LeaderboardGenerator": (
        "bioplausible.leaderboard.generator",
        "LeaderboardGenerator",
    ),
    # Lightning Integration
    "BioLightningModule": ("bioplausible.lightning_", "BioLightningModule"),
    "BioOptunaPruner": ("bioplausible.lightning_", "BioOptunaPruner"),
    "BioPrecisionCallback": ("bioplausible.lightning_", "BioPrecisionCallback"),
    "BioPrecisionMixin": ("bioplausible.lightning_", "BioPrecisionMixin"),
    "BioRayTuneSearch": ("bioplausible.lightning_", "BioRayTuneSearch"),
    "EnergyConvergenceCallback": (
        "bioplausible.lightning_",
        "EnergyConvergenceCallback",
    ),
    "build_trainer": ("bioplausible.lightning_", "build_trainer"),
    "run_nas_search": ("bioplausible.lightning_", "run_nas_search"),
    "run_pl_trial": ("bioplausible.lightning_", "run_pl_trial"),
    "run_pl_trial_with_wandb": ("bioplausible.lightning_", "run_pl_trial_with_wandb"),
    # Utilities
    "count_parameters": ("bioplausible.utils", "count_parameters"),
    # Select zoo symbols (mep presets + eqprop/fa models & propagators)
    "muon_backprop": ("bioplausible.zoo.mep.presets", "muon_backprop"),
    "smep": ("bioplausible.zoo.mep.presets", "smep"),
    "smep_fast": ("bioplausible.zoo.mep.presets", "smep_fast"),
    "BackpropMLP": ("bioplausible.zoo.models.eqprop", "BackpropMLP"),
    "ConvEqProp": ("bioplausible.zoo.models.eqprop", "ConvEqProp"),
    "LoopedMLP": ("bioplausible.zoo.models.eqprop", "LoopedMLP"),
    "MemoryEfficientLoopedMLP": (
        "bioplausible.zoo.models.eqprop",
        "MemoryEfficientLoopedMLP",
    ),
    "TransformerEqProp": ("bioplausible.zoo.models.eqprop", "TransformerEqProp"),
    "EqProp": ("bioplausible.core.local_learning.rules.eqprop", "EqProp"),
    "DirectFA": ("bioplausible.core.local_learning.rules.fa", "DirectFA"),
    "FeedbackAlignment": ("bioplausible.core.local_learning.rules.fa", "FeedbackAlignment"),
}

__all__ = sorted(_LAZY) + ["__version__"]


def __getattr__(name: str) -> object:
    """Lazily import a top-level symbol on first access (PEP 562)."""
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = _LAZY[name]
    module = __import__(module_name, fromlist=[attr] if attr else ["*"])
    value: object = module if attr is None else getattr(module, attr)
    setattr(__import__(__name__), name, value)  # cache on the module
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
