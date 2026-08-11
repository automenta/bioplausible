"""
EquiTile Configuration Classes
==============================

Consolidated configuration for all EquiTile components.
"""

from dataclasses import dataclass, field, fields
from typing import Literal

from bioplausible.core.local_learning.config import LocalLearningConfig

# =============================================================================
# Core Configuration
# =============================================================================


__all__ = [
    "AsyncConfig",
    "CurriculumConfig",
    "DistributedConfig",
    "DynamicEquiTileConfig",
    "EnhancedEquiTileConfig",
    "EquiTileConfig",
    "MultiGPUConfig",
    "NCCLConfig",
    "TileGrowthConfig",
    "create_dynamic_config",
    "create_enhanced_config",
    "create_fast_config",
    "create_production_config",
    "create_research_config",
]


@dataclass(frozen=True, slots=True)
class EquiTileConfig(LocalLearningConfig):
    """Main EquiTile configuration.

    Extends :class:`~bioplausible.core.local_learning.config.LocalLearningConfig`
    (architecture, learning, dynamics, task fields inherited) with EquiTile's
    PC/EP/backprop energy-dynamics and importance-sparsity knobs. Fields stay
    flat for ease of use in CLI/Hyperopt.

    Dynamics & Mode (EquiTile-specific)
    ------------------------------------
    mode: 'pc' (predictive coding), 'ep' (equilibrium propagation), or 'backprop'
    lambda_error: Weight of prediction error term in energy
    beta: Nudge strength for EP mode
    beta_anneal: Beta decay factor per step/epoch
    inference_steps_free: Separate steps for free phase (EP)
    inference_steps_nudged: Separate steps for nudged phase (EP)
    use_symmetric_weights: Enforce symmetric weights (for strict energy function)
    ep_init_scale: Initialization scale for EP activities

    Importance & Sparsity (EquiTile-specific)
    ------------------------------------------
    importance_decay: EMA decay for importance tracking
    importance_reg_coef: Regularization coefficient for importance
    sparsity_penalty_coef: Penalty for non-sparse importance
    sparsity_threshold: Threshold for considering a tile "active"
    min_active_fraction: Minimum fraction of active tiles
    """

    # Importance & sparsity (EquiTile-specific)
    importance_decay: float = 0.95
    importance_reg_coef: float = 0.01
    sparsity_penalty_coef: float = 0.05
    sparsity_threshold: float = 0.01
    min_active_fraction: float = 0.1

    # Dynamics & mode (EquiTile-specific)
    mode: Literal["pc", "ep", "backprop"] = "pc"
    lambda_error: float = 0.1
    beta: float = 0.1
    beta_anneal: float = 1.0
    inference_steps_free: int | None = None
    inference_steps_nudged: int | None = None
    use_symmetric_weights: bool = False
    ep_init_scale: float = 0.1

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

    def validate(self):
        """Validate configuration parameters."""
        super().validate()

        if not (0 <= self.sparsity_threshold <= 1):
            raise ValueError(
                f"sparsity_threshold must be in [0, 1], got {self.sparsity_threshold}"
            )
        if not (0 <= self.importance_decay <= 1):
            raise ValueError(
                f"importance_decay must be in [0, 1], got {self.importance_decay}"
            )

        if self.mode not in ("pc", "ep", "backprop"):
            raise ValueError(
                f"Invalid mode {self.mode}, must be one of 'pc', 'ep', 'backprop'"
            )


@dataclass(frozen=True, slots=True)
class EnhancedEquiTileConfig(EquiTileConfig):
    """
    Enhanced configuration for EquiTile with all improvements.
    Inherits from :class:`EquiTileConfig` to avoid field duplication.
    """

    # Normalization
    use_layer_norm: bool = True
    use_batch_norm: bool = False
    norm_eps: float = 1e-6

    # Error Propagation
    use_residual_errors: bool = True
    residual_error_weight: float = 0.1
    use_error_momentum: bool = False
    error_momentum: float = 0.9

    # Learning Rate Adaptation
    per_tile_lr: bool = True
    lr_adaptation_rate: float = 0.01
    lr_adaptation_decay: float = 0.99
    min_lr_ratio: float = 0.1
    max_lr_ratio: float = 10.0

    # Momentum for Weight Updates
    use_weight_momentum: bool = True
    weight_momentum: float = 0.9

    # Weight Initialization
    deep_init: bool = True
    init_scale_factor: float = 1.0

    # Architecture Improvements
    use_skip_connections: bool = True
    skip_connection_weight: float = 0.5

    # Enhanced Tile Importance
    enhanced_importance: bool = True
    importance_competition: bool = True
    importance_entropy_weight: float = 0.01

    # Activity Improvements
    use_activity_clipping: bool = True
    activity_clip_value: float = 5.0
    use_activity_scaling: bool = False

    # Gradient Improvements
    use_gradient_centralization: bool = False

    # Curriculum Learning
    use_curriculum: bool = False
    curriculum_stages: int = 5

    # Monitoring
    track_tile_statistics: bool = True

    @classmethod
    def preset_minimal(cls) -> EnhancedEquiTileConfig:
        """Minimal configuration (all improvements disabled)."""
        return cls(
            use_layer_norm=False,
            use_batch_norm=False,
            use_residual_errors=False,
            per_tile_lr=False,
            use_weight_momentum=False,
            deep_init=False,
            use_skip_connections=False,
            enhanced_importance=False,
            use_curriculum=False,
        )

    @classmethod
    def preset_vision(cls) -> EnhancedEquiTileConfig:
        """Optimized for vision tasks (CNN-like behavior)."""
        return cls(
            use_layer_norm=True,
            use_batch_norm=True,
            use_residual_errors=True,
            per_tile_lr=True,
            use_weight_momentum=True,
            deep_init=True,
            use_skip_connections=True,
            enhanced_importance=True,
            use_gradient_centralization=True,
            dropout=0.2,
            use_curriculum=True,
        )

    @classmethod
    def preset_language(cls) -> EnhancedEquiTileConfig:
        """Optimized for language modeling."""
        return cls(
            use_layer_norm=True,
            use_batch_norm=False,
            use_residual_errors=True,
            per_tile_lr=True,
            use_weight_momentum=True,
            deep_init=True,
            use_skip_connections=False,  # Skip connections can hurt language modeling
            enhanced_importance=True,
            dropout=0.1,
            use_curriculum=True,
        )

    @classmethod
    def preset_rl(cls) -> EnhancedEquiTileConfig:
        """Optimized for reinforcement learning (CartPole, etc.)."""
        return cls(
            use_layer_norm=True,
            use_batch_norm=False,
            use_residual_errors=True,
            per_tile_lr=True,
            use_weight_momentum=True,
            deep_init=True,
            use_skip_connections=True,
            enhanced_importance=True,
            dropout=0.0,  # No dropout for RL
            use_curriculum=False,
        )


# =============================================================================
# Distributed Training Configuration
# =============================================================================


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""

    device_ids: list[int] = field(default_factory=list)
    tile_balance: Literal["round_robin", "layered", "balanced"] = "round_robin"
    communication_backend: Literal["nccl", "gloo", "mpi"] = "nccl"
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    mixed_precision_dtype: Literal["float16", "bfloat16"] = "float16"
    overlap_communication: bool = True
    sync_frequency: int = 1


@dataclass
class MultiGPUConfig:
    """Configuration for multi-GPU training."""

    device_ids: list[int] = field(default_factory=list)
    tile_assignment: Literal["round_robin", "layered", "balanced"] = "round_robin"
    sync_frequency: int = 1
    overlap_comm: bool = True
    async_execution: bool = True
    gradient_accumulation: int = 1

    def __post_init__(self) -> None:
        valid_assignments = {"round_robin", "layered", "balanced"}
        if self.tile_assignment not in valid_assignments:
            raise ValueError(f"tile_assignment must be one of {valid_assignments}")


@dataclass(frozen=True, slots=True)
class NCCLConfig:
    """NCCL communication configuration."""

    world_size: int = 1
    rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "29500"
    backend: str = "nccl"
    timeout_minutes: int = 30
    init_method: str = "env://"

    def to_env(self) -> dict[str, str]:
        return {
            "MASTER_ADDR": self.master_addr,
            "MASTER_PORT": self.master_port,
            "WORLD_SIZE": str(self.world_size),
            "RANK": str(self.rank),
        }


# =============================================================================
# Async Execution Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class AsyncConfig:
    """Configuration for async tile execution."""

    n_workers: int = 4
    use_processes: bool = False
    device_ids: list[int] = field(default_factory=list)
    batch_threshold: int = 32
    priority_alpha: float = 0.5
    priority_beta: float = 0.5


# =============================================================================
# =============================================================================
# Curriculum Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class CurriculumConfig:
    """Curriculum learning configuration."""

    enabled: bool = False
    curriculum_type: Literal["difficulty", "uncertainty", "loss"] = "difficulty"
    n_stages: int = 5
    samples_per_stage: int = 1000
    difficulty_metric: Literal["error", "loss", "uncertainty"] = "error"
    start_easy: bool = True
    auto_progress: bool = True
    progress_threshold: float = 0.1


# =============================================================================
# Tile Dynamics Configuration
# =============================================================================


@dataclass
class TileGrowthConfig:
    """Tile growth and pruning configuration."""

    # Growth
    growth_enabled: bool = True
    growth_threshold: float = 0.5
    growth_cooldown: int = 100
    max_tiles: int = 100
    max_tiles_per_layer: int = 16

    # Pruning
    prune_enabled: bool = True
    prune_threshold: float = 0.05
    prune_cooldown: int = 200
    min_tiles: int = 2
    min_tiles_per_layer: int = 1

    # Merging
    merge_enabled: bool = False
    merge_threshold: float = 0.8
    merge_cooldown: int = 500

    # Splitting
    split_enabled: bool = False
    split_threshold: float = 1.0
    split_cooldown: int = 300

    # General
    error_ema_decay: float = 0.95
    min_age_for_modify: int = 50


@dataclass
class DynamicEquiTileConfig:
    """Dynamic tile architecture configuration."""

    growth: TileGrowthConfig = field(default_factory=TileGrowthConfig)
    merge_enabled: bool = False
    split_enabled: bool = False
    track_history: bool = True
    max_history: int = 1000


# =============================================================================
# Convenience Factory Functions
# =============================================================================


def create_production_config(
    neurons_per_tile: int = 64,
    num_layers: int = 4,
    tiles_per_layer: int = 4,
    **kwargs: object,
) -> EquiTileConfig:
    """Create a production-ready configuration."""
    return EquiTileConfig(
        neurons_per_tile=neurons_per_tile,
        num_layers=num_layers,
        tiles_per_layer=tiles_per_layer,
        mode="pc",
        dropout=0.1,
        weight_decay=1e-4,
        gradient_clip=1.0,
        **kwargs,
    )


def create_research_config(
    neurons_per_tile: int = 64,
    num_layers: int = 4,
    tiles_per_layer: int = 4,
    **kwargs: object,
) -> EquiTileConfig:
    """Create a research configuration for EP studies."""
    return EquiTileConfig(
        neurons_per_tile=neurons_per_tile,
        num_layers=num_layers,
        tiles_per_layer=tiles_per_layer,
        mode="ep",
        beta=0.1,
        beta_anneal=0.99,
        inference_steps_free=15,
        inference_steps_nudged=15,
        **kwargs,
    )


def create_fast_config(
    neurons_per_tile: int = 32,
    num_layers: int = 3,
    tiles_per_layer: int = 2,
    **kwargs: object,
) -> EquiTileConfig:
    """Create a fast configuration for prototyping."""
    return EquiTileConfig(
        neurons_per_tile=neurons_per_tile,
        num_layers=num_layers,
        tiles_per_layer=tiles_per_layer,
        inference_steps=5,
        dropout=0.0,
        **kwargs,
    )


def create_enhanced_config(
    use_layer_norm: bool = True,
    use_curriculum: bool = True,
    curriculum_stages: int = 5,
    **kwargs: object,
) -> EnhancedEquiTileConfig:
    """Create enhanced EP configuration."""
    return EnhancedEquiTileConfig(
        use_layer_norm=use_layer_norm,
        use_curriculum=use_curriculum,
        curriculum_stages=curriculum_stages,
        **kwargs,
    )


def create_dynamic_config(
    growth_enabled: bool = True,
    prune_enabled: bool = True,
    **kwargs: object,
) -> DynamicEquiTileConfig:
    """Create dynamic tile configuration."""
    growth_fields = {f.name for f in fields(TileGrowthConfig)}
    reserved = {"growth_enabled", "prune_enabled"}
    growth_kwargs = {
        k: v for k, v in kwargs.items() if k in growth_fields and k not in reserved
    }
    dynamic_kwargs = {k: v for k, v in kwargs.items() if k not in growth_fields}
    return DynamicEquiTileConfig(
        growth=TileGrowthConfig(
            growth_enabled=growth_enabled,
            prune_enabled=prune_enabled,
            **growth_kwargs,
        ),
        **dynamic_kwargs,
    )
