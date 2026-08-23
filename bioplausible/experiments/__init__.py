"""Bioplausible Experiments Package.

Flagship experiments for publishable bio-plausible learning results.
"""

# Import joint experiments (new architecture)
try:
    from .joint import (
        adaptation_efficiency,
        compute_efficiency,
        structural_robustness,
        algorithm_migration,
        z3_fixed_weights,
    )
except ImportError:
    pass

# Legacy experiments (may have missing dependencies)
try:
    from .cross_domain_transfer import TransferConfig, run_transfer_experiment
    from .eqprop_vision_parity import EqPropParityConfig, run_eqprop_parity
    from .fa_depth_scaling import FADepthConfig, run_fa_depth_scaling
    from .mep_tournament import MEPExperimentConfig, run_mep_tournament
    from .mot_ablation import MoTAblationConfig, run_mot_ablation
    from .tile_algorithm_comparison import (
        TileAlgorithmConfig,
        run_tile_algorithm_comparison,
    )
    from .tile_scaling import ScalingConfig, run_scaling_sweep
except ImportError:
    # Some legacy experiments have missing dependencies
    pass

__all__ = [
    # Joint architecture experiments
    "adaptation_efficiency",
    "compute_efficiency",
    "structural_robustness",
    "algorithm_migration",
    "z3_fixed_weights",
    # Legacy (conditionally available)
    "EqPropParityConfig",
    "FADepthConfig",
    "MEPExperimentConfig",
    "MoTAblationConfig",
    "ScalingConfig",
    "TileAlgorithmConfig",
    "TransferConfig",
    "run_eqprop_parity",
    "run_fa_depth_scaling",
    "run_mep_tournament",
    "run_mot_ablation",
    "run_scaling_sweep",
    "run_tile_algorithm_comparison",
    "run_transfer_experiment",
]
