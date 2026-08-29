"""SystemTrainer package: Orchestrates the 5-Layer and 6-D Ontology Pipeline.

This package provides:
- SystemTrainer: Training loop for 5-D composable systems
- Factory functions for composing 5-D and 6-D systems
- Configuration classes and protocols
"""

from computronium.core.system_trainer.config import (
    JointSystem,
    SystemTrainerConfig,
    _DataProvider,
    TS,
    TG,
    TD,
    TC,
    TU,
)
from computronium.core.system_trainer.factory import (
    compose_system,
    compose_system_from_configs,
    create_backprop_system,
    create_eqprop_system,
    create_fa_system,
    extract_config,
)
from computronium.core.system_trainer.joint import (
    compose_joint_system,
    compose_joint_system_from_configs,
    create_routing_eqprop_system,
    create_fast_weight_eqprop_system,
)
from computronium.core.system_trainer.trainer import SystemTrainer

# Continual Learning (extracted to computronium.core.continual)
# Re-export for backward compatibility
from computronium.core.continual import (
    CL_CLASSES_PER_TASK,
    CL_NUM_TASKS,
    CL_TOTAL_CLASSES,
    SPLIT_MNIST_TASKS,
    CLConfig,
    CLMetrics,
    ContinualJointSystem,
    LwFLoss,
    ReplayBuffer,
    SynapticIntelligence,
    _continual_step,
    _lwf_train_step,
    _masked_task_loss,
    _si_train_step,
    check_stability,
    compute_cl_metrics,
    create_backprop_arm,
    create_ewc_arm,
    create_fast_weight_arm,
    create_lwf_arm,
    create_replay_arm,
    create_si_arm,
    create_stability_guard,
    make_composite_state,
    make_transition_fn,
    run_continual_learning,
    run_continual_learning_suite,
    run_continual_train_step,
)

__all__ = [
    "SystemTrainer",
    "SystemTrainerConfig",
    "compose_joint_system",
    "compose_joint_system_from_configs",
    "compose_system",
    "compose_system_from_configs",
    "create_backprop_system",
    "create_eqprop_system",
    "create_fa_system",
    "create_routing_eqprop_system",
    "create_fast_weight_eqprop_system",
    "extract_config",
    # Continual Learning re-exports
    "CL_CLASSES_PER_TASK",
    "CL_NUM_TASKS",
    "CL_TOTAL_CLASSES",
    "SPLIT_MNIST_TASKS",
    "CLConfig",
    "CLMetrics",
    "ContinualJointSystem",
    "LwFLoss",
    "ReplayBuffer",
    "SynapticIntelligence",
    "_continual_step",
    "_lwf_train_step",
    "_masked_task_loss",
    "_si_train_step",
    "check_stability",
    "compute_cl_metrics",
    "create_backprop_arm",
    "create_ewc_arm",
    "create_fast_weight_arm",
    "create_lwf_arm",
    "create_replay_arm",
    "create_si_arm",
    "create_stability_guard",
    "make_composite_state",
    "make_transition_fn",
    "run_continual_learning",
    "run_continual_learning_suite",
    "run_continual_train_step",
]