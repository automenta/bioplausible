"""Continual Learning module for Computronium.

This module contains the continual learning subsystem extracted from system_trainer.py,
including:

- ContinualJointSystem: Joint system adapted for continual learning with task masking
- Arm factories: create_fast_weight_arm, create_ewc_arm, create_backprop_arm,
  create_replay_arm, create_lwf_arm, create_si_arm
- Supporting classes: ReplayBuffer, LwFLoss, SynapticIntelligence
- Configuration & metrics: CLConfig, CLMetrics, compute_cl_metrics
- Stability helpers: create_stability_guard, make_transition_fn, check_stability
- Training functions: run_continual_train_step, _continual_step, _lwf_train_step, _si_train_step
- Runner: run_continual_learning, run_continual_learning_suite
- Constants: CL_NUM_TASKS, CL_CLASSES_PER_TASK, CL_TOTAL_CLASSES, SPLIT_MNIST_TASKS
"""

from computronium.core.continual.arms import (
    create_backprop_arm,
    create_ewc_arm,
    create_fast_weight_arm,
    create_lwf_arm,
    create_replay_arm,
    create_si_arm,
)
from computronium.core.continual.buffers import ReplayBuffer
from computronium.core.continual.constants import (
    CL_CLASSES_PER_TASK,
    CL_NUM_TASKS,
    CL_TOTAL_CLASSES,
    SPLIT_MNIST_TASKS,
)
from computronium.core.continual.losses import LwFLoss, SynapticIntelligence
from computronium.core.continual.metrics import CLConfig, CLMetrics, compute_cl_metrics
from computronium.core.continual.runner import (
    run_continual_learning,
    run_continual_learning_suite,
)
from computronium.core.continual.stability import (
    check_stability,
    create_stability_guard,
    make_composite_state,
    make_transition_fn,
)
from computronium.core.continual.system import ContinualJointSystem
from computronium.core.continual.training import (
    _continual_step,
    _lwf_train_step,
    _masked_task_loss,
    _si_train_step,
    run_continual_train_step,
)

__all__ = [
    # System
    "ContinualJointSystem",
    # Arms
    "create_fast_weight_arm",
    "create_ewc_arm",
    "create_backprop_arm",
    "create_replay_arm",
    "create_lwf_arm",
    "create_si_arm",
    # Buffers
    "ReplayBuffer",
    # Losses
    "LwFLoss",
    "SynapticIntelligence",
    # Metrics & Config
    "CLConfig",
    "CLMetrics",
    "compute_cl_metrics",
    # Stability
    "create_stability_guard",
    "make_transition_fn",
    "make_composite_state",
    "check_stability",
    # Training
    "run_continual_train_step",
    "_continual_step",
    "_lwf_train_step",
    "_si_train_step",
    "_masked_task_loss",
    # Runner
    "run_continual_learning",
    "run_continual_learning_suite",
    # Constants
    "CL_NUM_TASKS",
    "CL_CLASSES_PER_TASK",
    "CL_TOTAL_CLASSES",
    "SPLIT_MNIST_TASKS",
]
