"""Trainers.

RLTrainer is a standalone trainer for reinforcement learning trajectories,
decoupled from the supervised SystemTrainer because the RL flow has a
different shape (no fixed DataLoader; samples come from an environment).
"""

from .rl import RLTrainer

__all__ = ["RLTrainer"]
