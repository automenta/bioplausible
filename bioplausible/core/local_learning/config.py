"""Base configuration for tile-based local-learning algorithms."""

from dataclasses import dataclass
from typing import Literal

__all__ = ["LocalLearningConfig"]

TaskType = Literal["classification", "regression", "binary", "multilabel"]
ActivationName = Literal["tanh", "relu", "gelu", "silu"]


@dataclass(frozen=True, slots=True)
class LocalLearningConfig:
    """Base config for any tile-based local learning algorithm.

    Architecture, learning, dynamics, and task fields shared by EquiTile
    (PC/EP/backprop) and future tile-based algorithms (FA, target prop,
    hierarchical PC). Concrete algorithms extend this with their own knobs;
    validation lives in ``validate()`` which subclasses chain via
    ``super().validate()``.
    """

    # Architecture
    neurons_per_tile: int = 64
    num_layers: int = 4
    tiles_per_layer: int = 4

    # Learning
    learning_rate: float = 0.01
    importance_lr: float = 0.001
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    dropout: float = 0.1

    # Dynamics shared across algorithms
    inference_steps: int = 10
    step_size: float = 0.1
    clamp_activities: bool = True
    activity_clamp_min: float = -5.0
    activity_clamp_max: float = 5.0
    relaxation_tolerance: float = 1e-4

    # Task & activation
    task_type: TaskType = "classification"
    activation: ActivationName = "gelu"

    def validate(self) -> None:
        """Validate configuration parameters (chain via super() in subclasses)."""
        if self.neurons_per_tile <= 0:
            raise ValueError(
                f"neurons_per_tile must be positive, got {self.neurons_per_tile}"
            )
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.tiles_per_layer <= 0:
            raise ValueError(
                f"tiles_per_layer must be positive, got {self.tiles_per_layer}"
            )

        if self.learning_rate < 0:
            raise ValueError(
                f"learning_rate must be non-negative, got {self.learning_rate}"
            )
        if self.importance_lr < 0:
            raise ValueError(
                f"importance_lr must be non-negative, got {self.importance_lr}"
            )
        if self.weight_decay < 0:
            raise ValueError(
                f"weight_decay must be non-negative, got {self.weight_decay}"
            )

        if not (0 <= self.dropout <= 1):
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")

        if self.inference_steps < 0:
            raise ValueError(
                f"inference_steps must be non-negative, got {self.inference_steps}"
            )
