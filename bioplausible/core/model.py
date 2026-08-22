"""
Bio-Plausible Model Base Class.

Extracted from ``zoo/base.py`` so that ``equitile/`` can depend on
``core/`` instead of ``zoo/``.
"""

from abc import ABC

import torch
from torch import nn

from bioplausible.config.experiment import ModelConfig
from bioplausible.core.checkpoint_mixin import CheckpointMixin
from bioplausible.core.losses import compute_accuracy
from bioplausible.core.spectral_mixin import SpectralMixin
from bioplausible.core.training_mixin import TrainingMixin

__all__ = [
    "BioModel",
]


class BioModel(nn.Module, ABC, TrainingMixin, SpectralMixin, CheckpointMixin):
    """
    Abstract base class for all bio-plausible models/algorithms.

    Unifies:
    - NEBCBase (Spectral Norm, Lipschitz) via SpectralMixin
    - BaseAlgorithm (train_step, config) via TrainingMixin
    - Checkpointing via CheckpointMixin
    """

    algorithm_name: str = "BioModel"

    # Default activation for _get_activation; subclasses can override.
    default_activation: str = "relu"

    # Capability declaration for Registry (REFACTOR3 §4).
    provides: list[str] = ["transition_graph", "standard_autograd"]

    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__()

        self.config = config

        # Shortcuts for convenience
        self.input_dim = self.config.input_dim
        self.output_dim = self.config.output_dim
        self.hidden_dim = self.config.hidden_dims[0] if self.config.hidden_dims else 0
        self.use_spectral_norm = self.config.use_spectral_norm
        self.max_steps = self.config.max_steps
        self.learning_rate = self.config.learning_rate
        self.beta = self.config.beta
        self.lipschitz_mode = self.config.lipschitz_mode
        self.spectral_norm_power_iterations = getattr(
            self.config,
            "spectral_norm_power_iterations",
            5,
        )

        # Helper for activation
        self.activation = self._get_activation(self.config.activation)

        # TrainingMixin expects _step_count
        self._step_count = 0

        # NEBCBase compatibility: Check for _build_layers hook
        if hasattr(self, "_build_layers"):
            self._build_layers()

    def _get_activation(self, name: str) -> nn.Module:
        from bioplausible.core.utils.activations import get_activation

        return get_activation(name, default=self.default_activation)

    def train(self, mode: bool = True):
        """Override train to clear caches."""
        # Call SpectralMixin.train() which calls super().train()
        super().train(mode)
        return self

    def _forward_train(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """Single training forward pass.

        Returns:
            (logits, aux_dict) where aux_dict contains additional metrics
            to include in the train_step return value.

        Override this for models that follow the standard TrainingMixin protocol.
        Models with custom train_step (e.g. EquiTile, EqPropModel) can override
        train_step directly instead.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _forward_train or override train_step"
        )

    def compute_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute loss from logits and targets. Override for custom losses."""
        return torch.nn.functional.cross_entropy(logits, y)

    def compute_metrics(self, logits: torch.Tensor, y: torch.Tensor) -> float:
        """Compute accuracy from logits and targets."""
        return compute_accuracy(logits, y)

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        """Execute one training step using TrainingMixin protocol."""
        return super().train_step(x, y)

    @classmethod
    def create_pair(
        cls, input_dim: int, hidden_dim: int, output_dim: int, **kwargs
    ) -> tuple[BioModel, BioModel]:
        """Create a pair of models: with and without spectral norm (for ablation)."""
        with_sn = cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            use_spectral_norm=True,
            **kwargs,
        )
        without_sn = cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            use_spectral_norm=False,
            **kwargs,
        )
        return with_sn, without_sn

    # ------------------------------------------------------------------
    # TransitionGraph protocol (REFACTOR3 §1)
    # ------------------------------------------------------------------
    def transition_modules(self) -> list[nn.Module]:
        """Modules called in order during one forward step.

        Auto-discovers from common patterns:
        ``self.layers: nn.ModuleList`` or ``self.forward_layers: nn.ModuleList``.

        Subclasses with non-standard structure (e.g. ``LoopedMLP``,
        ``HomeostaticEqProp``, ``NeuralCube``) MUST override this method.
        """
        layers = getattr(self, "layers", None)
        if isinstance(layers, nn.ModuleList):
            return list(layers)
        forward_layers = getattr(self, "forward_layers", None)
        if isinstance(forward_layers, nn.ModuleList):
            return list(forward_layers)

        raise NotImplementedError(
            f"{type(self).__name__} has no transition_modules(). "
            "Define `self.layers: nn.ModuleList[nn.Module]` or implement "
            "transition_modules()."
        )

    def initial_state(self, x: torch.Tensor) -> torch.Tensor:
        """Default: use the input as the initial state."""
        return x

    def readout(self, final_state: torch.Tensor) -> torch.Tensor:
        """Default: return the final state as the output."""
        return final_state

    def num_settling_steps(self) -> int:
        """Default: 1 (feedforward). Override for settling-based models."""
        return 1
