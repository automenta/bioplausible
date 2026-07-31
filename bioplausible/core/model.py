"""
Bio-Plausible Model Base Class.

Extracted from ``zoo/base.py`` so that ``equitile/`` can depend on
``core/`` instead of ``zoo/``.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

from bioplausible.core.config import (
    LayerRole,
    ModelConfig,
    _build_model_config,
)

__all__ = [
    "BioModel",
]


class BioModel(nn.Module, ABC):
    """
    Abstract base class for all bio-plausible models/algorithms.

    Unifies:
    - NEBCBase (Spectral Norm, Lipschitz)
    - BaseAlgorithm (train_step, config)
    """

    algorithm_name: str = "BioModel"

    # Capability declaration for Registry (REFACTOR3 §4).
    provides: list[str] = ["transition_graph", "standard_autograd"]

    def __init__(
        self,
        config: ModelConfig | None = None,
        # Legacy/Direct init support
        input_dim: int | None = None,
        hidden_dim: int | None = None,
        output_dim: int | None = None,
        use_spectral_norm: bool = True,
        max_steps: int = 30,
        lipschitz_mode: str = "power_iteration",
        **kwargs,
    ):
        super().__init__()

        # Handle config vs direct args
        if config is None:
            # Legacy/Direct init
            self.config = ModelConfig(
                name=self.algorithm_name,
                input_dim=input_dim if input_dim is not None else 0,
                output_dim=output_dim if output_dim is not None else 0,
                hidden_dims=[hidden_dim] if hidden_dim else [],
                use_spectral_norm=use_spectral_norm,
                max_steps=max_steps,
                lipschitz_mode=lipschitz_mode,
                extra=kwargs,
            )
        else:
            self.config = config
            if "max_steps" in kwargs:
                self.config.max_steps = kwargs["max_steps"]
                self.config.equilibrium_steps = kwargs["max_steps"]

        # Shortcuts for convenience
        self.input_dim = self.config.input_dim
        self.output_dim = self.config.output_dim
        self.hidden_dim = self.config.hidden_dims[0] if self.config.hidden_dims else 0
        self.use_spectral_norm = self.config.use_spectral_norm
        self.max_steps = self.config.max_steps
        self.lipschitz_mode = self.config.lipschitz_mode

        # Helper for activation
        self.activation = self._get_activation(self.config.activation)

        # NEBCBase compatibility: Check for _build_layers hook
        if hasattr(self, "_build_layers"):
            self._build_layers()

    def _get_activation(self, name: str) -> nn.Module:
        match name:
            case "silu":
                return nn.SiLU()
            case "relu":
                return nn.ReLU()
            case "tanh":
                return nn.Tanh()
            case "gelu":
                return nn.GELU()
            case _:
                return nn.ReLU()

    def apply_spectral_norm(
        self,
        layer: nn.Module,
        layer_role: LayerRole = "hidden",
    ) -> nn.Module:
        """Apply spectral normalization to a layer if enabled.

        Parameters
        ----------
        layer : nn.Module
            The layer to normalize.
        layer_role : LayerRole
            Whether this is a ``"hidden"`` or ``"output"`` layer.
            When ``output_scaling_mode == "mupc"`` and ``layer_role == "output"``,
            the weight is rescaled to remove the √L fan-in factor that is
            present in the default kaiming initialization but should not
            apply to output nodes under μPC.

        Returns
        -------
        nn.Module
            The normalized layer (wrapped or as-is).
        """
        if self.use_spectral_norm and isinstance(layer, (nn.Linear, nn.Conv2d)):
            layer = spectral_norm(layer, n_power_iterations=5)
            if (
                self.config.output_scaling_mode == "mupc"
                and layer_role == "output"
                and isinstance(layer, nn.Linear)
            ):
                fan_in = layer.weight.size(1)
                if fan_in > 0:
                    with torch.no_grad():
                        # μPC: output layer weights should NOT include the √L factor.
                        # Default kaiming init sets std = gain * √(2 / fan_in).
                        # For μPC output, we rescale to std = gain (no fan_in denom).
                        gain = nn.init.calculate_gain("linear")
                        std = gain * (2.0 / fan_in) ** 0.5
                        target_std = gain  # no √L factor
                        layer.weight.mul_(target_std / max(std, 1e-12))
            return layer
        return layer

    def _get_spectral_normalized_weight(self, layer: nn.Module) -> torch.Tensor:
        """Get spectral normalized weight, with caching in eval mode."""
        # Check for cached weight in eval mode
        if not self.training and hasattr(layer, "_cached_sn_weight"):
            return layer._cached_sn_weight

        # Compute normalized weight (.weight triggers spectral_norm if present)
        if hasattr(layer, "parametrizations") and hasattr(
            layer.parametrizations, "weight"
        ):
            weight = layer.weight
        else:
            weight = layer.weight

        # Cache in eval mode
        if not self.training:
            layer._cached_sn_weight = weight.detach()

        return weight

    def train(self, mode: bool = True):
        """Override train to clear caches."""
        super().train(mode)
        if mode:  # Entering training mode, clear cache
            for module in self.modules():
                if hasattr(module, "_cached_sn_weight"):
                    delattr(module, "_cached_sn_weight")
        return self

    def compute_lipschitz(self) -> float:
        """Compute the maximum Lipschitz constant across all layers."""
        max_L = 0.0
        with torch.no_grad():
            for module in self.modules():
                # Access .weight property if available (handles spectral_norm)
                if hasattr(module, "weight") and isinstance(
                    module.weight, torch.Tensor
                ):
                    w = module.weight
                    if w.dim() >= 2:
                        if self.lipschitz_mode == "power_iteration":
                            # Optimization: Use Power Iteration (O(N^2))
                            L = self._approx_spectral_norm(w)
                        elif self.lipschitz_mode == "svd":
                            # Exact SVD (O(N^3))
                            w_mat = w.view(w.size(0), -1)
                            s = torch.linalg.svdvals(w_mat)
                            L = s[0].item() if s.numel() > 0 else 0.0
                        else:
                            # Fallback to SVD for safety
                            w_mat = w.view(w.size(0), -1)
                            s = torch.linalg.svdvals(w_mat)
                            L = s[0].item() if s.numel() > 0 else 0.0

                        max_L = max(max_L, L)
        return max_L

    def _approx_spectral_norm(self, weight: torch.Tensor, n_iter: int = 10) -> float:
        """Approximate spectral norm using power iteration (faster than SVD)."""
        if weight.dim() < 2:
            return 0.0

        w_mat = weight.view(weight.size(0), -1)
        out_dim, in_dim = w_mat.shape

        u = torch.randn(out_dim, device=weight.device)

        # Power iteration
        for _ in range(n_iter):
            # v = W^T u / ||W^T u||
            v = torch.mv(w_mat.t(), u)
            v = F.normalize(v, dim=0, eps=1e-12)

            # u = W v / ||W v||
            u = torch.mv(w_mat, v)
            u = F.normalize(u, dim=0, eps=1e-12)

        # sigma = u^T W v
        return torch.dot(u, torch.mv(w_mat, v)).item()

    def get_stats(self) -> dict[str, float]:
        """Get algorithm-specific statistics for reporting."""
        return {
            "lipschitz": self.compute_lipschitz(),
            "num_params": sum(p.numel() for p in self.parameters()),
            "spectral_norm": self.use_spectral_norm,
        }

    @classmethod
    def create_pair(
        cls, input_dim: int, hidden_dim: int, output_dim: int, **kwargs
    ) -> tuple[BioModel, BioModel]:
        """Create a pair of models: with and without spectral norm (for ablation)."""
        # Note: Uses direct init assuming arguments match __init__
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

    @abstractmethod
    def forward(self, x: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        """Forward pass."""

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        """
        Custom training step.
        Override this for algorithms that don't use standard autograd
        (e.g. EqProp, FA). If not overridden, EqPropTrainer will assume
        standard BPTT/Autograd can be used if this returns None or raises
        NotImplementedError, or EqPropTrainer handles BPTT.
        """
        raise NotImplementedError(
            "Model does not implement custom train_step. Use BPTT."
        )

    @classmethod
    def build(
        cls,
        spec,
        input_dim,
        output_dim,
        hidden_dim,
        num_layers,
        device,
        task_type,
        **kwargs,
    ):
        config = _build_model_config(
            spec,
            input_dim,
            output_dim,
            hidden_dim,
            num_layers,
            kwargs,
            learning_rate=getattr(spec, "default_lr", 0.001),
            beta=0.1,
            equilibrium_steps=20,
            use_spectral_norm=True,
        )
        return cls(config=config).to(device)

    # ------------------------------------------------------------------
    # TransitionGraph protocol (REFACTOR3 §1)
    # ------------------------------------------------------------------
    def transition_modules(self) -> list[nn.Module]:
        """Modules called in order during one forward step.

        Auto-discovers from common patterns:
        ``self.layers: nn.ModuleList``, ``self.forward_layers: nn.ModuleList``,
        or a fallback scan of direct ``nn.Linear``/``nn.Conv*`` children.

        Subclasses with non-standard structure (e.g. ``LoopedMLP``,
        ``HomeostaticEqProp``, ``NeuralCube``) MUST override this method.
        """
        # 1. Explicit ModuleList (most common).
        layers = getattr(self, "layers", None)
        if isinstance(layers, nn.ModuleList):
            return list(layers)
        # 2. Forward layers (DirectedEP).
        forward_layers = getattr(self, "forward_layers", None)
        if isinstance(forward_layers, nn.ModuleList):
            return list(forward_layers)
        # 3. Fallback: scan direct children for Linear/Conv (backward compat).
        modules = [
            m
            for m in self.children()
            if isinstance(
                m,
                (
                    nn.Linear,
                    nn.Conv1d,
                    nn.Conv2d,
                    nn.Conv3d,
                ),
            )
        ]
        if modules:
            return modules
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
