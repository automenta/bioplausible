"""
Bio-Plausible Model Base Classes

Unified foundation for all biologically plausible learning algorithms and models.
Combines functionality for:
- Spectral Normalization (Stability)
- Lipschitz Constant Tracking
- Custom Training Steps (Heuristic/Contrastive updates)
- Configuration Management
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

# Backward-compat re-export: register_model was moved to core/registry.py
from bioplausible.core.registry import register_model  # noqa: F401

__all__ = [
    "BioModel",
    "LayerRole",
    "ModelConfig",
    "compute_hidden_dims",
    "resolve_hidden_dims",
]
LayerRole = Literal["hidden", "output"]


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Configuration for a bio-plausible model."""

    name: str
    input_dim: int
    output_dim: int
    hidden_dims: list[int] = field(default_factory=list)

    # Training hyperparameters
    learning_rate: float = 0.001
    beta: float = 0.2  # For EqProp
    # Equilibrium Steps (also known as max_steps)
    equilibrium_steps: int = 30
    max_steps: int = 30  # Alias for equilibrium_steps to match NEBCBase

    # Architecture
    use_spectral_norm: bool = True
    activation: str = "silu"
    lipschitz_mode: str = "power_iteration"  # "power_iteration" or "svd"

    # μPC (Maximal Update Parameterization) output-node scaling
    # "mupc": output layer skips the √L scaling factor applied to hidden layers
    # "uniform": all layers get the same scaling (backward compat / ablation)
    output_scaling_mode: Literal["uniform", "mupc"] = "mupc"

    # Additional kwargs
    extra: dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration."""
        # input_dim can be 0 for Conv models (placeholder)
        val = self.input_dim
        if isinstance(val, tuple):
            import math

            val = math.prod(val)
        if val < 0:
            raise ValueError(f"input_dim must be >= 0, got {val}")
        # Use object.__setattr__ because frozen=True
        if isinstance(self.input_dim, tuple):
            object.__setattr__(self, "input_dim", val)
        if self.output_dim <= 0:
            raise ValueError(f"output_dim must be > 0, got {self.output_dim}")

        # Sync steps if one is changed
        if self.equilibrium_steps != 30 and self.max_steps == 30:
            object.__setattr__(self, "max_steps", self.equilibrium_steps)
        elif self.max_steps != 30 and self.equilibrium_steps == 30:
            object.__setattr__(self, "equilibrium_steps", self.max_steps)


def resolve_hidden_dims(
    config: ModelConfig | None, hidden_dim: int | None
) -> list[int]:
    """Resolve the ``hidden_dims`` list from a ``ModelConfig`` or fallback.

    Returns ``config.hidden_dims`` if non-empty; otherwise falls back to
    ``[hidden_dim]`` if set; otherwise ``[]``.
    """
    if config is not None and config.hidden_dims:
        return config.hidden_dims
    if hidden_dim is not None:
        return [hidden_dim]
    return []


def compute_hidden_dims(
    hidden_dim: int | None, num_layers: int, max_layers: int = 5
) -> list[int]:
    """Compute a ``hidden_dims`` list for a ``build`` classmethod.

    Returns ``[hidden_dim] * min(num_layers, max_layers)`` when
    ``hidden_dim`` is set, else ``[]``.
    """
    if hidden_dim is None:
        return []
    return [hidden_dim] * min(num_layers, max_layers)


def _build_model_config(
    spec,
    input_dim: int,
    output_dim: int,
    hidden_dim: int | None,
    num_layers: int,
    kwargs: dict[str, object],
    *,
    learning_rate: float | None = None,
    beta: float | None = None,
    equilibrium_steps: int | None = None,
    use_spectral_norm: bool | None = None,
) -> ModelConfig:
    """Construct a ``ModelConfig`` from the standard ``build`` classmethod parameters.

    Handles the common ``spec.name``, ``compute_hidden_dims``, and
    ``kwargs`` wiring. Optional overrides are passed through to the
    ``ModelConfig`` constructor; if *not* provided, the corresponding
    ``ModelConfig`` defaults apply.
    """
    # Extract kwargs overrides that match ModelConfig fields, so we can
    # pass them in the constructor (ModelConfig is frozen).
    effective_lr = learning_rate
    effective_beta = beta
    effective_eq_steps = equilibrium_steps

    kw_beta = kwargs.get("beta")
    if isinstance(kw_beta, float | int):
        effective_beta = kw_beta  # type: ignore[assignment]

    kw_eq_steps = kwargs.get("equilibrium_steps")
    if isinstance(kw_eq_steps, int):
        effective_eq_steps = kw_eq_steps

    config = ModelConfig(
        name=spec.name,
        input_dim=input_dim if input_dim is not None else 0,
        output_dim=output_dim,
        hidden_dims=compute_hidden_dims(hidden_dim, num_layers),
        extra=kwargs,
    )
    # Apply overrides after construction (frozen — use object.__setattr__).
    if effective_lr is not None:
        object.__setattr__(config, "learning_rate", effective_lr)
    if effective_beta is not None:
        object.__setattr__(config, "beta", effective_beta)
    if effective_eq_steps is not None:
        object.__setattr__(config, "equilibrium_steps", effective_eq_steps)
        object.__setattr__(config, "max_steps", effective_eq_steps)
    if use_spectral_norm is not None:
        object.__setattr__(config, "use_spectral_norm", use_spectral_norm)

    return config


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
            if input_dim is None or output_dim is None:
                # If inherited directly without config/dims (e.g. specialized subclass),
                # allow skipping, but warn/fail if methods need them.
                # However, for consistency with NEBCBase, we might need these.
                # Let's assume subclasses will call super().__init__ properly.
                pass

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
            # Ensure max_steps override from kwargs if provided
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
        if name == "silu":
            return nn.SiLU()
        if name == "relu":
            return nn.ReLU()
        if name == "tanh":
            return nn.Tanh()
        if name == "gelu":
            return nn.GELU()
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

        # For BaseAlgorithm compatibility, allow abstract or default to BPTT.
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
