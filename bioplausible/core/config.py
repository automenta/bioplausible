"""
Core Configuration.

Frozen dataclass for model configuration, extracted from ``zoo/base.py``
so that ``equitile/`` can depend on ``core/`` instead of ``zoo/``.
"""

from dataclasses import dataclass, field
from typing import Literal

__all__ = [
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
