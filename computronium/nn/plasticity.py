"""Plasticity mechanisms for ComputroniumLinear.

Self-contained, torch-only implementations of episode-local plasticity.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import Tensor


class PlasticityType(StrEnum):
    """Enumeration of supported plasticity types."""

    NULL = "null"
    FAST_WEIGHTS = "fast_weights"


@dataclass(frozen=True, slots=True)
class PlasticityConfig:
    """Configuration for plasticity mechanism."""

    plasticity_type: PlasticityType = PlasticityType.NULL
    # FAST_WEIGHTS: dimension of fast weight memory (projected)
    fast_weight_dim: int = 512
    # FAST_WEIGHTS: decay factor per forward pass
    decay: float = 0.9
    # FAST_WEIGHTS: learning rate for Hebbian update
    learning_rate: float = 0.1
    # FAST_WEIGHTS: scale for outer product
    outer_product_scale: float = 1.0
    # FAST_WEIGHTS: modulation scale for output modulation
    modulation_scale: float = 0.01


class NullPlasticity:
    """Null plasticity: no plastic state, no modification."""

    def __init__(self, config: PlasticityConfig | None = None) -> None:
        self.config = config or PlasticityConfig()

    @property
    def psi(self) -> None:
        return None

    def initial_psi(
        self, _batch_size: int = 1, _device: torch.device | None = None
    ) -> dict[str, Tensor]:
        return {}

    def step(
        self, _psi: dict[str, Tensor] | None, _x: Tensor, _output: Tensor
    ) -> dict[str, Tensor]:
        return {}

    def modulate_output(self, output: Tensor, _psi: dict[str, Tensor]) -> Tensor:
        return output

    def get_gradient_contribution(self) -> Tensor | None:
        return None

    def to(self, _device: torch.device) -> NullPlasticity:
        return self


class FastWeightPlasticity:
    """Fast weight plasticity: episode-local Hebbian associative memory.

    Maintains a plastic state psi that accumulates Hebbian associations
    (pre-synaptic activity ⊗ post-synaptic activity) projected to a
    lower-dimensional space. Can modulate output and contributes to
    weight gradients (simulating consolidation).

    ψ dynamics:
        ψ_{t+1} = decay * ψ_t + lr * Proj(outer(pre_t, post_t))

    Modulation:
        output += modulation_scale * (readout_proj @ ψ)  # associative recall
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: PlasticityConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        cfg = config or PlasticityConfig()
        self.in_features = in_features
        self.out_features = out_features
        self.config = cfg
        self._device = device

        # Random projection matrices (fixed per instance, deterministic by dims)
        outer_dim = in_features * out_features
        self._proj_write: Tensor | None = None
        self._proj_read: Tensor | None = None
        self._outer_dim = outer_dim

        # Plastic state ψ (buffer)
        self._psi: Tensor | None = None

    def _init_projections(self, device: torch.device) -> None:
        """Lazy-initialize fixed random projection matrices."""
        if self._proj_write is not None:
            return

        generator = torch.Generator(device=device)
        seed = (
            self.in_features * 12345
            + self.out_features * 67890
            + self.config.fast_weight_dim * 42
        ) % (2**32)
        generator.manual_seed(seed)

        # Write projection: outer_dim -> fast_weight_dim
        self._proj_write = torch.randn(
            self.config.fast_weight_dim,
            self._outer_dim,
            generator=generator,
            device=device,
        ) / (self._outer_dim**0.5)

        # Read projection: fast_weight_dim -> out_features (for modulation)
        self._proj_read = torch.randn(
            self.out_features,
            self.config.fast_weight_dim,
            generator=generator,
            device=device,
        ) / (self.config.fast_weight_dim**0.5)

    @property
    def psi(self) -> Tensor | None:
        return self._psi

    def initial_psi(
        self, batch_size: int = 1, device: torch.device | None = None
    ) -> dict[str, Tensor]:
        """Create initial plastic state ψ (zero vector)."""
        _ = batch_size  # used for API consistency, not for zero tensor
        dev = device or self._device or torch.device("cpu")
        self._init_projections(dev)
        self._psi = torch.zeros(self.config.fast_weight_dim, device=dev)
        return {"fast_weights": self._psi.clone()}

    def step(
        self,
        psi: dict[str, Tensor] | None,
        x: Tensor,
        output: Tensor,
    ) -> dict[str, Tensor]:
        """Update ψ via Hebbian association between input and output.

        Args:
            psi: Current plastic state dict with 'fast_weights' key.
            x: Input activations (batch, in_features).
            output: Output activations (batch, out_features).
        """
        if psi is None or "fast_weights" not in psi:
            psi = self.initial_psi(batch_size=x.shape[0], device=x.device)

        fast_weights = psi["fast_weights"]
        self._init_projections(x.device)
        if self._proj_write is None:
            return {"fast_weights": fast_weights}

        batch_size = x.shape[0]
        psi_update = torch.zeros_like(fast_weights)

        for b in range(batch_size):
            pre = x[b].flatten()
            post = output[b].flatten()
            outer = torch.outer(pre, post).flatten()
            projected = self._proj_write @ outer
            psi_update += projected

        psi_update /= batch_size
        psi_update *= self.config.outer_product_scale

        new_fast_weights = (
            self.config.decay * fast_weights + self.config.learning_rate * psi_update
        )

        self._psi = new_fast_weights
        return {"fast_weights": self._psi.clone()}

    def modulate_output(self, output: Tensor, psi: dict[str, Tensor] | None) -> Tensor:
        """Modulate output using associative recall from fast weights."""
        if psi is None or "fast_weights" not in psi:
            return output

        fast_weights = psi["fast_weights"]
        if fast_weights is None or fast_weights.numel() == 0:
            return output

        self._init_projections(output.device)
        if self._proj_read is None:
            return output
        modulation = self._proj_read @ fast_weights
        return output + self.config.modulation_scale * modulation.unsqueeze(0)

    def get_gradient_contribution(self) -> Tensor | None:
        """Get the fast-weight contribution to persistent weight gradient.

        Returns a tensor of shape (out_features, in_features) representing
        the Hebbian association stored in ψ, un-projected for consolidation.
        """
        if self._psi is None or self._psi.numel() == 0 or self._proj_write is None:
            return None

        outer_approx = self._proj_write.T @ self._psi
        weight_grad = outer_approx.view(self.in_features, self.out_features).T
        return self.config.learning_rate * self.config.outer_product_scale * weight_grad

    def to(self, device: torch.device) -> FastWeightPlasticity:
        """Move internal tensors to device."""
        self._device = device
        if self._proj_write is not None:
            self._proj_write = self._proj_write.to(device)
        if self._proj_read is not None:
            self._proj_read = self._proj_read.to(device)
        if self._psi is not None:
            self._psi = self._psi.to(device)
        return self


def create_plasticity(
    plasticity_type: PlasticityType,
    in_features: int,
    out_features: int,
    config: PlasticityConfig | None = None,
    device: torch.device | None = None,
) -> NullPlasticity | FastWeightPlasticity:
    """Factory to create plasticity mechanism."""
    match plasticity_type:
        case PlasticityType.NULL:
            return NullPlasticity(config)
        case PlasticityType.FAST_WEIGHTS:
            return FastWeightPlasticity(in_features, out_features, config, device)
        case _:
            raise ValueError("Unknown plasticity type")
