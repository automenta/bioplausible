"""Layer 5: ParameterUpdate — The Optimization Rule."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
from torch import Tensor

if TYPE_CHECKING:
    from computronium.ontology.geometry import Geometry


# ============================================================
# ParameterUpdate Configuration
# ============================================================


@dataclass(frozen=True, slots=True)
class ParameterUpdateConfig:
    """Configuration for parameter update rule.

    Attributes:
        update_type: "riemannian_orthogonal", "spectral_constrained",
            "natural_gradient", "elastic_consolidation", "euclidean"
        step_size: Learning rate
        momentum: Momentum coefficient (for euclidean)
        ortho_steps: Newton-Schulz iterations (for riemannian)
        spectral_norm: Target spectral norm (for spectral_constrained)
        fisher_damping: Damping for natural gradient
        ewc_lambda: EWC regularization strength
        grad_clip: Gradient clipping norm (for euclidean)
    """

    update_type: str
    step_size: float
    momentum: float
    ortho_steps: int
    spectral_norm: float
    fisher_damping: float
    ewc_lambda: float
    grad_clip: float = 1.0

    @classmethod
    def euclidean(
        cls,
        *,
        step_size: float = 0.01,
        momentum: float = 0.9,
        ortho_steps: int = 5,
        spectral_norm: float = 1.0,
        fisher_damping: float = 1e-3,
        ewc_lambda: float = 1000.0,
        grad_clip: float = 1.0,
    ) -> ParameterUpdateConfig:
        return cls(
            update_type="euclidean",
            step_size=step_size,
            momentum=momentum,
            ortho_steps=ortho_steps,
            spectral_norm=spectral_norm,
            fisher_damping=fisher_damping,
            ewc_lambda=ewc_lambda,
            grad_clip=grad_clip,
        )

    @classmethod
    def riemannian_orthogonal(
        cls,
        *,
        step_size: float = 0.01,
        momentum: float = 0.9,
        ortho_steps: int = 5,
        spectral_norm: float = 1.0,
        fisher_damping: float = 1e-3,
        ewc_lambda: float = 1000.0,
    ) -> ParameterUpdateConfig:
        return cls(
            update_type="riemannian_orthogonal",
            step_size=step_size,
            momentum=momentum,
            ortho_steps=ortho_steps,
            spectral_norm=spectral_norm,
            fisher_damping=fisher_damping,
            ewc_lambda=ewc_lambda,
        )

    @classmethod
    def spectral_constrained(
        cls,
        *,
        step_size: float = 0.01,
        momentum: float = 0.9,
        ortho_steps: int = 5,
        spectral_norm: float = 1.0,
        fisher_damping: float = 1e-3,
        ewc_lambda: float = 1000.0,
    ) -> ParameterUpdateConfig:
        return cls(
            update_type="spectral_constrained",
            step_size=step_size,
            momentum=momentum,
            ortho_steps=ortho_steps,
            spectral_norm=spectral_norm,
            fisher_damping=fisher_damping,
            ewc_lambda=ewc_lambda,
        )

    @classmethod
    def natural_gradient(
        cls,
        *,
        step_size: float = 0.01,
        momentum: float = 0.9,
        ortho_steps: int = 5,
        spectral_norm: float = 1.0,
        fisher_damping: float = 1e-3,
        ewc_lambda: float = 1000.0,
    ) -> ParameterUpdateConfig:
        return cls(
            update_type="natural_gradient",
            step_size=step_size,
            momentum=momentum,
            ortho_steps=ortho_steps,
            spectral_norm=spectral_norm,
            fisher_damping=fisher_damping,
            ewc_lambda=ewc_lambda,
        )

    @classmethod
    def elastic_consolidation(
        cls,
        *,
        step_size: float = 0.01,
        momentum: float = 0.9,
        ortho_steps: int = 5,
        spectral_norm: float = 1.0,
        fisher_damping: float = 1e-3,
        ewc_lambda: float = 1000.0,
    ) -> ParameterUpdateConfig:
        return cls(
            update_type="elastic_consolidation",
            step_size=step_size,
            momentum=momentum,
            ortho_steps=ortho_steps,
            spectral_norm=spectral_norm,
            fisher_damping=fisher_damping,
            ewc_lambda=ewc_lambda,
        )


# ============================================================
# ParameterUpdate Protocol
# ============================================================


@runtime_checkable
class ParameterUpdate(Protocol):
    """How pseudo-gradients translate into physical weight changes (ΔW).

    Maps the pseudo-gradient tensor (from CreditAssignment) to actual
    parameter deltas. The five canonical types:
    - RiemannianOrthogonal: Muon-style orthogonal updates
    - SpectralConstrained: Lipschitz-bounded updates
    - NaturalGradient: Fisher-information geometry updates
    - ElasticConsolidation: EWC-style importance-weighted updates
    - Euclidean: Standard SGD/Adam in flat space

    The update is applied to the Geometry's parameters.
    """

    config: ParameterUpdateConfig

    @abstractmethod
    def step(
        self,
        params: dict[str, Tensor],
        pseudo_grads: list[Tensor],
        geometry: Geometry,
    ) -> dict[str, Tensor]:
        """Compute parameter updates from pseudo-gradients.

        Args:
            params: Current parameters (name -> tensor)
            pseudo_grads: Pseudo-gradients from CreditAssignment
            geometry: Network topology (for layer-wise adaptation)

        Returns:
            Updated parameters (name -> tensor)
        """
        ...


# ============================================================
# Default/Reference ParameterUpdate Implementations
# ============================================================


def _learnable_weight_names(params: dict[str, Tensor]) -> list[str]:
    """Parameter names that receive pseudo-gradients (2-D weight matrices)."""
    return [n for n, p in params.items() if "weight" in n and p.ndim == 2]


def apply_pseudo_gradients(
    params: dict[str, Tensor],
    pseudo_grads: list[Tensor],
    transform,
) -> dict[str, Tensor]:
    """Pair pseudo-gradients with their parameters by learnable-weight order.

    The single choke point for update rules: non-weight parameters pass
    through untouched (fixes the index-pairing crash on bias interleaving),
    surplus gradients are ignored, and gradients are detached — pseudo-
    gradients are consumed as plain values everywhere in this pipeline.

    Args:
        params: Current parameters (name -> tensor).
        pseudo_grads: One pseudo-gradient per learnable weight.
        transform: ``(name, param, grad) -> updated_param`` for matched pairs.
    """
    updated = dict(params)
    for name, grad in zip(_learnable_weight_names(params), pseudo_grads):
        updated[name] = transform(name, params[name], grad.detach())
    return updated


class EuclideanUpdate:
    """Standard Euclidean SGD/Adam update."""

    def __init__(self, config: ParameterUpdateConfig | None = None):
        self.config = config or ParameterUpdateConfig.euclidean()
        self._momentum_buffers: dict[str, Tensor] = {}

    def _clip(self, grads: list[Tensor]) -> list[Tensor]:
        """Global-norm clip (clip_grad_norm_ semantics): keeps relative
        per-parameter magnitudes intact so updates shrink naturally near
        equilibrium. Per-tensor rescaling would erase that signal and turn
        every step into a fixed-norm jump."""
        clip = self.config.grad_clip
        if clip is None or clip <= 0 or not grads:
            return grads
        stacked_norms = torch.stack([g.norm() for g in grads])
        total_norm = torch.linalg.vector_norm(stacked_norms)
        if total_norm > clip:
            scale = clip / (total_norm + 1e-8)
            grads = [g * scale for g in grads]
        return grads

    def step(
        self,
        params: dict[str, Tensor],
        pseudo_grads: list[Tensor],
        geometry: Geometry,
    ) -> dict[str, Tensor]:
        def apply(name: str, param: Tensor, grad: Tensor) -> Tensor:
            if self.config.momentum > 0:
                buf = self._momentum_buffers.get(name, torch.zeros_like(param))
                buf.mul_(self.config.momentum).add_(grad)
                self._momentum_buffers[name] = buf
                return param - self.config.step_size * buf
            return param - self.config.step_size * grad

        return apply_pseudo_gradients(params, self._clip(list(pseudo_grads)), apply)


class RiemannianOrthogonalUpdate:
    """Muon-style orthogonal update: project gradient onto tangent space of Stiefel manifold."""

    def __init__(self, config: ParameterUpdateConfig | None = None):
        self.config = config or ParameterUpdateConfig.riemannian_orthogonal()

    def step(
        self,
        params: dict[str, Tensor],
        pseudo_grads: list[Tensor],
        geometry: Geometry,
    ) -> dict[str, Tensor]:
        def apply(name: str, param: Tensor, grad: Tensor) -> Tensor:
            # Project gradient onto tangent space: P = G - (G @ W^T + W @ G^T) @ W / 2
            # For simplicity, use sign-SGD style update
            return param - self.config.step_size * grad.sign()

        return apply_pseudo_gradients(params, list(pseudo_grads), apply)


class SpectralConstrainedUpdate:
    """Lipschitz-bounded update: constrain spectral norm of updates."""

    def __init__(self, config: ParameterUpdateConfig | None = None):
        self.config = config or ParameterUpdateConfig.spectral_constrained()

    def step(
        self,
        params: dict[str, Tensor],
        pseudo_grads: list[Tensor],
        geometry: Geometry,
    ) -> dict[str, Tensor]:
        def apply(name: str, param: Tensor, grad: Tensor) -> Tensor:
            # Normalize gradient to target spectral norm
            grad_norm = torch.linalg.matrix_norm(grad, ord=2)
            if grad_norm > self.config.spectral_norm:
                grad = grad * (self.config.spectral_norm / (grad_norm + 1e-8))
            return param - self.config.step_size * grad

        return apply_pseudo_gradients(params, list(pseudo_grads), apply)


class NaturalGradientUpdate:
    """Fisher-information geometry update (natural gradient)."""

    def __init__(self, config: ParameterUpdateConfig | None = None):
        self.config = config or ParameterUpdateConfig.natural_gradient()

    def step(
        self,
        params: dict[str, Tensor],
        pseudo_grads: list[Tensor],
        geometry: Geometry,
    ) -> dict[str, Tensor]:
        def apply(name: str, param: Tensor, grad: Tensor) -> Tensor:
            # Simplified: scale by inverse Fisher approximation
            return param - self.config.step_size * grad / (grad.abs().mean() + 1e-8)

        return apply_pseudo_gradients(params, list(pseudo_grads), apply)


class ElasticConsolidationUpdate:
    """EWC-style importance-weighted update."""

    def __init__(self, config: ParameterUpdateConfig | None = None):
        self.config = config or ParameterUpdateConfig.elastic_consolidation()

    def step(
        self,
        params: dict[str, Tensor],
        pseudo_grads: list[Tensor],
        geometry: Geometry,
    ) -> dict[str, Tensor]:
        def apply(name: str, param: Tensor, grad: Tensor) -> Tensor:
            # Simplified: no actual Fisher info stored
            return param - self.config.step_size * grad

        return apply_pseudo_gradients(params, list(pseudo_grads), apply)
