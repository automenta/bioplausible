"""Layer 5: ParameterUpdate — The Optimization Rule."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
from torch import Tensor

from computronium.ontology.utils import apply_pseudo_gradients

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
    beta2: float = 0.999
    eps: float = 1e-8

    @classmethod
    def euclidean(
        cls,
        *,
        step_size: float = 0.01,
        momentum: float = 0.9,
        ortho_steps: int = 0,
        spectral_norm: float = 1.0,
        fisher_damping: float = 1e-3,
        ewc_lambda: float = 1000.0,
        grad_clip: float = 1.0,
    ) -> ParameterUpdateConfig:
        """Euclidean SGD update config."""
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
        ortho_steps: int = 0,
        spectral_norm: float = 1.0,
        fisher_damping: float = 1e-3,
        ewc_lambda: float = 1000.0,
    ) -> ParameterUpdateConfig:
        """Riemannian-orthogonal (Muon-class) update config.

        ``ortho_steps == 0`` (default) selects the EXACT SVD polar factor —
        full-spectrum whitening, the configuration under which D13's
        FF×Muon lift and D15's depth-frontier claims are measured. Values
        > 0 select Newton–Schulz iteration (Muon's cheaper recipe, partial
        whitening): measured at width 32 it PRESERVES the BP×Muon lift but
        COLLAPSES FF×Muon (0.29 vs 0.838) — the local-credit lift is
        whitening-driven, so NS is an opt-in variant, never the default.
        """
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

    @classmethod
    def adam(
        cls,
        *,
        step_size: float = 1e-3,
        momentum: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        grad_clip: float = 1.0,
    ) -> ParameterUpdateConfig:
        """Adam update config.

        ``momentum`` is β1 (first-moment decay), ``beta2`` the
        second-moment decay, ``eps`` the denominator floor. Distinct from
        ``euclidean`` (plain SGD + momentum): the per-coordinate
        second-moment normalization is a different optimizer family, not a
        step-size variant — the D14 jpc-faithful regime showed it is
        load-bearing for deep local learning, and the D16 coverage map
        never swept it (a known instrument gap).
        """
        return cls(
            update_type="adam",
            step_size=step_size,
            momentum=momentum,
            ortho_steps=0,
            spectral_norm=1.0,
            fisher_damping=1e-3,
            ewc_lambda=1000.0,
            grad_clip=grad_clip,
            beta2=beta2,
            eps=eps,
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


class EuclideanUpdate:
    """Standard Euclidean update: plain SGD (optionally with momentum).

    Not Adam — the per-coordinate second-moment family is
    :class:`AdamUpdate`. The ParameterUpdate Protocol docstring's
    "SGD/Adam" phrasing refers to the update *shapes* both realize
    (ΔW = f(∇) in flat space), not to the same algorithm.
    """

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
                buf = self._momentum_buffers.get(name)
                if buf is not None and buf.shape != param.shape:
                    msg = (
                        f"Momentum buffer for {name!r} has shape "
                        f"{tuple(buf.shape)} but parameter has "
                        f"{tuple(param.shape)} — the update instance is "
                        "being reused across different geometries; create "
                        "one update per system (optimizer state is "
                        "system-scoped)"
                    )
                    raise RuntimeError(msg)
                if buf is None:
                    buf = torch.zeros_like(param)
                buf.mul_(self.config.momentum).add_(grad)
                self._momentum_buffers[name] = buf
                return param - self.config.step_size * buf
            return param - self.config.step_size * grad

        return apply_pseudo_gradients(params, self._clip(list(pseudo_grads)), apply)


class AdamUpdate:
    """Adam (Kingma & Ba 2015) on pseudo-gradients.

    Per-coordinate first/second-moment estimates with bias correction.
    Optimizer state is system-scoped: reusing one instance across
    geometries fails loud (the D13 momentum-buffer lesson). A distinct
    optimizer family from EuclideanUpdate's plain SGD+momentum — the
    U-axis coverage map (D16) measured only the SGD family, which the
    D14 jpc-faithful regime showed is the wrong default at depth.
    """

    def __init__(self, config: ParameterUpdateConfig | None = None):
        self.config = config or ParameterUpdateConfig.adam()
        self._m: dict[str, Tensor] = {}
        self._v: dict[str, Tensor] = {}
        self._t = 0

    def _clip(self, grads: list[Tensor]) -> list[Tensor]:
        clip = self.config.grad_clip
        if clip is None or clip <= 0 or not grads:
            return grads
        stacked_norms = torch.stack([g.norm() for g in grads])
        total_norm = torch.linalg.vector_norm(stacked_norms)
        if total_norm > clip:
            scale = clip / (total_norm + 1e-8)
            grads = [g * scale for g in grads]
        return grads

    def _state(self, name: str, param: Tensor, store: dict[str, Tensor]) -> Tensor:
        buf = store.get(name)
        if buf is not None and buf.shape != param.shape:
            msg = (
                f"Adam state for {name!r} has shape {tuple(buf.shape)} but "
                f"parameter has {tuple(param.shape)} — the update instance "
                "is being reused across different geometries; create one "
                "update per system (optimizer state is system-scoped)"
            )
            raise RuntimeError(msg)
        if buf is None:
            buf = torch.zeros_like(param)
            store[name] = buf
        return buf

    def step(
        self,
        params: dict[str, Tensor],
        pseudo_grads: list[Tensor],
        geometry: Geometry,
    ) -> dict[str, Tensor]:
        grads = self._clip(list(pseudo_grads))
        self._t += 1
        beta1 = self.config.momentum
        beta2 = self.config.beta2
        bias1 = 1 - beta1**self._t
        bias2 = 1 - beta2**self._t

        def apply(name: str, param: Tensor, grad: Tensor) -> Tensor:
            m = self._state(name, param, self._m)
            v = self._state(name, param, self._v)
            m.mul_(beta1).add_(grad, alpha=1 - beta1)
            v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            m_hat = m / bias1
            v_hat = v / bias2
            denom = v_hat.sqrt().add_(self.config.eps)
            return param - self.config.step_size * m_hat / denom

        return apply_pseudo_gradients(params, grads, apply)


class RiemannianOrthogonalUpdate:
    """Muon-style orthogonal update: orthogonalize the momentum buffer.

    The orthogonalizer is Newton–Schulz iteration (Muon's actual recipe,
    ``ortho_steps`` iterations — MEP fast paths: Triton kernel on Triton
    targets, CUDA kernel on CUDA, torch otherwise), NOT the full SVD:
    the SVD polar factor was the placeholder, its per-matrix-per-step
    cost dominating deep sweeps. ``ortho_steps == 0`` selects the exact
    SVD polar factor as an audit escape hatch.
    """

    def __init__(self, config: ParameterUpdateConfig | None = None):
        self.config = config or ParameterUpdateConfig.riemannian_orthogonal()
        self._momentum_buffers: dict[str, Tensor] = {}

    def _orthogonalize(self, grad: Tensor) -> Tensor:
        if self.config.ortho_steps <= 0:
            # Exact polar factor: U @ Vh is the nearest orthogonal matrix
            # to grad in Frobenius norm. Reduced QR is NOT a substitute:
            # its R-diagonal is sign-arbitrary, so the resulting direction
            # is uncorrelated with the gradient (measured cos ≈ 0).
            U, _, Vh = torch.linalg.svd(grad, full_matrices=False)
            return U @ Vh
        from computronium.core.optimization.strategies.update import (
            newton_schulz5,
        )

        return newton_schulz5(grad, self.config.ortho_steps)

    def step(
        self,
        params: dict[str, Tensor],
        pseudo_grads: list[Tensor],
        geometry: Geometry,
    ) -> dict[str, Tensor]:
        def apply(name: str, param: Tensor, grad: Tensor) -> Tensor:
            # Muon orthogonalizes the MOMENTUM, not the raw single-batch
            # gradient: orthogonalization amplifies the noise floor, so the
            # EMA buffer must accumulate signal across batches first.
            if self.config.momentum > 0:
                buf = self._momentum_buffers.get(name)
                if buf is None or buf.shape != param.shape:
                    buf = torch.zeros_like(param)
                buf.mul_(self.config.momentum).add_(grad)
                self._momentum_buffers[name] = buf
                grad = buf
            ortho_grad = self._orthogonalize(grad)
            return param - self.config.step_size * ortho_grad

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
                grad = grad * (self.config.spectral_norm / (grad_norm + 1e-8))  # ruff: ignore[non-augmented-assignment]
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
    """EWC-style importance-weighted update.

    In standard EWC (Kirkpatrick et al., 2017):
    - Fisher information F = (param - old_param)^2 (squared distance from old task)
    - Regularization: lambda * F * (param - old_param)
    - lambda (ewc_lambda) is the importance weight controlling regularization strength

    This implementation separates:
    - fisher_damping: small constant added to Fisher for numerical stability
    - ewc_lambda: importance weight for the EWC regularization term
    """

    def __init__(self, config: ParameterUpdateConfig | None = None):
        self.config = config or ParameterUpdateConfig.elastic_consolidation()
        self._old_params: dict[str, Tensor] = {}
        self._fisher: dict[str, Tensor] = {}
        self._baseline: dict[str, Tensor] = {}

    def consolidate(
        self, params: dict[str, Tensor], old_params: dict[str, Tensor] | None = None
    ) -> None:
        """Store old parameters and compute Fisher importance (squared distance from old).

        With ``old_params`` given, importance is ``(params - old_params)^2``.
        Single-argument form anchors the current snapshot for future tasks and
        derives importance from drift since the previous consolidation
        baseline (uniform damping on the first call).
        """
        if old_params is None:
            old_params = self._baseline or params
        self._old_params = {k: v.clone().detach() for k, v in old_params.items()}
        # Fisher = (current - old)^2 + fisher_damping (for numerical stability)
        self._fisher = {
            k: (params[k].detach() - old_params[k].detach()) ** 2
            + self.config.fisher_damping
            for k in params
            if k in old_params
        }
        self._baseline = {k: v.clone().detach() for k, v in params.items()}

    def step(
        self,
        params: dict[str, Tensor],
        pseudo_grads: list[Tensor],
        geometry: Geometry,
    ) -> dict[str, Tensor]:
        def apply(name: str, param: Tensor, grad: Tensor) -> Tensor:
            # EWC update: param - lr * grad - lr * ewc_lambda * fisher * (param - old_param)
            if name in self._old_params and name in self._fisher:
                ewc_term = (
                    self.config.ewc_lambda
                    * self._fisher[name]
                    * (param - self._old_params[name])
                )
                return (
                    param
                    - self.config.step_size * grad
                    - self.config.step_size * ewc_term
                )
            return param - self.config.step_size * grad

        return apply_pseudo_gradients(params, list(pseudo_grads), apply)
