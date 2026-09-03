"""
Update transformation strategies.

Plain SGD and Muon orthogonalization reuse the generic implementations from
:mod:`computronium.core.optimization.strategies` (REFACTOR.md §7); the MEP
subclass adds the CUDA Newton-Schulz fast path. Dion (low-rank SVD) and
Fisher-whitened updates remain MEP-specific.
"""

from typing import Literal, cast

import torch
from torch import nn

from computronium.acceleration.triton_kernels import MEP_TritonOps
from computronium.core.optimization.strategies import (
    MuonUpdate as _CoreMuonUpdate,
)
from computronium.core.optimization.strategies import (
    PlainUpdate,
)

__all__ = [
    "DionUpdate",
    "FisherUpdate",
    "MuonUpdate",
    "PlainUpdate",
]

type Backend = Literal["pytorch", "triton"]
try:
    from ...cuda.kernels import dion_update_cuda, newton_schulz_cuda

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


class MuonUpdate(_CoreMuonUpdate):
    """Newton-Schulz orthogonalization with the MEP CUDA/Triton fast path."""

    def __init__(self, ns_steps: int = 5, backend: Backend = "pytorch"):
        super().__init__(ns_steps=ns_steps)
        self.backend = backend

    def _newton_schulz(
        self, G: torch.Tensor, steps: int, epsilon: float = 1e-4
    ) -> torch.Tensor:
        if self.backend == "triton":
            return cast(
                "torch.Tensor", MEP_TritonOps.muon_orthogonalize(G, ns_steps=steps)
            )
        if CUDA_AVAILABLE and G.is_cuda:
            return cast(
                "torch.Tensor", newton_schulz_cuda(G, steps=steps, epsilon=epsilon)
            )
        return super()._newton_schulz(G, steps, epsilon)


class DionUpdate:
    """
    Low-rank SVD update with error feedback.

    For large matrices (numel > threshold), uses low-rank SVD:
        G ≈ U @ S @ V^T
        update = U @ V^T  (scale-invariant)

    For smaller matrices, falls back to Muon orthogonalization.
    """

    def __init__(
        self,
        rank_frac: float = 0.2,
        threshold: int = 100000,
        muon_fallback: MuonUpdate | None = None,
        backend: Backend = "pytorch",
    ):
        self.rank_frac = rank_frac
        self.threshold = threshold
        self.backend = backend
        self.muon_fallback = muon_fallback or MuonUpdate(backend=backend)

    @staticmethod
    def _apply_error_feedback(
        gradient: torch.Tensor,
        update: torch.Tensor,
        state: dict,
        group_config: dict,
    ) -> None:
        if not group_config.get("use_error_feedback", True):
            return
        residual = gradient - update
        error_beta = group_config.get("error_beta", 0.9)
        if "error_buffer" not in state:
            state["error_buffer"] = torch.zeros_like(residual)
        state["error_buffer"].mul_(error_beta).add_(residual)

    def transform_gradient(  # ruff: ignore[too-many-locals]
        self,
        param: nn.Parameter,
        gradient: torch.Tensor,
        state: dict,
        group_config: dict,
    ) -> torch.Tensor:
        # Use gradient numel if param is None (for testing)
        numel = param.numel() if param is not None else gradient.numel()

        if numel <= self.threshold:
            return self.muon_fallback.transform_gradient(
                param, gradient, state, group_config
            )

        # Low-rank SVD for large matrices
        if gradient.ndim != 2:
            orig_shape = gradient.shape
            gradient = gradient.view(gradient.shape[0], -1)
        else:
            orig_shape = None

        rank = max(1, int(min(gradient.shape) * self.rank_frac))
        max_rank = min(gradient.shape)
        rank = min(rank, max_rank)

        try:  # ruff: ignore[too-many-statements-in-try-clause]
            # Gradient clipping
            max_norm = group_config.get("max_grad_norm", 10.0)
            grad_norm = gradient.norm()
            if grad_norm > max_norm:
                gradient = gradient * (max_norm / (grad_norm + 1e-8))  # ruff: ignore[non-augmented-assignment]

            # Low-rank SVD
            if self.backend == "triton":
                update = MEP_TritonOps.dion_update(gradient, rank=rank)
                self._apply_error_feedback(gradient, update, state, group_config)
            elif CUDA_AVAILABLE and gradient.is_cuda:
                error_buf = state.get("error_buffer")
                error_beta = group_config.get("error_beta", 0.9)
                use_feedback = group_config.get("use_error_feedback", True)

                if use_feedback and error_buf is not None:
                    update, new_buf = dion_update_cuda(
                        gradient,
                        rank=rank,
                        error_buffer=error_buf,
                        error_beta=error_beta,
                    )
                    state["error_buffer"] = new_buf
                else:
                    update, _ = dion_update_cuda(gradient, rank=rank)
            else:
                U, _S, V = torch.svd_lowrank(gradient, q=rank)
                update = U @ V.T

                self._apply_error_feedback(gradient, update, state, group_config)

            if orig_shape is not None:
                update = update.view(orig_shape)

            return update  # ruff: ignore[try-consider-else]

        except RuntimeError, torch.linalg.LinAlgError:
            # Fallback to Muon
            return self.muon_fallback.transform_gradient(
                param, gradient, state, group_config
            )


class FisherUpdate:
    """
    Fisher-whitened gradient with Muon orthogonalization.

    Applies natural gradient preconditioning:
        whitened = g @ (F + λI)^-1

    Then orthogonalizes via Newton-Schulz.
    """

    def __init__(
        self,
        damping: float = 1e-3,
        ns_steps: int = 5,
        use_diagonal: bool = False,
        beta: float = 0.95,
        backend: Backend = "pytorch",
    ):
        self.damping = damping
        self.ns_steps = ns_steps
        self.use_diagonal = use_diagonal
        self.beta = beta
        self.backend = backend
        self.muon = MuonUpdate(ns_steps=ns_steps, backend=backend)

    def transform_gradient(
        self,
        param: nn.Parameter,
        gradient: torch.Tensor,
        state: dict,
        group_config: dict,
    ) -> torch.Tensor:
        # Handle ND tensors by flattening
        orig_shape = None
        if gradient.ndim > 2:
            orig_shape = gradient.shape
            gradient = gradient.view(gradient.shape[0], -1)
        elif gradient.ndim < 2:
            return gradient

        # Check for new Fisher estimate on parameter
        if hasattr(param, "fisher"):
            fisher_estimate = getattr(param, "fisher")
            delattr(param, "fisher")  # Consume it

            if "fisher" not in state:
                state["fisher"] = fisher_estimate
            else:
                state["fisher"].mul_(self.beta).add_(
                    fisher_estimate, alpha=1 - self.beta
                )

        fisher = state.get("fisher")

        if fisher is not None:
            if self.use_diagonal:
                # Diagonal whitening
                F = fisher + self.damping
                whitened = gradient / F.unsqueeze(0)
            else:
                # Full whitening: solve (F + λI) @ X = g^T
                F = fisher + self.damping * torch.eye(
                    fisher.shape[0], device=fisher.device, dtype=fisher.dtype
                )
                try:
                    whitened = torch.linalg.solve(F, gradient.T).T
                    if torch.isnan(whitened).any():
                        whitened = gradient
                except RuntimeError:
                    whitened = gradient
        else:
            whitened = gradient

        update = self.muon._newton_schulz(whitened, self.ns_steps)
        if orig_shape is not None:
            update = update.view(orig_shape)

        return update
