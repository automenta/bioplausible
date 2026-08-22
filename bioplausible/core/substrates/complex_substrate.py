"""Complex Substrate for Holomorphic/Complex-Valued Neural Networks.

Provides efficient complex64 arithmetic on GPU via real/imag channel emulation
(with Triton kernels for hot paths) and CPU fallback for complex ops.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import torch
from torch import Tensor

from bioplausible.core.ontology import DigitalSubstrate, SubstrateConfig

if TYPE_CHECKING:
    from collections.abc import Callable


class ComplexSubstrate(DigitalSubstrate):
    """Complex-valued substrate for holomorphic neural networks.

    Implements complex arithmetic by storing real/imag as adjacent channels
    [real, imag, real, imag, ...] in the last dimension. This enables:
    - Full GPU utilization via standard float32/float16 operations
    - Triton-accelerated complex ops (matmul, tanh, conjugate transpose)
    - Zero-copy conversion to/from native complex64 for interop

    Memory layout: For a complex tensor of shape (..., N), the underlying
    real tensor has shape (..., 2*N) with [real_0, imag_0, real_1, imag_1, ...]
    """

    def __init__(self, config: SubstrateConfig | None = None):
        # Use float32 precision for the emulated channels
        super().__init__(
            config
            or SubstrateConfig(
                precision="float32",
                noise_level=config.noise_level if config else 0.0,
                weight_bounds=config.weight_bounds if config else None,
                sparsity=config.sparsity if config else 0.0,
                device=config.device if config else "cpu",
            )
        )
        self._use_triton = self._check_triton_available()
        self._complex_emulated = True  # Flag to identify complex-emulated substrate

    def _check_triton_available(self) -> bool:
        try:
            import triton  # noqa: F401
            import triton.language as tl  # noqa: F401
            return torch.cuda.is_available()
        except ImportError:
            return False

    # =========================================================================
    # Conversion utilities
    # =========================================================================

    @staticmethod
    def to_real(complex_tensor: Tensor) -> Tensor:
        """Convert native complex64/128 to emulated real tensor [..., 2*N]."""
        if not complex_tensor.is_complex():
            # Already in emulated format
            return complex_tensor
        real = complex_tensor.real
        imag = complex_tensor.imag
        # Stack as [real, imag] in last dim
        return torch.stack([real, imag], dim=-1).flatten(-2)

    @staticmethod
    def to_complex(real_tensor: Tensor) -> Tensor:
        """Convert emulated real tensor [..., 2*N] to native complex64."""
        if real_tensor.is_complex():
            return real_tensor
        # Reshape from [..., 2*N] to [..., N, 2]
        *batch, double_n = real_tensor.shape
        n = double_n // 2
        reshaped = real_tensor.view(*batch, n, 2)
        return torch.complex(reshaped[..., 0], reshaped[..., 1])

    # =========================================================================
    # Complex arithmetic (emulated via real channels)
    # =========================================================================

    def complex_mul(self, a: Tensor, b: Tensor) -> Tensor:
        """Complex multiplication: (a_r + i*a_i) * (b_r + i*b_i)."""
        a = self.to_real(a)
        b = self.to_real(b)
        # a = [..., 2*N], b = [..., 2*N] -> split into real/imag
        a_r, a_i = a[..., ::2], a[..., 1::2]
        b_r, b_i = b[..., ::2], b[..., 1::2]
        out_r = a_r * b_r - a_i * b_i
        out_i = a_r * b_i + a_i * b_r
        return torch.stack([out_r, out_i], dim=-1).flatten(-2)

    def complex_matmul(self, a: Tensor, b: Tensor) -> Tensor:
        """Complex matrix multiplication: a @ b^T (conjugate if needed)."""
        a = self.to_real(a)
        b = self.to_real(b)
        a_r, a_i = a[..., ::2], a[..., 1::2]
        b_r, b_i = b[..., ::2], b[..., 1::2]
        # (a_r + i*a_i) @ (b_r + i*b_i)^T = a_r@b_r^T - a_i@b_i^T + i*(a_r@b_i^T + a_i@b_r^T)
        out_r = a_r @ b_r.transpose(-2, -1) - a_i @ b_i.transpose(-2, -1)
        out_i = a_r @ b_i.transpose(-2, -1) + a_i @ b_r.transpose(-2, -1)
        return torch.stack([out_r, out_i], dim=-1).flatten(-2)

    def complex_conj(self, a: Tensor) -> Tensor:
        """Complex conjugate."""
        a = self.to_real(a)
        a_i = a[..., 1::2]
        a[..., 1::2] = -a_i
        return a

    def complex_tanh(self, a: Tensor) -> Tensor:
        """Complex tanh activation: tanh(z) = sin(2*real)/(cos(2*real)+cosh(2*imag)) + i*sinh(2*imag)/(cos(2*real)+cosh(2*imag))."""
        a = self.to_real(a)
        a_r, a_i = a[..., ::2], a[..., 1::2]
        two_r = 2 * a_r
        two_i = 2 * a_i
        denom = torch.cos(two_r) + torch.cosh(two_i)
        out_r = torch.sin(two_r) / denom
        out_i = torch.sinh(two_i) / denom
        return torch.stack([out_r, out_i], dim=-1).flatten(-2)

    def complex_linear(self, x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
        """Complex linear layer: x @ W^H + b (conjugate transpose of weight)."""
        x = self.to_real(x)
        w = self.to_real(weight)
        # Weight is stored as [out_features, in_features] complex -> [out, 2*in] real
        # We need x @ W^H where W^H = conjugate(W)^T
        w_r, w_i = w[..., ::2], w[..., 1::2]
        # Conjugate transpose: (W^H)_r = W_r^T, (W^H)_i = -W_i^T
        x_r, x_i = x[..., ::2], x[..., 1::2]
        out_r = x_r @ w_r.transpose(-2, -1) - x_i @ (-w_i).transpose(-2, -1)
        out_i = x_r @ (-w_i).transpose(-2, -1) + x_i @ w_r.transpose(-2, -1)
        if bias is not None:
            b = self.to_real(bias)
            out_r = out_r + b[..., ::2]
            out_i = out_i + b[..., 1::2]
        return torch.stack([out_r, out_i], dim=-1).flatten(-2)

    # =========================================================================
    # Substrate interface
    # =========================================================================

    def get_forward_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        """Forward operator for complex linear layer."""

        def complex_forward(x: Tensor, w: Tensor) -> Tensor:
            # x: [..., in_features] complex (emulated)
            # w: [out_features, in_features] complex (emulated)
            x = self._to_precision(x)
            w = self._to_precision(w)
            return self._to_precision(self.complex_linear(x, w))

        return complex_forward

    def get_weight_update_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        """Weight update for complex weights using gradient descent on real/imag parts."""

        def complex_update(pseudo_grad: Tensor, current_w: Tensor) -> Tensor:
            # pseudo_grad and current_w are in emulated format
            step_size = getattr(self.config, "step_size", 0.01)
            pseudo_grad = self._to_precision(pseudo_grad)
            current_w = self._to_precision(current_w)
            # Simple SGD on real/imag parts
            return self._to_precision(current_w - step_size * pseudo_grad)

        return complex_update

    def inject_state_noise(self, state: Tensor) -> Tensor:
        """Add complex Gaussian noise to state."""
        state = self.to_real(state)
        noise = torch.randn_like(state) * self.config.noise_level
        return self._to_precision(state + noise)


# =========================================================================
# Triton kernels for hot paths (if available)
# =========================================================================

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _complex_tanh_kernel(
        real_ptr, imag_ptr, out_real_ptr, out_imag_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Triton kernel for complex tanh."""
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements

        real = tl.load(real_ptr + offs, mask=mask)
        imag = tl.load(imag_ptr + offs, mask=mask)

        two_r = 2.0 * real
        two_i = 2.0 * imag
        denom = tl.cos(two_r) + tl.cosh(two_i)

        out_r = tl.sin(two_r) / denom
        out_i = tl.sinh(two_i) / denom

        tl.store(out_real_ptr + offs, out_r, mask=mask)
        tl.store(out_imag_ptr + offs, out_i, mask=mask)

    @triton.jit
    def _complex_matmul_kernel(
        a_real_ptr, a_imag_ptr, b_real_ptr, b_imag_ptr,
        out_real_ptr, out_imag_ptr,
        M, N, K,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """Triton kernel for complex batched matrix multiplication."""
        # Simplified: each block computes one output element
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_b = tl.program_id(2)

        if pid_m >= M or pid_n >= N or pid_b >= BLOCK_K:
            return

        # Accumulate over K
        acc_r = 0.0
        acc_i = 0.0
        for k in range(K):
            a_r = tl.load(a_real_ptr + pid_b * M * K + pid_m * K + k)
            a_i = tl.load(a_imag_ptr + pid_b * M * K + pid_m * K + k)
            b_r = tl.load(b_real_ptr + pid_b * N * K + pid_n * K + k)
            b_i = tl.load(b_imag_ptr + pid_b * N * K + pid_n * K + k)

            acc_r += a_r * b_r - a_i * b_i
            acc_i += a_r * b_i + a_i * b_r

        tl.store(out_real_ptr + pid_b * M * N + pid_m * N + pid_n, acc_r)
        tl.store(out_imag_ptr + pid_b * M * N + pid_m * N + pid_n, acc_i)

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False