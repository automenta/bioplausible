"""
Triton Kernels for EqProp Acceleration

Provides fused kernels for Equilibrium Propagation dynamics to maximize
GPU throughput by reducing memory bandwidth usage.
"""

import math

import torch

from bioplausible.acceleration.backends import HAS_CUPY, HAS_TRITON


class TritonEqPropOps:
    """EqProp operations with optional Triton/CUDA acceleration.

    Provides step functions for equilibrium propagation with automatic
    fallback to PyTorch when Triton is not available.
    """

    _triton_kernel = None

    @classmethod
    def is_available(cls) -> bool:
        return HAS_TRITON

    @classmethod
    def _init_triton(cls):
        if cls._triton_kernel is None and HAS_TRITON:
            try:
                import triton
                import triton.language as tl
                from triton.language.extra import libdevice

                @triton.jit
                def _step_kernel(
                    h_ptr,
                    pre_act_ptr,
                    bias_ptr,
                    out_ptr,
                    alpha,
                    n_elements,
                    bias_n,
                    BLOCK_SIZE: tl.constexpr,
                ):
                    pid = tl.program_id(0)
                    block_start = pid * BLOCK_SIZE
                    offsets = block_start + tl.arange(0, BLOCK_SIZE)
                    mask = offsets < n_elements

                    h = tl.load(h_ptr + offsets, mask=mask)
                    pre_act = tl.load(pre_act_ptr + offsets, mask=mask)
                    bias = (
                        tl.load(bias_ptr + offsets % bias_n, mask=mask)
                        if bias_ptr is not None
                        else 0.0
                    )

                    out = (1.0 - alpha) * h + alpha * libdevice.tanh(pre_act + bias)
                    tl.store(out_ptr + offsets, out, mask=mask)

                cls._triton_kernel = _step_kernel
            except ImportError:
                cls._triton_kernel = False

    @classmethod
    def step(
        cls,
        h: torch.Tensor,
        pre_act: torch.Tensor,
        alpha: float,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if HAS_TRITON and h.is_cuda and pre_act.is_cuda:
            cls._init_triton()
            if cls._triton_kernel:
                out = torch.empty_like(h)
                n_elements = h.numel()
                BLOCK_SIZE = 1024
                grid = (math.ceil(n_elements / BLOCK_SIZE),)
                bias_ptr = bias if bias is not None else None
                bias_n = bias.numel() if bias is not None else 0
                cls._triton_kernel[grid](
                    h,
                    pre_act,
                    bias_ptr,
                    out,
                    alpha,
                    n_elements,
                    bias_n,
                    BLOCK_SIZE=BLOCK_SIZE,
                )
                return out

        bias_val = bias if bias is not None else 0.0
        return (1.0 - alpha) * h + alpha * torch.tanh(pre_act + bias_val)

    @classmethod
    def step_linear(
        cls,
        h: torch.Tensor,
        h_target: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        return (1.0 - alpha) * h + alpha * h_target

    @classmethod
    def step_linear_cupy(cls, h, h_target, alpha):
        if not HAS_CUPY:
            raise ImportError("CuPy not available")
        return cls.step_linear(h, h_target, alpha)

    # ── fused layered MLP-block forward step ─────────────────────────────────
    # One kernel launch replaces the ~6 separate CuPy ops (layernorm mean/std,
    # W1 matmul, tanh, W2 matmul, residual) the NumPy path issues per settle
    # step, which is where the GPU kernel lost wall-clock time to the PyTorch
    # engine. h_next = (1-gamma)*h + gamma*(ffn_out + x_emb) where
    #   h_norm   = layernorm(h)
    #   ffn_hid  = tanh(h_norm @ W1^T + b1)
    #   ffn_out  = ffn_hid @ W2^T + b2
    _layered_kernel = None

    @classmethod
    def _init_layered(cls):
        if cls._layered_kernel is None and HAS_TRITON:
            import triton
            import triton.language as tl
            from triton.language.extra import libdevice

            @triton.jit
            def _layered_step_kernel(
                h_ptr,
                x_emb_ptr,
                w1_ptr,
                b1_ptr,
                w2_ptr,
                b2_ptr,
                out_ptr,
                hnorm_ptr,
                ffnhid_ptr,
                gamma,
                M,
                K: tl.constexpr,
                H: tl.constexpr,
                BLOCK_M: tl.constexpr,
            ):
                pid = tl.program_id(0)
                offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
                mask_m = offs_m < M
                offs_k = tl.arange(0, K)

                # Load h and x_emb tiles (BLOCK_M, K)
                h = tl.load(
                    h_ptr + offs_m[:, None] * K + offs_k[None, :],
                    mask=mask_m[:, None],
                )
                x_emb = tl.load(
                    x_emb_ptr + offs_m[:, None] * K + offs_k[None, :],
                    mask=mask_m[:, None],
                )

                # LayerNorm over K (per row)
                mean = tl.sum(h, axis=1) / K
                h_centered = h - mean[:, None]
                var = tl.sum(h_centered * h_centered, axis=1) / K
                std = tl.sqrt(var + 1e-5)
                h_norm = h_centered / std[:, None]
                tl.store(
                    hnorm_ptr + offs_m[:, None] * K + offs_k[None, :],
                    h_norm,
                    mask=mask_m[:, None],
                )

                # ffn_hidden = tanh(h_norm @ W1^T + b1); W1 is (H, K)
                offs_h = tl.arange(0, H)
                w1 = tl.load(w1_ptr + offs_h[:, None] * K + offs_k[None, :])
                b1 = tl.load(b1_ptr + offs_h)
                ffn = tl.dot(h_norm, tl.trans(w1))  # (BLOCK_M, H)
                ffn = libdevice.tanh(ffn + b1[None, :])
                tl.store(
                    ffnhid_ptr + offs_m[:, None] * H + offs_h[None, :],
                    ffn,
                    mask=mask_m[:, None],
                )

                # ffn_out = ffn @ W2^T + b2; W2 is (K, H)
                w2 = tl.load(w2_ptr + offs_k[:, None] * H + offs_h[None, :])
                b2 = tl.load(b2_ptr + offs_k)
                ffn_out = tl.dot(ffn, tl.trans(w2))  # (BLOCK_M, K)
                ffn_out = ffn_out + b2[None, :]

                # h_next = (1-gamma)*h + gamma*(ffn_out + x_emb)
                h_next = (1.0 - gamma) * h + gamma * (ffn_out + x_emb)
                tl.store(
                    out_ptr + offs_m[:, None] * K + offs_k[None, :],
                    h_next,
                    mask=mask_m[:, None],
                )

            cls._layered_kernel = _layered_step_kernel

    @classmethod
    def step_layered_cupy(
        cls,
        h,
        x_emb,
        w1,
        b1,
        w2,
        b2,
        gamma,
        out=None,
        hnorm_out=None,
        ffnhid_out=None,
    ) -> tuple[object, object, object] | None:
        """Fused layered MLP-block forward step on CuPy arrays.

        Computes ``(h_next, h_norm, ffn_hidden)`` in a single launch (the NumPy
        path needs all three for the contrastive Hebbian update). Writes into
        ``out``/``hnorm_out``/``ffnhid_out`` when provided and returns them;
        returns ``None`` if Triton is unavailable.
        """
        if not (HAS_CUPY and HAS_TRITON):
            return None
        cls._init_layered()
        if cls._layered_kernel is None:
            return None
        import cupy as cp
        import torch

        h_t = torch.as_tensor(cp.asnumpy(h), device="cuda").float()
        x_emb_t = torch.as_tensor(cp.asnumpy(x_emb), device="cuda").float()
        w1_t = torch.as_tensor(cp.asnumpy(w1), device="cuda").float()
        b1_t = torch.as_tensor(cp.asnumpy(b1), device="cuda").float()
        w2_t = torch.as_tensor(cp.asnumpy(w2), device="cuda").float()
        b2_t = torch.as_tensor(cp.asnumpy(b2), device="cuda").float()

        M, K = h_t.shape
        H = w1_t.shape[0]
        BLOCK_M = 16

        out_t = torch.empty_like(h_t)
        hnorm_t = torch.empty_like(h_t)
        ffnhid_t = torch.empty((M, H), device="cuda", dtype=torch.float32)
        grid = ((M + BLOCK_M - 1) // BLOCK_M,)
        cls._layered_kernel[grid](
            h_t,
            x_emb_t,
            w1_t,
            b1_t,
            w2_t,
            b2_t,
            out_t,
            hnorm_t,
            ffnhid_t,
            gamma,
            M,
            K=K,
            H=H,
            BLOCK_M=BLOCK_M,
        )

        def _store(dst, src):
            if dst is None:
                return cp.asarray(src)
            dst[:] = cp.asarray(src)
            return dst

        h_next = _store(out, out_t)
        h_norm = _store(hnorm_out, hnorm_t)
        ffn_hidden = _store(ffnhid_out, ffnhid_t)
        return h_next, h_norm, ffn_hidden


__all__ = [
    "HAS_CUPY",
    "HAS_TRITON",
    "TritonEqPropOps",
]
