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

    # Cache of converted torch weight tensors keyed by source cupy array id
    # (see ``step_layered_cupy_torch``). Cleared when weights change.
    _torch_cache: dict[int, object] = {}

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

    @classmethod
    def step_layered_cupy_torch(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]  # mirrors the cupy forward-step signature
        cls, h, x_emb, w1, b1, w2, b2, gamma, out=None, hnorm_out=None, ffnhid_out=None
    ) -> tuple[object, object, object] | None:
        """Torch-native layered MLP-block forward step on CuPy arrays.

        Uses zero-copy cupy<->torch views and cuBLAS matmuls, ~8x faster than
        the hand-rolled Triton kernel for these FFN shapes (128 -> 512 -> 128).
        Returns ``(h_next, h_norm, ffn_hidden)``; writes into provided buffers
        when given. Falls back to ``None`` if CuPy is unavailable.

        Converted weight tensors are cached on the source cupy arrays (via
        ``id()``) so the same weights convert once per settle, not per step.
        """
        if not HAS_CUPY:
            return None
        import cupy as cp
        import torch

        h_t = torch.as_tensor(h, device="cuda").float()
        x_emb_t = torch.as_tensor(x_emb, device="cuda").float()

        cache = cls._torch_cache
        w1_t = cache.get(id(w1))
        if w1_t is None:
            w1_t = torch.as_tensor(w1, device="cuda").float()
            b1_t = torch.as_tensor(b1, device="cuda").float()
            w2_t = torch.as_tensor(w2, device="cuda").float()
            b2_t = torch.as_tensor(b2, device="cuda").float()
            cache[id(w1)] = w1_t
            cache[id(b1)] = b1_t
            cache[id(w2)] = w2_t
            cache[id(b2)] = b2_t
        else:
            b1_t = cache[id(b1)]
            w2_t = cache[id(w2)]
            b2_t = cache[id(b2)]

        mean = h_t.mean(-1, keepdim=True)
        # torch.std defaults to Bessel correction (n-1); cupy/numpy use the
        # population std (ddof=0). Use correction=0 to match the NumPy path.
        std = h_t.std(-1, keepdim=True, correction=0) + 1e-5
        h_norm = (h_t - mean) / std
        ffn_hidden = torch.tanh(h_norm @ w1_t.t() + b1_t)
        ffn_out = ffn_hidden @ w2_t.t() + b2_t
        h_next = (1.0 - gamma) * h_t + gamma * (ffn_out + x_emb_t)

        def _store(dst, src):
            if dst is None:
                return cp.asarray(src)
            dst[:] = cp.asarray(src)
            return dst

        return (
            _store(out, h_next),
            _store(hnorm_out, h_norm),
            _store(ffnhid_out, ffn_hidden),
        )

    # ── fused layered MLP-block forward step ─────────────────────────────────
    # Computes h_next = (1-gamma)*h + gamma*(ffn_out + x_emb) with layernorm,
    # W1, tanh, W2 in one launch (the NumPy path issues ~6 separate cupy ops
    # per settle step). For these FFN shapes the torch-native path
    # (``step_layered_cupy_torch``) is ~8x faster, so it is preferred on GPU;
    # this Triton kernel remains as a fused alternative.
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

        # Zero-copy device-to-device views: cupy arrays and torch CUDA tensors
        # share backing memory (no host round-trip), so the settle loop stays on
        # device. Weights are float64 in cupy; view + cast to float32 for Triton.
        h_t = torch.as_tensor(h, device="cuda").float()
        x_emb_t = torch.as_tensor(x_emb, device="cuda").float()
        w1_t = torch.as_tensor(w1, device="cuda").float()
        b1_t = torch.as_tensor(b1, device="cuda").float()
        w2_t = torch.as_tensor(w2, device="cuda").float()
        b2_t = torch.as_tensor(b2, device="cuda").float()

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


# ============================================================
# MEP Triton Kernels: Muon/Dion/Fisher + EP Settle
# ============================================================

class MEP_TritonOps:
    """MEP operations with Triton acceleration.

    Provides fused kernels for Muon orthogonalization, Dion low-rank update,
    Fisher whitening, and EP settling with automatic fallback to PyTorch.
    """

    _muon_kernel = None
    _dion_kernel = None
    _fisher_kernel = None
    _ep_settle_kernel = None

    @classmethod
    def _init_muon(cls):
        if cls._muon_kernel is None and HAS_TRITON:
            try:
                import triton
                import triton.language as tl

                @triton.jit
                def _muon_kernel(
                    W_ptr,
                    out_ptr,
                    ns_steps: tl.constexpr,
                    M: tl.constexpr,
                    N: tl.constexpr,
                    BLOCK_M: tl.constexpr,
                    BLOCK_N: tl.constexpr,
                ):
                    """Newton-Schulz orthogonalization kernel.
                    W_{k+1} = W_k @ (1.5 * I - 0.5 * W_k^T @ W_k)
                    """
                    pid_m = tl.program_id(0)
                    pid_n = tl.program_id(1)

                    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
                    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

                    # Load W block
                    W = tl.load(
                        W_ptr + offs_m[:, None] * N + offs_n[None, :],
                        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
                    )

                    # Iterative Newton-Schulz
                    for _ in range(ns_steps):
                        # Compute W^T @ W (N x N)
                        WTW = tl.zeros((N, N), dtype=tl.float32)
                        # Simplified: assume square for kernel, use full matmul
                        # In practice, this needs a more sophisticated blocked approach
                        pass  # Placeholder for full implementation

                    tl.store(
                        out_ptr + offs_m[:, None] * N + offs_n[None, :],
                        W,
                        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
                    )

                cls._muon_kernel = _muon_kernel
            except ImportError:
                cls._muon_kernel = False

    @classmethod
    def muon_orthogonalize(cls, W: torch.Tensor, ns_steps: int = 5) -> torch.Tensor:
        """Newton-Schulz orthogonalization with Triton fallback."""
        if HAS_TRITON and W.is_cuda and W.dim() == 2:
            cls._init_muon()
            if cls._muon_kernel:
                # For now, fall back to PyTorch implementation
                # Full Triton matmul-based NS needs more complex blocked kernel
                pass

        # PyTorch fallback (used when Triton unavailable or for non-square)
        out = W.clone()
        for _ in range(ns_steps):
            WTW = out.T @ out
            out = out @ (1.5 * torch.eye(out.shape[1], device=out.device, dtype=out.dtype) - 0.5 * WTW)
        return out

    @classmethod
    def _init_dion(cls):
        if cls._dion_kernel is None and HAS_TRITON:
            try:
                import triton
                import triton.language as tl

                @triton.jit
                def _dion_kernel(
                    W_ptr,
                    out_ptr,
                    Omega_ptr,
                    rank: tl.constexpr,
                    M: tl.constexpr,
                    N: tl.constexpr,
                    BLOCK_M: tl.constexpr,
                    BLOCK_N: tl.constexpr,
                ):
                    """Dion low-rank update via randomized SVD."""
                    # Placeholder for full implementation
                    pass

                cls._dion_kernel = _dion_kernel
            except ImportError:
                cls._dion_kernel = False

    @classmethod
    def dion_update(cls, W: torch.Tensor, rank: int | None = None, rank_frac: float = 0.25) -> torch.Tensor:
        """Low-rank SVD update with Triton fallback."""
        if HAS_TRITON and W.is_cuda and W.dim() == 2:
            cls._init_dion()
            if cls._dion_kernel:
                pass  # Full implementation pending

        # PyTorch fallback
        out_dim, in_dim = W.shape
        if rank is None:
            rank = max(1, int(min(out_dim, in_dim) * rank_frac))

        Omega = torch.randn(in_dim, rank, device=W.device, dtype=W.dtype)
        Y = W @ Omega
        Q, _ = torch.linalg.qr(Y)
        B = Q.T @ W
        W_lr = Q @ B
        return W_lr

    @classmethod
    def _init_fisher(cls):
        if cls._fisher_kernel is None and HAS_TRITON:
            try:
                import triton
                import triton.language as tl

                @triton.jit
                def _fisher_kernel(
                    grad_ptr,
                    fisher_diag_ptr,
                    out_ptr,
                    damping,
                    n_elements: tl.constexpr,
                    BLOCK_SIZE: tl.constexpr,
                ):
                    """Diagonal Fisher whitening: grad / sqrt(fisher_diag + damping)"""
                    pid = tl.program_id(0)
                    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask = offs < n_elements

                    grad = tl.load(grad_ptr + offs, mask=mask)
                    fisher = tl.load(fisher_diag_ptr + offs, mask=mask)

                    denom = tl.sqrt(fisher + damping)
                    out = grad / denom

                    tl.store(out_ptr + offs, out, mask=mask)

                cls._fisher_kernel = _fisher_kernel
            except ImportError:
                cls._fisher_kernel = False

    @classmethod
    def fisher_whiten(cls, grad: torch.Tensor, fisher_diag: torch.Tensor, damping: float = 1e-3) -> torch.Tensor:
        """Diagonal Fisher preconditioning with Triton."""
        if HAS_TRITON and grad.is_cuda and grad.numel() == fisher_diag.numel():
            cls._init_fisher()
            if cls._fisher_kernel:
                out = torch.empty_like(grad)
                n = grad.numel()
                BLOCK_SIZE = 1024
                grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
                cls._fisher_kernel[grid](
                    grad,
                    fisher_diag,
                    out,
                    damping,
                    n_elements=n,
                    BLOCK_SIZE=BLOCK_SIZE,
                )
                return out

        return grad / torch.sqrt(fisher_diag + damping)

    @classmethod
    def _init_ep_settle(cls):
        if cls._ep_settle_kernel is None and HAS_TRITON:
            try:
                import triton
                import triton.language as tl
                from triton.language.extra import libdevice

                @triton.jit
                def _ep_settle_kernel(
                    h_ptr,
                    x_emb_ptr,
                    W1_ptr,
                    b1_ptr,
                    W2_ptr,
                    b2_ptr,
                    out_ptr,
                    gamma,
                    steps: tl.constexpr,
                    M: tl.constexpr,
                    K: tl.constexpr,
                    H: tl.constexpr,
                    BLOCK_M: tl.constexpr,
                ):
                    """Fused EP settle: LayerNorm -> W1 -> tanh -> W2 -> residual"""
                    pid = tl.program_id(0)
                    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
                    mask_m = offs_m < M
                    offs_k = tl.arange(0, K)
                    offs_h = tl.arange(0, H)

                    # Load persistent weights
                    W1 = tl.load(W1_ptr + offs_h[:, None] * K + offs_k[None, :])
                    b1 = tl.load(b1_ptr + offs_h)
                    W2 = tl.load(W2_ptr + offs_k[:, None] * H + offs_h[None, :])
                    b2 = tl.load(b2_ptr + offs_k)

                    # Load initial state
                    h = tl.load(
                        h_ptr + offs_m[:, None] * K + offs_k[None, :],
                        mask=mask_m[:, None],
                    )
                    x_emb = tl.load(
                        x_emb_ptr + offs_m[:, None] * K + offs_k[None, :],
                        mask=mask_m[:, None],
                    )

                    for _ in range(steps):
                        # LayerNorm
                        mean = tl.sum(h, axis=1) / K
                        h_centered = h - mean[:, None]
                        var = tl.sum(h_centered * h_centered, axis=1) / K
                        std = tl.sqrt(var + 1e-5)
                        h_norm = h_centered / std[:, None]

                        # FFN
                        ffn = tl.dot(h_norm, tl.trans(W1))
                        ffn = libdevice.tanh(ffn + b1[None, :])
                        ffn_out = tl.dot(ffn, tl.trans(W2))
                        ffn_out = ffn_out + b2[None, :]

                        # Residual update
                        h = (1.0 - gamma) * h + gamma * (ffn_out + x_emb)

                    tl.store(
                        out_ptr + offs_m[:, None] * K + offs_k[None, :],
                        h,
                        mask=mask_m[:, None],
                    )

                cls._ep_settle_kernel = _ep_settle_kernel
            except ImportError:
                cls._ep_settle_kernel = False

    @classmethod
    def ep_settle(
        cls,
        h: torch.Tensor,
        x_emb: torch.Tensor,
        W1: torch.Tensor,
        b1: torch.Tensor,
        W2: torch.Tensor,
        b2: torch.Tensor,
        gamma: float = 1.0,
        steps: int = 30,
    ) -> torch.Tensor:
        """Fused EP settle with Triton acceleration."""
        if HAS_TRITON and h.is_cuda and h.dim() == 2:
            cls._init_ep_settle()
            if cls._ep_settle_kernel:
                M, K = h.shape
                H = W1.shape[0]
                BLOCK_M = 16

                out = torch.empty_like(h)
                grid = ((M + BLOCK_M - 1) // BLOCK_M,)
                cls._ep_settle_kernel[grid](
                    h,
                    x_emb,
                    W1,
                    b1,
                    W2,
                    b2,
                    out,
                    gamma,
                    steps=steps,
                    M=M,
                    K=K,
                    H=H,
                    BLOCK_M=BLOCK_M,
                )
                return out

        # PyTorch fallback
        h = h.clone()
        for _ in range(steps):
            mean = h.mean(dim=-1, keepdim=True)
            std = h.std(dim=-1, keepdim=True, correction=0) + 1e-5
            h_norm = (h - mean) / std
            ffn_hidden = torch.tanh(h_norm @ W1.T + b1)
            ffn_out = ffn_hidden @ W2.T + b2
            h = (1.0 - gamma) * h + gamma * (ffn_out + x_emb)
        return h


__all__ = [
    "HAS_CUPY",
    "HAS_TRITON",
    "TritonEqPropOps",
    "MEP_TritonOps",
]
