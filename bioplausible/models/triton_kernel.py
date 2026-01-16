
"""
Triton Kernels for EqProp Acceleration (Priority 2)

Provides fused kernels for Equilibrium Propagation dynamics to maximize
GPU throughput by reducing memory bandwidth usage.
"""

import torch
from typing import Optional

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None

if HAS_TRITON:
    @triton.jit
    def _eqprop_step_kernel(
        h_ptr,          # Current hidden state
        pre_act_ptr,    # Linear projection (Wx + Wh)
        bias_ptr,       # Bias
        out_ptr,        # Output pointer
        alpha,          # Nudge factor
        n_elements,     # Total elements
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fused kernel for: h_new = (1 - alpha) * h + alpha * tanh(pre_act + bias)
        """
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Load data
        h = tl.load(h_ptr + offsets, mask=mask)
        pre = tl.load(pre_act_ptr + offsets, mask=mask)

        # Load bias (optional broadcasting handling could be added, but assuming flattened for now)
        # If bias is [hidden_dim] and others are [batch, hidden_dim], we need modulo indexing
        # But for simplicity, let's assume the caller expands bias or adds it to pre_act beforehand.
        # Actually, adding bias to pre_act beforehand in PyTorch is fast.
        # Let's assume pre_act ALREADY includes bias for maximum simplicity or we handle it here.
        # Handling it here is better for fusion.
        # BUT, standard matmul result doesn't include bias.
        # Let's keep it simple: caller adds bias to pre_act using addmm or similar,
        # OR we pass bias pointer.
        # If bias is vector, we need the column index.
        # offsets is linear index. col = offsets % hidden_dim.
        # That requires passing hidden_dim.

        # Let's assume pre_act includes bias for now to avoid complex indexing in v1.

        val = tl.math.tanh(pre)
        out = (1.0 - alpha) * h + alpha * val

        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _eqprop_step_kernel_with_bias(
        h_ptr,          # Current hidden state
        pre_act_ptr,    # Linear projection (Wx + Wh)
        bias_ptr,       # Bias vector
        out_ptr,        # Output pointer
        alpha,          # Nudge factor
        n_rows,         # Batch size
        n_cols,         # Hidden dim
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fused kernel including bias addition:
        h_new = (1 - alpha) * h + alpha * tanh(pre_act + bias)
        """
        # Program ID covers the flattened array
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)

        n_elements = n_rows * n_cols
        mask = offsets < n_elements

        # Map linear offset to row/col
        # row = offsets // n_cols  # Not needed
        col = offsets % n_cols

        # Load
        h = tl.load(h_ptr + offsets, mask=mask)
        pre = tl.load(pre_act_ptr + offsets, mask=mask)
        b = tl.load(bias_ptr + col, mask=mask) # Broadcast bias

        val = tl.math.tanh(pre + b)
        out = (1.0 - alpha) * h + alpha * val

        tl.store(out_ptr + offsets, out, mask=mask)


class TritonEqPropOps:
    """Interface for Triton kernels."""

    @staticmethod
    def is_available():
        return HAS_TRITON and torch.cuda.is_available()

    @staticmethod
    def step(h: torch.Tensor, pre_act: torch.Tensor, alpha: float, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Perform one EqProp step: h <- (1-a)h + a*tanh(pre_act + bias)
        """
        if not TritonEqPropOps.is_available():
            # Fallback
            if bias is not None:
                return (1 - alpha) * h + alpha * torch.tanh(pre_act + bias)
            return (1 - alpha) * h + alpha * torch.tanh(pre_act)

        # Ensure contiguity for safe pointer access
        if not h.is_contiguous():
            h = h.contiguous()
        if not pre_act.is_contiguous():
            pre_act = pre_act.contiguous()
        if bias is not None and not bias.is_contiguous():
            bias = bias.contiguous()

        # Prepare output
        out = torch.empty_like(h)

        n_elements = h.numel()

        # Heuristic for block size
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        if bias is not None:
            # Check shapes
            assert bias.dim() == 1
            assert h.dim() == 2
            n_rows, n_cols = h.shape
            assert bias.shape[0] == n_cols

            _eqprop_step_kernel_with_bias[grid](
                h, pre_act, bias, out,
                alpha, n_rows, n_cols,
                BLOCK_SIZE=BLOCK_SIZE
            )
        else:
            _eqprop_step_kernel[grid](
                h, pre_act, None, out,  # bias ptr unused
                alpha, n_elements,
                BLOCK_SIZE=BLOCK_SIZE
            )

        return out
