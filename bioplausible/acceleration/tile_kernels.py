"""Tile Substrate Kernel Backend.

Tile-parallel contrastive kernels extending core/tile/kernels.py.
"""

from __future__ import annotations

import torch
from torch import Tensor

from bioplausible.acceleration.kernel_backend import (
    AlgorithmFamily,
    HardwareTarget,
    KernelConfig,
    KernelRegistry,
    LocalityLevel,
)

# ──────────────────────────────────────────────
# Triton Kernels for Tile Substrate
# ──────────────────────────────────────────────

try:
    import triton
    import triton.language as tl

    # ── Fused Tile Activity Update ─────────────────────────────────────────
    # Computes: activity = clamp(activity - step_size * importance * (error + lambda*activity + sum(feedback)))
    @triton.jit
    def _tile_activity_update_kernel(
        activity_ptr,
        error_ptr,
        feedback_ptr,  # [num_feedback, B, N] flattened
        feedback_strides,  # [num_feedback, 3] = [stride_b, stride_n, stride_f]
        num_feedback,
        out_ptr,
        step_size,
        importance,
        lambda_error,
        clamp_min,
        clamp_max,
        clamp,
        B,
        N,
        BLOCK_B: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Fused tile activity update for EP/PC/SNN algorithms."""
        pid_b = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        mask_b = offs_b < B
        mask_n = offs_n < N

        # Load activity and error
        activity = tl.load(
            activity_ptr + offs_b[:, None] * N + offs_n[None, :],
            mask=mask_b[:, None] & mask_n[None, :],
            other=0.0,
        )
        error = tl.load(
            error_ptr + offs_b[:, None] * N + offs_n[None, :],
            mask=mask_b[:, None] & mask_n[None, :],
            other=0.0,
        )

        # Accumulate feedback from all sources
        grad = error + lambda_error * activity
        for f in range(num_feedback):
            fb = tl.load(
                feedback_ptr
                + f * feedback_strides[0]
                + offs_b[:, None] * feedback_strides[1]
                + offs_n[None, :] * feedback_strides[2],
                mask=mask_b[:, None] & mask_n[None, :],
                other=0.0,
            )
            grad = grad + fb

        delta = step_size * importance * grad
        new_activity = activity - delta

        if clamp:
            new_activity = tl.maximum(new_activity, clamp_min)
            new_activity = tl.minimum(new_activity, clamp_max)

        tl.store(
            out_ptr + offs_b[:, None] * N + offs_n[None, :],
            new_activity,
            mask=mask_b[:, None] & mask_n[None, :],
        )

    # ── Fused Tile Prediction ──────────────────────────────────────────────
    # Computes: prediction = sum(inputs) + bias
    @triton.jit
    def _tile_prediction_kernel(
        input_ptrs,  # [num_inputs] array of pointers
        input_strides,  # [num_inputs, 2] = [stride_b, stride_n]
        num_inputs,
        bias_ptr,
        out_ptr,
        B,
        N,
        BLOCK_B: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Fused tile prediction: sum of weighted inputs + bias."""
        pid_b = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        mask_b = offs_b < B
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)

        for i in range(num_inputs):
            inp = tl.load(
                input_ptrs[i] + offs_b[:, None] * input_strides[i, 1] + offs_n[None, :],
                mask=mask_b[:, None] & mask_n[None, :],
                other=0.0,
            )
            acc += inp

        if bias_ptr != 0:
            bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
            acc += bias[None, :]

        tl.store(
            out_ptr + offs_b[:, None] * N + offs_n[None, :],
            acc,
            mask=mask_b[:, None] & mask_n[None, :],
        )

    # ── Fused Contrastive Hebbian Update ───────────────────────────────────
    # Computes: delta = lr/beta * (src_free.T @ dst_free - src_nudged.T @ dst_nudged) / B
    @triton.jit
    def _tile_contrastive_update_kernel(
        src_free_ptr,
        dst_free_ptr,
        src_nudged_ptr,
        dst_nudged_ptr,
        delta_ptr,
        lr,
        beta,
        B,
        D_in,
        D_out,
        BLOCK_IN: tl.constexpr,
        BLOCK_OUT: tl.constexpr,
    ):
        """Fused contrastive Hebbian weight update per tile."""
        pid_in = tl.program_id(0)
        pid_out = tl.program_id(1)

        offs_in = pid_in * BLOCK_IN + tl.arange(0, BLOCK_IN)
        offs_out = pid_out * BLOCK_OUT + tl.arange(0, BLOCK_OUT)

        mask_in = offs_in < D_in
        mask_out = offs_out < D_out

        acc_free = tl.zeros((BLOCK_OUT, BLOCK_IN), dtype=tl.float32)
        acc_nudged = tl.zeros((BLOCK_OUT, BLOCK_IN), dtype=tl.float32)

        for b in range(B):
            # Free phase
            pre_f = tl.load(
                src_free_ptr + b * D_in + offs_in[None, :],
                mask=mask_in[None, :],
                other=0.0,
            )
            post_f = tl.load(
                dst_free_ptr + b * D_out + offs_out[:, None],
                mask=mask_out[:, None],
                other=0.0,
            )
            acc_free += tl.dot(tl.trans(post_f), pre_f)

            # Nudged phase
            pre_n = tl.load(
                src_nudged_ptr + b * D_in + offs_in[None, :],
                mask=mask_in[None, :],
                other=0.0,
            )
            post_n = tl.load(
                dst_nudged_ptr + b * D_out + offs_out[:, None],
                mask=mask_out[:, None],
                other=0.0,
            )
            acc_nudged += tl.dot(tl.trans(post_n), pre_n)

        acc_free = acc_free / B
        acc_nudged = acc_nudged / B

        delta = (lr / beta) * (acc_free - acc_nudged)

        tl.store(
            delta_ptr + offs_out[:, None] * D_in + offs_in[None, :],
            delta,
            mask=mask_out[:, None] & mask_in[None, :],
        )

    # ── Fused Hebbian Update ───────────────────────────────────────────────
    # Computes: delta = importance * (src.T @ dst) / B
    @triton.jit
    def _tile_hebbian_update_kernel(
        src_ptr,
        dst_ptr,
        weight_ptr,
        delta_ptr,
        importance,
        B,
        D_in,
        D_out,
        BLOCK_IN: tl.constexpr,
        BLOCK_OUT: tl.constexpr,
    ):
        """Fused Hebbian weight update per tile."""
        pid_in = tl.program_id(0)
        pid_out = tl.program_id(1)

        offs_in = pid_in * BLOCK_IN + tl.arange(0, BLOCK_IN)
        offs_out = pid_out * BLOCK_OUT + tl.arange(0, BLOCK_OUT)

        mask_in = offs_in < D_in
        mask_out = offs_out < D_out

        acc = tl.zeros((BLOCK_OUT, BLOCK_IN), dtype=tl.float32)

        for b in range(B):
            pre = tl.load(
                src_ptr + b * D_in + offs_in[None, :],
                mask=mask_in[None, :],
                other=0.0,
            )
            post = tl.load(
                dst_ptr + b * D_out + offs_out[:, None],
                mask=mask_out[:, None],
                other=0.0,
            )
            acc += tl.dot(tl.trans(post), pre)

        acc = acc / B
        delta = importance * acc

        # Oja's subtraction term if weight provided
        if weight_ptr != 0:
            post_sq = tl.zeros((BLOCK_OUT, 1), dtype=tl.float32)
            for b in range(B):
                post = tl.load(
                    dst_ptr + b * D_out + offs_out[:, None],
                    mask=mask_out[:, None],
                    other=0.0,
                )
                post_sq += post * post
            post_sq = post_sq / B
            weight = tl.load(
                weight_ptr + offs_out[:, None] * D_in + offs_in[None, :],
                mask=mask_out[:, None] & mask_in[None, :],
                other=0.0,
            )
            delta = delta - post_sq * weight

        tl.store(
            delta_ptr + offs_out[:, None] * D_in + offs_in[None, :],
            delta,
            mask=mask_out[:, None] & mask_in[None, :],
        )

    # ── Tile Routing Kernels (MoT) ─────────────────────────────────────────

    @triton.jit
    def _tile_topk_routing_kernel(
        logits_ptr,
        topk_indices_ptr,
        topk_values_ptr,
        B,
        N,
        K,
        BLOCK_B: tl.constexpr,
    ):
        """Top-K tile routing: select top K tiles per sample."""
        pid_b = tl.program_id(0)

        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        mask_b = offs_b < B

        for b_idx in range(BLOCK_B):
            b = pid_b * BLOCK_B + b_idx
            if not mask_b[b_idx]:
                continue

            # Load logits for this sample
            logits = tl.load(
                logits_ptr + b * N + tl.arange(0, N),
                mask=tl.arange(0, N) < N,
                other=-float("inf"),
            )

            # Simple top-k via selection (for small K, N)
            # In practice, use a more efficient algorithm
            for k in range(K):
                max_val = -float("inf")
                max_idx = 0
                for n in range(N):
                    if logits[n] > max_val:
                        max_val = logits[n]
                        max_idx = n
                topk_indices_ptr[b * K + k] = max_idx
                topk_values_ptr[b * K + k] = max_val
                logits[max_idx] = -float("inf")

    @triton.jit
    def _tile_random_routing_kernel(
        logits_ptr,
        topk_indices_ptr,
        topk_values_ptr,
        B,
        N,
        K,
        seed,
        BLOCK_B: tl.constexpr,
    ):
        """Random tile routing: sample K tiles per sample."""
        pid_b = tl.program_id(0)

        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        mask_b = offs_b < B

        for b_idx in range(BLOCK_B):
            b = pid_b * BLOCK_B + b_idx
            if not mask_b[b_idx]:
                continue

            # Use deterministic random based on seed + batch index
            rng_state = seed + b * 12345 + 67890
            for k in range(K):
                rng_state = rng_state * 1664525 + 1013904223
                idx = rng_state % N
                topk_indices_ptr[b * K + k] = idx
                # Value from logits
                val = tl.load(logits_ptr + b * N + idx)
                topk_values_ptr[b * K + k] = val

    @triton.jit
    def _tile_learned_routing_kernel(
        logits_ptr,
        router_weights_ptr,
        router_bias_ptr,
        topk_indices_ptr,
        topk_values_ptr,
        B,
        N,
        K,
        router_dim,
        BLOCK_B: tl.constexpr,
    ):
        """Learned tile routing: small MLP router."""
        pid_b = tl.program_id(0)

        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        mask_b = offs_b < B

        for b_idx in range(BLOCK_B):
            b = pid_b * BLOCK_B + b_idx
            if not mask_b[b_idx]:
                continue

            # Load logits
            logits = tl.load(
                logits_ptr + b * N + tl.arange(0, N),
                mask=tl.arange(0, N) < N,
                other=0.0,
            )

            # Router: logits -> router_weights -> router_bias -> softmax
            # Simplified: use logits directly with learned temperature
            # Full MLP would require more shared memory
            temp = tl.load(router_weights_ptr) if router_weights_ptr != 0 else 1.0
            logits = logits * temp

            if router_bias_ptr != 0:
                bias = tl.load(router_bias_ptr + tl.arange(0, N), mask=tl.arange(0, N) < N, other=0.0)
                logits = logits + bias

            # Softmax
            max_logit = tl.max(logits, axis=0)
            exp_logits = tl.exp(logits - max_logit)
            sum_exp = tl.sum(exp_logits, axis=0)
            probs = exp_logits / sum_exp

            # Top-k from probs
            for k in range(K):
                max_val = -float("inf")
                max_idx = 0
                for n in range(N):
                    if probs[n] > max_val:
                        max_val = probs[n]
                        max_idx = n
                topk_indices_ptr[b * K + k] = max_idx
                topk_values_ptr[b * K + k] = max_val
                probs[max_idx] = -float("inf")

    HAS_TRITON_TILE = True

except ImportError:
    HAS_TRITON_TILE = False


# ──────────────────────────────────────────────
# Multi-GPU Tile Sharding (NCCL)
# ──────────────────────────────────────────────

class TileShardedBackend:
    """Multi-GPU tile sharding via NCCL.

    Distributes tiles across GPUs, each GPU computes local updates,
    then all-reduces gradients.
    """

    def __init__(self, world_size: int, rank: int, device: torch.device) -> None:
        self.world_size = world_size
        self.rank = rank
        self.device = device
        self._process_group = None
        self._init_nccl()

    def _init_nccl(self) -> None:
        if not torch.distributed.is_initialized():
            # Single-GPU or non-distributed
            return
        self._process_group = torch.distributed.group.WORLD

    def all_reduce_gradients(self, gradients: dict[str, Tensor]) -> dict[str, Tensor]:
        """All-reduce weight gradients across GPUs."""
        if self.world_size <= 1 or self._process_group is None:
            return gradients

        reduced = {}
        for name, grad in gradients.items():
            if grad.is_cuda:
                torch.distributed.all_reduce(
                    grad, op=torch.distributed.ReduceOp.SUM, group=self._process_group
                )
                grad = grad / self.world_size
            reduced[name] = grad
        return reduced

    def broadcast_params(self, params: dict[str, Tensor]) -> dict[str, Tensor]:
        """Broadcast parameters from rank 0 to all."""
        if self.world_size <= 1 or self._process_group is None:
            return params

        broadcast = {}
        for name, param in params.items():
            if param.is_cuda:
                torch.distributed.broadcast(param, src=0, group=self._process_group)
            broadcast[name] = param
        return broadcast


# ──────────────────────────────────────────────
# Tile Kernel Backend with Triton Acceleration
# ──────────────────────────────────────────────

class TileKernelBackend:
    """Tile substrate kernel backend.

    Implements tile-parallel contrastive learning:
    - Each tile is a local compute unit
    - Tiles communicate via message passing
    - Contrastive Hebbian updates per tile
    - Triton-accelerated kernels for GPU
    """

    name = AlgorithmFamily.TILE
    supported_dtypes = (torch.float32, torch.float16, torch.bfloat16)
    supports_autograd = False
    requires_settle = True
    memory_complexity = "O(1)"  # Per-tile O(1), global O(tiles)
    locality_level = LocalityLevel.LOCAL

    def __init__(self) -> None:
        self._config: KernelConfig | None = None
        self._num_tiles: int = 0
        self._neurons_per_tile: int = 0
        self._tiles_per_layer: int = 0
        self._num_hidden_layers: int = 0
        self._beta: float = 0.5
        self._lr: float = 0.01
        self._device: torch.device = torch.device("cpu")
        self._dtype: torch.dtype = torch.float32
        self._last_settle_telemetry: dict[str, object] | None = None
        self._sharded: TileShardedBackend | None = None

    def initialize(self, config: KernelConfig) -> None:
        self._config = config
        self._device = torch.device(
            "cuda"
            if config.hardware in (HardwareTarget.CUDA, HardwareTarget.TRITON)
            else "cpu"
        )
        self._dtype = config.dtype

        extra = config.extra
        self._neurons_per_tile = extra.get("neurons_per_tile", 32)
        self._tiles_per_layer = extra.get("tiles_per_layer", 8)
        self._num_hidden_layers = extra.get("num_hidden_layers", 3)
        self._beta = config.beta
        self._lr = config.extra.get("learning_rate", 0.01)

        self._num_tiles = self._tiles_per_layer * self._num_hidden_layers

        # Multi-GPU sharding
        if config.hardware in (HardwareTarget.CUDA, HardwareTarget.TRITON):
            world_size = extra.get("world_size", 1)
            rank = extra.get("rank", 0)
            if world_size > 1 and torch.distributed.is_initialized():
                self._sharded = TileShardedBackend(world_size, rank, self._device)

    def set_model_ref(self, tile_algorithm) -> None:
        """Set reference to TileAlgorithm instance."""
        self._tile_algo = tile_algorithm

    def _launch_activity_update(
        self,
        activity: Tensor,
        error: Tensor,
        feedback: list[Tensor],
        step_size: float,
        importance: float,
        lambda_error: float,
        clamp_min: float,
        clamp_max: float,
        clamp: bool,
    ) -> Tensor:
        """Launch fused activity update kernel."""
        B, N = activity.shape

        if HAS_TRITON_TILE and self._device.type == "cuda":
            # Stack feedback tensors
            if feedback:
                num_fb = len(feedback)
                fb_stack = torch.stack(feedback, dim=0)  # [num_fb, B, N]
                fb_ptr = fb_stack.data_ptr()
                fb_strides = torch.tensor(
                    [fb_stack.stride(0), fb_stack.stride(1), fb_stack.stride(2)],
                    device="cpu",
                    dtype=torch.int64,
                )
            else:
                num_fb = 0
                fb_ptr = 0
                fb_strides = torch.zeros(3, dtype=torch.int64)

            out = torch.empty_like(activity)
            BLOCK_B = 16
            BLOCK_N = 32
            grid = ((B + BLOCK_B - 1) // BLOCK_B, (N + BLOCK_N - 1) // BLOCK_N)

            _tile_activity_update_kernel[grid](
                activity.data_ptr(),
                error.data_ptr(),
                fb_ptr,
                fb_strides.data_ptr() if num_fb > 0 else 0,
                num_fb,
                out.data_ptr(),
                step_size,
                importance,
                lambda_error,
                clamp_min,
                clamp_max,
                clamp,
                B,
                N,
                BLOCK_B=BLOCK_B,
                BLOCK_N=BLOCK_N,
            )
            return out

        # PyTorch fallback: runs on same device as inputs (CUDA or CPU)
        from bioplausible.core.tile.kernels import compute_activity_update

        return compute_activity_update(
            activity=activity,
            error=error,
            fwd_feedback=feedback,
            importance=importance,
            step_size=step_size,
            lambda_error=lambda_error,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            clamp=clamp,
        )

    def _launch_prediction(
        self,
        inputs: list[Tensor],
        bias: Tensor | None,
    ) -> Tensor:
        """Launch fused prediction kernel."""
        if not inputs:
            if bias is not None:
                return bias.unsqueeze(0).expand(inputs[0].shape[0], -1) if inputs else bias.unsqueeze(0)
            return torch.zeros(1, self._neurons_per_tile, device=self._device, dtype=self._dtype)

        B, N = inputs[0].shape

        if HAS_TRITON_TILE and self._device.type == "cuda":
            num_inputs = len(inputs)
            input_ptrs = torch.tensor([inp.data_ptr() for inp in inputs], dtype=torch.int64, device="cuda")
            input_strides = torch.tensor(
                [[inp.stride(0), inp.stride(1)] for inp in inputs],
                dtype=torch.int64,
                device="cuda",
            )
            bias_ptr = bias.data_ptr() if bias is not None else 0
            out = torch.empty_like(inputs[0])

            BLOCK_B = 16
            BLOCK_N = 32
            grid = ((B + BLOCK_B - 1) // BLOCK_B, (N + BLOCK_N - 1) // BLOCK_N)

            _tile_prediction_kernel[grid](
                input_ptrs,
                input_strides,
                num_inputs,
                bias_ptr,
                out.data_ptr(),
                B,
                N,
                BLOCK_B=BLOCK_B,
                BLOCK_N=BLOCK_N,
            )
            return out

        # PyTorch fallback
        from bioplausible.core.tile.kernels import compute_tile_prediction

        return compute_tile_prediction(inputs, bias)

    def _launch_contrastive_update(
        self,
        src_free: Tensor,
        dst_free: Tensor,
        src_nudged: Tensor,
        dst_nudged: Tensor,
        lr: float,
        beta: float,
    ) -> Tensor:
        """Launch fused contrastive Hebbian update kernel.

        Returns only the weight delta (not bias) for compatibility with the
        test harness which expects dict[str, Tensor] for grads.
        """
        B, D_in = src_free.shape
        _, D_out = dst_free.shape

        if HAS_TRITON_TILE and self._device.type == "cuda":
            delta = torch.empty(D_out, D_in, device=self._device, dtype=self._dtype)
            BLOCK_IN = 32
            BLOCK_OUT = 32
            grid = ((D_in + BLOCK_IN - 1) // BLOCK_IN, (D_out + BLOCK_OUT - 1) // BLOCK_OUT)

            _tile_contrastive_update_kernel[grid](
                src_free.data_ptr(),
                dst_free.data_ptr(),
                src_nudged.data_ptr(),
                dst_nudged.data_ptr(),
                delta.data_ptr(),
                lr,
                beta,
                B,
                D_in,
                D_out,
                BLOCK_IN=BLOCK_IN,
                BLOCK_OUT=BLOCK_OUT,
            )
            return delta

        # PyTorch fallback: compute_contrastive_hebbian_update returns (weight, bias)
        # We only need the weight delta for the test harness
        from bioplausible.core.tile.kernels import compute_contrastive_hebbian_update

        weight_delta, _ = compute_contrastive_hebbian_update(
            src_free=src_free,
            dst_free=dst_free,
            src_nudged=src_nudged,
            dst_nudged=dst_nudged,
            learning_rate=lr,
            beta=beta,
            batch_size=B,
        )
        return weight_delta

    def _launch_hebbian_update(
        self,
        src: Tensor,
        dst: Tensor,
        weight: Tensor | None,
        importance: float,
        use_oja: bool,
    ) -> Tensor:
        """Launch fused Hebbian update kernel."""
        B, D_in = src.shape
        _, D_out = dst.shape

        if HAS_TRITON_TILE and self._device.type == "cuda":
            delta = torch.empty(D_out, D_in, device=self._device, dtype=self._dtype)
            BLOCK_IN = 32
            BLOCK_OUT = 32
            grid = ((D_in + BLOCK_IN - 1) // BLOCK_IN, (D_out + BLOCK_OUT - 1) // BLOCK_OUT)

            _tile_hebbian_update_kernel[grid](
                src.data_ptr(),
                dst.data_ptr(),
                weight.data_ptr() if weight is not None and use_oja else 0,
                delta.data_ptr(),
                importance,
                B,
                D_in,
                D_out,
                BLOCK_IN=BLOCK_IN,
                BLOCK_OUT=BLOCK_OUT,
            )
            return delta

        # PyTorch fallback
        from bioplausible.core.tile.kernels import compute_hebbian_update

        return compute_hebbian_update(
            src_act=src, dst_err=dst, importance=importance, batch_size=B
        )

    def route_tiles(
        self,
        logits: Tensor,
        num_routes: int,
        strategy: str = "topk",
        router_weights: Tensor | None = None,
        router_bias: Tensor | None = None,
        seed: int = 42,
    ) -> tuple[Tensor, Tensor]:
        """Route tiles using specified strategy.

        Args:
            logits: Routing logits [B, num_tiles]
            num_routes: Number of tiles to route to (K)
            strategy: "topk" | "random" | "learned"
            router_weights: Optional learned router weights
            router_bias: Optional learned router bias
            seed: Random seed for stochastic routing

        Returns:
            (indices [B, K], values [B, K])
        """
        B, N = logits.shape
        device = logits.device

        indices = torch.empty(B, num_routes, dtype=torch.int64, device=device)
        values = torch.empty(B, num_routes, dtype=logits.dtype, device=device)

        if HAS_TRITON_TILE and device.type == "cuda":
            BLOCK_B = 16
            grid = ((B + BLOCK_B - 1) // BLOCK_B,)

            if strategy == "topk":
                _tile_topk_routing_kernel[grid](
                    logits.data_ptr(),
                    indices.data_ptr(),
                    values.data_ptr(),
                    B,
                    N,
                    num_routes,
                    BLOCK_B=BLOCK_B,
                )
            elif strategy == "random":
                _tile_random_routing_kernel[grid](
                    logits.data_ptr(),
                    indices.data_ptr(),
                    values.data_ptr(),
                    B,
                    N,
                    num_routes,
                    seed,
                    BLOCK_B=BLOCK_B,
                )
            elif strategy == "learned":
                router_w_ptr = router_weights.data_ptr() if router_weights is not None else 0
                router_b_ptr = router_bias.data_ptr() if router_bias is not None else 0
                router_dim = router_weights.shape[0] if router_weights is not None else N
                _tile_learned_routing_kernel[grid](
                    logits.data_ptr(),
                    router_w_ptr,
                    router_b_ptr,
                    indices.data_ptr(),
                    values.data_ptr(),
                    B,
                    N,
                    num_routes,
                    router_dim,
                    BLOCK_B=BLOCK_B,
                )
        # PyTorch fallback
        elif strategy == "topk":
            values, indices = torch.topk(logits, num_routes, dim=1)
        elif strategy == "random":
            gen = torch.Generator(device=device).manual_seed(seed)
            indices = torch.multinomial(torch.softmax(logits, dim=1), num_routes, generator=gen)
            values = torch.gather(logits, 1, indices)
        elif strategy == "learned":
            if router_weights is not None:
                logits = logits @ router_weights.t()
            if router_bias is not None:
                logits = logits + router_bias
            probs = torch.softmax(logits, dim=1)
            values, indices = torch.topk(probs, num_routes, dim=1)

        return indices, values

    def tile_forward(
        self,
        x: Tensor,
        tile_states: list[Tensor] | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        """Forward pass through tile substrate."""
        x = x.to(device=self._device, dtype=self._dtype)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        batch_size = x.shape[0]

        if tile_states is None:
            tile_states = [
                torch.zeros(
                    batch_size,
                    self._neurons_per_tile,
                    device=self._device,
                    dtype=self._dtype,
                )
                for _ in range(self._num_tiles)
            ]

        current_acts = x

        for layer_idx in range(self._num_hidden_layers):
            layer_tiles = tile_states[
                layer_idx * self._tiles_per_layer : (layer_idx + 1)
                * self._tiles_per_layer
            ]

            new_tile_states = []
            for tile_idx, tile_state in enumerate(layer_tiles):
                new_state = self._tile_local_update(
                    tile_state, current_acts, layer_idx, tile_idx
                )
                new_tile_states.append(new_state)

            tile_states[
                layer_idx * self._tiles_per_layer : (layer_idx + 1)
                * self._tiles_per_layer
            ] = new_tile_states

            current_acts = torch.cat(new_tile_states, dim=1)

        return current_acts, tile_states

    def _tile_local_update(
        self,
        tile_state: Tensor,
        input_acts: Tensor,
        layer_idx: int,
        tile_idx: int,
    ) -> Tensor:
        """Single tile local update (simplified)."""
        return torch.tanh(tile_state + input_acts.mean(dim=1, keepdim=True))

    def settle(
        self,
        x: Tensor,
        beta: float = 0.0,
        steps: int = 10,
    ) -> tuple[list[Tensor], dict[str, float]]:
        """Settle tile substrate to equilibrium."""
        tile_states = None
        telemetry = {"steps": steps, "converged": False, "final_delta": 0.0}
        prev_tile_states: list[Tensor] | None = None

        for step in range(steps):
            output, tile_states = self.tile_forward(x, tile_states)

            if step > 0 and prev_tile_states is not None:
                delta = sum(
                    (s - prev_s).abs().max().item()
                    for s, prev_s in zip(tile_states, prev_tile_states)
                )
                telemetry["final_delta"] = delta
                if delta < 1e-4:
                    telemetry["converged"] = True
                    telemetry["steps"] = step + 1
                    break

            prev_tile_states = [s.clone() for s in tile_states]

        self._last_settle_telemetry = telemetry
        return tile_states, telemetry

    def backward_contrastive(
        self,
        free_states: list[Tensor],
        nudged_states: list[Tensor],
    ) -> dict[str, Tensor]:
        """Contrastive Hebbian update for tile substrate.

        Uses Triton-accelerated kernels when available.
        """
        weight_deltas: dict[str, Tensor] = {}

        for layer_idx in range(self._num_hidden_layers):
            for tile_idx in range(self._tiles_per_layer):
                idx = layer_idx * self._tiles_per_layer + tile_idx

                free_pre = (
                    free_states[idx] if idx < len(free_states) else free_states[-1]
                )
                free_post = free_states[idx]
                nudged_pre = (
                    nudged_states[idx]
                    if idx < len(nudged_states)
                    else nudged_states[-1]
                )
                nudged_post = nudged_states[idx]

                # Use Triton-accelerated contrastive update
                delta = self._launch_contrastive_update(
                    free_pre, free_post, nudged_pre, nudged_post, self._lr, self._beta
                )

                weight_deltas[f"tiles.layer{layer_idx}.tile{tile_idx}.weight"] = delta

        # Multi-GPU all-reduce
        if self._sharded is not None:
            weight_deltas = self._sharded.all_reduce_gradients(weight_deltas)

        return weight_deltas

    def backward_hebbian(
        self,
        activations: list[Tensor],
        importance: float = 1.0,
        use_oja: bool = True,
    ) -> dict[str, Tensor]:
        """Pure Hebbian update for all tile edges."""
        weight_deltas: dict[str, Tensor] = {}

        for layer_idx in range(self._num_hidden_layers):
            for tile_idx in range(self._tiles_per_layer):
                idx = layer_idx * self._tiles_per_layer + tile_idx

                src = activations[idx] if idx < len(activations) else activations[-1]
                dst = activations[idx + 1] if idx + 1 < len(activations) else activations[-1]

                weight = None
                if hasattr(self, "_tile_algo") and self._tile_algo is not None:
                    # Get weight from tile algorithm
                    src_id = layer_idx * self._tiles_per_layer + tile_idx - self._tiles_per_layer
                    dst_id = idx
                    if src_id >= 0 and hasattr(self._tile_algo, "_weight_lookup"):
                        weight = self._tile_algo._weight_lookup(src_id, dst_id)

                delta = self._launch_hebbian_update(src, dst, weight, importance, use_oja)
                weight_deltas[f"tiles.layer{layer_idx}.tile{tile_idx}.weight"] = delta

        if self._sharded is not None:
            weight_deltas = self._sharded.all_reduce_gradients(weight_deltas)

        return weight_deltas

    def update_weights(self, gradients: dict[str, Tensor], lr: float = 1.0) -> None:
        """Apply weight updates to tile algorithm."""
        if hasattr(self, "_tile_algo") and self._tile_algo is not None:
            self._tile_algo.apply_weight_updates(gradients, lr)

    def get_memory_stats(self) -> dict[str, float]:
        tile_params = self._num_tiles * self._neurons_per_tile * self._neurons_per_tile
        state_mb = self._num_tiles * self._neurons_per_tile * 4 / 1e6

        return {
            "tile_params_mb": tile_params * 4 / 1e6,
            "tile_states_mb": state_mb,
            "activations_mb": 0.0,
        }

    def get_settle_telemetry(self) -> dict[str, object] | None:
        return self._last_settle_telemetry


# Register backend for all HardwareTargets
for hw in HardwareTarget:
    KernelRegistry.register(AlgorithmFamily.TILE, hw, TileKernelBackend)


__all__ = [
    "HAS_TRITON_TILE",
    "TileKernelBackend",
    "TileShardedBackend",
]
