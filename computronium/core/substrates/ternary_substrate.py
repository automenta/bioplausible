"""Ternary Substrate for { -1, 0, +1 } Weight Quantization with STE.

Implements ternary weight networks (TWN) with Straight-Through Estimator
for gradient backpropagation through the quantization function.

Key features:
- Ternary quantization: weights ∈ {-α, 0, +α} with learnable scale α
- STE for gradient estimation through quantization
- Delta ternary networks: separate positive/negative scales
- Support for weight decay on full-precision latent weights
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
from torch import Tensor, nn

from computronium.core.ontology import DigitalSubstrate, SubstrateConfig


class TernaryQuantize(torch.autograd.Function):
    """Straight-Through Estimator for ternary quantization.

    Forward:  w_q = α * sign(w) * (|w| > threshold)
    Backward: ∂L/∂w = ∂L/∂w_q (straight-through for |w| < threshold)
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        w: Tensor,
        threshold: float,
        alpha: Tensor,
    ) -> Tensor:
        ctx.save_for_backward(w, alpha)
        ctx.threshold = threshold
        # Ternary: sign(w) * alpha where |w| > threshold, else 0
        w_q = alpha * torch.sign(w) * (w.abs() > threshold).to(w.dtype)
        return w_q

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: Tensor
    ) -> tuple[Tensor, None, Tensor]:
        w, alpha = ctx.saved_tensors
        threshold = ctx.threshold
        # STE: gradient passes through for |w| < threshold, zero elsewhere
        # This encourages weights to move toward ternary values
        mask = (w.abs() <= threshold).to(w.dtype)
        grad_w = grad_output * mask
        # Gradient for alpha: sum of grad_output * sign(w) * (|w| > threshold)
        grad_alpha = (
            grad_output * torch.sign(w) * (w.abs() > threshold).to(w.dtype)
        ).sum()
        return grad_w, None, grad_alpha


class TernarySubstrate(DigitalSubstrate):
    """Ternary weight substrate with STE-based gradient estimation.

    Implements Ternary Weight Networks (TWN) and variants:
    - Standard TWN: single scale α for both +1 and -1
    - Delta Ternary: separate α_pos, α_neg for asymmetric quantization
    - Trained Threshold: threshold learned via STE

    The substrate maintains full-precision latent weights for optimization
    and quantizes to ternary on the forward pass.
    """

    def __init__(
        self,
        config: SubstrateConfig | None = None,
        *,
        ternary_type: Literal["standard", "delta", "trained_threshold"] = "standard",
        threshold_init: float = 0.05,
        learn_threshold: bool = False,
        weight_decay: float = 0.0,
        alpha_init: float = 1.0,
    ):
        super().__init__(
            config
            or SubstrateConfig(
                precision="float32",
                noise_level=0.0,
                weight_bounds=(-1.0, 1.0),
                sparsity=0.0,  # Sparsity emerges from thresholding
                device="cpu",
            )
        )
        self.ternary_type = ternary_type
        self.threshold_init = threshold_init
        self.learn_threshold = learn_threshold
        self.weight_decay = weight_decay
        self.alpha_init = alpha_init

        # State for quantization parameters
        self._alpha: dict[str, nn.Parameter] = {}
        self._alpha_neg: dict[str, nn.Parameter] = {}
        self._threshold: dict[str, nn.Parameter] = {}
        self._latent_weights: dict[
            str, Tensor
        ] = {}  # Full-precision weights for optimizer

    @classmethod
    def from_config(cls, config: SubstrateConfig) -> TernarySubstrate:
        """Create TernarySubstrate from SubstrateConfig."""
        return cls(config=config)

    # =========================================================================
    # Parameter Management
    # =========================================================================

    def _get_or_create_params(
        self, weight: Tensor, name: str
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Get or create quantization parameters for a weight matrix."""
        device = weight.device

        if name not in self._alpha:
            # α initialized from latent-weight magnitude, not a fixed 1.0:
            # unit-magnitude quantized weights give wide layers a settling
            # gain that explodes (ρ ~ 1e8 on composed systems), while
            # mean(|w|) scales with fan-in like the init distribution.
            with torch.no_grad():
                scale = max(weight.abs().mean().item() * self.alpha_init, 1e-8)
            if self.ternary_type == "delta":
                self._alpha[name] = nn.Parameter(torch.tensor(scale, device=device))
                self._alpha_neg[name] = nn.Parameter(torch.tensor(scale, device=device))
            else:
                self._alpha[name] = nn.Parameter(torch.tensor(scale, device=device))

            if self.learn_threshold:
                self._threshold[name] = nn.Parameter(
                    torch.tensor(self.threshold_init, device=device)
                )

            # Store reference to full-precision latent weights
            self._latent_weights[name] = weight.detach().clone()

        alpha = self._alpha[name]
        alpha_neg = self._alpha_neg.get(name, alpha)
        threshold = self._threshold.get(
            name, torch.tensor(self.threshold_init, device=device)
        )

        return alpha, alpha_neg, threshold

    def _compute_scale(self, weight: Tensor) -> Tensor:
        """Compute optimal scale α = mean(|w|) for standard ternary."""
        return weight.abs().mean()

    # =========================================================================
    # Substrate Interface
    # =========================================================================

    def quantize_weights(self, w: Tensor) -> Tensor:
        """Quantize weights to ternary {-α, 0, +α} using STE."""
        name = getattr(w, "_param_name", "default")
        alpha, alpha_neg, threshold = self._get_or_create_params(w, name)

        # Update latent weights if provided (from optimizer step)
        if name in self._latent_weights and self._latent_weights[name] is not w:
            self._latent_weights[name] = w.detach().clone()

        # Use latent weights for quantization if available
        latent_w = self._latent_weights.get(name, w)

        if self.ternary_type == "delta":
            # Delta ternary: separate scales for positive and negative
            w_pos = TernaryQuantize.apply(latent_w, threshold, alpha)
            w_neg = TernaryQuantize.apply(-latent_w, threshold, alpha_neg)
            w_q = w_pos - w_neg
        else:
            # Standard ternary
            w_q = TernaryQuantize.apply(latent_w, threshold, alpha)

        # Apply weight decay to latent weights
        if self.weight_decay > 0 and name in self._latent_weights:
            with torch.no_grad():
                self._latent_weights[name].mul_(1 - self.weight_decay)

        return self._to_precision(w_q)

    def inject_state_noise(self, s: Tensor) -> Tensor:
        """Add noise to activations."""
        return super().inject_state_noise(s)

    def get_forward_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        """Forward operator with ternary weights."""

        def ternary_forward(x: Tensor, w: Tensor) -> Tensor:
            x = self._to_precision(x)
            w = self._to_precision(w)
            w_q = self.quantize_weights(w)
            return self._to_precision(x @ w_q.T)

        return ternary_forward

    def get_weight_update_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        """Weight update operator for ternary substrate.

        The pseudo-gradient is applied to the full-precision latent weights.
        The ternary quantization is handled by STE in quantize_weights().
        """

        def ternary_update(pseudo_grad: Tensor, current_w: Tensor) -> Tensor:
            pseudo_grad = self._to_precision(pseudo_grad)
            current_w = self._to_precision(current_w)
            name = getattr(current_w, "_param_name", "default")

            # Update latent weights (full precision)
            if name in self._latent_weights:
                latent_w = self._latent_weights[name]
                # Apply weight decay
                if self.weight_decay > 0:
                    latent_w = latent_w * (1 - self.weight_decay)
                # Apply pseudo-gradient (SGD step)
                step_size = getattr(self.config, "step_size", 0.01)
                latent_w = latent_w - step_size * pseudo_grad
                self._latent_weights[name] = latent_w
                # Return quantized weights for next forward pass
                return self.quantize_weights(latent_w)

            # Fallback: direct update on current weights
            step_size = getattr(self.config, "step_size", 0.01)
            return self._to_precision(current_w - step_size * pseudo_grad)

        return ternary_update

    def initial_state(self, x: Tensor) -> Tensor:
        return x

    def get_quantization_params(self, name: str) -> dict[str, Tensor] | None:
        """Get current quantization parameters for inspection."""
        if name not in self._alpha:
            return None
        params = {"alpha": self._alpha[name].data.clone()}
        if self.ternary_type == "delta":
            params["alpha_neg"] = self._alpha_neg[name].data.clone()
        if self.learn_threshold:
            params["threshold"] = self._threshold[name].data.clone()
        return params

    def set_latent_weights(self, name: str, weights: Tensor) -> None:
        """Set full-precision latent weights (e.g., from checkpoint)."""
        self._latent_weights[name] = weights.detach().clone()

    def get_latent_weights(self, name: str) -> Tensor | None:
        """Get full-precision latent weights."""
        return self._latent_weights.get(name)

    def get_ternary_weights(self, name: str) -> Tensor | None:
        """Get current ternary quantized weights."""
        latent = self._latent_weights.get(name)
        if latent is None:
            return None
        return self.quantize_weights(latent)

    def sparsity_stats(self) -> dict[str, float]:
        """Return sparsity statistics (fraction of zero weights)."""
        stats = {}
        for name in self._latent_weights:
            alpha, alpha_neg, threshold = self._get_or_create_params(
                self._latent_weights[name], name
            )
            w_q = self.quantize_weights(self._latent_weights[name])
            sparsity = (w_q == 0).float().mean().item()
            stats[name] = sparsity
        return stats
