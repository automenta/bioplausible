"""Equilibrium Propagation model variants."""

import torch
import torch.nn.functional as F
from torch import nn

from computronium.core.model_status import status_tag
from computronium.core.registry import LocalityLevel, register_model
from computronium.zoo.models.transitions import TransitionGraphMixin

__all__ = [
    "TernaryEqProp",
    "TernaryLinear",
    "TernaryQuantize",
]


class TernaryQuantize(torch.autograd.Function):
    """
    Ternary quantization with Straight-Through Estimator.

    Forward: Quantize weights to {-1, 0, +1}
    Backward: Pass gradients through unchanged (STE)
    """

    @staticmethod
    def forward(ctx, weight: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        ctx.save_for_backward(weight)

        ternary = torch.zeros_like(weight)
        ternary[weight > threshold] = 1.0
        ternary[weight < -threshold] = -1.0

        return ternary

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (_weight,) = ctx.saved_tensors
        grad_weight = grad_output.clone()
        return grad_weight, None


class TernaryLinear(nn.Module):
    """Linear layer with ternary weights."""

    def __init__(self, in_features: int, out_features: int, threshold: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        nn.init.xavier_uniform_(self.weight, gain=0.8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ternary_weight = TernaryQuantize.apply(self.weight, self.threshold)
        return F.linear(x, ternary_weight, self.bias)

    def get_weight_stats(self) -> dict:
        w = self.weight.detach()
        threshold = self.threshold

        n_pos = (w > threshold).sum().item()
        n_neg = (w < -threshold).sum().item()
        n_zero = w.numel() - n_pos - n_neg

        total = w.numel()
        return {
            "positive": n_pos / total,
            "zero": n_zero / total,
            "negative": n_neg / total,
            "sparsity": n_zero / total,
        }


@register_model(
    "ternary_eqprop",
    locality_level=LocalityLevel.EQUILIBRIUM,
    bio_plausibility_score=0.85,
    credit_assignment_type="equilibrium",
    requires_backward=False,
    memory_complexity="O(1)",
    family="eqprop",
    typical_lr_range=(0.001, 0.01),
    tags=["eqprop", "ternary", "quantized", status_tag("experimental")],
    extra={"quantization": "ternary", "parity_threshold": 0.1},
)
class TernaryEqProp(TransitionGraphMixin, nn.Module):
    """
    Equilibrium Propagation with Ternary Weights.

    Combines recurrent fixed-point dynamics with extreme quantization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        threshold: float = 0.5,
        max_steps: int = 30,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.threshold = threshold
        self.max_steps = max_steps

        self.W_in = TernaryLinear(input_dim, hidden_dim, threshold)
        self.W_rec = TernaryLinear(hidden_dim, hidden_dim, threshold)
        self.W_out = TernaryLinear(hidden_dim, output_dim, threshold)

    def transition_modules(self) -> list[nn.Module]:
        """Modules called in order during one forward step.

        :returns: ``[self.W_in, self.W_rec, self.W_out]``
        """
        return [self.W_in, self.W_rec, self.W_out]

    def forward(
        self,
        x: torch.Tensor,
        steps: int | None = None,
    ) -> torch.Tensor:
        steps = steps or self.max_steps
        batch_size = x.shape[0]

        h = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)

        x_proj = self.W_in(x)

        for _ in range(steps):
            h = torch.tanh(x_proj + self.W_rec(h))

        return self.W_out(h)

    def get_model_stats(self) -> dict:
        stats = {
            "W_in": self.W_in.get_weight_stats(),
            "W_rec": self.W_rec.get_weight_stats(),
            "W_out": self.W_out.get_weight_stats(),
        }

        total_zero = sum(s["sparsity"] for s in stats.values())
        stats["overall_sparsity"] = total_zero / 3

        return stats

    def count_bit_operations(self) -> dict:
        in_ops = self.input_dim * self.hidden_dim
        rec_ops = self.hidden_dim * self.hidden_dim
        out_ops = self.hidden_dim * self.output_dim
        total_ops = in_ops + rec_ops * self.max_steps + out_ops

        float32_ops = total_ops * 2

        sparsity = self.get_model_stats()["overall_sparsity"]
        ternary_ops = int(total_ops * (1 - sparsity))

        return {
            "float32_operations": float32_ops,
            "ternary_operations": ternary_ops,
            "speedup_factor": (
                float32_ops / ternary_ops if ternary_ops > 0 else float("inf")
            ),
            "sparsity_used": sparsity,
        }

    @classmethod
    def build(
        cls,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        device: str,
        **_kwargs,
    ):
        """Factory method for registry instantiation."""
        return cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        ).to(device)
