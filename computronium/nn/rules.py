"""Credit assignment rules for ComputroniumLinear.

Self-contained, torch-only implementations of bio-plausible learning rules.
Each rule computes pseudo-gradients for a single linear layer given
input, output, and upstream gradient (grad_output).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import Tensor


class CreditRule(StrEnum):
    """Enumeration of supported credit assignment rules."""

    BACKPROP = "backprop"
    FA = "fa"
    HEBBIAN = "hebbian"
    EQPROP = "eqprop"


@dataclass(frozen=True, slots=True)
class CreditRuleConfig:
    """Configuration for a credit rule."""

    rule: CreditRule = CreditRule.BACKPROP
    # FA: fixed random feedback matrix scale
    feedback_scale: float = 1.0
    # HEBBIAN: activation function for post-synaptic signal
    hebbian_nonlinearity: str = "sign"  # "sign" | "tanh" | "identity"
    # EQPROP: contrastive scaling factor (beta)
    eqprop_beta: float = 0.1
    # EQPROP: number of settling steps (if we do internal settling)
    eqprop_steps: int = 20


def _generate_feedback_matrix(
    in_features: int, out_features: int, scale: float, device: torch.device
) -> Tensor:
    """Generate a fixed random feedback matrix B of shape (out_features, in_features).

    Used by FA rule for upstream gradient propagation instead of W^T.
    Deterministic per (in_features, out_features, scale) triple.
    """
    generator = torch.Generator(device=device)
    seed = (in_features * 12345 + out_features * 67890 + int(scale * 1000) * 42) % (
        2**32
    )
    generator.manual_seed(seed)
    return (
        torch.randn(out_features, in_features, generator=generator, device=device)
        * scale
    )


def _hebbian_post(grad_output: Tensor, nonlinearity: str) -> Tensor:
    """Transform grad_output into a Hebbian post-synaptic signal."""
    match nonlinearity:
        case "sign":
            return torch.sign(grad_output)
        case "tanh":
            return torch.tanh(grad_output)
        case "identity":
            return grad_output
        case _:
            raise ValueError("Unknown hebbian nonlinearity")


def backprop_pseudo_grad(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    grad_output: Tensor,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Exact native backprop pseudo-gradients for a linear layer.

    Returns:
        grad_x: (batch, in_features) — gradient w.r.t. input
        grad_weight: (out_features, in_features) — gradient w.r.t. weight
        grad_bias: (out_features,) or None — gradient w.r.t. bias
    """
    grad_x = grad_output @ weight
    grad_weight = grad_output.T @ x
    grad_bias = grad_output.sum(0) if bias is not None else None
    return grad_x, grad_weight, grad_bias


def fa_pseudo_grad(
    x: Tensor,
    _weight: Tensor,
    bias: Tensor | None,
    grad_output: Tensor,
    feedback: Tensor,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Feedback Alignment pseudo-gradients.

    Uses fixed random feedback matrix B for upstream gradient propagation.
    Weight gradient uses standard form (same as backprop); input gradient uses B.
    """
    grad_weight = grad_output.T @ x
    grad_x = grad_output @ feedback
    grad_bias = grad_output.sum(0) if bias is not None else None
    return grad_x, grad_weight, grad_bias


def hebbian_pseudo_grad(
    x: Tensor,
    _weight: Tensor,
    bias: Tensor | None,
    grad_output: Tensor,
    nonlinearity: str = "sign",
) -> tuple[Tensor | None, Tensor, Tensor | None]:
    """Hebbian pseudo-gradients: purely local, no upstream propagation.

    Weight update: post-synaptic signal ⊗ pre-synaptic activity.
    grad_x = None (zero) — no error signal propagates upstream.
    """
    post = _hebbian_post(grad_output, nonlinearity)
    pre = x
    grad_weight = post.T @ pre
    grad_bias = post.sum(0) if bias is not None else None
    grad_x = None
    return grad_x, grad_weight, grad_bias


def eqprop_pseudo_grad(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    grad_output: Tensor,
    beta: float,
    _steps: int,
    free_output: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Equilibrium Propagation pseudo-gradients for a linear layer.

    For a single linear layer, EqProp free/nudged contrastive gradient
    reduces to standard backprop scaled by 1/beta. The free phase output
    can be provided or computed.

    In a full network, EqProp requires free-phase settling, then nudged
    phase settling with target. For an isolated linear layer, this
    simplifies to the contrastive gradient proportional to backprop.
    """
    if free_output is None:
        free_output = x @ weight.T
        if bias is not None:
            free_output += bias

    scale = 1.0 / beta if beta != 0 else 1.0
    grad_x, grad_weight, grad_bias = backprop_pseudo_grad(x, weight, bias, grad_output)
    return (
        grad_x * scale,
        grad_weight * scale,
        grad_bias * scale if grad_bias is not None else None,
    )


def compute_pseudo_gradients(
    rule: CreditRule,
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    grad_output: Tensor,
    config: CreditRuleConfig,
    feedback: Tensor | None = None,
    free_output: Tensor | None = None,
) -> tuple[Tensor | None, Tensor, Tensor | None]:
    """Dispatch to the appropriate pseudo-gradient function."""
    match rule:
        case CreditRule.BACKPROP:
            return backprop_pseudo_grad(x, weight, bias, grad_output)
        case CreditRule.FA:
            if feedback is None:
                feedback = _generate_feedback_matrix(
                    x.shape[-1], weight.shape[0], config.feedback_scale, x.device
                )
            return fa_pseudo_grad(x, weight, bias, grad_output, feedback)
        case CreditRule.HEBBIAN:
            return hebbian_pseudo_grad(
                x, weight, bias, grad_output, config.hebbian_nonlinearity
            )
        case CreditRule.EQPROP:
            return eqprop_pseudo_grad(
                x,
                weight,
                bias,
                grad_output,
                config.eqprop_beta,
                config.eqprop_steps,
                free_output,
            )
        case _:
            raise ValueError("Unknown credit rule")
