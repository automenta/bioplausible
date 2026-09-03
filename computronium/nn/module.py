"""ComputroniumLinear: Drop-in PyTorch layer with pluggable bio-plausible credit assignment.

Replaces `nn.Linear` + optimizer with a single layer that internally handles
free/nudged phases, settling loops, and plastic state (ψ) bookkeeping.

NullPlasticity + Backprop rule falls back to native behavior bit-for-bit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from computronium.nn.plasticity import (
    FastWeightPlasticity,
    NullPlasticity,
    PlasticityConfig,
    PlasticityType,
    create_plasticity,
)
from computronium.nn.rules import (
    CreditRule,
    CreditRuleConfig,
    _generate_feedback_matrix,
    compute_pseudo_gradients,
)


@dataclass(frozen=True, slots=True)
class ComputroniumLinearConfig:
    """Configuration for ComputroniumLinear layer."""

    # Credit assignment rule
    rule: CreditRule = CreditRule.BACKPROP
    # Plasticity mechanism
    plasticity: PlasticityType = PlasticityType.NULL
    # Credit rule config
    credit_config: CreditRuleConfig = field(default_factory=CreditRuleConfig)
    # Plasticity config
    plasticity_config: PlasticityConfig = field(default_factory=PlasticityConfig)


class _ComputroniumLinearFn(torch.autograd.Function):
    """Custom autograd Function for ComputroniumLinear.

    Forward: native linear transform (bit-for-bit compatible with nn.Linear).
    Backward: computes pseudo-gradients per the configured credit rule.
    """

    @staticmethod
    def forward(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
        ctx,
        x: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        rule: CreditRule,
        credit_config: CreditRuleConfig,
        plasticity: PlasticityType,
        plasticity_config: PlasticityConfig,
        in_features: int,
        out_features: int,
        feedback: Tensor | None,
        free_output: Tensor | None,
    ) -> Tensor:
        # Save for backward
        ctx.save_for_backward(x, weight, bias if bias is not None else torch.tensor([]))
        ctx.rule = rule
        ctx.credit_config = credit_config
        ctx.plasticity = plasticity
        ctx.plasticity_config = plasticity_config
        ctx.in_features = in_features
        ctx.out_features = out_features
        ctx.feedback = feedback
        ctx.free_output = free_output

        # Native forward: x @ W^T + b
        out = x @ weight.T
        if bias is not None:
            out += bias
        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor | None, ...]:
        x, weight, bias = ctx.saved_tensors
        if bias.numel() == 0:
            bias = None

        # Compute pseudo-gradients per rule
        grad_x, grad_weight, grad_bias = compute_pseudo_gradients(
            rule=ctx.rule,
            x=x,
            weight=weight,
            bias=bias,
            grad_output=grad_output,
            config=ctx.credit_config,
            feedback=ctx.feedback,
            free_output=ctx.free_output,
        )

        # For HEBBIAN, grad_x is None -> no upstream gradient
        # Return tuple: (grad_x, grad_weight, grad_bias, *None for config args)
        # The config args don't need gradients
        return (
            grad_x,
            grad_weight,
            grad_bias,
            None,  # rule
            None,  # credit_config
            None,  # plasticity
            None,  # plasticity_config
            None,  # in_features
            None,  # out_features
            None,  # feedback
            None,  # free_output
        )


class ComputroniumLinear(nn.Linear):
    """Drop-in replacement for `nn.Linear` with bio-plausible learning rules.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If set to False, the layer will not learn an additive bias.
        rule: Credit assignment rule ("backprop", "fa", "hebbian", "eqprop").
        plasticity: Plasticity mechanism ("null", "fast_weights").
        credit_config: Configuration for the credit rule.
        plasticity_config: Configuration for the plasticity mechanism.
        device: Device for parameters and buffers.
        dtype: Data type for parameters.

    Example:
        >>> # Drop-in replacement (exact native behavior)
        >>> layer = ComputroniumLinear(784, 10, rule="backprop", plasticity="null")
        >>> x = torch.randn(32, 784)
        >>> y = layer(x)
        >>> loss = F.cross_entropy(y, target)
        >>> loss.backward()  # Exact native gradients

        >>> # Bio-plausible: Feedback Alignment
        >>> layer = ComputroniumLinear(784, 10, rule="fa", plasticity="null")
        >>> y = layer(x)
        >>> loss = F.cross_entropy(y, target)
        >>> loss.backward()  # FA pseudo-gradients

        >>> # With fast-weight plasticity
        >>> layer = ComputroniumLinear(
        ...     784, 10, rule="backprop", plasticity="fast_weights"
        ... )
        >>> y = layer(x)  # Updates internal ψ, modulates output
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        rule: str | CreditRule = CreditRule.BACKPROP,
        plasticity: str | PlasticityType = PlasticityType.NULL,
        credit_config: CreditRuleConfig | None = None,
        plasticity_config: PlasticityConfig | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        # Store configs before calling super().__init__ (which calls reset_parameters)
        self._credit_rule = CreditRule(rule) if isinstance(rule, str) else rule
        self._plasticity_type = (
            PlasticityType(plasticity) if isinstance(plasticity, str) else plasticity
        )
        self._credit_config = credit_config or CreditRuleConfig(rule=self._credit_rule)
        self._plasticity_config = plasticity_config or PlasticityConfig(
            plasticity_type=self._plasticity_type
        )

        # Pre-generate feedback matrix for FA (fixed, no grad) - before super()
        # to avoid reset_parameters race
        self._feedback: Tensor | None = None
        if self._credit_rule == CreditRule.FA:
            self._feedback = _generate_feedback_matrix(
                in_features,
                out_features,
                self._credit_config.feedback_scale,
                device or torch.device("cpu"),
            )

        # Initialize parent nn.Linear (sets up weight, bias, in_features, out_features)
        super().__init__(in_features, out_features, bias, device, dtype)

        # Register feedback as buffer after super() init
        if self._feedback is not None:
            self.register_buffer("_feedback_buffer", self._feedback, persistent=False)

        # Plasticity mechanism
        self._plasticity = create_plasticity(
            self._plasticity_type,
            in_features,
            out_features,
            self._plasticity_config,
            device,
        )

        # Internal state
        self._free_output: Tensor | None = None
        self._psi: dict[str, Tensor] | None = None

    def reset_parameters(self) -> None:
        """Reinitialize parameters (weight, bias) using nn.Linear's default."""
        super().reset_parameters()
        # Regenerate feedback matrix if FA
        if self._credit_rule == CreditRule.FA and self._feedback is not None:
            self._feedback = _generate_feedback_matrix(
                self.in_features,
                self.out_features,
                self._credit_config.feedback_scale,
                self.weight.device,
            )
            if hasattr(self, "_feedback_buffer"):
                self._feedback_buffer = self._feedback
            else:
                self.register_buffer(
                    "_feedback_buffer", self._feedback, persistent=False
                )

    def forward(self, x: Tensor) -> Tensor:
        # Store free output for eqprop contrastive learning
        if self._credit_rule == CreditRule.EQPROP:
            # Compute free output (native linear transform, no grad needed)
            with torch.no_grad():
                self._free_output = x @ self.weight.T
                if self.bias is not None:
                    self._free_output += self.bias

        # Native forward via custom autograd Function
        out = _ComputroniumLinearFn.apply(
            x,
            self.weight,
            self.bias,
            self._credit_rule,
            self._credit_config,
            self._plasticity_type,
            self._plasticity_config,
            self.in_features,
            self.out_features,
            self._feedback,
            self._free_output,
        )

        # Apply plasticity modulation (in forward, detached from autograd)
        if self._plasticity_type == PlasticityType.FAST_WEIGHTS:
            self._psi = self._plasticity.step(self._psi, x, out.detach())
            out = self._plasticity.modulate_output(out, self._psi)

        return out

    def extra_repr(self) -> str:
        base = super().extra_repr()
        return (
            f"{base}, rule={self._credit_rule.value}, "
            f"plasticity={self._plasticity_type.value}"
        )

    @property
    def plasticity(self) -> NullPlasticity | FastWeightPlasticity:
        """Access the plasticity mechanism for inspection/control."""
        return self._plasticity

    @property
    def psi(self) -> dict[str, Tensor] | None:
        """Current plastic state ψ."""
        return self._psi

    def reset_psi(self, batch_size: int = 1) -> None:
        """Reset plastic state ψ (e.g., at episode boundaries)."""
        self._psi = self._plasticity.initial_psi(batch_size, self.weight.device)

    def get_fast_weight_gradient_contribution(self) -> Tensor | None:
        """Get the fast-weight consolidation contribution to weight gradient.

        Returns:
            Tensor of shape (out_features, in_features) or None.
        """
        if isinstance(self._plasticity, FastWeightPlasticity):
            return self._plasticity.get_gradient_contribution()
        return None

    def to(self, *args, **kwargs) -> ComputroniumLinear:
        """Move to device, handling plasticity internal tensors."""
        moved_self = super().to(*args, **kwargs)
        device = args[0] if args else kwargs.get("device")
        if device is not None:
            self._plasticity.to(device)
            # _feedback_buffer is moved by super().to() since it's a registered buffer
            # Update local reference
            self._feedback = getattr(self, "_feedback_buffer", None)
        return moved_self

    def cuda(self, device: torch.device | int | None = None) -> ComputroniumLinear:
        """Move to CUDA, ensuring feedback buffer reference is updated."""
        return self.to(device or torch.device("cuda"))

    def cpu(self) -> ComputroniumLinear:
        """Move to CPU, ensuring feedback buffer reference is updated."""
        return self.to(torch.device("cpu"))


def replace_linear_with_computronium(
    module: nn.Module,
    rule: str | CreditRule = CreditRule.BACKPROP,
    plasticity: str | PlasticityType = PlasticityType.NULL,
    **kwargs,
) -> nn.Module:
    """Recursively replace all nn.Linear layers in a module with ComputroniumLinear.

    Args:
        module: The module to convert.
        rule: Default credit rule for new layers.
        plasticity: Default plasticity for new layers.
        **kwargs: Additional args passed to ComputroniumLinear constructor.

    Returns:
        The same module instance with Linear layers replaced in-place.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and not isinstance(child, ComputroniumLinear):
            new_layer = ComputroniumLinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                rule=rule,
                plasticity=plasticity,
                device=child.weight.device,
                dtype=child.weight.dtype,
                **kwargs,
            )
            # Copy weight and bias
            new_layer.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                new_layer.bias.data.copy_(child.bias.data)
            setattr(module, name, new_layer)
        else:
            replace_linear_with_computronium(child, rule, plasticity, **kwargs)
    return module
