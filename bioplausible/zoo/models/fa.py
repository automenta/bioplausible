"""
Combined Feedback Alignment Models
===================================

Aggregates all FA-family models into a single module for the model zoo.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

from bioplausible.config.unified import (
    ModelConfig,
    _build_model_config,
    resolve_hidden_dims,
)
from bioplausible.core.model import BioModel
from bioplausible.core.model_status import status_tag
from bioplausible.core.registry import register_model
from bioplausible.core.training_mixin import supervised_step
from bioplausible.core.utils.optimizer import OptimizerConfig, create_optimizer

from ..nebc_base import NEBCBase
from .base import EqPropModel

# ---------------------------------------------------------------------------
# Shared FA helpers
# ---------------------------------------------------------------------------


__all__ = [
    "AdaptiveFeedbackAlignment",
    "ContrastiveFeedbackAlignment",
    "DeepDFAEqProp",
    "DirectFeedbackAlignmentEqProp",
    "EnergyGuidedFA",
    "EnergyMinimizingFA",
    "EquilibriumAlignment",
    "FeedbackAlignmentEqProp",
    "FeedbackAlignmentLayer",
    "LayerwiseEquilibriumFA",
    "StandardFA",
    "StochasticFA",
]


def _fa_apply_activation_derivative(
    grad_h: torch.Tensor,
    h_curr: torch.Tensor,
    activation: nn.Module,
) -> torch.Tensor:
    """Apply the activation function derivative for FA backward.

    Handles SiLU (swish), ReLU, Tanh, and falls back to ReLU derivative.
    """
    if isinstance(activation, nn.SiLU):
        sig = torch.sigmoid(h_curr)
        return grad_h * sig * (1 + h_curr * (1 - sig))
    if isinstance(activation, nn.ReLU):
        return grad_h * (h_curr > 0).float()
    if isinstance(activation, nn.Tanh):
        return grad_h * (1 - h_curr**2)
    return grad_h * (h_curr > 0).float()


def _fa_forward(
    model: nn.Module,
    x: torch.Tensor,
) -> list[torch.Tensor]:
    """Run a standard feedforward pass and return per-layer activations.

    Returns ``[x, h1, h2, ..., output]`` where hidden layers have the
    model's activation applied and the output layer does not.
    """
    activations: list[torch.Tensor] = [x]
    h = x
    if h.dim() > 2:
        h = h.view(h.size(0), -1)
    for i, layer in enumerate(model.layers):  # type: ignore[attr-defined]
        h = layer(h)
        if i < len(model.layers) - 1:  # type: ignore[attr-defined]
            h = model.activation(h)  # type: ignore[attr-defined]
        activations.append(h)
    activations[0] = activations[0].view(activations[0].size(0), -1)
    return activations


def _fa_backward_loop(
    activations: list[torch.Tensor],
    error: torch.Tensor,
    feedback_weights: list[torch.Tensor] | nn.ParameterList,
    activation: nn.Module,
    num_layers: int,
    batch_size: int,
    *,
    dropout_prob: float = 0.0,
) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
    """Compute FA weight and bias gradients via manual backward loop.

    Args:
        activations: Per-layer activations from ``_fa_forward``.
        error: Output error signal ``(output - target)``.
        feedback_weights: Per-layer feedback weight matrices.
        activation: Activation module (used for derivative computation).
        num_layers: Number of weight layers (``len(model.layers)``).
        batch_size: Batch size (``x.size(0)``).
        dropout_prob: If > 0, apply dropout to feedback weights for
            stochastic feedback alignment.

    Returns:
        ``(weight_grads, bias_grads)`` where each list has ``num_layers``
        entries (``weight_grads[i]`` and ``bias_grads[i]`` correspond to
        ``layers[i]``).  ``bias_grads[i]`` is ``None`` when the layer has
        no bias.
    """
    weight_grads: list[torch.Tensor] = [None] * num_layers  # type: ignore[assignment]
    bias_grads: list[torch.Tensor | None] = [None] * num_layers

    propagated_error = error
    for i in reversed(range(num_layers)):
        h_prev = activations[i]

        if i < num_layers - 1:
            B = feedback_weights[i + 1]

            if dropout_prob > 0.0:
                B_device = B.to(propagated_error.device) if hasattr(B, "to") else B  # type: ignore[attr-defined]
                mask = (torch.rand_like(B_device) > dropout_prob).float()
                B_eff = B_device * mask * (1.0 / (1.0 - dropout_prob))
                grad_h = torch.mm(propagated_error, B_eff)
            else:
                grad_h = torch.mm(
                    propagated_error,
                    B.to(propagated_error.device) if hasattr(B, "to") else B,
                )  # type: ignore[attr-defined]

            h_curr = activations[i + 1]
            grad_h = _fa_apply_activation_derivative(grad_h, h_curr, activation)
        else:
            grad_h = propagated_error

        weight_grads[i] = torch.mm(grad_h.T, h_prev) / batch_size
        bias_grads[i] = grad_h.mean(0)
        propagated_error = grad_h

    return weight_grads, bias_grads


def _fa_train_step_body(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, *, dropout_prob: float = 0.0
) -> tuple[
    list[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    list[torch.Tensor],
    list[torch.Tensor | None],
]:
    """Forward, loss, error, and backward loop shared by FA classes.

    Returns ``(activations, output, loss, wgrads, bgrads)``.
    """
    activations = _fa_forward(model, x)
    output = activations[-1]
    loss = model.criterion(output, y)  # type: ignore[attr-defined]
    error = output - F.one_hot(y, model.config.output_dim).float()  # type: ignore[attr-defined]
    wgrads, bgrads = _fa_backward_loop(
        activations,
        error,
        model.feedback_weights,  # type: ignore[attr-defined]
        model.activation,  # type: ignore[attr-defined]
        len(model.layers),
        x.size(0),
        dropout_prob=dropout_prob,
    )
    return activations, output, loss, wgrads, bgrads


def _apply_fa_grads_to_optim(
    layers: nn.ModuleList,
    wgrads: list[torch.Tensor],
    bgrads: list[torch.Tensor | None],
    *,
    feedback_evolution: dict[str, list[nn.Module]] | None = None,
) -> None:
    """Set ``.grad`` buffers on layers from FA backward-loop output.

    Used by classes that step an optimizer after setting gradients.

    Parameters
    ----------
    feedback_evolution:
        If provided, expects keys ``"feedback"`` and ``"forward"`` for
        evolution toward the forward weight transpose.
    """
    for i, layer in enumerate(layers):
        if wgrads[i] is not None:
            if layer.weight.grad is None:
                layer.weight.grad = wgrads[i]
            else:
                layer.weight.grad += wgrads[i]
        if layer.bias is not None and bgrads[i] is not None:
            if layer.bias.grad is None:
                layer.bias.grad = bgrads[i]
            else:
                layer.bias.grad += bgrads[i]

    if feedback_evolution is not None:
        fwd_list = feedback_evolution.get("forward")
        fb_list = feedback_evolution.get("feedback")
        if fwd_list is not None and fb_list is not None:
            for i in range(len(layers)):
                if i < len(fb_list) - 1 and i + 1 < len(fwd_list):
                    target_B = fwd_list[i + 1].weight.data
                    current_B = fb_list[i + 1].data
                    grad_B = -(target_B - current_B)
                    if fb_list[i + 1].grad is None:
                        fb_list[i + 1].grad = grad_B
                    else:
                        fb_list[i + 1].grad += grad_B


def _apply_fa_grads_inplace(
    layers: nn.ModuleList,
    wgrads: list[torch.Tensor],
    bgrads: list[torch.Tensor | None],
    lr: float,
) -> None:
    """Apply FA gradients as in-place SGD update (no optimizer)."""
    for i, layer in enumerate(layers):
        if wgrads[i] is not None:
            layer.weight.data -= lr * wgrads[i]
        if layer.bias is not None and bgrads[i] is not None:
            layer.bias.data -= lr * bgrads[i]


def _ensure_optimizer(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    """Return ``model.optimizer``, creating it on first call.

    Fixes the bug where ``StandardFA``, ``EnergyMinimizingFA``, and
    ``LayerwiseEquilibriumFA`` created a new ``torch.optim.Adam`` on every
    ``train_step`` call (losing momentum between steps).
    """
    if not hasattr(model, "optimizer") or model.optimizer is None:  # type: ignore[attr-defined]
        model.optimizer = create_optimizer(  # type: ignore[attr-defined]
            [p for p in model.parameters() if p.requires_grad],
            OptimizerConfig(name="adam", lr=lr),
        )
    return model.optimizer  # type: ignore[attr-defined]


# ============================================================================
# feedback_alignment.py - All FA variants
# ============================================================================


class FeedbackAlignmentLayer(nn.Module):
    """Linear layer with separate forward and feedback weights."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        feedback_mode: str = "random",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.feedback_mode = feedback_mode

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        if feedback_mode == "random":
            self.register_buffer(
                "feedback_weight", torch.randn(in_features, out_features)
            )
        elif feedback_mode == "evolving":
            self.feedback_weight = nn.Parameter(torch.randn(in_features, out_features))
        else:
            self.feedback_weight = None

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.weight, gain=0.8)
        if hasattr(self, "feedback_weight") and self.feedback_weight is not None:
            nn.init.xavier_uniform_(self.feedback_weight, gain=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

    def get_feedback_weight(self) -> torch.Tensor:
        if self.feedback_mode == "symmetric" or self.feedback_weight is None:
            return self.weight.t()
        return self.feedback_weight

    def get_alignment_angle(self) -> float:
        W_flat = self.weight.t().flatten()
        B_flat = self.get_feedback_weight().flatten()
        cos_sim = F.cosine_similarity(W_flat.unsqueeze(0), B_flat.unsqueeze(0))
        return cos_sim.item()


@register_model(
    "feedback_alignment",
    family="fa",
    tags=["fa", "feedback-alignment", status_tag("stable")],
)
class FeedbackAlignmentEqProp(BioModel):
    """
    Equilibrium Propagation with Feedback Alignment.
    Uses asymmetric weights: forward W and feedback B.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        alpha: float = 0.5,
        feedback_mode: str = "random",
        use_spectral_norm: bool = True,
        config: ModelConfig | None = None,
        **kwargs,
    ):
        super().__init__(
            config,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            use_spectral_norm=use_spectral_norm,
            **kwargs,
        )

        self.alpha = alpha
        self.feedback_mode = feedback_mode

        self.W_in = nn.Linear(input_dim, hidden_dim)
        if use_spectral_norm:
            self.W_in = spectral_norm(self.W_in)

        self.layers = nn.ModuleList([
            FeedbackAlignmentLayer(hidden_dim, hidden_dim, feedback_mode)
            for _ in range(num_layers)
        ])

        self.head = nn.Linear(hidden_dim, output_dim)

    def forward_step(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.W_in(x)

        for layer in self.layers:
            h = torch.lerp(h, torch.tanh(x_proj + layer(h)), self.alpha)

        return h

    def forward(self, x: torch.Tensor, steps: int = 30) -> torch.Tensor:
        batch_size = x.size(0)
        h = torch.zeros(
            batch_size,
            self.config.hidden_dims[0] if self.config.hidden_dims else 256,
            device=x.device,
        )

        for _ in range(steps):
            h = self.forward_step(h, x)

        return self.head(h)

    def get_alignment_angles(self) -> dict[str, float]:
        angles = {}
        for i, layer in enumerate(self.layers):
            angles[f"layer_{i}"] = layer.get_alignment_angle()
        return angles

    def get_mean_alignment(self) -> float:
        angles = self.get_alignment_angles()
        if not angles:
            return 0.0
        return sum(angles.values()) / len(angles)


@register_model(
    "adaptive_feedback_alignment",
    family="fa",
    tags=["fa", "adaptive-feedback-alignment", status_tag("experimental")],
)
class AdaptiveFeedbackAlignment(BioModel):
    """FA with slow adaptive feedback evolution."""

    def __init__(self, config: ModelConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

        if not hasattr(self, "layers") or not self.layers:
            self.layers = nn.ModuleList()
            hidden_dims = resolve_hidden_dims(self.config, self.hidden_dim)
            dims = [self.input_dim] + hidden_dims + [self.output_dim]

            n_layers = len(dims) - 1
            for i in range(n_layers):
                layer = nn.Linear(dims[i], dims[i + 1])
                role = "output" if i == n_layers - 1 else "hidden"
                layer = self.apply_spectral_norm(layer, layer_role=role)
                self.layers.append(layer)

            self.to(kwargs.get("device", "cpu"))

        self.feedback_weights = nn.ParameterList()
        if config is None:
            config = self.config

        hidden_dims = resolve_hidden_dims(config, self.hidden_dim)
        dims = [config.input_dim] + hidden_dims + [config.output_dim]

        for i in range(len(dims) - 1):
            B = torch.randn(dims[i + 1], dims[i]) * 0.1
            self.feedback_weights.append(nn.Parameter(B, requires_grad=True))

        self.criterion = nn.CrossEntropyLoss()

        self.w_optimizer = create_optimizer(
            self.layers, OptimizerConfig(name="adam", lr=self.config.learning_rate)
        )
        self.b_optimizer = create_optimizer(
            self.feedback_weights,
            OptimizerConfig(name="adam", lr=self.config.learning_rate * 0.001),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        self.w_optimizer.zero_grad()
        self.b_optimizer.zero_grad()

        _, output, loss, wgrads, bgrads = _fa_train_step_body(self, x, y)

        with torch.no_grad():
            _apply_fa_grads_to_optim(
                self.layers,
                wgrads,
                bgrads,
                feedback_evolution={
                    "forward": list(self.layers),
                    "feedback": list(self.feedback_weights),
                },
            )

        self.w_optimizer.step()
        self.b_optimizer.step()

        return {
            "loss": loss.item(),
            "accuracy": (output.argmax(1) == y).float().mean().item(),
        }

    @classmethod
    def build(
        cls,
        spec,
        input_dim,
        output_dim,
        hidden_dim,
        num_layers,
        device,
        task_type,
        **kwargs,
    ):
        return cls(
            config=_build_model_config(
                spec, input_dim, output_dim, hidden_dim, num_layers, kwargs
            )
        ).to(device)


@register_model(
    "stochastic_fa",
    family="fa",
    tags=["fa", "stochastic", status_tag("experimental")],
)
class StochasticFA(BioModel):
    """FA with dropout on feedback signals."""

    def __init__(self, config: ModelConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

        if not hasattr(self, "layers") or not self.layers:
            self.layers = nn.ModuleList()
            hidden_dims = resolve_hidden_dims(self.config, self.hidden_dim)
            dims = [self.input_dim] + hidden_dims + [self.output_dim]

            n_layers = len(dims) - 1
            for i in range(n_layers):
                layer = nn.Linear(dims[i], dims[i + 1])
                role = "output" if i == n_layers - 1 else "hidden"
                layer = self.apply_spectral_norm(layer, layer_role=role)
                self.layers.append(layer)

            self.to(kwargs.get("device", "cpu"))

        self.feedback_weights = []
        dims = (
            [self.input_dim]
            + (
                self.config.hidden_dims
                if self.config.hidden_dims
                else [self.hidden_dim]
            )
            + [self.output_dim]
        )
        for i in range(len(dims) - 1):
            B = torch.randn(dims[i + 1], dims[i]) * 0.1
            self.feedback_weights.append(B)

        self.criterion = nn.CrossEntropyLoss()
        self.drop_prob = 0.5

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        self.zero_grad()

        _, output, loss, wgrads, bgrads = _fa_train_step_body(
            self,
            x,
            y,
            dropout_prob=self.drop_prob,
        )

        _apply_fa_grads_inplace(
            self.layers,
            wgrads,
            bgrads,
            self.config.learning_rate,
        )

        return {
            "loss": loss.item(),
            "accuracy": (output.argmax(1) == y).float().mean().item(),
        }

    @classmethod
    def build(
        cls,
        spec,
        input_dim,
        output_dim,
        hidden_dim,
        num_layers,
        device,
        task_type,
        **kwargs,
    ):
        return cls(
            config=_build_model_config(
                spec, input_dim, output_dim, hidden_dim, num_layers, kwargs
            )
        ).to(device)


@register_model(
    "contrastive_feedback_alignment",
    family="fa",
    tags=["fa", "contrastive", status_tag("experimental")],
)
class ContrastiveFeedbackAlignment(BioModel):
    """Contrastive FA."""

    def __init__(self, config: ModelConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

        if not hasattr(self, "layers") or not self.layers:
            self.layers = nn.ModuleList()
            hidden_dims = resolve_hidden_dims(self.config, self.hidden_dim)
            dims = [self.input_dim] + hidden_dims + [self.output_dim]

            n_layers = len(dims) - 1
            for i in range(n_layers):
                layer = nn.Linear(dims[i], dims[i + 1])
                role = "output" if i == n_layers - 1 else "hidden"
                layer = self.apply_spectral_norm(layer, layer_role=role)
                self.layers.append(layer)

            self.to(kwargs.get("device", "cpu"))

        self.criterion = nn.CrossEntropyLoss()

        self.feedback_weights = nn.ParameterList()
        hidden_dims = resolve_hidden_dims(self.config, self.hidden_dim)
        dims = [self.input_dim] + hidden_dims + [self.output_dim]
        for i in range(len(dims) - 1):
            B = torch.randn(dims[i + 1], dims[i]) * 0.1
            self.feedback_weights.append(nn.Parameter(B, requires_grad=False))

        self.optimizer = create_optimizer(
            self, OptimizerConfig(name="adam", lr=self.config.learning_rate)
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        return supervised_step(self, self.optimizer, x, y)


# ============================================================================
# dfa_eqprop.py - DirectFeedbackAlignmentEqProp & DeepDFAEqProp
# ============================================================================


@register_model(
    "direct_feedback_alignment_eqprop",
    family="fa",
    tags=["fa", "dfa", status_tag("experimental")],
)
class DirectFeedbackAlignmentEqProp(NEBCBase):
    """
    Direct Feedback Alignment with EqProp-style dynamics.
    """

    algorithm_name = "DFA"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        use_spectral_norm: bool = True,
        max_steps: int = 30,
        alpha: float = 0.5,
    ):
        self.alpha = alpha
        super().__init__(
            input_dim, hidden_dim, output_dim, num_layers, use_spectral_norm, max_steps
        )

    def _build_layers(self):
        self.W_in = nn.Linear(self.input_dim, self.hidden_dim)
        if self.use_spectral_norm:
            self.W_in = spectral_norm(self.W_in, n_power_iterations=5)

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            layer = nn.Linear(self.hidden_dim, self.hidden_dim)
            if self.use_spectral_norm:
                layer = spectral_norm(layer, n_power_iterations=5)
            self.layers.append(layer)

        self.head = nn.Linear(self.hidden_dim, self.output_dim)
        if self.use_spectral_norm:
            self.head = spectral_norm(self.head, n_power_iterations=5)

        self.feedback_projections = nn.ModuleList()
        for i in range(self.num_layers):
            B = nn.Linear(self.output_dim, self.hidden_dim, bias=False)
            nn.init.xavier_uniform_(B.weight, gain=0.1)
            B.weight.requires_grad = False
            self.feedback_projections.append(B)

    def forward(self, x: torch.Tensor, steps: int | None = None) -> torch.Tensor:
        steps = steps or self.max_steps
        batch_size = x.size(0)

        h = [
            torch.zeros(batch_size, self.hidden_dim, device=x.device)
            for _ in range(self.num_layers)
        ]

        x_proj = self.W_in(x)

        for _ in range(steps):
            h[0] = (1 - self.alpha) * h[0] + self.alpha * torch.tanh(
                x_proj + self.layers[0](h[0])
            )

            for i in range(1, self.num_layers):
                h[i] = (1 - self.alpha) * h[i] + self.alpha * torch.tanh(
                    h[i - 1] + self.layers[i](h[i])
                )

        return self.head(h[-1])

    def get_feedback_alignment_angles(self) -> dict[str, float]:
        angles = {}
        for i, (layer, B) in enumerate(zip(self.layers, self.feedback_projections)):
            if hasattr(layer, "weight"):
                W = layer.weight
            else:
                W = layer.parametrizations.weight.original

            W_flat = W.flatten()
            B_flat = B.weight.flatten()

            min_len = min(len(W_flat), len(B_flat))
            cos_sim = F.cosine_similarity(
                W_flat[:min_len].unsqueeze(0), B_flat[:min_len].unsqueeze(0)
            )
            angles[f"layer_{i}"] = cos_sim.item()

        return angles

    def get_stats(self) -> dict[str, float]:
        stats = super().get_stats()
        angles = self.get_feedback_alignment_angles()
        stats["mean_alignment"] = sum(angles.values()) / len(angles) if angles else 0.0
        return stats

    @classmethod
    def build(
        cls,
        spec,
        input_dim,
        output_dim,
        hidden_dim,
        num_layers,
        device,
        task_type,
        **kwargs,
    ):
        return cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            use_spectral_norm=kwargs.get("use_spectral_norm", True),
            max_steps=kwargs.get("max_steps", 30),
            alpha=kwargs.get("alpha", 0.5),
        ).to(device)


@register_model(
    "dfa_deep",
    family="fa",
    tags=["fa", "dfa", "deep", status_tag("experimental")],
)
class DeepDFAEqProp(DirectFeedbackAlignmentEqProp):
    """
    DFA variant optimized for extreme depth (1000+ layers).
    """

    algorithm_name = "DeepDFA"

    def _build_layers(self):
        super()._build_layers()

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)
        ])

    def forward(self, x: torch.Tensor, steps: int | None = None) -> torch.Tensor:
        steps = steps or self.max_steps
        batch_size = x.size(0)

        h = [
            torch.zeros(batch_size, self.hidden_dim, device=x.device)
            for _ in range(self.num_layers)
        ]

        x_proj = self.W_in(x)

        for _ in range(steps):
            h_new = torch.tanh(x_proj + self.layers[0](h[0]))
            h[0] = self.layer_norms[0]((1 - self.alpha) * h[0] + self.alpha * h_new)

            for i in range(1, self.num_layers):
                h_new = torch.tanh(h[i - 1] + self.layers[i](h[i]))
                h[i] = self.layer_norms[i]((1 - self.alpha) * h[i] + self.alpha * h_new)

        return self.head(h[-1])

    @classmethod
    def build(
        cls,
        spec,
        input_dim,
        output_dim,
        hidden_dim,
        num_layers,
        device,
        task_type,
        **kwargs,
    ):
        return cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            use_spectral_norm=kwargs.get("use_spectral_norm", True),
            max_steps=kwargs.get("max_steps", 30),
            alpha=kwargs.get("alpha", 0.5),
        ).to(device)


# ============================================================================
# simple_fa.py - StandardFA
# ============================================================================


@register_model(
    "standard_fa",
    family="fa",
    tags=["fa", "standard", status_tag("stable")],
)
class StandardFA(BioModel):
    """Feedback Alignment with random fixed backward weights."""

    def __init__(self, config: ModelConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

        self.feedback_weights = nn.ParameterList()
        hidden_dims = resolve_hidden_dims(self.config, self.hidden_dim)
        dims = [self.input_dim] + hidden_dims + [self.output_dim]

        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

            B = torch.randn(dims[i + 1], dims[i]) * 0.1
            p = nn.Parameter(B, requires_grad=False)
            self.feedback_weights.append(p)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = create_optimizer(
            [p for p in self.parameters() if p.requires_grad],
            OptimizerConfig(name="adam", lr=self.config.learning_rate),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> dict[str, float]:
        self.optimizer.zero_grad()

        _, output, loss, wgrads, bgrads = _fa_train_step_body(self, x, y)

        with torch.no_grad():
            _apply_fa_grads_to_optim(self.layers, wgrads, bgrads)

        self.optimizer.step()

        return {
            "loss": loss.item(),
            "accuracy": (output.argmax(1) == y).float().mean().item(),
        }


# ============================================================================
# eg_fa.py - EnergyGuidedFA
# ============================================================================


@register_model(
    "energy_guided_fa",
    family="fa",
    tags=["fa", "energy-guided", status_tag("experimental")],
)
class EnergyGuidedFA(BioModel):
    """Energy Guided FA."""

    def __init__(self, config: ModelConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

        if not hasattr(self, "layers") or not self.layers:
            self.layers = nn.ModuleList()
            hidden_dims = resolve_hidden_dims(self.config, self.hidden_dim)
            dims = [self.input_dim] + hidden_dims + [self.output_dim]

            n_layers = len(dims) - 1
            for i in range(n_layers):
                layer = nn.Linear(dims[i], dims[i + 1])
                role = "output" if i == n_layers - 1 else "hidden"
                layer = self.apply_spectral_norm(layer, layer_role=role)
                self.layers.append(layer)

            self.to(kwargs.get("device", "cpu"))

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        return supervised_step(
            self, _ensure_optimizer(self, self.config.learning_rate), x, y
        )

    @classmethod
    def build(
        cls,
        spec,
        input_dim,
        output_dim,
        hidden_dim,
        num_layers,
        device,
        task_type,
        **kwargs,
    ):
        return cls(
            config=_build_model_config(
                spec, input_dim, output_dim, hidden_dim, num_layers, kwargs
            )
        ).to(device)


# ============================================================================
# em_fa.py - EnergyMinimizingFA
# ============================================================================


@register_model(
    "energy_minimizing_fa",
    family="fa",
    tags=["fa", "energy-minimizing", status_tag("experimental")],
)
class EnergyMinimizingFA(BioModel):
    """EqProp dynamics + FA updates."""

    def __init__(self, config: ModelConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

        if not hasattr(self, "layers") or not self.layers:
            self.layers = nn.ModuleList()
            hidden_dims = resolve_hidden_dims(self.config, self.hidden_dim)
            dims = [self.input_dim] + hidden_dims + [self.output_dim]

            n_layers = len(dims) - 1
            for i in range(n_layers):
                layer = nn.Linear(dims[i], dims[i + 1])
                role = "output" if i == n_layers - 1 else "hidden"
                layer = self.apply_spectral_norm(layer, layer_role=role)
                self.layers.append(layer)

            self.to(kwargs.get("device", "cpu"))

        self.criterion = nn.CrossEntropyLoss()

        self.feedback_weights = nn.ParameterList()
        dims = (
            [self.input_dim]
            + (
                self.config.hidden_dims
                if self.config.hidden_dims
                else [self.hidden_dim]
            )
            + [self.output_dim]
        )
        for i in range(len(dims) - 1):
            B = torch.randn(dims[i + 1], dims[i]) * 0.1
            self.feedback_weights.append(nn.Parameter(B, requires_grad=False))

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        return supervised_step(
            self, _ensure_optimizer(self, self.config.learning_rate), x, y
        )

    @classmethod
    def build(
        cls,
        spec,
        input_dim,
        output_dim,
        hidden_dim,
        num_layers,
        device,
        task_type,
        **kwargs,
    ):
        return cls(
            config=_build_model_config(
                spec, input_dim, output_dim, hidden_dim, num_layers, kwargs
            )
        ).to(device)


# ============================================================================
# leq_fa.py - LayerwiseEquilibriumFA
# ============================================================================


@register_model(
    "layerwise_equilibrium_fa",
    family="fa",
    tags=["fa", "layerwise-equilibrium", status_tag("experimental")],
)
class LayerwiseEquilibriumFA(BioModel):
    """Layerwise Equilibrium FA."""

    def __init__(self, config: ModelConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

        if not hasattr(self, "layers") or not self.layers:
            self.layers = nn.ModuleList()
            hidden_dims = resolve_hidden_dims(self.config, self.hidden_dim)
            dims = [self.input_dim] + hidden_dims + [self.output_dim]

            n_layers = len(dims) - 1
            for i in range(n_layers):
                layer = nn.Linear(dims[i], dims[i + 1])
                role = "output" if i == n_layers - 1 else "hidden"
                layer = self.apply_spectral_norm(layer, layer_role=role)
                self.layers.append(layer)

            self.to(kwargs.get("device", "cpu"))

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        return supervised_step(
            self, _ensure_optimizer(self, self.config.learning_rate), x, y
        )

    @classmethod
    def build(
        cls,
        spec,
        input_dim,
        output_dim,
        hidden_dim,
        num_layers,
        device,
        task_type,
        **kwargs,
    ):
        return cls(
            config=_build_model_config(
                spec, input_dim, output_dim, hidden_dim, num_layers, kwargs
            )
        ).to(device)


# ============================================================================
# eq_align.py - EquilibriumAlignment
# ============================================================================


@register_model(
    "equilibrium_alignment",
    family="fa",
    tags=["fa", "equilibrium-alignment", status_tag("broken")],
)
class EquilibriumAlignment(EqPropModel):
    """
    Equilibrium Alignment (EqAlign) - Native Implementation.

    Combines Equilibrium Propagation's fixed-point dynamics with
    Feedback Alignment (FA) training signals.
    """

    algorithm_name = "EquilibriumAlignment"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        max_steps: int = 30,
        use_spectral_norm: bool = True,
        learning_rate: float = 0.001,
        **kwargs,
    ):
        self.learning_rate = learning_rate
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            max_steps=max_steps,
            use_spectral_norm=use_spectral_norm,
            **kwargs,
        )

    @classmethod
    def build(
        cls,
        spec,
        input_dim,
        output_dim,
        hidden_dim,
        num_layers,
        device,
        task_type,
        **kwargs,
    ):
        return cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            max_steps=30,
            use_spectral_norm=True,
            learning_rate=spec.default_lr,
        ).to(device)

    def _build_layers(self):
        self.W_in = nn.Linear(self.input_dim, self.hidden_dim)
        self.W_rec = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.W_out = nn.Linear(self.hidden_dim, self.output_dim)

        if self.use_spectral_norm:
            self.W_in = spectral_norm(self.W_in)
            self.W_rec = spectral_norm(self.W_rec)
            self.W_out = spectral_norm(self.W_out)

        self.B_out = nn.Parameter(
            torch.randn(self.output_dim, self.hidden_dim) * 0.1, requires_grad=False
        )

    def transition_modules(self) -> list[nn.Module]:
        return [self.W_in, self.W_rec, self.W_out]

    def _initialize_hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return torch.zeros(
            (batch_size, self.hidden_dim), device=x.device, dtype=x.dtype
        )

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_in(x)

    def forward_step(
        self, h: torch.Tensor, x_transformed: torch.Tensor
    ) -> torch.Tensor:
        return torch.tanh(x_transformed + self.W_rec(h))

    def _output_projection(self, h: torch.Tensor) -> torch.Tensor:
        return self.W_out(h)

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        with torch.no_grad():
            x_transformed = self._transform_input(x)
            h = self._initialize_hidden_state(x)
            for _ in range(self.max_steps):
                h = self.forward_step(h, x_transformed)
            h_star = h

            logits = self._output_projection(h_star)

            loss = nn.functional.cross_entropy(logits, y)

            acc = (logits.argmax(dim=1) == y).float().mean().item()

            if y.dim() == 1:
                target = nn.functional.one_hot(y, self.output_dim).float()
            else:
                target = y

            probs = torch.softmax(logits, dim=1)
            delta_out = probs - target

        delta_h = torch.mm(delta_out, self.B_out)

        delta_h = delta_h * (1 - h_star**2)

        batch_size = x.size(0)

        grad_W_out = torch.mm(delta_out.T, h_star) / batch_size
        grad_W_rec = torch.mm(delta_h.T, h_star) / batch_size
        grad_W_in = torch.mm(delta_h.T, x) / batch_size

        grad_b_out = delta_out.mean(0)
        grad_b_rec = delta_h.mean(0)

        def update_layer(layer, grad_w, grad_b=None):
            if hasattr(layer, "parametrizations"):
                weight_param = layer.parametrizations.weight.original
            else:
                weight_param = layer.weight

            weight_param.data -= self.learning_rate * grad_w

            if layer.bias is not None and grad_b is not None:
                layer.bias.data -= self.learning_rate * grad_b

        update_layer(self.W_out, grad_W_out, grad_b_out)
        update_layer(self.W_rec, grad_W_rec, grad_b_rec)
        update_layer(self.W_in, grad_W_in, grad_b_rec)

        return {"loss": loss.item(), "accuracy": acc}
