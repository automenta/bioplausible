"""Generic gradient computation strategies (REFACTOR.md §7).

Backpropagation, feedback-alignment, target-propagation, and Hebbian gradients.
MEP-specific EP strategies live in ``zoo.mep.optimizers.strategies.gradient``.
"""

from typing import Protocol, cast

import torch
import torch.nn.functional as F
from torch import nn

from .base import GradientStrategy

__all__ = ["BackpropGradient", "FAGradient", "HebbianGradient", "TargetPropGradient"]


class _ForwardNetLayer(Protocol):
    """A target-prop layer exposing its forward and learned inverse networks."""

    forward_net: nn.Module
    inverse_net: nn.Module


class _TargetPropModel(Protocol):
    """Structural interface for target-propagation models."""

    layers: list[_ForwardNetLayer]
    out_layer: nn.Module


class _TransitionModel(Protocol):
    """Structural interface for models exposing ordered transition modules."""

    hebbian_lr: float

    def transition_modules(self) -> list[nn.Module]: ...


class _HebbianLayer(Protocol):
    """Optional local-update hook on a transition module."""

    def hebbian_update(self, x: torch.Tensor, y: torch.Tensor) -> None: ...


class BackpropGradient(GradientStrategy):
    """Standard backpropagation via ``loss.backward()``."""

    def __init__(self, loss_fn: nn.Module | None = None):
        self.loss_fn = loss_fn

    def compute_gradients(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target: torch.Tensor | None,
        loss_fn: nn.Module | None = None,
        **kwargs: object,  # ruff: ignore[unused-method-argument]
    ) -> None:
        """Compute gradients via backpropagation."""
        fn = loss_fn or self.loss_fn
        if fn is None:
            raise ValueError("loss_fn must be provided to BackpropGradient")
        output = model(x)
        loss = fn(output, target)
        loss.backward()


class FAGradient(GradientStrategy):
    """Feedback alignment: errors routed through fixed random matrices.

    The loss error propagates backward through frozen random ``B_l`` matrices
    (one per linear layer) instead of the transposed forward weights,
    avoiding symmetric weight transport while keeping a global loss signal.
    """

    def __init__(self, loss_fn: nn.Module | None = None, feedback_seed: int = 0):
        self.loss_fn = loss_fn
        self.feedback_seed = feedback_seed
        self._feedback: dict[int, torch.Tensor] = {}

    def _feedback_for(self, module: nn.Linear) -> torch.Tensor:
        """Return (and lazily build) the fixed random feedback for a layer.

        The feedback `B_l` has shape ``(out, in)`` so a size-``out`` error
        maps back to the layer input dimension: ``delta @ B_l``.
        """
        weight = module.weight
        key = id(module)
        fb = self._feedback.get(key)
        if fb is None:
            g = torch.Generator(device=weight.device).manual_seed(self.feedback_seed)
            fb = torch.randn(
                weight.shape[0], weight.shape[1], device=weight.device, generator=g
            )
            self._feedback[key] = fb
        return fb

    @staticmethod
    def _linear_modules(model: nn.Module) -> list[nn.Linear]:
        return [m for m in model.modules() if isinstance(m, nn.Linear)]

    def compute_gradients(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target: torch.Tensor | None,
        loss_fn: nn.Module | None = None,
        **kwargs: object,  # ruff: ignore[unused-method-argument]
    ) -> None:
        """Compute feedback-alignment gradients for a sequential linear model.

        The model must be a feed-forward ``nn.Sequential`` of ``Linear`` and
        activation layers; each layer's input is recorded during the forward
        pass so ``grad_W_l = delta_l @ a_{l-1}^T``.
        """
        fn = loss_fn or self.loss_fn
        if fn is None:
            raise ValueError("loss_fn must be provided to FAGradient")
        if target is None:
            raise ValueError("target must be provided to FAGradient")

        layers = self._linear_modules(model)
        if not layers:
            raise ValueError("FAGradient requires at least one Linear layer")

        inputs: list[torch.Tensor] = []
        current = x
        with torch.enable_grad():
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    inputs.append(current)
                    current = module(current)

            loss = fn(current, target)
            error = torch.autograd.grad(loss, current, retain_graph=True)[0]

        # Backward through fixed random feedback matrices.
        delta = error.detach()
        for layer, layer_input in reversed(list(zip(layers, inputs))):
            weight = layer.weight
            weight.grad = delta.T @ layer_input.detach()
            if layer.bias is not None:
                layer.bias.grad = delta.sum(0)
            delta = delta @ self._feedback_for(layer)

        for name, param in model.named_parameters():
            if param.grad is None and param.requires_grad:
                param.grad = torch.zeros_like(param)


class TargetPropGradient(GradientStrategy):
    """Target Propagation: errors routed through learned approximate inverses.

    Computes layer-wise targets by propagating output targets backward
    through learned inverse mappings, then trains each forward network
    to match its target via local MSE loss.

    Assumes the model has a structure compatible with DifferenceTargetProp:
    - ``model.layers``: list of modules with ``forward_net`` and ``inverse_net``
    - ``model.out_layer``: output linear layer
    - ``model.criterion``: loss function (e.g., CrossEntropyLoss)
    - ``model.target_lr``: target learning rate
    """

    requires_energy = True  # needs x/target inside step(); forwarded via energy path

    def __init__(
        self,
        loss_fn: nn.Module | None = None,
        target_lr: float = 0.1,
    ):
        self.loss_fn = loss_fn
        self.target_lr = target_lr

    @staticmethod
    def _validate_model(model: _TargetPropModel) -> None:
        if not hasattr(model, "layers") or not hasattr(model, "out_layer"):
            raise ValueError(
                "TargetPropGradient requires model with 'layers' "
                "(with forward_net/inverse_net) and 'out_layer'"
            )
        for i, layer in enumerate(model.layers):
            if not hasattr(layer, "forward_net"):
                raise ValueError(f"Layer {i} must have a 'forward_net' attribute")
            if i > 0 and not hasattr(layer, "inverse_net"):
                raise ValueError(
                    f"Layer {i} must have 'inverse_net' for target propagation"
                )

    @staticmethod
    def _forward_pass(
        model: _TargetPropModel, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        hs = [x]
        h = x
        for layer in model.layers:
            h = layer.forward_net(h)
            hs.append(h)
        out = model.out_layer(h)
        return hs, out

    def _compute_output_target(
        self,
        model: _TargetPropModel,
        h: torch.Tensor,
        target: torch.Tensor,
        fn: nn.Module,
    ) -> torch.Tensor:
        t = h.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            out_t = model.out_layer(t)
            loss_t = fn(out_t, target)
            grad_t = torch.autograd.grad(loss_t, t)[0]
        with torch.no_grad():
            return h - self.target_lr * grad_t

    @staticmethod
    def _propagate_targets(
        model: _TargetPropModel,
        hs: list[torch.Tensor],
        initial_target: torch.Tensor,
    ) -> list[torch.Tensor]:
        targets = [initial_target]
        for i in reversed(range(len(model.layers))):
            if i > 0:
                h_prev = hs[i]
                t_curr = targets[-1]
                layer = model.layers[i]
                with torch.no_grad():
                    t_prev = (
                        h_prev
                        - layer.inverse_net(hs[i + 1])
                        + layer.inverse_net(t_curr)
                    )
                    targets.append(t_prev)
        return targets

    @staticmethod
    def _train_forward_nets(
        model: _TargetPropModel,
        hs: list[torch.Tensor],
        targets: list[torch.Tensor],
    ) -> None:
        # Layer i's forward net predicts hs[i+1], whose target sits at
        # targets[-(i+1)] (targets[0] = t_target for the last hidden layer).
        for i in reversed(range(len(model.layers))):
            layer = model.layers[i]
            t_curr = targets[-(i + 1)]
            h_prev_det = hs[i].detach()
            pred_h = layer.forward_net(h_prev_det)
            loss_f = F.mse_loss(pred_h, t_curr)
            loss_f.backward()

    @staticmethod
    def _train_inverse_nets(
        model: _TargetPropModel,
        hs: list[torch.Tensor],
    ) -> None:
        for i in reversed(range(1, len(model.layers))):
            layer = model.layers[i]
            layer.inverse_net.zero_grad()
            h_prev = hs[i]
            # Use detached forward pass output for cycle consistency
            with torch.no_grad():
                pred_h = layer.forward_net(h_prev.detach())
            inv_out = layer.inverse_net(pred_h.detach())
            loss_g = F.mse_loss(inv_out, h_prev.detach())
            loss_g.backward()

    @staticmethod
    def _zero_missing_grads(model: nn.Module) -> None:
        for param in model.parameters():
            if param.grad is None and param.requires_grad:
                param.grad = torch.zeros_like(param)

    def compute_gradients(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target: torch.Tensor | None,
        loss_fn: nn.Module | None = None,
        **kwargs: object,  # ruff: ignore[unused-method-argument]
    ) -> None:
        """Compute target propagation gradients."""
        fn = loss_fn or self.loss_fn
        if fn is None:
            raise ValueError("loss_fn must be provided to TargetPropGradient")
        if target is None:
            raise ValueError("target must be provided to TargetPropGradient")
        tpm = cast("_TargetPropModel", model)

        self._validate_model(tpm)

        hs, out = self._forward_pass(tpm, x)
        loss = fn(out, target)

        # Update output layer first (via standard backprop)
        out_layer_params = list(tpm.out_layer.parameters())
        for p in out_layer_params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        loss.backward()

        # Compute target for output layer
        t_target = self._compute_output_target(tpm, hs[-1], target, fn)

        # Backward target propagation
        targets = self._propagate_targets(tpm, hs, t_target)

        # Train forward nets to hit targets
        self._train_forward_nets(tpm, hs, targets)

        # Train inverse nets for cycle consistency
        self._train_inverse_nets(tpm, hs)

        # Zero gradients for untouched parameters
        self._zero_missing_grads(model)


class HebbianGradient(GradientStrategy):
    """Local Hebbian learning with Oja's normalization rule.

    Applies local Hebbian updates per layer: Delta W = eta * (y @ x.T - y^2 @ W)
    The output layer receives a supervised delta update.

    Assumes the model has:
    - ``model.transition_modules()``: list of modules in forward order
    - Layers with optional ``hebbian_update(x, y)`` method
    - ``model.hebbian_lr``: Hebbian learning rate
    - ``model.use_oja``: whether to apply Oja's normalization
    """

    requires_energy = True  # needs x/target inside step(); forwarded via energy path

    def __init__(
        self,
        hebbian_lr: float = 0.01,
        use_oja: bool = True,
    ):
        self.hebbian_lr = hebbian_lr
        self.use_oja = use_oja

    @staticmethod
    def _validate_model(model: _TransitionModel) -> list[nn.Module]:
        if not hasattr(model, "transition_modules"):
            raise AttributeError(
                "HebbianGradient requires model with 'transition_modules()' method"
            )
        return model.transition_modules()

    def _get_hebbian_lr(self, model: _TransitionModel) -> float:
        return getattr(model, "hebbian_lr", self.hebbian_lr)

    @staticmethod
    def _compute_head_update(
        head: nn.Module,
        error: torch.Tensor,
        prev_activation: torch.Tensor,
        hebbian_lr: float,
        batch_size: int,
    ) -> None:
        if hasattr(head, "parametrizations"):
            head_w = dict(head.named_parameters())["parametrizations.weight.original"]
        elif hasattr(head, "weight"):
            head_w = head.weight
        else:
            head_w = None
        if head_w is not None:
            cast("torch.Tensor", head_w).addmm_(
                error.T,
                prev_activation,
                alpha=hebbian_lr / batch_size,
            )

    @staticmethod
    def _zero_missing_grads(model: nn.Module) -> None:
        for param in model.parameters():
            if param.grad is None and param.requires_grad:
                param.grad = torch.zeros_like(param)

    def compute_gradients(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target: torch.Tensor | None,
        loss_fn: nn.Module | None = None,  # ruff: ignore[unused-method-argument]
        **kwargs: object,  # ruff: ignore[unused-method-argument]
    ) -> None:
        """Compute local Hebbian gradients."""
        if target is None:
            raise ValueError("target must be provided to HebbianGradient")

        ttm = cast("_TransitionModel", model)
        transitions = self._validate_model(ttm)
        hebbian_lr = self._get_hebbian_lr(ttm)

        with torch.no_grad():
            h = x
            activations = [h]
            for layer in transitions:
                h = layer(h)
                if hasattr(layer, "hebbian_update"):
                    cast("_HebbianLayer", layer).hebbian_update(activations[-1], h)
                activations.append(h)

            # Supervised update for the output head (the last transition module).
            logits = h
            y_onehot = torch.zeros_like(logits)
            y_onehot.scatter_(1, target.unsqueeze(1), 1.0)
            error = y_onehot - torch.softmax(logits, dim=1)
            head = transitions[-1]

            self._compute_head_update(
                head, error, activations[-2], hebbian_lr, x.shape[0]
            )

        # No autograd graph, no gradients to accumulate for backprop
        # But StrategyOptimizer expects gradients to exist. Set zero grads
        # for params not touched by local updates so optimizer doesn't error.
        self._zero_missing_grads(model)
