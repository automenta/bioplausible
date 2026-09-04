"""Substrate-operator-native settle kernel (EqProp family).

P4 kernel port: the settle family executes its relaxation loop through the
Substrate operator API (forward operator, weight quantization, state noise,
weight update operator) instead of raw nn.Linear / legacy NumPy kernels.

Only the true forward passes (bottom-up and output layer) route through the
substrate's forward operator. The top-down and recurrent passes are
mathematical transpose operations, not physical substrate processes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor

if TYPE_CHECKING:
    from computronium.ontology.geometry import Geometry
    from computronium.ontology.substrate import Substrate


@dataclass(frozen=True, slots=True)
class LayeredParams:
    """Extracted linear layer parameters from a geometry.

    Used by the settle kernel to route forward passes through the
    substrate's forward operator. ``transitions`` is the interleaved
    ``(weight, bias, activation chain)`` schedule (R3.1) — populated only
    when the geometry exposes its raw module stack; assembled block views
    (tile meshes) leave it empty.
    """

    weights: tuple[Tensor, ...]
    biases: tuple[Tensor | None, ...]
    activations: tuple[torch.nn.Module, ...]
    recurrent_weight: Tensor | None
    transitions: tuple[
        tuple[Tensor, Tensor | None, tuple[torch.nn.Module, ...]], ...
    ] = ()


def group_transitions(
    modules: list[torch.nn.Module],
) -> tuple[tuple[Tensor, Tensor | None, tuple[torch.nn.Module, ...]], ...]:
    """Group a module stack into ``(weight, bias, activations)`` transitions.

    Each ``nn.Linear`` opens a transition; consecutive non-Linear modules
    close it as the transition's activation chain. Deep-linear stacks yield
    empty activation chains — error injection still applies (R3.1, moved
    from ePC's private ``_transitions``).
    """
    transitions: list[tuple[Tensor, Tensor | None, tuple[torch.nn.Module, ...]]] = []
    current: tuple[Tensor, Tensor | None, tuple[torch.nn.Module, ...]] | None = None
    for module in modules:
        if isinstance(module, torch.nn.Linear):
            if current is not None:
                transitions.append(current)
            current = (module.weight, module.bias, ())
        elif current is not None:
            current = (current[0], current[1], (*current[2], module))
    if current is not None:
        transitions.append(current)
    return tuple(transitions)


def extract_layered_params(geometry: Geometry) -> LayeredParams | None:
    """Extract linear layer stack from a layered geometry.

    Returns None if the geometry is not layer-structured. TileGeometry
    answers through its assembled block view (R11.1.4).
    """
    layers = getattr(geometry, "_layers", None)
    if isinstance(layers, torch.nn.ModuleList):
        modules = list(layers)
        linears = [m for m in modules if isinstance(m, torch.nn.Linear)]
        activations = [m for m in modules if not isinstance(m, torch.nn.Linear)]
        recurrent = getattr(geometry, "_recurrent_weight", None)

        return LayeredParams(
            weights=tuple(layer.weight for layer in linears),
            biases=tuple(layer.bias for layer in linears),
            activations=tuple(activations),
            recurrent_weight=recurrent if isinstance(recurrent, Tensor) else None,
            transitions=group_transitions(modules),
        )
    assembler = getattr(geometry, "layered_params", None)
    if callable(assembler):
        return cast("LayeredParams", assembler())
    return None


def _one_hot(target: Tensor, like: Tensor) -> Tensor:
    """Convert class indices to one-hot matching 'like' tensor."""
    if target.dim() == 1:
        out = torch.zeros_like(like)
        out.scatter_(1, target.unsqueeze(1), 1.0)
        return out
    return target


class SubstrateSettleKernel:
    """EqProp settle kernel executing through the Substrate operator API.

    True forward passes (bottom-up, output) route through
    ``Substrate.get_forward_operator()``; weight synchronization quantizes
    through ``Substrate.quantize_weights()``; the contrastive consolidation
    applies ``Substrate.get_weight_update_operator()``.

    Top-down and recurrent passes use raw matmul (mathematical transposes,
    not physical substrate processes).
    """

    def __init__(
        self,
        substrate: Substrate,
        params: LayeredParams,
        step_size: float,
        momentum: float = 0.0,
    ) -> None:
        self.substrate = substrate
        self._op = substrate.get_forward_operator()
        self._update_op = substrate.get_weight_update_operator()
        self.params = params
        self.step_size = step_size
        self.momentum = momentum

    def _bottom_up(self, x: Tensor, idx: int) -> Tensor:
        """Bottom-up pass: x @ W.T + b (substrate forward operator + bias)."""
        out = self._op(x, self.params.weights[idx])
        b = self.params.biases[idx]
        return out if b is None else out + b

    def _top_down(self, x: Tensor, w: Tensor) -> Tensor:
        """Top-down pass: x @ W (raw matmul, mathematical transpose)."""
        return x @ w

    def _recurrent(self, x: Tensor, w: Tensor) -> Tensor:
        """Recurrent pass: x @ W.T (raw matmul)."""
        return x @ w.T

    def step(
        self,
        all_acts: list[Tensor],
        beta: float,
        target: Tensor | None,
        velocity: list[Tensor] | None,
    ) -> tuple[list[Tensor], list[Tensor] | None]:
        """One relaxation step; mirrors legacy `_settle_step` exactly."""
        p = self.params
        num_hidden = len(all_acts) - 2
        new_acts = [all_acts[0]]
        new_velocity: list[Tensor] | None = (
            [] if self.momentum > 0 and velocity is not None else None
        )

        for i in range(num_hidden):
            pre = self._bottom_up(all_acts[i], i)

            if p.recurrent_weight is not None and i == num_hidden - 1:
                pre += self._recurrent(all_acts[i + 1], p.recurrent_weight)

            top_down = self._top_down(all_acts[i + 2], p.weights[i + 1])

            total = pre + top_down

            if new_velocity is not None and velocity is not None:
                total = self.momentum * velocity[i] + total
                new_velocity.append(total.detach().clone())

            target_h = p.activations[i](total) if i < len(p.activations) else total
            h_new = all_acts[i + 1] + self.step_size * (target_h - all_acts[i + 1])

            new_acts.append(h_new)

        out = self._bottom_up(new_acts[-1], len(p.weights) - 1)

        if beta > 0 and target is not None:
            out += beta * (_one_hot(target, out) - out)

        new_acts.append(out)

        return new_acts, new_velocity

    def pseudo_gradient(
        self,
        free_acts: list[Tensor],
        nudged_acts: list[Tensor],
        beta: float,
    ) -> list[Tensor]:
        """Contrastive Hebbian pseudo-gradients: (free - nudged) / (beta * N).

        Matches ThermodynamicContrast.compute_pseudo_gradient math exactly.
        """
        if beta <= 0:
            return []

        batch = free_acts[0].shape[0]
        n_layers = len(free_acts) - 1
        grads = []

        for layer_idx in range(n_layers):
            if layer_idx < len(self.params.weights):
                free_pre = free_acts[layer_idx]
                free_post = free_acts[layer_idx + 1]
                free_corr = free_pre.T @ free_post

                nudged_pre = nudged_acts[layer_idx]
                nudged_post = nudged_acts[layer_idx + 1]
                nudged_corr = nudged_pre.T @ nudged_post

                contrast = (free_corr - nudged_corr) / beta / batch
                grads.append(contrast.T)

        return grads

    def apply_weight_update(self, grads: list[Tensor], lr: float) -> None:
        """Consolidate ΔW through the substrate's physical update operator.

        For each layer: ΔW = update_op(-lr * grad, current_W); W += ΔW
        """
        for g, w in zip(grads, self.params.weights):
            delta = self._update_op(-lr * g, w)
            with torch.no_grad():
                w.add_(delta)

    def effective_weights(self) -> tuple[Tensor, ...]:
        """Return substrate-quantized view of current weights (for inspection)."""
        return tuple(self.substrate.quantize_weights(w) for w in self.params.weights)
