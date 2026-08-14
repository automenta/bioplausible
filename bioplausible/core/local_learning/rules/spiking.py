"""
Spiking Neural Network propagator (STDP).

Classes: STDPLearningRule

A self-contained spike-timing-dependent-plasticity local rule that operates on
any model exposing ``transition_modules()`` returning ``nn.Linear`` layers. It
rate-encodes the input, propagates spikes forward through the linear layers, and
applies an STDP-style update (pre-before-post potentiation minus a trace-based
anti-Hebbian term) to each layer's weights. This runs entirely via PyTorch and
does not require snnTorch — ``SpikingSTDP`` (``zoo.models.spiking``) remains the
full LIF/snnTorch model that owns STDP internally; this propagator provides an
equivalent local rule through the Zoo propagator interface.
"""

import torch
from torch import nn

from bioplausible.core.registry import LocalityLevel, register_propagator

from .base import LearningRuleOptimizer

__all__ = [
    "STDPLearningRule",
]


@register_propagator(
    "stdp",
    family="spiking",
    locality_level=LocalityLevel.LOCAL,
    bio_plausibility_score=0.95,
    credit_assignment_type="spiking",
    requires_backward=False,
    memory_complexity="O(N)",
    requires=["transition_graph"],
    tags=["spiking", "stdp", "hebbian", "local"],
    description=(
        "Spike-Timing-Dependent Plasticity (STDP): local weight update from"
        " correlations between pre- and post-synaptic spike traces."
    ),
)
class STDPLearningRule(LearningRuleOptimizer):
    """Spike-Timing-Dependent Plasticity (STDP) local learning rule.

    Rate-encodes the input as spikes, propagates them once forward through the
    model's linear transition layers, and updates each ``nn.Linear`` weight in
    place from the running spike traces    ::

        dw = lr * (A+ * post^T @ pre_trace - A- * post_trace^T @ pre) / batch

    where ``post`` is the current post-synaptic spikes, ``pre_trace`` the
    pre-synaptic eligibility trace, ``post_trace`` the post-synaptic trace, and
    ``pre`` the current pre-synaptic spikes. The first (potentiating) term
    strengthens co-active pre/post targeting; the second (depressing) term,
    scaled by the distinct amplitude ``A-``, yields the characteristic STDP
    curve (net LTP for strongly correlated pre/post).
    """

    def __init__(
        self,
        params,
        model: nn.Module,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        num_steps: int = 10,
        threshold: float = 0.5,
        tau_pre: float = 0.9,
        tau_post: float = 0.9,
        a_plus: float = 1.0,
        a_minus: float = 0.1,
    ):
        super().__init__(params, model, lr, momentum, weight_decay)
        self.lr = lr
        self.num_steps = num_steps
        self.threshold = threshold
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.a_plus = a_plus
        self.a_minus = a_minus

    def step(self, x: torch.Tensor, target: torch.Tensor | None = None) -> None:
        layers = self._get_transitions()
        if not layers:
            raise ValueError(
                "STDP requires a model with at least one linear transition layer"
            )

        state = self._encode(x)
        with torch.no_grad():
            for layer in layers:
                inputs = state
                pre_trace = torch.zeros_like(inputs)
                post_trace = torch.zeros_like(layer(inputs))
                spikes = inputs
                can_feedback = layer.in_features == layer.out_features
                for _ in range(self.num_steps):
                    pre_trace = self.tau_pre * pre_trace + inputs
                    post = (layer(spikes) > self.threshold).float()
                    post_trace = self.tau_post * post_trace + post

                    # Canonical STDP: potentiation scales with the pre-synaptic
                    # trace (pre fired before post), depression with the
                    # post-synaptic trace. Distinct amplitudes A+ / A- give the
                    # characteristic asymmetric learning window (net LTP for
                    # strongly correlated pre/post).
                    pot = self.a_plus * (post.t() @ pre_trace)
                    dep = self.a_minus * (post_trace.t() @ inputs)
                    dw = self.lr * (pot - dep) / inputs.shape[0]
                    layer.weight.data += dw

                    if not can_feedback:
                        break
                    spikes = post
                state = spikes

    def _get_transitions(self) -> list[nn.Module]:
        if not hasattr(self.model, "transition_modules"):
            raise TypeError(
                f"STDP requires a model implementing TransitionGraph. "
                f"{type(self.model).__name__} does not implement "
                f"transition_modules()."
            )
        return [m for m in self.model.transition_modules() if isinstance(m, nn.Linear)]

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(x)
        return (torch.rand_like(probs) < probs).float()
