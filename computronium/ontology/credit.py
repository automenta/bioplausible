"""Layer 4: CreditAssignment — Error Routing & Pseudo-Gradients."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

import torch
from torch import Tensor

from computronium.ontology.utils import _learnable_weight_names

if TYPE_CHECKING:
    from collections.abc import Mapping

    from computronium.ontology.geometry import Geometry
    from computronium.ontology.system import SystemState


# ============================================================
# CreditAssignment Configuration
# ============================================================


@dataclass(frozen=True, slots=True)
class CreditAssignmentConfig:
    """Configuration for credit assignment.

    Attributes:
        credit_type: "thermodynamic_contrast", "random_projections",
            "local_goodness", "temporal_trace", "target_inversion"
        beta: Nudge strength (for thermodynamic contrast)
        feedback_matrix: Optional fixed feedback matrix
            (for random projections)
        local_objective: Layer-local loss type (for local goodness)
        orthogonal_init: Initialize feedback matrices with orthogonal weights
        feedback_scale: Scaling factor for feedback matrices
        a_plus: STDP potentiation amplitude (for temporal_trace)
        a_minus: STDP depression amplitude (for temporal_trace)
        tau: STDP time constant (for temporal_trace, legacy)
        tau_pre: STDP pre-synaptic trace time constant (for timing-asymmetric STDP)
        tau_post: STDP post-synaptic trace time constant (for timing-asymmetric STDP)
        homeostatic_target: Target activation norm for homeostatic scaling
    """

    credit_type: str
    beta: float
    feedback_matrix: Tensor | None
    local_objective: str
    orthogonal_init: bool
    feedback_scale: float
    a_plus: float = 1.0
    a_minus: float = 1.0
    tau: float = 20.0
    tau_pre: float = 0.9
    tau_post: float = 0.9
    homeostatic_target: float = 1.0

    @classmethod
    def thermodynamic_contrast(
        cls,
        *,
        beta: float = 0.5,
        feedback_matrix: Tensor | None = None,
        local_objective: str = "mse",
        orthogonal_init: bool = False,
        feedback_scale: float = 0.01,
    ) -> CreditAssignmentConfig:
        return cls(
            credit_type="thermodynamic_contrast",
            beta=beta,
            feedback_matrix=feedback_matrix,
            local_objective=local_objective,
            orthogonal_init=orthogonal_init,
            feedback_scale=feedback_scale,
        )

    @classmethod
    def random_projections(
        cls,
        *,
        beta: float = 0.5,
        feedback_matrix: Tensor | None = None,
        local_objective: str = "mse",
        orthogonal_init: bool = False,
        feedback_scale: float = 0.01,
    ) -> CreditAssignmentConfig:
        return cls(
            credit_type="random_projections",
            beta=beta,
            feedback_matrix=feedback_matrix,
            local_objective=local_objective,
            orthogonal_init=orthogonal_init,
            feedback_scale=feedback_scale,
        )

    @classmethod
    def local_goodness(
        cls,
        *,
        beta: float = 0.5,
        feedback_matrix: Tensor | None = None,
        local_objective: str = "mse",
        orthogonal_init: bool = False,
        feedback_scale: float = 0.01,
    ) -> CreditAssignmentConfig:
        return cls(
            credit_type="local_goodness",
            beta=beta,
            feedback_matrix=feedback_matrix,
            local_objective=local_objective,
            orthogonal_init=orthogonal_init,
            feedback_scale=feedback_scale,
        )

    @classmethod
    def temporal_trace(
        cls,
        *,
        beta: float = 0.5,
        feedback_matrix: Tensor | None = None,
        local_objective: str = "mse",
        orthogonal_init: bool = False,
        feedback_scale: float = 0.01,
        a_plus: float = 1.0,
        a_minus: float = 0.5,
        tau: float = 20.0,
        tau_pre: float = 0.9,
        tau_post: float = 0.9,
    ) -> CreditAssignmentConfig:
        """Potentiation/depression weights must differ: the rate-coded
        surrogate correlates the same (pre, post) activity pair in both
        directions, so ``a_plus == a_minus`` yields an identically-zero
        pseudo-gradient.

        For timing-asymmetric STDP (when spike rasters are available),
        ``tau_pre`` and ``tau_post`` control the exponential decay of the
        pre- and post-synaptic eligibility traces.
        """
        return cls(
            credit_type="temporal_trace",
            beta=beta,
            feedback_matrix=feedback_matrix,
            local_objective=local_objective,
            orthogonal_init=orthogonal_init,
            feedback_scale=feedback_scale,
            a_plus=a_plus,
            a_minus=a_minus,
            tau=tau,
            tau_pre=tau_pre,
            tau_post=tau_post,
        )

    @classmethod
    def target_inversion(
        cls,
        *,
        beta: float = 0.5,
        feedback_matrix: Tensor | None = None,
        local_objective: str = "mse",
        orthogonal_init: bool = False,
        feedback_scale: float = 0.01,
    ) -> CreditAssignmentConfig:
        return cls(
            credit_type="target_inversion",
            beta=beta,
            feedback_matrix=feedback_matrix,
            local_objective=local_objective,
            orthogonal_init=orthogonal_init,
            feedback_scale=feedback_scale,
        )

    @classmethod
    def homeostatic(
        cls,
        *,
        beta: float = 0.5,
        feedback_matrix: Tensor | None = None,
        local_objective: str = "mse",
        orthogonal_init: bool = False,
        feedback_scale: float = 0.01,
    ) -> CreditAssignmentConfig:
        return cls(
            credit_type="homeostatic",
            beta=beta,
            feedback_matrix=feedback_matrix,
            local_objective=local_objective,
            orthogonal_init=orthogonal_init,
            feedback_scale=feedback_scale,
        )

    @classmethod
    def gradient(
        cls,
        *,
        beta: float = 0.5,
        feedback_matrix: Tensor | None = None,
        local_objective: str = "mse",
        orthogonal_init: bool = False,
        feedback_scale: float = 0.01,
    ) -> CreditAssignmentConfig:
        return cls(
            credit_type="gradient",
            beta=beta,
            feedback_matrix=feedback_matrix,
            local_objective=local_objective,
            orthogonal_init=orthogonal_init,
            feedback_scale=feedback_scale,
        )


# ============================================================
# CreditAssignment Protocol
# ============================================================


class Phase(StrEnum):
    """Settling phase declared by a credit rule.

    Credits declare exactly the phases they consume (``phases`` ClassVar);
    the pipeline settles only declared phases, removing wasted settles for
    families that ignore free or nudged states.
    """

    FREE = "free"
    NUDGED = "nudged"


@runtime_checkable
class CreditAssignment(Protocol):
    """How the network computes the direction of learning (pseudo-gradient).

    Uses only locally available signals to compute a gradient-like update
    direction. The five canonical types:
    - ThermodynamicContrast: (nudged - free) / beta (EqProp)
    - RandomProjections: Fixed/adaptive random feedback matrices (FA, DFA)
    - LocalGoodness: Layer-local contrastive objectives (FF, PEPITA)
    - TemporalTrace: Spike-timing correlations (STDP)
    - TargetInversion: Propagating local targets (Target Prop)

    Capabilities are declared, not assumed:
    - ``phases``: settling phases the rule consumes; the pipeline settles
      only these.
    - ``requires_autograd``: True when the rule needs autograd through
      settling (the default detached/no-grad path is bypassed only then).

    Output is a list of pseudo-gradients, one per learnable weight layer,
    matching the Geometry's parameter structure.
    """

    config: CreditAssignmentConfig

    phases: ClassVar[tuple[Phase, ...]]
    requires_autograd: ClassVar[bool]

    @abstractmethod
    def compute_pseudo_gradient(
        self,
        states: Mapping[Phase, SystemState],
        loss: Tensor | None,
        geometry: Geometry,
    ) -> list[Tensor]:
        """Compute pseudo-gradients from the settled phase states.

        Args:
            states: Phase-keyed settled states; contains exactly the phases
                this rule declares.
            loss: Task loss at the pipeline's output state (None-safe).
            geometry: Network topology (provides layer structure)

        Returns:
            List of pseudo-gradient tensors, one per learnable weight layer
        """
        ...

    # DEFAULT METHOD — non-breaking, only overridden by LocalGoodness/TargetInversion
    def surrogate_objective(
        self,
        free_state: SystemState,
        nudged_state: SystemState,
        geometry: Geometry,
    ) -> Tensor:
        """Compute layer-local surrogate loss for gradient checking.

        Only LocalGoodnessCredit and TargetInversionCredit override this.
        Others raise NotImplementedError.
        """
        raise NotImplementedError(
            "Surrogate objective not defined for this credit rule"
        )


def _acts_list(activations: list[Tensor] | Tensor | None) -> list[Tensor]:
    """Normalize activations to a list of layer tensors."""
    if activations is None:
        return []
    if isinstance(activations, list):
        return activations
    return [activations]


def _block_transition_acts(
    acts: list[Tensor], geometry: Geometry
) -> list[Tensor] | None:
    """The settled acts when the geometry declares a block transition table
    and they align with it (tile meshes, R11.1.4); else None."""
    count = getattr(geometry, "block_act_count", None)
    if count is not None and len(acts) == count:
        return acts
    return None


def _propagate_targets(
    acts: list[Tensor],
    y: Tensor,
    weight_names: list[str],
    geometry: Geometry,
) -> list[Tensor | None]:
    """Transpose-feedback target propagation: t_L = one-hot(y), t_l = t_{l+1} @ W_{l+1}."""
    out_dim = acts[-1].shape[-1]
    targets: list[Tensor | None] = [None] * len(acts)
    targets[-1] = torch.nn.functional.one_hot(y, num_classes=out_dim).float()
    for l in range(min(len(weight_names), len(acts) - 1) - 1, -1, -1):  # ruff: ignore[ambiguous-variable-name]
        nxt = targets[l + 1]
        if nxt is None:
            break
        w = geometry.params[weight_names[l]]
        if nxt.shape[-1] != w.shape[0]:
            # Ragged weight (e.g. tile meshes): the transition contract
            # does not hold — stop propagating rather than crash.
            break
        targets[l] = nxt @ w
    return targets


def _weight_acts(
    weight_names: list[str], acts: list[Tensor], geometry: Geometry
) -> list[tuple[Tensor, Tensor] | None]:
    """(pre, post) activation pair each learnable weight consumes.

    Positional over act transitions: weight i maps acts[i] -> acts[i+1].
    Surplus weights (recurrent self-connections) consume the last hidden
    layer as both pre and post — the correlate the settle kernel itself
    feeds them. Weights whose shape does not match their act-pair widths
    (tile/ragged meshes) map to None — credits emit zeros for them.
    """
    n_trans = len(acts) - 1
    pairs: list[tuple[Tensor, Tensor] | None] = []
    for i, name in enumerate(weight_names):
        w = geometry.params[name]
        if i < n_trans:
            pre, post = acts[i], acts[i + 1]
        elif n_trans >= 1:
            pre = post = acts[n_trans - 1]
        else:
            pairs.append(None)
            continue
        if w.shape[0] == post.shape[-1] and w.shape[1] == pre.shape[-1]:
            pairs.append((pre, post))
        else:
            pairs.append(None)
    return pairs


# ============================================================
# Default/Reference CreditAssignment Implementations
# ============================================================


class ThermodynamicContrast:
    """Equilibrium Propagation credit assignment: (nudged - free) / beta.

    For feedforward networks, computes parameter pseudo-gradients using the
    contrastive Hebbian rule: ΔW = (free_pre @ free_post - nudged_pre @ nudged_post) / β
    """

    phases: ClassVar[tuple[Phase, ...]] = (Phase.FREE, Phase.NUDGED)
    requires_autograd: ClassVar[bool] = False

    def __init__(self, config: CreditAssignmentConfig | None = None):
        self.config = config or CreditAssignmentConfig.thermodynamic_contrast()

    def compute_pseudo_gradient(
        self,
        states: Mapping[Phase, SystemState],
        loss: Tensor | None,
        geometry: Geometry,
    ) -> list[Tensor]:
        free_state = states.get(Phase.FREE)
        nudged_state = states.get(Phase.NUDGED)
        if free_state is None or nudged_state is None:
            return []
        if free_state.activations is None or nudged_state.activations is None:
            return []

        free_acts = (
            free_state.activations
            if isinstance(free_state.activations, list)
            else [free_state.activations]
        )
        nudged_acts = (
            nudged_state.activations
            if isinstance(nudged_state.activations, list)
            else [nudged_state.activations]
        )

        # Tile meshes: per-transition block contrast scattered to per-edge
        # weights (R11.1.4).
        if _block_transition_acts(free_acts, geometry) is not None and (
            len(nudged_acts) == len(free_acts)
        ):
            batch = free_acts[0].shape[0]
            block_grads = [
                (
                    free_acts[i].T @ free_acts[i + 1]
                    - nudged_acts[i].T @ nudged_acts[i + 1]
                ).T
                / self.config.beta
                / batch
                for i in range(len(free_acts) - 1)
            ]
            return geometry.scatter_block_grads(block_grads)

        weight_names = _learnable_weight_names(geometry.params)

        grads = []
        # Compute contrastive Hebbian gradients for each weight matrix
        # Assume activations are ordered: [input, hidden1, hidden2, ..., output]
        n_layers = len(free_acts) - 1
        for l in range(n_layers):  # ruff: ignore[ambiguous-variable-name]
            if l < len(weight_names):
                # Free phase correlation: free_pre^T @ free_post
                free_pre = free_acts[l]  # (batch, in_dim)
                free_post = free_acts[l + 1]  # (batch, out_dim)
                free_corr = free_pre.T @ free_post  # (in_dim, out_dim)

                # Nudged phase correlation
                nudged_pre = nudged_acts[l]
                nudged_post = nudged_acts[l + 1]
                nudged_corr = nudged_pre.T @ nudged_post

                # Contrastive gradient: (free - nudged) / β (standard EqProp)
                contrast = (
                    (free_corr - nudged_corr) / self.config.beta / free_pre.shape[0]
                )
                # Transpose to match weight shape (out_dim, in_dim)
                grads.append(contrast.T)

        return grads

    def surrogate_objective(
        self,
        free_state: SystemState,
        nudged_state: SystemState,
        geometry: Geometry,
    ) -> Tensor:
        """Surrogate objective for EqProp: negative free energy difference."""
        if free_state.energy is not None and nudged_state.energy is not None:
            return torch.as_tensor(nudged_state.energy - free_state.energy)
        return torch.tensor(0.0)


# Alias for backwards compatibility with README/docs
ThermodynamicContrastCredit = ThermodynamicContrast


class RandomProjectionsCredit:
    """Fixed random feedback pathways (FA/DFA) with an autograd readout.

    Top error δ_L = ∂L/∂a_L via autograd on the task loss (no weight
    transport at the readout), propagated down through FIXED random
    matrices: δ_i = δ_{i+1} @ B_i with B_i ~ feedback_scale · N(0,1),
    fixed at first use. Per-layer pseudo-gradient ΔW_i = δ_{i+1}ᵀ a_i / batch;
    recurrent self-connections project the last hidden layer's error
    through their own fixed feedback. When the settle graph is unavailable
    the signal is zeros — never fabricated noise.
    """

    phases: ClassVar[tuple[Phase, ...]] = (Phase.FREE, Phase.NUDGED)
    requires_autograd: ClassVar[bool] = True

    def __init__(self, config: CreditAssignmentConfig | None = None):
        self.config = config or CreditAssignmentConfig.random_projections()
        self._feedback_weights: dict[str, Tensor] = {}

    def _init_feedback_weights(
        self, geometry: Geometry, device: torch.device | None = None
    ) -> None:
        """Initialize fixed random feedback matrices for each learnable weight.

        Feedback matrices are initialized once and kept constant throughout training.
        This implements the FA/DFA assumption of fixed feedback pathways.
        """
        if self._feedback_weights:
            return  # Already initialized
        for name in _learnable_weight_names(geometry.params):
            param = geometry.params[name]
            # Initialize with small random values
            fb = torch.randn_like(param, device=device) * self.config.feedback_scale
            self._feedback_weights[name] = fb

    def compute_pseudo_gradient(
        self,
        states: Mapping[Phase, SystemState],
        loss: Tensor | None,
        geometry: Geometry,
    ) -> list[Tensor]:
        """Compute FA pseudo-gradients through fixed feedback matrices."""
        nudged_state = states.get(Phase.NUDGED)
        if loss is None or nudged_state is None or nudged_state.activations is None:
            return []

        acts = _acts_list(nudged_state.activations)
        weight_names = _learnable_weight_names(geometry.params)
        if len(acts) < 2 or not weight_names:
            return []

        self._init_feedback_weights(geometry, acts[-1].device)
        logits = acts[-1]
        if not logits.requires_grad:
            # Settle graph not preserved (e.g. detached settle paths): no
            # error signal exists — zeros, never fabricated signal.
            return [torch.zeros_like(geometry.params[n]) for n in weight_names]

        delta_out = torch.autograd.grad(loss, logits)[0].detach()
        n_trans = len(acts) - 1
        batch = acts[0].shape[0]

        # Tile meshes: feedback blocks assembled per transition, error
        # walked back over the block layout, grads scattered to per-edge
        # weights (R11.1.4). B_e shares its weight's shape, so the layered
        # contract (B maps act_{k+1} widths down to act_k) holds per edge.
        if _block_transition_acts(acts, geometry) is not None:
            blocks = geometry.assemble_blocks(self._feedback_weights)
            err = delta_out
            block_grads: list[Tensor] = []
            for i in range(n_trans - 1, -1, -1):
                block_grads.append(err.T @ acts[i] / batch)
                err = err @ blocks[i]  # ruff: ignore[non-augmented-assignment]
            block_grads.reverse()
            return geometry.scatter_block_grads(block_grads)

        # Layered contract: feedback B_k must map the act-space of layer
        # k+1 down to layer k. Tile/ragged weights (e.g. per-tile matrices
        # not aligned with the act widths) break the chain — zeros, never
        # fabricated signal.
        for k in range(n_trans):
            b = self._feedback_weights[weight_names[k]]
            if b.shape[0] != acts[k + 1].shape[-1] or b.shape[1] != acts[k].shape[-1]:
                return [torch.zeros_like(geometry.params[n]) for n in weight_names]

        # Feedback-propagated error at each act layer:
        # err_at[L] = ∂L/∂a_L; err_at[k] = err_at[k+1] @ B_k.
        # During the sweep, err is err_at[i + 1] on entering iteration i.
        grads: list[Tensor] = []
        err = delta_out
        hidden_err = delta_out
        for i in range(n_trans - 1, -1, -1):
            grads.append(err.T @ acts[i] / batch)
            err = err @ self._feedback_weights[weight_names[i]]  # ruff: ignore[non-augmented-assignment]
            if i == n_trans - 1:
                # Error at the last hidden layer: upstream of the recurrent
                # self-connection.
                hidden_err = err
        grads.reverse()
        for _name in weight_names[n_trans:]:
            # Recurrent self-connection at the last hidden layer: its
            # upstream error is the propagated error at that layer.
            grads.append(hidden_err.T @ acts[n_trans - 1] / batch)
        return grads

    def surrogate_objective(
        self,
        free_state: SystemState,
        nudged_state: SystemState,
        geometry: Geometry,
    ) -> Tensor:
        """Surrogate objective not defined for RandomProjectionsCredit."""
        raise NotImplementedError(
            "Surrogate objective not defined for RandomProjectionsCredit"
        )


class LocalGoodnessCredit:
    """Layer-local contrastive objective (Forward-Forward, PEPITA).

    Goodness G_l = mean(acts_l^2) per layer. The pseudo-gradient descends
    (G_free - G_nudged) so nudged goodness increases, free goodness decreases.
    """

    phases: ClassVar[tuple[Phase, ...]] = (Phase.FREE, Phase.NUDGED)
    requires_autograd: ClassVar[bool] = True

    def __init__(self, config: CreditAssignmentConfig | None = None):
        self.config = config or CreditAssignmentConfig.local_goodness()

    def compute_pseudo_gradient(
        self,
        states: Mapping[Phase, SystemState],
        loss: Tensor | None,
        geometry: Geometry,
    ) -> list[Tensor]:
        free_state = states.get(Phase.FREE)
        nudged_state = states.get(Phase.NUDGED)
        if free_state is None or nudged_state is None:
            return []

        free_acts = _acts_list(free_state.activations)
        nudged_acts = _acts_list(nudged_state.activations)
        if not free_acts or not nudged_acts:
            return []

        weight_names = _learnable_weight_names(geometry.params)
        if not weight_names:
            return []

        # Layer-local goodness objective summed over act transitions
        # (surplus weights — recurrent self-connections — receive their
        # gradient through the shared hidden-layer goodness).
        n_trans = min(len(free_acts), len(nudged_acts)) - 1
        if n_trans < 1 or not nudged_acts[-1].requires_grad:
            # Settle graph not preserved: no autograd signal — zeros.
            return [torch.zeros_like(geometry.params[n]) for n in weight_names]

        total = torch.zeros((), device=nudged_acts[-1].device)
        for i in range(1, n_trans + 1):
            total += free_acts[i].pow(2).mean() - nudged_acts[i].pow(2).mean()

        params = [geometry.params[n] for n in weight_names]
        grads = torch.autograd.grad(
            total, params, retain_graph=False, create_graph=False, allow_unused=True
        )
        return [
            g if g is not None else torch.zeros_like(p)
            for p, g in zip(params, grads, strict=True)
        ]

    def surrogate_objective(
        self,
        free_state: SystemState,
        nudged_state: SystemState,
        geometry: Geometry,
    ) -> Tensor:
        """Sum of layer-local goodness differences."""
        free_acts = _acts_list(free_state.activations)
        nudged_acts = _acts_list(nudged_state.activations)
        if not free_acts or not nudged_acts:
            return torch.tensor(0.0)
        total = torch.tensor(0.0)
        for i in range(1, min(len(free_acts), len(nudged_acts))):
            total = total + free_acts[i].pow(2).mean() - nudged_acts[i].pow(2).mean()
        return total


class TemporalTraceCredit:
    """Spike-timing correlations (STDP).

    Supports two modes:
    - Rate-coded surrogate: uses settled (pre, post) activity pairs as
      weighted Hebbian correlation. This is the fallback when spike timing
      data is unavailable.
    - Timing-asymmetric STDP: uses per-neuron per-step spike rasters from
      ``SpikeIntegrationDynamics`` to compute eligibility traces (pre/post
      traces) and applies the canonical STDP update:
        Δw ∝ a_plus * post^T @ pre_trace - a_minus * post_trace^T @ pre
      where traces are exponential filters over spike history.
    """

    phases: ClassVar[tuple[Phase, ...]] = (Phase.FREE,)
    requires_autograd: ClassVar[bool] = False

    def __init__(self, config: CreditAssignmentConfig | None = None):
        self.config = config or CreditAssignmentConfig.temporal_trace()

    def compute_pseudo_gradient(
        self,
        states: Mapping[Phase, SystemState],
        loss: Tensor | None,
        geometry: Geometry,
    ) -> list[Tensor]:
        free_state = states.get(Phase.FREE)
        if free_state is None or free_state.activations is None:
            return []

        acts = _acts_list(free_state.activations)
        if len(acts) < 2:
            return []

        weight_names = _learnable_weight_names(geometry.params)
        if not weight_names:
            return []

        # Check if spike rasters are available for timing-asymmetric STDP
        spike_rasters = getattr(free_state, "spike_rasters", None)
        use_timing_stdp = (
            spike_rasters is not None
            and len(spike_rasters) > 0
            and isinstance(spike_rasters[0], list)
        )

        batch = acts[0].shape[0]
        grads = []

        if use_timing_stdp:
            # Timing-asymmetric STDP using eligibility traces
            # spike_rasters[layer][step] = [batch, neurons] for that layer's output
            tau_pre = getattr(self.config, "tau_pre", 0.9)
            tau_post = getattr(self.config, "tau_post", 0.9)
            a_plus = self.config.a_plus
            a_minus = self.config.a_minus

            # For each weight layer, we need pre and post spike rasters
            # Weight i connects acts[i] (pre) -> acts[i+1] (post)
            # Post-synaptic spikes for weight i: spike_rasters[i] (output of layer i)
            # Pre-synaptic spikes for weight i: spike_rasters[i-1] (output of layer i-1)
            # For i=0 (first layer), pre-synaptic is the input - rate encode it
            for w_idx, name in enumerate(weight_names):
                if w_idx >= len(spike_rasters):
                    grads.append(torch.zeros_like(geometry.params[name]))
                    continue

                post_rasters = spike_rasters[w_idx]  # [step] -> [batch, post_neurons]
                if not post_rasters:
                    grads.append(torch.zeros_like(geometry.params[name]))
                    continue

                # Get pre-synaptic spikes
                if w_idx == 0:
                    # First layer: rate-encode input as pre-synaptic spikes
                    # Input is acts[0] = x, shape [batch, in_dim]
                    # We need to encode it as spikes for each time step
                    # Use the same encoding as STDPLearningRule: sigmoid + Bernoulli
                    x = acts[0]
                    probs = torch.sigmoid(x)
                    # Generate spikes for each time step (same pattern each step for rate coding)
                    pre_rasters = [
                        (torch.rand_like(probs) < probs).float()
                        for _ in range(len(post_rasters))
                    ]
                else:
                    pre_rasters = spike_rasters[w_idx - 1]

                if not pre_rasters or len(pre_rasters) != len(post_rasters):
                    grads.append(torch.zeros_like(geometry.params[name]))
                    continue

                # Compute eligibility traces
                # pre_trace accumulates pre-synaptic spikes with exponential decay
                pre_trace = torch.zeros_like(pre_rasters[0])  # [batch, pre_neurons]
                for pre_spikes in pre_rasters:
                    pre_trace = tau_pre * pre_trace + pre_spikes

                # post_trace accumulates post-synaptic spikes with exponential decay
                post_trace = torch.zeros_like(post_rasters[0])  # [batch, post_neurons]
                for post_spikes in post_rasters:
                    post_trace = tau_post * post_trace + post_spikes

                # STDP update: pot = a_plus * post^T @ pre_trace, dep = a_minus * post_trace^T @ pre
                # We need the final pre and post activity (last time step or average)
                pre_final = pre_rasters[-1]  # [batch, pre_neurons]
                post_final = post_rasters[-1]  # [batch, post_neurons]

                # Potentiation: post^T @ pre_trace -> [post, pre]
                pot = a_plus * (post_final.T @ pre_trace) / batch
                # Depression: post_trace^T @ pre -> [post, pre]
                dep = a_minus * (post_trace.T @ pre_final) / batch

                stdp_grad = -(pot - dep)
                grads.append(stdp_grad)
        else:
            # Rate-coded surrogate (fallback)
            for pair, name in zip(
                _weight_acts(weight_names, acts, geometry), weight_names, strict=True
            ):
                if pair is None:
                    grads.append(torch.zeros_like(geometry.params[name]))
                    continue
                pre, post = pair

                # Causal correlation: post^T @ pre -> [out_dim, in_dim] (matches weight shape)
                causal = post.T @ pre / batch
                # Anti-causal: pre^T @ post -> [in_dim, out_dim], transpose for weight shape
                anticausal = pre.T @ post / batch
                anticausal_w = anticausal.T

                # STDP: potentiate causal, depress anti-causal
                # Pseudo-gradient descended: -(a_plus * causal - a_minus * anticausal_w)
                stdp_grad = -(
                    self.config.a_plus * causal - self.config.a_minus * anticausal_w
                )
                grads.append(stdp_grad)

        return grads

    def compute_stdp_window(
        self,
        pre_spikes: Tensor,  # [batch, 1] or [batch] - spike times (unused for window function)
        post_spikes: Tensor,  # [batch, 1] or [batch] - spike times (unused for window function)
        dt: Tensor,  # [n_dt] - time lag grid
    ) -> Tensor:
        """Compute STDP window W(Δt) over the lag grid dt.

        The STDP window is an antisymmetric function of the time lag Δt:
        W(Δt) = A+ exp(-Δt/τ) for Δt > 0 (potentiation),
              = -A- exp(Δt/τ) for Δt < 0 (depression),
              = 0 for Δt = 0.

        The pre_spikes and post_spikes arguments are retained for API consistency
        (e.g., batching multiple spike pairs) but the window function itself
        depends only on the time lag grid dt.

        Returns: Tensor of shape [batch, n_dt] where each row is W(dt).
        """
        # Standard STDP window function evaluated at dt grid
        pos_mask = dt > 0
        neg_mask = dt < 0

        window = torch.zeros_like(dt)
        window[pos_mask] = self.config.a_plus * torch.exp(
            -dt[pos_mask] / self.config.tau
        )
        window[neg_mask] = -self.config.a_minus * torch.exp(
            dt[neg_mask] / self.config.tau
        )

        # Expand to [batch, n_dt] for API consistency (same window for all pairs in batch)
        batch_size = pre_spikes.shape[0] if pre_spikes.ndim > 0 else 1
        return window.expand(batch_size, -1)


class TargetInversionCredit:
    """Propagating local targets (Target Prop) with transpose feedback.

    Output target = one-hot(y). Local targets propagated backward through
    transpose of weight matrices: t_l = t_{l+1} @ W_{l+1}.
    Per-layer pseudo-gradient = (acts_l - t_l)^T @ acts_{l-1} / batch.
    """

    phases: ClassVar[tuple[Phase, ...]] = (Phase.FREE, Phase.NUDGED)
    requires_autograd: ClassVar[bool] = True

    def __init__(self, config: CreditAssignmentConfig | None = None):
        self.config = config or CreditAssignmentConfig.target_inversion()

    def compute_pseudo_gradient(
        self,
        states: Mapping[Phase, SystemState],
        loss: Tensor | None,
        geometry: Geometry,
    ) -> list[Tensor]:
        nudged_state = states.get(Phase.NUDGED)
        if nudged_state is None or nudged_state.activations is None:
            return []

        acts = _acts_list(nudged_state.activations)
        if len(acts) < 2:
            return []

        weight_names = _learnable_weight_names(geometry.params)
        if not weight_names:
            return []

        # Target at output layer: one-hot of class indices
        y = nudged_state.y
        if y is None:
            return [torch.zeros_like(geometry.params[n]) for n in weight_names]

        n_trans = len(acts) - 1

        # Tile meshes: targets propagate over the assembled blocks, grads
        # scattered to per-edge weights (R11.1.4).
        if _block_transition_acts(acts, geometry) is not None:
            blocks = geometry.assemble_blocks(geometry.params)
            targets: list[Tensor | None] = [None] * len(acts)
            targets[-1] = torch.nn.functional.one_hot(
                y, num_classes=acts[-1].shape[-1]
            ).float()
            batch = acts[0].shape[0]
            for i in range(n_trans - 1, -1, -1):
                nxt = targets[i + 1]
                targets[i] = nxt @ blocks[i] if nxt is not None else None
            block_grads = [
                (acts[i + 1] - targets[i + 1]).T @ acts[i] / batch
                if targets[i + 1] is not None
                else torch.zeros(acts[i + 1].shape[-1], acts[i].shape[-1])
                for i in range(n_trans)
            ]
            return geometry.scatter_block_grads(block_grads)

        targets = _propagate_targets(acts, y, weight_names[:n_trans], geometry)
        pairs = _weight_acts(weight_names, acts, geometry)

        # Layer-local deltas and pseudo-gradients
        grads = []
        for i, name in enumerate(weight_names):
            pair = pairs[i]
            if pair is None or i >= n_trans:
                # Surplus/ragged weights (recurrent self-connections, tile
                # meshes) receive no propagated target — zeros, not
                # fabricated signal.
                grads.append(torch.zeros_like(geometry.params[name]))
                continue
            pre, post = pair
            tgt = targets[i + 1]
            if tgt is None:
                grads.append(torch.zeros_like(geometry.params[name]))
                continue
            delta = post - tgt  # [batch, out]
            # pseudo_grad = delta^T @ pre / batch -> [out, in]
            grad = delta.T @ pre / pre.shape[0]
            grads.append(grad)

        return grads

    def surrogate_objective(
        self,
        free_state: SystemState,
        nudged_state: SystemState,
        geometry: Geometry,
    ) -> Tensor:
        """Sum of layer-local target matching errors."""
        nudged_acts = _acts_list(nudged_state.activations)
        if not nudged_acts or nudged_state.y is None:
            return torch.tensor(0.0)
        y = nudged_state.y
        weight_names = _learnable_weight_names(geometry.params)
        targets = _propagate_targets(nudged_acts, y, weight_names, geometry)
        total = torch.tensor(0.0, device=nudged_acts[-1].device)
        for i in range(1, len(nudged_acts)):
            tgt = targets[i]
            if tgt is None:
                continue
            delta = nudged_acts[i] - tgt
            total = total + (delta**2).mean()  # ruff: ignore[non-augmented-assignment]
        return total


class HomeostaticCredit:
    """Homeostatic credit assignment (autonomous Lipschitz scaling).

    Per-layer scaling to keep activation norms near homeostatic_target.
    Pseudo-gradient: (mean|post| - target) * W / |W|_F (directional).
    """

    phases: ClassVar[tuple[Phase, ...]] = (Phase.FREE, Phase.NUDGED)
    requires_autograd: ClassVar[bool] = False

    def __init__(self, config: CreditAssignmentConfig | None = None):
        self.config = config or CreditAssignmentConfig.homeostatic()

    def compute_pseudo_gradient(
        self,
        states: Mapping[Phase, SystemState],
        loss: Tensor | None,
        geometry: Geometry,
    ) -> list[Tensor]:
        free_state = states.get(Phase.FREE)
        if free_state is None or free_state.activations is None:
            return []

        acts = _acts_list(free_state.activations)
        if len(acts) < 2:
            return []

        weight_names = _learnable_weight_names(geometry.params)
        if not weight_names:
            return []

        target = self.config.homeostatic_target
        grads = []
        for pair, name in zip(
            _weight_acts(weight_names, acts, geometry), weight_names, strict=True
        ):
            if pair is None:
                grads.append(torch.zeros_like(geometry.params[name]))
                continue
            _, post = pair
            weight = geometry.params[name]  # [out_dim, in_dim]

            # Mean post-activation norm (L2 across features, then mean across batch)
            post_norm = post.norm(dim=1).mean()  # scalar
            # Error from target
            err = post_norm - target
            # Direction: scale W by error, normalized by Frobenius norm
            weight_norm = weight.norm()
            if weight_norm > 0:
                grad = err * weight / weight_norm
            else:
                grad = torch.zeros_like(weight)
            grads.append(grad)

        return grads


class GradientCredit:
    """Standard backprop credit (for comparison)."""

    phases: ClassVar[tuple[Phase, ...]] = (Phase.FREE, Phase.NUDGED)
    requires_autograd: ClassVar[bool] = True

    def __init__(self, config: CreditAssignmentConfig | None = None):
        self.config = config or CreditAssignmentConfig.gradient()

    def compute_pseudo_gradient(
        self,
        states: Mapping[Phase, SystemState],
        loss: Tensor | None,
        geometry: Geometry,
    ) -> list[Tensor]:
        """Compute true gradients via autograd on nudged state loss."""
        if loss is None:
            return []

        weight_names = _learnable_weight_names(geometry.params)
        params = [geometry.params[n] for n in weight_names]
        grads = torch.autograd.grad(
            loss, params, retain_graph=False, create_graph=False, allow_unused=True
        )
        detached = [n for n, g in zip(weight_names, grads, strict=True) if g is None]
        if detached:
            raise RuntimeError(
                f"GradientCredit: no gradient reached {detached}. The loss "
                "graph does not touch every learnable weight — a dynamics "
                "that detaches activations would silently degrade learning "
                "to the reached layers only. Zero-filling hides that failure."
            )
        return list(grads)

    def surrogate_objective(
        self,
        free_state: SystemState,
        nudged_state: SystemState,
        geometry: Geometry,
    ) -> Tensor:
        """The nudged loss is the surrogate objective for true gradients."""
        loss = nudged_state.loss
        if isinstance(loss, Tensor):
            return loss
        return torch.tensor(0.0)


# Alias for backwards compatibility
BackpropCredit = GradientCredit
