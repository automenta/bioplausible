"""Layer 4: CreditAssignment — Error Routing & Pseudo-Gradients."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

import torch
from torch import Tensor

from computronium.ontology.utils import _learnable_weight_names

if TYPE_CHECKING:
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
        tau: STDP time constant (for temporal_trace)
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
        a_minus: float = 1.0,
        tau: float = 20.0,
    ) -> CreditAssignmentConfig:
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
    for l in range(len(weight_names) - 1, -1, -1):
        nxt = targets[l + 1]
        if nxt is None:
            break
        targets[l] = nxt @ geometry.params[weight_names[l]]
    return targets


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

        weight_names = _learnable_weight_names(geometry.params)

        grads = []
        # Compute contrastive Hebbian gradients for each weight matrix
        # Assume activations are ordered: [input, hidden1, hidden2, ..., output]
        n_layers = len(free_acts) - 1
        for l in range(n_layers):
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
    """Random feedback alignment credit (FA, DFA)."""

    phases: ClassVar[tuple[Phase, ...]] = (Phase.FREE, Phase.NUDGED)
    requires_autograd: ClassVar[bool] = False

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
        weight_names = _learnable_weight_names(geometry.params)
        for name in weight_names:
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
        """Compute FA pseudo-gradients using fixed feedback matrices."""
        free_state = states.get(Phase.FREE)
        nudged_state = states.get(Phase.NUDGED)
        if free_state is None or nudged_state is None:
            return []

        weight_names = _learnable_weight_names(geometry.params)
        if not self._feedback_weights:
            device = (
                next(iter(geometry.params.values())).device if geometry.params else None
            )
            self._init_feedback_weights(geometry, device)

        # FA error signal: difference between nudged and free activations
        free_acts = free_state.activations
        nudged_acts = nudged_state.activations
        # Handle activations as list (take last element) or tensor
        if isinstance(free_acts, list):
            free_acts = free_acts[-1] if free_acts else None
        if isinstance(nudged_acts, list):
            nudged_acts = nudged_acts[-1] if nudged_acts else None
        if free_acts is not None and nudged_acts is not None:
            error = (nudged_acts - free_acts).detach()
        else:
            error = (
                torch.ones_like(nudged_acts)
                if nudged_acts is not None
                else torch.tensor(1.0)
            )

        # Project error through feedback matrices (one per layer)
        # Apply feedback_scale to gradient magnitude during training (not just init)
        grads = []
        for name in weight_names:
            fb = self._feedback_weights.get(name)
            if fb is not None:
                # FA gradient: feedback @ error, scaled by feedback_scale
                grad = fb * error.abs().mean() * self.config.feedback_scale
            else:
                grad = torch.zeros_like(geometry.params[name])
            grads.append(grad)
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
        if len(weight_names) != len(free_acts) - 1:
            # Mismatch: return zeros for all weights
            return [torch.zeros_like(geometry.params[n]) for n in weight_names]

        grads = []
        for i, name in enumerate(weight_names):
            # Layer i maps acts[i] -> acts[i+1]
            post_free = free_acts[i + 1]
            post_nudged = nudged_acts[i + 1]

            # Goodness = mean squared activation
            g_free = post_free.pow(2).mean()
            g_nudged = post_nudged.pow(2).mean()

            # Local objective: descend (G_free - G_nudged) -> raise nudged, lower free
            local_obj = g_free - g_nudged

            # Autograd through the settled activations (requires_autograd=True)
            if not post_nudged.requires_grad:
                # Graph not available - return zeros
                grads.append(torch.zeros_like(geometry.params[name]))
                continue

            grad = torch.autograd.grad(
                local_obj,
                geometry.params[name],
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )[0]
            grads.append(
                grad if grad is not None else torch.zeros_like(geometry.params[name])
            )

        return grads

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

    Rate-coded surrogate: causal (pre->post) potentiation with a_plus,
    anti-causal depression with a_minus. Pseudo-gradient descends:
    -(a_plus * post^T@pre - a_minus * pre^T@post) / batch.
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
        if len(weight_names) != len(acts) - 1:
            return [torch.zeros_like(geometry.params[n]) for n in weight_names]

        batch = acts[0].shape[0]
        grads = []
        for i in range(len(weight_names)):
            pre = acts[i]  # [batch, in_dim]
            post = acts[i + 1]  # [batch, out_dim]

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
        if len(weight_names) != len(acts) - 1:
            return [torch.zeros_like(geometry.params[n]) for n in weight_names]

        # Target at output layer: one-hot of class indices
        y = nudged_state.y
        if y is None:
            return [torch.zeros_like(geometry.params[n]) for n in weight_names]

        targets = _propagate_targets(acts, y, weight_names, geometry)

        # Layer-local deltas and pseudo-gradients
        grads = []
        for i, name in enumerate(weight_names):
            tgt = targets[i + 1] if i + 1 < len(targets) else None
            if tgt is None or i + 1 >= len(acts):
                grads.append(torch.zeros_like(geometry.params[name]))
                continue
            delta = acts[i + 1] - tgt  # [batch, out]
            pre = acts[i]  # [batch, in]
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
            total = total + (delta**2).mean()
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
        if len(weight_names) != len(acts) - 1:
            return [torch.zeros_like(geometry.params[n]) for n in weight_names]

        target = self.config.homeostatic_target
        grads = []
        for i, name in enumerate(weight_names):
            post = acts[i + 1]  # [batch, out_dim]
            W = geometry.params[name]  # [out_dim, in_dim]

            # Mean post-activation norm (L2 across features, then mean across batch)
            post_norm = post.norm(dim=1).mean()  # scalar
            # Error from target
            err = post_norm - target
            # Direction: scale W by error, normalized by Frobenius norm
            W_norm = W.norm()
            if W_norm > 0:
                grad = err * W / W_norm
            else:
                grad = torch.zeros_like(W)
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
        # Return None grads as zeros of correct shape
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
        """The nudged loss is the surrogate objective for true gradients."""
        loss = nudged_state.loss
        if isinstance(loss, Tensor):
            return loss
        return torch.tensor(0.0)


# Alias for backwards compatibility
BackpropCredit = GradientCredit
