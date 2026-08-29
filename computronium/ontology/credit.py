"""Layer 4: CreditAssignment — Error Routing & Pseudo-Gradients."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

import torch
from torch import Tensor

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
    """

    credit_type: str
    beta: float
    feedback_matrix: Tensor | None
    local_objective: str
    orthogonal_init: bool
    feedback_scale: float

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
    ) -> CreditAssignmentConfig:
        return cls(
            credit_type="temporal_trace",
            beta=beta,
            feedback_matrix=feedback_matrix,
            local_objective=local_objective,
            orthogonal_init=orthogonal_init,
            feedback_scale=feedback_scale,
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


# ============================================================
# Default/Reference CreditAssignment Implementations
# ============================================================


def _learnable_weight_names(params: dict[str, Tensor]) -> list[str]:
    """Parameter names that receive pseudo-gradients (2-D weight matrices).

    Credits emit exactly one pseudo-gradient per learnable weight, in this
    order. Biases and other auxiliary parameters never receive gradients
    from the local learning rules.
    """
    return [n for n, p in params.items() if "weight" in n and p.ndim == 2]


def apply_pseudo_gradients(
    params: dict[str, Tensor],
    pseudo_grads: list[Tensor],
    transform,
) -> dict[str, Tensor]:
    """Pair pseudo-gradients with their parameters by learnable-weight order.

    The single choke point for update rules: non-weight parameters pass
    through untouched (fixes the index-pairing crash on bias interleaving),
    surplus gradients are ignored, and gradients are detached — pseudo-
    gradients are consumed as plain values everywhere in this pipeline.

    Args:
        params: Current parameters (name -> tensor).
        pseudo_grads: One pseudo-gradient per learnable weight.
        transform: ``(name, param, grad) -> updated_param`` for matched pairs.
    """
    updated = dict(params)
    for name, grad in zip(_learnable_weight_names(params), pseudo_grads):
        updated[name] = transform(name, params[name], grad.detach())
    return updated


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

    def compute_pseudo_gradient(
        self,
        states: Mapping[Phase, SystemState],
        loss: Tensor | None,
        geometry: Geometry,
    ) -> list[Tensor]:
        # Simplified: return random projections of loss gradient
        free_state = states.get(Phase.FREE)
        nudged_state = states.get(Phase.NUDGED)
        if free_state is None or nudged_state is None:
            return []

        weight_names = _learnable_weight_names(geometry.params)
        grads = []
        for _ in weight_names:
            # Random projection pseudo-gradient
            grads.append(torch.randn(1))
        return grads


class LocalGoodnessCredit:
    """Layer-local contrastive objective (Forward-Forward, PEPITA)."""

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
        # Simplified
        return []


class TemporalTraceCredit:
    """Spike-timing correlations (STDP)."""

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
        # Simplified
        return []


class TargetInversionCredit:
    """Propagating local targets (Target Prop)."""

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
        # Simplified
        return []


class HomeostaticCredit:
    """Homeostatic credit assignment."""

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
        # Simplified
        return []


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
        # Simplified
        return []


# Alias for backwards compatibility
BackpropCredit = GradientCredit
