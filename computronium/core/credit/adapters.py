"""Cross-Credit Adapters.

Enables translation between different credit assignment paradigms,
allowing hybrid local/global credit, comparison of learning rules,
and adapter-based composition of credit mechanisms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import torch
from torch import Tensor

from computronium.core.ontology import (
    BackpropCredit,
    CreditAssignment,
    CreditAssignmentConfig,
    Geometry,
    HomeostaticCredit,
    LocalGoodnessCredit,
    Phase,
    RandomProjectionsCredit,
    SystemState,
    TargetInversionCredit,
    TemporalTraceCredit,
    ThermodynamicContrast,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

# ============================================================
# Base Adapter Class
# ============================================================


class CreditAdapter:
    """Base class for cross-credit adapters.

    Wraps a source credit assignment and emulates target credit behavior.

    Capabilities cover the hybrid by default (both declared phases,
    no autograd); backprop-involving hybrids override ``requires_autograd``.
    """

    phases: ClassVar[tuple[Phase, ...]] = (Phase.FREE, Phase.NUDGED)
    requires_autograd: ClassVar[bool] = False

    def __init__(
        self,
        source_credit: CreditAssignment,
        target_config: CreditAssignmentConfig | None = None,
    ):
        self._source = source_credit
        self._target_config = target_config or source_credit.config
        self.config = self._target_config

    @property
    def source_credit(self) -> CreditAssignment:
        return self._source

    @property
    def target_config(self) -> CreditAssignmentConfig:
        return self._target_config

    def compute_pseudo_gradient(
        self,
        states: Mapping[Phase, SystemState],
        loss: Tensor | None,
        geometry: Geometry,
    ) -> list[Tensor]:
        return self._source.compute_pseudo_gradient(states, loss, geometry)

    def surrogate_objective(
        self,
        free_state: SystemState,
        nudged_state: SystemState,
        geometry: Geometry,
    ) -> Tensor:
        return self._source.surrogate_objective(free_state, nudged_state, geometry)


# ============================================================
# ThermodynamicContrast -> Backprop (EqProp to BPTT Comparison)
# ============================================================


class ThermodynamicToBackpropAdapter(CreditAdapter):
    requires_autograd: ClassVar[bool] = True
    """Compare local EqProp gradients with global backprop gradients.

    Computes both contrastive Hebbian (EqProp) and autograd (backprop)
    gradients, enabling direct comparison and hybrid credit.

    Source: ThermodynamicContrast (equilibrium contrast)
    Target: BackpropCredit (global autograd)
    """

    def __init__(
        self,
        source_credit: ThermodynamicContrast,
        target_config: CreditAssignmentConfig | None = None,
        *,
        hybrid_weight: float = 0.5,  # Weight for hybrid: 0=pure EqProp, 1=pure backprop
        compute_both: bool = True,
    ):
        target_config = target_config or CreditAssignmentConfig.gradient()
        super().__init__(source_credit, target_config)
        self._hybrid_weight = hybrid_weight
        self._compute_both = compute_both
        self._backprop_credit = BackpropCredit(target_config)
        self._last_comparison: dict[str, dict] = {}

    def compute_pseudo_gradient(
        self,
        states: Mapping[Phase, SystemState],
        loss: Tensor | None,
        geometry: Geometry,
    ) -> list[Tensor]:
        """Compute both EqProp and backprop gradients."""
        # EqProp pseudo-gradient
        eqprop_grads = self._source.compute_pseudo_gradient(states, loss, geometry)

        if not self._compute_both:
            return eqprop_grads

        # Backprop gradients (requires autograd)
        backprop_grads = self._backprop_credit.compute_pseudo_gradient(
            states, loss, geometry
        )

        # Store comparison for analysis
        self._last_comparison = self._compare_gradients(eqprop_grads, backprop_grads)

        # Hybrid combination
        if self._hybrid_weight > 0 and self._hybrid_weight < 1:
            return self._combine_gradients(eqprop_grads, backprop_grads)
        if self._hybrid_weight >= 1:
            return backprop_grads
        return eqprop_grads

    @staticmethod
    def _compare_gradients(
        eqprop: list[Tensor], backprop: list[Tensor]
    ) -> dict[str, dict]:
        """Compare gradient magnitudes and angles."""
        comparison = {}
        for i, (eq, bp) in enumerate(zip(eqprop, backprop, strict=False)):
            if eq.shape == bp.shape and eq.numel() > 0:
                # Cosine similarity
                eq_flat = eq.flatten()
                bp_flat = bp.flatten()
                cos_sim = torch.nn.functional.cosine_similarity(
                    eq_flat.unsqueeze(0), bp_flat.unsqueeze(0)
                ).item()
                # Relative magnitude
                mag_ratio = (eq.norm() / (bp.norm() + 1e-8)).item()
                comparison[f"layer_{i}"] = {
                    "cosine_similarity": cos_sim,
                    "magnitude_ratio": mag_ratio,
                    "eqprop_norm": eq.norm().item(),
                    "backprop_norm": bp.norm().item(),
                }
        return comparison

    @staticmethod
    def _combine_gradients(
        eqprop: list[Tensor], backprop: list[Tensor]
    ) -> list[Tensor]:
        """Hybrid combination of gradients."""
        combined = []
        for eq, bp in zip(eqprop, backprop, strict=False):
            if eq.shape == bp.shape:
                combined.append((1 - 0.5) * eq + 0.5 * bp)
            else:
                combined.append(eq)
        return combined

    def get_gradient_comparison(self) -> dict[str, dict]:
        """Return last gradient comparison results."""
        return self._last_comparison


# ============================================================
# RandomProjections -> ThermodynamicContrast (FA to EqProp Hybrid)
# ============================================================


class RandomProjectionsToThermodynamicAdapter(CreditAdapter):
    """Hybrid Feedback Alignment + Equilibrium Propagation.

    Uses fixed random feedback matrices for hidden layers and
    equilibrium contrast for output layer, combining local and
    global credit signals.

    Source: RandomProjectionsCredit (FA/DFA)
    Target: ThermodynamicContrast (EqProp)
    """

    def __init__(
        self,
        source_credit: RandomProjectionsCredit,
        target_config: CreditAssignmentConfig | None = None,
        *,
        eqprop_weight: float = 0.5,  # Weight for EqProp component
        use_fa_for_hidden: bool = True,
        use_eqprop_for_output: bool = True,
    ):
        target_config = target_config or CreditAssignmentConfig.thermodynamic_contrast()
        super().__init__(source_credit, target_config)
        self._eqprop_weight = eqprop_weight
        self._use_fa_for_hidden = use_fa_for_hidden
        self._use_eqprop_for_output = use_eqprop_for_output
        self._thermodynamic_credit = ThermodynamicContrast(target_config)

    def compute_pseudo_gradient(
        self,
        states: Mapping[Phase, SystemState],
        loss: Tensor | None,
        geometry: Geometry,
    ) -> list[Tensor]:
        """Hybrid FA + EqProp pseudo-gradients."""
        # FA gradients for all layers
        fa_grads = self._source.compute_pseudo_gradient(states, loss, geometry)

        if not self._use_eqprop_for_output:
            return fa_grads

        # EqProp gradients
        eqprop_grads = self._thermodynamic_credit.compute_pseudo_gradient(
            states, loss, geometry
        )

        # Combine: FA for hidden, EqProp for output (or weighted blend)
        combined = []
        n_layers = len(fa_grads)
        for i, (fa, eq) in enumerate(zip(fa_grads, eqprop_grads, strict=False)):
            if i == n_layers - 1 and self._use_eqprop_for_output:
                # Output layer: use EqProp
                combined.append(eq)
            elif self._use_fa_for_hidden:
                # Hidden layers: blend or use FA
                if self._eqprop_weight > 0 and eq.shape == fa.shape:
                    combined.append(
                        (1 - self._eqprop_weight) * fa + self._eqprop_weight * eq
                    )
                else:
                    combined.append(fa)
            else:
                combined.append(fa)

        return combined


# ============================================================
# LocalGoodness -> ThermodynamicContrast (FF/PEPITA to EqProp Hybrid)
# ============================================================


class LocalGoodnessToThermodynamicAdapter(CreditAdapter):
    """Hybrid Forward-Forward/PEPITA + Equilibrium Propagation.

    Combines layer-local goodness objectives with global
    equilibrium contrast for improved credit assignment.

    Source: LocalGoodnessCredit (FF/PEPITA)
    Target: ThermodynamicContrast (EqProp)
    """

    def __init__(
        self,
        source_credit: LocalGoodnessCredit,
        target_config: CreditAssignmentConfig | None = None,
        *,
        eqprop_weight: float = 0.5,
        layer_weights: list[float] | None = None,
    ):
        target_config = target_config or CreditAssignmentConfig.thermodynamic_contrast()
        super().__init__(source_credit, target_config)
        self._eqprop_weight = eqprop_weight
        self._layer_weights = layer_weights
        self._thermodynamic_credit = ThermodynamicContrast(target_config)

    def compute_pseudo_gradient(
        self,
        states: Mapping[Phase, SystemState],
        loss: Tensor | None,
        geometry: Geometry,
    ) -> list[Tensor]:
        """Hybrid local goodness + EqProp gradients."""
        # Local goodness gradients
        lg_grads = self._source.compute_pseudo_gradient(states, loss, geometry)

        # EqProp gradients
        eqprop_grads = self._thermodynamic_credit.compute_pseudo_gradient(
            states, loss, geometry
        )

        # Combine per-layer
        combined = []
        for i, (lg, eq) in enumerate(zip(lg_grads, eqprop_grads, strict=False)):
            if lg.shape == eq.shape:
                w = (
                    self._layer_weights[i]
                    if self._layer_weights
                    else self._eqprop_weight
                )
                combined.append((1 - w) * lg + w * eq)
            else:
                combined.append(eq if eq.numel() > 0 else lg)

        return combined

    def surrogate_objective(
        self,
        free_state: SystemState,
        nudged_state: SystemState,
        geometry: Geometry,
    ) -> Tensor:
        """Combined surrogate: local goodness + energy difference."""
        lg_obj = self._source.surrogate_objective(free_state, nudged_state, geometry)
        eq_obj = self._thermodynamic_credit.surrogate_objective(
            free_state, nudged_state, geometry
        )
        return (1 - self._eqprop_weight) * lg_obj + self._eqprop_weight * eq_obj


# ============================================================
# ThermodynamicContrast -> HomeostaticCredit (EqProp + Homeostasis)
# ============================================================


class ThermodynamicToHomeostaticAdapter(CreditAdapter):
    """Equilibrium Propagation with homeostatic stability control.

    Adds dynamic Lipschitz estimation and velocity-based braking/boosting
    to standard EqProp for autonomous stability.

    Source: ThermodynamicContrast (EqProp)
    Target: HomeostaticCredit (EqProp + homeostasis)
    """

    def __init__(
        self,
        source_credit: ThermodynamicContrast,
        target_config: CreditAssignmentConfig | None = None,
    ):
        target_config = target_config or CreditAssignmentConfig(
            credit_type="homeostatic",
            beta=source_credit.config.beta,
            feedback_matrix=None,
            local_objective="mse",
            orthogonal_init=False,
            feedback_scale=0.01,
        )
        super().__init__(source_credit, target_config)
        self._homeostatic_credit = HomeostaticCredit(target_config)

    def compute_pseudo_gradient(
        self,
        states: Mapping[Phase, SystemState],
        loss: Tensor | None,
        geometry: Geometry,
    ) -> list[Tensor]:
        """EqProp gradients with homeostatic scaling."""
        # Apply homeostatic scaling
        scaled_grads = self._homeostatic_credit.compute_pseudo_gradient(
            states, loss, geometry
        )

        return scaled_grads

    def apply_homeostasis(self, geometry: Geometry) -> dict:
        """Apply homeostatic adaptation after gradient computation."""
        return self._homeostatic_credit.apply_homeostasis(geometry)

    def get_stability_report(self) -> str:
        """Get homeostatic stability report."""
        return self._homeostatic_credit.get_stability_report()


# ============================================================
# TemporalTrace -> ThermodynamicContrast (STDP + EqProp)
# ============================================================


class TemporalTraceToThermodynamicAdapter(CreditAdapter):
    """Spike-Timing-Dependent Plasticity with Equilibrium Contrast.

    Combines STDP spike correlations with EqProp contrastive
    learning for spiking equilibrium networks.

    Source: TemporalTraceCredit (STDP)
    Target: ThermodynamicContrast (EqProp)
    """

    def __init__(
        self,
        source_credit: TemporalTraceCredit,
        target_config: CreditAssignmentConfig | None = None,
        *,
        stdp_weight: float = 0.5,
    ):
        target_config = target_config or CreditAssignmentConfig.thermodynamic_contrast()
        super().__init__(source_credit, target_config)
        self._stdp_weight = stdp_weight
        self._thermodynamic_credit = ThermodynamicContrast(target_config)

    def compute_pseudo_gradient(
        self,
        states: Mapping[Phase, SystemState],
        loss: Tensor | None,
        geometry: Geometry,
    ) -> list[Tensor]:
        """Hybrid STDP + EqProp gradients."""
        # STDP gradients (requires spike data)
        stdp_grads = self._source.compute_pseudo_gradient(states, loss, geometry)

        # EqProp gradients
        eqprop_grads = self._thermodynamic_credit.compute_pseudo_gradient(
            states, loss, geometry
        )

        # Combine
        combined = []
        for stdp, eq in zip(stdp_grads, eqprop_grads, strict=False):
            if stdp.shape == eq.shape:
                combined.append(self._stdp_weight * stdp + (1 - self._stdp_weight) * eq)
            elif stdp.numel() > 0:
                combined.append(stdp)
            else:
                combined.append(eq)

        return combined


# ============================================================
# TargetInversion -> ThermodynamicContrast (Target Prop + EqProp)
# ============================================================


class TargetInversionToThermodynamicAdapter(CreditAdapter):
    """Target Propagation with Equilibrium Contrast.

    Uses learned inverse maps for layer-wise target propagation
    combined with global equilibrium contrast.

    Source: TargetInversionCredit (Target Prop)
    Target: ThermodynamicContrast (EqProp)
    """

    def __init__(
        self,
        source_credit: TargetInversionCredit,
        target_config: CreditAssignmentConfig | None = None,
        *,
        eqprop_weight: float = 0.5,
    ):
        target_config = target_config or CreditAssignmentConfig.thermodynamic_contrast()
        super().__init__(source_credit, target_config)
        self._eqprop_weight = eqprop_weight
        self._thermodynamic_credit = ThermodynamicContrast(target_config)

    def compute_pseudo_gradient(
        self,
        states: Mapping[Phase, SystemState],
        loss: Tensor | None,
        geometry: Geometry,
    ) -> list[Tensor]:
        """Hybrid Target Prop + EqProp gradients."""
        # Target Prop gradients
        tp_grads = self._source.compute_pseudo_gradient(states, loss, geometry)

        # EqProp gradients
        eqprop_grads = self._thermodynamic_credit.compute_pseudo_gradient(
            states, loss, geometry
        )

        # Combine
        combined = []
        for tp, eq in zip(tp_grads, eqprop_grads, strict=False):
            if tp.shape == eq.shape:
                combined.append(
                    self._eqprop_weight * eq + (1 - self._eqprop_weight) * tp
                )
            elif tp.numel() > 0:
                combined.append(tp)
            else:
                combined.append(eq)

        return combined


# ============================================================
# Backprop -> ThermodynamicContrast (Backprop as Teacher)
# ============================================================


class BackpropToThermodynamicAdapter(CreditAdapter):
    requires_autograd: ClassVar[bool] = True
    """Backprop-guided Equilibrium Propagation.

    Uses backprop gradients as a teacher signal to guide
    the equilibrium contrastive learning, useful for
    distillation and initialization.

    Source: BackpropCredit (global gradients)
    Target: ThermodynamicContrast (EqProp)
    """

    def __init__(
        self,
        source_credit: BackpropCredit,
        target_config: CreditAssignmentConfig | None = None,
        *,
        distillation_weight: float = 0.3,
    ):
        target_config = target_config or CreditAssignmentConfig.thermodynamic_contrast()
        super().__init__(source_credit, target_config)
        self._distillation_weight = distillation_weight
        self._thermodynamic_credit = ThermodynamicContrast(target_config)

    def compute_pseudo_gradient(
        self,
        states: Mapping[Phase, SystemState],
        loss: Tensor | None,
        geometry: Geometry,
    ) -> list[Tensor]:
        """Backprop-distilled EqProp gradients."""
        # Backprop gradients (teacher)
        bp_grads = self._source.compute_pseudo_gradient(states, loss, geometry)

        # EqProp gradients (student)
        eqprop_grads = self._thermodynamic_credit.compute_pseudo_gradient(
            states, loss, geometry
        )

        # Distill: blend with backprop as regularization
        combined = []
        for bp, eq in zip(bp_grads, eqprop_grads, strict=False):
            if bp.shape == eq.shape:
                combined.append(
                    (1 - self._distillation_weight) * eq
                    + self._distillation_weight * bp
                )
            else:
                combined.append(eq)

        return combined


# ============================================================
# Factory Function
# ============================================================


def create_credit_adapter(
    source_type: str,
    target_type: str,
    source_credit: CreditAssignment,
    target_config: CreditAssignmentConfig | None = None,
    **kwargs,
) -> CreditAdapter:
    """Factory for cross-credit adapters.

    Args:
        source_type: Source credit ("thermodynamic_contrast", "random_projections",
                     "local_goodness", "temporal_trace", "target_inversion", "gradient")
        target_type: Target credit
        source_credit: Source credit assignment instance
        target_config: Optional target config
        **kwargs: Adapter-specific parameters

    Returns:
        Configured CreditAdapter instance
    """
    adapter_map: dict[tuple[str, str], type[CreditAdapter]] = {
        ("thermodynamic_contrast", "gradient"): ThermodynamicToBackpropAdapter,
        (
            "random_projections",
            "thermodynamic_contrast",
        ): RandomProjectionsToThermodynamicAdapter,
        (
            "local_goodness",
            "thermodynamic_contrast",
        ): LocalGoodnessToThermodynamicAdapter,
        ("thermodynamic_contrast", "homeostatic"): ThermodynamicToHomeostaticAdapter,
        (
            "temporal_trace",
            "thermodynamic_contrast",
        ): TemporalTraceToThermodynamicAdapter,
        (
            "target_inversion",
            "thermodynamic_contrast",
        ): TargetInversionToThermodynamicAdapter,
        ("gradient", "thermodynamic_contrast"): BackpropToThermodynamicAdapter,
    }

    key = (source_type, target_type)
    if key not in adapter_map:
        available = list(adapter_map.keys())
        msg = f"No adapter for {source_type} -> {target_type}. Available: {available}"
        raise ValueError(msg)

    adapter_class = adapter_map[key]
    return adapter_class(source_credit, target_config, **kwargs)


__all__ = [
    "BackpropToThermodynamicAdapter",
    "CreditAdapter",
    "LocalGoodnessToThermodynamicAdapter",
    "RandomProjectionsToThermodynamicAdapter",
    "TargetInversionToThermodynamicAdapter",
    "TemporalTraceToThermodynamicAdapter",
    "ThermodynamicToBackpropAdapter",
    "ThermodynamicToHomeostaticAdapter",
    "create_credit_adapter",
]
