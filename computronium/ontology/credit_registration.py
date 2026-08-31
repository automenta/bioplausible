"""Credit-assignment primitives as CREDIT_ASSIGNMENT registry components.

First-class ontology-side registrations so the AutoScientist's model x
propagator composition has propagators to draw from (the deprecated zoo never
registered any).
"""

from computronium.core.registry import LocalityLevel, register_credit_assignment
from computronium.ontology.credit import (
    GradientCredit,
    HomeostaticCredit,
    LocalGoodnessCredit,
    RandomProjectionsCredit,
    TargetInversionCredit,
    TemporalTraceCredit,
    ThermodynamicContrast,
)

__all__ = [
    "GradientCredit",
    "HomeostaticCredit",
    "LocalGoodnessCredit",
    "RandomProjectionsCredit",
    "TargetInversionCredit",
    "TemporalTraceCredit",
    "ThermodynamicContrast",
]


@register_credit_assignment(
    "thermodynamic_contrast",
    family="eqprop",
    bio_plausibility_score=0.85,
    locality_level=LocalityLevel.EQUILIBRIUM,
    credit_assignment_type="equilibrium",
    requires_backward=False,
)
class ThermodynamicContrastComponent(ThermodynamicContrast):
    """EqProp contrastive credit: (nudged - free) / beta."""


@register_credit_assignment(
    "random_projections",
    family="fa",
    bio_plausibility_score=0.8,
    locality_level=LocalityLevel.FORWARD_ONLY,
    credit_assignment_type="forward-only",
    requires_backward=False,
)
class RandomProjectionsComponent(RandomProjectionsCredit):
    """Fixed random feedback matrices (FA / DFA)."""


@register_credit_assignment(
    "local_goodness",
    family="forward_only",
    bio_plausibility_score=0.9,
    locality_level=LocalityLevel.FORWARD_ONLY,
    credit_assignment_type="local",
)
class LocalGoodnessComponent(LocalGoodnessCredit):
    """Layer-local contrastive goodness (FF / PEPITA)."""


@register_credit_assignment(
    "temporal_trace",
    family="hebbian",
    bio_plausibility_score=0.95,
    locality_level=LocalityLevel.LOCAL,
    credit_assignment_type="hebbian",
    requires_backward=False,
)
class TemporalTraceComponent(TemporalTraceCredit):
    """Spike-timing correlations (STDP)."""


@register_credit_assignment(
    "target_inversion",
    family="target_prop",
    bio_plausibility_score=0.6,
    locality_level=LocalityLevel.LAYERWISE,
    credit_assignment_type="target",
)
class TargetInversionComponent(TargetInversionCredit):
    """Local target propagation."""


@register_credit_assignment(
    "homeostatic",
    family="homeostatic",
    bio_plausibility_score=0.75,
    locality_level=LocalityLevel.LOCAL,
    credit_assignment_type="local",
    requires_backward=False,
)
class HomeostaticComponent(HomeostaticCredit):
    """Autonomous Lipschitz scaling."""


@register_credit_assignment(
    "gradient",
    family="backprop",
    bio_plausibility_score=0.0,
    locality_level=LocalityLevel.GLOBAL,
    credit_assignment_type="gradient",
)
class GradientComponent(GradientCredit):
    """Full backprop baseline."""
