"""Computronium neural network layers with bio-plausible credit assignment.

Public API for drop-in PyTorch layer replacements.
"""

from __future__ import annotations

from computronium.nn.module import (
    ComputroniumLinear,
    ComputroniumLinearConfig,
    replace_linear_with_computronium,
)
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
)

__all__ = [
    "ComputroniumLinear",
    "ComputroniumLinearConfig",
    "CreditRule",
    "CreditRuleConfig",
    "FastWeightPlasticity",
    "NullPlasticity",
    "PlasticityConfig",
    "PlasticityType",
    "create_plasticity",
    "replace_linear_with_computronium",
]
