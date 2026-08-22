"""Plasticity Primitives: Non-null plasticity laws for the joint architecture."""

from __future__ import annotations

from bioplausible.core.joint.transition import (
    NullPlasticity,
    PlasticityConfig,
    PlasticityPrimitive,
)

__all__ = [
    "PlasticityConfig",
    "PlasticityPrimitive",
    "NullPlasticity",
]