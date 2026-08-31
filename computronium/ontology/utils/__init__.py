"""Ontology utilities package."""

from computronium.ontology.utils.config import ConfigFactory
from computronium.ontology.utils.params import (
    _learnable_weight_names,
    apply_pseudo_gradients,
)

__all__ = [
    "ConfigFactory",
    "_learnable_weight_names",
    "apply_pseudo_gradients",
]
