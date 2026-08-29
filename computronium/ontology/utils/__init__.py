"""Ontology utilities package."""

from computronium.ontology.utils.config import ConfigFactory
from computronium.ontology.utils.geometry import _layer_stack, _recurrent_weight
from computronium.ontology.utils.params import (
    _learnable_weight_names,
    _set_param_name,
    apply_pseudo_gradients,
)
from computronium.ontology.utils.state import (
    StateProtocol,
    _get_state_activations,
    _get_state_activity,
    _get_state_free_state,
    _get_state_loss,
    _get_state_metrics,
    _get_state_nudged_state,
    _get_state_x,
    _is_composite_state,
    _set_state_activations,
    _set_state_free_state,
    _set_state_loss,
    _set_state_metrics,
    _set_state_nudged_state,
    _set_state_x,
)

__all__ = [
    "ConfigFactory",
    "StateProtocol",
    "_layer_stack",
    "_recurrent_weight",
    "_learnable_weight_names",
    "_set_param_name",
    "apply_pseudo_gradients",
    "_is_composite_state",
    "_get_state_x",
    "_get_state_activations",
    "_get_state_free_state",
    "_get_state_nudged_state",
    "_get_state_loss",
    "_get_state_metrics",
    "_get_state_activity",
    "_set_state_x",
    "_set_state_activations",
    "_set_state_free_state",
    "_set_state_nudged_state",
    "_set_state_loss",
    "_set_state_metrics",
]
