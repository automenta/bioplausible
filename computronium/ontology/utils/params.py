"""Shared parameter utilities for ontology modules."""

from collections.abc import Callable

from torch import Tensor


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
    transform: Callable[[str, Tensor, Tensor], Tensor],
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
        param = params[name]
        updated[name] = transform(name, param, grad.detach().to(param.device))
    return updated
