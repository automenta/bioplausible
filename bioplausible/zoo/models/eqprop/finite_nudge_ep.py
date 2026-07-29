"""Equilibrium Propagation model variants."""

from ...base import (
    ModelConfig,
    _build_model_config,
    register_model,
)
from .standard_eqprop import StandardEqProp


@register_model(
    "finite_nudge_ep",
    family="eqprop",
    tags=["eqprop", "finite-nudge"],
)
class FiniteNudgeEP(StandardEqProp):
    """
    Finite-Nudge EqProp.
    Operates with large beta values (e.g. beta=1.0) where the infinitesimal
    approximation of the gradient is replaced by a finite difference
    that optimizes a global energy bound.
    """

    def __init__(self, config: ModelConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

        if "beta" in kwargs:
            self.beta = kwargs["beta"]
        elif self.config and self.config.extra and "beta" in self.config.extra:
            self.beta = self.config.extra["beta"]

        if self.beta < 0.5:
            self.beta = 1.0

    @classmethod
    def build(
        cls,
        spec,
        input_dim,
        output_dim,
        hidden_dim,
        num_layers,
        device,
        task_type,
        **kwargs,
    ):
        return cls(
            config=_build_model_config(
                spec, input_dim, output_dim, hidden_dim, num_layers, kwargs
            )
        ).to(device)
