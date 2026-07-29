"""Equilibrium Propagation model variants."""

from ...base import ModelConfig, compute_hidden_dims, register_model
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
        config = ModelConfig(
            name=spec.name,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=compute_hidden_dims(hidden_dim, num_layers),
            extra=kwargs,
        )

        if "equilibrium_steps" in kwargs:
            config.equilibrium_steps = kwargs["equilibrium_steps"]
            config.max_steps = kwargs["equilibrium_steps"]
        if "beta" in kwargs:
            config.beta = kwargs["beta"]

        return cls(config=config).to(device)
