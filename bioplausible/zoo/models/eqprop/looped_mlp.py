"""Equilibrium Propagation model variants.

``LoopedMLP`` is the single-registered-name facade (``eqprop_mlp``) over the
consolidated deep-eqprop engine in :mod:`bioplausible.zoo.models.eqprop._energy`.
It exists only to preserve the 3-arg positional legacy constructor used by the
validation tracks and hardware facades; the engine itself, the depth handling
(``num_layers``→real hidden layers) and the energy-contrastive rule live in
:class:`EquilibriumMLP` — there is no second architecture here.

``BackpropMLP`` is the plain feedforward MLP used as the parity baseline
against all the bio-plausible families.
"""

import math

import torch

from bioplausible.config.unified import ModelConfig
from bioplausible.core.model_status import status_tag
from bioplausible.core.ontology import System
from bioplausible.core.registry import LocalityLevel, register_model
from bioplausible.models.native.eqprop_native import native_eqprop_mlp

from ._energy import EquilibriumMLP

__all__ = [
    "LoopedMLP",
]


def _kernel_backend_step(
    engine: object,
    x: torch.Tensor,
    y: torch.Tensor,
) -> dict[str, float] | None:
    """Run a train step via the NumPy/CuPy kernel engine.

    Returns ``None`` when the engine is unavailable (caller falls back to
    the PyTorch implementation). The kernel engine is single-hidden-state
    only; layered ``LoopedMLP`` (``num_layers > 1``) must keep ``backend``
    on ``"pytorch"`` and never reach here — the consolidated engine's
    ``train_step`` path is then used instead.
    """
    if engine is None:
        return None
    x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y
    if x_np.ndim > 2:
        x_np = x_np.reshape(x_np.shape[0], -1)
    return engine.train_step(x_np, y_np)


# Register native eqprop_mlp factory (bypasses ModelAdapter for 5-D composition)
@register_model(
    "eqprop_mlp",
    locality_level=LocalityLevel.EQUILIBRIUM,
    bio_plausibility_score=0.9,
    credit_assignment_type="equilibrium",
    requires_backward=False,
    memory_complexity="O(1)",
    family="eqprop",
    typical_lr_range=(0.001, 0.01),
    tags=["eqprop", "looped_mlp", "equilibrium", status_tag("stable")],
    extra={"parity_threshold": 0.05},
)
def _native_eqprop_mlp_factory(**kwargs) -> System:
    return native_eqprop_mlp(**kwargs)


class LoopedMLP(EquilibriumMLP):
    """Canonical eqprop MLP — alias of :class:`EquilibriumMLP`.

    The registration metadata lives here (rather than on ``EquilibriumMLP``
    itself) so the historic search-space key ``"eqprop_mlp`` keeps resolving
    to the consolidated layered engine. Architecture, depth handling
    (``num_layers``→real hidden layers), and energy-contrastive update are
    inherited unmodified: there is no separate "looped" implementation, this
    is the unified eqprop MLP seen under the old name.

    The positional 3-arg constructor ``LoopedMLP(input_dim, hidden_dim,
    output_dim, ...)`` historically used by the validation/hardware tracks is
    preserved by translating it into a ``ModelConfig``; ``config=`` remains
    the preferred entrypoint for the construction layer.

    .. deprecated:: 0.1
       Use native 5-D composition via ``Registry.to_system("eqprop_mlp")`` instead.
       This class is kept for backward compatibility with validation tracks.
    """

    variant = "plain"  # type: ignore[assignment]

    def __init__(
        self,
        input_dim: int | tuple[int, ...] | None = None,
        hidden_dim: int | None = None,
        output_dim: int | None = None,
        num_layers: int = 1,
        *,
        config: ModelConfig | None = None,
        use_spectral_norm: bool = True,
        max_steps: int = 30,
        gradient_method: str = "equilibrium",
        backend: str = "pytorch",
        **kwargs,
    ) -> None:
        if config is None:
            if input_dim is None or hidden_dim is None or output_dim is None:
                raise ValueError(
                    "LoopedMLP positional init requires input_dim, hidden_dim and "
                    "output_dim (or pass config=ModelConfig(...))."
                )
            if isinstance(input_dim, tuple):
                input_dim = math.prod(input_dim)
            config = ModelConfig(
                name="eqprop_mlp",
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=[hidden_dim] * max(num_layers, 1),
                use_spectral_norm=use_spectral_norm,
                max_steps=max_steps,
                extra={
                    "gradient_method": gradient_method,
                    "backend": backend,
                    **kwargs,
                },
            )
        # ``backend`` is recorded for trainer compatibility (the kernel backend
        # used to live here); the consolidated layered engine ignores it and
        # always runs through ``settle_activations_list``. Read it from the
        # config's ``extra`` when constructed via the canonical trainer path
        # (where ``backend`` lands in ``ModelConfig.extra``), falling back to
        # the explicit kwarg.
        self.backend = str(config.extra.get("backend", backend))
        super().__init__(config=config, **kwargs)

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        """Route to the correct training path.

        * ``gradient_method="equilibrium"`` + single hidden layer → return ``None``
          so the trainer's EnergyModel path (Phase 1) runs the O(1)-memory
          implicit equilibrium backward — the historic fast path.
        * ``gradient_method="contrastive"`` or multi-layer → run the consolidated
          contrastive free/nudged step via ``EquilibriumMLP.train_step``.
        """
        if self.gradient_method == "equilibrium" and len(self._hidden_dims()) == 1:
            # Single-hidden eqprop MLPs keep the O(1)-memory implicit
            # differentiation backward (memory flat in settle steps).
            return None  # type: ignore[return-value]
        if self.backend == "kernel":
            metrics = _kernel_backend_step(getattr(self, "_engine", None), x, y)
            if metrics is not None:
                return metrics
        return super().train_step(x, y)
