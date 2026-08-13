"""Spectral mixin — Lipschitz and spectral normalization utilities.

Consolidates the spectral norm computation and power iteration logic that was
duplicated across ``core/model.py`` and ``equitile/core/model.py``.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

from bioplausible.core.utils.activations import approx_spectral_norm
from bioplausible.utils import count_parameters


class SpectralMixin:
    """Mixin providing spectral normalization and Lipschitz computation.

    Expected attributes on the host class:
        - ``use_spectral_norm: bool``
        - ``config.lipschitz_mode: str`` (``"power_iteration"`` or ``"svd"``)
        - ``config.output_scaling_mode: str`` (``"mupc"`` or ``"uniform"``)
        - ``spectral_norm_power_iterations: int``
        - ``config: ModelConfig`` (with ``output_scaling_mode`` field)
    """

    def apply_spectral_norm(
        self,
        layer: nn.Module,
        layer_role: str = "hidden",
    ) -> nn.Module:
        """Apply spectral normalization to a linear or conv layer.

        When ``output_scaling_mode == "mupc"`` and the layer is an output
        ``nn.Linear``, the weight is rescaled to remove the √fan_in factor
        from kaiming initialization.

        Args:
            layer: The layer to normalize.
            layer_role: ``"hidden"`` or ``"output"``.

        Returns:
            The normalized layer (wrapped or as-is).
        """
        if self.use_spectral_norm and isinstance(layer, (nn.Linear, nn.Conv2d)):
            layer = spectral_norm(
                layer, n_power_iterations=self.spectral_norm_power_iterations
            )
            if (
                self.config.output_scaling_mode == "mupc"
                and layer_role == "output"
                and isinstance(layer, nn.Linear)
            ):
                fan_in = layer.weight.size(1)
                if fan_in > 0:
                    with torch.no_grad():
                        gain = nn.init.calculate_gain("linear")
                        std = gain * (2.0 / fan_in) ** 0.5
                        target_std = gain
                        layer.weight.mul_(target_std / max(std, 1e-12))
            return layer
        return layer

    def _get_spectral_normalized_weight(self, layer: nn.Module) -> torch.Tensor:
        """Get spectral normalized weight, cached in eval mode for inference."""
        if torch.is_grad_enabled():
            weight = layer.weight
            return weight

        if not self.training and hasattr(layer, "_cached_sn_weight"):
            return layer._cached_sn_weight

        weight = layer.weight

        if not self.training:
            layer._cached_sn_weight = weight.detach()

        return weight

    def train(self: nn.Module, mode: bool = True) -> nn.Module:
        """Override train to clear spectral norm caches."""
        super().train(mode)  # type: ignore[misc]
        if mode:
            for module in self.modules():
                if hasattr(module, "_cached_sn_weight"):
                    delattr(module, "_cached_sn_weight")
        return self

    def compute_lipschitz(self) -> float:
        """Compute the maximum Lipschitz constant across all layers."""
        max_L = 0.0
        with torch.no_grad():
            for module in self.modules():
                if hasattr(module, "weight") and isinstance(
                    module.weight, torch.Tensor
                ):
                    w = module.weight
                    if w.dim() >= 2:
                        if self.config.lipschitz_mode == "power_iteration":
                            L = approx_spectral_norm(w, n_iter=10)
                        else:
                            w_mat = w.view(w.size(0), -1)
                            s = torch.linalg.svdvals(w_mat)
                            L = s[0].item() if s.numel() > 0 else 0.0
                        max_L = max(max_L, L)
        return max_L

    def get_stats(self) -> dict[str, float]:
        """Get algorithm-specific statistics for reporting."""
        return {
            "lipschitz": self.compute_lipschitz(),
            "num_params": count_parameters(
                cast("nn.Module", self), trainable_only=False
            ),
            "spectral_norm": self.use_spectral_norm,
        }
