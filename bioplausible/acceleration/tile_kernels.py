"""Tile Substrate Kernel Backend.

Tile-parallel contrastive kernels extending core/tile/kernels.py.
"""

from __future__ import annotations

import torch
from torch import Tensor

from bioplausible.acceleration.contrastive_primitives import (
    contrastive_hebbian_update,
)
from bioplausible.acceleration.kernel_backend import (
    AlgorithmFamily,
    HardwareTarget,
    KernelConfig,
    KernelRegistry,
    LocalityLevel,
)


class TileKernelBackend:
    """Tile substrate kernel backend.

    Implements tile-parallel contrastive learning:
    - Each tile is a local compute unit
    - Tiles communicate via message passing
    - Contrastive Hebbian updates per tile
    """

    name = AlgorithmFamily.TILE
    supported_dtypes = (torch.float32, torch.float16, torch.bfloat16)
    supports_autograd = False
    requires_settle = True
    memory_complexity = "O(1)"  # Per-tile O(1), global O(tiles)
    locality_level = LocalityLevel.LOCAL

    def __init__(self) -> None:
        self._config: KernelConfig | None = None
        self._num_tiles: int = 0
        self._neurons_per_tile: int = 0
        self._tiles_per_layer: int = 0
        self._num_hidden_layers: int = 0
        self._beta: float = 0.5
        self._lr: float = 0.01
        self._device: torch.device = torch.device("cpu")
        self._dtype: torch.dtype = torch.float32

    def initialize(self, config: KernelConfig) -> None:
        self._config = config
        self._device = torch.device(
            "cuda"
            if config.hardware in (HardwareTarget.CUDA, HardwareTarget.TRITON)
            else "cpu"
        )
        self._dtype = config.dtype

        extra = config.extra
        self._neurons_per_tile = extra.get("neurons_per_tile", 32)
        self._tiles_per_layer = extra.get("tiles_per_layer", 8)
        self._num_hidden_layers = extra.get("num_hidden_layers", 3)
        self._beta = config.beta
        self._lr = config.extra.get("learning_rate", 0.01)

        self._num_tiles = self._tiles_per_layer * self._num_hidden_layers

    def set_model_ref(self, tile_algorithm) -> None:
        """Set reference to TileAlgorithm instance."""
        self._tile_algo = tile_algorithm

    def tile_forward(
        self,
        x: Tensor,
        tile_states: list[Tensor] | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        """Forward pass through tile substrate.

        Args:
            x: Input [B, D_in]
            tile_states: Optional per-tile states [num_tiles, B, neurons_per_tile]

        Returns:
            (output, tile_states)
        """
        x = x.to(device=self._device, dtype=self._dtype)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        batch_size = x.shape[0]

        if tile_states is None:
            # Initialize tile states
            tile_states = [
                torch.zeros(
                    batch_size,
                    self._neurons_per_tile,
                    device=self._device,
                    dtype=self._dtype,
                )
                for _ in range(self._num_tiles)
            ]

        # Input projection to first layer tiles
        # (simplified - full implementation uses tile topology)
        current_acts = x

        for layer_idx in range(self._num_hidden_layers):
            layer_tiles = tile_states[
                layer_idx * self._tiles_per_layer : (layer_idx + 1)
                * self._tiles_per_layer
            ]

            # Process each tile in parallel (batched)
            new_tile_states = []
            for tile_idx, tile_state in enumerate(layer_tiles):
                # Tile computation: local update
                # In practice, this uses the tile's weight matrix
                new_state = self._tile_local_update(
                    tile_state, current_acts, layer_idx, tile_idx
                )
                new_tile_states.append(new_state)

            tile_states[
                layer_idx * self._tiles_per_layer : (layer_idx + 1)
                * self._tiles_per_layer
            ] = new_tile_states

            # Pool across tiles for next layer input
            current_acts = torch.cat(new_tile_states, dim=1)

        return current_acts, tile_states

    def _tile_local_update(
        self,
        tile_state: Tensor,
        input_acts: Tensor,
        layer_idx: int,
        tile_idx: int,
    ) -> Tensor:
        """Single tile local update (simplified)."""
        # This would use the tile's actual weights
        # For now, simple linear transform
        return torch.tanh(tile_state + input_acts.mean(dim=1, keepdim=True))

    def settle(
        self,
        x: Tensor,
        beta: float = 0.0,
        steps: int = 10,
    ) -> tuple[list[Tensor], dict[str, float]]:
        """Settle tile substrate to equilibrium.

        Args:
            x: Input
            beta: Nudge strength (0 = free, >0 = nudged)
            steps: Settle steps

        Returns:
            (final_tile_states, telemetry)
        """
        tile_states = None
        telemetry = {"steps": steps, "converged": False, "final_delta": 0.0}
        prev_tile_states: list[Tensor] | None = None

        for step in range(steps):
            output, tile_states = self.tile_forward(x, tile_states)

            # Check convergence (simplified)
            if step > 0 and prev_tile_states is not None:
                delta = sum(
                    (s - prev_s).abs().max().item()
                    for s, prev_s in zip(tile_states, prev_tile_states)
                )
                telemetry["final_delta"] = delta
                if delta < 1e-4:
                    telemetry["converged"] = True
                    telemetry["steps"] = step + 1
                    break

            prev_tile_states = [s.clone() for s in tile_states]

        return tile_states, telemetry

    def backward_contrastive(
        self,
        free_states: list[Tensor],
        nudged_states: list[Tensor],
    ) -> dict[str, Tensor]:
        """Contrastive Hebbian update for tile substrate.

        Uses core/tile/kernels.py compute_contrastive_hebbian_update.

        Returns:
            Weight deltas per tile
        """
        weight_deltas: dict[str, Tensor] = {}

        for layer_idx in range(self._num_hidden_layers):
            for tile_idx in range(self._tiles_per_layer):
                idx = layer_idx * self._tiles_per_layer + tile_idx

                free_pre = (
                    free_states[idx] if idx < len(free_states) else free_states[-1]
                )
                free_post = free_states[idx]
                nudged_pre = (
                    nudged_states[idx]
                    if idx < len(nudged_states)
                    else nudged_states[-1]
                )
                nudged_post = nudged_states[idx]

                # Use shared contrastive primitive
                delta = contrastive_hebbian_update(
                    free_pre, free_post, nudged_pre, nudged_post, self._lr, self._beta
                )

                weight_deltas[f"tiles.layer{layer_idx}.tile{tile_idx}.weight"] = delta

        return weight_deltas

    def update_weights(self, gradients: dict[str, Tensor], lr: float = 1.0) -> None:
        """Apply weight updates to tile algorithm."""
        if hasattr(self, "_tile_algo") and self._tile_algo is not None:
            # Delegate to tile algorithm's update method
            self._tile_algo.apply_weight_updates(gradients, lr)

    def get_memory_stats(self) -> dict[str, float]:
        # Estimate: num_tiles * neurons_per_tile^2 params
        tile_params = self._num_tiles * self._neurons_per_tile * self._neurons_per_tile
        state_mb = self._num_tiles * self._neurons_per_tile * 4 / 1e6  # per batch

        return {
            "tile_params_mb": tile_params * 4 / 1e6,
            "tile_states_mb": state_mb,
            "activations_mb": 0.0,
        }

    def get_settle_telemetry(self) -> dict[str, object] | None:
        return None


# Register backend
KernelRegistry.register(AlgorithmFamily.TILE, HardwareTarget.CPU, TileKernelBackend)
KernelRegistry.register(AlgorithmFamily.TILE, HardwareTarget.CUDA, TileKernelBackend)
KernelRegistry.register(AlgorithmFamily.TILE, HardwareTarget.TRITON, TileKernelBackend)


__all__ = ["TileKernelBackend"]
