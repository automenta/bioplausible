"""Equilibrium Propagation model variants."""

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

from bioplausible.acceleration.kernels import HAS_CUPY, EqPropKernel
from bioplausible.acceleration.triton_kernels import TritonEqPropOps
from bioplausible.core.registry import Domain, LocalityLevel

from ....acceleration import compile_settling_loop
from ...base import BioModel, ModelConfig, register_model
from ...utils import spectral_conv2d, spectral_linear
from ..base import EqPropModel



@register_model("neural_cube")
class NeuralCube(nn.Module):
    """
    A 3D lattice neural network where neurons exist in 3D space.

    Each neuron connects only to its 26 neighbors (3x3x3 local patch minus self).
    This mimics biological neural tissue where connectivity is spatially local.
    """

    def __init__(
        self,
        cube_size: int = 6,
        input_dim: int = 64,
        output_dim: int = 10,
        max_steps: int = 30,
    ):
        super().__init__()
        self.cube_size = cube_size
        self.n_neurons = cube_size**3
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_steps = max_steps

        self.W_in = nn.Linear(input_dim, self.n_neurons)

        self.W_local = nn.Parameter(torch.zeros(self.n_neurons, 27))

        self.W_out = nn.Linear(self.n_neurons, output_dim)

        self.register_buffer("neighbor_indices", self._build_neighbor_indices())

        self._init_weights()

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
        cube_size = int(round(hidden_dim ** (1 / 3)))
        return cls(
            cube_size=max(4, cube_size),
            input_dim=input_dim,
            output_dim=output_dim,
        ).to(device)

    def _build_neighbor_indices(self) -> torch.Tensor:
        size = self.cube_size
        indices = torch.full((self.n_neurons, 27), self.n_neurons, dtype=torch.long)

        for z in range(size):
            for y in range(size):
                for x in range(size):
                    neuron_idx = z * size * size + y * size + x
                    neighbor_count = 0

                    for dz in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                nz, ny, nx = z + dz, y + dy, x + dx

                                if 0 <= nz < size and 0 <= ny < size and 0 <= nx < size:
                                    neighbor_idx = nz * size * size + ny * size + nx
                                    indices[neuron_idx, neighbor_count] = neighbor_idx

                                neighbor_count += 1

        return indices

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_in.weight, gain=0.5)
        nn.init.zeros_(self.W_in.bias)
        nn.init.normal_(self.W_local, mean=0, std=0.1)
        nn.init.xavier_uniform_(self.W_out.weight, gain=0.5)
        nn.init.zeros_(self.W_out.bias)

    def local_update(self, h: torch.Tensor) -> torch.Tensor:
        if (
            hasattr(TritonEqPropOps, "neural_cube_update")
            and TritonEqPropOps.is_available()
            and h.is_cuda
        ):
            return TritonEqPropOps.neural_cube_update(h, self.W_local, self.cube_size)

        batch_size = h.shape[0]

        h_padded = F.pad(h, (0, 1))

        indices_expanded = self.neighbor_indices.unsqueeze(0).expand(batch_size, -1, -1)
        h_expanded = h_padded.unsqueeze(1).expand(-1, self.n_neurons, -1)
        neighbor_activations = torch.gather(h_expanded, 2, indices_expanded)

        weighted = (neighbor_activations * self.W_local.unsqueeze(0)).sum(dim=2)

        return weighted

    def forward(
        self,
        x: torch.Tensor,
        steps: int = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        steps = steps or self.max_steps
        batch_size = x.shape[0]
        device = x.device

        h = torch.zeros(batch_size, self.n_neurons, device=device, dtype=x.dtype)

        x_proj = self.W_in(x)

        trajectory = [h.detach()] if return_trajectory else None

        for _ in range(steps):
            local_contrib = self.local_update(h)

            h = torch.tanh(x_proj + local_contrib)

            if return_trajectory:
                trajectory.append(h.detach())

        out = self.W_out(h)

        if return_trajectory:
            return out, trajectory
        return out

    def get_topology_stats(self) -> dict:
        active_weights = (self.W_local.abs() > 0.01).float().mean().item()

        fully_connected = self.n_neurons * self.n_neurons
        local_connections = self.n_neurons * 27

        return {
            "cube_size": self.cube_size,
            "n_neurons": self.n_neurons,
            "local_connections": local_connections,
            "fully_connected_equivalent": fully_connected,
            "connection_reduction": 1 - (local_connections / fully_connected),
            "active_weight_fraction": active_weights,
        }

    def get_cube_slice(self, h: torch.Tensor, z: int) -> torch.Tensor:
        size = self.cube_size
        start = z * size * size
        end = (z + 1) * size * size

        slice_flat = h[..., start:end]
        return slice_flat.reshape(*h.shape[:-1], size, size)

    def visualize_cube_ascii(self, h: torch.Tensor, sample_idx: int = 0) -> str:
        chars = " .dbBF"
        size = self.cube_size

        lines = []
        lines.append(f"Neural Cube {size}x{size}x{size} (z-slices)")
        lines.append("=" * (size * 3 + 10))

        h_sample = h[sample_idx].detach().cpu()
        h_norm = (h_sample - h_sample.min()) / (h_sample.max() - h_sample.min() + 1e-8)

        for z in range(size):
            lines.append(f"\nz={z}:")
            for y in range(size):
                row = ""
                for x in range(size):
                    idx = z * size * size + y * size + x
                    val = h_norm[idx].item()
                    char_idx = min(int(val * (len(chars) - 1)), len(chars) - 1)
                    row += chars[char_idx] * 2
                lines.append(f"  {row}")

        return "\n".join(lines)


