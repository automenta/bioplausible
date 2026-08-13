"""Equilibrium Propagation model variants."""

import torch
import torch.nn.functional as F
from torch import nn

from bioplausible.acceleration.triton_kernels import TritonEqPropOps
from bioplausible.core.losses import compute_accuracy
from bioplausible.core.model_status import status_tag
from bioplausible.core.registry import register_model
from bioplausible.zoo._settling import settle_state
from bioplausible.zoo.models.transitions import TransitionGraphMixin

__all__ = [
    "LocalUpdateModule",
    "NeuralCube",
]


class LocalUpdateModule(nn.Module):
    """Wraps NeuralCube's W_local + neighbor logic into a callable module."""

    def __init__(
        self, W_local: nn.Parameter, neighbor_indices: torch.Tensor, cube_size: int
    ):
        super().__init__()
        self.W_local = W_local
        self.register_buffer("neighbor_indices", neighbor_indices)
        self.cube_size = cube_size

    def forward(self, h: torch.Tensor, x: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, n_neurons = h.shape[0], self.neighbor_indices.shape[0]
        h_padded = torch.nn.functional.pad(h, (0, 1))
        indices_expanded = self.neighbor_indices.unsqueeze(0).expand(batch_size, -1, -1)
        h_expanded = h_padded.unsqueeze(1).expand(-1, n_neurons, -1)
        neighbor_activations = torch.gather(h_expanded, 2, indices_expanded)
        return (neighbor_activations * self.W_local.unsqueeze(0)).sum(dim=2)


@register_model(
    "neural_cube",
    family="eqprop",
    tags=["eqprop", "neural-cube", status_tag("broken")],
)
class NeuralCube(TransitionGraphMixin, nn.Module):
    """
    A 3D lattice neural network where neurons exist in 3D space.

    Each neuron connects only to its 26 neighbors (3x3x3 local patch minus self).
    This mimics biological neural tissue where connectivity is spatially local.

    Status: ``broken`` — the structural axis is ``cube_size`` (a 3D lattice),
    not ``num_layers``; sampled ``num_layers`` is silently dropped at
    construction. See ``docs/phantom_knob_audit.md``.
    """

    def __init__(
        self,
        cube_size: int = 6,
        input_dim: int = 64,
        output_dim: int = 10,
        max_steps: int = 30,
        convergence_threshold: float = 1e-3,
        convergence_start: int = 5,
        learning_rate: float = 0.01,
        beta: float = 0.1,
    ):
        super().__init__()
        self.cube_size = cube_size
        self.n_neurons = cube_size**3
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_steps = max_steps
        # P1: adopt the shared settle protocol — convergence knobs are real
        # constructor parameters, so a search space may legitimately sweep them.
        self.convergence_threshold = convergence_threshold
        self.convergence_start = convergence_start

        self.W_in = nn.Linear(input_dim, self.n_neurons)

        self.W_local = nn.Parameter(torch.zeros(self.n_neurons, 27))

        self.W_out = nn.Linear(self.n_neurons, output_dim)

        self.register_buffer("neighbor_indices", self._build_neighbor_indices())
        self.local_update_mod = LocalUpdateModule(
            self.W_local, self.neighbor_indices, self.cube_size
        )

        self._init_weights()
        self.learning_rate = learning_rate
        self.beta = beta

    def transition_modules(self) -> list[nn.Module]:
        """Modules called in order during one forward step.

        :returns: ``[self.W_in, self.local_update_mod, self.W_out]``
        """
        return [self.W_in, self.local_update_mod, self.W_out]

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
        import math

        if isinstance(input_dim, tuple):
            input_dim = math.prod(input_dim)
        cube_size = int(round(hidden_dim ** (1 / 3)))
        # Allow cube_size=3 (27 neurons) for hidden_dim=32 to enable
        # fair parameter-count comparison with backprop MLP
        return cls(
            cube_size=max(3, cube_size),
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

    def _initialize_hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        """Zero hidden state over the cube's neurons (P1 settle protocol)."""
        return torch.zeros((x.shape[0], self.n_neurons), device=x.device, dtype=x.dtype)

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        """Project the raw input into the cube's neuron space (P1 protocol)."""
        return self.W_in(x)

    def _forward_step_impl(
        self, h: torch.Tensor, x_transform: torch.Tensor
    ) -> torch.Tensor:
        """One recurrent settle step: input projection + local-neighbor update."""
        return torch.tanh(x_transform + self.local_update(h))

    def forward(
        self,
        x: torch.Tensor,
        steps: int | None = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        steps = steps or self.max_steps

        if not return_trajectory:
            h, steps_taken, converged = settle_state(self, x, steps=steps)
            self._last_settle_steps = steps_taken
            self._last_settle_converged = converged
            return self.W_out(h)

        # Trajectory path (visualization/analysis only): hand-rolled loop so the
        # per-step snapshots are returned. Reuses the same recurrence and the
        # same convergence knobs as :func:`settle_state`.
        x_transform = self._transform_input(x)
        h = self._initialize_hidden_state(x)
        trajectory = [h.detach()]
        for step_idx in range(steps):
            h_new = self._forward_step_impl(h, x_transform)
            trajectory.append(h_new.detach())
            if (
                step_idx > self.convergence_start
                and torch.dist(h_new, h, p=float("inf")).item()
                < self.convergence_threshold
            ):
                self._last_settle_steps = step_idx + 1
                self._last_settle_converged = True
                return self.W_out(h_new), trajectory
            h = h_new
        self._last_settle_steps = steps
        self._last_settle_converged = False
        return self.W_out(h), trajectory

    def _settle_nudged(
        self, x: torch.Tensor, nudge: torch.Tensor | None, steps: int
    ) -> torch.Tensor:
        """Settle to fixed point, optionally with output-layer nudging.

        For the free phase ``nudge`` is None. For the nudged phase, the nudge
        tensor ``v`` (gradient w.r.t. logits) is projected back through W_out.
        """
        x_transform = self._transform_input(x)
        h = self._initialize_hidden_state(x)

        for _ in range(steps):
            if nudge is None:
                h_new = self._forward_step_impl(h, x_transform)
            else:
                # Nudged: h ← tanh(W_in x + local_update(h)) − beta * (v · W_out)
                h_new = self._forward_step_impl(h, x_transform)
                h_new = h_new - self.beta * torch.mm(nudge, self.W_out.weight)
            delta = torch.dist(h_new, h, p=float("inf")).item()
            h = h_new
            if delta < self.convergence_threshold:
                break
        return h

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        """Energy-contrastive EqProp train step for NeuralCube.

        Free settle → compute dL/dlogits → nudged settle → energy gradient
        difference → manual weight update. No external optimizer.
        """
        x_flat = x.reshape(x.size(0), -1) if x.dim() > 2 else x

        # --- Free phase ---
        with torch.no_grad():
            h_free = self._settle_nudged(x_flat, None, self.max_steps)
            logits_free = self.W_out(h_free)
            loss_free = F.cross_entropy(logits_free, y)

        # dL/dlogits at free equilibrium
        logits_det = logits_free.clone().detach().requires_grad_(True)
        v = torch.autograd.grad(
            F.cross_entropy(logits_det, y), logits_det, retain_graph=False
        )[0]

        # --- Nudged phase ---
        with torch.no_grad():
            h_nudged = self._settle_nudged(x_flat, v, max(3, self.max_steps // 3))

        # --- Energy gradients: ∇_θ Σ(0.5 h² − h · pre_activation) ---
        def _energy_grads(h: torch.Tensor) -> list[torch.Tensor]:
            h_req = h.detach().requires_grad_(True)
            pre_act = x_flat @ self.W_in.weight.T + self.W_in.bias
            pre_act = pre_act + self.local_update(h_req)
            energy = torch.sum(0.5 * h_req**2 - h_req * pre_act)
            grads = torch.autograd.grad(
                energy,
                self.parameters(),
                retain_graph=True,
                allow_unused=True,
            )
            return [
                g if g is not None else torch.zeros_like(p)
                for g, p in zip(grads, self.parameters())
            ]

        gf = _energy_grads(h_free)
        gn = _energy_grads(h_nudged)

        # --- Manual weight updates ---
        with torch.no_grad():
            for p, gf_p, gn_p in zip(self.parameters(), gf, gn):
                if p is self.W_out.weight:
                    p -= self.learning_rate * torch.mm(v.T, h_free)
                elif p is self.W_out.bias:
                    p -= self.learning_rate * v.sum(0)
                else:
                    p -= self.learning_rate * (gf_p - gn_p) / self.beta

        acc = compute_accuracy(logits_free, y)
        return {"loss": loss_free.item(), "accuracy": acc}

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
