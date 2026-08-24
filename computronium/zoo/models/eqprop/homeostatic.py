"""Equilibrium Propagation model variants."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from computronium.zoo.models.transitions import TransitionGraphMixin

__all__ = [
    "HomeostasisMetrics",
    "HomeostaticEqProp",
]


@dataclass(frozen=True, slots=True)
class HomeostasisMetrics:
    avg_velocity: float
    lipschitz_estimate: float
    brake_applied: float
    boost_applied: float
    layers_braked: int
    layers_boosted: int


class HomeostaticEqProp(TransitionGraphMixin, nn.Module):
    """
    EqProp with Dynamic Lipschitz Scaling for autonomous stability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 5,
        alpha: float = 0.5,
        target_lipschitz: float = 0.95,
        velocity_threshold_high: float = 0.1,
        velocity_threshold_low: float = 0.01,
        adaptation_rate: float = 0.01,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.alpha = alpha
        self.target_lipschitz = target_lipschitz
        self.velocity_threshold_high = velocity_threshold_high
        self.velocity_threshold_low = velocity_threshold_low
        self.adaptation_rate = adaptation_rate

        self.W_in = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.register_buffer("layer_scales", torch.ones(num_layers))

        self.head = nn.Linear(hidden_dim, output_dim)

        for layer in self.layers:
            nn.init.orthogonal_(layer.weight)
            with torch.no_grad():
                layer.weight.mul_(0.7)

        self.last_velocities: dict[int, float] = {}
        self.homeostasis_history: list[HomeostasisMetrics] = []

    def transition_modules(self) -> list[nn.Module]:
        """Modules called in order during one forward step.

        :returns: ``[self.W_in, *self.layers, self.head]``
        """
        return [self.W_in, *self.layers, self.head]

    def _estimate_layer_lipschitz(self, layer_idx: int) -> float:
        original_weight = self.layers[layer_idx].weight
        scaled_weight = original_weight * self.layer_scales[layer_idx]

        with torch.no_grad():
            W = scaled_weight
            u = torch.randn(W.shape[1], device=W.device)
            u = F.normalize(u, dim=0)
            for _ in range(3):
                v = F.normalize(W @ u, dim=0)
                u = F.normalize(W.T @ v, dim=0)
            sigma = torch.norm(W @ u).item()
        return sigma

    def forward_step(
        self,
        h_states: dict[int, torch.Tensor],
        x: torch.Tensor,
        track_velocity: bool = False,
    ) -> tuple[dict[int, torch.Tensor], dict[int, float]]:
        new_states = {}
        velocities = {}
        x_emb = self.W_in(x)

        for i, layer in enumerate(self.layers):
            pre = x_emb if i == 0 else h_states.get(i - 1, torch.zeros_like(x_emb))
            h_curr = h_states.get(i, torch.zeros_like(pre))

            scale = self.layer_scales[i]
            h_target = torch.tanh(F.linear(pre, layer.weight * scale, layer.bias))

            h_new = (1 - self.alpha) * h_curr + self.alpha * h_target
            new_states[i] = h_new

            if track_velocity:
                velocity = torch.mean(torch.abs(h_new - h_curr)).item()
                velocities[i] = velocity

        return new_states, velocities

    def apply_homeostasis(self, velocities: dict[int, float]) -> HomeostasisMetrics:
        brake_total = 0.0
        boost_total = 0.0
        layers_braked = 0
        layers_boosted = 0

        for i, velocity in velocities.items():
            current_L = self._estimate_layer_lipschitz(i)

            if velocity > self.velocity_threshold_high or current_L > (
                self.target_lipschitz + 0.1
            ):
                error_v = max(0, velocity - self.velocity_threshold_high)
                error_l = max(0, current_L - self.target_lipschitz)

                error = error_v + error_l

                factor = 1.0 - (self.adaptation_rate * (1.0 + 10.0 * error))
                factor = max(0.5, factor)

                self.layer_scales[i] *= factor
                brake_total += 1.0 - factor
                layers_braked += 1

            elif velocity < self.velocity_threshold_low:
                current_L = self._estimate_layer_lipschitz(i)
                if current_L < self.target_lipschitz:
                    error = self.velocity_threshold_low - velocity
                    factor = 1.0 + (self.adaptation_rate * (1.0 + 5.0 * error))
                    factor = min(1.5, factor)

                    self.layer_scales[i] *= factor
                    boost_total += factor - 1.0
                    layers_boosted += 1

        self.layer_scales.clamp_(0.1, 3.0)

        avg_velocity = sum(velocities.values()) / len(velocities) if velocities else 0.0
        avg_lipschitz = (
            sum(self._estimate_layer_lipschitz(i) for i in range(self.num_layers))
            / self.num_layers
        )

        metrics = HomeostasisMetrics(
            avg_velocity=avg_velocity,
            lipschitz_estimate=avg_lipschitz,
            brake_applied=brake_total,
            boost_applied=boost_total,
            layers_braked=layers_braked,
            layers_boosted=layers_boosted,
        )

        self.homeostasis_history.append(metrics)
        self.last_velocities = velocities

        return metrics

    def forward(
        self, x: torch.Tensor, steps: int = 30, apply_homeostasis: bool = True
    ) -> torch.Tensor:
        batch_size = x.size(0)
        h_states = {
            i: torch.zeros(batch_size, self.hidden_dim, device=x.device)
            for i in range(self.num_layers)
        }

        all_velocities = []
        for step in range(steps):
            track = step >= steps // 2
            h_states, velocities = self.forward_step(h_states, x, track_velocity=track)
            if track:
                all_velocities.append(velocities)

        if apply_homeostasis and all_velocities:
            avg_velocities = {}
            for i in range(self.num_layers):
                avg_velocities[i] = sum(v.get(i, 0) for v in all_velocities) / len(
                    all_velocities
                )
            self.apply_homeostasis(avg_velocities)

        return self.head(h_states[self.num_layers - 1])

    def get_stability_report(self) -> str:
        lipschitz = [self._estimate_layer_lipschitz(i) for i in range(self.num_layers)]
        max_L = max(lipschitz) if lipschitz else 0.0
        status = "STABLE" if max_L < 1.0 else "UNSTABLE"

        lines = [
            f"Max Lipschitz: {max_L:.4f} {status}",
            f"Layer Scales: {[f'{s:.3f}' for s in self.layer_scales.tolist()]}",
        ]
        if self.homeostasis_history:
            last = self.homeostasis_history[-1]
            lines.append(
                f"Last Action: {last.layers_braked} braked, {last.layers_boosted} boosted"
            )

        return "\n".join(lines)
