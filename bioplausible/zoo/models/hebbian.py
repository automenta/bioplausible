"""
Combined Hebbian Models
=======================

Aggregates all Hebbian-family models into a single module for the model zoo.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

from bioplausible.core.model_status import status_tag
from bioplausible.core.registry import LocalityLevel, register_model

from ..nebc_base import NEBCBase
from .transitions import TransitionGraphMixin

# ============================================================================
# hebbian_chain.py - DeepHebbianChain, HebbianLayer, HebbianCube
# ============================================================================


__all__ = [
    "DeepHebbianChain",
    "HebbianCube",
    "HebbianLayer",
    "ThreeFactorHebbian",
]


class HebbianLayer(nn.Module):
    """
    Single Hebbian layer with Oja's normalization rule.

    Update: Delta W = eta * (y @ x.T - y^2 @ W)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        learning_rate: float = 0.01,
        use_oja: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.learning_rate = learning_rate
        self.use_oja = use_oja

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.orthogonal_(self.weight, gain=1.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)

    def hebbian_update(self, x: torch.Tensor, y: torch.Tensor):
        batch_size = x.size(0)

        if hasattr(self, "weight_orig"):
            target_weight = self.weight_orig
        else:
            target_weight = self.weight

        with torch.no_grad():
            target_weight.addmm_(y.T, x, alpha=self.learning_rate / batch_size)

            if self.use_oja:
                y_sq = y.pow(2).mean(dim=0, keepdim=True).T
                target_weight.addcmul_(y_sq, self.weight, value=-self.learning_rate)


@register_model(
    "deep_hebbian",
    family="hebbian",
    locality_level=LocalityLevel.LOCAL,
    tags=["hebbian", "deep", status_tag("broken")],
)
@register_model(
    "hebbian_chain",
    family="hebbian",
    locality_level=LocalityLevel.LOCAL,
    tags=[status_tag("broken")],
)
class DeepHebbianChain(NEBCBase):
    """
    Deep Hebbian Chain with spectral normalization.

    Tests signal propagation through 1000+ layers with pure Hebbian learning.
    """

    algorithm_name = "HebbianChain"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 100,
        use_spectral_norm: bool = True,
        max_steps: int = 1,
        hebbian_lr: float = 0.01,
        use_oja: bool = True,
        spectral_norm_power_iterations: int = 5,
        learning_rate: float | None = None,
    ):
        # ``learning_rate`` is the canonical knob name (the construction layer
        # surfaces sampled LRs as ``learning_rate``, not ``hebbian_lr``); accept
        # it as an alias so the sweep can actually vary this model's LR.
        if learning_rate is not None:
            hebbian_lr = learning_rate
        self.hebbian_lr = hebbian_lr
        self.use_oja = use_oja
        self.spectral_norm_power_iterations = spectral_norm_power_iterations
        super().__init__(
            input_dim, hidden_dim, output_dim, num_layers, use_spectral_norm, max_steps
        )

    def transition_modules(self) -> list[nn.Module]:
        """Modules called in order during one forward step.

        Must include input projection (W_in), hidden chain, and output head
        so that propagators (e.g., ContrastiveHebbianLearning) can run the
        full free and clamped forward passes from input to output.
        """
        return [self.W_in, *self.chain, self.head]

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
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            use_spectral_norm=kwargs.get("use_spectral_norm", True),
            hebbian_lr=float(kwargs.get("learning_rate", kwargs.get("lr", 0.01))),
            use_oja=kwargs.get("use_oja", True),
            spectral_norm_power_iterations=int(
                kwargs.get("spectral_norm_power_iterations", 5)
            ),
        ).to(device)

    def _build_layers(self):
        self.W_in = nn.Linear(self.input_dim, self.hidden_dim)
        if self.use_spectral_norm:
            self.W_in = spectral_norm(
                self.W_in, n_power_iterations=self.spectral_norm_power_iterations
            )

        self.chain = nn.ModuleList()
        for i in range(self.num_layers):
            layer = HebbianLayer(
                self.hidden_dim,
                self.hidden_dim,
                learning_rate=self.hebbian_lr,
                use_oja=self.use_oja,
            )

            if self.use_spectral_norm:
                layer = spectral_norm(
                    layer, n_power_iterations=self.spectral_norm_power_iterations
                )

            self.chain.append(layer)

        self.head = nn.Linear(self.hidden_dim, self.output_dim)
        if self.use_spectral_norm:
            self.head = spectral_norm(
                self.head, n_power_iterations=self.spectral_norm_power_iterations
            )

    def forward(
        self,
        x: torch.Tensor,
        steps: int | None = None,
        return_signal_norms: bool = False,
    ) -> torch.Tensor:
        if not self.training and self.use_spectral_norm:
            w = self._get_spectral_normalized_weight(self.W_in)
            b = self.W_in.bias
            h = F.linear(x, w, b)
        else:
            h = self.W_in(x)

        h.tanh_()

        norms = [h.abs().max().item()]

        for layer in self.chain:
            if not self.training and self.use_spectral_norm:
                w = self._get_spectral_normalized_weight(layer)
                h = F.linear(h, w)
            else:
                h = layer(h)

            h.tanh_()

            if return_signal_norms:
                norms.append(h.abs().max().item())

        if not self.training and self.use_spectral_norm:
            w = self._get_spectral_normalized_weight(self.head)
            b = self.head.bias
            output = F.linear(h, w, b)
        else:
            output = self.head(h)

        if return_signal_norms:
            return output, norms
        return output

    def measure_signal_propagation(self, x: torch.Tensor) -> dict[str, float]:
        _, norms = self.forward(x, return_signal_norms=True)

        initial = norms[0]
        final = norms[-1]
        decay = final / initial if initial > 1e-10 else 0.0

        return {
            "initial_norm": initial,
            "final_norm": final,
            "decay_ratio": decay,
            "norms": norms,
        }

    def get_stats(self) -> dict[str, float]:
        stats = super().get_stats()
        stats["hebbian_lr"] = self.hebbian_lr
        stats["use_oja"] = self.use_oja
        return stats

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        """Local Hebbian (Oja) update per layer + supervised output head.

        Free phase forward pass streams activations; each Hebbian layer updates
        its weights via Oja's rule. The output head (``self.head``) receives a
        supervised delta update so the network actually produces useful logits.
        No autograd graph, no BPTT fallback.
        """
        self.train()
        transitions = self.transition_modules()

        with torch.no_grad():
            h = x
            activations = [h]
            for layer in transitions:
                h = layer(h)
                if hasattr(layer, "hebbian_update"):
                    layer.hebbian_update(activations[-1], h)
                activations.append(h)

            # Supervised update for the output head (the last transition module).
            # Delta W_out = lr * (y_onehot - softmax(logits)) @ h_prev
            logits = h
            y_onehot = torch.zeros_like(logits)
            y_onehot.scatter_(1, y.unsqueeze(1), 1.0)
            error = y_onehot - torch.softmax(logits, dim=1)
            head = transitions[-1]
            # For spectral-norm parametrized layers, update the *original*
            # parameter (``parametrizations.weight.original``), not the
            # computed ``weight`` property — otherwise the update is silently
            # discarded.
            if hasattr(head, "parametrizations"):
                head_w = dict(head.named_parameters())[
                    "parametrizations.weight.original"
                ]
            elif hasattr(head, "weight"):
                head_w = head.weight
            else:
                head_w = None
            if head_w is not None:
                head_w.addmm_(
                    error.T,
                    activations[-2],
                    alpha=self.hebbian_lr / x.shape[0],
                )

        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean().item()
        return {"loss": loss.item(), "accuracy": acc}


@register_model(
    "hebbian_3d",
    family="hebbian",
    locality_level=LocalityLevel.LOCAL,
    tags=[status_tag("broken")],
)
class HebbianCube(TransitionGraphMixin, NEBCBase):
    """
    3D Hebbian lattice for testing spatial organization.
    """

    algorithm_name = "HebbianCube"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 10,
        cube_size: int = 8,
        use_spectral_norm: bool = True,
        max_steps: int = 1,
    ):
        self.cube_size = cube_size
        super().__init__(
            input_dim, hidden_dim, output_dim, num_layers, use_spectral_norm, max_steps
        )

    def transition_modules(self) -> list[nn.Module]:
        """Return transition modules in forward order."""
        return [self.input_proj, *self.conv_layers, self.head]

    def _build_layers(self):
        cube_neurons = self.cube_size**3
        self.input_proj = nn.Linear(self.input_dim, min(self.hidden_dim, cube_neurons))
        if self.use_spectral_norm:
            self.input_proj = spectral_norm(self.input_proj, n_power_iterations=5)

        channels = max(1, self.hidden_dim // cube_neurons)
        self.conv_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
            if self.use_spectral_norm:
                conv = spectral_norm(conv, n_power_iterations=5)
            self.conv_layers.append(conv)

        self.head = nn.Linear(min(self.hidden_dim, cube_neurons), self.output_dim)
        if self.use_spectral_norm:
            self.head = spectral_norm(self.head, n_power_iterations=5)

        self._cube_neurons = min(self.hidden_dim, cube_neurons)
        self._channels = channels

    def forward(self, x: torch.Tensor, steps: int | None = None) -> torch.Tensor:
        batch_size = x.size(0)

        h = self.input_proj(x)
        h = torch.tanh(h)

        c = self._channels
        s = self.cube_size
        if h.size(1) >= c * s * s * s:
            h_3d = h[:, : c * s * s * s].view(batch_size, c, s, s, s)
        else:
            h_padded = F.pad(h, (0, c * s * s * s - h.size(1)))
            h_3d = h_padded.view(batch_size, c, s, s, s)

        for conv in self.conv_layers:
            h_3d = torch.tanh(conv(h_3d))

        h_flat = h_3d.view(batch_size, -1)[:, : self._cube_neurons]

        return self.head(h_flat)


# ============================================================================
# three_factor.py - ThreeFactorHebbian
# ============================================================================


@register_model(
    "three_factor_hebbian",
    family="hebbian",
    locality_level=LocalityLevel.LOCAL,
    tags=["hebbian", "three-factor", status_tag("experimental")],
)
class ThreeFactorHebbian(TransitionGraphMixin, nn.Module):
    """
    Three-Factor Learning: Delta w = eta * M * pre * post
    where M is a neuromodulatory signal (dopamine-like global reward).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        learning_rate: float = 0.005,
    ):
        super().__init__()
        if isinstance(input_dim, tuple):
            input_dim = math.prod(input_dim)
        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim, bias=False)])
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
        self.out_layer = nn.Linear(hidden_dim, output_dim, bias=False)
        self.relu = nn.ReLU()
        self.lr = learning_rate

    @classmethod
    def build(
        cls,
        spec,
        input_dim,
        output_dim,
        hidden_dim,
        num_layers=2,
        device="cpu",
        task_type="vision",
        **kwargs,
    ):
        model = cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            learning_rate=float(kwargs.get("learning_rate", kwargs.get("lr", 0.005))),
        )
        model = model.to(device)
        return model

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = self.relu(layer(h))
        return self.out_layer(h)

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        hs = [x]
        h = x
        for layer in self.layers:
            h = self.relu(layer(h))
            hs.append(h)
        out = self.out_layer(h)

        # --- Compute modulator: graded error signal, not binary ---
        # M_i = softmax(out_i) - onehot(y_i) → continuous, ranges [-1, 1]
        # Positive M → prediction was too high (should decrease weight)
        # Negative M → prediction was too low (should increase weight)
        # This is the 3rd factor: modulates Hebbian updates by error magnitude
        with torch.no_grad():
            pred_probs = torch.softmax(out, dim=1)
            y_onehot = torch.zeros_like(out, device=out.device)
            y_onehot.scatter_(1, y.unsqueeze(1), 1.0)
            output_modulator = y_onehot - pred_probs  # [batch, classes]

        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                pre = hs[i]
                post = hs[i + 1]
                # Backproject output modulator to hidden layer via output weights
                hidden_modulator = torch.mm(
                    output_modulator, self.out_layer.weight
                )  # [B, hidden]
                # Normalize to prevent NaN: scale by hidden dim
                hidden_modulator = hidden_modulator / max(
                    hidden_modulator.abs().max().item(), 1.0
                )
                post_mod = post * hidden_modulator  # [B, hidden]
                layer.weight.data += self.lr * torch.mm(post_mod.T, pre) / x.shape[0]

            y_onehot = torch.zeros_like(out, device=out.device)
            y_onehot.scatter_(1, y.unsqueeze(1), 1.0)
            error = y_onehot - out
            self.out_layer.weight.data += (
                self.lr * torch.mm(error.T, hs[-1]) / x.shape[0]
            )

        preds = out.argmax(1)
        correct = (preds == y).float()
        loss = nn.functional.cross_entropy(out, y).item()
        return {"loss": loss, "accuracy": correct.mean().item()}
