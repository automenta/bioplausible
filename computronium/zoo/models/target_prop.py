"""
Target Propagation Models
==========================

Difference Target Propagation model for the model zoo.
"""

import math

import torch
from torch import nn

from computronium.core.losses import compute_accuracy
from computronium.core.model_status import status_tag
from computronium.core.registry import LocalityLevel, register_model
from computronium.core.utils.optimizer import OptimizerConfig, create_optimizer
from computronium.zoo.models.transitions import TransitionGraphMixin

__all__ = [
    "DTPLayer",
    "DifferenceTargetProp",
]


class DTPLayer(nn.Module):
    def __init__(self, in_features, out_features, learning_rate: float = 0.001):
        super().__init__()
        self.forward_net = nn.Sequential(
            nn.Linear(in_features, out_features), nn.Tanh()
        )
        self.inverse_net = nn.Sequential(
            nn.Linear(out_features, in_features), nn.Tanh()
        )
        self.opt_f = create_optimizer(
            self.forward_net,
            OptimizerConfig(name="adam", lr=learning_rate, weight_decay=0.0),
        )
        self.opt_g = create_optimizer(
            self.inverse_net,
            OptimizerConfig(name="adam", lr=learning_rate, weight_decay=0.0),
        )


@register_model(
    "diff_target_prop",
    family="target_prop",
    locality_level=LocalityLevel.LAYERWISE,
    tags=["target-prop", "diffprop", status_tag("stable")],
)
class DifferenceTargetProp(TransitionGraphMixin, nn.Module):
    """
    Difference Target Propagation (Lee et al. 2015).

    Propagates targets (not gradients) backward using learned approximate inverses.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        learning_rate: float = 0.001,
        target_lr: float = 0.1,
    ):
        super().__init__()
        if isinstance(input_dim, tuple):
            input_dim = math.prod(input_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.target_lr = target_lr
        self.layers = nn.ModuleList([DTPLayer(input_dim, hidden_dim, learning_rate)])
        for _ in range(num_layers - 1):
            self.layers.append(DTPLayer(hidden_dim, hidden_dim, learning_rate))
        self.out_layer = nn.Linear(hidden_dim, output_dim)

        self.out_opt = create_optimizer(
            self.out_layer,
            OptimizerConfig(name="adam", lr=learning_rate, weight_decay=0.0),
        )
        self.criterion = nn.CrossEntropyLoss()

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
        return cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            learning_rate=float(kwargs.get("learning_rate", 0.001)),
            target_lr=float(kwargs.get("target_lr", 0.1)),
        ).to(device)

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = layer.forward_net(h)
        return self.out_layer(h)

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        # Forward pass collecting activations
        hs = [x]
        h = x
        for layer in self.layers:
            h = layer.forward_net(h)
            hs.append(h)
        out = self.out_layer(h)

        loss = self.criterion(out, y)

        # --- Update output layer first (before backward target propagation
        # modifies hidden layer weights, which would invalidate the forward graph) ---
        self.out_opt.zero_grad()
        loss.backward()
        self.out_opt.step()

        # --- Compute target for output layer ---
        # t_target = h - target_lr * dL/dh (where dL/dh via output layer)
        t = h.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            out_t = self.out_layer(t)
            loss_t = self.criterion(out_t, y)
            grad_t = torch.autograd.grad(loss_t, t)[0]

        with torch.no_grad():
            t_target = h - self.target_lr * grad_t

        targets = [t_target]

        # --- Backward target propagation ---
        # Propagate target backward through inverse mappings
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if i > 0:
                h_prev = hs[i]
                h_curr = hs[i + 1]
                t_curr = targets[-1]

                with torch.no_grad():
                    t_prev = (
                        h_prev - layer.inverse_net(h_curr) + layer.inverse_net(t_curr)
                    )
                    targets.append(t_prev)

            # --- Train forward net to hit target ---
            t_curr = targets[-len(targets)]
            h_prev_det = hs[i].detach()
            layer.opt_f.zero_grad()
            pred_h = layer.forward_net(h_prev_det)
            loss_f = nn.functional.mse_loss(pred_h, t_curr)
            loss_f.backward()

            if i > 0:
                # --- Train inverse net for cycle consistency ---
                # Detach pred_h to avoid graph conflict after loss_f.backward freed it
                layer.opt_g.zero_grad()
                inv_out = layer.inverse_net(pred_h.detach())
                loss_g = nn.functional.mse_loss(inv_out, h_prev_det)
                loss_g.backward()
                layer.opt_g.step()

            layer.opt_f.step()

        acc = compute_accuracy(out, y)
        return {"loss": loss.item(), "accuracy": acc}
