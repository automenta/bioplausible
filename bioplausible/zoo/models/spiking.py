"""
Spiking Neural Network Models
=============================

SpikingSTDP model for the model zoo.
"""

import math
import torch
from snntorch import surrogate
import snntorch as snn
from torch import nn

from bioplausible.core.model_status import status_tag
from bioplausible.core.registry import LocalityLevel, register_model
from bioplausible.zoo.models.transitions import TransitionGraphMixin

__all__ = [
    "SpikingSTDP",
]


@register_model(
    "spiking_stdp",
    family="spiking",
    locality_level=LocalityLevel.LOCAL,
    tags=["spiking", "stdp", status_tag("experimental")],
)
class SpikingSTDP(TransitionGraphMixin, nn.Module):
    """
    Leaky Integrate-and-Fire neurons with 3-factor Spike-Timing-Dependent Plasticity.

    Uses snnTorch for LIF dynamics; custom 3-factor STDP rule overlaid.
    The modulator (3rd factor) is an error signal backprojected from the
    output layer, enabling supervised credit assignment to hidden layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_steps: int = 10,
        learning_rate: float = 0.01,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.lr = learning_rate
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        # Fixed random feedback weights for error backprojection to hidden layer
        self.register_buffer(
            "W_fb",
            torch.empty(output_dim, hidden_dim).uniform_(-0.5, 0.5),
        )

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
        if isinstance(input_dim, tuple):
            input_dim = math.prod(input_dim)
        return cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            learning_rate=float(kwargs.get("learning_rate", 0.01)),
        ).to(device)

    def transition_modules(self) -> list[nn.Module]:
        """Return weight layers in order: fc1, lif1, fc2, lif2."""
        return [self.fc1, self.lif1, self.fc2, self.lif2]

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []
        for step in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)

        return torch.stack(spk2_rec, dim=0).sum(0)

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        """3-factor STDP: pre x post x modulator (error signal).

        Runs the LIF simulation, computes output error, backprojects it
        to hidden layer, then does a second pass with 3-factor STDP updates.
        """
        # --- Pass 1: Forward to get output and compute error ---
        with torch.no_grad():
            out = self.forward(x)

        y_onehot = torch.zeros_like(out)
        y_onehot.scatter_(1, y.unsqueeze(1), self.num_steps * 0.8)
        output_error = y_onehot - out  # [batch, n_class]

        # Backproject error to hidden layer: e_hidden = e_out @ W_fb
        error_hidden = torch.mm(output_error, self.W_fb)  # [batch, hidden_dim]

        # --- Pass 2: Simulate with 3-factor STDP updates ---
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        pre_trace = torch.zeros(x.shape[0], x.shape[1], device=x.device)
        post_trace1 = torch.zeros(x.shape[0], self.fc1.out_features, device=x.device)
        pre_trace2 = torch.zeros(x.shape[0], self.fc1.out_features, device=x.device)
        post_trace2 = torch.zeros(x.shape[0], self.fc2.out_features, device=x.device)

        spk2_rec = []

        with torch.no_grad():
            for step in range(self.num_steps):
                # Hidden layer
                cur1 = self.fc1(x)
                spk1, mem1 = self.lif1(cur1, mem1)

                # 3-factor STDP for fc1:
                # dw[i,j] = lr * (post_i * pre_trace_j * mod_i - post_trace_i * pre_j * mod_i)
                # mod_i = error_hidden[:, i] (modulator for post-synaptic neuron i)
                pre_trace = 0.9 * pre_trace + x
                post_trace1 = 0.9 * post_trace1 + spk1

                mod1 = error_hidden  # [B, hidden]
                # Potentiation: post * (pre_trace * mod) → outer product
                pot1 = mod1.unsqueeze(2) * pre_trace.unsqueeze(1)  # [B, hidden, input]
                dw1 = self.lr * (spk1.unsqueeze(2) * pot1).sum(0) / x.shape[0]
                # Depression: (post_trace * mod) * pre → outer product
                dep1 = mod1.unsqueeze(2) * x.unsqueeze(1)  # [B, hidden, input]
                dw1 -= self.lr * (post_trace1.unsqueeze(2) * dep1).sum(0) / x.shape[0]
                self.fc1.weight.data += dw1

                # Output layer
                cur2 = self.fc2(spk1)
                spk2, mem2 = self.lif2(cur2, mem2)

                # 3-factor STDP for fc2: modulator = output_error
                pre_trace2 = 0.9 * pre_trace2 + spk1
                post_trace2 = 0.9 * post_trace2 + spk2

                mod2 = output_error  # [B, out_dim]
                pot2 = mod2.unsqueeze(2) * pre_trace2.unsqueeze(1)  # [B, out, hidden]
                dw2 = self.lr * (spk2.unsqueeze(2) * pot2).sum(0) / x.shape[0]
                dep2 = mod2.unsqueeze(2) * spk1.unsqueeze(1)  # [B, out, hidden]
                dw2 -= self.lr * (post_trace2.unsqueeze(2) * dep2).sum(0) / x.shape[0]
                self.fc2.weight.data += dw2

                spk2_rec.append(spk2)

        out = torch.stack(spk2_rec, dim=0).sum(0)
        loss = (output_error**2).mean().item()
        acc = (out.argmax(1) == y).float().mean().item()
        return {"loss": loss, "accuracy": acc}
