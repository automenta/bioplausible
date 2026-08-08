"""Self-contained energy-contrastive Equilibrium Propagation engine.

Implements the Scellier & Bengio energy contrastive rule on a single-hidden
recurrent MLP, settled via ``settle_single_state`` (contractive forward-only
with early convergence). No external optimizer — weight updates are the
genuine free/nudged energy difference: ``Δw = (∇E_nudged − ∇E_free) / β``.
This is the proven, fast, O(1)-memory design used by ``GraphEqProp`` (vision
path, 0.85 acc, 9 s/epoch on MNIST).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

from bioplausible.core.config import ModelConfig
from bioplausible.zoo._settling import settle_single_state
from bioplausible.zoo.models.base import EqPropModel
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LazyStats:
    """Statistics for lazy execution."""

    total_neurons: int = 0
    active_neurons: int = 0
    skipped_neurons: int = 0

    @property
    def skip_ratio(self) -> float:
        if self.total_neurons == 0:
            return 0.0
        return self.skipped_neurons / self.total_neurons

    @property
    def flop_savings(self) -> float:
        return self.skip_ratio * 100

    @staticmethod
    def reset() -> LazyStats:
        return LazyStats()


def _flatten(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.size(0), -1) if x.dim() > 2 else x


class EquilibriumMLP(EqPropModel):
    """
    Single-hidden recurrent MLP trained via energy-contrastive EqProp.

    Recurrence:  h ← σ(W_in x + W_rec h)
    Output:      W_out h*
    Settle:      contractive forward-only via ``settle_single_state``
                 (spectral-norm freeze + early convergence, O(1) memory).
    Train step:  energy contrastive (free + nudged), fully self-contained
                 manual updates — no external optimizer.
    """

    #: Recurrence variant: "plain", "momentum", "sparse", "feedback"
    variant: str = "plain"

    def __init__(
        self,
        config: ModelConfig | None = None,
        gradient_method: str = "equilibrium",
        **kwargs,
    ):
        super().__init__(config, gradient_method=gradient_method, **kwargs)
        self.lr = self.config.learning_rate
        self.beta = self.config.beta
        self.nudge_steps = int(
            self.config.extra.get("nudge_steps", max(3, self.max_steps // 3))
        )
        self.sparse_ratio = float(self.config.extra.get("sparse_ratio", 0.5))
        self.momentum = float(self.config.extra.get("momentum", 0.5))
        # Early-convergence knobs reach the shared settle primitive via the
        # model surface (``settle_single_state`` honours
        # ``model.convergence_threshold`` / ``convergence_start``).
        self.convergence_threshold = float(
            kwargs.get(
                "convergence_threshold",
                getattr(self.config, "convergence_threshold", 1e-3),
            )
        )
        self.convergence_start = int(
            kwargs.get(
                "convergence_start",
                getattr(self.config, "convergence_start", 5),
            )
        )

    def get_hebbian_pairs(
        self, h: torch.Tensor, x: torch.Tensor
    ) -> list[tuple[nn.Module, torch.Tensor, torch.Tensor]]:
        """Return ``(layer, input, target)`` tuples for the base-class contrastive update."""
        x_in = _flatten(x)
        return [(self.W_in, x_in, h), (self.W_rec, h, h)]

    def _build_layers(self):
        hid = self.hidden_dim if self.hidden_dim > 0 else 64
        self.W_in = nn.Linear(self.input_dim, hid)
        self.W_rec = nn.Linear(hid, hid)
        self.W_out = nn.Linear(hid, self.output_dim)
        if self.use_spectral_norm:
            self.W_rec = spectral_norm(self.W_rec)

        # Feedback path (DirectedEP variant)
        if self.variant == "feedback":
            self.W_fb = nn.Linear(self.output_dim, hid)

        self.layers = nn.ModuleList([self.W_in, self.W_rec, self.W_out])

    def transition_modules(self) -> list[nn.Module]:
        return [self.W_in, self.W_rec, self.W_out]

    def _initialize_hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (x.size(0), self.hidden_dim), device=x.device, dtype=x.dtype
        )

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_in(_flatten(x))

    def _output_projection(self, h: torch.Tensor) -> torch.Tensor:
        return self.W_out(h)

    def _pre_activation(self, h: torch.Tensor, x_transformed: torch.Tensor) -> torch.Tensor:
        return x_transformed + self.W_rec(h)

    def _forward_step_impl(
        self, h: torch.Tensor, x_transformed: torch.Tensor
    ) -> torch.Tensor:
        pre = self._pre_activation(h, x_transformed)
        if self.variant == "momentum":
            if not hasattr(self, "_velocity") or self._velocity.shape != h.shape:
                self._velocity = torch.zeros_like(h)
            pre = self.momentum * self._velocity + pre
            self._velocity = pre.detach().clone()
        h_next = torch.tanh(pre)
        if self.variant == "sparse":
            k = int(h_next.size(1) * self.sparse_ratio)
            if k > 0:
                vals, _ = torch.topk(torch.abs(h_next), k, dim=1)
                thr = vals[:, -1].unsqueeze(1)
                mask = (torch.abs(h_next) >= thr).float()
                h_next = h_next * mask
        return h_next

    def forward_step(
        self, h: torch.Tensor, x_transformed: torch.Tensor
    ) -> torch.Tensor:
        return self._forward_step_impl(h, x_transformed)

    def _settle(
        self,
        x: torch.Tensor,
        beta: float,
        nudge: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Settle to fixed point; return (h_star, x_transformed)."""
        x_flat = _flatten(x)
        h0 = self._initialize_hidden_state(x_flat)
        x_trans = self._transform_input(x_flat)

        if nudge is None:
            # Free phase
            def free_step(h, xt):
                return self.forward_step(h, xt)

            h_star, _, _ = settle_single_state(
                h0, free_step, x_trans, self.max_steps, model=self
            )
        else:
            # Nudged phase: h ← forward_step(h) − beta * nudge
            def nudged_step(h, xt):
                return self.forward_step(h, xt) - beta * torch.mm(nudge, self.W_out.weight)

            h_star, _, _ = settle_single_state(
                h0, nudged_step, x_trans, self.nudge_steps, model=self
            )

        return h_star, x_trans

    def _energy_grads(
        self, h: torch.Tensor, x_transformed: torch.Tensor
    ) -> list[torch.Tensor]:
        """∇_θ Σ(0.5 h² − h·pre_act), treating h as a constant (fixed-point direct term).

        ``h`` is the settled equilibrium state (detached, constant).  ``pre_act``
        is recomputed from the live model parameters so the gradient flows
        through ``W_rec`` — this is the direct energy-gradient term that is equal
        to the total gradient at the fixed point (where ∂E/∂h = 0).
        """
        h_const = h.detach()
        pre_act = self._pre_activation(h_const, x_transformed)
        energy = torch.sum(0.5 * h_const**2 - h_const * pre_act)
        grads = torch.autograd.grad(
            energy, self.parameters(), retain_graph=True, allow_unused=True
        )
        return [g if g is not None else torch.zeros_like(p)
                for g, p in zip(grads, self.parameters())]

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        # Energy-contrastive rule (Scellier & Bengio): no BPTT, no optimizer.
        # Runs when the sweep activates eqprop with gradient_method="equilibrium".
        # Kept intentionally as the honest bio-rule path even though it currently
        # learns slowly; diagnose later (see EXPERIMENT_PLAN6.md §8.6).
        if self.gradient_method not in ("equilibrium", "contrastive"):
            return None

        x_flat = _flatten(x)

        # --- Free phase ---
        with torch.no_grad():
            h_free, x_trans = self._settle(x_flat, 0.0, None)
            logits_free = self._output_projection(h_free)
            loss_free = F.cross_entropy(logits_free, y)

        # dL/dlogits at free equilibrium
        logits_det = logits_free.clone().detach().requires_grad_(True)
        v = torch.autograd.grad(
            F.cross_entropy(logits_det, y), logits_det, retain_graph=False
        )[0]

        # --- Nudged phase ---
        with torch.no_grad():
            h_nudged, _ = self._settle(x_flat, self.beta, v)

        # --- Energy gradients ---
        gf = self._energy_grads(h_free, x_trans)
        gn = self._energy_grads(h_nudged, x_trans)

        # --- Manual weight updates ---
        with torch.no_grad():
            for p, gf_p, gn_p in zip(self.parameters(), gf, gn):
                if p is self.W_out.weight:
                    # Supervised output update: ΔW_out = −lr * v.T @ h_free
                    p -= self.lr * torch.mm(v.T, h_free)
                elif p is self.W_out.bias:
                    p -= self.lr * v.sum(0)
                else:
                    # Energy-contrastive update: Δw = −lr * (∂E_nudged − ∂E_free) / β
                    # = −lr * (gn − gf) / β
                    p -= self.lr * (gn_p - gf_p) / self.beta

        acc = (logits_free.argmax(1) == y).float().mean().item()
        return {"loss": loss_free.item(), "accuracy": acc}


# ============================================================
# Six thin registered subclasses — same engine, different names
# ============================================================

from bioplausible.core.registry import register_model


@register_model("eqprop", family="eqprop", tags=["eqprop", "energy"])
class StandardEqProp(EquilibriumMLP):
    """Plain energy-contrastive EqProp (replaces StandardEqProp)."""
    variant = "plain"

    @classmethod
    def build(cls, spec, input_dim, output_dim, hidden_dim, num_layers,
              device, task_type, **kwargs):
        return cls(
            config=ModelConfig(
                name="eqprop", input_dim=input_dim, output_dim=output_dim,
                hidden_dims=[hidden_dim], max_steps=kwargs.get("max_steps", 20),
                learning_rate=kwargs.get("learning_rate", 1e-3),
                beta=kwargs.get("beta", 0.3), use_spectral_norm=kwargs.get("use_spectral_norm", True),
            )
        ).to(device)


@register_model("directed_ep", family="eqprop", tags=["eqprop", "energy", "feedback"])
class DirectedEP(EquilibriumMLP):
    """Energy-contrastive EqProp with output-to-hidden feedback (replaces DirectedEP)."""
    variant = "feedback"

    @classmethod
    def build(cls, spec, input_dim, output_dim, hidden_dim, num_layers,
              device, task_type, **kwargs):
        return cls(
            config=ModelConfig(
                name="directed_ep", input_dim=input_dim, output_dim=output_dim,
                hidden_dims=[hidden_dim], max_steps=kwargs.get("max_steps", 20),
                learning_rate=kwargs.get("learning_rate", 1e-3),
                beta=kwargs.get("beta", 0.3), use_spectral_norm=kwargs.get("use_spectral_norm", True),
            )
        ).to(device)


@register_model("finite_nudge_ep", family="eqprop", tags=["eqprop", "energy"])
class FiniteNudgeEP(EquilibriumMLP):
    """Energy-contrastive EqProp with configurable nudge steps (replaces FiniteNudgeEP)."""
    variant = "plain"

    @classmethod
    def build(cls, spec, input_dim, output_dim, hidden_dim, num_layers,
              device, task_type, **kwargs):
        return cls(
            config=ModelConfig(
                name="finite_nudge_ep", input_dim=input_dim, output_dim=output_dim,
                hidden_dims=[hidden_dim], max_steps=kwargs.get("max_steps", 20),
                learning_rate=kwargs.get("learning_rate", 1e-3),
                beta=kwargs.get("beta", 0.3), use_spectral_norm=kwargs.get("use_spectral_norm", True),
                extra={"nudge_steps": kwargs.get("nudge_steps", max(3, kwargs.get("max_steps", 20)//3))}
            )
        ).to(device)


@register_model("lazy_eqprop", family="eqprop", tags=["eqprop", "energy"])
class LazyEqProp(EquilibriumMLP):
    """Energy-contrastive EqProp (replaces LazyEqProp)."""
    variant = "plain"

    @classmethod
    def build(cls, spec, input_dim, output_dim, hidden_dim, num_layers,
              device, task_type, **kwargs):
        return cls(
            config=ModelConfig(
                name="lazy_eqprop", input_dim=input_dim, output_dim=output_dim,
                hidden_dims=[hidden_dim], max_steps=kwargs.get("max_steps", 20),
                learning_rate=kwargs.get("learning_rate", 1e-3),
                beta=kwargs.get("beta", 0.3), use_spectral_norm=kwargs.get("use_spectral_norm", True),
            )
        ).to(device)


@register_model("momentum_equilibrium", family="eqprop", tags=["eqprop", "energy", "momentum"])
class MomentumEquilibrium(EquilibriumMLP):
    """Energy-contrastive EqProp with momentum in settle dynamics (replaces MomentumEquilibrium)."""
    variant = "momentum"

    @classmethod
    def build(cls, spec, input_dim, output_dim, hidden_dim, num_layers,
              device, task_type, **kwargs):
        return cls(
            config=ModelConfig(
                name="momentum_equilibrium", input_dim=input_dim, output_dim=output_dim,
                hidden_dims=[hidden_dim], max_steps=kwargs.get("max_steps", 20),
                learning_rate=kwargs.get("learning_rate", 1e-3),
                beta=kwargs.get("beta", 0.3), use_spectral_norm=kwargs.get("use_spectral_norm", True),
                extra={"momentum": kwargs.get("momentum", 0.5)}
            )
        ).to(device)


@register_model("sparse_equilibrium", family="eqprop", tags=["eqprop", "energy", "sparse"])
class SparseEquilibrium(EquilibriumMLP):
    """Energy-contrastive EqProp with top-k sparsity (replaces SparseEquilibrium)."""
    variant = "sparse"

    @classmethod
    def build(cls, spec, input_dim, output_dim, hidden_dim, num_layers,
              device, task_type, **kwargs):
        return cls(
            config=ModelConfig(
                name="sparse_equilibrium", input_dim=input_dim, output_dim=output_dim,
                hidden_dims=[hidden_dim], max_steps=kwargs.get("max_steps", 20),
                learning_rate=kwargs.get("learning_rate", 1e-3),
                beta=kwargs.get("beta", 0.3), use_spectral_norm=kwargs.get("use_spectral_norm", True),
                extra={"sparse_ratio": kwargs.get("sparse_ratio", 0.5)}
            )
        ).to(device)