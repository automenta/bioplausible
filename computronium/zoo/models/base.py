from abc import abstractmethod

import torch
import torch.nn.functional as F
from torch import autograd, nn

from computronium.config.unified import ModelConfig
from computronium.core.local_learning.settling import (
    EquilibriumFunction,
    settle_single_state,
)
from computronium.core.logging import get_logger
from computronium.core.losses import compute_accuracy
from computronium.core.model import BioModel
from computronium.core.utils.optimizer import OptimizerConfig, create_optimizer

__all__ = [
    "EqPropModel",
    "logger",
]
logger = get_logger()


class EqPropModel(BioModel):
    """
    Abstract base class for Equilibrium Propagation models.
    """

    def __init__(
        self,
        config: ModelConfig | None = None,
        max_steps: int = 30,
        gradient_method: str = "equilibrium",
        **kwargs,
    ):
        """
        Args:
            config: Optional ``ModelConfig`` (preferred).  If given,
                ``input_dim``, ``hidden_dims``, ``output_dim``,
                ``max_steps``, ``use_spectral_norm``, and
                ``lipschitz_mode`` are extracted from the config.
                ``gradient_method`` falls back to ``config.extra``.
            max_steps: Number of equilibrium steps (ignored if ``config``
                is given).
            gradient_method: ``'bptt'``, ``'equilibrium'`` (implicit, O(1)
                memory — the default), or ``'contrastive'`` (Hebbian).
        """
        if config is not None:
            input_dim = config.input_dim
            hidden_dim = config.hidden_dims[0] if config.hidden_dims else 0
            output_dim = config.output_dim
            use_spectral_norm = config.use_spectral_norm
            lipschitz_mode = config.lipschitz_mode
            max_steps = config.max_steps
            gm = config.extra.get("gradient_method")
            gradient_method = gm if isinstance(gm, str) else gradient_method
            beta = config.beta
        else:
            input_dim = kwargs.pop("input_dim", 0)
            hidden_dim = kwargs.pop("hidden_dim", 0)
            output_dim = kwargs.pop("output_dim", 0)
            use_spectral_norm = kwargs.pop("use_spectral_norm", True)
            lipschitz_mode = kwargs.pop("lipschitz_mode", "power_iteration")
            beta = kwargs.get("beta", 0.1)

        super().__init__(
            config=config,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            max_steps=max_steps,
            use_spectral_norm=use_spectral_norm,
            lipschitz_mode=lipschitz_mode,
            **kwargs,
        )
        self.gradient_method = gradient_method
        self.beta = beta
        self.hebbian_lr = kwargs.get("learning_rate", 0.001)
        self.internal_optimizer = None

    @abstractmethod
    def _build_layers(self):
        """Build layers. Required by NEBCBase/BioModel, implemented by subclasses."""

    @abstractmethod
    def forward_step(
        self, h: torch.Tensor, x_transformed: torch.Tensor
    ) -> torch.Tensor:
        """Single equilibrium iteration step."""

    @abstractmethod
    def _initialize_hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        """Initialize the hidden state tensor based on input x."""

    @abstractmethod
    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        """Transform raw input x into the form used in the loop."""

    @abstractmethod
    def _output_projection(self, h: torch.Tensor) -> torch.Tensor:
        """Project hidden state to output."""

    def get_hebbian_pairs(
        self, h: torch.Tensor, x: torch.Tensor
    ) -> list[tuple[nn.Module, torch.Tensor, torch.Tensor]]:
        """
        Return list of (layer_module, input, output_target) for Hebbian updates.

        This defines the topology for contrastive learning.
        For a layer y = f(W, u), we typically return (layer, u, y).
        The generic update will compute gradients of (layer(u) * y).sum().

        Args:
            h: Hidden state at equilibrium
            x: Raw input

        Returns:
            List of tuples: (layer, input_to_layer, target_output_of_layer)
        """
        raise NotImplementedError(
            "Subclasses must implement get_hebbian_pairs for contrastive learning."
        )

    def contrastive_update(
        self,
        h_free: torch.Tensor,
        h_nudged: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
    ):
        """
        Perform generic contrastive Hebbian update using 'get_hebbian_pairs'.

        Implements: Delta W ~ grad(Layer(x) @ y_nudged) - grad(Layer(x) @ y_free)
        """
        batch_size = x.shape[0]
        scale = 1.0 / (self.beta * batch_size)

        # 1. Get pairs for Free and Nudged states
        # Note: We recompute 'transform_input' if needed, but 'get_hebbian_pairs'
        # usually takes raw x and h.
        pairs_free = self.get_hebbian_pairs(h_free, x)
        pairs_nudged = self.get_hebbian_pairs(h_nudged, x)

        # 2. Aggregate Proxy Losses and Compute Gradients Once
        # Optimization: Sum proxy losses to reduce autograd overhead
        total_loss_free = 0.0
        total_loss_nudged = 0.0

        for (layer, inp_f, tgt_f), (_, inp_n, tgt_n) in zip(pairs_free, pairs_nudged):
            # Free Phase
            # Detach inputs to prevent backprop through layers (preserve local learning)
            out_f = layer(inp_f.detach())
            total_loss_free = total_loss_free + torch.sum(out_f * tgt_f.detach())

            # Nudged Phase
            out_n = layer(inp_n.detach())
            total_loss_nudged = total_loss_nudged + torch.sum(out_n * tgt_n.detach())

        # Compute gradients for all parameters at once
        params = list(self.parameters())
        grads_f = autograd.grad(
            total_loss_free, params, retain_graph=True, allow_unused=True
        )
        grads_n = autograd.grad(
            total_loss_nudged, params, retain_graph=True, allow_unused=True
        )

        # Apply update
        for param, gf, gn in zip(params, grads_f, grads_n):
            if param.requires_grad:
                # Delta W ~ (Nudged - Free)
                g_update = 0.0
                if gn is not None:
                    g_update += gn
                if gf is not None:
                    g_update -= gf

                if isinstance(g_update, float) and g_update == 0.0:
                    continue

                grad_term = scale * g_update

                if param.grad is None:
                    param.grad = grad_term
                else:
                    param.grad.add_(grad_term)

        # 3. Output Layer (Standard Backprop on Nudged or Free?)
        # Standard EqProp: W_out update is just gradient of
        # Cost function at Free phase.
        logits = self._output_projection(h_free)
        loss = F.cross_entropy(logits, y)

        # Update W_out (supervised component).
        # We use autograd.grad on loss, but only apply it to parameters
        # that haven't been updated by the Hebbian phase (i.e., parameters
        # with .grad is None). This assumes W_out is not part of the
        # Hebbian dynamics.

        grads_loss = autograd.grad(loss, self.parameters(), allow_unused=True)
        for param, g in zip(self.parameters(), grads_loss):
            if g is not None:
                if param.grad is None:
                    # This param wasn't updated by Hebbian loop -> Must
                    # be W_out or similar
                    param.grad = g
                else:
                    # Already has Hebbian grad -> Do not add Loss grad
                    # (unless hybrid?). Pure EqProp: Internal weights only
                    # update via Hebbian.
                    pass

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        """
        Perform a single training step.
        If gradient_method is 'contrastive', this runs the EqProp loop manually.
        Otherwise, it returns None to let CoreTrainer handle BPTT/Implicit.
        """
        if self.gradient_method != "contrastive":
            return None  # Delegate to standard trainer

        # Initialize optimizer on first call
        if self.internal_optimizer is None:
            self.internal_optimizer = create_optimizer(
                self, OptimizerConfig(name="adam", lr=self.hebbian_lr, weight_decay=0.0)
            )

        self.internal_optimizer.zero_grad()

        # 1. Free Phase
        with torch.no_grad():
            h_free = self._initialize_hidden_state(x)
            x_transformed = self._transform_input(x)

            for _ in range(self.max_steps):
                h_free = self.forward_step(h_free, x_transformed)

            logits_free = self._output_projection(h_free)

        # 2. Nudged Phase
        # We need to compute gradients of the loss w.r.t h to nudge.
        # But for 'contrastive', we typically nudge via a top-down drive
        # or explicit gradient injection.

        # Enable grad just for the nudge calculation
        h_nudged = h_free.clone().detach().requires_grad_(True)

        # Run one step to connect h to output (if needed) or just project
        # Ideally we settle in the nudged phase with a constant nudge.
        # Nudge term: - beta * dL/dh

        # Calculate dL/dh at equilibrium
        logits_nudge_init = self._output_projection(h_nudged)
        loss = F.cross_entropy(logits_nudge_init, y)
        grads_h = autograd.grad(loss, h_nudged)[0]

        # Stability Check 1: Gradients
        if torch.isnan(grads_h).any() or torch.isinf(grads_h).any():
            raise RuntimeError(
                "EqProp divergence detected (NaN/Inf gradients) — training unstable, check learning rate/beta"
            )

        # Nudged dynamics: h <- forward_step(h) - beta * dL/dh
        # Note: In continuous time, dot_h = -h + sigma(...)
        # - beta * dL/dh. In discrete step: h_new = forward_step(h)
        # - beta * dL/dh.

        # We perform fixed point iteration with the nudge
        # Nudge should be constant if dL/dh is approx constant
        # locally, or updated? Standard EqProp keeps the nudge target
        # fixed (y) but dL/dh changes as h changes.

        with torch.no_grad():
            h_nudged = h_free.clone()

            # Simple implementation: Apply constant nudge derived from free phase error?
            # Or recompute nudge each step?
            # Scellier 2017: weakly clamp output units.
            # Here output is a projection. We inject gradient.

            # We'll use a constant nudge vector derived from free phase
            # for stability/speed
            nudge_vec = -self.beta * grads_h

            for _ in range(
                self.max_steps // 2
            ):  # Typically fewer steps for nudged phase
                # h = f(h) + nudge
                h_next = self.forward_step(h_nudged, x_transformed)
                h_nudged = h_next + nudge_vec

        # 3. Weight Update
        self.contrastive_update(h_free, h_nudged, x, y)

        self.internal_optimizer.step()

        # Compute metrics
        with torch.no_grad():
            if torch.isnan(logits_free).any():
                raise RuntimeError(
                    "Model collapse (NaN logits) — check weight initialization or gradient clipping"
                )
            else:
                acc = compute_accuracy(logits_free, y)
                loss_val = F.cross_entropy(logits_free, y).item()

        return {"loss": loss_val, "accuracy": acc}

    def forward(
        self,
        x: torch.Tensor,
        steps: int | None = None,
        return_trajectory: bool = False,
        return_dynamics: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, list[torch.Tensor]]
        | tuple[torch.Tensor, dict[str, object]]
    ):
        """
        Forward pass: iterate to equilibrium.

        Args:
            x: Input tensor
            steps: Override number of iteration steps
            return_trajectory: If True, return all hidden states
            return_dynamics: If True, return detailed convergence metrics

        Returns:
            Output logits
            (optionally) trajectory of hidden states or dynamics dict
        """
        steps = steps or self.max_steps

        # Initialize
        h = self._initialize_hidden_state(x)
        x_transformed = self._transform_input(x)

        if (
            return_trajectory
            or return_dynamics
            or self.gradient_method in ["bptt", "contrastive"]
        ):
            # Standard unrolling (BPTT, Analysis, or Contrastive Inference)
            h, trajectory, dynamics = settle_single_state(
                h_0=h,
                forward_step=self.forward_step,
                x_transformed=x_transformed,
                steps=steps,
                model=self,
                return_trajectory=return_trajectory,
                return_dynamics=return_dynamics,
            )

            out = self._output_projection(h)

            if return_dynamics:
                return out, dynamics
            if return_trajectory:
                return out, trajectory
            return out

        elif self.gradient_method == "equilibrium":
            # O(1) memory implicit differentiation
            # We must pass params to apply so they are captured by ctx for
            # backward. Note: We use list(self.parameters()) to get all
            # parameters including weight_orig. Only trainable params are
            # forwarded: passing a `requires_grad=False` param (e.g. the fixed
            # random feedback `B_out` in EquilibriumAlignment) makes the adjoint
            # `autograd.grad` raise "One of the differentiated Tensors does not
            # require grad".
            params = [p for p in self.parameters() if p.requires_grad]
            h_star = EquilibriumFunction.apply(self, x_transformed, h, *params)
            out = self._output_projection(h_star)
            return out

        else:
            raise ValueError(f"Unknown gradient_method: {self.gradient_method}")

    def inject_noise_and_relax(
        self,
        x: torch.Tensor,
        noise_level: float = 1.0,
        injection_step: int = 15,
        total_steps: int = 30,
    ) -> dict[str, float]:
        """Demonstrate self-healing: inject noise and measure damping."""
        h = self._initialize_hidden_state(x)
        x_transformed = self._transform_input(x)

        # Run to injection point
        for _ in range(injection_step):
            h = self.forward_step(h, x_transformed)

        # Inject noise
        h_clean = h.clone()
        h_noisy = h + torch.randn_like(h) * noise_level

        # Use torch.dist(p=2) instead of manually computing diff.norm()
        initial_noise_norm = torch.dist(h_noisy, h_clean, p=2).item() / h.numel() ** 0.5

        # Run remaining steps
        steps_remaining = total_steps - injection_step
        for _ in range(steps_remaining):
            h_noisy = self.forward_step(h_noisy, x_transformed)
            h_clean = self.forward_step(h_clean, x_transformed)

        final_noise_norm = torch.dist(h_noisy, h_clean, p=2).item() / h.numel() ** 0.5

        ratio = (
            final_noise_norm / initial_noise_norm if initial_noise_norm > 1e-9 else 0.0
        )

        return {
            "initial_noise": initial_noise_norm,
            "final_noise": final_noise_norm,
            "damping_ratio": ratio,
            "damping_percent": (1 - ratio) * 100,
        }
