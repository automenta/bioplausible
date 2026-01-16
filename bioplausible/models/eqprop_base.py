import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from typing import Optional, List, Tuple, Dict, Union, Any
from abc import abstractmethod
from .nebc_base import NEBCBase


class EquilibriumFunction(autograd.Function):
    """
    Implicit differentiation for Equilibrium Propagation models.

    Implements O(1) memory backpropagation using the equilibrium property:
    dL/dtheta = dL/dh * dh/dtheta
    where dh/dtheta = (I - J)^-1 * df/dtheta

    The backward pass solves for the adjoint state delta:
    delta = (I - J^T)^-1 * dL/dh
    via fixed-point iteration:
    delta_{t+1} = J^T * delta_t + dL/dh
    """

    @staticmethod
    def forward(
        ctx: Any,
        model: nn.Module,
        x_transformed: torch.Tensor,
        h_init: torch.Tensor,
        *params: torch.Tensor
    ) -> torch.Tensor:
        ctx.model = model

        # 1. Find fixed point (no gradient tracking needed for the loop itself)
        # We assume h_init is close to the fixed point if we are continuing from previous state,
        # or we iterate enough steps to converge.
        with torch.no_grad():
            h = h_init
            for _ in range(model.max_steps):
                h = model.forward_step(h, x_transformed)

        # Save tensors for backward
        # Note: We must save params to ensure autograd knows they participate in the graph
        ctx.save_for_backward(h, x_transformed, *params)
        return h

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        h_star, x_transformed, *params = ctx.saved_tensors
        model = ctx.model

        # Capture training state
        was_training = model.training
        # Set to eval to prevent buffer updates (e.g. Spectral Norm) during backward fixed-point iteration
        # This is critical because Spectral Norm updates 'u' and 'v' buffers in .train() mode,
        # which would cause in-place modification errors or incorrect gradients during the backward loop.
        model.eval()

        try:
            # 2. Compute adjoint state (delta) via fixed-point iteration
            # Initial guess for delta is dL/dh (grad_output)
            delta = grad_output.clone()

            # Use detached X for the VJP loop to avoid any graph entanglement with input gradients yet
            x_transformed_detached = x_transformed.detach()

            # Check if model has _forward_step_impl (uncompiled) to avoid torch.compile overhead in loop
            forward_fn = getattr(model, "_forward_step_impl", model.forward_step)

            # Iterate to equilibrium for the backward pass (solving for delta)
            # delta_{t+1} = (df/dh)^T * delta_t + grad_output
            for _ in range(model.max_steps):
                with torch.enable_grad():
                    # Create a new leaf for h_star at each step for local VJP calc
                    h_star_loop = h_star.detach().requires_grad_(True)

                    # Compute f(h, x)
                    f_h = forward_fn(h_star_loop, x_transformed_detached)

                    # VJP: v = (df/dh)^T @ delta
                    # retain_graph=False ensures we free the f_h graph immediately.
                    # We detach delta because for the purpose of the VJP, delta is a constant vector.
                    vjp = autograd.grad(
                        f_h,
                        h_star_loop,
                        grad_outputs=delta.detach(),
                        retain_graph=False,
                        create_graph=False
                    )[0]

                    # Update delta
                    # Crucial: detach delta to prevent graph growth during the fixed-point iteration
                    # The VJP loop is purely for finding the value of the adjoint state.
                    delta = (vjp + grad_output).detach()

            # 3. Compute gradients for parameters and input using the converged delta
            delta = delta.detach()

            with torch.enable_grad():
                h_star_detached = h_star.detach()

                # A. Compute gradients for parameters
                # dL/dtheta = (df/dtheta)^T @ delta

                # CRITICAL: Detach x_transformed here.
                # If we don't detach, autograd will trace d(f_h)/d(x) * d(x)/d(theta)
                # effectively double-counting the gradient for params that affect x_transformed.
                x_detached = x_transformed.detach()

                params_with_grad = [p for p in params if p.requires_grad]
                grads_params_list = [None] * len(params)

                if params_with_grad:
                    # Re-run forward step to build graph from params to f_h
                    # Use uncompiled function here too for consistency.
                    f_h_params = forward_fn(h_star_detached, x_detached)

                    computed_grads = autograd.grad(
                        f_h_params,
                        params,
                        grad_outputs=delta,
                        allow_unused=True,
                        retain_graph=False
                    )
                    grads_params_list = list(computed_grads)

                # B. Compute gradients for input (x_transformed)
                # dL/dx = (df/dx)^T @ delta
                grad_x = None
                if x_transformed.requires_grad:
                     # Use attached x_transformed to get gradients w.r.t input
                     f_h_x = model.forward_step(h_star_detached, x_transformed)
                     grad_x = autograd.grad(
                         f_h_x,
                         x_transformed,
                         grad_outputs=delta,
                         retain_graph=False
                     )[0]

        finally:
            # Restore original training state
            model.train(was_training)

        # Return gradients corresponding to inputs of forward:
        # ctx, model, x_transformed, h_init, *params
        # model and h_init don't get gradients
        return (None, grad_x, None, *grads_params_list)


class EqPropModel(NEBCBase):
    """
    Abstract base class for Equilibrium Propagation models.
    """

    def __init__(self, max_steps: int = 30, gradient_method: str = "bptt", **kwargs):
        """
        Args:
            max_steps: Number of equilibrium steps
            gradient_method: 'bptt', 'equilibrium' (implicit diff), or 'contrastive' (Hebbian)
        """
        input_dim = kwargs.get("input_dim", 0)
        hidden_dim = kwargs.get("hidden_dim", 0)
        output_dim = kwargs.get("output_dim", 0)

        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            max_steps=max_steps,
            use_spectral_norm=kwargs.get("use_spectral_norm", True),
        )
        self.max_steps = max_steps
        self.gradient_method = gradient_method

        # Contrastive Hebbian specific params
        self.beta = kwargs.get("beta", 0.1)
        self.hebbian_lr = kwargs.get("learning_rate", 0.001)
        self.internal_optimizer = None

    @abstractmethod
    def _build_layers(self):
        """Build layers. Required by NEBCBase, implemented by subclasses."""
        pass

    @abstractmethod
    def forward_step(
        self, h: torch.Tensor, x_transformed: torch.Tensor
    ) -> torch.Tensor:
        """Single equilibrium iteration step."""
        pass

    @abstractmethod
    def _initialize_hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        """Initialize the hidden state tensor based on input x."""
        pass

    @abstractmethod
    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        """Transform raw input x into the form used in the loop."""
        pass

    @abstractmethod
    def _output_projection(self, h: torch.Tensor) -> torch.Tensor:
        """Project hidden state to output."""
        pass

    def contrastive_update(
        self,
        h_free: torch.Tensor,
        h_nudged: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
    ):
        """
        Perform contrastive Hebbian update.
        Must be implemented by subclasses if gradient_method='contrastive'.
        """
        raise NotImplementedError("Subclasses must implement contrastive_update for Hebbian learning.")

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """
        Perform a single training step.
        If gradient_method is 'contrastive', this runs the EqProp loop manually.
        Otherwise, it returns None to let SupervisedTrainer handle BPTT/Implicit.
        """
        if self.gradient_method != "contrastive":
            return None  # Delegate to standard trainer

        # Initialize optimizer on first call
        if self.internal_optimizer is None:
            self.internal_optimizer = torch.optim.Adam(self.parameters(), lr=self.hebbian_lr)

        self.internal_optimizer.zero_grad()

        # 1. Free Phase
        with torch.no_grad():
            h_free = self._initialize_hidden_state(x)
            x_transformed = self._transform_input(x)

            for _ in range(self.max_steps):
                h_free = self.forward_step(h_free, x_transformed)

            logits_free = self._output_projection(h_free)

        # 2. Nudged Phase
        # We need to compute gradients of the loss w.r.t h to nudge
        # But for 'contrastive', we typically nudge via a top-down drive or explicit gradient injection

        # Enable grad just for the nudge calculation
        h_nudged = h_free.clone().detach().requires_grad_(True)

        # Run one step to connect h to output (if needed) or just project
        # Ideally we settle in the nudged phase with a constant nudge.
        # Nudge term: - beta * dL/dh

        # Calculate dL/dh at equilibrium
        logits_nudge_init = self._output_projection(h_nudged)
        loss = F.cross_entropy(logits_nudge_init, y)
        grads_h = autograd.grad(loss, h_nudged)[0]

        # Nudged dynamics: h <- forward_step(h) - beta * dL/dh
        # Note: In continuous time, dot_h = -h + sigma(...) - beta * dL/dh
        # In discrete step: h_new = forward_step(h) - beta * dL/dh

        # We perform fixed point iteration with the nudge
        # Nudge should be constant if dL/dh is approx constant locally, or updated?
        # Standard EqProp keeps the nudge target fixed (y) but dL/dh changes as h changes.

        with torch.no_grad():
            h_nudged = h_free.clone()

            # Simple implementation: Apply constant nudge derived from free phase error?
            # Or recompute nudge each step?
            # Scellier 2017: weakly clamp output units.
            # Here output is a projection. We inject gradient.

            # We'll use a constant nudge vector derived from free phase for stability/speed
            nudge_vec = -self.beta * grads_h

            for _ in range(self.max_steps // 2): # Typically fewer steps for nudged phase
                # h = f(h) + nudge
                h_next = self.forward_step(h_nudged, x_transformed)
                h_nudged = h_next + nudge_vec

            logits_nudged = self._output_projection(h_nudged)

        # 3. Weight Update
        self.contrastive_update(h_free, h_nudged, x, y)

        self.internal_optimizer.step()

        # Compute metrics
        with torch.no_grad():
            acc = (logits_free.argmax(dim=1) == y).float().mean().item()
            loss_val = F.cross_entropy(logits_free, y).item()

        return {"loss": loss_val, "accuracy": acc}

    def forward(
        self,
        x: torch.Tensor,
        steps: Optional[int] = None,
        return_trajectory: bool = False,
        return_dynamics: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, List[torch.Tensor]],
        Tuple[torch.Tensor, Dict[str, Any]],
    ]:
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

        if return_trajectory or return_dynamics or self.gradient_method == "bptt":
            # Standard unrolling (BPTT or Analysis)
            trajectory = [h]
            deltas = []

            for _ in range(steps):
                h_new = self.forward_step(h, x_transformed)

                if return_dynamics:
                    # Compute change norm
                    delta = (h_new - h).norm().item()
                    deltas.append(delta)

                h = h_new
                if return_trajectory:
                    trajectory.append(h)

            out = self._output_projection(h)

            if return_dynamics:
                return out, {
                    "trajectory": trajectory if return_trajectory else None,
                    "deltas": deltas,
                    "final_delta": deltas[-1] if deltas else 0.0
                }

            if return_trajectory:
                return out, trajectory
            return out

        elif self.gradient_method == "equilibrium":
            # O(1) memory implicit differentiation
            # We must pass params to apply so they are captured by ctx for backward
            # Note: We use list(self.parameters()) to get all parameters including weight_orig
            params = list(self.parameters())
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
    ) -> Dict[str, float]:
        """Demonstrate self-healing: inject noise and measure damping."""
        h = self._initialize_hidden_state(x)
        x_transformed = self._transform_input(x)

        # Run to injection point
        for _ in range(injection_step):
            h = self.forward_step(h, x_transformed)

        # Inject noise
        h_clean = h.clone()
        h_noisy = h + torch.randn_like(h) * noise_level

        initial_noise_norm = (h_noisy - h_clean).norm().item() / h.numel() ** 0.5

        # Run remaining steps
        steps_remaining = total_steps - injection_step
        for _ in range(steps_remaining):
            h_noisy = self.forward_step(h_noisy, x_transformed)
            h_clean = self.forward_step(h_clean, x_transformed)

        final_noise_norm = (h_noisy - h_clean).norm().item() / h.numel() ** 0.5

        ratio = (
            final_noise_norm / initial_noise_norm if initial_noise_norm > 1e-9 else 0.0
        )

        return {
            "initial_noise": initial_noise_norm,
            "final_noise": final_noise_norm,
            "damping_ratio": ratio,
            "damping_percent": (1 - ratio) * 100,
        }
