import torch
import torch.nn as nn
import torch.autograd as autograd
from typing import Optional, List, Tuple, Dict, Union, Any
from abc import ABC, abstractmethod
from .nebc_base import NEBCBase


class EquilibriumFunction(autograd.Function):
    """
    Implicit differentiation for Equilibrium Propagation models.
    Allows O(1) memory training by computing gradients via fixed-point iteration
    instead of unrolling the graph (BPTT).
    """

    @staticmethod
    def forward(ctx, model, x_transformed, h_init, *params):
        ctx.model = model

        # 1. Find fixed point (no gradient tracking needed for the loop itself)
        with torch.no_grad():
            h = h_init
            for _ in range(model.max_steps):
                h = model.forward_step(h, x_transformed)

        # Save tensors for backward
        ctx.save_for_backward(h, x_transformed, *params)
        return h

    @staticmethod
    def backward(ctx, grad_output):
        h_star, x_transformed, *params = ctx.saved_tensors
        model = ctx.model

        # Capture training state
        was_training = model.training
        # Set to eval to prevent buffer updates (e.g. Spectral Norm) during backward
        model.eval()

        try:
            # 2. Compute adjoint state (delta) via fixed-point iteration
            # delta = VJP(f, h*, delta) + grad_output
            # This solves delta = (I - J)^(-T) @ grad_output

            delta = grad_output.clone()

            # Use detached X for the VJP loop to avoid any graph entanglement
            x_transformed_detached = x_transformed.detach()

            # Iterate to equilibrium for the backward pass (solving for delta)
            for _ in range(model.max_steps):
                with torch.enable_grad():
                    # Need to create a new leaf for h_star at each step for local VJP calc
                    # And x_transformed is detached constant.
                    h_star_loop = h_star.detach().requires_grad_(True)
                    f_h = model.forward_step(h_star_loop, x_transformed_detached)

                    # VJP: v = grad(f(h), h) @ delta
                    # delta must be detached to stop graph from growing across iterations
                    # retain_graph=False ensures we free the f_h graph immediately.

                    vjp = autograd.grad(
                        f_h,
                        h_star_loop,
                        grad_outputs=delta.detach(),
                        retain_graph=False,
                    )[0]

                    # Update delta
                    delta = vjp + grad_output

            # 3. Compute gradients for parameters and input
            # dL/d(params) = grad(f(h*), params) @ delta
            # dL/dx = grad(f(h*), x) @ delta

            delta = delta.detach()

            # IMPORTANT: We must NOT use torch.enable_grad() here for x_transformed if it's from saved_tensors
            # because it belongs to the FORWARD graph, which is immutable now.
            # However, we need to create a NEW graph connecting (params, x_transformed) -> f_h

            with torch.enable_grad():
                # Re-compute one step to form graph connecting params and x to output
                # We use the original x_transformed (attached to graph) here

                # CRITICAL FIX: We must detach h_star!
                h_star_detached = h_star.detach()

                f_h = model.forward_step(h_star_detached, x_transformed)

                inputs = list(params)
                if x_transformed.requires_grad:
                    inputs.append(x_transformed)

                # Use retain_graph=True to avoid freeing graph buffers that might be needed
                # if autograd needs to access x_transformed's history later?
                # Actually, for x_transformed, we are at the "leaf" of this local graph.
                # But x_transformed itself is an intermediate node in the global graph.
                grads = autograd.grad(
                    f_h,
                    inputs,
                    grad_outputs=delta,
                    allow_unused=True,
                    retain_graph=True,
                )

                grad_params = grads[: len(params)]
                if x_transformed.requires_grad:
                    grad_x = grads[-1]
                else:
                    grad_x = None

        finally:
            # Restore training state
            model.train(was_training)

        return (None, grad_x, None, *grad_params)


class EqPropModel(NEBCBase):
    """
    Abstract base class for Equilibrium Propagation models.
    """

    def __init__(self, max_steps: int = 30, gradient_method: str = "bptt", **kwargs):
        """
        Args:
            max_steps: Number of equilibrium steps
            gradient_method: 'bptt' (default) or 'equilibrium' (O(1) memory implicit diff)
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

    def forward(
        self,
        x: torch.Tensor,
        steps: Optional[int] = None,
        return_trajectory: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass: iterate to equilibrium.

        Args:
            x: Input tensor
            steps: Override number of iteration steps
            return_trajectory: If True, return all hidden states

        Returns:
            Output logits
            (optionally) trajectory of hidden states
        """
        steps = steps or self.max_steps

        # Initialize
        h = self._initialize_hidden_state(x)
        x_transformed = self._transform_input(x)

        if return_trajectory or self.gradient_method == "bptt":
            # Standard unrolling (BPTT)
            trajectory = [h] if return_trajectory else None
            for _ in range(steps):
                h = self.forward_step(h, x_transformed)
                if return_trajectory:
                    trajectory.append(h)

            out = self._output_projection(h)
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
