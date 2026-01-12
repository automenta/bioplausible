import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict, Union
from abc import ABC, abstractmethod

class EqPropModel(nn.Module, ABC):
    """
    Abstract base class for Equilibrium Propagation models.

    Provides common functionality for:
    - Equilibrium iteration loop
    - Hidden state initialization
    - Trajectory tracking
    - Lipschitz constant computation
    - Noise injection for stability analysis
    """

    def __init__(self, max_steps: int = 30):
        super().__init__()
        self.max_steps = max_steps

    @abstractmethod
    def forward_step(self, h: torch.Tensor, x_transformed: torch.Tensor) -> torch.Tensor:
        """
        Single equilibrium iteration step.

        Args:
            h: Current hidden state
            x_transformed: Input data (possibly projected/embedded)

        Returns:
            Next hidden state
        """
        pass

    @abstractmethod
    def _initialize_hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        """
        Initialize the hidden state tensor based on input x.

        Args:
            x: Input tensor

        Returns:
            Initial hidden state (usually zeros)
        """
        pass

    @abstractmethod
    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform raw input x into the form used in the loop (e.g. projection).

        Args:
            x: Raw input tensor

        Returns:
            Transformed input tensor
        """
        pass

    @abstractmethod
    def _output_projection(self, h: torch.Tensor) -> torch.Tensor:
        """
        Project hidden state to output.

        Args:
            h: Hidden state

        Returns:
            Output (logits)
        """
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

        trajectory = [h] if return_trajectory else None

        # Iterate
        for _ in range(steps):
            h = self.forward_step(h, x_transformed)
            if return_trajectory:
                trajectory.append(h)

        out = self._output_projection(h)

        if return_trajectory:
            return out, trajectory
        return out

    def compute_lipschitz(self) -> float:
        """
        Compute the maximum Lipschitz constant across all layers.
        Default implementation checks all parameters for spectral norms.
        """
        max_L = 0.0
        with torch.no_grad():
            for module in self.modules():
                # Access .weight property if available.
                # This triggers spectral_norm forward hook if present.
                if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
                    w = module.weight
                    # Only consider weights that look like matrices or filters
                    if w.dim() >= 2:
                        w_mat = w.view(w.size(0), -1)
                        s = torch.linalg.svdvals(w_mat)
                        if s.numel() > 0:
                            max_L = max(max_L, s[0].item())
        return max_L

    def inject_noise_and_relax(
        self,
        x: torch.Tensor,
        noise_level: float = 1.0,
        injection_step: int = 15,
        total_steps: int = 30,
    ) -> Dict[str, float]:
        """
        Demonstrate self-healing: inject noise and measure damping.

        Args:
            x: Input tensor
            noise_level: Magnitude of injected noise
            injection_step: Step at which to inject noise
            total_steps: Total number of steps to run

        Returns:
            Dictionary containing noise metrics and damping information
        """
        h = self._initialize_hidden_state(x)
        x_transformed = self._transform_input(x)

        # Run to injection point
        for _ in range(injection_step):
            h = self.forward_step(h, x_transformed)

        # Inject noise
        h_clean = h.clone()
        h_noisy = h + torch.randn_like(h) * noise_level

        initial_noise_norm = (h_noisy - h_clean).norm().item() / h.numel()**0.5 # Normalized by size

        # Run remaining steps
        steps_remaining = total_steps - injection_step
        for _ in range(steps_remaining):
            h_noisy = self.forward_step(h_noisy, x_transformed)
            h_clean = self.forward_step(h_clean, x_transformed)

        final_noise_norm = (h_noisy - h_clean).norm().item() / h.numel()**0.5

        ratio = final_noise_norm / initial_noise_norm if initial_noise_norm > 1e-9 else 0.0

        return {
            'initial_noise': initial_noise_norm,
            'final_noise': final_noise_norm,
            'damping_ratio': ratio,
            'damping_percent': (1 - ratio) * 100,
        }
