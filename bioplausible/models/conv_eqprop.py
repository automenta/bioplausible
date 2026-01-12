import torch
import torch.nn as nn
from .utils import spectral_conv2d
from .eqprop_base import EqPropModel

# =============================================================================
# ConvEqProp - Convolutional EqProp for Vision Tasks
# =============================================================================

class ConvEqProp(EqPropModel):
    """
    Convolutional Equilibrium Propagation Model.

    Uses ResNet-like loop structure with spectral normalization.
    Suitable for image classification tasks (MNIST, CIFAR-10).

    Example:
        >>> model = ConvEqProp(1, 32, 10)  # MNIST
        >>> x = torch.randn(32, 1, 28, 28)
        >>> output = model(x, steps=25)  # [32, 10]
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        output_dim: int,
        gamma: float = 0.5,
        use_spectral_norm: bool = True,
        max_steps: int = 25
    ) -> None:
        super().__init__(max_steps=max_steps)
        self.hidden_channels = hidden_channels
        self.gamma = gamma

        # Input embedding
        self.embed = spectral_conv2d(
            input_channels, hidden_channels, kernel_size=3, padding=1,
            use_sn=use_spectral_norm
        )

        # Recurrent weights
        self.W1 = spectral_conv2d(
            hidden_channels, hidden_channels * 2, kernel_size=3, padding=1,
            use_sn=use_spectral_norm
        )
        self.W2 = spectral_conv2d(
            hidden_channels * 2, hidden_channels, kernel_size=3, padding=1,
            use_sn=use_spectral_norm
        )

        self.norm = nn.GroupNorm(8, hidden_channels)

        # Classifier head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_channels, output_dim)
        )

        # Initialize for stability
        with torch.no_grad():
            self.W1.weight.mul_(0.5)
            self.W2.weight.mul_(0.5)

    def _initialize_hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        """Initialize the hidden state tensor."""
        B, _, H, W = x.shape
        return torch.zeros(B, self.hidden_channels, H, W, device=x.device, dtype=x.dtype)

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        """Transform input: embed(x)"""
        return self.embed(x)

    def forward_step(self, h: torch.Tensor, x_transformed: torch.Tensor) -> torch.Tensor:
        """
        Single equilibrium iteration step.

        Args:
            h: Current hidden state
            x_transformed: Embedded input tensor (x_emb)

        Returns:
            Next hidden state
        """
        h_norm = self.norm(h)

        pre_act = self.W1(h_norm)
        hidden = torch.tanh(pre_act)
        ffn_out = self.W2(hidden)

        h_target = ffn_out + x_transformed
        # Use torch.lerp for more efficient interpolation
        h_next = torch.lerp(h, h_target, self.gamma)
        return h_next

    def _output_projection(self, h: torch.Tensor) -> torch.Tensor:
        """Output projection."""
        return self.head(h)
