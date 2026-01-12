import torch
import torch.nn as nn
from .utils import spectral_conv2d

# =============================================================================
# ConvEqProp - Convolutional EqProp for Vision Tasks
# =============================================================================

class ConvEqProp(nn.Module):
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
        use_spectral_norm: bool = True
    ) -> None:
        super().__init__()
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

    def forward_step(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Single equilibrium iteration step.

        Args:
            h: Current hidden state
            x: Input tensor

        Returns:
            Next hidden state
        """
        h_norm = self.norm(h)
        x_emb = self.embed(x)

        pre_act = self.W1(h_norm)
        hidden = torch.tanh(pre_act)
        ffn_out = self.W2(hidden)

        h_target = ffn_out + x_emb
        # Use torch.lerp for more efficient interpolation
        h_next = torch.lerp(h, h_target, self.gamma)
        return h_next

    def forward(self, x: torch.Tensor, steps: int = 25) -> torch.Tensor:
        """
        Forward pass: iterate to equilibrium.

        Args:
            x: Input tensor [batch, channels, height, width]
            steps: Number of equilibrium steps

        Returns:
            Output logits [batch, output_dim]
        """
        B, _, H, W = x.shape
        h = self._create_hidden_state_tensor(B, H, W, x)

        for _ in range(steps):
            h = self.forward_step(h, x)

        return self.head(h)

    def _create_hidden_state_tensor(self, batch_size: int, height: int, width: int, reference_tensor: torch.Tensor) -> torch.Tensor:
        """Create the initial hidden state tensor for ConvEqProp.

        Args:
            batch_size: Size of the batch dimension
            height: Height of the spatial dimensions
            width: Width of the spatial dimensions
            reference_tensor: Reference tensor to get device from

        Returns:
            Initialized hidden state tensor
        """
        return torch.zeros(batch_size, self.hidden_channels, height, width, device=reference_tensor.device)
