"""Equilibrium Propagation model variants."""

import torch
import torch.nn.functional as F
from torch import nn

from bioplausible.core.registry import register_model

from ..transitions import TransitionGraphMixin
from .modern_conv_eqprop import SimpleConvEqProp

__all__ = [
    "EqPropDiffusion",
]


@register_model(
    "eqprop_diffusion",
    family="eqprop",
    tags=["eqprop", "diffusion"],
)
class EqPropDiffusion(TransitionGraphMixin, nn.Module):
    """
    Equilibrium Propagation Diffusion Model.

    Hypothesis: Denoising diffusion is energy minimization.
    Energy Formulation: E(x,t) = ||x - Denoise(x_t,t)||² + lambda R(x)

    This model predicts the clean image x_0 from x_t.
    """

    def __init__(
        self,
        img_channels=1,
        hidden_channels=64,
        gradient_method="bptt",
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs: dict | None = None,
    ):
        super().__init__()
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs or {"lr": 1e-3}
        self.denoiser = SimpleConvEqProp(
            input_channels=img_channels + 1,
            hidden_channels=hidden_channels,
            output_dim=img_channels,
            pool_output=False,
            use_spectral_norm=True,
            gradient_method=gradient_method,
        )
        self.img_channels = img_channels

        T = 1000
        self.T = T
        beta = torch.linspace(1e-4, 0.02, T)
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)
        posterior_variance = beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("alpha_bar_prev", alpha_bar_prev)
        self.register_buffer("posterior_variance", posterior_variance)

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
        """Build an ``EqPropDiffusion`` from explicit configuration.

        ``input_dim`` is the total flattened image size ``C * H * W``. Channel
        count is taken explicitly from ``img_channels`` (default 1) rather than
        reverse-engineered from magic pixel totals (784/3072, etc.). Spatial
        size is set via ``img_size`` (default 28); no other kwargs are passed
        through to the constructor.
        """
        img_channels = int(kwargs.get("img_channels", 1))
        return cls(
            img_channels=img_channels,
            hidden_channels=hidden_dim,
        ).to(device)

    def train_step(self, x, y=None):
        device = x.device
        batch_size = x.shape[0]

        t = torch.randint(0, self.T, (batch_size,), device=device).long()

        noise = torch.randn_like(x)
        sqrt_ab = torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1, 1)
        sqrt_omab = torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1, 1, 1)
        x_noisy = sqrt_ab * x + sqrt_omab * noise

        pred = self(x_noisy, t)

        loss = F.mse_loss(pred, x)

        if not hasattr(self, "optimizer") or not self.optimizer:
            self.optimizer = self.optimizer_class(
                self.parameters(), **self.optimizer_kwargs
            )

        if self.training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return {"loss": loss.item()}

    def val_step(self, x, y=None):
        device = x.device
        batch_size = x.shape[0]

        t = torch.randint(0, self.T, (batch_size,), device=device).long()

        noise = torch.randn_like(x)
        sqrt_ab = torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1, 1)
        sqrt_omab = torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1, 1, 1)
        x_noisy = sqrt_ab * x + sqrt_omab * noise

        pred = self(x_noisy, t)

        loss = F.mse_loss(pred, x).item()

        accuracy = 1.0 / (1.0 + loss)

        return {"loss": loss, "accuracy": accuracy}

    def predict_x0(self, x_t, t):
        batch_size, _, h, w = x_t.shape

        t_norm = t.float() / self.T
        t_emb = t_norm.view(batch_size, 1, 1, 1).expand(batch_size, 1, h, w)

        x_input = torch.cat([x_t, t_emb], dim=1)
        return self.denoiser(x_input)

    def denoise_step(self, x_t, t_norm, steps=30):
        batch_size, _, h, w = x_t.shape

        if t_norm.dim() == 1:
            t_emb = t_norm.view(batch_size, 1, 1, 1).expand(batch_size, 1, h, w)
        else:
            t_emb = t_norm.expand(batch_size, 1, h, w)

        x_input = torch.cat([x_t, t_emb], dim=1)

        return self.denoiser(x_input, steps=steps)

    def forward(self, x, t=None):
        if t is None:
            if x.shape[1] == self.img_channels + 1:
                return self.denoiser(x)
            raise ValueError("t must be provided for diffusion forward pass")

        return self.predict_x0(x, t)

    def transition_modules(self) -> list[nn.Module]:
        """Delegate to the internal denoiser's transition modules."""
        return self.denoiser.transition_modules()

    @torch.no_grad()
    def sample(self, num_samples=16, img_size=(1, 28, 28), device="cpu", steps=None):
        """Generate samples via the reverse diffusion process.

        Args:
            num_samples: Number of samples to generate.
            img_size: Image dimensions ``(C, H, W)``.
            device: Target device.
            steps: Number of reverse steps (default ``self.T``, i.e. full 1000).

        Returns:
            Tensor of shape ``(num_samples, C, H, W)`` with values in ``[-1, 1]``.
        """
        self.eval()
        B = num_samples
        C, H, W = img_size
        T = steps if steps is not None else self.T

        x = torch.randn(B, C, H, W, device=device)

        stride = self.T // T if T < self.T else 1
        timesteps = list(reversed(range(0, self.T, stride)))[:T]

        for i in timesteps:
            t = torch.full((B,), i, device=device, dtype=torch.long)

            x_0_pred = self.predict_x0(x, t)

            alpha_t = self.alpha[t].view(B, 1, 1, 1)
            alpha_bar_t = self.alpha_bar[t].view(B, 1, 1, 1)
            alpha_bar_prev_t = self.alpha_bar_prev[t].view(B, 1, 1, 1)
            beta_t = self.beta[t].view(B, 1, 1, 1)

            coeff1 = torch.sqrt(alpha_bar_prev_t) * beta_t / (1.0 - alpha_bar_t)
            coeff2 = (
                torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev_t) / (1.0 - alpha_bar_t)
            )

            mean = coeff1 * x_0_pred + coeff2 * x

            if i > 0:
                noise = torch.randn_like(x)
                var = self.posterior_variance[t].view(B, 1, 1, 1)
                sigma = torch.sqrt(var)
                x = mean + sigma * noise
            else:
                x = mean

        self.train()
        return x.clamp(-1, 1)
