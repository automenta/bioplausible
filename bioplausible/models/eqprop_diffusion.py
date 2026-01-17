import torch
import torch.nn as nn
from bioplausible.models import SimpleConvEqProp


class EqPropDiffusion(nn.Module):
    """
    Equilibrium Propagation Diffusion Model.

    Hypothesis: Denoising diffusion is energy minimization.
    Energy Formulation: E(x,t) = ||x - Denoise(x_t,t)||² + λR(x)
    """

    def __init__(self, img_channels=1, hidden_channels=64):
        super().__init__()
        # Input channels + 1 for time embedding (concatenated as a channel)
        # Using SimpleConvEqProp allows Triton acceleration
        # Use pool_output=False for spatial output (Dense prediction)
        self.denoiser = SimpleConvEqProp(
            input_channels=img_channels + 1,
            hidden_channels=hidden_channels,
            output_dim=img_channels,  # Output channels
            pool_output=False,
            use_spectral_norm=True,
        )
        self.img_channels = img_channels

    def energy(self, x_noisy, x_pred, t):
        """Energy function for denoising."""
        # Simple reconstruction error energy
        recon_error = ((x_noisy - x_pred) ** 2).sum()
        # Simple prior: penalize high-frequency noise (Total Variation)
        prior = self.total_variation(x_pred)
        return recon_error + 0.1 * prior

    def total_variation(self, x):
        """Smoothness prior."""
        dx = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().sum()
        dy = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().sum()
        return dx + dy

    def denoise_step(self, x_noisy, t, steps=10):
        """Single denoising step via equilibrium."""
        batch_size, _, h, w = x_noisy.shape

        # Embed time: simple broadcast of t
        # t is expected to be [batch_size]
        t_emb = t.view(batch_size, 1, 1, 1).expand(batch_size, 1, h, w)

        # Concatenate noisy image and time embedding
        x_input = torch.cat([x_noisy, t_emb], dim=1)

        # In a real EqProp diffusion, we might iterate to equilibrium *on the input* x_noisy
        # or use the network to predict the clean image and then take an energy step.
        # Here we follow the simplified TODO.md implementation:
        # Use the network to predict the clean image (or noise), then relax towards it.

        # Run forward pass of EqProp network to get prediction
        h_pred = self.denoiser(x_input)
        # Output is already [B, C, H, W] if pool_output=False

        # Refine prediction via energy gradient (simplified)
        # In full EqProp, this happens inside the layers.
        # Here we simulate an outer loop energy minimization if needed,
        # but the TODO spec implies the "denoise step" *is* the equilibrium process.
        # However, ConvEqProp itself has an internal equilibrium loop.
        # So h_pred is already the "equilibrium" output of the network.

        # If we want to implement the specific logic from TODO.md:
        # h = x_noisy
        # for _ in range(steps):
        #     h_pred = self.denoiser(x_input) ...
        #     grad = 2 * (h - h_pred)
        #     h = h - 0.1 * grad

        h = x_noisy.clone()
        for _ in range(steps):
            # Recalculate input with current estimate?
            # Or just use the fixed x_noisy input to predict "target"?
            # usage in TODO.md: h_pred = self.denoiser(x_input)
            # This implies x_input is fixed (the noisy image).
            # And 'h' is the latent code or the cleaned image being refined?

            # Let's interpret the TODO.md strictly:
            # h starts as x_noisy. We want to move h towards the "clean" manifold.
            # The network predicts a "chemically pure" version given the noisy input.

            # Since self.denoiser is stateless between calls unless we pass state,
            # and it returns a prediction.

            x_pred = self.denoiser(x_input)

            # Gradient descent on Energy = ||h - x_pred||^2
            # grad_h E = 2 * (h - x_pred)
            grad = 2 * (h - x_pred)
            h = h - 0.1 * grad

        return h

    def forward(self, x, t=None):
        """
        Forward pass for training/inference.

        If training (no t provided):
            Returns the 'denoised' prediction from the network directly.
            This is used by the trainer to compute loss against the clean image.

        Args:
            x: Noisy image batch (if t provided) or Input batch (if t embedded in x?)
               Actually, for training, x usually contains both if not handling t separately.
            t: Time steps (optional)
        """
        # If input x has extra channels, it might already include t embedding?
        # self.denoiser input channels = img_channels + 1

        if x.shape[1] == self.img_channels + 1:
            # Already has time embedding concatenated
            return self.denoiser(x)

        if t is None:
            # Assume t is zero or handle gracefully?
            # Or assume this is just the denoiser call?
            # For simplicity, if standard forward is called without t,
            # we might just return the denoiser output assuming x is formatted correctly
            # or fail if shape mismatch.
            return self.denoiser(x)

        # If t is provided, embed and concat
        batch_size, _, h, w = x.shape
        t_emb = t.view(batch_size, 1, 1, 1).expand(batch_size, 1, h, w)
        x_input = torch.cat([x, t_emb], dim=1)

        return self.denoiser(x_input)
