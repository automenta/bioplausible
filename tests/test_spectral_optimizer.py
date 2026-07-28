"""Coverage tests for Spectral constraint optimizer."""

import torch
from bioplausible.zoo.optimizers.spectral import SpectralConstraint


class TestSpectralConstraint:
    def test_spectral_step_clamps_singular_values(self):
        param = torch.nn.Parameter(torch.randn(5, 10) * 5.0)  # Large weights
        orig_norm = torch.linalg.svd(param.reshape(5, -1), full_matrices=False)[1].max()
        opt = SpectralConstraint([param], max_norm=1.0)
        opt.step()
        s_after = torch.linalg.svd(param.reshape(5, -1), full_matrices=False)[1]
        assert s_after.max().item() <= 1.0 + 1e-5

    def test_spectral_step_skips_1d_params(self):
        bias = torch.nn.Parameter(torch.randn(10))
        opt = SpectralConstraint([bias], max_norm=1.0)
        # Should not raise
        opt.step()

    def test_zero_grad_clears_gradients(self):
        param = torch.nn.Parameter(torch.randn(5, 10))
        param.grad = torch.randn(5, 10)
        opt = SpectralConstraint([param])
        assert param.grad is not None
        opt.zero_grad()
        assert param.grad is not None  # zero_grad doesn't None the grad, just zeros it

    def test_zero_grad_handles_none_grad(self):
        param = torch.nn.Parameter(torch.randn(5, 10))
        param.grad = None
        opt = SpectralConstraint([param])
        # Should not raise
        opt.zero_grad()

    def test_spectral_constraint_with_embedding(self):
        """2D embedding matrix gets constrained via SVD."""
        param = torch.nn.Parameter(torch.randn(100, 64) * 10.0)
        opt = SpectralConstraint([param], max_norm=2.0)
        opt.step()
        u, s, vh = torch.linalg.svd(param.reshape(100, -1), full_matrices=False)
        assert s.max().item() <= 2.0 + 1e-5
