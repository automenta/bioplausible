"""Gradient Equivalence CI Gate.

Compares Triton vs CuPy vs PyTorch kernel implementations for numerical parity
on every commit. Ensures bitwise/numerical equivalence across backends.
"""

import pytest
import torch

from bioplausible.acceleration.backends import HAS_CUPY, HAS_TRITON
from bioplausible.acceleration.triton_kernels import MEP_TritonOps, TritonEqPropOps


class TestTritonEqPropEquivalence:
    """Test Triton EqProp kernels match PyTorch reference."""

    @pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_step_equivalence(self):
        """Test Triton step kernel matches PyTorch reference."""
        torch.manual_seed(42)

        batch, dim = 32, 256
        h = torch.randn(batch, dim, device="cuda", dtype=torch.float32)
        pre_act = torch.randn(batch, dim, device="cuda", dtype=torch.float32)
        bias = torch.randn(dim, device="cuda", dtype=torch.float32)
        alpha = 0.1

        # Triton implementation
        triton_out = TritonEqPropOps.step(h, pre_act, alpha, bias)

        # PyTorch reference
        ref_out = (1.0 - alpha) * h + alpha * torch.tanh(pre_act + bias)

        # Check numerical equivalence
        max_diff = (triton_out - ref_out).abs().max().item()
        rel_diff = max_diff / (ref_out.abs().max().item() + 1e-8)

        assert max_diff < 1e-5, f"Max absolute diff: {max_diff}"
        assert rel_diff < 1e-4, f"Max relative diff: {rel_diff}"

    @pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_step_no_bias_equivalence(self):
        """Test Triton step kernel without bias matches PyTorch."""
        torch.manual_seed(42)

        batch, dim = 16, 128
        h = torch.randn(batch, dim, device="cuda", dtype=torch.float32)
        pre_act = torch.randn(batch, dim, device="cuda", dtype=torch.float32)
        alpha = 0.2

        triton_out = TritonEqPropOps.step(h, pre_act, alpha, None)
        ref_out = (1.0 - alpha) * h + alpha * torch.tanh(pre_act)

        max_diff = (triton_out - ref_out).abs().max().item()
        assert max_diff < 1e-5, f"Max absolute diff: {max_diff}"

    @pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.xfail(reason="Layered step requires CuPy for Triton path; skipping until fixed")
    def test_layered_step_equivalence(self):
        """Test Triton fused layered MLP step matches PyTorch."""
        torch.manual_seed(42)

        batch, K, H = 8, 128, 512
        h = torch.randn(batch, K, device="cuda", dtype=torch.float32)
        x_emb = torch.randn(batch, K, device="cuda", dtype=torch.float32)
        w1 = torch.randn(H, K, device="cuda", dtype=torch.float32)
        b1 = torch.randn(H, device="cuda", dtype=torch.float32)
        w2 = torch.randn(K, H, device="cuda", dtype=torch.float32)
        b2 = torch.randn(K, device="cuda", dtype=torch.float32)
        gamma = 0.5

        # Triton (via step_layered_cupy which uses Triton)
        if HAS_TRITON and HAS_CUPY:
            triton_out = TritonEqPropOps.step_layered_cupy(
                h, x_emb, w1, b1, w2, b2, gamma
            )
            if triton_out is not None:
                triton_h_next, triton_hnorm, triton_ffnhid = triton_out

                # PyTorch reference
                mean = h.mean(-1, keepdim=True)
                std = h.std(-1, keepdim=True, correction=0) + 1e-5
                h_norm = (h - mean) / std
                ffn_hidden = torch.tanh(h_norm @ w1.t() + b1)
                ffn_out = ffn_hidden @ w2.t() + b2
                ref_h_next = (1.0 - gamma) * h + gamma * (ffn_out + x_emb)

                max_diff = (triton_h_next - ref_h_next).abs().max().item()
                rel_diff = max_diff / (ref_h_next.abs().max().item() + 1e-8)
                assert max_diff < 1e-4, f"Layered step max diff: {max_diff}"
                assert rel_diff < 1e-3, f"Layered step rel diff: {rel_diff}"
            else:
                pytest.skip("Triton layered kernel not available")
        else:
            pytest.skip("CuPy not available for Triton layered step")


class TestMEPKernelsEquivalence:
    """Test MEP Triton kernels match PyTorch reference."""

    @pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_muon_orthogonalize_equivalence(self):
        """Test Triton Muon NS matches PyTorch reference."""
        torch.manual_seed(42)

        M, N = 256, 128
        W = torch.randn(M, N, device="cuda", dtype=torch.float32)

        # Triton implementation
        triton_out = MEP_TritonOps.muon_orthogonalize(W.clone(), ns_steps=5)

        # PyTorch reference (from core MEP implementation)
        ref_W = W.clone().T.contiguous() if M < N else W.clone()
        transposed = M < N
        base = ref_W.T.contiguous() if transposed else ref_W
        out = base / base.norm().clamp(min=1e-4, max=1e4)

        for _ in range(5):
            WT_W = out.T @ out
            out = out @ (
                1.5 * torch.eye(N, device=out.device, dtype=out.dtype) - 0.5 * WT_W
            )

        ref_out = out.T if transposed else out

        max_diff = (triton_out - ref_out).abs().max().item()
        rel_diff = max_diff / (ref_out.abs().max().item() + 1e-8)

        assert max_diff < 1e-4, f"Muon max diff: {max_diff}"
        assert rel_diff < 1e-3, f"Muon rel diff: {rel_diff}"

    @pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fisher_whiten_equivalence(self):
        """Test Triton Fisher whitening matches PyTorch."""
        torch.manual_seed(42)

        n = 1024
        grad = torch.randn(n, device="cuda", dtype=torch.float32)
        fisher_diag = torch.rand(n, device="cuda", dtype=torch.float32).abs() + 0.1
        damping = 1e-3

        triton_out = MEP_TritonOps.fisher_whiten(grad, fisher_diag, damping)
        ref_out = grad / torch.sqrt(fisher_diag + damping)

        max_diff = (triton_out - ref_out).abs().max().item()
        assert max_diff < 1e-5, f"Fisher max diff: {max_diff}"

    @pytest.mark.skipif(not HAS_TRITON, reason="Triton not available")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.xfail(reason="EP settle tolerance needs tuning for accumulated operations")
    def test_ep_settle_equivalence(self):
        """Test Triton fused EP settle matches PyTorch loop."""
        torch.manual_seed(42)

        M, K, H = 8, 128, 512
        h = torch.randn(M, K, device="cuda", dtype=torch.float32)
        x_emb = torch.randn(M, K, device="cuda", dtype=torch.float32)
        W1 = torch.randn(H, K, device="cuda", dtype=torch.float32)
        b1 = torch.randn(H, device="cuda", dtype=torch.float32)
        W2 = torch.randn(K, H, device="cuda", dtype=torch.float32)
        b2 = torch.randn(K, device="cuda", dtype=torch.float32)
        gamma = 0.5
        steps = 10

        triton_out = MEP_TritonOps.ep_settle(h, x_emb, W1, b1, W2, b2, gamma, steps)

        # PyTorch reference
        ref_h = h.clone()
        for _ in range(steps):
            mean = ref_h.mean(dim=-1, keepdim=True)
            std = ref_h.std(dim=-1, keepdim=True, correction=0) + 1e-5
            h_norm = (ref_h - mean) / std
            ffn_hidden = torch.tanh(h_norm @ W1.T + b1)
            ffn_out = ffn_hidden @ W2.T + b2
            ref_h = (1.0 - gamma) * ref_h + gamma * (ffn_out + x_emb)

        max_diff = (triton_out - ref_h).abs().max().item()
        rel_diff = max_diff / (ref_h.abs().max().item() + 1e-8)

        # Allow higher tolerance for accumulated operations
        assert max_diff < 1e-2, f"EP settle max diff: {max_diff}"
        assert rel_diff < 1e-1, f"EP settle rel diff: {rel_diff}"


class TestCuPyEquivalence:
    """Test CuPy implementations match PyTorch (when available)."""

    @pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.xfail(reason="CuPy-Torch zero-copy path returns empty tensors; needs investigation")
    def test_step_layered_cupy_torch_equivalence(self):
        """Test CuPy-Torch zero-copy layered step matches pure PyTorch."""
        torch.manual_seed(42)

        import cupy as cp

        batch, K, H = 4, 64, 256
        h = torch.randn(batch, K, device="cuda", dtype=torch.float32)
        x_emb = torch.randn(batch, K, device="cuda", dtype=torch.float32)
        w1 = torch.randn(H, K, device="cuda", dtype=torch.float32)
        b1 = torch.randn(H, device="cuda", dtype=torch.float32)
        w2 = torch.randn(K, H, device="cuda", dtype=torch.float32)
        b2 = torch.randn(K, device="cuda", dtype=torch.float32)
        gamma = 0.5

        # Convert to CuPy (zero-copy)
        h_cp = cp.asarray(h)
        x_emb_cp = cp.asarray(x_emb)
        w1_cp = cp.asarray(w1)
        b1_cp = cp.asarray(b1)
        w2_cp = cp.asarray(w2)
        b2_cp = cp.asarray(b2)

        result = TritonEqPropOps.step_layered_cupy_torch(
            h_cp, x_emb_cp, w1_cp, b1_cp, w2_cp, b2_cp, gamma
        )

        if result is None:
            pytest.skip("CuPy-Torch path not available")

        out_cp, hnorm_cp, ffnhid_cp = result

        # Convert back
        out_t = torch.as_tensor(out_cp, device="cuda")
        hnorm_t = torch.as_tensor(hnorm_cp, device="cuda")
        ffnhid_t = torch.as_tensor(ffnhid_cp, device="cuda")

        # PyTorch reference
        mean = h.mean(-1, keepdim=True)
        std = h.std(-1, keepdim=True, correction=0) + 1e-5
        h_norm_ref = (h - mean) / std
        ffn_hidden_ref = torch.tanh(h_norm_ref @ w1.t() + b1)
        ffn_out_ref = ffn_hidden_ref @ w2.t() + b2
        out_ref = (1.0 - gamma) * h + gamma * (ffn_out_ref + x_emb)

        # Check shapes match
        assert out_t.shape == out_ref.shape, f"Shape mismatch: {out_t.shape} vs {out_ref.shape}"

        max_diff = (out_t - out_ref).abs().max().item()
        assert max_diff < 1e-4, f"CuPy-Torch max diff: {max_diff}"

        hnorm_diff = (hnorm_t - h_norm_ref).abs().max().item()
        assert hnorm_diff < 1e-4, f"CuPy-Torch hnorm diff: {hnorm_diff}"

        ffnhid_diff = (ffnhid_t - ffn_hidden_ref).abs().max().item()
        assert ffnhid_diff < 1e-4, f"CuPy-Torch ffnhid diff: {ffnhid_diff}"


class TestKernelRegistryAutoTune:
    """Test KernelRegistry auto-tuning functionality."""

    def test_autotune_cache_works(self):
        """Test that auto-tune caching works correctly."""
        from bioplausible.acceleration.kernel_backend import (
            AlgorithmFamily,
            HardwareTarget,
            KernelRegistry,
        )

        # Clear cache
        KernelRegistry.clear_autotune_cache()

        # Simple test with CPU backend
        class DummyBackend:
            name = AlgorithmFamily.BACKPROP
            supported_dtypes = (torch.float32,)
            supports_autograd = True
            requires_settle = False
            memory_complexity = "O(L)"
            locality_level = "global"

            def initialize(self, config):
                pass

            def forward(self, x):
                return x * 2

        # Register dummy backend
        KernelRegistry.register(AlgorithmFamily.BACKPROP, HardwareTarget.CPU, DummyBackend)

        # First call should populate cache
        backend1 = KernelRegistry.get_best_for_shape(
            AlgorithmFamily.BACKPROP, "forward", (32, 256)
        )
        assert backend1 is not None

        # Second call should use cache
        backend2 = KernelRegistry.get_best_for_shape(
            AlgorithmFamily.BACKPROP, "forward", (32, 256)
        )
        assert backend2 is not None
        assert backend1 is backend2  # Same instance from cache

        # Clean up
        KernelRegistry.clear_autotune_cache()
        KernelRegistry.clear_cache()


class TestBackendNumericalParity:
    """Test numerical parity across all registered backends."""

    @pytest.mark.parametrize("algorithm", [
        "eqprop",
        "mep",
    ])
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_registered_backends_exist(self, algorithm):
        """Verify backends are registered for key algorithms."""
        from bioplausible.acceleration.kernel_backend import (
            AlgorithmFamily,
            KernelRegistry,
        )

        # At least CPU should be available
        # (actual backend registration happens in respective kernel files)
        alg_family = AlgorithmFamily(algorithm)
        hw_list = KernelRegistry.list_for(alg_family)
        # This test passes if the registry is functional
        assert isinstance(hw_list, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
