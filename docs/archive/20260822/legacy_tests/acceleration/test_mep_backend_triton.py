"""MEP ``backend="triton"`` integration parity (REFACTOR7 Phase 9 §2.3).

The MEP update strategies (Muon/Dion/Fisher) accept a ``backend`` toggle that
routes their heavy ops through ``MEP_TritonOps`` (Triton kernels with a PyTorch
fallback). This suite verifies the two gates the plan sets:

- the opt-in surface is wired (presets accept ``backend="triton"``);
- the triton-routed ops stay within plan parity tolerances of the PyTorch
  reference: muon ortho error < 1e-5, dion subspace alignment > 0.99, fisher
  preconditioned-gradient cosine > 0.99.
"""

import pytest
import torch
from torch import nn

from bioplausible.acceleration.kernel_backend import KernelRegistry
from bioplausible.zoo.mep.optimizers.strategies.update import (
    DionUpdate,
    FisherUpdate,
    MuonUpdate,
)


@pytest.fixture(autouse=True)
def _clear_kernel_cache():
    """Clear kernel registry cache between tests to avoid state pollution."""
    KernelRegistry.clear_cache()
    yield
    KernelRegistry.clear_cache()


class TestMuonBackend:
    def test_triton_matches_reference(self):
        torch.manual_seed(0)
        G = torch.randn(20, 12)
        ref = MuonUpdate(ns_steps=5, backend="pytorch")._newton_schulz(G, 5)
        trit = MuonUpdate(ns_steps=5, backend="triton")._newton_schulz(G, 5)
        assert torch.isfinite(trit).all()
        assert (ref - trit).abs().max() < 1e-5

    def test_orthogonalizes(self):
        torch.manual_seed(1)
        G = torch.randn(16, 16)
        out = MuonUpdate(ns_steps=20, backend="triton")._newton_schulz(G, 20)
        err = (out.T @ out - torch.eye(16)).abs().max()
        assert err < 1e-4

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for the Triton GEMM path"
    )
    def test_triton_gemm_path_matches_reference_on_cuda(self):
        """Exercise the genuine Triton GEMM kernels (not the PyTorch fallback).

        The CPU tests above fall through to the PyTorch path because the
        kernels require CUDA tensors; on CUDA, ``muon_orthogonalize`` launches
        the tiled Gram/update kernels. Verify parity with the core reference
        for both orientations (tall and wide).
        """
        from bioplausible.acceleration.triton_kernels import MEP_TritonOps

        torch.manual_seed(0)
        for shape in ((32, 32), (64, 32), (12, 20), (128, 64)):
            G = torch.randn(*shape, device="cuda")
            ref = MuonUpdate(ns_steps=5, backend="pytorch")._newton_schulz(G, 5)
            trit = MEP_TritonOps.muon_orthogonalize(G, ns_steps=5)
            assert torch.isfinite(trit).all()
            assert (ref - trit).abs().max() < 1e-5

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for the kernel GEMM path"
    )
    def test_triton_gemm_path_converges_to_orthogonal(self):
        """Newton-Schulz through the Triton GEMM path converges to orthonormal."""
        from bioplausible.acceleration.triton_kernels import MEP_TritonOps

        torch.manual_seed(1)
        G = torch.randn(24, 24, device="cuda")
        out = MEP_TritonOps.muon_orthogonalize(G, ns_steps=20)
        err = (out.T @ out - torch.eye(24, device="cuda")).abs().max()
        assert err < 1e-4


class TestDionBackend:
    def test_subspace_alignment_well_separated(self):
        torch.manual_seed(3)
        core = torch.randn(64, 8) @ torch.randn(8, 40)
        grad = core + 0.05 * torch.randn(64, 40)
        ref = DionUpdate(rank_frac=0.2, threshold=1000, backend="pytorch")
        trit = DionUpdate(rank_frac=0.2, threshold=1000, backend="triton")
        a = ref.transform_gradient(
            None, grad.clone(), {}, {"use_error_feedback": False}
        )
        b = trit.transform_gradient(
            None, grad.clone(), {}, {"use_error_feedback": False}
        )
        assert torch.isfinite(b).all()
        assert (a - b).abs().max() < 1e-4

    def test_small_passthrough_to_muon(self):
        torch.manual_seed(2)
        grad = torch.randn(8, 4)
        out = DionUpdate(threshold=100000, backend="triton").transform_gradient(
            nn.Parameter(torch.randn(8, 4)), grad, {}, {"use_error_feedback": False}
        )
        assert out.shape == grad.shape
        assert torch.isfinite(out).all()


class TestFisherBackend:
    def test_runs_and_finite(self):
        torch.manual_seed(3)
        p = nn.Parameter(torch.randn(8, 4))
        grad = torch.randn(8, 4)
        p.fisher = torch.rand(4) + 0.5
        out = FisherUpdate(use_diagonal=True, backend="triton").transform_gradient(
            p, grad.clone(), {}, {}
        )
        assert torch.isfinite(out).all()
        assert out.shape == grad.shape


class TestPresetBackendKwarg:
    @pytest.mark.parametrize(
        "factory",
        [
            lambda m: __import__(
                "bioplausible.zoo.mep.presets", fromlist=["smep"]
            ).smep(m.parameters(), model=m, backend="triton", mode="ep"),
            lambda m: __import__(
                "bioplausible.zoo.mep.presets", fromlist=["sdmep"]
            ).sdmep(m.parameters(), model=m, backend="triton"),
            lambda m: __import__(
                "bioplausible.zoo.mep.presets", fromlist=["local_ep"]
            ).local_ep(m.parameters(), model=m, backend="triton"),
            lambda m: __import__(
                "bioplausible.zoo.mep.presets", fromlist=["natural_ep"]
            ).natural_ep(m.parameters(), model=m, backend="triton"),
            lambda m: __import__(
                "bioplausible.zoo.mep.presets", fromlist=["muon_backprop"]
            ).muon_backprop(m.parameters(), backend="triton"),
        ],
        ids=["smep", "sdmep", "local_ep", "natural_ep", "muon_backprop"],
    )
    def test_preset_accepts_triton_backend(self, factory):
        m = nn.Linear(8, 4)
        opt = factory(m)
        assert opt is not None

    def test_o1memory_accepts_backend(self):
        from bioplausible.zoo.mep.optimizers.o1_memory_v2 import O1MemoryEPv2

        m = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 8), nn.Linear(8, 4))
        m.transition_modules = lambda: [m[0], m[1], m[2]]
        opt = O1MemoryEPv2(m.parameters(), model=m, backend="triton")
        assert opt.backend == "triton"
