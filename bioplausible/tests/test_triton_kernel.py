
import unittest
import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
root_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_path))

from bioplausible.models.triton_kernel import TritonEqPropOps

class TestTritonKernel(unittest.TestCase):

    def setUp(self):
        self.batch_size = 32
        self.hidden_dim = 128
        self.alpha = 0.5

        # Use GPU if available for Triton
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.h = torch.randn(self.batch_size, self.hidden_dim, device=self.device)
        self.pre_act = torch.randn(self.batch_size, self.hidden_dim, device=self.device)
        self.bias = torch.randn(self.hidden_dim, device=self.device)

    def test_fallback_cpu(self):
        """Test the fallback logic on CPU (or GPU if Triton unavailable)."""
        # Force fallback by checking logic
        # If we are on CPU, TritonEqPropOps.is_available() is False

        if self.device.type == 'cpu':
            out = TritonEqPropOps.step(self.h, self.pre_act, self.alpha, self.bias)

            # Expected
            expected = (1 - self.alpha) * self.h + self.alpha * torch.tanh(self.pre_act + self.bias)

            self.assertTrue(torch.allclose(out, expected, atol=1e-5))
            print("CPU Fallback Verified")

    def test_triton_match(self):
        """Test that Triton kernel matches PyTorch implementation."""
        if not TritonEqPropOps.is_available():
            print("Skipping Triton test (Triton/GPU not available)")
            return

        # Run Triton
        out_triton = TritonEqPropOps.step(self.h, self.pre_act, self.alpha, self.bias)

        # Run PyTorch
        expected = (1 - self.alpha) * self.h + self.alpha * torch.tanh(self.pre_act + self.bias)

        # Compare
        self.assertTrue(torch.allclose(out_triton, expected, atol=1e-5))
        print("Triton Output Matches PyTorch")

    def test_no_bias(self):
        """Test without bias."""
        if not TritonEqPropOps.is_available():
            if self.device.type == 'cpu':
                 out = TritonEqPropOps.step(self.h, self.pre_act, self.alpha, None)
                 expected = (1 - self.alpha) * self.h + self.alpha * torch.tanh(self.pre_act)
                 self.assertTrue(torch.allclose(out, expected, atol=1e-5))
            return

        out_triton = TritonEqPropOps.step(self.h, self.pre_act, self.alpha, None)
        expected = (1 - self.alpha) * self.h + self.alpha * torch.tanh(self.pre_act)

        self.assertTrue(torch.allclose(out_triton, expected, atol=1e-5))

    def test_linear_fallback_cpu(self):
        """Test linear step fallback on CPU."""
        if self.device.type == "cpu":
            h_target = self.h + self.pre_act  # Just some target
            out = TritonEqPropOps.step_linear(self.h, h_target, self.alpha)
            expected = (1 - self.alpha) * self.h + self.alpha * h_target
            self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_linear_triton_match(self):
        """Test linear step Triton matches PyTorch."""
        if not TritonEqPropOps.is_available():
            return

        h_target = self.h + self.pre_act
        out_triton = TritonEqPropOps.step_linear(self.h, h_target, self.alpha)
        expected = (1 - self.alpha) * self.h + self.alpha * h_target
        self.assertTrue(torch.allclose(out_triton, expected, atol=1e-5))

    def test_cupy_integration(self):
        """Test CuPy integration with Triton."""
        try:
            import cupy as cp
        except ImportError:
            print("Skipping CuPy test (CuPy not installed)")
            return

        # This test requires GPU and Triton to be meaningful
        if not TritonEqPropOps.is_available():
            print("Skipping CuPy Triton test (Triton/GPU not available)")
            return

        # Create CuPy arrays
        h_cp = cp.random.randn(self.batch_size, self.hidden_dim, dtype=cp.float32)
        target_cp = cp.random.randn(self.batch_size, self.hidden_dim, dtype=cp.float32)

        # Run Triton CuPy kernel
        out_cp = TritonEqPropOps.step_linear_cupy(h_cp, target_cp, self.alpha)

        # Run NumPy/CPU baseline
        h_np = cp.asnumpy(h_cp)
        target_np = cp.asnumpy(target_cp)
        expected_np = (1 - self.alpha) * h_np + self.alpha * target_np

        # Verify
        self.assertTrue(np.allclose(cp.asnumpy(out_cp), expected_np, atol=1e-5))
        print("Triton CuPy Integration Verified")


if __name__ == "__main__":
    unittest.main()
