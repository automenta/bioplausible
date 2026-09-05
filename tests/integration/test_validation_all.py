import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F  # ruff: ignore[lowercase-imported-as-non-lowercase]
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from computronium.models.native.backprop_native import create_native_backprop_mlp
from computronium.models.native.eqprop_native import create_native_eqprop_mlp
from computronium.models.native.fa_native import create_native_fa_mlp
from computronium.models.native.pepita_native import create_native_pepita_mlp
from computronium.models.native.tile_native import (
    create_native_tile_ep,
    create_native_tile_fa,
    create_native_tile_hebbian,
    create_native_tile_tp,
)

# Add parent to path for in-package testing
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))


class TestValidationAll:
    """
    Generalized validation suite for all models.
    Verifies that minimal instances of every model can actually learn a simple task.

    Migrated to native compositions after legacy zoo removal.
    """

    def setup_method(self):
        # Seed torch at setup so the per-test random tensors are independent
        # of upstream test RNG usage (which can flip pass/fail rates of the
        # inherently-flaky `f_loss < i_loss` checks in this suite).
        torch.manual_seed(42)
        self.device = "cpu"
        self.epochs = 5  # Minimal epochs — enough to verify f_loss < i_loss
        self.batch_size = 4
        self.sample_size = 20
        self.input_dim = 10
        self.output_dim = 2
        self.vocab_size = 10
        self.seq_len = 5

        # 1. Standard MLP Data
        self.x = torch.randn(self.sample_size, self.input_dim).to(self.device)
        self.y = torch.randint(0, self.output_dim, (self.sample_size,)).to(self.device)
        self.loader = DataLoader(
            TensorDataset(self.x, self.y), batch_size=self.batch_size, shuffle=True
        )

        # 2. Convolutional Data (CIFAR-like shape: B, 3, 16, 16)
        # Note: Native models don't yet support ConvGeometry - DEFERRED per TODO7.md
        # ConvEqProp, ModernConvEqProp, SimpleConvEqProp tests skipped
        self.x_conv = torch.randn(self.sample_size, 3, 16, 16).to(self.device)
        self.y_conv = torch.randint(0, 10, (self.sample_size,)).to(
            self.device
        )  # Models default to 10 classes

    def _train_and_assert_learns(
        self, model: nn.Module, x: torch.Tensor, y: torch.Tensor, name: str
    ) -> None:
        """Train for a few epochs and assert loss decreases."""
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        losses = []
        for epoch in range(self.epochs):
            opt.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0], (
            f"{name}: loss did not decrease ({losses[0]:.4f} -> {losses[-1]:.4f})"
        )

    def _train_system_and_assert_learns(
        self, system, x: torch.Tensor, y: torch.Tensor, name: str
    ) -> None:
        """Train a native System for a few epochs and assert loss decreases."""
        system.train()  # type: ignore[attr-defined]
        losses = []
        for epoch in range(self.epochs):
            metrics = system.train_step(x, y)
            losses.append(metrics.get("loss", 0.0))
        assert losses[-1] < losses[0], (
            f"{name}: loss did not decrease ({losses[0]:.4f} -> {losses[-1]:.4f})"
        )

    # --- Native MLP Models ---

    def test_native_backprop_mlp(self):
        """Native Backprop MLP learns."""
        model = create_native_backprop_mlp(
            self.input_dim, 16, self.output_dim, num_layers=2, lr=1e-3
        )
        self._train_system_and_assert_learns(
            model, self.x, self.y, "native_backprop_mlp"
        )

    def test_native_eqprop_mlp(self):
        """Native EqProp MLP learns."""
        model = create_native_eqprop_mlp(
            input_dim=self.input_dim,
            hidden_dim=16,
            output_dim=self.output_dim,
            num_layers=2,
            beta=0.5,
            settle_steps=10,
            lr=1e-3,
        )
        self._train_system_and_assert_learns(model, self.x, self.y, "native_eqprop_mlp")

    def test_native_fa_mlp(self):
        """Native FA MLP learns - un-xfailed by the R3.2 repair: FA routes
        the autograd top error through fixed feedback matrices, so
        InstantaneousDynamics pairs are live (previously Δθ was exactly 0)."""
        model = create_native_fa_mlp(
            self.input_dim, 16, self.output_dim, num_layers=2, lr=1e-3
        )
        self._train_system_and_assert_learns(model, self.x, self.y, "native_fa_mlp")

    def test_native_pepita_mlp(self):
        """Native PEPITA MLP learns (realized local_objective="pepita")."""
        model = create_native_pepita_mlp(
            self.input_dim, 16, self.output_dim, num_layers=2, lr=1e-3
        )
        self._train_system_and_assert_learns(model, self.x, self.y, "native_pepita_mlp")

    # --- Native Tile Models ---

    def test_native_tile_ep(self):
        """Native Tile EP learns."""
        model = create_native_tile_ep(
            self.input_dim,
            16,
            self.output_dim,
            num_layers=2,
            neurons_per_tile=8,
            tiles_per_layer=2,
            lr=1e-3,
            beta=0.1,
        )
        self._train_system_and_assert_learns(model, self.x, self.y, "native_tile_ep")

    def test_native_tile_fa(self):
        """Native Tile FA learns."""
        model = create_native_tile_fa(
            self.input_dim,
            16,
            self.output_dim,
            num_layers=2,
            neurons_per_tile=8,
            tiles_per_layer=2,
            lr=1e-3,
        )
        self._train_system_and_assert_learns(model, self.x, self.y, "native_tile_fa")

    def test_native_tile_tp(self):
        """Native Tile TP learns."""
        model = create_native_tile_tp(
            self.input_dim,
            16,
            self.output_dim,
            num_layers=2,
            neurons_per_tile=8,
            tiles_per_layer=2,
            lr=1e-3,
            beta=0.1,
        )
        self._train_system_and_assert_learns(model, self.x, self.y, "native_tile_tp")

    def test_native_tile_hebbian(self):
        """Native Tile Hebbian learns."""
        model = create_native_tile_hebbian(
            self.input_dim,
            16,
            self.output_dim,
            num_layers=2,
            neurons_per_tile=8,
            tiles_per_layer=2,
            lr=1e-3,
        )
        self._train_system_and_assert_learns(
            model, self.x, self.y, "native_tile_hebbian"
        )

    # --- Skipped: Conv/Graph/Attention Models ---
    # These require ConvGeometry, GraphGeometry, AttentionGeometry which are DEFERRED per TODO7.md

    @pytest.mark.skip(reason="ConvGeometry not implemented - DEFERRED per TODO7.md")
    def test_conv_eqprop(self):
        pass

    @pytest.mark.skip(reason="ConvGeometry not implemented - DEFERRED per TODO7.md")
    def test_modern_conv_eqprop(self):
        pass

    @pytest.mark.skip(reason="ConvGeometry not implemented - DEFERRED per TODO7.md")
    def test_simple_conv_eqprop(self):
        pass

    @pytest.mark.skip(
        reason="AttentionGeometry not implemented - DEFERRED per TODO7.md"
    )
    def test_transformer_eqprop(self):
        pass

    @pytest.mark.skip(reason="GraphGeometry not implemented - DEFERRED per TODO7.md")
    def test_full_eqprop_lm(self):
        pass

    @pytest.mark.skip(reason="GraphGeometry not implemented - DEFERRED per TODO7.md")
    def test_recurrent_eqprop_lm(self):
        pass

    @pytest.mark.skip(
        reason="Homeostatic credit not implemented - DEFERRED per TODO7.md"
    )
    def test_homeostatic_eqprop(self):
        pass


if __name__ == "__main__":
    unittest.main()  # ruff: ignore[undefined-name]
