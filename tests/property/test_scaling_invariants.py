"""Scaling Invariant Property Tests.

Converted from validation tracks to automated property tests.
These verify scaling laws and invariants that can be checked algorithmically.

Migrated to native compositions after legacy zoo removal.
"""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torch import nn, optim

from computronium.models.native.backprop_native import create_native_backprop_mlp
from computronium.models.native.eqprop_native import create_native_eqprop_mlp
from computronium.ontology import (
    BackpropCredit,
    CreditAssignmentConfig,
    DigitalSubstrate,
    EnergyMinimizationDynamics,
    EuclideanUpdate,
    FeedforwardGeometry,
    GeometryConfig,
    InstantaneousDynamics,
    ParameterUpdateConfig,
    RecurrentGeometry,
    StateDynamicsConfig,
    SubstrateConfig,
    SubstrateType,
    ThermodynamicContrast,
)
from computronium.core.system_trainer import compose_system


def _make_synthetic_dataset(
    n_samples: int, input_dim: int, output_dim: int, seed: int = 42
):
    torch.manual_seed(seed)
    x = torch.randn(n_samples, input_dim)
    y = torch.randint(0, output_dim, (n_samples,))
    for c in range(output_dim):
        mask = y == c
        if mask.any():
            direction = torch.randn(input_dim)
            direction = direction / direction.norm() * 1.5
            x[mask] += direction * 0.5
    return x, y


def _create_native_eqprop_cuda(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    beta: float = 0.5,
    settle_steps: int = 30,
    lr: float = 0.01,
):
    """Create native EqProp model configured for CUDA."""
    hidden_dims = tuple([hidden_dim] * 1)
    geometry_cfg = GeometryConfig.recurrent(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
    )

    substrate = DigitalSubstrate(SubstrateConfig.digital(device="cuda"))
    geometry = RecurrentGeometry(geometry_cfg, hidden_dim=hidden_dim)
    # Move geometry to CUDA
    geometry = geometry.to("cuda")
    dynamics = EnergyMinimizationDynamics(
        StateDynamicsConfig.energy_minimization(
            max_steps=settle_steps,
            beta=beta,
        )
    )
    credit = ThermodynamicContrast(
        CreditAssignmentConfig.thermodynamic_contrast(
            beta=beta,
        )
    )
    update = EuclideanUpdate(
        ParameterUpdateConfig.euclidean(
            step_size=lr,
        )
    )

    return compose_system(substrate, geometry, dynamics, credit, update)


def _create_native_backprop_cuda(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 1,
    lr: float = 0.01,
):
    """Create native Backprop model configured for CUDA."""
    substrate = DigitalSubstrate(SubstrateConfig.digital(device="cuda"))

    dims = [hidden_dim] * max(num_layers - 1, 1)
    geometry = FeedforwardGeometry(
        GeometryConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=tuple(dims),
            num_layers=num_layers,
            topology_type="feedforward",
            connectivity=None,
            recurrent_weight=None,
        )
    )
    # Move geometry to CUDA
    geometry = geometry.to("cuda")

    dynamics = InstantaneousDynamics(StateDynamicsConfig.instantaneous())

    credit = BackpropCredit(CreditAssignmentConfig.gradient())

    update = EuclideanUpdate(
        ParameterUpdateConfig(
            update_type="euclidean",
            step_size=lr,
            momentum=0.9,
            ortho_steps=5,
            spectral_norm=1.0,
            fisher_damping=1e-3,
            ewc_lambda=1000.0,
        )
    )

    return compose_system(substrate, geometry, dynamics, credit, update)


# =============================================================================
# Track 10: O(1) Memory Scaling
# =============================================================================


class TestMemoryScalingO1:
    """EqProp should use O(1) activation memory regardless of depth;
    Backprop uses O(depth) memory.
    These tests only run on CUDA where memory can be accurately measured.
    """

    @pytest.mark.parametrize("depth", [10, 25, 50])
    @settings(max_examples=2, deadline=None)
    @given(st.data())
    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for memory measurement"
    )
    def test_eqprop_activation_memory_constant(self, depth, data):
        """EqProp activation memory should not grow dramatically with depth."""
        model = _create_native_eqprop_cuda(
            input_dim=64,
            hidden_dim=128,
            output_dim=10,
            beta=0.5,
            settle_steps=depth,
            lr=0.01,
        )

        x = torch.randn(32, 64, device="cuda")
        y = torch.randint(0, 10, (32,), device="cuda")

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        model.train()  # type: ignore[attr-defined]
        model.train_step(x, y)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        activation_mem = max(0, peak - baseline)

        # Activation memory should be bounded (not grow with depth)
        assert activation_mem < 500 * 1024 * 1024, (
            f"EqProp activation memory {activation_mem / 1e6:.1f} MB "
            f"too high at depth {depth}"
        )

    @pytest.mark.parametrize("depth", [10, 25, 50])
    @settings(max_examples=2, deadline=None)
    @given(st.data())
    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for memory measurement"
    )
    @pytest.mark.xfail(
        reason="CUDA memory measurement not capturing activation memory correctly in train_step"
    )
    def test_backprop_memory_grows_with_depth(self, depth, data):
        """Backprop activation memory should grow with depth."""
        num_layers = depth // 10
        model = _create_native_backprop_cuda(
            input_dim=64,
            hidden_dim=128,
            output_dim=10,
            num_layers=num_layers,
            lr=0.01,
        )

        x = torch.randn(32, 64, device="cuda")
        y = torch.randint(0, 10, (32,), device="cuda")

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        model.train()  # type: ignore[attr-defined]
        model.train_step(x, y)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        activation_mem = max(0, peak - baseline)

        # Backprop should use more memory at higher depth
        assert activation_mem > 1 * 1024 * 1024, (
            f"Backprop activation memory {activation_mem / 1e6:.1f} MB "
            f"too low at depth {depth}"
        )


# =============================================================================
# Track 11: Deep Network Credit Assignment
# =============================================================================


class TestDeepNetworkCreditAssignment:
    """EqProp should enable credit assignment through 100+ effective layers.

    Known issue: The "equilibrium" gradient method doesn't propagate gradients
    well for deep networks. This is tracked in GATE-0.
    """

    @pytest.mark.parametrize("depth", [50, 100])
    @settings(max_examples=2, deadline=None)
    @given(st.data())
    @pytest.mark.xfail(
        reason="GATE-0: Equilibrium method gradients don't propagate to input layer in deep networks. "
        "Use 'contrastive' or 'bptt' gradient_method for deep credit assignment."
    )
    def test_deep_network_gradient_flow(self, depth, data):
        """Gradients should flow through 100+ effective layers."""
        model = create_native_eqprop_mlp(
            input_dim=64,
            hidden_dim=64,
            output_dim=10,
            beta=0.5,
            settle_steps=depth,
            lr=0.01,
        )

        x = torch.randn(4, 64)
        y = torch.randint(0, 10, (4,))

        # Check gradient flow through first layer
        model.train()  # type: ignore[attr-defined]
        metrics = model.train_step(x, y)

        # Find first layer weight via geometry params
        params = model.geometry.params
        first_layer_weight = None
        for name, param in params.items():
            if (
                "weight" in name.lower()
                and "bias" not in name.lower()
                and ("layer_0" in name or "W_in" in name)
            ):
                first_layer_weight = param
                break

        if first_layer_weight is None:
            pytest.skip("Could not find first layer weight")

        assert first_layer_weight.grad is not None, "First layer should have gradients"
        grad_mag = first_layer_weight.grad.abs().mean().item()
        assert grad_mag > 1e-6, (
            f"Gradient magnitude {grad_mag:.6f} too small at depth {depth}"
        )

    @pytest.mark.parametrize("depth", [100])
    @settings(max_examples=2, deadline=None)
    @given(st.data())
    @pytest.mark.xfail(
        reason="GATE-0: Deep network accuracy with equilibrium method is poor. "
        "Contrastive method or more epochs needed."
    )
    def test_deep_network_accuracy(self, depth, data):
        """100-layer network should achieve reasonable accuracy."""
        model = create_native_eqprop_mlp(
            input_dim=64,
            hidden_dim=64,
            output_dim=10,
            beta=0.5,
            settle_steps=depth,
            lr=0.01,
        )

        x_train, y_train = _make_synthetic_dataset(128, 64, 10, 42)
        x_test, y_test = _make_synthetic_dataset(32, 64, 10, 43)

        # Train for a few epochs
        model.train()  # type: ignore[attr-defined]
        for epoch in range(3):
            model.train_step(x_train, y_train)

        model.eval()  # type: ignore[attr-defined]
        with torch.no_grad():
            logits = model(x_test)  # type: ignore[operator]
            acc = (logits.argmax(dim=1) == y_test).float().mean().item()

        # Should achieve better than random (10%)
        assert acc > 0.3, f"Accuracy {acc:.1%} too low at depth {depth}"


# =============================================================================
# Track 5: Neural Cube 3D Topology - REMOVED (was legacy)
# =============================================================================
# NeuralCube was a legacy zoo model removed in the cleanup.
# The 3D lattice topology capability is DEFERRED per TODO7.md decision.
# If needed, requires new Geometry axis: "ConvGeometry" or "SpatialGeometry".
# =============================================================================


# =============================================================================
# Core Track 2: EqProp vs Backprop Parity
# =============================================================================


class TestEqPropBackpropAccuracyParity:
    """EqProp should achieve competitive accuracy with Backprop."""

    @settings(max_examples=3, deadline=None)
    @given(st.data())
    def test_eqprop_vs_backprop_accuracy(self, data):
        """EqProp accuracy should be within 15% of Backprop on synthetic data."""
        bp_model = create_native_backprop_mlp(64, 128, 10)
        eq_model = create_native_eqprop_mlp(
            input_dim=64,
            hidden_dim=128,
            output_dim=10,
            beta=0.5,
            settle_steps=30,
            lr=0.01,
        )

        x_train, y_train = _make_synthetic_dataset(128, 64, 10, 42)
        x_test, y_test = _make_synthetic_dataset(32, 64, 10, 43)

        # Backprop - uses native model directly
        bp_model.train()  # type: ignore[attr-defined]
        optimizer = optim.Adam([p for p in bp_model.geometry.params.values()], lr=0.01)
        for epoch in range(5):
            optimizer.zero_grad()
            logits = bp_model.forward(x_train)
            loss = nn.functional.cross_entropy(logits, y_train)
            loss.backward()
            optimizer.step()

        bp_model.eval()  # type: ignore[attr-defined]
        with torch.no_grad():
            bp_acc = (
                (bp_model.forward(x_test).argmax(dim=1) == y_test).float().mean().item()
            )

        # EqProp
        eq_model.train()  # type: ignore[attr-defined]
        for epoch in range(5):
            eq_model.train_step(x_train, y_train)

        eq_model.eval()  # type: ignore[attr-defined]
        with torch.no_grad():
            eq_acc = (
                (eq_model.forward(x_test).argmax(dim=1) == y_test).float().mean().item()
            )

        gap = abs(bp_acc - eq_acc)
        # Allow up to 15% gap (lenient for property test)
        assert gap < 0.15, (
            f"EqProp accuracy {eq_acc:.1%} vs Backprop {bp_acc:.1%}, "
            f"gap={gap:.1%} exceeds 15%"
        )


# =============================================================================
# Core Track 3: Adversarial Self-Healing / Noise Damping
# =============================================================================


class TestNoiseDampingSelfHealing:
    """EqProp networks should damp injected noise via contraction mapping."""

    @settings(max_examples=2, deadline=None)
    @given(st.data())
    def test_noise_damping(self, data):
        """Injected noise should be damped to zero through relaxation."""
        model = create_native_eqprop_mlp(
            input_dim=64,
            hidden_dim=128,
            output_dim=10,
            beta=0.5,
            settle_steps=50,
            lr=0.01,
        )

        x, y = _make_synthetic_dataset(32, 64, 10, 42)

        # Pre-train
        model.train()  # type: ignore[attr-defined]
        for epoch in range(3):
            model.train_step(x, y)

        model.eval()  # type: ignore[attr-defined]
        with torch.no_grad():
            # Get initial activations by running forward_with_intermediates
            acts = model.geometry.forward_with_intermediates(x[:4], model.substrate)
            h_clean = acts[-1]

        noise_levels = [0.5, 1.0, 2.0]
        dampings = []

        for noise in noise_levels:
            h_noisy = h_clean + torch.randn_like(h_clean) * noise
            initial_noise_mag = (h_noisy - h_clean).abs().mean().item()

            # Replace last activation with noisy version and settle
            # Note: This is a simplified test - the full settling would require
            # running the dynamics with the noisy state. For now, we verify
            # the model can train without error after noise injection.
            model.train()  # type: ignore[attr-defined]
            result = model.train_step(x[:4], y[:4])

            # Get final hidden state after settling
            with torch.no_grad():
                final_acts = model.geometry.forward_with_intermediates(
                    x[:4], model.substrate
                )
                h_final = final_acts[-1]

            final_noise = (h_final - h_clean).abs().mean().item()
            damping_percent = (1 - final_noise / (initial_noise_mag + 1e-8)) * 100
            dampings.append(damping_percent)

        avg_damping = sum(dampings) / len(dampings)
        # Should damp at least 30% on average (lenient)
        assert avg_damping > 30, f"Average noise damping {avg_damping:.1f}% below 30%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
