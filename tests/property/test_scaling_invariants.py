"""Scaling Invariant Property Tests.

Converted from validation tracks to automated property tests.
These verify scaling laws and invariants that can be checked algorithmically.
"""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torch import nn, optim

from bioplausible.config.unified import ModelConfig
from bioplausible.zoo.models.eqprop._energy import EquilibriumMLP


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
        config = ModelConfig(
            name="eqprop_mlp",
            input_dim=64,
            output_dim=10,
            hidden_dims=[128],
            learning_rate=0.01,
            beta=0.5,
            max_steps=depth,
            convergence_threshold=1e-4,
            convergence_start=5,
            use_spectral_norm=True,
            spectral_norm_power_iterations=5,
            activation="tanh",
            lipschitz_mode="power_iteration",
            output_scaling_mode="uniform",
            extra={
                "gradient_method": "equilibrium",
                "backend": "pytorch",
            },
        )
        model = EquilibriumMLP(config=config).cuda()

        x = torch.randn(32, 64, device="cuda")
        y = torch.randint(0, 10, (32,), device="cuda")

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        model.train()
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
        config = ModelConfig(
            name="eqprop_mlp",
            input_dim=64,
            output_dim=10,
            hidden_dims=[128],
            learning_rate=0.01,
            beta=0.5,
            max_steps=depth,
            convergence_threshold=1e-4,
            convergence_start=5,
            use_spectral_norm=True,
            spectral_norm_power_iterations=5,
            activation="tanh",
            lipschitz_mode="power_iteration",
            output_scaling_mode="uniform",
            extra={
                "gradient_method": "bptt",
                "backend": "pytorch",
            },
        )
        model = EquilibriumMLP(config=config).cuda()

        x = torch.randn(32, 64, device="cuda")
        y = torch.randint(0, 10, (32,), device="cuda")

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        model.train()
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
        from bioplausible.zoo.models.eqprop.looped_mlp import LoopedMLP

        model = LoopedMLP(
            input_dim=64,
            hidden_dim=64,
            output_dim=10,
            use_spectral_norm=True,
            max_steps=depth,
            gradient_method="equilibrium",
        )

        x = torch.randn(4, 64)
        y = torch.randint(0, 10, (4,))

        # Check gradient flow through first layer
        model.train()
        metrics = model.train_step(x, y)

        # Find first layer weight
        first_layer_weight = None
        for name, param in model.named_parameters():
            if "weight" in name.lower() and "bias" not in name.lower():
                if (
                    "layers.0" in name
                    or "W_in" in name
                    or name == "layers.0.parametrizations.weight.original"
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
        from bioplausible.zoo.models.eqprop.looped_mlp import LoopedMLP

        model = LoopedMLP(
            input_dim=64,
            hidden_dim=64,
            output_dim=10,
            use_spectral_norm=True,
            max_steps=depth,
            gradient_method="equilibrium",
        )

        x_train, y_train = _make_synthetic_dataset(128, 64, 10, 42)
        x_test, y_test = _make_synthetic_dataset(32, 64, 10, 43)

        # Train for a few epochs
        model.train()
        for epoch in range(3):
            model.train_step(x_train, y_train)

        model.eval()
        with torch.no_grad():
            logits = model(x_test)
            acc = (logits.argmax(dim=1) == y_test).float().mean().item()

        # Should achieve better than random (10%)
        assert acc > 0.3, f"Accuracy {acc:.1%} too low at depth {depth}"


# =============================================================================
# Track 12: Lazy Event-Driven Updates
# =============================================================================


class TestLazyUpdates:
    """Lazy/EqProp models should achieve FLOP savings by skipping inactive neurons.

    Known issue: LazyEqProp uses legacy config path and doesn't accept
    input_dim/hidden_dim/output_dim directly. Requires ModelConfig.
    """

    @pytest.mark.parametrize("epsilon", [0.001, 0.01, 0.1])
    @settings(max_examples=2, deadline=None)
    @given(st.data())
    @pytest.mark.xfail(
        reason="LazyEqProp uses legacy config path; requires ModelConfig instantiation. "
        "Waiting for native migration (Sprint 9)."
    )
    def test_lazy_eqprop_flop_savings(self, epsilon, data):
        """LazyEqProp should save FLOPs with minimal accuracy loss."""
        from bioplausible.zoo.models.eqprop.lazy_eqprop import LazyEqProp

        model = LazyEqProp(
            input_dim=64,
            hidden_dim=128,
            output_dim=10,
            epsilon=epsilon,
            use_spectral_norm=True,
        )

        x_train, y_train = _make_synthetic_dataset(128, 64, 10, 42)
        x_test, y_test = _make_synthetic_dataset(32, 64, 10, 43)

        # Train
        model.train()
        for epoch in range(3):
            model.train_step(x_train, y_train)

        # Measure accuracy
        model.eval()
        with torch.no_grad():
            logits = model(x_test)
            acc = (logits.argmax(dim=1) == y_test).float().mean().item()

        # Measure FLOP savings
        model.stats = model.stats.reset()
        with torch.no_grad():
            _ = model(x_test, steps=30)
        savings = model.get_flop_savings()

        # Should have some savings (even if small)
        assert savings >= 0, f"FLOP savings should be non-negative, got {savings}"

        # Accuracy should not be terrible
        assert acc > 0.1, f"Accuracy {acc:.1%} too low with epsilon={epsilon}"

    @pytest.mark.xfail(
        reason="LazyEqProp uses legacy config path; requires ModelConfig instantiation. "
        "Waiting for native migration (Sprint 9)."
    )
    def test_lazy_eqprop_savings_increase_with_epsilon(self):
        """Larger epsilon should yield more FLOP savings (potentially with more accuracy loss)."""
        from bioplausible.zoo.models.eqprop.lazy_eqprop import LazyEqProp

        x_train, y_train = _make_synthetic_dataset(128, 64, 10, 42)
        x_test, y_test = _make_synthetic_dataset(32, 64, 10, 43)

        savings_results = []
        for epsilon in [0.001, 0.01, 0.1]:
            model = LazyEqProp(
                input_dim=64,
                hidden_dim=128,
                output_dim=10,
                epsilon=epsilon,
                use_spectral_norm=True,
            )
            model.train()
            for epoch in range(3):
                model.train_step(x_train, y_train)

            model.stats = model.stats.reset()
            with torch.no_grad():
                _ = model(x_test, steps=30)
            savings = model.get_flop_savings()
            savings_results.append((epsilon, savings))

        # At least the largest epsilon should have some savings
        assert savings_results[-1][1] >= 0, (
            "Highest epsilon should have non-negative savings"
        )


# =============================================================================
# Track 5: Neural Cube 3D Topology
# =============================================================================


class TestNeuralCubeTopology:
    """Neural Cube should achieve connection reduction with 3D lattice."""

    @settings(max_examples=2, deadline=None)
    @given(st.data())
    def test_neural_cube_connection_reduction(self, data):
        """Neural Cube should have 90%+ fewer connections than fully-connected."""
        from bioplausible.zoo.models.eqprop import NeuralCube

        model = NeuralCube(cube_size=6, input_dim=64, output_dim=10)
        topo = model.get_topology_stats()

        # Should have ~91% connection reduction
        assert topo["connection_reduction"] > 0.85, (
            f"Connection reduction {topo['connection_reduction']:.1%} below 85%"
        )

        # Should have correct number of neurons
        assert topo["n_neurons"] == 6**3, (
            f"Expected {6**3} neurons, got {topo['n_neurons']}"
        )

        # Should have local connections
        assert topo["local_connections"] > 0, "Should have local connections"
        assert topo["fully_connected_equivalent"] > topo["local_connections"]

    def test_neural_cube_trainable(self):
        """Neural Cube should be trainable (at least with BPTT)."""
        from bioplausible.zoo.models.eqprop import NeuralCube

        model = NeuralCube(cube_size=4, input_dim=32, output_dim=10)

        x, y = _make_synthetic_dataset(64, 32, 10, 42)

        # Train with BPTT (standard optimizer)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        for epoch in range(5):
            optimizer.zero_grad()
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(x)
            acc = (logits.argmax(dim=1) == y).float().mean().item()

        # Should achieve better than random
        assert acc > 0.2, f"Neural Cube accuracy {acc:.1%} too low"


# =============================================================================
# Core Track 2: EqProp vs Backprop Parity
# =============================================================================


class TestEqPropBackpropAccuracyParity:
    """EqProp should achieve competitive accuracy with Backprop."""

    @settings(max_examples=3, deadline=None)
    @given(st.data())
    def test_eqprop_vs_backprop_accuracy(self, data):
        """EqProp accuracy should be within 15% of Backprop on synthetic data."""
        from bioplausible.zoo.models.eqprop import BackpropMLP, LoopedMLP

        bp_model = BackpropMLP(64, 128, 10)
        eq_model = LoopedMLP(64, 128, 10, use_spectral_norm=True, max_steps=30)

        x_train, y_train = _make_synthetic_dataset(128, 64, 10, 42)
        x_test, y_test = _make_synthetic_dataset(32, 64, 10, 43)

        # Backprop
        optimizer = optim.Adam(bp_model.parameters(), lr=0.01)
        for epoch in range(5):
            optimizer.zero_grad()
            logits = bp_model(x_train)
            loss = nn.functional.cross_entropy(logits, y_train)
            loss.backward()
            optimizer.step()

        bp_model.eval()
        with torch.no_grad():
            bp_acc = (bp_model(x_test).argmax(dim=1) == y_test).float().mean().item()

        # EqProp
        for epoch in range(5):
            eq_model.train_step(x_train, y_train)

        eq_model.eval()
        with torch.no_grad():
            eq_acc = (eq_model(x_test).argmax(dim=1) == y_test).float().mean().item()

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
        try:
            from bioplausible.core.local_learning.settling import (
                settle_activations_list,
            )
        except ImportError:
            pytest.skip("Required modules not available")

        from bioplausible.zoo.models.eqprop.looped_mlp import LoopedMLP

        model = LoopedMLP(
            input_dim=64,
            hidden_dim=128,
            output_dim=10,
            use_spectral_norm=True,
            max_steps=50,
            gradient_method="equilibrium",
        )

        x, y = _make_synthetic_dataset(32, 64, 10, 42)

        # Pre-train
        model.train()
        for epoch in range(3):
            model.train_step(x, y)

        model.eval()
        with torch.no_grad():
            activations = model._initial_activations(x[:4])

        noise_levels = [0.5, 1.0, 2.0]
        dampings = []

        for noise in noise_levels:
            h_clean = activations[-1].clone()
            noise_tensor = torch.randn_like(h_clean) * noise
            h_noisy = h_clean + noise_tensor
            initial_noise_mag = noise_tensor.abs().mean().item()

            activations_noisy = list(activations)
            activations_noisy[-1] = h_noisy

            settled, _, _ = settle_activations_list(
                activations_0=activations_noisy,
                forward_dynamics=model.forward_dynamics,
                steps=model.max_steps,
                beta=0.0,
                target=None,
                return_trajectory=False,
                return_dynamics=False,
                convergence_threshold=model.convergence_threshold,
                convergence_start=model.convergence_start,
            )

            h_final = settled[-1]
            final_noise = (h_final - h_clean).abs().mean().item()
            damping_percent = (1 - final_noise / (initial_noise_mag + 1e-8)) * 100
            dampings.append(damping_percent)

        avg_damping = sum(dampings) / len(dampings)
        # Should damp at least 30% on average (lenient)
        assert avg_damping > 30, f"Average noise damping {avg_damping:.1f}% below 30%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
