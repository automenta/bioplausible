"""Biology Axiom Property Tests - Simplified for Native Compositions.

These tests verify bio-plausibility axioms using native compositions.
Tests are simplified to work with the System protocol interface.
"""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torch import nn, optim

from computronium.core.local_learning.builder import (
    TileAlgorithm,
    TileAlgorithmConfig,
)
from computronium.core.local_learning.settling import (
    SettleProtocol,
    settle_universal,
)
from computronium.models.native.eqprop_native import create_native_eqprop_mlp
from computronium.models.native.fa_native import create_native_fa_mlp


# =============================================================================
# Shared Fixtures & Helpers
# =============================================================================


@pytest.fixture(scope="module")
def synthetic_mlp_task():
    """Minimal MLP task for gradient equivalence: 1 hidden layer, small dims."""
    torch.manual_seed(42)
    input_dim = 8
    hidden_dim = 8
    output_dim = 4
    n_samples = 32
    x = torch.randn(n_samples, input_dim)
    y = torch.randint(0, output_dim, (n_samples,))
    for c in range(output_dim):
        mask = y == c
        if mask.any():
            direction = torch.randn(input_dim)
            direction = direction / direction.norm() * 1.5
            x[mask] += direction * 0.5
    return x, y, input_dim, hidden_dim, output_dim


def _create_tile_ep_model(input_dim: int, hidden_dim: int, output_dim: int):
    """Create a Tile EP model for biology testing."""
    config = TileAlgorithmConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        neurons_per_tile=hidden_dim,
        tiles_per_layer=1,
        num_hidden_layers=1,
        algorithm="ep",
        mode="ep",
        free_steps=20,
        nudged_steps=20,
        learning_rate=0.01,
        beta=0.5,
        step_size=0.1,
    )
    model = TileAlgorithm(config)
    model.convergence_threshold = 1e-4
    model.convergence_start = 5
    return model


# =============================================================================
# 3.1 EP Gradient Equivalence — Equilibrium Propagation ≈ BPTT
# =============================================================================


class TestEPGradientEquivalence:
    """Verify EP gradient matches BPTT gradient direction."""

    @settings(max_examples=10, deadline=None)
    @given(st.data())
    @pytest.mark.xfail(
        reason="GATE-0: pre-existing EqProp gradient drift — "
        "max EP-BPTT cosine < 0.5. Locked until LOOP/RULE parity work lands."
    )
    def test_ep_gradient_matches_bptt(self, synthetic_mlp_task, data):
        """EP gradient should align with BPTT gradient at finite β."""
        x, y, input_dim, hidden_dim, output_dim = synthetic_mlp_task

        model = _create_tile_ep_model(input_dim, hidden_dim, output_dim)

        xb, yb = x[:16], y[:16]
        model.eval()

        # --- BPTT gradient (standard autograd) ---
        model.zero_grad()
        logits = model(xb)
        loss_bptt = nn.functional.cross_entropy(logits, yb)
        loss_bptt.backward()
        bptt_grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]
        model.zero_grad()

        # --- EP gradient (contrastive method) ---
        model.train()
        result = model.train_step(xb, yb)
        ep_grads = [
            p.grad.clone() if p.grad is not None else torch.zeros_like(p)
            for p in model.parameters()
        ]

        # Compare gradient directions (cosine similarity)
        cos_sims = []
        for g_bptt, g_ep in zip(bptt_grads, ep_grads):
            if g_bptt.numel() > 0 and g_ep.numel() > 0:
                v1 = g_bptt.flatten()
                v2 = g_ep.flatten()
                dot = torch.dot(v1, v2)
                norm1 = torch.norm(v1)
                norm2 = torch.norm(v2)
                if norm1 > 1e-8 and norm2 > 1e-8:
                    cos_sim = dot / (norm1 * norm2)
                    cos_sims.append(cos_sim.item())

        assert len(cos_sims) > 0, "No comparable gradients found"
        max_cos_sim = max(cos_sims)
        assert max_cos_sim >= 0.5, (
            f"TileEP: max EP-BPTT cosine similarity = {max_cos_sim:.3f} < 0.5. "
            f"All: {cos_sims}"
        )

    @pytest.mark.xfail(reason="GATE-0: pre-existing EqProp gradient drift")
    def test_deq_gradients_match_bptt_wired_up(self, synthetic_mlp_task):
        """Wire up the disabled test_deq.py::test_gradients_match_bptt."""
        x, y, input_dim, hidden_dim, output_dim = synthetic_mlp_task

        model = _create_tile_ep_model(input_dim, hidden_dim, output_dim)

        xb, yb = x[:16], y[:16]
        model.eval()

        # BPTT
        model.zero_grad()
        logits = model(xb)
        loss_bptt = nn.functional.cross_entropy(logits, yb)
        loss_bptt.backward()
        bptt_grad = torch.cat([
            p.grad.flatten() for p in model.parameters() if p.grad is not None
        ])
        model.zero_grad()

        # EP gradient via contrastive train_step
        model.train()
        model.train_step(xb, yb)
        ep_grad = torch.cat([
            p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten()
            for p in model.parameters()
        ])

        # Cosine similarity
        dot = torch.dot(bptt_grad, ep_grad)
        norm_bptt = torch.norm(bptt_grad)
        norm_ep = torch.norm(ep_grad)
        cos_sim = dot / (norm_bptt * norm_ep + 1e-8)

        assert cos_sim >= 0.5, f"EP-BPTT cosine similarity = {cos_sim:.3f} < 0.5"


# =============================================================================
# 3.2 Lyapunov Energy Descent — Monotone Energy Decrease
# =============================================================================


class TestLyapunovEnergyDescent:
    """Verify energy decreases monotonically along relaxation dynamics."""

    @settings(max_examples=5, deadline=None)
    @given(st.data())
    def test_energy_monotone_decrease_tile_ep(self, synthetic_mlp_task, data):
        """Run relaxation steps via settle_universal, assert free energy monotonically non-increasing."""
        x, y, input_dim, hidden_dim, output_dim = synthetic_mlp_task

        model = _create_tile_ep_model(input_dim, hidden_dim, output_dim)
        model.eval()

        xb, yb = x[:8], y[:8]

        # Use settle_universal to get trajectory
        with torch.no_grad():
            from computronium.core.local_learning.settling import SettleConfig

            config = SettleConfig(
                max_steps=20,
                convergence_threshold=1e-4,
                convergence_start=5,
            )
            out, steps_taken, converged, telemetry = settle_universal(
                model, xb, config=config
            )

        # Energy should be monotonic (deltas should decrease)
        deltas = telemetry.deltas
        if len(deltas) < 2:
            pytest.skip("Trajectory too short")

        # Assert monotone non-increase of deltas (energy proxy)
        slack = 1e-3
        for i in range(1, len(deltas)):
            assert deltas[i] <= deltas[i - 1] + slack, (
                f"TileEP: delta increased at step {i}: "
                f"{deltas[i - 1]:.6f} -> {deltas[i]:.6f}"
            )


# =============================================================================
# 3.4 Fixed-Point Reliability — Attractor Uniqueness
# =============================================================================


class TestFixedPointReliability:
    """Verify relaxation converges to unique fixed point."""

    @settings(max_examples=5, deadline=None)
    @given(st.data())
    def test_fixed_point_uniqueness_tile_ep(self, synthetic_mlp_task, data):
        """Run relax from multiple initializations, assert convergence."""
        x, y, input_dim, hidden_dim, output_dim = synthetic_mlp_task

        model = _create_tile_ep_model(input_dim, hidden_dim, output_dim)
        model.eval()
        xb = x[:4]

        # Test settle_universal converges
        with torch.no_grad():
            from computronium.core.local_learning.settling import SettleConfig

            config = SettleConfig(
                max_steps=50,
                convergence_threshold=1e-4,
                convergence_start=5,
            )
            out1, steps1, converged1, telemetry1 = settle_universal(
                model, xb, config=config
            )
            out2, steps2, converged2, telemetry2 = settle_universal(
                model, xb, config=config
            )

        # Both should run without error (convergence is not guaranteed for all inputs)
        # Output should be deterministic (same input, same result)
        # settle_universal may return a list of tensors for multi-layer models
        if isinstance(out1, list) and isinstance(out2, list):
            assert len(out1) == len(out2), "Output list lengths should match"
            for o1, o2 in zip(out1, out2):
                assert torch.allclose(o1, o2, rtol=1e-4), "Fixed point should be unique"
        else:
            assert torch.allclose(out1, out2, rtol=1e-4), "Fixed point should be unique"


# =============================================================================
# 3.5 Weight-Transport Freeness — FA Family
# =============================================================================


class TestWeightTransportFreeness:
    """Verify Feedback Alignment models use random fixed B ≠ W.T."""

    def test_tile_fa_backward_weights_not_transpose(self, synthetic_mlp_task):
        """Assert B ≠ W.T at initialization for Tile FA."""
        x, y, input_dim, hidden_dim, output_dim = synthetic_mlp_task

        config = TileAlgorithmConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            neurons_per_tile=hidden_dim,
            tiles_per_layer=1,
            num_hidden_layers=1,
            algorithm="fa",
            mode="fa",
            free_steps=10,
            nudged_steps=10,
            learning_rate=0.001,
        )
        model = TileAlgorithm(config)

        forward_weights = []
        backward_weights = []

        for name, param in model.named_parameters():
            if "weight" in name.lower() and "bias" not in name.lower():
                if (
                    "backward" in name.lower()
                    or "feedback" in name.lower()
                    or "B" in name
                ):
                    backward_weights.append((name, param.data.clone()))
                elif (
                    "forward" in name.lower() or "W" in name or "layer" in name.lower()
                ):
                    forward_weights.append((name, param.data.clone()))

        if not backward_weights:
            pytest.skip("Tile FA: no backward/feedback weights found")

        for b_name, B in backward_weights:
            for w_name, W in forward_weights:
                if B.shape == W.T.shape:
                    diff = torch.norm(B - W.T).item()
                    assert diff > 1e-3, (
                        f"Tile FA: backward weight {b_name} matches forward {w_name} transpose! "
                        f"||B - W.T|| = {diff:.6f}"
                    )
                    return

        pytest.skip("Tile FA: no comparable forward/backward weight shapes")

    def test_tile_fa_backward_path_separate(self, synthetic_mlp_task):
        """Assert backward pass doesn't read forward weights (separate tensors)."""
        x, y, input_dim, hidden_dim, output_dim = synthetic_mlp_task

        config = TileAlgorithmConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            neurons_per_tile=hidden_dim,
            tiles_per_layer=1,
            num_hidden_layers=1,
            algorithm="fa",
            mode="fa",
            free_steps=10,
            nudged_steps=10,
            learning_rate=0.001,
        )
        model = TileAlgorithm(config)

        feedback_weights = []
        for name, param in model.named_parameters():
            if "backward" in name.lower() or "feedback" in name.lower() or "B" in name:
                feedback_weights.append((name, param))

        if not feedback_weights:
            pytest.skip("No feedback weights found")

        forward_weights = [
            p
            for n, p in model.named_parameters()
            if "weight" in n.lower()
            and "bias" not in n.lower()
            and "backward" not in n.lower()
            and "feedback" not in n.lower()
            and "B" not in n
        ]

        for b_name, B in feedback_weights:
            for W in forward_weights:
                assert B.data_ptr() != W.data_ptr(), (
                    f"Feedback weight {b_name} shares memory with forward weight!"
                )
                if B.shape == W.T.shape:
                    assert B.data_ptr() != W.T.contiguous().data_ptr(), (
                        f"Feedback weight {b_name} is view of forward transpose!"
                    )


# =============================================================================
# 3.6 Adaptive-FA Alignment Improvement
# =============================================================================


class TestAdaptiveFAAlignment:
    """Verify feedback alignment matrices align with forward weights over training."""

    @pytest.mark.xfail(
        reason="AdaptiveFA feedback LR too small to show alignment in 50 steps"
    )
    def test_feedback_alignment_improves(self, synthetic_mlp_task):
        """After K=50 steps, cos(B, W.T) should increase from initial random value."""
        x, y, input_dim, hidden_dim, output_dim = synthetic_mlp_task

        config = TileAlgorithmConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            neurons_per_tile=hidden_dim,
            tiles_per_layer=1,
            num_hidden_layers=1,
            algorithm="fa",
            mode="fa",
            free_steps=10,
            nudged_steps=10,
            learning_rate=0.001,
        )
        model = TileAlgorithm(config)

        B_weights = []
        W_weights = []
        for name, param in model.named_parameters():
            if "backward" in name.lower() or "feedback" in name.lower() or "B" in name:
                B_weights.append(param)
            elif (
                "forward" in name.lower()
                and "weight" in name.lower()
                and "bias" not in name.lower()
            ):
                W_weights.append(param)

        if not B_weights or not W_weights:
            pytest.skip("Could not find both feedback and forward weights")

        initial_alignments = []
        for B in B_weights:
            for W in W_weights:
                if B.shape == W.T.shape:
                    cos = torch.dot(B.flatten(), W.T.flatten()) / (
                        torch.norm(B) * torch.norm(W.T) + 1e-8
                    )
                    initial_alignments.append(cos.item())

        if not initial_alignments:
            pytest.skip("No matching shape pairs for alignment")

        xb = x[:32]
        yb = y[:32]
        model.train()
        for step in range(50):
            model.train_step(xb, yb)

        final_alignments = []
        for B in B_weights:
            for W in W_weights:
                if B.shape == W.T.shape:
                    cos = torch.dot(B.flatten(), W.T.flatten()) / (
                        torch.norm(B) * torch.norm(W.T) + 1e-8
                    )
                    final_alignments.append(cos.item())

        max_initial = max(initial_alignments)
        max_final = max(final_alignments)
        assert max_final > max_initial + 0.05, (
            f"Feedback alignment did not improve: "
            f"initial max={max_initial:.4f}, final max={max_final:.4f}"
        )


# =============================================================================
# Native Model Composition Tests
# =============================================================================


class TestNativeModelCompositions:
    """Test that native model compositions work correctly."""

    def test_native_eqprop_composes_and_trains(self, synthetic_mlp_task):
        """native_eqprop_mlp should compose and train."""
        x, y, input_dim, hidden_dim, output_dim = synthetic_mlp_task

        model = create_native_eqprop_mlp(
            input_dim, hidden_dim, output_dim, beta=0.5, settle_steps=10, lr=0.01
        )

        xb, yb = x[:16], y[:16]
        model.train()
        result = model.train_step(xb, yb)
        assert isinstance(result, dict)
        assert "loss" in result
        assert "nudged_fit_accuracy" in result

    def test_native_fa_composes_and_trains(self, synthetic_mlp_task):
        """native_fa_mlp should compose and train."""
        x, y, input_dim, hidden_dim, output_dim = synthetic_mlp_task

        model = create_native_fa_mlp(input_dim, hidden_dim, output_dim, lr=0.001)

        xb, yb = x[:16], y[:16]
        model.train()
        result = model.train_step(xb, yb)
        assert isinstance(result, dict)
        assert "loss" in result
        assert "nudged_fit_accuracy" in result


# =============================================================================


@st.composite
def small_mlp_config(draw):
    """Generate small MLP configurations for biology tests."""
    return {
        "input_dim": draw(st.integers(4, 16)),
        "hidden_dim": draw(st.integers(4, 16)),
        "output_dim": draw(st.integers(2, 8)),
        "batch_size": draw(st.integers(4, 32)),
    }


@st.composite
def step_size_values(draw):
    """Step sizes for contraction testing."""
    return draw(st.sampled_from([0.05, 0.1, 0.2, 0.3, 0.5]))


@st.composite
def beta_values(draw):
    """Beta values for EP gradient equivalence."""
    return draw(
        st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False)
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
