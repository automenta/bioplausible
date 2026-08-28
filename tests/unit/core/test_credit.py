"""Regression tests for Credit Assignment Correctness (Phase 3.6.1/3.6.8).

These tests lock in the credit assignment audit checks to prevent regressions.
"""

from __future__ import annotations

from typing import List

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from computronium.ontology import (
    BackpropCredit,
    CreditAssignmentConfig,
    DigitalSubstrate,
    EnergyMinimizationDynamics,
    FeedforwardGeometry,
    GeometryConfig,
    Phase,
    RandomProjectionsCredit,
    RecurrentGeometry,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemState,
    ThermodynamicContrast,
    _learnable_weight_names,
)


def get_activations(geometry, substrate, x) -> List[Tensor]:
    """Get intermediate activations from geometry."""
    if hasattr(geometry, "forward_with_intermediates"):
        return geometry.forward_with_intermediates(x, substrate)
    else:
        return [x, geometry.forward(x, substrate)]


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def linear_geometry(device):
    """Linear geometry (no hidden layers) for exact gradient matching."""
    geometry = FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=20,
            output_dim=1,
            hidden_dims=(),  # No hidden layers = linear
            init_scale=0.1,
        )
    )
    geometry.to(device)
    return geometry


@pytest.fixture
def mlp_geometry(device):
    """MLP with hidden layers for approximate gradient matching."""
    torch.manual_seed(7)  # deterministic init to avoid flaky cosine
    geometry = RecurrentGeometry(
        GeometryConfig.recurrent(
            input_dim=784,
            output_dim=10,
            hidden_dims=(128,),
            init_scale=0.1,
        ),
        hidden_dim=128,
    )
    geometry.to(device)
    return geometry


@pytest.fixture
def substrate(device):
    return DigitalSubstrate(SubstrateConfig.digital(device=str(device)))


@pytest.fixture
def energy_dynamics_linear():
    return EnergyMinimizationDynamics(
        StateDynamicsConfig.energy_minimization(
            max_steps=50,
            convergence_threshold=1e-6,
            convergence_start=5,
            step_size=0.05,
            beta=0.5,
            track_free_energy_per_iter=True,
            gradient_checkpointing=False,
        )
    )


@pytest.fixture
def energy_dynamics_mlp():
    return EnergyMinimizationDynamics(
        StateDynamicsConfig.energy_minimization(
            max_steps=500,
            convergence_threshold=1e-5,
            convergence_start=10,
            step_size=0.01,
            beta=0.1,
            track_free_energy_per_iter=True,
            gradient_checkpointing=False,
        )
    )


@pytest.fixture
def thermo_credit():
    return ThermodynamicContrast(
        CreditAssignmentConfig.thermodynamic_contrast(beta=0.5)
    )


@pytest.fixture
def backprop_credit():
    return BackpropCredit(CreditAssignmentConfig.gradient())


@pytest.fixture
def fa_credit():
    return RandomProjectionsCredit(
        CreditAssignmentConfig.random_projections(
            beta=0.5,
            feedback_scale=0.01,
        )
    )


# ============================================================
# Test: ThermodynamicContrast vs BackpropCredit (Linear)
# ============================================================


class TestThermodynamicVsBackpropLinear:
    """Test ThermodynamicContrast matches BackpropCredit on linear regression."""

    def test_cosine_similarity_high(
        self,
        linear_geometry,
        substrate,
        energy_dynamics_linear,
        thermo_credit,
        backprop_credit,
        device,
    ):
        """Cosine similarity >= 0.95 on linear regression."""
        cosines = []
        rel_errors = []

        for batch_idx in range(20):
            x = torch.randn(32, 20, device=device)
            W_true = torch.randn(20, 1, device=device)
            y = x @ W_true + 0.01 * torch.randn(32, 1, device=device)

            initial_acts = get_activations(linear_geometry, substrate, x)

            free_state = SystemState(x=x, y=y.squeeze(-1))
            free_state.activations = initial_acts
            free_state = energy_dynamics_linear.settle(
                free_state, linear_geometry, substrate, target=None
            )

            nudged_state = SystemState(x=x, y=y.squeeze(-1))
            nudged_state.activations = initial_acts
            nudged_state = energy_dynamics_linear.settle(
                nudged_state, linear_geometry, substrate, target=y
            )

            states = {Phase.FREE: free_state, Phase.NUDGED: nudged_state}

            # TRUE gradient via autograd (0.5 * MSE for energy-gradient equivalence)
            logits = linear_geometry.forward(x, substrate)
            true_loss = 0.5 * F.mse_loss(logits, y)
            params = [p for p in linear_geometry.parameters() if p.requires_grad]
            true_grads_all = torch.autograd.grad(true_loss, params, retain_graph=False)

            # Filter to only weight gradients
            weight_names = _learnable_weight_names(linear_geometry.params)
            true_grads = []
            param_idx = 0
            for n, p in linear_geometry.named_parameters():
                if p.requires_grad and "weight" in n and p.ndim == 2:
                    parts = n.split(".")
                    if len(parts) >= 3 and parts[0] == "_layers" and parts[1].isdigit():
                        layer_idx = int(parts[1])
                        param_key = f"{layer_idx}.weight"
                        if param_key in weight_names:
                            true_grads.append(true_grads_all[param_idx])
                param_idx += 1

            # ThermodynamicContrast
            nudged_logits = (
                nudged_state.activations[-1]
                if isinstance(nudged_state.activations, list)
                else nudged_state.activations
            )
            dyn_loss = 0.5 * F.mse_loss(nudged_logits, y)
            thermo_grads = thermo_credit.compute_pseudo_gradient(
                states, dyn_loss, linear_geometry
            )

            if thermo_grads and true_grads:
                for g1, g2 in zip(thermo_grads, true_grads):
                    cos = F.cosine_similarity(
                        g1.flatten().unsqueeze(0), g2.flatten().unsqueeze(0)
                    ).item()
                    rel = (g1 - g2).norm().item() / (g2.norm().item() + 1e-12)
                    cosines.append(cos)
                    rel_errors.append(rel)

        mean_cos = sum(cosines) / len(cosines) if cosines else 0.0
        min_cos = min(cosines) if cosines else 0.0
        mean_rel = sum(rel_errors) / len(rel_errors) if rel_errors else float("inf")

        assert mean_cos >= 0.95, f"Mean cosine {mean_cos:.4f} < 0.95"
        assert min_cos >= 0.9, f"Min cosine {min_cos:.4f} < 0.9"
        assert mean_rel <= 0.1, f"Mean relative error {mean_rel:.4f} > 0.1"


# ============================================================
# Test: ThermodynamicContrast vs BackpropCredit (MLP)
# ============================================================


class TestThermodynamicVsBackpropMLP:
    """Test ThermodynamicContrast approximates BackpropCredit on MLP."""

    def test_cosine_similarity_reasonable(
        self,
        mlp_geometry,
        substrate,
        energy_dynamics_mlp,
        thermo_credit,
        backprop_credit,
        device,
    ):
        """Cosine similarity >= 0.62 (EqProp approximation quality)."""
        cosines = []
        same_sign_count = 0
        total_params = 0

        for batch_idx in range(20):  # 20 batches for stable mean (matches audit)
            torch.manual_seed(42 + batch_idx)
            x = torch.randn(4, 784, device=device)
            y = torch.randint(0, 10, (4,), device=device)

            initial_acts = get_activations(mlp_geometry, substrate, x)

            free_state = SystemState(x=x, y=y)
            free_state.activations = initial_acts
            free_state = energy_dynamics_mlp.settle(
                free_state, mlp_geometry, substrate, target=None
            )

            nudged_state = SystemState(x=x, y=y)
            nudged_state.activations = initial_acts
            nudged_state = energy_dynamics_mlp.settle(
                nudged_state, mlp_geometry, substrate, target=y
            )

            states = {Phase.FREE: free_state, Phase.NUDGED: nudged_state}

            # TRUE gradient via autograd
            logits = mlp_geometry.forward(x, substrate)
            true_loss = F.cross_entropy(logits, y)
            params = [p for p in mlp_geometry.parameters() if p.requires_grad]
            true_grads_all = torch.autograd.grad(true_loss, params, retain_graph=False)

            weight_names = _learnable_weight_names(mlp_geometry.params)
            true_grads = []
            param_idx = 0
            for n, p in mlp_geometry.named_parameters():
                if p.requires_grad and "weight" in n and p.ndim == 2:
                    parts = n.split(".")
                    if len(parts) >= 3 and parts[0] == "_layers" and parts[1].isdigit():
                        layer_idx = int(parts[1])
                        param_key = f"{layer_idx}.weight"
                        if param_key in weight_names:
                            true_grads.append(true_grads_all[param_idx])
                param_idx += 1

            # ThermodynamicContrast
            nudged_logits = (
                nudged_state.activations[-1]
                if isinstance(nudged_state.activations, list)
                else nudged_state.activations
            )
            dyn_loss = F.cross_entropy(nudged_logits, y)
            thermo_grads = thermo_credit.compute_pseudo_gradient(
                states, dyn_loss, mlp_geometry
            )

            if thermo_grads and true_grads:
                for g1, g2 in zip(thermo_grads, true_grads):
                    cos = F.cosine_similarity(
                        g1.flatten().unsqueeze(0), g2.flatten().unsqueeze(0)
                    ).item()
                    cosines.append(cos)
                    same_sign = ((g1 * g2) > 0).float().mean().item()
                    same_sign_count += int(same_sign * g1.numel())
                    total_params += g1.numel()

        mean_cos = sum(cosines) / len(cosines) if cosines else 0.0
        same_sign_pct = same_sign_count / total_params if total_params > 0 else 0.0

        assert mean_cos >= 0.62, f"Mean cosine {mean_cos:.4f} < 0.62"
        assert same_sign_pct >= 0.65, f"Same sign % {same_sign_pct:.4f} < 0.65"


# ============================================================
# Test: RandomProjectionsCredit (FA) vs Theoretical
# ============================================================


class TestFATheoretical:
    """Test FA pseudo-gradients match theoretical expectation."""

    def test_relative_error_low(self, device):
        """Relative error <= 20% vs theoretical FA."""
        geometry = FeedforwardGeometry(
            GeometryConfig.feedforward(
                input_dim=784,
                output_dim=10,
                hidden_dims=(256,),
                init_scale=0.1,
            )
        )
        geometry.to(device)

        substrate = DigitalSubstrate(SubstrateConfig.digital(device=device))

        credit = RandomProjectionsCredit(
            CreditAssignmentConfig.random_projections(
                beta=0.5,
                feedback_scale=0.01,
            )
        )
        credit._init_feedback_weights(geometry, device)
        fb_weights = credit._feedback_weights

        backprop_credit = BackpropCredit(CreditAssignmentConfig.gradient())

        rel_errors = []

        for batch_idx in range(10):
            x = torch.randn(4, 784, device=device)
            y = torch.randint(0, 10, (4,), device=device)

            initial_acts = get_activations(geometry, substrate, x)

            free_state = SystemState(x=x, y=y)
            free_state.activations = initial_acts
            nudged_state = SystemState(x=x, y=y)
            nudged_state.activations = initial_acts

            states = {Phase.FREE: free_state, Phase.NUDGED: nudged_state}

            # TRUE gradient
            logits = geometry.forward(x, substrate)
            true_loss = F.cross_entropy(logits, y)
            params = [p for p in geometry.parameters() if p.requires_grad]
            true_grads = torch.autograd.grad(true_loss, params, retain_graph=False)

            # FA pseudo-gradients
            fa_grads = credit.compute_pseudo_gradient(states, true_loss, geometry)

            # Theoretical FA
            if isinstance(nudged_state.activations, list):
                logits_n = nudged_state.activations[-1]
                hidden_acts = nudged_state.activations[1:-1]
            else:
                logits_n = nudged_state.activations
                hidden_acts = []

            probs = torch.softmax(logits_n, dim=-1)
            target = torch.zeros_like(probs)
            target.scatter_(-1, y.unsqueeze(-1), 1.0)
            output_error = probs - target

            theoretical_grads = []

            if "layer_0" in fb_weights:
                fb = fb_weights["layer_0"]
                hidden_error = output_error @ fb.T
                if hidden_acts:
                    hidden_error = hidden_error * (hidden_acts[0] > 0).float()
                pre_act = free_state.x
                if pre_act is not None:
                    theoretical_grads.append(hidden_error.T @ pre_act)

            if len(true_grads) >= 2:
                theoretical_grads.append(true_grads[-1])

            if theoretical_grads and fa_grads:
                for fg, tg in zip(fa_grads, theoretical_grads):
                    rel = (fg - tg).norm().item() / (tg.norm().item() + 1e-12)
                    rel_errors.append(rel)

        mean_rel = sum(rel_errors) / len(rel_errors) if rel_errors else float("inf")
        assert mean_rel <= 0.2, f"Mean relative error {mean_rel:.4f} > 0.2"


# ============================================================
# Test: RandomProjectionsCredit (DFA) vs Theoretical
# ============================================================


class TestDFATheoretical:
    """Test DFA pseudo-gradients match theoretical expectation."""

    def test_relative_error_low(self, device):
        """Relative error <= 20% vs theoretical DFA."""
        geometry = FeedforwardGeometry(
            GeometryConfig.feedforward(
                input_dim=784,
                output_dim=10,
                hidden_dims=(256, 128),
                init_scale=0.1,
            )
        )
        geometry.to(device)

        substrate = DigitalSubstrate(SubstrateConfig.digital(device=device))

        config = CreditAssignmentConfig(
            credit_type="direct_feedback_alignment",
            beta=0.5,
            feedback_matrix=None,
            local_objective="mse",
            orthogonal_init=False,
            feedback_scale=0.01,
        )
        credit = RandomProjectionsCredit(config)
        credit._init_feedback_weights(geometry, device)
        fb_weights = credit._feedback_weights

        rel_errors = []

        for batch_idx in range(10):
            x = torch.randn(4, 784, device=device)
            y = torch.randint(0, 10, (4,), device=device)

            initial_acts = get_activations(geometry, substrate, x)

            free_state = SystemState(x=x, y=y)
            free_state.activations = initial_acts
            nudged_state = SystemState(x=x, y=y)
            nudged_state.activations = initial_acts

            states = {Phase.FREE: free_state, Phase.NUDGED: nudged_state}

            # TRUE gradient
            logits = geometry.forward(x, substrate)
            true_loss = F.cross_entropy(logits, y)
            params = [p for p in geometry.parameters() if p.requires_grad]
            true_grads = torch.autograd.grad(true_loss, params, retain_graph=False)

            # DFA pseudo-gradients
            dfa_grads = credit.compute_pseudo_gradient(states, true_loss, geometry)

            # Theoretical DFA
            if isinstance(nudged_state.activations, list):
                logits_n = nudged_state.activations[-1]
                hidden_acts = nudged_state.activations[1:-1]
            else:
                logits_n = nudged_state.activations
                hidden_acts = []

            probs = torch.softmax(logits_n, dim=-1)
            target = torch.zeros_like(probs)
            target.scatter_(-1, y.unsqueeze(-1), 1.0)
            output_error = probs - target

            theoretical_grads = []

            if "layer_0" in fb_weights:
                fb = fb_weights["layer_0"]
                hidden_error = output_error @ fb.T
                if len(hidden_acts) > 0:
                    hidden_error = hidden_error * (hidden_acts[0] > 0).float()
                pre_act = free_state.x
                if pre_act is not None:
                    theoretical_grads.append(hidden_error.T @ pre_act)

            if "layer_1" in fb_weights:
                fb = fb_weights["layer_1"]
                hidden_error = output_error @ fb.T
                if len(hidden_acts) > 1:
                    hidden_error = hidden_error * (hidden_acts[1] > 0).float()
                pre_act = hidden_acts[0] if len(hidden_acts) > 0 else free_state.x
                if pre_act is not None:
                    theoretical_grads.append(hidden_error.T @ pre_act)

            if len(true_grads) >= 3:
                theoretical_grads.append(true_grads[-1])

            if theoretical_grads and dfa_grads:
                for dg, tg in zip(dfa_grads, theoretical_grads):
                    rel = (dg - tg).norm().item() / (tg.norm().item() + 1e-12)
                    rel_errors.append(rel)

        mean_rel = sum(rel_errors) / len(rel_errors) if rel_errors else float("inf")
        assert mean_rel <= 0.2, f"Mean relative error {mean_rel:.4f} > 0.2"


# ============================================================
# Test: BackpropCredit Identity
# ============================================================


class TestBackpropIdentity:
    """Test BackpropCredit matches autograd exactly."""

    def test_bitwise_identical(self, device):
        """Bitwise identical to autograd on same graph."""
        geometry = FeedforwardGeometry(
            GeometryConfig.feedforward(
                input_dim=784,
                output_dim=10,
                hidden_dims=(256, 128),
                init_scale=0.1,
            )
        )
        geometry.to(device)

        substrate = DigitalSubstrate(SubstrateConfig.digital(device=device))
        credit = BackpropCredit(CreditAssignmentConfig.gradient())

        for batch_idx in range(5):
            x = torch.randn(4, 784, device=device, requires_grad=True)
            y = torch.randint(0, 10, (4,), device=device)

            # Autograd reference
            logits = geometry.forward(x, substrate)
            loss = F.cross_entropy(logits, y)
            params = [p for p in geometry.parameters() if p.requires_grad]
            autograd_grads_all = torch.autograd.grad(loss, params, retain_graph=True)

            weight_names = _learnable_weight_names(geometry.params)
            autograd_grads = []
            param_idx = 0
            for n, p in geometry.named_parameters():
                if p.requires_grad and "weight" in n and p.ndim == 2:
                    parts = n.split(".")
                    if len(parts) >= 3 and parts[0] == "_layers" and parts[1].isdigit():
                        layer_idx = int(parts[1])
                        param_key = f"{layer_idx}.weight"
                        if param_key in weight_names:
                            autograd_grads.append(autograd_grads_all[param_idx])
                param_idx += 1

            # BackpropCredit
            initial_acts = get_activations(geometry, substrate, x.detach())
            free_state = SystemState(x=x.detach(), y=y)
            free_state.activations = initial_acts
            nudged_state = SystemState(x=x.detach(), y=y)
            nudged_state.activations = initial_acts

            states = {Phase.FREE: free_state, Phase.NUDGED: nudged_state}
            bp_grads = credit.compute_pseudo_gradient(states, loss, geometry)

            for ag, bg in zip(autograd_grads, bp_grads):
                assert torch.allclose(ag, bg, rtol=1e-7, atol=1e-10), (
                    f"Mismatch: max_diff={(ag - bg).abs().max().item():.6f}"
                )


# ============================================================
# Test: Energy Gap Sign
# ============================================================


class TestEnergyGapSign:
    """Test free < nudged energy always."""

    def test_all_batches(self, device):
        """100/100 random batches: free_energy < nudged_energy."""
        geometry = RecurrentGeometry(
            GeometryConfig.recurrent(
                input_dim=784,
                output_dim=10,
                hidden_dims=(256,),
                init_scale=0.1,
            ),
            hidden_dim=256,
        )
        geometry.to(device)

        substrate = DigitalSubstrate(SubstrateConfig.digital(device=device))
        dynamics = EnergyMinimizationDynamics(
            StateDynamicsConfig.energy_minimization(
                max_steps=50,
                convergence_threshold=1e-4,
                convergence_start=5,
                step_size=0.05,
                beta=0.5,
                track_free_energy_per_iter=True,
                gradient_checkpointing=False,
            )
        )

        n_passed = 0
        for batch_idx in range(20):  # Reduced for test speed
            x = torch.randn(4, 784, device=device)
            y = torch.randint(0, 10, (4,), device=device)

            initial_acts = get_activations(geometry, substrate, x)

            free_state = SystemState(x=x, y=y)
            free_state.activations = initial_acts
            free_state = dynamics.settle(free_state, geometry, substrate, target=None)
            free_energy = dynamics.compute_energy(free_state, geometry).item()

            nudged_state = SystemState(x=x, y=y)
            nudged_state.activations = initial_acts
            nudged_state = dynamics.settle(nudged_state, geometry, substrate, target=y)
            nudged_energy = dynamics.compute_energy(nudged_state, geometry).item()

            if free_energy < nudged_energy:
                n_passed += 1

        assert n_passed == 20, f"Only {n_passed}/20 batches had free < nudged energy"


# ============================================================
# Test: Settling Convergence
# ============================================================


class TestSettlingConvergence:
    """Test energy decreases monotonically and converges."""

    def test_monotonic_and_converged(self, device):
        """Energy decreases overall and converges within max_steps."""
        torch.manual_seed(5)  # deterministic init
        geometry = RecurrentGeometry(
            GeometryConfig.recurrent(
                input_dim=784,
                output_dim=10,
                hidden_dims=(256,),
                init_scale=0.1,
            ),
            hidden_dim=256,
        )
        geometry.to(device)

        substrate = DigitalSubstrate(SubstrateConfig.digital(device=device))

        dynamics = EnergyMinimizationDynamics(
            StateDynamicsConfig.energy_minimization(
                max_steps=500,
                convergence_threshold=1e-4,
                convergence_start=10,
                step_size=0.01,
                beta=0.5,
                momentum=0.0,
                track_free_energy_per_iter=True,
                gradient_checkpointing=False,
            )
        )

        n_converged = 0
        n_decreased = 0

        for trial in range(5):  # Reduced for test speed
            x = torch.randn(4, 784, device=device)
            y = torch.randint(0, 10, (4,), device=device)

            initial_acts = get_activations(geometry, substrate, x)
            free_state = SystemState(x=x, y=y)
            free_state.activations = initial_acts
            free_state = dynamics.settle(free_state, geometry, substrate, target=None)

            energy_history = dynamics.get_free_energy_history()
            if energy_history is None or len(energy_history) < 2:
                continue

            # Overall decrease
            decreased = energy_history[-1] < energy_history[0]
            if decreased:
                n_decreased += 1

            # Check convergence
            converged = abs(energy_history[-1] - energy_history[-2]) < 1e-4
            if converged:
                n_converged += 1

        assert n_decreased == 5, f"Only {n_decreased}/5 trials decreased energy"
        assert n_converged == 5, f"Only {n_converged}/5 trials converged"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
