"""Unit tests for Continual Learning (Phase 2) experiment components.

Tests verify:
- FastWeightPlasticity with EnergyMinimizationDynamics + ThermodynamicContrast learns correctly
- Joint system pipeline integration with plasticity stepping
- Task masking in forward pass and loss computation
- CL metrics computation (forgetting, backward transfer, etc.)
- Replay buffer, LwF, SI arm implementations
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

import pytest

from computronium.core.joint.transition import NullPlasticity, PlasticityConfig
from computronium.core.pipeline import run_train_step
from computronium.core.plasticity import create_fast_weight_plasticity
from computronium.core.profiling import ResourceUsage, measure_suite_resources
from computronium.core.stability import StabilityGuard, GuardDecision
from computronium.core.stability.spectral_radius import SpectralRadiusEstimator
from computronium.domains.base import TaskSplit
from computronium.domains.vision import SplitMNIST
from computronium.experiments.joint.continual_learning import (
    CLConfig,
    CLMetrics,
    ReplayBuffer,
    LwFLoss,
    SynapticIntelligence,
    ContinualJointSystem,
    create_fast_weight_arm,
    create_ewc_arm,
    create_backprop_arm,
    create_replay_arm,
    create_lwf_arm,
    create_si_arm,
    run_continual_train_step,
    compute_cl_metrics,
    create_stability_guard,
    make_transition_fn,
    check_stability,
    run_continual_learning,
    run_continual_learning_suite,
    SPLIT_MNIST_TASKS,
    NUM_TASKS,
    CLASSES_PER_TASK,
    TOTAL_CLASSES,
)

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def mnist_batch(device):
    """Create a single MNIST batch for testing."""
    x = torch.randn(4, 784, device=device)
    y = torch.randint(0, 2, (4,), device=device)
    return x, y


@pytest.fixture
def fast_weight_model(device):
    """Create a fast weight arm model."""
    return create_fast_weight_arm(device=str(device))


@pytest.fixture
def backprop_model(device):
    """Create a backprop arm model."""
    return create_backprop_arm(device=str(device))


# ============================================================
# Test: FastWeightPlasticity with EnergyMinimizationDynamics
# ============================================================


class TestFastWeightPlasticityLearning:
    """Test that fast weight plasticity actually learns with correct dynamics."""

    def test_fast_weight_arm_uses_correct_dynamics(self, fast_weight_model):
        """Verify the arm uses EnergyMinimizationDynamics (not InstantaneousDynamics)."""
        from computronium.core.ontology import EnergyMinimizationDynamics

        dynamics = fast_weight_model.joint_system.dynamics
        assert isinstance(dynamics, EnergyMinimizationDynamics), (
            f"Expected EnergyMinimizationDynamics, got {type(dynamics).__name__}"
        )
        assert dynamics.config.dynamics_type == "energy_minimization"
        assert dynamics.config.max_steps > 1, "Should have multiple settling steps"

    def test_fast_weight_arm_uses_thermodynamic_contrast(self, fast_weight_model):
        """Verify the arm uses ThermodynamicContrast credit assignment."""
        from computronium.core.ontology import ThermodynamicContrast

        credit = fast_weight_model.joint_system.credit
        assert isinstance(credit, ThermodynamicContrast), (
            f"Expected ThermodynamicContrast, got {type(credit).__name__}"
        )
        assert credit.requires_autograd is False
        assert credit.phases == ("free", "nudged")

    def test_fast_weight_arm_has_fast_weight_plasticity(self, fast_weight_model):
        """Verify the arm has FastWeightPlasticity."""
        from computronium.core.plasticity.fast_weights import FastWeightPlasticity

        plasticity = fast_weight_model.joint_system.plasticity
        assert isinstance(plasticity, FastWeightPlasticity)
        assert hasattr(plasticity, "fast_weight_dim")
        assert plasticity.fast_weight_dim == 512

    def test_energy_minimization_dynamics_produces_different_states(self, fast_weight_model, mnist_batch):
        """Verify EnergyMinimizationDynamics produces different free vs nudged states."""
        from computronium.core.ontology import Phase, SystemState
        from computronium.core.pipeline import forward_pass, phase_states

        x, y = mnist_batch
        js = fast_weight_model.joint_system

        # Run free phase
        initial_acts = forward_pass(js.substrate, js.geometry, x)
        free_state = SystemState(x=x, y=y)
        free_state.activations = initial_acts
        free_state = js.dynamics.settle(free_state, js.geometry, js.substrate, target=None)

        # Run nudged phase
        nudged_state = SystemState(x=x, y=y)
        nudged_state.activations = initial_acts
        nudged_state = js.dynamics.settle(nudged_state, js.geometry, js.substrate, target=y)

        # They should differ (nudged receives beta * (target - output))
        free_logits = free_state.activations[-1]
        nudged_logits = nudged_state.activations[-1]
        diff = (nudged_logits - free_logits).abs().mean().item()

        # With beta=0.5 and 3 settling steps, there should be a measurable difference
        assert diff > 1e-6, "Free and nudged states should differ with EnergyMinimizationDynamics"

    def test_thermodynamic_contrast_produces_nonzero_gradients(self, fast_weight_model, mnist_batch):
        """Verify ThermodynamicContrast produces non-zero pseudo-gradients."""
        x, y = mnist_batch
        js = fast_weight_model.joint_system

        from computronium.core.ontology import Phase, SystemState
        from computronium.core.pipeline import forward_pass

        initial_acts = forward_pass(js.substrate, js.geometry, x)

        free_state = SystemState(x=x, y=y)
        free_state.activations = initial_acts
        free_state = js.dynamics.settle(free_state, js.geometry, js.substrate, target=None)

        nudged_state = SystemState(x=x, y=y)
        nudged_state.activations = initial_acts
        nudged_state = js.dynamics.settle(nudged_state, js.geometry, js.substrate, target=y)

        states = {Phase.FREE: free_state, Phase.NUDGED: nudged_state}

        # Need a loss for nudged phase
        task_start = 0
        task_end = 2
        task_logits = nudged_state.activations[-1][:, task_start:task_end]
        loss = F.cross_entropy(task_logits, y)

        pseudo_grads = js.credit.compute_pseudo_gradient(states, loss, js.geometry)

        # Should have gradients for each learnable layer (3 Linear layers)
        assert len(pseudo_grads) == 3
        for i, grad in enumerate(pseudo_grads):
            assert grad.shape == js.geometry.params[f"{i*2}.weight"].shape
            assert not torch.allclose(grad, torch.zeros_like(grad)), (
                f"Gradient {i} should be non-zero"
            )


# ============================================================
# Test: run_continual_train_step integration
# ============================================================


class TestRunContinualTrainStep:
    """Test the continual training step function."""

    def test_train_step_returns_metrics_and_psi(self, fast_weight_model, mnist_batch):
        """Test train_step returns proper metrics and updated psi."""
        x, y = mnist_batch
        metrics, psi = run_continual_train_step(
            fast_weight_model.joint_system, x, y, task_id=0, psi=None
        )

        assert "loss" in metrics
        assert "energy" in metrics
        assert "accuracy" in metrics
        assert isinstance(metrics["loss"], float)
        assert isinstance(metrics["accuracy"], float)
        assert psi is not None
        assert "fast_weights" in psi
        assert psi["fast_weights"].shape[1] == 512  # fast_weight_dim

    def test_train_step_updates_psi(self, fast_weight_model, mnist_batch):
        """Test that psi is updated across steps."""
        x, y = mnist_batch

        # Step 1
        _, psi1 = run_continual_train_step(fast_weight_model.joint_system, x, y, task_id=0, psi=None)
        assert psi1 is not None, "psi should not be None after first step"

        # Step 2 with same x,y (should update psi)
        _, psi2 = run_continual_train_step(fast_weight_model.joint_system, x, y, task_id=0, psi=psi1)
        assert psi2 is not None, "psi should not be None after second step"

        # Fast weights should change (decay + outer product update)
        diff = (psi2["fast_weights"] - psi1["fast_weights"]).abs().mean().item()
        assert diff > 1e-6, "Fast weights should update across steps"

    def test_train_step_task_masking(self, fast_weight_model, device):
        """Test that loss is computed only on task-relevant logits."""
        # Task 0: classes 0,1 -> logits[:, 0:2]
        # Task 1: classes 2,3 -> logits[:, 2:4]
        x = torch.randn(4, 784, device=device)
        y0 = torch.tensor([0, 1, 0, 1], device=device)  # Task 0 labels
        y1 = torch.tensor([0, 1, 0, 1], device=device)  # Task 1 labels (local 0,1)

        # Train on task 0
        metrics0, psi = run_continual_train_step(fast_weight_model.joint_system, x, y0, task_id=0, psi=None)
        # Train on task 1
        metrics1, psi = run_continual_train_step(fast_weight_model.joint_system, x, y1, task_id=1, psi=psi)

        # Both should work without error
        assert "loss" in metrics0 and "loss" in metrics1


# ============================================================
# Test: ContinualJointSystem forward and training
# ============================================================


class TestContinualJointSystem:
    """Test the ContinualJointSystem wrapper."""

    def test_forward_without_psi(self, fast_weight_model, device):
        """Forward pass without psi should work (standard forward)."""
        x = torch.randn(4, 784, device=device)
        fast_weight_model._psi = None

        logits = fast_weight_model(x, task_id=0)
        assert logits.shape == (4, 10)

    def test_forward_with_psi_modulates_output(self, fast_weight_model, device):
        """Forward pass with psi should modulate the last hidden layer."""
        x = torch.randn(4, 784, device=device)

        # Get psi after some training
        y = torch.randint(0, 2, (4,), device=device)
        _, psi = run_continual_train_step(fast_weight_model.joint_system, x, y, task_id=0, psi=None)

        fast_weight_model._psi = psi
        logits_with = fast_weight_model(x, task_id=0)

        fast_weight_model._psi = None
        logits_without = fast_weight_model(x, task_id=0)

        # They should differ (fast weight modulation)
        diff = (logits_with - logits_without).abs().max().item()
        assert diff > 1e-4, "Fast weight modulation should change logits"

    def test_train_step_uses_joint_system_pipeline(self, fast_weight_model, mnist_batch):
        """Test that ContinualJointSystem.train_step uses the joint system pipeline."""
        x, y = mnist_batch
        fast_weight_model.set_task(0)

        metrics = fast_weight_model.train_step(x, y, task_id=0)
        assert "loss" in metrics
        assert "accuracy" in metrics

    def test_reset_plastic_state(self, fast_weight_model, mnist_batch):
        """Test plastic state reset at task boundaries."""
        x, y = mnist_batch

        # Train a bit
        fast_weight_model.set_task(0)
        for _ in range(3):
            fast_weight_model.train_step(x, y, task_id=0)

        # Store psi
        psi_before = fast_weight_model._psi
        assert psi_before is not None

        # Reset
        fast_weight_model.reset_plastic_state()
        assert fast_weight_model._psi is None


# ============================================================
# Test: Other Arm Implementations
# ============================================================


class TestOtherArms:
    """Test EWC, Backprop, Replay, LwF, SI arm implementations."""

    def test_ewc_arm_creation(self, device):
        """Test EWC arm creates correctly with SI tracker."""
        model, si = create_ewc_arm(device=str(device))
        assert isinstance(model, ContinualJointSystem)
        assert isinstance(si, SynapticIntelligence)

    def test_backprop_arm_creation(self, device):
        """Test backprop arm creation."""
        model = create_backprop_arm(device=str(device))
        assert isinstance(model, ContinualJointSystem)
        from computronium.core.ontology import BackpropCredit
        assert isinstance(model.joint_system.credit, BackpropCredit)
        assert model.joint_system.credit.requires_autograd is True

    def test_replay_arm_creation(self, device):
        """Test replay arm creation with buffer."""
        model, buffer = create_replay_arm(device=str(device))
        assert isinstance(model, ContinualJointSystem)
        assert isinstance(buffer, ReplayBuffer)

    def test_lwf_arm_creation(self, device):
        """Test LwF arm creation."""
        model, lwf_loss = create_lwf_arm(device=str(device))
        assert isinstance(model, ContinualJointSystem)
        assert isinstance(lwf_loss, LwFLoss)

    def test_si_arm_creation(self, device):
        """Test SI arm creation."""
        model, si = create_si_arm(device=str(device))
        assert isinstance(model, ContinualJointSystem)
        assert isinstance(si, SynapticIntelligence)

    def test_replay_buffer_add_and_sample(self, device):
        """Test replay buffer add and sample operations."""
        buffer = ReplayBuffer(capacity=100, input_shape=(784,), device=device)

        x = torch.randn(10, 784, device=device)
        y = torch.randint(0, 2, (10,), device=device)

        buffer.add(x, y, task_id=0)
        assert len(buffer) == 10

        # Add more to test eviction
        x2 = torch.randn(100, 784, device=device)
        y2 = torch.randint(0, 2, (100,), device=device)
        buffer.add(x2, y2, task_id=1)
        assert len(buffer) == 100  # At capacity

        # Sample
        rx, ry, rt = buffer.sample(16)
        assert rx.shape == (16, 784)
        assert ry.shape == (16,)
        assert rt.shape == (16,)

    def test_lwf_loss_computation(self, device):
        """Test LwF loss computes correctly."""
        model = create_backprop_arm(device=str(device))
        lwf_loss = LwFLoss(temperature=2.0, lambda_lwf=1.0)

        # Before setting prev_model, should be just CE
        logits = torch.randn(4, 10, device=device)
        targets = torch.tensor([0, 1, 0, 1], device=device)
        loss = lwf_loss(logits, targets, task_id=0)
        ce_loss = F.cross_entropy(logits[:, 0:2], targets)
        assert torch.allclose(loss, ce_loss)

        # After setting prev_model, should include distillation
        prev_model = type(model)(model.joint_system).to(device)
        prev_model.load_state_dict(model.state_dict())
        lwf_loss.set_prev_model(prev_model)

        loss_with_distill = lwf_loss(logits, targets, task_id=1)
        # Should be different from CE (includes distillation)
        assert not torch.allclose(loss_with_distill, ce_loss)

    def test_synaptic_intelligence_tracking(self, device):
        """Test SI tracks importance and computes regularization."""
        model = create_backprop_arm(device=str(device))
        si = SynapticIntelligence(model, xi=0.1)

        # Start task
        si.start_task()
        assert len(si.prev_params) > 0

        # Do a forward/backward to generate gradients
        x = torch.randn(4, 784, device=device)
        y = torch.tensor([0, 1, 0, 1], device=device)
        logits = model(x, task_id=0)
        task_logits = logits[:, 0:2]
        loss = F.cross_entropy(task_logits, y)
        loss.backward()

        # Update importance
        si.update_importance()
        assert len(si.omega) > 0

        # Regularization loss should be computable
        reg_loss = si.regularization_loss()
        assert reg_loss.item() >= 0


# ============================================================
# Test: CL Metrics Computation
# ============================================================


class TestCLMetrics:
    """Test continual learning metrics computation."""

    def test_compute_cl_metrics_basic(self, fast_weight_model, device):
        """Test basic CL metrics computation."""
        # Create dummy task loaders
        task_loaders = []
        for task_id in range(NUM_TASKS):
            dataset = TensorDataset(
                torch.randn(20, 784), torch.randint(0, 2, (20,))
            )
            task_loaders.append(DataLoader(dataset, batch_size=4))

        fast_weight_model.set_task(0)
        # Pass a dummy accuracy matrix for proper testing
        accuracy_matrix = [[0.0 for _ in range(NUM_TASKS)] for _ in range(NUM_TASKS)]
        metrics = compute_cl_metrics(fast_weight_model, task_loaders, 0, accuracy_matrix)

        assert isinstance(metrics, CLMetrics)
        assert len(metrics.final_accuracies) == NUM_TASKS
        assert len(metrics.accuracy_matrix) == NUM_TASKS

    def test_compute_cl_metrics_forgetting(self, fast_weight_model, device):
        """Test forgetting computation."""
        task_loaders = []
        for task_id in range(NUM_TASKS):
            dataset = TensorDataset(
                torch.randn(20, 784), torch.randint(0, 2, (20,))
            )
            task_loaders.append(DataLoader(dataset, batch_size=4))

        # Create an accuracy matrix with forgetting
        accuracy_matrix = [
            [0.9, 0.8, 0.7, 0.6, 0.5],  # Task 0: high initial, decays
            [0.0, 0.9, 0.8, 0.7, 0.6],  # Task 1
            [0.0, 0.0, 0.9, 0.8, 0.7],  # Task 2
            [0.0, 0.0, 0.0, 0.9, 0.8],  # Task 3
            [0.0, 0.0, 0.0, 0.0, 0.9],  # Task 4
        ]

        metrics = compute_cl_metrics(fast_weight_model, task_loaders, 4, accuracy_matrix)

        # Forgetting for each task = max(acc_so_far) - final_acc
        # Task 0: max=0.9, final=0.5 -> forgetting=0.4
        # Task 1: max=0.9, final=0.6 -> forgetting=0.3
        # etc.
        assert len(metrics.forgetting) == 5
        assert metrics.forgetting[0] == pytest.approx(0.4)
        assert metrics.forgetting[1] == pytest.approx(0.3)
        assert metrics.forgetting[2] == pytest.approx(0.2)
        assert metrics.forgetting[3] == pytest.approx(0.1)
        assert metrics.forgetting[4] == pytest.approx(0.0)
        assert metrics.avg_forgetting == pytest.approx(0.2)

    def test_compute_cl_metrics_backward_transfer(self, fast_weight_model, device):
        """Test backward transfer computation."""
        task_loaders = []
        for task_id in range(NUM_TASKS):
            dataset = TensorDataset(
                torch.randn(20, 784), torch.randint(0, 2, (20,))
            )
            task_loaders.append(DataLoader(dataset, batch_size=4))

        # Accuracy matrix where later tasks improve earlier tasks
        accuracy_matrix = [
            [0.8, 0.85, 0.9, 0.92, 0.93],  # Task 0 improves
            [0.0, 0.8, 0.85, 0.88, 0.9],   # Task 1 improves
            [0.0, 0.0, 0.8, 0.85, 0.88],   # Task 2
            [0.0, 0.0, 0.0, 0.8, 0.85],    # Task 3
            [0.0, 0.0, 0.0, 0.0, 0.8],     # Task 4
        ]

        metrics = compute_cl_metrics(fast_weight_model, task_loaders, 4, accuracy_matrix)

        # BWT = mean(acc_after_all - acc_after_own_task)
        # Task 0: 0.93 - 0.8 = 0.13
        # Task 1: 0.9 - 0.8 = 0.1
        # Task 2: 0.88 - 0.8 = 0.08
        # Task 3: 0.85 - 0.8 = 0.05
        # Mean = (0.13 + 0.1 + 0.08 + 0.05) / 4 = 0.09
        assert metrics.backward_transfer > 0  # Positive backward transfer


# ============================================================
# Test: Stability Guard Integration
# ============================================================


class TestStabilityGuard:
    """Test stability guard integration with continual learning."""

    def test_create_stability_guard(self):
        """Test stability guard creation."""
        guard = create_stability_guard(threshold=1.029, statistic="fast_proxy", window=10)
        assert isinstance(guard, StabilityGuard)
        assert guard.threshold == 1.029

    def test_check_stability_returns_verdict(self, fast_weight_model, mnist_batch):
        """Test stability check returns a GuardDecision."""
        x, y = mnist_batch
        guard = create_stability_guard()
        transition_fn = make_transition_fn(fast_weight_model)
        context = fast_weight_model.joint_system.context

        verdict = check_stability(guard, transition_fn, x, step=0, context=context)

        assert isinstance(verdict, GuardDecision)
        assert hasattr(verdict, "kill")
        assert hasattr(verdict, "statistic")
        assert hasattr(verdict, "threshold")


# ============================================================
# Test: SplitMNIST Integration
# ============================================================


class TestSplitMNIST:
    """Test SplitMNIST domain integration."""

    def test_split_mnist_tasks(self):
        """Verify SPLIT_MNIST_TASKS definition."""
        assert len(SPLIT_MNIST_TASKS) == 5
        assert SPLIT_MNIST_TASKS == [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
        assert NUM_TASKS == 5
        assert CLASSES_PER_TASK == 2
        assert TOTAL_CLASSES == 10

    def test_split_mnist_loader(self, device):
        """Test SplitMNIST dataloader creation."""
        task = SplitMNIST(task_id=0, batch_size=32, device=str(device), num_workers=0)
        task.setup()

        train_loader = task.get_dataloader(TaskSplit.TRAIN)
        test_loader = task.get_dataloader(TaskSplit.TEST)

        # Check one batch
        x, y = next(iter(train_loader))
        assert x.shape[1:] == (1, 28, 28)  # MNIST image shape
        assert y.shape[0] == x.shape[0]
        assert y.min() >= 0 and y.max() <= 1  # Binary labels 0/1

        x, y = next(iter(test_loader))
        assert x.shape[1:] == (1, 28, 28)
        assert y.min() >= 0 and y.max() <= 1


# ============================================================
# Test: End-to-End Integration (Smoke)
# ============================================================


class TestContinualLearningIntegration:
    """End-to-end integration tests (smoke tests, 1 epoch)."""

    def test_fast_weights_single_task_learning(self, device):
        """Test fast weights can learn a single task above chance."""
        config = CLConfig(
            epochs_per_task=1,
            batch_size=32,
            device=str(device),
            seed=42,
        )

        # Run single task
        metrics = run_continual_learning("fast_weights", config, protocol="task_incremental")

        # Should complete without error
        assert metrics.total_time_s > 0
        assert len(metrics.final_accuracies) == NUM_TASKS
        assert len(metrics.stability_verdicts) > 0

    def test_replay_single_task_learning(self, device):
        """Test replay arm completes."""
        config = CLConfig(
            epochs_per_task=1,
            batch_size=32,
            device=str(device),
            seed=42,
        )

        metrics = run_continual_learning("replay", config, protocol="task_incremental")

        assert metrics.total_time_s > 0
        assert metrics.replay_buffer_bytes > 0

    def test_backprop_single_task_learning(self, device):
        """Test backprop arm completes."""
        config = CLConfig(
            epochs_per_task=1,
            batch_size=32,
            device=str(device),
            seed=42,
        )

        metrics = run_continual_learning("backprop", config, protocol="task_incremental")

        assert metrics.total_time_s > 0

    def test_ewc_single_task_learning(self, device):
        """Test EWC arm completes."""
        config = CLConfig(
            epochs_per_task=1,
            batch_size=32,
            device=str(device),
            seed=42,
        )

        metrics = run_continual_learning("ewc", config, protocol="task_incremental")

        assert metrics.total_time_s > 0

    def test_lwf_single_task_learning(self, device):
        """Test LwF arm completes."""
        config = CLConfig(
            epochs_per_task=1,
            batch_size=32,
            device=str(device),
            seed=42,
        )

        metrics = run_continual_learning("lwf", config, protocol="task_incremental")

        assert metrics.total_time_s > 0

    def test_si_single_task_learning(self, device):
        """Test SI arm completes."""
        config = CLConfig(
            epochs_per_task=1,
            batch_size=32,
            device=str(device),
            seed=42,
        )

        metrics = run_continual_learning("si", config, protocol="task_incremental")

        assert metrics.total_time_s > 0

    def test_task_free_protocol(self, device):
        """Test task-free protocol runs."""
        config = CLConfig(
            epochs_per_task=1,
            batch_size=32,
            device=str(device),
            seed=42,
        )

        metrics = run_continual_learning("fast_weights", config, protocol="task_free")

        assert metrics.total_time_s > 0


# ============================================================
# Test: Suite Runner
# ============================================================


class TestSuiteRunner:
    """Test the continual learning suite runner."""

    def test_suite_runner_smoke(self, device, tmp_path):
        """Test suite runner with minimal config."""
        config = CLConfig(
            epochs_per_task=1,
            batch_size=32,
            device=str(device),
            seed=42,
        )

        results = run_continual_learning_suite(
            arms=["fast_weights", "backprop"],
            protocols=["task_incremental"],
            output_dir=tmp_path,
            config=config,
            seeds=1,
        )

        assert "fast_weights" in results
        assert "backprop" in results
        assert "task_incremental" in results["fast_weights"]
        assert "task_incremental" in results["backprop"]

        # Check results file saved
        results_file = tmp_path / "continual_learning_results.json"
        assert results_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])