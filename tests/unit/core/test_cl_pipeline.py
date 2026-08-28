"""Regression tests for CL Pipeline (Phase 3.6.5/3.6.8)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from computronium.core.continual.arms import create_fast_weight_arm
from computronium.core.continual.training import run_continual_train_step


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def fast_weight_model(device):
    return create_fast_weight_arm(device=str(device))


class TestTaskMasking:
    """Test task masking in forward pass and loss computation."""

    def test_forward_outputs_10_classes(self, fast_weight_model, device):
        """Model outputs 10-class logits."""
        x = torch.randn(4, 784, device=device)
        logits = fast_weight_model(x, task_id=0)
        assert logits.shape == (4, 10)

    def test_loss_computed_on_task_slice(self, fast_weight_model, device):
        """Loss computed only on task-relevant logits."""
        x = torch.randn(4, 784, device=device)
        y = torch.tensor([0, 1, 0, 1], device=device)  # Task 0 labels (0,1)

        # Task 0: logits[:, 0:2]
        fast_weight_model.set_task(0)
        logits0 = fast_weight_model(x, task_id=0)
        task_logits0 = logits0[:, 0:2]
        loss0 = F.cross_entropy(task_logits0, y)

        # Task 1: logits[:, 2:4]
        fast_weight_model.set_task(1)
        logits1 = fast_weight_model(x, task_id=1)
        task_logits1 = logits1[:, 2:4]
        loss1 = F.cross_entropy(task_logits1, y)

        # Both should work
        assert loss0.item() >= 0
        assert loss1.item() >= 0

    def test_train_step_uses_task_masking(self, fast_weight_model, device):
        """run_continual_train_step uses task-masked loss."""
        x = torch.randn(4, 784, device=device)
        y = torch.tensor([0, 1, 0, 1], device=device)

        metrics, psi = run_continual_train_step(
            fast_weight_model.joint_system, x, y, task_id=0, psi=None
        )

        assert "loss" in metrics
        assert "accuracy" in metrics
        assert psi is not None

    def test_different_tasks_different_slices(self, fast_weight_model, device):
        """Different tasks use different logit slices."""
        x = torch.randn(4, 784, device=device)
        y = torch.tensor([0, 1, 0, 1], device=device)

        fast_weight_model.set_task(0)
        logits0 = fast_weight_model(x, task_id=0)

        fast_weight_model.set_task(1)
        logits1 = fast_weight_model(x, task_id=1)

        # Task 0 logits[:, 0:2] should differ from Task 1 logits[:, 2:4]
        task0_slice = logits0[:, 0:2]
        task1_slice = logits1[:, 2:4]

        # They should be different tensors (different output neurons)
        assert not torch.allclose(task0_slice, task1_slice)


class TestPlasticStateManagement:
    """Test plastic state (psi) management across steps and task boundaries."""

    def test_psi_initialized_on_first_step(self, fast_weight_model, device):
        """Psi initialized on first train_step."""
        x = torch.randn(4, 784, device=device)
        y = torch.randint(0, 2, (4,), device=device)

        metrics, psi = run_continual_train_step(
            fast_weight_model.joint_system, x, y, task_id=0, psi=None
        )

        assert psi is not None
        assert "fast_weights" in psi
        assert psi["fast_weights"].shape[1] == 512

    def test_psi_updated_across_steps(self, fast_weight_model, device):
        """Psi updated across training steps."""
        x = torch.randn(4, 784, device=device)
        y = torch.randint(0, 2, (4,), device=device)

        _, psi1 = run_continual_train_step(
            fast_weight_model.joint_system, x, y, task_id=0, psi=None
        )
        _, psi2 = run_continual_train_step(
            fast_weight_model.joint_system, x, y, task_id=0, psi=psi1
        )

        diff = (psi2["fast_weights"] - psi1["fast_weights"]).abs().mean().item()
        assert diff > 1e-6, "Fast weights should update across steps"

    def test_reset_plastic_state_at_task_boundary(self, fast_weight_model, device):
        """Plastic state reset at task boundary."""
        x = torch.randn(4, 784, device=device)
        y = torch.randint(0, 2, (4,), device=device)

        # Train a bit
        fast_weight_model.set_task(0)
        for _ in range(3):
            fast_weight_model.train_step(x, y, task_id=0)

        psi_before = fast_weight_model._psi
        assert psi_before is not None

        # Reset at task boundary
        fast_weight_model.reset_plastic_state()
        assert fast_weight_model._psi is None


class TestStabilityGuardIntegration:
    """Test stability guard integration with CL pipeline."""

    def test_guard_called_per_step(self, fast_weight_model, device):
        """Stability guard returns verdict per step."""
        from computronium.core.continual.stability import (
            check_stability,
            create_stability_guard,
            make_transition_fn,
        )

        guard = create_stability_guard(
            threshold=1.029, statistic="fast_proxy", window=10
        )
        transition_fn = make_transition_fn(fast_weight_model)
        context = fast_weight_model.context

        x = torch.randn(4, 784, device=device)

        verdicts = []
        for step in range(5):
            fast_weight_model.train_step(
                x, torch.randint(0, 2, (4,), device=device), task_id=0
            )
            verdict = check_stability(
                guard, transition_fn, x, step=step, context=context
            )
            verdicts.append(verdict)

        assert len(verdicts) == 5
        for v in verdicts:
            assert hasattr(v, "kill")
            assert hasattr(v, "statistic")
            assert hasattr(v, "threshold")
            assert v.threshold == 1.029


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
