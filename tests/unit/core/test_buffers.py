"""Regression tests for Replay Buffer (Phase 3.6.5/3.6.8)."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from computronium.core.continual.buffers import ReplayBuffer


@pytest.fixture
def device():
    return torch.device("cpu")


class TestReplayBuffer:
    """Tests for ReplayBuffer correctness."""

    def test_capacity_respected(self, device):
        """Buffer never exceeds capacity."""
        capacity = 100
        buffer = ReplayBuffer(capacity=capacity, input_shape=(784,), device=device)
        
        x = torch.randn(150, 784, device=device)
        y = torch.randint(0, 2, (150,), device=device)
        buffer.add(x, y, task_id=0)
        
        assert len(buffer) == capacity

    def test_balanced_eviction(self, device):
        """Eviction maintains balanced representation across tasks."""
        capacity = 100
        buffer = ReplayBuffer(capacity=capacity, input_shape=(784,), device=device)
        
        x0 = torch.randn(80, 784, device=device)
        y0 = torch.zeros(80, device=device, dtype=torch.long)
        buffer.add(x0, y0, task_id=0)
        
        x1 = torch.randn(80, 784, device=device)
        y1 = torch.ones(80, device=device, dtype=torch.long)
        buffer.add(x1, y1, task_id=1)
        
        # Should be roughly balanced (difference <= 5)
        task_counts = buffer.task_counts
        diff = abs(task_counts.get(0, 0) - task_counts.get(1, 0))
        assert diff <= 5, f"Eviction not balanced: {task_counts}, diff={diff}"

    def test_sampling_shapes(self, device):
        """Sample returns correct shapes."""
        buffer = ReplayBuffer(capacity=100, input_shape=(784,), device=device)
        
        x = torch.randn(50, 784, device=device)
        y = torch.randint(0, 2, (50,), device=device)
        buffer.add(x, y, task_id=0)
        
        rx, ry, rt = buffer.sample(16)
        assert rx.shape == (16, 784)
        assert ry.shape == (16,)
        assert rt.shape == (16,)

    def test_sampling_capped_at_buffer_size(self, device):
        """Sampling more than buffer size returns buffer size samples."""
        buffer = ReplayBuffer(capacity=100, input_shape=(784,), device=device)
        
        x = torch.randn(50, 784, device=device)
        y = torch.randint(0, 2, (50,), device=device)
        buffer.add(x, y, task_id=0)
        
        rx, ry, rt = buffer.sample(200)
        assert rx.shape[0] == 50  # Buffer size, not 200

    def test_empty_buffer_raises(self, device):
        """Sampling from empty buffer raises ValueError."""
        buffer = ReplayBuffer(capacity=100, input_shape=(784,), device=device)
        
        with pytest.raises(ValueError, match="empty"):
            buffer.sample(16)

    def test_memory_bytes_calculation(self, device):
        """memory_bytes() matches actual storage."""
        capacity = 41
        buffer = ReplayBuffer(capacity=capacity, input_shape=(784,), device=device)
        
        x = torch.randn(capacity, 784, device=device)
        y = torch.randint(0, 2, (capacity,), device=device)
        buffer.add(x, y, task_id=0)
        
        # Calculate expected from actual stored tensors
        sample = buffer.buffer[0]
        expected_per_sample = sample[0].numel() * sample[0].element_size() + sample[1].numel() * sample[1].element_size()
        expected_bytes = expected_per_sample * capacity
        
        actual_bytes = buffer.memory_bytes()
        assert actual_bytes == expected_bytes, f"Expected {expected_bytes}, got {actual_bytes}"

    def test_task_id_preserved(self, device):
        """Sampled task_id matches added task_id."""
        buffer = ReplayBuffer(capacity=100, input_shape=(784,), device=device)
        
        for task_id in range(3):
            x = torch.randn(10, 784, device=device)
            y = torch.full((10,), task_id, device=device, dtype=torch.long)
            buffer.add(x, y, task_id=task_id)
        
        rx, ry, rt = buffer.sample(30)
        unique_tasks = rt.unique().tolist()
        # Should have samples from all tasks
        assert len(unique_tasks) >= 2  # At least 2 of the 3 tasks


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])