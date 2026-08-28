"""Replay buffer for continual learning."""

from __future__ import annotations

import torch
from torch import Tensor


class ReplayBuffer:
    """Fixed-capacity replay buffer for continual learning.

    Stores (input, target, task_id) tuples. When full, evicts uniformly
    to maintain balanced representation across seen tasks.
    """

    def __init__(
        self, capacity: int, input_shape: tuple[int, ...], device: torch.device
    ):
        self.capacity = capacity
        self.input_shape = input_shape
        self.device = device
        self.buffer: list[tuple[Tensor, Tensor, int]] = []
        self.task_counts: dict[int, int] = {}

    def add(self, x: Tensor, y: Tensor, task_id: int) -> None:
        """Add a batch to the buffer."""
        batch_size = x.shape[0]
        x_cpu = x.detach().cpu()
        y_cpu = y.detach().cpu()

        for i in range(batch_size):
            if len(self.buffer) >= self.capacity:
                # Evict from the task with most samples
                if self.task_counts:
                    evict_task = max(
                        self.task_counts.keys(), key=lambda k: self.task_counts[k]
                    )
                    # Find and remove one sample from that task
                    for idx, (_, _, t) in enumerate(self.buffer):
                        if t == evict_task:
                            self.buffer.pop(idx)
                            self.task_counts[evict_task] -= 1
                            if self.task_counts[evict_task] == 0:
                                del self.task_counts[evict_task]
                            break

            self.buffer.append((x_cpu[i], y_cpu[i], task_id))
            self.task_counts[task_id] = self.task_counts.get(task_id, 0) + 1

    def sample(self, batch_size: int) -> tuple[Tensor, Tensor, Tensor]:
        """Sample a batch from the buffer."""
        if not self.buffer:
            raise ValueError("Replay buffer is empty")
        indices = torch.randperm(len(self.buffer))[:batch_size]
        samples = [self.buffer[i] for i in indices]
        x = torch.stack([s[0] for s in samples]).to(self.device)
        y = torch.stack([s[1] for s in samples]).to(self.device)
        t = torch.tensor([s[2] for s in samples], device=self.device)
        return x, y, t

    def __len__(self) -> int:
        return len(self.buffer)

    def memory_bytes(self) -> int:
        """Estimate memory footprint in bytes."""
        if not self.buffer:
            return 0
        sample = self.buffer[0]
        per_sample = (
            sample[0].numel() * sample[0].element_size()
            + sample[1].numel() * sample[1].element_size()
        )
        return per_sample * len(self.buffer)


__all__ = ["ReplayBuffer"]
