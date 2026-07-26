"""
Curriculum Learning Support

Defines curriculum schedules for progressive training (easy to hard).
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections.abc import Callable


class Curriculum(ABC):
    """Base class for curriculum schedules."""

    @abstractmethod
    def get_difficulty(self, epoch: int, total_epochs: int) -> float:
        """
        Return difficulty level in [0.0, 1.0] for the given epoch.
        0.0 = easiest, 1.0 = hardest.
        """
        pass

    @abstractmethod
    def description(self) -> str:
        """Human-readable description of this curriculum."""
        pass


class FixedCurriculum(Curriculum):
    """Always use the same difficulty level."""

    def __init__(self, difficulty: float = 1.0):
        self._difficulty = difficulty

    def get_difficulty(self, epoch: int, total_epochs: int) -> float:
        return self._difficulty

    def description(self) -> str:
        return f"Fixed(difficulty={self._difficulty})"


class ProgressiveCurriculum(Curriculum):
    """Linearly increase difficulty from start to end."""

    def __init__(self, start: float = 0.0, end: float = 1.0):
        self.start = start
        self.end = end

    def get_difficulty(self, epoch: int, total_epochs: int) -> float:
        if total_epochs <= 1:
            return self.end
        progress = epoch / (total_epochs - 1)
        return self.start + (self.end - self.start) * progress

    def description(self) -> str:
        return f"Progressive({self.start}->{self.end})"


class AntiCurriculum(Curriculum):
    """Linearly decrease difficulty (start hard, get easier)."""

    def __init__(self, start: float = 1.0, end: float = 0.0):
        self.start = start
        self.end = end

    def get_difficulty(self, epoch: int, total_epochs: int) -> float:
        if total_epochs <= 1:
            return self.end
        progress = epoch / (total_epochs - 1)
        return self.start - (self.start - self.end) * progress

    def description(self) -> str:
        return f"AntiCurriculum({self.start}->{self.end})"


class CurriculumScheduler:
    """
    Scheduler that wraps a Curriculum and applies it to training.

    Can be used with:
    - Data filtering (use easy examples first)
    - Task difficulty (easy tasks first)
    - Model capacity (small model first, then grow)
    """

    def __init__(
        self,
        curriculum: Curriculum,
        apply_fn: Callable[[float], None] | None = None,
    ):
        self.curriculum = curriculum
        self.apply_fn = apply_fn
        self.current_difficulty = 0.0

    def step(self, epoch: int, total_epochs: int) -> float:
        """Advance scheduler and return current difficulty."""
        self.current_difficulty = self.curriculum.get_difficulty(epoch, total_epochs)
        if self.apply_fn:
            self.apply_fn(self.current_difficulty)
        return self.current_difficulty

    def description(self) -> str:
        return self.curriculum.description()


# Pre-built curriculum schedules
CURRICULA = {
    "default": FixedCurriculum(1.0),
    "progressive": ProgressiveCurriculum(0.0, 1.0),
    "anti": AntiCurriculum(1.0, 0.0),
    "easy_first": ProgressiveCurriculum(0.0, 0.5),
    "hard_first": ProgressiveCurriculum(0.5, 1.0),
}
