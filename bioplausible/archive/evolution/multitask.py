"""
Multi-Task Evaluation Strategy

Ensures fair evaluation across multiple tasks (MNIST, Fashion-MNIST, CIFAR-10, Shakespeare)
to discover breakthroughs in ANY domain, not just architectures that dominate all tasks.

Key principles:
1. Each architecture is evaluated on ONE task (sampled fairly)
2. Task assignment is round-robin to ensure even coverage
3. Time budget is fairly allocated (harder tasks don't disadvantage)
4. Breakthrough detection is per-task
5. Final rankings consider task difficulty

This discovers specialists, not just generalists.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import defaultdict
import numpy as np


@dataclass
class TaskSpec:
    """Specification for a task."""
    name: str
    input_dim: int
    output_dim: int
    difficulty: float  # Relative difficulty (1.0 = baseline)
    type: str  # classification, language_modeling
    expected_accuracy: float  # Expected baseline accuracy
    

# Available tasks with metadata
TASK_POOL = {
    'mnist': TaskSpec(
        name='mnist',
        input_dim=784,
        output_dim=10,
        difficulty=1.0,
        type='classification',
        expected_accuracy=0.95,
    ),
    'fashion': TaskSpec(
        name='fashion',
        input_dim=784,
        output_dim=10,
        difficulty=1.2,
        type='classification',
        expected_accuracy=0.87,
    ),
    'cifar10': TaskSpec(
        name='cifar10',
        input_dim=3072,
        output_dim=10,
        difficulty=2.0,
        type='classification',
        expected_accuracy=0.65,
    ),
    'shakespeare': TaskSpec(
        name='shakespeare',
        input_dim=65,
        output_dim=65,
        difficulty=1.5,
        type='language_modeling',
        expected_accuracy=0.40,  # character-level
    ),
    
    # RL Domains
    'cartpole': TaskSpec(
        name='cartpole',
        input_dim=4,  # cart position, velocity, pole angle, angular velocity
        output_dim=2,  # left or right
        difficulty=1.0,
        type='rl_control',
        expected_accuracy=0.95,  # 195/200 reward threshold
    ),
    'acrobot': TaskSpec(
        name='acrobot',
        input_dim=6,  # 2 angles + 2 angular velocities (cos, sin for each)
        output_dim=3,  # torque: -1, 0, +1
        difficulty=1.3,
        type='rl_control',
        expected_accuracy=0.50,  # Harder, -100 reward threshold
    ),
    'mountaincar': TaskSpec(
        name='mountaincar',
        input_dim=2,  # position, velocity
        output_dim=3,  # left, none, right
        difficulty=1.4,
        type='rl_control',
        expected_accuracy=0.55,  # Also harder
    ),
}


class TaskAssigner:
    """
    Assigns tasks to individuals fairly using round-robin.
    
    Ensures even coverage of all tasks across the population.
    """
    
    def __init__(
        self,
        tasks: List[str] = None,
        seed: int = 42,
    ):
        self.tasks = tasks or list(TASK_POOL.keys())
        self.rng = np.random.default_rng(seed)
        self._current_idx = 0
        self._assignments: Dict[int, str] = {}  # individual_id -> task
        
        # Shuffle task order for variety
        self.rng.shuffle(self.tasks)
    
    def assign_task(self, individual_id: int) -> str:
        """Assign a task to an individual."""
        # Round-robin through tasks
        task = self.tasks[self._current_idx % len(self.tasks)]
        self._current_idx += 1
        
        self._assignments[individual_id] = task
        return task
    
    def get_assignment(self, individual_id: int) -> Optional[str]:
        """Get previously assigned task."""
        return self._assignments.get(individual_id)
    
    def get_coverage_stats(self) -> Dict[str, int]:
        """Get how many individuals assigned to each task."""
        counts = defaultdict(int)
        for task in self._assignments.values():
            counts[task] += 1
        return dict(counts)
    
    def reset_for_generation(self):
        """Reset assignments for new generation (but keep round-robin position)."""
        self._assignments = {}


class TaskNormalizedFitness:
    """
    Computes task-normalized fitness scores.
    
    Accounts for task difficulty to enable fair comparison across tasks.
    """
    
    @staticmethod
    def normalize_accuracy(accuracy: float, task: str) -> float:
        """
        Normalize accuracy relative to task difficulty.
        
        Returns a score where 1.0 = meeting expected baseline,
        >1.0 = exceeding baseline (breakthrough!)
        """
        spec = TASK_POOL.get(task)
        if spec is None:
            return accuracy
        
        # Normalize by expected accuracy
        # This way, 95% on MNIST and 65% on CIFAR-10 are both "meeting baseline"
        normalized = accuracy / spec.expected_accuracy
        
        # Apply difficulty scaling
        # Harder tasks get bonus points
        difficulty_bonus = 1.0 + (spec.difficulty - 1.0) * 0.2
        
        return normalized * difficulty_bonus
    
    @staticmethod
    def compute_composite(
        accuracy: float,
        task: str,
        lipschitz: float,
        speed: float,
        memory_mb: float,
    ) -> float:
        """
        Compute task-aware composite fitness score.
        """
        # Task-normalized accuracy (most important)
        norm_acc = TaskNormalizedFitness.normalize_accuracy(accuracy, task)
        
        # Stability bonus (L ≤ 1 is critical)
        stability_score = 1.0 if lipschitz <= 1.05 else max(0.0, 2.0 - lipschitz)
        
        # Efficiency (speed vs memory tradeoff)
        efficiency = speed / max(memory_mb, 100)
        
        # Composite
        composite = (
            norm_acc * 2.0 +
            stability_score * 1.0 +
            min(efficiency / 10, 1.0) * 0.5
        )
        
        return composite


class MultiTaskBreakthroughDetector:
    """
    Detects breakthroughs per-task.
    
    Allows finding specialists that excel at specific tasks,
    not just generalists that are mediocre at everything.
    """
    
    def __init__(self):
        self.best_per_task: Dict[str, float] = defaultdict(float)
        self.breakthroughs: List[Dict] = []
    
    def check_breakthrough(
        self,
        task: str,
        accuracy: float,
        config: Dict,
    ) -> bool:
        """Check if this is a breakthrough for this task."""
        spec = TASK_POOL.get(task)
        if spec is None:
            return False
        
        # Breakthrough = exceeds expected baseline by 5%
        threshold = spec.expected_accuracy * 1.05
        
        if accuracy >= threshold:
            # Check if this is the best for this task
            if accuracy > self.best_per_task[task]:
                self.best_per_task[task] = accuracy
                
                self.breakthroughs.append({
                    'task': task,
                    'accuracy': accuracy,
                    'config': config,
                    'improvement_pct': (accuracy - spec.expected_accuracy) / spec.expected_accuracy * 100,
                })
                return True
        
        return False
    
    def get_best_per_task(self) -> Dict[str, Dict]:
        """Get best result per task."""
        results = {}
        for bt in self.breakthroughs:
            task = bt['task']
            if task not in results or bt['accuracy'] > results[task]['accuracy']:
                results[task] = bt
        return results
    
    def summarize(self) -> str:
        """Generate summary of breakthroughs."""
        lines = ["## Multi-Task Breakthroughs", ""]
        
        best_per_task = self.get_best_per_task()
        
        if not best_per_task:
            lines.append("No breakthroughs yet.")
        else:
            lines.append("| Task | Best Accuracy | Improvement | Expected |")
            lines.append("|------|---------------|-------------|----------|")
            
            for task, bt in best_per_task.items():
                spec = TASK_POOL[task]
                lines.append(
                    f"| {task} | {bt['accuracy']:.4f} | "
                    f"+{bt['improvement_pct']:.1f}% | {spec.expected_accuracy:.4f} |"
                )
        
        return "\n".join(lines)


def select_tasks_for_portfolio(
    n_individuals: int,
    tasks: List[str] = None,
) -> List[str]:
    """
    Select tasks for a portfolio of individuals.
    
    Ensures even distribution across all tasks.
    
    Args:
        n_individuals: Number of individuals to assign
        tasks: Available tasks (default: all)
        
    Returns:
        List of task names in assignment order
    """
    tasks = tasks or list(TASK_POOL.keys())
    
    # Calculate how many of each task
    per_task = n_individuals // len(tasks)
    remainder = n_individuals % len(tasks)
    
    assignment = []
    for i, task in enumerate(tasks):
        count = per_task + (1 if i < remainder else 0)
        assignment.extend([task] * count)
    
    # Shuffle to avoid bias
    np.random.shuffle(assignment)
    
    return assignment


# Example usage
if __name__ == '__main__':
    # Demo task assignment
    assigner = TaskAssigner()
    
    print("Assigning tasks to 20 individuals:")
    for i in range(20):
        task = assigner.assign_task(i)
        print(f"  Individual {i}: {task}")
    
    print(f"\nCoverage: {assigner.get_coverage_stats()}")
    
    # Demo breakthrough detection
    detector = MultiTaskBreakthroughDetector()
    
    # Simulate some results
    detector.check_breakthrough('mnist', 0.97, {'model': 'test'})
    detector.check_breakthrough('cifar10', 0.72, {'model': 'test2'})
    
    print(f"\n{detector.summarize()}")
