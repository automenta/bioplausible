"""
Multi-Objective Analysis for EqProp Evolution

Implements Pareto-optimal solution detection and efficiency metrics.

Key objectives:
1. Maximize accuracy
2. Minimize training time (wall clock)
3. Minimize parameter count
4. Maximize Lipschitz stability (L ≤ 1)

Efficiency metrics:
- Accuracy per parameter
- Accuracy per second of training
- Performance density (accuracy / (params * time))
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict


@dataclass
class MultiObjectiveMetrics:
    """Multi-objective performance metrics."""
    accuracy: float
    train_time_sec: float
    param_count: int
    lipschitz: float
    
    # Efficiency metrics
    accuracy_per_param: float = 0.0
    accuracy_per_second: float = 0.0
    performance_density: float = 0.0  # accuracy / (params * time)
    
    # Metadata
    model_type: str = "unknown"
    task: str = "unknown"
    config: Dict = None
    
    def __post_init__(self):
        """Compute derived efficiency metrics."""
        if self.param_count > 0:
            self.accuracy_per_param = self.accuracy / self.param_count * 1e6  # per million params
        
        if self.train_time_sec > 0:
            self.accuracy_per_second = self.accuracy / self.train_time_sec
        
        if self.param_count > 0 and self.train_time_sec > 0:
            self.performance_density = self.accuracy / (self.param_count * self.train_time_sec) * 1e6
    
    def dominates(self, other: 'MultiObjectiveMetrics') -> bool:
        """Check if this solution Pareto-dominates another.
        
        Returns True if this is better in at least one objective
        and not worse in any objective.
        
        Objectives:
        - Maximize accuracy
        - Minimize time
        - Minimize params
        """
        better_in_any = False
        worse_in_any = False
        
        # Accuracy (maximize)
        if self.accuracy > other.accuracy:
            better_in_any = True
        elif self.accuracy < other.accuracy:
            worse_in_any = True
        
        # Time (minimize)
        if self.train_time_sec < other.train_time_sec:
            better_in_any = True
        elif self.train_time_sec > other.train_time_sec:
            worse_in_any = True
        
        # Parameters (minimize)
        if self.param_count < other.param_count:
            better_in_any = True
        elif self.param_count > other.param_count:
            worse_in_any = True
        
        return better_in_any and not worse_in_any


class ParetoFrontAnalyzer:
    """Analyze and identify Pareto-optimal solutions."""
    
    def __init__(self):
        self.all_solutions: List[MultiObjectiveMetrics] = []
        self.pareto_fronts: Dict[str, List[MultiObjectiveMetrics]] = defaultdict(list)
    
    def add_solution(self, metrics: MultiObjectiveMetrics):
        """Add a solution to analysis."""
        self.all_solutions.append(metrics)
        
        # Compute Pareto front for this task
        task = metrics.task
        self._update_pareto_front(task)
    
    def _update_pareto_front(self, task: str):
        """Update Pareto front for a task."""
        task_solutions = [s for s in self.all_solutions if s.task == task]
        
        pareto_front = []
        for solution in task_solutions:
            dominated = False
            for other in task_solutions:
                if other.dominates(solution):
                    dominated = True
                    break
            
            if not dominated:
                pareto_front.append(solution)
        
        self.pareto_fronts[task] = pareto_front
    
    def get_pareto_front(self, task: str) -> List[MultiObjectiveMetrics]:
        """Get Pareto-optimal solutions for a task."""
        return self.pareto_fronts.get(task, [])
    
    def get_best_by_objective(
        self,
        task: str,
        objective: str = 'accuracy'
    ) -> Optional[MultiObjectiveMetrics]:
        """Get best solution for a specific objective."""
        solutions = [s for s in self.all_solutions if s.task == task]
        if not solutions:
            return None
        
        if objective == 'accuracy':
            return max(solutions, key=lambda s: s.accuracy)
        elif objective == 'speed':
            return min(solutions, key=lambda s: s.train_time_sec)
        elif objective == 'params':
            return min(solutions, key=lambda s: s.param_count)
        elif objective == 'efficiency':
            return max(solutions, key=lambda s: s.accuracy_per_param)
        elif objective == 'performance_density':
            return max(solutions, key=lambda s: s.performance_density)
        else:
            return None
    
    def analyze_tradeoffs(self, task: str) -> Dict[str, any]:
        """Analyze trade-offs for a task."""
        # Filter to only successful solutions
        solutions = [s for s in self.all_solutions if s.task == task and s.accuracy > 0]
        if not solutions:
            return {}
        
        pareto = [p for p in self.get_pareto_front(task) if p.accuracy > 0]
        
        # Compute statistics
        accuracies = [s.accuracy for s in solutions]
        times = [s.train_time_sec for s in solutions]
        params = [s.param_count for s in solutions]
        
        return {
            'n_solutions': len(solutions),
            'n_pareto_optimal': len(pareto),
            'pareto_percentage': len(pareto) / len(solutions) * 100 if solutions else 0,
            
            # Best by objective
            'best_accuracy': max(accuracies),
            'fastest_time': min(times),
            'smallest_params': min(params),
            
            # Efficiency leaders
            'best_accuracy_per_param': max(s.accuracy_per_param for s in solutions),
            'best_performance_density': max(s.performance_density for s in solutions),
            
            # Correlations
            'accuracy_param_correlation': np.corrcoef(accuracies, params)[0, 1] if len(solutions) > 1 else 0,
            'accuracy_time_correlation': np.corrcoef(accuracies, times)[0, 1] if len(solutions) > 1 else 0,
        }
    
    def generate_report(self) -> str:
        """Generate multi-objective analysis report."""
        lines = [
            "# Multi-Objective Analysis",
            "",
            "## Pareto-Optimal Solutions by Task",
            "",
        ]
        
        # Filter to only tasks with successful solutions
        tasks_with_success = []
        for task in sorted(self.pareto_fronts.keys()):
            if any(s.accuracy > 0 for s in self.all_solutions if s.task == task):
                tasks_with_success.append(task)
        
        if not tasks_with_success:
            lines.append("_No successful solutions yet._")
            return "\n".join(lines)
        
        for task in tasks_with_success:
            tradeoffs = self.analyze_tradeoffs(task)
            if not tradeoffs:  # Skip if no successful solutions
                continue
            
            pareto = [p for p in self.get_pareto_front(task) if p.accuracy > 0]
            
            lines.extend([
                f"### {task.upper()}",
                "",
                f"**Pareto-optimal solutions**: {len(pareto)} / {tradeoffs.get('n_solutions', 0)} "
                f"({tradeoffs.get('pareto_percentage', 0):.1f}%)",
                "",
                "| Solution | Accuracy | Params | Time (s) | Acc/Param | Acc/Sec | Model |",
                "|----------|----------|--------|----------|-----------|---------|-------|",
            ])
            
            # Sort Pareto front by accuracy
            pareto_sorted = sorted(pareto, key=lambda s: s.accuracy, reverse=True)
            
            for sol in pareto_sorted[:10]:  # Top 10
                lines.append(
                    f"| {sol.model_type[:12]} | {sol.accuracy:.4f} | "
                    f"{sol.param_count//1000}K | {sol.train_time_sec:.1f} | "
                    f"{sol.accuracy_per_param:.2f} | {sol.accuracy_per_second:.3f} | "
                    f"{sol.model_type} |"
                )
            
            # Trade-off analysis
            lines.extend([
                "",
                "**Best by Objective:**",
                f"- **Accuracy leader**: {tradeoffs.get('best_accuracy', 0):.4f}",
                f"- **Smallest model**: {tradeoffs.get('smallest_params', 0)//1000}K params",
                f"- **Best efficiency**: {tradeoffs.get('best_accuracy_per_param', 0):.2f} acc/M-param",
                f"- **Fastest training**: {tradeoffs.get('fastest_time', 0):.1f}s",
                "",
                "**Trade-off Correlations:**",
                f"- Accuracy vs Params: {tradeoffs.get('accuracy_param_correlation', 0):.3f}",
                f"- Accuracy vs Time: {tradeoffs.get('accuracy_time_correlation', 0):.3f}",
                "",
            ])
        
        return "\n".join(lines)
    
    def get_efficiency_leaders(self, task: str, top_k: int = 5) -> Dict[str, List]:
        """Get top performers for each efficiency metric."""
        solutions = [s for s in self.all_solutions if s.task == task]
        
        return {
            'accuracy_per_param': sorted(
                solutions,
                key=lambda s: s.accuracy_per_param,
                reverse=True
            )[:top_k],
            
            'accuracy_per_second': sorted(
                solutions,
                key=lambda s: s.accuracy_per_second,
                reverse=True
            )[:top_k],
            
            'performance_density': sorted(
                solutions,
                key=lambda s: s.performance_density,
                reverse=True
            )[:top_k],
            
            'balanced': sorted(
                solutions,
                key=lambda s: s.accuracy * s.accuracy_per_param * s.accuracy_per_second,
                reverse=True
            )[:top_k],
        }


def identify_algorithm_strategies(solutions: List[MultiObjectiveMetrics]) -> Dict[str, List]:
    """
    Identify patterns and strategies across solutions.
    
    Groups solutions by characteristics to understand what works.
    """
    strategies = defaultdict(list)
    
    for sol in solutions:
        config = sol.config or {}
        
        # Size category
        if sol.param_count < 100_000:
            strategies['tiny_models'].append(sol)
        elif sol.param_count < 1_000_000:
            strategies['small_models'].append(sol)
        elif sol.param_count < 10_000_000:
            strategies['medium_models'].append(sol)
        else:
            strategies['large_models'].append(sol)
        
        # Speed category
        if sol.train_time_sec < 10:
            strategies['fast_training'].append(sol)
        elif sol.train_time_sec < 60:
            strategies['moderate_training'].append(sol)
        else:
            strategies['slow_training'].append(sol)
        
        # Efficiency category
        if sol.accuracy_per_param > 1.0:
            strategies['highly_efficient'].append(sol)
        
        if sol.accuracy_per_second > 0.1:
            strategies['fast_learners'].append(sol)
        
        # Model type
        strategies[f"type_{sol.model_type}"].append(sol)
    
    # Find best strategy for each metric
    best_strategies = {}
    for category, sols in strategies.items():
        if sols:
            avg_acc = np.mean([s.accuracy for s in sols])
            avg_eff = np.mean([s.accuracy_per_param for s in sols])
            best_strategies[category] = {
                'count': len(sols),
                'avg_accuracy': avg_acc,
                'avg_efficiency': avg_eff,
            }
    
    return best_strategies
