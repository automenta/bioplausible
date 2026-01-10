"""
Breakthrough Detection for EqProp+SN Evolution

Automatically identifies variants that exceed known baselines:
- Detects new state-of-the-art configurations
- Ranks variants by breakthrough potential
- Generates breakthrough reports
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime
from pathlib import Path

from .fitness import FitnessScore


@dataclass 
class BreakthroughThresholds:
    """Thresholds that define a breakthrough."""
    accuracy: float = 0.0
    perplexity: float = float('inf')
    memory_mb: float = float('inf')
    lipschitz: float = 1.1
    speed: float = 0.0


# Baseline thresholds for each task
BASELINES: Dict[str, BreakthroughThresholds] = {
    'mnist': BreakthroughThresholds(
        accuracy=0.98,
        memory_mb=500,
        lipschitz=1.05,
        speed=50,
    ),
    'fashion': BreakthroughThresholds(
        accuracy=0.90,
        memory_mb=500,
        lipschitz=1.05,
        speed=50,
    ),
    'cifar10': BreakthroughThresholds(
        accuracy=0.75,
        memory_mb=2000,
        lipschitz=1.05,
        speed=20,
    ),
    'shakespeare': BreakthroughThresholds(
        perplexity=2.0,
        memory_mb=1000,
        lipschitz=1.05,
    ),
}


@dataclass
class BreakthroughReport:
    """Report for a breakthrough variant."""
    config: Dict[str, Any]
    fitness: FitnessScore
    task: str
    breakthrough_dims: List[str]  # Which dimensions exceeded baseline
    improvement_pct: Dict[str, float]  # Percent improvement over baseline
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'config': self.config,
            'fitness': {
                'accuracy': self.fitness.accuracy,
                'perplexity': self.fitness.perplexity,
                'speed': self.fitness.speed,
                'memory_mb': self.fitness.memory_mb,
                'lipschitz': self.fitness.lipschitz,
                'generalization': self.fitness.generalization,
            },
            'task': self.task,
            'breakthrough_dims': self.breakthrough_dims,
            'improvement_pct': self.improvement_pct,
            'timestamp': self.timestamp,
        }


class BreakthroughDetector:
    """Identify variants that exceed known baselines."""
    
    def __init__(
        self,
        custom_baselines: Optional[Dict[str, BreakthroughThresholds]] = None,
        output_dir: str = 'results/breakthroughs',
    ):
        self.baselines = {**BASELINES, **(custom_baselines or {})}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.breakthroughs: List[BreakthroughReport] = []
    
    def is_breakthrough(
        self,
        result: FitnessScore,
        task: str = 'mnist',
    ) -> bool:
        """Check if result exceeds baseline thresholds."""
        if task not in self.baselines:
            return False
        
        baseline = self.baselines[task]
        
        # Check each dimension
        if result.accuracy > baseline.accuracy:
            return True
        if result.perplexity < baseline.perplexity:
            return True
        if result.speed > baseline.speed:
            return True
        if result.lipschitz < baseline.lipschitz and result.accuracy > 0.5:
            return True
        
        return False
    
    def get_breakthrough_dims(
        self,
        result: FitnessScore,
        task: str = 'mnist',
    ) -> Tuple[List[str], Dict[str, float]]:
        """Get which dimensions exceeded baseline and by how much."""
        if task not in self.baselines:
            return [], {}
        
        baseline = self.baselines[task]
        dims = []
        improvements = {}
        
        # Accuracy (higher is better)
        if result.accuracy > baseline.accuracy:
            dims.append('accuracy')
            improvements['accuracy'] = (
                (result.accuracy - baseline.accuracy) / baseline.accuracy * 100
            )
        
        # Perplexity (lower is better)
        if baseline.perplexity < float('inf') and result.perplexity < baseline.perplexity:
            dims.append('perplexity')
            improvements['perplexity'] = (
                (baseline.perplexity - result.perplexity) / baseline.perplexity * 100
            )
        
        # Speed (higher is better)
        if result.speed > baseline.speed:
            dims.append('speed')
            improvements['speed'] = (
                (result.speed - baseline.speed) / max(baseline.speed, 0.01) * 100
            )
        
        # Memory (lower is better)
        if result.memory_mb < baseline.memory_mb:
            dims.append('memory')
            improvements['memory'] = (
                (baseline.memory_mb - result.memory_mb) / baseline.memory_mb * 100
            )
        
        # Lipschitz (lower is better, target ≤ 1.0)
        if result.lipschitz < baseline.lipschitz:
            dims.append('lipschitz')
            improvements['lipschitz'] = (
                (baseline.lipschitz - result.lipschitz) / baseline.lipschitz * 100
            )
        
        return dims, improvements
    
    def check_and_record(
        self,
        result: FitnessScore,
        config: Dict[str, Any],
        task: str = 'mnist',
    ) -> Optional[BreakthroughReport]:
        """Check if breakthrough and record if so."""
        if not self.is_breakthrough(result, task):
            return None
        
        dims, improvements = self.get_breakthrough_dims(result, task)
        
        report = BreakthroughReport(
            config=config,
            fitness=result,
            task=task,
            breakthrough_dims=dims,
            improvement_pct=improvements,
        )
        
        self.breakthroughs.append(report)
        self._save_report(report)
        
        return report
    
    def rank_breakthroughs(
        self,
        results: List[FitnessScore],
        task: str = 'mnist',
    ) -> List[Tuple[int, float, List[str]]]:
        """
        Rank variants by breakthrough potential.
        
        Returns: List of (index, score, breakthrough_dims) sorted by score descending.
        """
        rankings = []
        
        for i, result in enumerate(results):
            dims, improvements = self.get_breakthrough_dims(result, task)
            
            # Compute breakthrough score
            score = 0.0
            for dim, pct in improvements.items():
                weight = {
                    'accuracy': 3.0,
                    'perplexity': 2.0,
                    'speed': 1.0,
                    'memory': 1.0,
                    'lipschitz': 2.0,
                }.get(dim, 1.0)
                score += weight * pct
            
            rankings.append((i, score, dims))
        
        # Sort by score descending
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
    
    def get_pareto_breakthroughs(
        self,
        results: List[FitnessScore],
    ) -> List[int]:
        """Get indices of Pareto-optimal breakthroughs."""
        pareto_indices = []
        
        for i, result_i in enumerate(results):
            is_dominated = False
            for j, result_j in enumerate(results):
                if i != j and result_i.is_pareto_dominated_by(result_j):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_indices.append(i)
        
        return pareto_indices
    
    def _save_report(self, report: BreakthroughReport) -> None:
        """Save breakthrough report to disk."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"breakthrough_{report.task}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
    
    def generate_summary(self) -> str:
        """Generate markdown summary of all breakthroughs."""
        if not self.breakthroughs:
            return "# Breakthrough Summary\n\nNo breakthroughs detected yet.\n"
        
        lines = [
            "# Breakthrough Summary",
            "",
            f"**Total Breakthroughs**: {len(self.breakthroughs)}",
            "",
            "## Top Breakthroughs",
            "",
            "| Task | Accuracy | Perplexity | Lipschitz | Improvements |",
            "|------|----------|------------|-----------|--------------|",
        ]
        
        # Sort by composite score
        sorted_reports = sorted(
            self.breakthroughs,
            key=lambda r: r.fitness.composite_score(),
            reverse=True
        )
        
        for report in sorted_reports[:10]:
            improvements = ", ".join([
                f"{dim}: +{pct:.1f}%" 
                for dim, pct in report.improvement_pct.items()
            ])
            lines.append(
                f"| {report.task} | {report.fitness.accuracy:.4f} | "
                f"{report.fitness.perplexity:.2f} | {report.fitness.lipschitz:.3f} | "
                f"{improvements} |"
            )
        
        lines.extend([
            "",
            "## Configuration Details",
            "",
        ])
        
        for i, report in enumerate(sorted_reports[:5], 1):
            lines.append(f"### #{i} - {report.task}")
            lines.append("```json")
            lines.append(json.dumps(report.config, indent=2))
            lines.append("```")
            lines.append("")
        
        return "\n".join(lines)
    
    def save_summary(self, filename: str = 'breakthrough_summary.md') -> Path:
        """Save summary to file."""
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            f.write(self.generate_summary())
        return filepath
