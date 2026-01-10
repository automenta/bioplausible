"""
Evolution Engine for EqProp+SN Variants

Main orchestration loop for evolutionary optimization:
- Population management
- Generation advancement
- Parallel evaluation
- Checkpoint/resume
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import numpy as np

from .breeder import VariationBreeder, ArchConfig
from .evaluator import VariationEvaluator, EvalTier
from .fitness import FitnessScore
from .breakthrough import BreakthroughDetector


@dataclass
class EvolutionConfig:
    """Configuration for evolution run."""
    population_size: int = 20
    n_generations: int = 50
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    elite_fraction: float = 0.2
    task: str = 'mnist'
    eval_tier: EvalTier = EvalTier.TIER_2_QUICK
    seed: int = 42
    timeout_hours: float = 24.0
    checkpoint_every: int = 5
    output_dir: str = 'results/evolution'


@dataclass
class Individual:
    """An individual in the population."""
    config: ArchConfig
    fitness: Optional[FitnessScore] = None
    evaluated: bool = False
    id: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'config': self.config.to_dict(),
            'fitness': asdict(self.fitness) if self.fitness else None,
            'evaluated': self.evaluated,
            'id': self.id,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Individual':
        config = ArchConfig.from_dict(d['config'])
        fitness = FitnessScore(**d['fitness']) if d.get('fitness') else None
        return cls(
            config=config,
            fitness=fitness,
            evaluated=d.get('evaluated', False),
            id=d.get('id', 0),
        )


@dataclass
class EvolutionState:
    """State of the evolution run."""
    generation: int = 0
    population: List[Individual] = field(default_factory=list)
    best_fitness: float = 0.0
    best_individual: Optional[Individual] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'generation': self.generation,
            'population': [ind.to_dict() for ind in self.population],
            'best_fitness': self.best_fitness,
            'best_individual': self.best_individual.to_dict() if self.best_individual else None,
            'history': self.history,
            'start_time': self.start_time,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'EvolutionState':
        state = cls(
            generation=d['generation'],
            population=[Individual.from_dict(ind) for ind in d['population']],
            best_fitness=d['best_fitness'],
            history=d.get('history', []),
            start_time=d.get('start_time', datetime.now().isoformat()),
        )
        if d.get('best_individual'):
            state.best_individual = Individual.from_dict(d['best_individual'])
        return state


class EvolutionEngine:
    """Main evolution loop for EqProp+SN variants."""
    
    def __init__(
        self,
        config: EvolutionConfig = None,
        breeder: VariationBreeder = None,
        evaluator: VariationEvaluator = None,
        detector: BreakthroughDetector = None,
    ):
        self.config = config or EvolutionConfig()
        self.breeder = breeder or VariationBreeder(
            mutation_rate=self.config.mutation_rate,
            crossover_rate=self.config.crossover_rate,
            seed=self.config.seed,
        )
        self.evaluator = evaluator or VariationEvaluator()
        self.detector = detector or BreakthroughDetector(
            output_dir=f"{self.config.output_dir}/breakthroughs"
        )
        
        self.state = EvolutionState()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._next_id = 0
        self._start_time = time.time()
    
    def run(
        self,
        n_generations: Optional[int] = None,
        verbose: bool = True,
    ) -> EvolutionState:
        """
        Run evolution for specified generations.
        
        Args:
            n_generations: Number of generations (overrides config)
            verbose: Print progress
            
        Returns:
            Final evolution state
        """
        n_gens = n_generations or self.config.n_generations
        timeout_sec = self.config.timeout_hours * 3600
        
        # Initialize population if empty
        if not self.state.population:
            self._initialize_population()
        
        if verbose:
            print(f"Starting evolution: {n_gens} generations, pop={self.config.population_size}")
            print(f"Task: {self.config.task}, Tier: {self.config.eval_tier}")
        
        for gen in range(self.state.generation, self.state.generation + n_gens):
            gen_start = time.time()
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"Generation {gen + 1}/{self.state.generation + n_gens}")
                print(f"{'='*60}")
            
            # Evaluate population
            self._evaluate_population(verbose)
            
            # Record generation stats
            gen_stats = self._record_generation_stats(gen, gen_start)
            
            if verbose:
                self._print_generation_stats(gen_stats)
            
            # Check for breakthroughs
            for ind in self.state.population:
                if ind.fitness:
                    self.detector.check_and_record(
                        ind.fitness,
                        ind.config.to_dict(),
                        self.config.task,
                    )
            
            # Evolve to next generation
            self._evolve_population()
            
            self.state.generation = gen + 1
            
            # Checkpoint
            if (gen + 1) % self.config.checkpoint_every == 0:
                self._save_checkpoint()
            
            # Timeout check
            if time.time() - self._start_time > timeout_sec:
                if verbose:
                    print(f"\nTimeout reached after {gen + 1} generations")
                break
        
        # Final checkpoint
        self._save_checkpoint()
        self._save_final_report()
        
        return self.state
    
    def _initialize_population(self) -> None:
        """Create initial random population."""
        self.state.population = []
        for _ in range(self.config.population_size):
            config = self.breeder.generate_random()
            ind = Individual(config=config, id=self._next_id)
            self._next_id += 1
            self.state.population.append(ind)
    
    def _evaluate_population(self, verbose: bool = True) -> None:
        """Evaluate all unevaluated individuals."""
        for i, ind in enumerate(self.state.population):
            if ind.evaluated:
                continue
            
            if verbose:
                print(f"  [{i+1}/{len(self.state.population)}] Evaluating {ind.config.model_type} "
                      f"(d={ind.config.depth}, w={ind.config.width})")
            
            try:
                ind.fitness = self.evaluator.evaluate(
                    ind.config,
                    tier=self.config.eval_tier,
                    task=self.config.task,
                )
                ind.evaluated = True
                
                # Update best
                if ind.fitness.composite_score() > self.state.best_fitness:
                    self.state.best_fitness = ind.fitness.composite_score()
                    self.state.best_individual = ind
                    if verbose:
                        print(f"    ★ New best: acc={ind.fitness.accuracy:.4f}, "
                              f"L={ind.fitness.lipschitz:.3f}")
                        
            except Exception as e:
                if verbose:
                    print(f"    ✗ Failed: {e}")
                ind.fitness = FitnessScore(accuracy=0.0)
                ind.evaluated = True
    
    def _evolve_population(self) -> None:
        """Create next generation through selection, crossover, and mutation."""
        # Sort by fitness
        evaluated = [ind for ind in self.state.population if ind.fitness]
        evaluated.sort(key=lambda x: x.fitness.composite_score(), reverse=True)
        
        # Elite selection
        n_elite = max(1, int(len(evaluated) * self.config.elite_fraction))
        elites = evaluated[:n_elite]
        
        # Generate offspring
        new_population = []
        
        # Keep elites
        for elite in elites:
            new_ind = Individual(config=elite.config.clone(), id=self._next_id)
            self._next_id += 1
            new_population.append(new_ind)
        
        # Fill rest with offspring
        while len(new_population) < self.config.population_size:
            if len(elites) >= 2 and np.random.random() < self.config.crossover_rate:
                # Crossover
                p1, p2 = np.random.choice(elites, size=2, replace=False)
                child_config = self.breeder.crossover(p1.config, p2.config)
            else:
                # Mutation only
                parent = np.random.choice(elites)
                child_config = parent.config.clone()
            
            # Apply mutation
            child_config = self.breeder.mutate(child_config)
            
            new_ind = Individual(config=child_config, id=self._next_id)
            self._next_id += 1
            new_population.append(new_ind)
        
        self.state.population = new_population
    
    def _record_generation_stats(self, gen: int, gen_start: float) -> Dict[str, Any]:
        """Record statistics for this generation."""
        fitnesses = [ind.fitness for ind in self.state.population if ind.fitness]
        
        stats = {
            'generation': gen,
            'time_sec': time.time() - gen_start,
            'n_evaluated': len(fitnesses),
        }
        
        if fitnesses:
            accs = [f.accuracy for f in fitnesses]
            stats.update({
                'best_accuracy': max(accs),
                'mean_accuracy': np.mean(accs),
                'std_accuracy': np.std(accs),
                'best_lipschitz': min(f.lipschitz for f in fitnesses),
                'mean_lipschitz': np.mean([f.lipschitz for f in fitnesses if f.lipschitz < float('inf')]),
            })
        
        self.state.history.append(stats)
        return stats
    
    def _print_generation_stats(self, stats: Dict[str, Any]) -> None:
        """Print generation statistics."""
        print(f"\n  Gen {stats['generation'] + 1} Stats:")
        print(f"    Evaluated: {stats['n_evaluated']}")
        print(f"    Time: {stats['time_sec']:.1f}s")
        if 'best_accuracy' in stats:
            print(f"    Best Acc: {stats['best_accuracy']:.4f}")
            print(f"    Mean Acc: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f}")
            print(f"    Best L: {stats['best_lipschitz']:.3f}")
    
    def _save_checkpoint(self) -> None:
        """Save checkpoint to disk."""
        checkpoint_path = self.output_dir / f"checkpoint_gen{self.state.generation}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(self.state.to_dict(), f, indent=2, default=str)
        
        # Also save as 'latest'
        latest_path = self.output_dir / "checkpoint_latest.json"
        with open(latest_path, 'w') as f:
            json.dump(self.state.to_dict(), f, indent=2, default=str)
    
    def _save_final_report(self) -> None:
        """Generate and save final report."""
        report_lines = [
            "# Evolution Run Report",
            "",
            f"**Task**: {self.config.task}",
            f"**Generations**: {self.state.generation}",
            f"**Population Size**: {self.config.population_size}",
            f"**Total Time**: {time.time() - self._start_time:.1f}s",
            "",
            "## Best Individual",
            "",
        ]
        
        if self.state.best_individual:
            best = self.state.best_individual
            report_lines.extend([
                f"- **Accuracy**: {best.fitness.accuracy:.4f}",
                f"- **Lipschitz**: {best.fitness.lipschitz:.3f}",
                f"- **Speed**: {best.fitness.speed:.1f} iter/s",
                f"- **Memory**: {best.fitness.memory_mb:.0f} MB",
                "",
                "### Configuration",
                "```json",
                json.dumps(best.config.to_dict(), indent=2),
                "```",
            ])
        
        report_lines.extend([
            "",
            "## Evolution Trajectory",
            "",
            "| Gen | Best Acc | Mean Acc | Best L |",
            "|-----|----------|----------|--------|",
        ])
        
        for stats in self.state.history:
            if 'best_accuracy' in stats:
                report_lines.append(
                    f"| {stats['generation'] + 1} | {stats['best_accuracy']:.4f} | "
                    f"{stats['mean_accuracy']:.4f} | {stats['best_lipschitz']:.3f} |"
                )
        
        report_path = self.output_dir / "evolution_report.md"
        with open(report_path, 'w') as f:
            f.write("\n".join(report_lines))
    
    @classmethod
    def resume(cls, checkpoint_path: str) -> 'EvolutionEngine':
        """Resume evolution from checkpoint."""
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
        
        state = EvolutionState.from_dict(data)
        engine = cls()
        engine.state = state
        
        return engine
