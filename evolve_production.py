#!/usr/bin/env python3
"""
Production Evolution Runner

Robust overnight evolution with:
- Diversity preservation (avoid concentration on any architecture type)
- Reliable long-running with checkpoints and crash recovery
- Comprehensive data collection for reports
- Incremental runs that accumulate data
- Report generation from accumulated results

Usage:
    # Start overnight run (8 hours)
    python evolve_production.py --hours 8 --output results/evolution
    
    # Resume from checkpoint
    python evolve_production.py --resume results/evolution
    
    # Generate report from accumulated data
    python evolve_production.py --report results/evolution
    
    # Continue an existing run for more hours
    python evolve_production.py --continue results/evolution --hours 4
"""

import argparse
import json
import time
import logging
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))

from evolution import (
    VariationBreeder, ArchConfig, VariationEvaluator, EvalTier,
    FitnessScore, BreakthroughDetector,
    AlgorithmBreeder, AlgorithmConfig, build_algorithm_variant,
    ALGORITHM_PRESETS,
)
from evolution.multitask import (
    TaskAssigner, TaskNormalizedFitness, MultiTaskBreakthroughDetector,
    TASK_POOL,
)
from evolution.pareto import (
    ParetoFrontAnalyzer, MultiObjectiveMetrics, identify_algorithm_strategies,
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ProductionConfig:
    """Configuration for production runs."""
    # Time limits
    max_hours: float = 8.0
    checkpoint_interval_min: int = 10
    
    # Population
    population_size: int = 30
    
    # Diversity settings
    min_per_model_type: int = 3  # Minimum individuals per model type
    max_per_model_type: int = 8  # Maximum to prevent concentration
    diversity_pressure: float = 0.3  # Weight for diversity in selection
    
    # Model types to explore
    model_types: List[str] = field(default_factory=lambda: [
        'looped_mlp', 'transformer', 'conv', 'hebbian', 'feedback_alignment'
    ])
    
    # Algorithm types to explore
    explore_algorithms: bool = True
    
    # Evaluation - MULTI-TASK
    eval_tier: int = 2  # Tier 2 (5 min) for overnight
    tasks: List[str] = field(default_factory=lambda: [
        'mnist', 'fashion', 'cifar10', 'shakespeare',
        'cartpole', 'acrobot',  # RL tasks
    ])  # Tasks to sample from
    
    # Genetic parameters
    mutation_rate: float = 0.3
    crossover_rate: float = 0.6
    elite_fraction: float = 0.15
    
    # Seeds for reproducibility
    base_seed: int = 42
    
    # Output
    output_dir: str = 'results/evolution'
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ProductionConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ============================================================================
# Data Collection
# ============================================================================

@dataclass
class EvaluationRecord:
    """Single evaluation result."""
    id: int
    timestamp: str
    model_type: str
    task: str  # NEW: Track which task
    config: Dict[str, Any]
    fitness: Dict[str, float]
    generation: int
    run_id: str
    success: bool
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DataCollector:
    """Collects and persists evaluation data."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_file = self.output_dir / 'evaluations.jsonl'
        self.summary_file = self.output_dir / 'summary.json'
        
        self._next_id = self._get_next_id()
        self._records: List[EvaluationRecord] = []
        self._run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.logger = logging.getLogger('DataCollector')
    
    def _get_next_id(self) -> int:
        """Get next ID by counting existing records."""
        if not self.data_file.exists():
            return 0
        count = 0
        with open(self.data_file, 'r') as f:
            for line in f:
                count += 1
        return count
    
    def record(
        self,
        model_type: str,
        task: str,  # NEW: task parameter
        config: Dict[str, Any],
        fitness: FitnessScore,
        generation: int,
        success: bool = True,
        error: Optional[str] = None,
    ) -> EvaluationRecord:
        """Record an evaluation result."""
        record = EvaluationRecord(
            id=self._next_id,
            timestamp=datetime.now().isoformat(),
            model_type=model_type,
            task=task,  # NEW
            config=config,
            fitness={
                'accuracy': fitness.accuracy,
                'perplexity': fitness.perplexity if fitness.perplexity < float('inf') else -1,
                'speed': fitness.speed,
                'memory_mb': fitness.memory_mb,
                'lipschitz': fitness.lipschitz if fitness.lipschitz < float('inf') else -1,
                'generalization': fitness.generalization,
                'composite': fitness.composite_score(),
                'parameter_count': fitness.parameter_count,  # NEW
                'train_time_sec': fitness.train_time_sec,  # NEW
            },
            generation=generation,
            run_id=self._run_id,
            success=success,
            error=error,
        )
        
        self._next_id += 1
        self._records.append(record)
        
        # Append to file immediately (crash-safe)
        with open(self.data_file, 'a') as f:
            f.write(json.dumps(record.to_dict()) + '\n')
        
        return record
    
    def load_all_records(self) -> List[Dict[str, Any]]:
        """Load all records from data file."""
        if not self.data_file.exists():
            return []
        
        records = []
        with open(self.data_file, 'r') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records
    
    def update_summary(self, state: Dict[str, Any]):
        """Update summary file."""
        with open(self.summary_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from all accumulated data."""
        records = self.load_all_records()
        
        if not records:
            return {'total': 0}
        
        # Group by model type AND task
        by_type = defaultdict(list)
        by_task = defaultdict(list)
        for r in records:
            by_type[r['model_type']].append(r)
            by_task[r.get('task', 'unknown')].append(r)
        
        # Compute stats
        stats = {
            'total': len(records),
            'successful': sum(1 for r in records if r['success']),
            'by_model_type': {},
            'by_task': {},  # NEW
            'best_per_type': {},
            'best_per_task': {},  # NEW
            'runs': list(set(r['run_id'] for r in records)),
        }
        
        for mtype, recs in by_type.items():
            successful = [r for r in recs if r['success'] and r['fitness']['accuracy'] > 0]
            if successful:
                accs = [r['fitness']['accuracy'] for r in successful]
                stats['by_model_type'][mtype] = {
                    'count': len(recs),
                    'successful': len(successful),
                    'mean_acc': np.mean(accs),
                    'max_acc': max(accs),
                    'std_acc': np.std(accs),
                }
                
                best = max(successful, key=lambda r: r['fitness']['accuracy'])
                stats['best_per_type'][mtype] = {
                    'accuracy': best['fitness']['accuracy'],
                    'config': best['config'],
                }
        
        # NEW: Stats per task
        for task_name, recs in by_task.items():
            successful = [r for r in recs if r['success'] and r['fitness']['accuracy'] > 0]
            if successful:
                accs = [r['fitness']['accuracy'] for r in successful]
                stats['by_task'][task_name] = {
                    'count': len(recs),
                    'successful': len(successful),
                    'mean_acc': np.mean(accs),
                    'max_acc': max(accs),
                    'std_acc': np.std(accs),
                }
                
                best = max(successful, key=lambda r: r['fitness']['accuracy'])
                stats['best_per_task'][task_name] = {
                    'accuracy': best['fitness']['accuracy'],
                    'model_type': best['model_type'],
                    'config': best['config'],
                }
        
        return stats


# ============================================================================
# Diversity-Preserving Population
# ============================================================================

class DiversePopulation:
    """Population manager that preserves diversity across model types."""
    
    def __init__(
        self,
        config: ProductionConfig,
        arch_breeder: VariationBreeder,
        algo_breeder: AlgorithmBreeder,
    ):
        self.config = config
        self.arch_breeder = arch_breeder
        self.algo_breeder = algo_breeder
        
        self.individuals: List[Dict[str, Any]] = []
        self._next_id = 0
        
        self.logger = logging.getLogger('DiversePopulation')
    
    def initialize(self):
        """Initialize population with guaranteed diversity."""
        self.individuals = []
        
        # Ensure minimum representation of each model type
        for mtype in self.config.model_types:
            for _ in range(self.config.min_per_model_type):
                ind = self._create_individual(model_type=mtype)
                self.individuals.append(ind)
        
        # Fill remaining slots with random types
        while len(self.individuals) < self.config.population_size:
            mtype = np.random.choice(self.config.model_types)
            ind = self._create_individual(model_type=mtype)
            self.individuals.append(ind)
        
        self.logger.info(f"Initialized population with {len(self.individuals)} individuals")
        self._log_diversity()
    
    def _create_individual(
        self, 
        model_type: Optional[str] = None,
        arch_config: Optional[ArchConfig] = None,
        algo_config: Optional[AlgorithmConfig] = None,
    ) -> Dict[str, Any]:
        """Create a new individual."""
        if arch_config is None:
            arch_config = self.arch_breeder.generate_random()
            if model_type:
                arch_config.model_type = model_type
        
        if algo_config is None and self.config.explore_algorithms:
            algo_config = self.algo_breeder.generate_informed()
        
        ind = {
            'id': self._next_id,
            'arch_config': arch_config,
            'algo_config': algo_config,
            'fitness': None,
            'evaluated': False,
            'generation': 0,
        }
        self._next_id += 1
        return ind
    
    def evolve_next_generation(self, generation: int):
        """Create next generation preserving diversity."""
        # Sort by fitness
        evaluated = [ind for ind in self.individuals if ind['fitness'] is not None]
        if not evaluated:
            return
        
        evaluated.sort(key=lambda x: x['fitness'].composite_score(), reverse=True)
        
        # Count by type
        type_counts = self._count_by_type()
        
        # Elite selection (top performers, but limit per type)
        new_pop = []
        elite_per_type = defaultdict(int)
        max_elite_per_type = 2
        
        for ind in evaluated:
            mtype = ind['arch_config'].model_type
            if elite_per_type[mtype] < max_elite_per_type:
                # Clone elite
                new_ind = self._clone_individual(ind, generation)
                new_pop.append(new_ind)
                elite_per_type[mtype] += 1
            
            if len(new_pop) >= int(self.config.population_size * self.config.elite_fraction):
                break
        
        # Fill with offspring, ensuring diversity
        while len(new_pop) < self.config.population_size:
            # Determine which type needs more representation
            current_counts = defaultdict(int)
            for ind in new_pop:
                current_counts[ind['arch_config'].model_type] += 1
            
            # Prioritize under-represented types
            underrep = [t for t in self.config.model_types 
                       if current_counts[t] < self.config.min_per_model_type]
            
            if underrep:
                target_type = np.random.choice(underrep)
            else:
                # Random, but avoid over-represented types
                weights = []
                for t in self.config.model_types:
                    if current_counts[t] >= self.config.max_per_model_type:
                        weights.append(0.01)
                    else:
                        weights.append(1.0)
                weights = np.array(weights) / sum(weights)
                target_type = np.random.choice(self.config.model_types, p=weights)
            
            # Select parents of target type if available
            type_parents = [ind for ind in evaluated 
                          if ind['arch_config'].model_type == target_type]
            
            if len(type_parents) >= 2 and np.random.random() < self.config.crossover_rate:
                p1, p2 = np.random.choice(type_parents, size=2, replace=False)
                child_arch = self.arch_breeder.crossover(p1['arch_config'], p2['arch_config'])
                if self.config.explore_algorithms and p1['algo_config'] and p2['algo_config']:
                    child_algo = self.algo_breeder.crossover(p1['algo_config'], p2['algo_config'])
                else:
                    child_algo = None
            elif type_parents:
                parent = np.random.choice(type_parents)
                child_arch = parent['arch_config'].clone()
                child_algo = parent['algo_config'].clone() if parent['algo_config'] else None
            else:
                # Generate new random
                child_arch = self.arch_breeder.generate_random()
                child_arch.model_type = target_type
                child_algo = self.algo_breeder.generate_informed() if self.config.explore_algorithms else None
            
            # Mutate
            child_arch = self.arch_breeder.mutate(child_arch)
            child_arch.model_type = target_type  # Preserve target type
            
            if child_algo:
                child_algo = self.algo_breeder.mutate(child_algo)
            
            new_ind = self._create_individual(
                arch_config=child_arch,
                algo_config=child_algo,
            )
            new_ind['generation'] = generation
            new_pop.append(new_ind)
        
        self.individuals = new_pop
        self._log_diversity()
    
    def _clone_individual(self, ind: Dict, generation: int) -> Dict:
        """Clone an individual for next generation."""
        return {
            'id': self._next_id,
            'arch_config': ind['arch_config'].clone(),
            'algo_config': ind['algo_config'].clone() if ind['algo_config'] else None,
            'fitness': None,
            'evaluated': False,
            'generation': generation,
        }
    
    def _count_by_type(self) -> Dict[str, int]:
        """Count individuals by model type."""
        counts = defaultdict(int)
        for ind in self.individuals:
            counts[ind['arch_config'].model_type] += 1
        return counts
    
    def _log_diversity(self):
        """Log population diversity."""
        counts = self._count_by_type()
        self.logger.info(f"Population diversity: {dict(counts)}")
    
    def get_unevaluated(self) -> List[Dict]:
        """Get individuals that need evaluation."""
        return [ind for ind in self.individuals if not ind['evaluated']]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize population."""
        return {
            'individuals': [
                {
                    'id': ind['id'],
                    'arch_config': ind['arch_config'].to_dict(),
                    'algo_config': ind['algo_config'].to_dict() if ind['algo_config'] else None,
                    'fitness': asdict(ind['fitness']) if ind['fitness'] else None,
                    'evaluated': ind['evaluated'],
                    'generation': ind['generation'],
                }
                for ind in self.individuals
            ],
            'next_id': self._next_id,
        }
    
    def from_dict(self, data: Dict[str, Any]):
        """Deserialize population."""
        self._next_id = data['next_id']
        self.individuals = []
        
        for d in data['individuals']:
            ind = {
                'id': d['id'],
                'arch_config': ArchConfig.from_dict(d['arch_config']),
                'algo_config': AlgorithmConfig.from_dict(d['algo_config']) if d['algo_config'] else None,
                'fitness': FitnessScore(**d['fitness']) if d['fitness'] else None,
                'evaluated': d['evaluated'],
                'generation': d['generation'],
            }
            self.individuals.append(ind)


# ============================================================================
# Production Runner
# ============================================================================

class ProductionRunner:
    """Robust evolution runner for overnight operation."""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Components
        self.arch_breeder = VariationBreeder(
            mutation_rate=config.mutation_rate,
            crossover_rate=config.crossover_rate,
            seed=config.base_seed,
        )
        self.algo_breeder = AlgorithmBreeder(
            mutation_rate=config.mutation_rate,
            crossover_rate=config.crossover_rate,
            seed=config.base_seed + 1000,
        )
        self.evaluator = VariationEvaluator(
            verbose=False,
        )
        self.collector = DataCollector(config.output_dir)
        self.population = DiversePopulation(
            config=config,
            arch_breeder=self.arch_breeder,
            algo_breeder=self.algo_breeder,
        )
        
        # NEW: Multi-task components
        self.task_assigner = TaskAssigner(
            tasks=config.tasks,
            seed=config.base_seed + 2000,
        )
        self.multitask_detector = MultiTaskBreakthroughDetector()
        
        # NEW: Pareto analysis
        self.pareto_analyzer = ParetoFrontAnalyzer()
        
        # State
        self.generation = 0
        self.start_time = None
        self.total_evaluations = 0
        self.successful_evaluations = 0
        
        self.logger = logging.getLogger('ProductionRunner')
    
    def _setup_logging(self):
        """Setup file and console logging."""
        log_file = self.output_dir / 'evolution.log'
        
        # Root logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ]
        )
    
    def run(self):
        """Run evolution for configured duration."""
        self.start_time = time.time()
        end_time = self.start_time + self.config.max_hours * 3600
        last_checkpoint = self.start_time
        
        self.logger.info(f"Starting production evolution run")
        self.logger.info(f"Config: {self.config.to_dict()}")
        self.logger.info(f"Will run until {datetime.fromtimestamp(end_time)}")
        
        # Initialize or restore population
        if not self.population.individuals:
            self.population.initialize()
        
        try:
            while time.time() < end_time:
                gen_start = time.time()
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"Generation {self.generation}")
                self.logger.info(f"{'='*60}")
                
                # Evaluate population
                self._evaluate_population()
                
                # Log generation stats
                self._log_generation_stats()
                
                # Evolve
                self.generation += 1
                self.population.evolve_next_generation(self.generation)
                
                # Checkpoint
                if time.time() - last_checkpoint > self.config.checkpoint_interval_min * 60:
                    self._save_checkpoint()
                    last_checkpoint = time.time()
                
                gen_time = time.time() - gen_start
                remaining = end_time - time.time()
                self.logger.info(f"Generation time: {gen_time:.1f}s, Remaining: {remaining/3600:.1f}h")
                
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Error: {e}")
            self.logger.error(traceback.format_exc())
        finally:
            self._save_checkpoint()
            self._generate_report()
    
    def _evaluate_population(self):
        """Evaluate all unevaluated individuals on diverse tasks."""
        unevaluated = self.population.get_unevaluated()
        
        # Reset task assigner for this generation
        self.task_assigner.reset_for_generation()
        
        for i, ind in enumerate(unevaluated):
            arch = ind['arch_config']
            
            # NEW: Assign task round-robin
            task = self.task_assigner.assign_task(ind['id'])
            
            self.logger.info(f"  [{i+1}/{len(unevaluated)}] Evaluating {arch.model_type} "
                           f"(d={arch.depth}, w={arch.width}) on {task}")
            
            try:
                fitness = self.evaluator.evaluate(
                    arch,
                    tier=EvalTier(self.config.eval_tier),
                    task=task,  # Use assigned task
                )
                ind['fitness'] = fitness
                ind['evaluated'] = True
                ind['task'] = task  # Store task
                self.successful_evaluations += 1
                
                # Record
                self.collector.record(
                    model_type=arch.model_type,
                    task=task,  # NEW
                    config=arch.to_dict(),
                    fitness=fitness,
                    generation=self.generation,
                    success=True,
                )
                
                # Check for breakthroughs
                self.multitask_detector.check_breakthrough(
                    task=task,
                    accuracy=fitness.accuracy,
                    config=arch.to_dict(),
                )
                
                # NEW: Record for Pareto analysis
                pareto_metrics = MultiObjectiveMetrics(
                    accuracy=fitness.accuracy,
                    train_time_sec=fitness.train_time_sec,
                    param_count=fitness.parameter_count,
                    lipschitz=fitness.lipschitz,
                    model_type=arch.model_type,
                    task=task,
                    config=arch.to_dict(),
                )
                self.pareto_analyzer.add_solution(pareto_metrics)
                
                self.logger.info(f"    Acc: {fitness.accuracy:.4f}, L: {fitness.lipschitz:.3f}")
                
            except Exception as e:
                self.logger.warning(f"    Failed: {e}")
                # Record failure
                ind['fitness'] = FitnessScore(accuracy=0.0)
                ind['evaluated'] = True
                ind['task'] = task
                
                self.collector.record(
                    model_type=arch.model_type,
                    task=task,  # NEW
                    config=arch.to_dict(),
                    fitness=FitnessScore(accuracy=0.0),
                    generation=self.generation,
                    success=False,
                    error=str(e),
                )
            
            self.total_evaluations += 1
        
        # Log task coverage
        coverage = self.task_assigner.get_coverage_stats()
        self.logger.info(f"  Task coverage: {coverage}")
    
    def _log_generation_stats(self):
        """Log statistics for current generation."""
        evaluated = [ind for ind in self.population.individuals if ind['fitness']]
        if not evaluated:
            return
        
        accs = [ind['fitness'].accuracy for ind in evaluated]
        best = max(evaluated, key=lambda x: x['fitness'].accuracy)
        
        self.logger.info(f"  Generation stats:")
        self.logger.info(f"    Best acc: {max(accs):.4f}")
        self.logger.info(f"    Mean acc: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        self.logger.info(f"    Best model: {best['arch_config'].model_type}")
    
    def _save_checkpoint(self):
        """Save checkpoint for recovery."""
        checkpoint = {
            'config': self.config.to_dict(),
            'generation': self.generation,
            'population': self.population.to_dict(),
            'total_evaluations': self.total_evaluations,
            'successful_evaluations': self.successful_evaluations,
            'timestamp': datetime.now().isoformat(),
        }
        
        checkpoint_file = self.output_dir / 'checkpoint.json'
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)
        
        self.logger.info(f"Checkpoint saved: gen={self.generation}, evals={self.total_evaluations}")
    
    def load_checkpoint(self):
        """Load checkpoint to resume."""
        checkpoint_file = self.output_dir / 'checkpoint.json'
        if not checkpoint_file.exists():
            self.logger.warning("No checkpoint found")
            return False
        
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        
        self.generation = checkpoint['generation']
        self.total_evaluations = checkpoint['total_evaluations']
        self.successful_evaluations = checkpoint['successful_evaluations']
        self.population.from_dict(checkpoint['population'])
        
        self.logger.info(f"Resumed from checkpoint: gen={self.generation}")
        return True
    
    def _generate_report(self):
        """Generate comprehensive report from accumulated data."""
        stats = self.collector.get_stats()
        
        lines = [
            "# Evolution Report",
            "",
            f"**Generated**: {datetime.now().isoformat()}",
            f"**Total Evaluations**: {stats['total']}",
            f"**Successful**: {stats['successful']}",
            f"**Runs**: {len(stats.get('runs', []))}",
            "",
            "## Best Per Model Type",
            "",
            "| Model Type | Best Accuracy | Mean ± Std | Count |",
            "|------------|---------------|------------|-------|",
        ]
        
        for mtype, mstats in stats.get('by_model_type', {}).items():
            lines.append(
                f"| {mtype} | {mstats['max_acc']:.4f} | "
                f"{mstats['mean_acc']:.4f} ± {mstats['std_acc']:.4f} | {mstats['count']} |"
            )
        
        
        lines.extend([
            "",
            "## Best Per Task",
            "",
            "| Task | Best Accuracy | Best Model | Mean ± Std | Count |",
            "|------|---------------|------------|------------|-------|",
        ])
        
        for task_name, tstats in stats.get('by_task', {}).items():
            best_model = stats.get('best_per_task', {}).get(task_name, {}).get('model_type', 'unknown')
            lines.append(
                f"| {task_name} | {tstats['max_acc']:.4f} | {best_model} | "
                f"{tstats['mean_acc']:.4f} ± {tstats['std_acc']:.4f} | {tstats['count']} |"
            )
        
        # Add breakthrough summary
        lines.extend([
            "",
            self.multitask_detector.summarize(),
            "",
        ])
        
        # NEW: Add Pareto analysis
        lines.extend([
            self.pareto_analyzer.generate_report(),
            "",
            "## Top Configurations",
            "",
        ])

        
        # Get top 10 overall
        records = self.collector.load_all_records()
        successful = [r for r in records if r['success'] and r['fitness']['accuracy'] > 0]
        top_10 = sorted(successful, key=lambda r: r['fitness']['accuracy'], reverse=True)[:10]
        
        for i, rec in enumerate(top_10, 1):
            lines.append(f"### #{i}: {rec['model_type']} - {rec['fitness']['accuracy']:.4f}")
            lines.append("```json")
            lines.append(json.dumps(rec['config'], indent=2))
            lines.append("```")
            lines.append("")
        
        report_file = self.output_dir / 'report.md'
        with open(report_file, 'w') as f:
            f.write('\n'.join(lines))
        
        self.logger.info(f"Report generated: {report_file}")
        return report_file


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Production evolution runner for overnight operation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Mode
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--resume', type=str, metavar='DIR',
                     help='Resume from checkpoint in directory')
    mode.add_argument('--report', type=str, metavar='DIR',
                     help='Generate report from existing data')
    mode.add_argument('--continue', dest='continue_dir', type=str, metavar='DIR',
                     help='Continue existing run for more time')
    
    # Duration
    parser.add_argument('--hours', type=float, default=8.0,
                       help='Maximum runtime in hours (default: 8)')
    
    
    # Population
    parser.add_argument('--population', '-p', type=int, default=30,
                       help='Population size (default: 30)')
    
    # Tier
    parser.add_argument('--tier', type=int, default=2,
                       choices=[1, 2, 3],
                       help='Evaluation tier (default: 2)')
    
    # Output
    parser.add_argument('--output', '-o', type=str, default='results/evolution',
                       help='Output directory')
    
    # Advanced
    parser.add_argument('--no-algorithms', action='store_true',
                       help='Disable algorithm evolution')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Report only mode
    if args.report:
        collector = DataCollector(args.report)
        stats = collector.get_stats()
        print(f"Total evaluations: {stats['total']}")
        print(f"Successful: {stats['successful']}")
        for mtype, mstats in stats.get('by_model_type', {}).items():
            print(f"  {mtype}: best={mstats['max_acc']:.4f}, "
                  f"mean={mstats['mean_acc']:.4f}±{mstats['std_acc']:.4f}, n={mstats['count']}")
        
        # Generate report
        config = ProductionConfig(output_dir=args.report)
        runner = ProductionRunner(config)
        runner._generate_report()
        return
    
    # Build config
    output_dir = args.resume or args.continue_dir or args.output
    
    config = ProductionConfig(
        max_hours=args.hours,
        population_size=args.population,
        eval_tier=args.tier,
        output_dir=output_dir,
        explore_algorithms=not args.no_algorithms,
        base_seed=args.seed,
    )
    
    # Create runner
    runner = ProductionRunner(config)
    
    # Resume if requested
    if args.resume or args.continue_dir:
        if not runner.load_checkpoint():
            print("No checkpoint to resume from, starting fresh")
    
    # Run
    print(f"\n{'='*60}")
    print("Production Evolution Runner")
    print(f"{'='*60}")
    print(f"Output: {config.output_dir}")
    print(f"Duration: {config.max_hours} hours")
    print(f"Population: {config.population_size}")
    print(f"Tasks: {', '.join(config.tasks)}")
    print(f"Tier: {config.eval_tier}")
    print(f"{'='*60}\n")
    
    runner.run()


if __name__ == '__main__':
    main()
