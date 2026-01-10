"""
Evolutionary Optimization Engine

Implements evolutionary search with Pareto-based selection for multi-objective optimization.
"""

from typing import List, Dict, Any, Optional, Callable
import numpy as np
from dataclasses import dataclass

from .search_space import SearchSpace, get_search_space
from .metrics import TrialMetrics, non_dominated_sort, crowding_distance, get_pareto_frontier
from .storage import HyperoptStorage


@dataclass
class OptimizationConfig:
    """Configuration for the optimization process."""
    population_size: int = 20
    n_generations: int = 10
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    elite_fraction: float = 0.2
    random_seed: int = 42


class EvolutionaryOptimizer:
    """Multi-objective evolutionary optimizer using NSGA-II principles."""
    
    def __init__(
        self,
        model_names: List[str],
        config: OptimizationConfig = None,
        storage: HyperoptStorage = None
    ):
        self.model_names = model_names
        self.config = config or OptimizationConfig()
        self.storage = storage or HyperoptStorage()
        self.rng = np.random.default_rng(self.config.random_seed)
        
        # Search spaces for each model
        self.search_spaces = {
            name: get_search_space(name) for name in model_names
        }
        
        # Current population (trial IDs)
        self.population: Dict[str, List[int]] = {name: [] for name in model_names}
        self.generation = 0
    
    def _sanitize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert numpy types to native Python types for JSON serialization."""
        sanitized = {}
        for k, v in config.items():
            if isinstance(v, (np.integer, np.int64, np.int32)):
                sanitized[k] = int(v)
            elif isinstance(v, (np.floating, np.float64, np.float32)):
                sanitized[k] = float(v)
            elif isinstance(v, np.ndarray):
                sanitized[k] = v.tolist()
            else:
                sanitized[k] = v
        return sanitized

    def initialize_population(self, model_name: str) -> List[int]:
        """Initialize random population for a model."""
        space = self.search_spaces[model_name]
        trial_ids = []
        
        for _ in range(self.config.population_size):
            config = space.sample(self.rng)
            config = self._sanitize_config(config)
            trial_id = self.storage.create_trial(model_name, config)
            trial_ids.append(trial_id)
        
        self.population[model_name] = trial_ids
        return trial_ids
    
    def select_parents(self, model_name: str) -> List[TrialMetrics]:
        """Select parents using tournament selection based on Pareto ranking."""
        # Get completed trials
        all_trials = self.storage.get_all_trials(model_name, status='completed')
        
        if len(all_trials) < 2:
            return []
        
        # Non-dominated sort
        fronts = non_dominated_sort(all_trials)
        
        # Assign fitness based on front rank and crowding distance
        fitness_scores = {}
        for front_idx, front_indices in enumerate(fronts):
            distances = crowding_distance(all_trials, front_indices)
            for i, trial_idx in enumerate(front_indices):
                # Lower front_idx = better
                # Higher distance = better (more diversity)
                fitness_scores[trial_idx] = (len(fronts) - front_idx) + distances[i] / 10.0
        
        # Select top performers
        sorted_trials = sorted(
            all_trials,
            key=lambda t: fitness_scores.get(all_trials.index(t), 0),
            reverse=True
        )
        
        n_parents = max(2, int(self.config.population_size * self.config.elite_fraction))
        return sorted_trials[:n_parents]
    
    def generate_offspring(
        self,
        model_name: str,
        parents: List[TrialMetrics],
        n_offspring: int
    ) -> List[int]:
        """Generate offspring through crossover and mutation."""
        if not parents:
            return []
        
        space = self.search_spaces[model_name]
        offspring_ids = []
        
        for _ in range(n_offspring):
            if self.rng.random() < self.config.crossover_rate and len(parents) >= 2:
                # Crossover
                p1, p2 = self.rng.choice(parents, size=2, replace=False)
                child_config = space.crossover(p1.config, p2.config, self.rng)
            else:
                # Random parent
                parent = self.rng.choice(parents)
                child_config = parent.config.copy()
            
            # Mutation
            child_config = space.mutate(child_config, self.config.mutation_rate, self.rng)
            child_config = self._sanitize_config(child_config)
            
            # Create trial
            trial_id = self.storage.create_trial(model_name, child_config)
            offspring_ids.append(trial_id)
        
        return offspring_ids
    
    def evolve_generation(self, model_name: str) -> List[int]:
        """Evolve to the next generation."""
        # Select parents
        parents = self.select_parents(model_name)
        
        if not parents:
            # No completed trials yet, return random population
            return self.initialize_population(model_name)
        
        # Generate offspring
        n_offspring = self.config.population_size - len(parents)
        offspring_ids = self.generate_offspring(model_name, parents, n_offspring)
        
        # New population = elite parents + offspring
        new_population = [p.trial_id for p in parents] + offspring_ids
        self.population[model_name] = new_population
        self.generation += 1
        
        return new_population
    
    def get_next_trial(self, model_name: str = None) -> Optional[int]:
        """Get the next pending trial to run.
        
        If model_name is None, returns the next trial from any model.
        """
        if model_name:
            models_to_check = [model_name]
        else:
            models_to_check = self.model_names
        
        for name in models_to_check:
            # Check if we need to initialize
            if not self.population.get(name):
                self.initialize_population(name)
            
            # Find a pending trial
            for trial_id in self.population[name]:
                trial = self.storage.get_trial(trial_id)
                if trial and trial.status == 'pending':
                    return trial_id
        
        return None
    
    def update_pareto_frontiers(self):
        """Update Pareto frontier markings for all models."""
        for model_name in self.model_names:
            trials = self.storage.get_all_trials(model_name, status='completed')
            if trials:
                frontier_indices = get_pareto_frontier(trials)
                frontier_ids = [trials[i].trial_id for i in frontier_indices]
                self.storage.mark_pareto_frontier(frontier_ids)
    
    def get_best_configs(self, model_name: str, top_k: int = 5) -> List[TrialMetrics]:
        """Get best configurations for a model."""
        trials = self.storage.get_all_trials(model_name, status='completed')
        if not trials:
            return []
        
        # Sort by composite score
        sorted_trials = sorted(trials, key=lambda t: t.composite_score(), reverse=True)
        return sorted_trials[:top_k]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        stats = {
            'generation': self.generation,
            'models': {}
        }
        
        for model_name in self.model_names:
            trials = self.storage.get_all_trials(model_name)
            completed = [t for t in trials if t.status == 'completed']
            
            stats['models'][model_name] = {
                'total_trials': len(trials),
                'completed': len(completed),
                'pending': len([t for t in trials if t.status == 'pending']),
                'failed': len([t for t in trials if t.status == 'failed']),
                'pareto_size': len(self.storage.get_all_trials(model_name)) if completed else 0
            }
            
            if completed:
                accuracies = [t.accuracy for t in completed]
                stats['models'][model_name]['best_accuracy'] = max(accuracies)
                stats['models'][model_name]['mean_accuracy'] = np.mean(accuracies)
        
        return stats
