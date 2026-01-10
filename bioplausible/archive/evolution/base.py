"""
Base abstractions for the evolution system.

Defines abstract interfaces that make the system extensible and modular.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .breeder import ArchConfig
from .fitness import FitnessScore


class ModelBuilder(ABC):
    """Abstract interface for building models from configurations."""
    
    @abstractmethod
    def build(self, config: ArchConfig, task: str) -> Any:
        """
        Build a model from architecture configuration.
        
        Args:
            config: Architecture configuration
            task: Task name (mnist, cifar10, etc.)
            
        Returns:
            Built model instance
        """
        pass
    
    @abstractmethod
    def validate_config(self, config: ArchConfig, task: str) -> bool:
        """Validate that config is compatible with this builder."""
        pass


class Evaluator(ABC):
    """Abstract interface for model evaluation."""
    
    @abstractmethod
    def evaluate(
        self,
        config: ArchConfig,
        **kwargs
    ) -> FitnessScore:
        """
        Evaluate a configuration.
        
        Args:
            config: Architecture to evaluate
            **kwargs: Additional evaluation parameters
            
        Returns:
            Fitness score
        """
        pass


class SelectionStrategy(ABC):
    """Abstract interface for parent selection in evolution."""
    
    @abstractmethod
    def select_parents(
        self,
        population: list,
        n_parents: int,
    ) -> list:
        """
        Select parents from population for breeding.
        
        Args:
            population: List of evaluated individuals
            n_parents: Number of parents to select
            
        Returns:
            Selected parent individuals
        """
        pass


class BreedingStrategy(ABC):
    """Abstract interface for breeding new individuals."""
    
    @abstractmethod
    def breed(
        self,
        parents: list,
        n_offspring: int,
    ) -> list:
        """
        Create offspring from parents.
        
        Args:
            parents: Parent configurations
            n_offspring: Number of offspring to create
            
        Returns:
            List of new configurations
        """
        pass


class TerminationCriterion(ABC):
    """Abstract interface for evolution termination conditions."""
    
    @abstractmethod
    def should_terminate(
        self,
        generation: int,
        population: list,
        elapsed_time: float,
    ) -> bool:
        """
        Check if evolution should terminate.
        
        Args:
            generation: Current generation number
            population: Current population
            elapsed_time: Elapsed time in seconds
            
        Returns:
            True if evolution should stop
        """
        pass


@dataclass
class EvaluationResult:
    """Standard result format for evaluations."""
    fitness: FitnessScore
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
