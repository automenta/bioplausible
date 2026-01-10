"""
Multi-Objective Fitness Function for EqProp+SN Evolution

Evaluates model variants across multiple dimensions:
- Accuracy/Perplexity (task performance)
- Speed (training efficiency)
- Memory (GPU usage)
- Stability (Lipschitz constant)
- Generalization (train-test gap)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class FitnessScore:
    """Multi-objective fitness score for a model variant."""
    
    # Performance metrics
    accuracy: float = 0.0           # Test accuracy (0-1, higher = better)
    perplexity: float = float('inf')  # LM perplexity (lower = better)
    
    # Efficiency metrics
    speed: float = 0.0              # Iterations/second
    memory_mb: float = float('inf')  # Peak GPU memory in MB
    train_time_sec: float = 0.0     # Total training time
    parameter_count: int = 0        # Total trainable parameters
    
    # Stability metrics
    lipschitz: float = float('inf')  # Final Lipschitz constant (target: ≤ 1.0)
    lipschitz_trajectory: list = field(default_factory=list)  # L(t) over training
    stability: float = 0.0          # 1 / variance across seeds
    
    # Generalization
    train_accuracy: float = 0.0
    generalization: float = 0.0     # 1 - abs(train - test) / train
    
    # Metadata
    config: Dict[str, Any] = field(default_factory=dict)
    task: str = ""
    seed: int = 42
    
    def composite_score(
        self, 
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Compute weighted composite score for ranking.
        
        Higher is always better for the composite score.
        """
        if weights is None:
            weights = {
                'accuracy': 2.0,
                'perplexity': -1.0,  # Negative because lower is better
                'speed': 0.5,
                'memory': -0.3,      # Negative because lower is better
                'lipschitz': -1.5,   # Negative, penalty for L > 1
                'stability': 0.5,
                'generalization': 1.0,
            }
        
        score = 0.0
        
        # Accuracy (0-1) - direct contribution
        score += weights.get('accuracy', 0) * self.accuracy
        
        # Perplexity - lower is better, use log scale
        if self.perplexity < float('inf'):
            # Convert to 0-1 scale where lower perplexity = higher score
            ppl_score = 1.0 / (1.0 + np.log(self.perplexity + 1))
            score += weights.get('perplexity', 0) * ppl_score
        
        # Speed - normalize assuming 100 iter/sec is excellent
        speed_score = min(1.0, self.speed / 100.0)
        score += weights.get('speed', 0) * speed_score
        
        # Memory - normalize, 1GB baseline, 8GB max
        if self.memory_mb < float('inf'):
            mem_score = 1.0 - min(1.0, (self.memory_mb - 1000) / 7000)
            score += weights.get('memory', 0) * mem_score
        
        # Lipschitz - heavy penalty for L > 1
        if self.lipschitz < float('inf'):
            if self.lipschitz <= 1.0:
                lip_score = 1.0  # Perfect
            elif self.lipschitz <= 1.1:
                lip_score = 0.5  # Acceptable
            else:
                lip_score = max(0, 1.0 - (self.lipschitz - 1.0))
            score += weights.get('lipschitz', 0) * lip_score
        
        # Stability
        score += weights.get('stability', 0) * min(1.0, self.stability)
        
        # Generalization
        score += weights.get('generalization', 0) * self.generalization
        
        return score
    
    def is_pareto_dominated_by(self, other: 'FitnessScore') -> bool:
        """Check if this solution is dominated by another."""
        dominated_dims = 0
        equal_dims = 0
        total_dims = 4  # accuracy, perplexity, speed, memory
        
        # Accuracy: higher is better
        if other.accuracy > self.accuracy:
            dominated_dims += 1
        elif other.accuracy == self.accuracy:
            equal_dims += 1
            
        # Perplexity: lower is better
        if other.perplexity < self.perplexity:
            dominated_dims += 1
        elif other.perplexity == self.perplexity:
            equal_dims += 1
            
        # Speed: higher is better
        if other.speed > self.speed:
            dominated_dims += 1
        elif other.speed == self.speed:
            equal_dims += 1
            
        # Memory: lower is better
        if other.memory_mb < self.memory_mb:
            dominated_dims += 1
        elif other.memory_mb == self.memory_mb:
            equal_dims += 1
        
        # Dominated if other is better in all dimensions
        return dominated_dims == total_dims - equal_dims and equal_dims < total_dims
    
    def __repr__(self) -> str:
        return (
            f"FitnessScore(acc={self.accuracy:.4f}, ppl={self.perplexity:.2f}, "
            f"L={self.lipschitz:.3f}, mem={self.memory_mb:.0f}MB)"
        )


def compute_fitness(
    model,
    train_loader,
    test_loader,
    epochs: int = 10,
    device: str = 'cuda'
) -> FitnessScore:
    """
    Compute fitness score for a model through training and evaluation.
    
    This is a simplified version - the full VariationEvaluator handles
    tiered evaluation and more sophisticated metrics.
    """
    import torch
    import torch.nn as nn
    import time
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Track metrics
    start_time = time.time()
    iterations = 0
    lipschitz_values = []
    
    # Training
    model.train()
    for epoch in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            iterations += 1
        
        # Track Lipschitz after each epoch
        if hasattr(model, 'compute_lipschitz'):
            L = model.compute_lipschitz()
            lipschitz_values.append(float(L))
    
    train_time = time.time() - start_time
    
    # Evaluation
    model.eval()
    correct_train = total_train = 0
    correct_test = total_test = 0
    
    with torch.no_grad():
        # Train accuracy
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=-1)
            correct_train += (pred == y).sum().item()
            total_train += y.size(0)
        
        # Test accuracy  
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=-1)
            correct_test += (pred == y).sum().item()
            total_test += y.size(0)
    
    train_acc = correct_train / total_train if total_train > 0 else 0
    test_acc = correct_test / total_test if total_test > 0 else 0
    
    # Memory tracking
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1e6
    else:
        peak_memory = 0
    
    # Compute generalization gap
    if train_acc > 0:
        gen_gap = 1.0 - abs(train_acc - test_acc) / train_acc
    else:
        gen_gap = 0.0
    
    return FitnessScore(
        accuracy=test_acc,
        train_accuracy=train_acc,
        speed=iterations / train_time if train_time > 0 else 0,
        memory_mb=peak_memory,
        train_time_sec=train_time,
        lipschitz=lipschitz_values[-1] if lipschitz_values else float('inf'),
        lipschitz_trajectory=lipschitz_values,
        generalization=gen_gap,
    )
