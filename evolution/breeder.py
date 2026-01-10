"""
Variation Breeder for EqProp+SN Architecture Evolution

Implements genetic operations for evolving model architectures:
- Architectural mutations (depth, width, activations)
- Connection topology (residual, skip connections)
- SN parameter tuning (n_power_iterations)
- Crossover between successful variants
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import copy


@dataclass
class ArchConfig:
    """Architecture configuration for an EqProp+SN variant."""
    
    # Core architecture
    model_type: str = "looped_mlp"  # looped_mlp, conv, transformer, hebbian
    depth: int = 10
    width: int = 256
    
    # Activation and normalization
    activation: str = "tanh"  # tanh, gelu, relu, silu
    normalization: str = "spectral"  # spectral, layernorm, groupnorm, none
    
    # EqProp parameters
    eq_steps: int = 30
    beta: float = 0.22
    alpha: float = 0.5  # Mixing coefficient for equilibrium update
    
    # Spectral normalization parameters
    use_sn: bool = True
    n_power_iterations: int = 1
    
    # Residual connections
    use_residual: bool = False
    residual_scale: float = 0.5
    
    # Training parameters
    lr: float = 0.001
    
    # For transformers
    num_heads: int = 4
    
    # Metadata
    generation: int = 0
    parent_ids: List[int] = field(default_factory=list)
    mutations_applied: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ArchConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    def clone(self) -> 'ArchConfig':
        """Create a deep copy."""
        return copy.deepcopy(self)


class VariationBreeder:
    """Genetic operations for architecture evolution."""
    
    # Valid values for categorical parameters
    MODEL_TYPES = ["looped_mlp", "conv", "transformer", "hebbian", "feedback_alignment"]
    ACTIVATIONS = ["tanh", "gelu", "relu", "silu"]
    NORMALIZATIONS = ["spectral", "layernorm", "groupnorm", "none"]
    
    # Parameter ranges
    RANGES = {
        "depth": (2, 100),
        "width": (32, 1024),
        "eq_steps": (5, 100),
        "beta": (0.05, 0.5),
        "alpha": (0.1, 0.9),
        "n_power_iterations": (1, 10),
        "residual_scale": (0.0, 1.0),
        "lr": (1e-5, 1e-2),
        "num_heads": (1, 16),
    }
    
    def __init__(
        self,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
        seed: int = 42,
    ):
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.rng = np.random.default_rng(seed)
    
    def mutate(self, config: ArchConfig) -> ArchConfig:
        """Apply random mutations to architecture config."""
        child = config.clone()
        child.mutations_applied = []
        child.generation = config.generation + 1
        child.parent_ids = [id(config)]
        
        # Structural mutations
        if self.rng.random() < self.mutation_rate:
            child.depth = self._mutate_int(child.depth, *self.RANGES["depth"])
            child.mutations_applied.append(f"depth→{child.depth}")
        
        if self.rng.random() < self.mutation_rate:
            child.width = self._mutate_int(child.width, *self.RANGES["width"])
            child.mutations_applied.append(f"width→{child.width}")
        
        # Activation mutation
        if self.rng.random() < self.mutation_rate * 0.5:
            child.activation = self.rng.choice([a for a in self.ACTIVATIONS if a != child.activation])
            child.mutations_applied.append(f"activation→{child.activation}")
        
        # EqProp parameter mutations
        if self.rng.random() < self.mutation_rate:
            child.eq_steps = self._mutate_int(child.eq_steps, *self.RANGES["eq_steps"])
            child.mutations_applied.append(f"eq_steps→{child.eq_steps}")
        
        if self.rng.random() < self.mutation_rate:
            child.beta = self._mutate_float(child.beta, *self.RANGES["beta"])
            child.mutations_applied.append(f"beta→{child.beta:.3f}")
        
        if self.rng.random() < self.mutation_rate:
            child.alpha = self._mutate_float(child.alpha, *self.RANGES["alpha"])
            child.mutations_applied.append(f"alpha→{child.alpha:.2f}")
        
        # SN parameter mutations
        if self.rng.random() < self.mutation_rate:
            child.n_power_iterations = self._mutate_int(
                child.n_power_iterations, 
                *self.RANGES["n_power_iterations"]
            )
            child.mutations_applied.append(f"n_power_iter→{child.n_power_iterations}")
        
        # Toggle residual connections
        if self.rng.random() < self.mutation_rate * 0.3:
            child.use_residual = not child.use_residual
            child.mutations_applied.append(f"residual→{child.use_residual}")
        
        if child.use_residual and self.rng.random() < self.mutation_rate:
            child.residual_scale = self._mutate_float(
                child.residual_scale, 
                *self.RANGES["residual_scale"]
            )
            child.mutations_applied.append(f"res_scale→{child.residual_scale:.2f}")
        
        # Learning rate mutation (log scale)
        if self.rng.random() < self.mutation_rate:
            child.lr = self._mutate_log(child.lr, *self.RANGES["lr"])
            child.mutations_applied.append(f"lr→{child.lr:.6f}")
        
        return child
    
    def mutate_architecture_type(self, config: ArchConfig) -> ArchConfig:
        """More aggressive mutation that can change model type."""
        child = self.mutate(config)
        
        if self.rng.random() < self.mutation_rate * 0.2:
            child.model_type = self.rng.choice([t for t in self.MODEL_TYPES if t != child.model_type])
            child.mutations_applied.append(f"model_type→{child.model_type}")
            
            # Adjust parameters for new architecture
            if child.model_type == "transformer":
                child.depth = min(12, child.depth)  # Transformers don't go as deep
                child.num_heads = self._ensure_divides(child.num_heads, child.width)
            elif child.model_type == "hebbian":
                child.use_sn = True  # SN is essential for deep Hebbian
                child.depth = max(50, child.depth)  # Hebbian scales deep
        
        return child
    
    def crossover(self, parent1: ArchConfig, parent2: ArchConfig) -> ArchConfig:
        """Create child by crossing over two parent configurations."""
        child = ArchConfig()
        child.generation = max(parent1.generation, parent2.generation) + 1
        child.parent_ids = [id(parent1), id(parent2)]
        child.mutations_applied = ["crossover"]
        
        # Uniform crossover for each parameter
        for field_name in ArchConfig.__dataclass_fields__:
            if field_name in ['generation', 'parent_ids', 'mutations_applied']:
                continue
            
            p1_val = getattr(parent1, field_name)
            p2_val = getattr(parent2, field_name)
            
            if self.rng.random() < 0.5:
                setattr(child, field_name, p1_val)
            else:
                setattr(child, field_name, p2_val)
        
        # Ensure valid configuration
        self._validate_config(child)
        
        return child
    
    def crossover_blend(self, parent1: ArchConfig, parent2: ArchConfig) -> ArchConfig:
        """Blend crossover - average numeric values."""
        child = self.crossover(parent1, parent2)
        
        # Blend numeric parameters
        for field_name in ['depth', 'width', 'eq_steps', 'beta', 'alpha', 'lr', 'residual_scale']:
            p1_val = getattr(parent1, field_name)
            p2_val = getattr(parent2, field_name)
            
            if isinstance(p1_val, int):
                blended = int((p1_val + p2_val) / 2)
            else:
                blended = (p1_val + p2_val) / 2
            
            setattr(child, field_name, blended)
        
        child.mutations_applied.append("blend")
        return child
    
    def generate_random(self) -> ArchConfig:
        """Generate a random architecture configuration."""
        return ArchConfig(
            model_type=self.rng.choice(self.MODEL_TYPES),
            depth=int(self.rng.integers(*self.RANGES["depth"])),
            width=int(self.rng.choice([64, 128, 256, 512])),
            activation=self.rng.choice(self.ACTIVATIONS),
            normalization=self.rng.choice(self.NORMALIZATIONS),
            eq_steps=int(self.rng.integers(*self.RANGES["eq_steps"])),
            beta=float(self.rng.uniform(*self.RANGES["beta"])),
            alpha=float(self.rng.uniform(*self.RANGES["alpha"])),
            use_sn=bool(self.rng.random() > 0.1),  # 90% chance of SN
            n_power_iterations=int(self.rng.integers(*self.RANGES["n_power_iterations"])),
            use_residual=bool(self.rng.random() > 0.5),
            residual_scale=float(self.rng.uniform(*self.RANGES["residual_scale"])),
            lr=float(np.exp(self.rng.uniform(np.log(1e-5), np.log(1e-2)))),
            num_heads=int(self.rng.choice([1, 2, 4, 8])),
            generation=0,
        )
    
    def _mutate_int(self, value: int, min_val: int, max_val: int) -> int:
        """Mutate integer parameter."""
        delta = int(self.rng.normal(0, (max_val - min_val) * 0.1))
        return int(np.clip(value + delta, min_val, max_val))
    
    def _mutate_float(self, value: float, min_val: float, max_val: float) -> float:
        """Mutate float parameter."""
        delta = self.rng.normal(0, (max_val - min_val) * 0.1)
        return float(np.clip(value + delta, min_val, max_val))
    
    def _mutate_log(self, value: float, min_val: float, max_val: float) -> float:
        """Mutate in log space for learning rate."""
        log_val = np.log(value) + self.rng.normal(0, 0.5)
        return float(np.clip(np.exp(log_val), min_val, max_val))
    
    def _ensure_divides(self, num_heads: int, width: int) -> int:
        """Ensure num_heads divides width evenly."""
        while width % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        return num_heads
    
    def _validate_config(self, config: ArchConfig) -> None:
        """Validate and fix configuration."""
        # Ensure valid ranges
        config.depth = int(np.clip(config.depth, *self.RANGES["depth"]))
        config.width = int(np.clip(config.width, *self.RANGES["width"]))
        config.eq_steps = int(np.clip(config.eq_steps, *self.RANGES["eq_steps"]))
        config.beta = float(np.clip(config.beta, *self.RANGES["beta"]))
        config.alpha = float(np.clip(config.alpha, *self.RANGES["alpha"]))
        config.lr = float(np.clip(config.lr, *self.RANGES["lr"]))
        
        # Transformer-specific fixes
        if config.model_type == "transformer":
            config.num_heads = self._ensure_divides(config.num_heads, config.width)
        
        # Hebbian requires SN
        if config.model_type == "hebbian":
            config.use_sn = True
