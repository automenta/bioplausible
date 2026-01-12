"""
Algorithm Genome - Evolvable EqProp Algorithm Variants

Defines the genetic encoding for evolving EqProp algorithm variations:
- Update rules (contrastive Hebbian, nudging, etc.)
- Equilibrium dynamics (fixed-point, damped oscillation, etc.)
- Gradient approximations (symmetric, finite-difference, etc.)
- Spectral normalization strategies
- Hybrid combinations from bio-plausible algorithms

Key Insight: SN enables stability for ANY contraction-based learning rule.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Callable, Tuple
from enum import Enum
import numpy as np
import copy


class UpdateRule(Enum):
    """Types of synaptic update rules."""
    CONTRASTIVE_HEBBIAN = "contrastive_hebbian"  # Classic: Δw ∝ h_nudged⊗h_nudged - h_free⊗h_free
    SYMMETRIC_DIFF = "symmetric_diff"  # Δw ∝ (h_nudged - h_free) ⊗ (pre_nudged - pre_free)
    ASYMMETRIC_NUDGE = "asymmetric_nudge"  # Only nudge output, backprop-like
    FINITE_DIFF = "finite_diff"  # Δw ∝ (L(β) - L(-β)) / 2β
    LOCAL_ENERGY = "local_energy"  # Δw ∝ -∂E_local/∂w
    PREDICTIVE_CODING = "predictive_coding"  # Minimize prediction error


class EquilibriumDynamics(Enum):
    """Types of equilibrium-seeking dynamics."""
    FIXED_POINT = "fixed_point"  # Standard iteration to fixed point
    DAMPED_OSCILLATION = "damped_oscillation"  # Allow oscillation with damping
    MOMENTUM = "momentum"  # Add momentum to settling
    ADAPTIVE_STEP = "adaptive_step"  # Adaptive step size based on error
    SYNCHRONOUS = "synchronous"  # All neurons update simultaneously
    ASYNCHRONOUS = "asynchronous"  # Random neuron updates


class GradientApprox(Enum):
    """Gradient approximation methods."""
    BPTT = "bptt"  # Backprop through time (exact but not bio-plausible)
    SYMMETRIC = "symmetric"  # Symmetric nudging (β, -β)
    ONE_SIDED = "one_sided"  # Single nudge direction
    RANDOM_FEEDBACK = "random_feedback"  # Feedback alignment
    DIRECT_FEEDBACK = "direct_feedback"  # Direct feedback alignment
    WEIGHT_PERTURBATION = "weight_perturbation"  # Random weight perturbation


class SNStrategy(Enum):
    """Spectral normalization application strategies."""
    ALL_LAYERS = "all_layers"  # Apply SN to all layers
    RECURRENT_ONLY = "recurrent_only"  # Only recurrent connections
    ADAPTIVE = "adaptive"  # Adjust SN strength during training
    LAYERWISE = "layerwise"  # Different SN per layer
    SOFT = "soft"  # Soft constraint (penalize instead of enforce)


class ActivationFunction(Enum):
    """Activation functions."""
    TANH = "tanh"
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"
    SOFTPLUS = "softplus"
    SIGMOID = "sigmoid"


@dataclass
class AlgorithmConfig:
    """
    Genetic encoding for an EqProp algorithm variant.
    
    This captures the fundamental algorithmic choices, not just hyperparameters.
    """
    # Core algorithm components
    update_rule: UpdateRule = UpdateRule.CONTRASTIVE_HEBBIAN
    equilibrium_dynamics: EquilibriumDynamics = EquilibriumDynamics.FIXED_POINT
    gradient_approx: GradientApprox = GradientApprox.SYMMETRIC
    sn_strategy: SNStrategy = SNStrategy.ALL_LAYERS
    activation: ActivationFunction = ActivationFunction.TANH
    
    # Equilibrium parameters
    eq_steps: int = 30
    beta: float = 0.22  # Nudge strength
    alpha: float = 0.5  # Mixing coefficient
    
    # Momentum/dynamics parameters
    momentum: float = 0.0  # For momentum dynamics
    damping: float = 0.9  # For damped oscillation
    
    # SN parameters
    use_sn: bool = True
    sn_strength: float = 1.0  # For soft SN: constraint strength
    n_power_iterations: int = 1
    
    # Gradient approximation parameters
    beta_negative: bool = True  # Use symmetric +/- beta
    perturbation_scale: float = 0.01  # For weight perturbation
    
    # Hybrid features (combine elements from different algorithms)
    use_local_energy: bool = False
    use_lateral_inhibition: bool = False
    use_homeostatic_plasticity: bool = False
    
    # Architecture-level algorithm choices
    bidirectional: bool = False  # Bidirectional connections
    skip_connections: bool = False
    layer_sharing: bool = False  # Share weights across equilibrium steps
    
    # Metadata
    generation: int = 0
    parent_ids: List[int] = field(default_factory=list)
    mutations_applied: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling enums."""
        d = asdict(self)
        # Convert enums to strings
        for key, val in d.items():
            if isinstance(val, Enum):
                d[key] = val.value
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'AlgorithmConfig':
        """Create from dictionary."""
        # Convert string values back to enums
        enum_fields = {
            'update_rule': UpdateRule,
            'equilibrium_dynamics': EquilibriumDynamics,
            'gradient_approx': GradientApprox,
            'sn_strategy': SNStrategy,
            'activation': ActivationFunction,
        }
        for field_name, enum_class in enum_fields.items():
            if field_name in d and isinstance(d[field_name], str):
                d[field_name] = enum_class(d[field_name])
        
        # Filter to valid fields
        valid_fields = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)
    
    def clone(self) -> 'AlgorithmConfig':
        """Create a deep copy."""
        return copy.deepcopy(self)
    
    def describe(self) -> str:
        """Human-readable description of this algorithm variant."""
        parts = [
            f"Update: {self.update_rule.value}",
            f"Dynamics: {self.equilibrium_dynamics.value}",
            f"Gradient: {self.gradient_approx.value}",
            f"SN: {self.sn_strategy.value}" if self.use_sn else "No SN",
        ]
        if self.use_homeostatic_plasticity:
            parts.append("+Homeostatic")
        if self.use_lateral_inhibition:
            parts.append("+LateralInhib")
        return " | ".join(parts)


class AlgorithmBreeder:
    """Genetic operations for algorithm evolution."""
    
    # All possible values for each enum
    UPDATE_RULES = list(UpdateRule)
    DYNAMICS = list(EquilibriumDynamics)
    GRADIENT_APPROX = list(GradientApprox)
    SN_STRATEGIES = list(SNStrategy)
    ACTIVATIONS = list(ActivationFunction)
    
    def __init__(
        self,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.6,
        seed: int = 42,
    ):
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.rng = np.random.default_rng(seed)
    
    def mutate(self, config: AlgorithmConfig) -> AlgorithmConfig:
        """Apply random mutations to algorithm config."""
        child = config.clone()
        child.mutations_applied = []
        child.generation = config.generation + 1
        child.parent_ids = [id(config)]
        
        # Core algorithm mutations (lower probability - these are significant changes)
        if self.rng.random() < self.mutation_rate * 0.5:
            child.update_rule = self.rng.choice([r for r in self.UPDATE_RULES if r != child.update_rule])
            child.mutations_applied.append(f"update_rule→{child.update_rule.value}")
        
        if self.rng.random() < self.mutation_rate * 0.5:
            child.equilibrium_dynamics = self.rng.choice([d for d in self.DYNAMICS if d != child.equilibrium_dynamics])
            child.mutations_applied.append(f"dynamics→{child.equilibrium_dynamics.value}")
        
        if self.rng.random() < self.mutation_rate * 0.5:
            child.gradient_approx = self.rng.choice([g for g in self.GRADIENT_APPROX if g != child.gradient_approx])
            child.mutations_applied.append(f"gradient→{child.gradient_approx.value}")
        
        if self.rng.random() < self.mutation_rate * 0.5:
            child.sn_strategy = self.rng.choice(self.SN_STRATEGIES)
            child.mutations_applied.append(f"sn_strategy→{child.sn_strategy.value}")
        
        # Activation mutation
        if self.rng.random() < self.mutation_rate:
            child.activation = self.rng.choice(self.ACTIVATIONS)
            child.mutations_applied.append(f"activation→{child.activation.value}")
        
        # Parameter mutations
        if self.rng.random() < self.mutation_rate:
            child.eq_steps = int(np.clip(child.eq_steps + self.rng.integers(-10, 11), 5, 100))
            child.mutations_applied.append(f"eq_steps→{child.eq_steps}")
        
        if self.rng.random() < self.mutation_rate:
            child.beta = float(np.clip(child.beta + self.rng.normal(0, 0.05), 0.01, 0.5))
            child.mutations_applied.append(f"beta→{child.beta:.3f}")
        
        if self.rng.random() < self.mutation_rate:
            child.alpha = float(np.clip(child.alpha + self.rng.normal(0, 0.1), 0.1, 0.95))
            child.mutations_applied.append(f"alpha→{child.alpha:.2f}")
        
        # Toggle hybrid features
        if self.rng.random() < self.mutation_rate * 0.3:
            child.use_homeostatic_plasticity = not child.use_homeostatic_plasticity
            child.mutations_applied.append(f"homeostatic→{child.use_homeostatic_plasticity}")
        
        if self.rng.random() < self.mutation_rate * 0.3:
            child.use_lateral_inhibition = not child.use_lateral_inhibition
            child.mutations_applied.append(f"lateral_inhib→{child.use_lateral_inhibition}")
        
        if self.rng.random() < self.mutation_rate * 0.3:
            child.bidirectional = not child.bidirectional
            child.mutations_applied.append(f"bidirectional→{child.bidirectional}")
        
        # Dynamics-specific parameters
        if child.equilibrium_dynamics == EquilibriumDynamics.MOMENTUM:
            if self.rng.random() < self.mutation_rate:
                child.momentum = float(np.clip(child.momentum + self.rng.normal(0, 0.1), 0, 0.99))
                child.mutations_applied.append(f"momentum→{child.momentum:.2f}")
        
        if child.equilibrium_dynamics == EquilibriumDynamics.DAMPED_OSCILLATION:
            if self.rng.random() < self.mutation_rate:
                child.damping = float(np.clip(child.damping + self.rng.normal(0, 0.05), 0.5, 0.99))
                child.mutations_applied.append(f"damping→{child.damping:.2f}")
        
        return child
    
    def crossover(self, p1: AlgorithmConfig, p2: AlgorithmConfig) -> AlgorithmConfig:
        """Create child by combining two parent algorithms."""
        child = AlgorithmConfig()
        child.generation = max(p1.generation, p2.generation) + 1
        child.parent_ids = [id(p1), id(p2)]
        child.mutations_applied = ["crossover"]
        
        # Randomly inherit each component
        child.update_rule = self.rng.choice([p1.update_rule, p2.update_rule])
        child.equilibrium_dynamics = self.rng.choice([p1.equilibrium_dynamics, p2.equilibrium_dynamics])
        child.gradient_approx = self.rng.choice([p1.gradient_approx, p2.gradient_approx])
        child.sn_strategy = self.rng.choice([p1.sn_strategy, p2.sn_strategy])
        child.activation = self.rng.choice([p1.activation, p2.activation])
        
        # Blend numeric parameters
        child.eq_steps = int((p1.eq_steps + p2.eq_steps) / 2)
        child.beta = (p1.beta + p2.beta) / 2
        child.alpha = (p1.alpha + p2.alpha) / 2
        child.momentum = (p1.momentum + p2.momentum) / 2
        child.damping = (p1.damping + p2.damping) / 2
        
        # Inherit hybrid features from either parent
        child.use_homeostatic_plasticity = self.rng.choice([p1.use_homeostatic_plasticity, p2.use_homeostatic_plasticity])
        child.use_lateral_inhibition = self.rng.choice([p1.use_lateral_inhibition, p2.use_lateral_inhibition])
        child.bidirectional = self.rng.choice([p1.bidirectional, p2.bidirectional])
        
        return child
    
    def generate_random(self) -> AlgorithmConfig:
        """Generate a random algorithm configuration."""
        return AlgorithmConfig(
            update_rule=self.rng.choice(self.UPDATE_RULES),
            equilibrium_dynamics=self.rng.choice(self.DYNAMICS),
            gradient_approx=self.rng.choice(self.GRADIENT_APPROX),
            sn_strategy=self.rng.choice(self.SN_STRATEGIES),
            activation=self.rng.choice(self.ACTIVATIONS),
            eq_steps=int(self.rng.integers(10, 50)),
            beta=float(self.rng.uniform(0.05, 0.4)),
            alpha=float(self.rng.uniform(0.2, 0.8)),
            momentum=float(self.rng.uniform(0, 0.9)) if self.rng.random() > 0.5 else 0.0,
            damping=float(self.rng.uniform(0.7, 0.99)),
            use_sn=True,  # Always use SN (our key insight)
            sn_strength=float(self.rng.uniform(0.8, 1.2)),
            use_homeostatic_plasticity=bool(self.rng.random() > 0.7),
            use_lateral_inhibition=bool(self.rng.random() > 0.8),
            bidirectional=bool(self.rng.random() > 0.7),
            generation=0,
        )
    
    def generate_informed(self) -> AlgorithmConfig:
        """
        Generate algorithm config informed by our validated insights:
        1. SN is essential (L < 1)
        2. Contrastive Hebbian works well
        3. Homeostatic plasticity helps stability
        """
        # Start with known-good base
        config = AlgorithmConfig(
            update_rule=self.rng.choice([
                UpdateRule.CONTRASTIVE_HEBBIAN,
                UpdateRule.SYMMETRIC_DIFF,
                UpdateRule.LOCAL_ENERGY,
            ]),
            equilibrium_dynamics=self.rng.choice([
                EquilibriumDynamics.FIXED_POINT,
                EquilibriumDynamics.MOMENTUM,
            ]),
            gradient_approx=self.rng.choice([
                GradientApprox.SYMMETRIC,
                GradientApprox.ONE_SIDED,
            ]),
            sn_strategy=SNStrategy.ALL_LAYERS,  # Most reliable
            activation=self.rng.choice([
                ActivationFunction.TANH,  # Bounded, works best
                ActivationFunction.GELU,
            ]),
            eq_steps=int(self.rng.integers(15, 40)),
            beta=float(self.rng.uniform(0.15, 0.3)),
            alpha=float(self.rng.uniform(0.4, 0.6)),
            use_sn=True,
            use_homeostatic_plasticity=bool(self.rng.random() > 0.5),
        )
        return config


# Pre-defined algorithm presets based on research insights
ALGORITHM_PRESETS = {
    "classic_eqprop": AlgorithmConfig(
        update_rule=UpdateRule.CONTRASTIVE_HEBBIAN,
        equilibrium_dynamics=EquilibriumDynamics.FIXED_POINT,
        gradient_approx=GradientApprox.SYMMETRIC,
        sn_strategy=SNStrategy.ALL_LAYERS,
        activation=ActivationFunction.TANH,
    ),
    
    "momentum_eqprop": AlgorithmConfig(
        update_rule=UpdateRule.CONTRASTIVE_HEBBIAN,
        equilibrium_dynamics=EquilibriumDynamics.MOMENTUM,
        gradient_approx=GradientApprox.SYMMETRIC,
        sn_strategy=SNStrategy.ALL_LAYERS,
        activation=ActivationFunction.TANH,
        momentum=0.9,
    ),
    
    "predictive_coding_hybrid": AlgorithmConfig(
        update_rule=UpdateRule.PREDICTIVE_CODING,
        equilibrium_dynamics=EquilibriumDynamics.FIXED_POINT,
        gradient_approx=GradientApprox.ONE_SIDED,
        sn_strategy=SNStrategy.ALL_LAYERS,
        activation=ActivationFunction.TANH,
        use_local_energy=True,
    ),
    
    "feedback_alignment_eqprop": AlgorithmConfig(
        update_rule=UpdateRule.CONTRASTIVE_HEBBIAN,
        equilibrium_dynamics=EquilibriumDynamics.FIXED_POINT,
        gradient_approx=GradientApprox.RANDOM_FEEDBACK,
        sn_strategy=SNStrategy.ALL_LAYERS,
        activation=ActivationFunction.TANH,
    ),
    
    "homeostatic_eqprop": AlgorithmConfig(
        update_rule=UpdateRule.CONTRASTIVE_HEBBIAN,
        equilibrium_dynamics=EquilibriumDynamics.FIXED_POINT,
        gradient_approx=GradientApprox.SYMMETRIC,
        sn_strategy=SNStrategy.ALL_LAYERS,
        activation=ActivationFunction.TANH,
        use_homeostatic_plasticity=True,
        use_lateral_inhibition=True,
    ),
    
    "adaptive_sn_eqprop": AlgorithmConfig(
        update_rule=UpdateRule.CONTRASTIVE_HEBBIAN,
        equilibrium_dynamics=EquilibriumDynamics.ADAPTIVE_STEP,
        gradient_approx=GradientApprox.SYMMETRIC,
        sn_strategy=SNStrategy.ADAPTIVE,
        activation=ActivationFunction.GELU,
    ),
}
