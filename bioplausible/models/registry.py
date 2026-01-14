"""
Model Registry for Bio-Plausible Algorithms

Defines specifications for available models and algorithms, used by experiments and UI.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelSpec:
    """Specification for a model."""

    name: str  # Display name
    description: str  # Short description
    model_type: str  # Internal type key (mapped to model class)
    variant: Optional[str] = None  # Variant for transformer models
    default_lr: float = 0.001
    default_beta: float = 0.22
    default_steps: int = 30
    has_beta: bool = False
    has_steps: bool = False
    color: str = "#888888"


# All models available - ordered by category
MODEL_REGISTRY = [
    # Baselines
    ModelSpec(
        name="Backprop Baseline",
        description="Standard Backprop baseline (Transformer/MLP)",
        model_type="backprop",
        default_lr=0.001,
        color="#ff6b6b",
    ),
    # EqProp MLP
    ModelSpec(
        name="EqProp MLP",
        description="Looped MLP with Spectral Norm",
        model_type="eqprop_mlp",
        default_lr=0.001,
        default_beta=0.22,
        default_steps=30,
        has_beta=True,
        has_steps=True,
        color="#4ecdc4",
    ),
    # Advanced EqProp Variants
    ModelSpec(
        name="Holomorphic EqProp",
        description="Complex-valued Equilibrium Propagation",
        model_type="holomorphic_ep",
        default_lr=0.001,
        default_beta=0.1,
        default_steps=30,
        has_beta=True,
        has_steps=True,
        color="#a55eea",
    ),
    ModelSpec(
        name="Directed EqProp (Deep EP)",
        description="Asymmetric forward and feedback weights",
        model_type="directed_ep",
        default_lr=0.001,
        default_beta=0.2,
        default_steps=30,
        has_beta=True,
        has_steps=True,
        color="#fd9644",
    ),
    ModelSpec(
        name="Finite-Nudge EqProp",
        description="EqProp with large beta (finite difference)",
        model_type="finite_nudge_ep",
        default_lr=0.001,
        default_beta=1.0,
        default_steps=30,
        has_beta=True,
        has_steps=True,
        color="#fc5c65",
    ),
    ModelSpec(
        name="Conv EqProp (CIFAR-10)",
        description="Convolutional EqProp optimized for CIFAR-10",
        model_type="modern_conv_eqprop",
        default_lr=0.0005,
        default_steps=15,
        has_steps=True,
        color="#26de81",
    ),
    # Other Bio-Plausible Algorithms
    ModelSpec(
        name="DFA (Direct Feedback Alignment)",
        description="Random feedback weights",
        model_type="dfa",
        default_lr=0.001,
        color="#45b7d1",
    ),
    ModelSpec(
        name="CHL (Contrastive Hebbian)",
        description="Contrastive Hebbian Learning",
        model_type="chl",
        default_lr=0.001,
        default_beta=0.1,
        default_steps=20,
        has_beta=True,
        has_steps=True,
        color="#f9ca24",
    ),
    ModelSpec(
        name="Deep Hebbian (Hundred-Layer)",
        description="100-layer Hebbian chain with SN",
        model_type="deep_hebbian",
        default_lr=0.0005,
        color="#6c5ce7",
    ),
    # EqProp Transformers (From Track 37 results) - SLOW MODELS LAST
    ModelSpec(
        name="EqProp Transformer (Attention Only)",
        description="Best variant: EqProp in attention only",
        model_type="eqprop_transformer",
        variant="attention_only",
        default_lr=0.0003,
        default_steps=10,
        has_steps=True,
        color="#2ecc71",
    ),
    ModelSpec(
        name="EqProp Transformer (Full)",
        description="All layers use equilibrium",
        model_type="eqprop_transformer",
        variant="full",
        default_lr=0.0003,
        default_steps=15,
        has_steps=True,
        color="#27ae60",
    ),
    ModelSpec(
        name="EqProp Transformer (Hybrid)",
        description="Standard layers + EqProp final layer",
        model_type="eqprop_transformer",
        variant="hybrid",
        default_lr=0.0003,
        default_steps=10,
        has_steps=True,
        color="#1abc9c",
    ),
    ModelSpec(
        name="EqProp Transformer (Recurrent)",
        description="Single recurrent block, parameter efficient",
        model_type="eqprop_transformer",
        variant="recurrent_core",
        default_lr=0.0003,
        default_steps=20,
        has_steps=True,
        color="#16a085",
    ),
]


def get_model_spec(name: str) -> ModelSpec:
    """Get model spec by name."""
    for spec in MODEL_REGISTRY:
        if spec.name == name:
            return spec
    raise ValueError(f"Unknown model: {name}")


def list_model_names() -> List[str]:
    """List all available model names."""
    return [spec.name for spec in MODEL_REGISTRY]
