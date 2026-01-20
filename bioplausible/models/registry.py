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
    task_compat: Optional[List[str]] = None  # ['vision', 'lm', 'rl'] or None for all applicable


# All models available - ordered by category
MODEL_REGISTRY = [
    # Baselines
    ModelSpec(
        name="Backprop Baseline",
        description="Standard Backprop baseline (Transformer/MLP)",
        model_type="backprop",
        default_lr=0.001,
        color="#ff6b6b",
        task_compat=["vision", "lm", "rl"],
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
        task_compat=["vision", "rl"],
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
        task_compat=["vision", "rl"],
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
        task_compat=["vision", "rl"],
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
        task_compat=["vision", "rl"],
    ),
    ModelSpec(
        name="Conv EqProp (CIFAR-10)",
        description="Convolutional EqProp optimized for CIFAR-10",
        model_type="modern_conv_eqprop",
        default_lr=0.0005,
        default_steps=15,
        has_steps=True,
        color="#26de81",
        task_compat=["vision"],
    ),
    ModelSpec(
        name="EqProp Diffusion",
        description="Generative Diffusion via Equilibrium Propagation",
        model_type="eqprop_diffusion",
        default_lr=0.001,
        color="#fdcb6e",
        task_compat=["vision"],
    ),
    # Hybrid & Experimental Algorithms
    ModelSpec(
        name="Adaptive Feedback Alignment",
        description="FA with slowly adapting feedback weights",
        model_type="adaptive_feedback_alignment",
        default_lr=0.001,
        color="#4b7bec",
        task_compat=["vision", "rl"],
    ),
    ModelSpec(
        name="Equilibrium Alignment",
        description="EqProp dynamics + Feedback Alignment",
        model_type="eq_align",
        default_lr=0.001,
        default_steps=30,
        has_steps=True,
        color="#d1d8e0",
        task_compat=["vision", "rl"],
    ),
    ModelSpec(
        name="Layerwise Equilibrium FA",
        description="Layerwise training with EqProp/FA hybrid",
        model_type="layerwise_equilibrium_fa",
        default_lr=0.001,
        color="#a5b1c2",
        task_compat=["vision", "rl"],
    ),
    ModelSpec(
        name="Energy Guided FA",
        description="Feedback Alignment guided by Energy Function",
        model_type="energy_guided_fa",
        default_lr=0.001,
        color="#778ca3",
        task_compat=["vision", "rl"],
    ),
    ModelSpec(
        name="Predictive Coding Hybrid",
        description="Hybrid of Predictive Coding and EqProp",
        model_type="predictive_coding_hybrid",
        default_lr=0.001,
        default_steps=20,
        has_steps=True,
        color="#3867d6",
        task_compat=["vision", "rl"],
    ),
    ModelSpec(
        name="Sparse Equilibrium",
        description="EqProp with sparsity constraints",
        model_type="sparse_equilibrium",
        default_lr=0.001,
        default_beta=0.1,
        has_beta=True,
        color="#8854d0",
        task_compat=["vision", "rl"],
    ),
    ModelSpec(
        name="Momentum Equilibrium",
        description="EqProp with momentum dynamics",
        model_type="momentum_equilibrium",
        default_lr=0.001,
        color="#45aaf2",
        task_compat=["vision", "rl"],
    ),
    ModelSpec(
        name="Stochastic FA",
        description="Feedback Alignment with stochastic weights",
        model_type="stochastic_fa",
        default_lr=0.001,
        color="#2bcbba",
        task_compat=["vision", "rl"],
    ),
    ModelSpec(
        name="Energy Minimizing FA",
        description="FA variant that minimizes local energy",
        model_type="energy_minimizing_fa",
        default_lr=0.001,
        color="#0fb9b1",
        task_compat=["vision", "rl"],
    ),
    # Other Bio-Plausible Algorithms
    ModelSpec(
        name="DFA (Direct Feedback Alignment)",
        description="Random feedback weights",
        model_type="dfa",
        default_lr=0.001,
        color="#45b7d1",
        task_compat=["vision", "rl"],
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
        task_compat=["vision", "rl"],
    ),
    ModelSpec(
        name="Deep Hebbian (Hundred-Layer)",
        description="100-layer Hebbian chain with SN",
        model_type="deep_hebbian",
        default_lr=0.0005,
        color="#6c5ce7",
        task_compat=["vision", "rl"],
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
        task_compat=["lm"],
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
        task_compat=["lm"],
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
        task_compat=["lm"],
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
        task_compat=["lm"],
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
