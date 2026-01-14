"""
Model Factory

Centralizes model creation logic for Experiment Runner and UI.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any

from bioplausible.models.registry import ModelSpec
from bioplausible.models.base import ModelConfig

# Standard Models
from bioplausible.models.looped_mlp import LoopedMLP, BackpropMLP
from bioplausible.models.backprop_transformer_lm import BackpropTransformerLM
from bioplausible.models.simple_fa import StandardFA
from bioplausible.models.cf_align import ContrastiveFeedbackAlignment
from bioplausible.models.hebbian_chain import DeepHebbianChain
from bioplausible.lm_models import create_eqprop_lm

# Advanced EqProp Models
from bioplausible.models.holomorphic_ep import HolomorphicEP
from bioplausible.models.deep_ep import DirectedEP
from bioplausible.models.finite_nudge_ep import FiniteNudgeEP
from bioplausible.models.modern_conv_eqprop import ModernConvEqProp

# Hybrid / Experimental Models
from bioplausible.models.ada_fa import AdaptiveFeedbackAlignment
from bioplausible.models.eq_align import EquilibriumAlignment
from bioplausible.models.leq_fa import LayerwiseEquilibriumFA
from bioplausible.models.eg_fa import EnergyGuidedFA
from bioplausible.models.pc_hybrid import PredictiveCodingHybrid
from bioplausible.models.sparse_eq import SparseEquilibrium
from bioplausible.models.mom_eq import MomentumEquilibrium
from bioplausible.models.sto_fa import StochasticFA
from bioplausible.models.em_fa import EnergyMinimizingFA


def create_model(
    spec: ModelSpec,
    input_dim: Optional[int],
    output_dim: int,
    hidden_dim: int = 128,
    num_layers: int = 4,
    device: str = "cpu",
    task_type: str = "lm",  # "lm", "vision", "rl"
) -> nn.Module:
    """
    Factory method to create a model instance from a specification.
    """
    model_type = spec.model_type

    # Decide if we need embeddings (LM only, usually)
    # If input_dim is provided, we assume vector input (Vision/RL)
    # Exclude models that handle their own embeddings (Transformers)
    use_embedding = (
        (input_dim is None)
        and (task_type == "lm")
        and (model_type not in ["backprop", "eqprop_transformer"])
    )

    input_size = input_dim if input_dim is not None else hidden_dim

    # Common config creation helper
    def make_config(name):
        return ModelConfig(
            name=name,
            input_dim=input_size,
            output_dim=output_dim,
            hidden_dims=[hidden_dim] * min(num_layers, 5),
            beta=spec.default_beta if spec.has_beta else 0.1,
            learning_rate=spec.default_lr,
            equilibrium_steps=spec.default_steps if spec.has_steps else 20,
            use_spectral_norm=True
        )

    # Embedding wrapper if needed
    embedding_layer = None
    if use_embedding:
        embedding_layer = nn.Embedding(output_dim, hidden_dim).to(device)

    model = None

    if model_type == "backprop":
        if task_type == "lm":
            # Use the robust BackpropTransformerLM
            model = BackpropTransformerLM(
                vocab_size=output_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                max_seq_len=256,
            )
        else:
            # Use BackpropMLP for Vision/RL
            model = BackpropMLP(
                input_dim=input_size,
                hidden_dim=hidden_dim,
                output_dim=output_dim
            )

    elif model_type == "eqprop_transformer":
        model = create_eqprop_lm(
            spec.variant,
            vocab_size=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_sn=True,
        )

    elif model_type == "eqprop_mlp":
        model = LoopedMLP(
            input_dim=input_size,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            use_spectral_norm=True,
        )

    elif model_type == "dfa":
        config = make_config("feedback_alignment")
        model = StandardFA(config=config)

    elif model_type == "chl":
        config = make_config("cf_align")
        model = ContrastiveFeedbackAlignment(config=config)

    elif model_type == "deep_hebbian":
        model = DeepHebbianChain(
            input_dim=input_size,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            use_spectral_norm=True,
            hebbian_lr=0.001,
            use_oja=True
        )

    elif model_type == "holomorphic_ep":
        config = make_config("holomorphic_ep")
        model = HolomorphicEP(config=config, device=device)

    elif model_type == "directed_ep":
        config = make_config("directed_ep")
        model = DirectedEP(config=config, device=device)

    elif model_type == "finite_nudge_ep":
        config = make_config("finite_nudge_ep")
        model = FiniteNudgeEP(config=config, device=device)

    elif model_type == "modern_conv_eqprop":
        # ModernConvEqProp init: (eq_steps, gamma, hidden_channels, use_spectral_norm)
        model = ModernConvEqProp(
            eq_steps=spec.default_steps if spec.has_steps else 30,
            hidden_channels=hidden_dim,
        )

    # --- Hybrid / Experimental Models ---

    elif model_type == "adaptive_feedback_alignment":
        config = make_config("adaptive_feedback_alignment")
        model = AdaptiveFeedbackAlignment(config=config)

    elif model_type == "eq_align":
        model = EquilibriumAlignment(
            input_dim=input_size,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            max_steps=spec.default_steps if spec.has_steps else 30,
            use_spectral_norm=True,
            learning_rate=spec.default_lr
        )

    elif model_type == "layerwise_equilibrium_fa":
        config = make_config("layerwise_equilibrium_fa")
        model = LayerwiseEquilibriumFA(config=config)

    elif model_type == "energy_guided_fa":
        config = make_config("energy_guided_fa")
        model = EnergyGuidedFA(config=config)

    elif model_type == "predictive_coding_hybrid":
        config = make_config("predictive_coding_hybrid")
        model = PredictiveCodingHybrid(config=config)

    elif model_type == "sparse_equilibrium":
        config = make_config("sparse_equilibrium")
        model = SparseEquilibrium(config=config)

    elif model_type == "momentum_equilibrium":
        config = make_config("momentum_equilibrium")
        model = MomentumEquilibrium(config=config)

    elif model_type == "stochastic_fa":
        config = make_config("stochastic_fa")
        model = StochasticFA(config=config)

    elif model_type == "energy_minimizing_fa":
        config = make_config("energy_minimizing_fa")
        model = EnergyMinimizingFA(config=config)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Attach embedding if created and not already integrated
    # Some models might handle embedding internally, but here we assume if we created it, we attach it.
    # Note: The caller needs to know if 'embedding_layer' was created.
    # But standard models don't have an 'embed' attribute by default.
    # We can attach it to the model instance.
    if embedding_layer is not None:
        model.embed = embedding_layer
        model.has_embed = True
    else:
        model.has_embed = False

    return model.to(device)
