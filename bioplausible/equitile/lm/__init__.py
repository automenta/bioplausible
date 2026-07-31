"""
EquiTile Fast LM
================

High-performance language modeling showcasing EquiTile's unique advantages:
- Mixture of Tiles (MoT) for conditional computation
- Tile-local attention for O(n) complexity
- Parameter efficiency (< 10M parameters)
- Fast training on commodity GPUs

Quick Start
-----------
>>> from bioplausible.equitile.lm import FastLMEquiTile, FastLMConfig
>>> config = FastLMConfig(vocab_size=1000, embed_dim=192, num_layers=6)
>>> model = FastLMEquiTile(config)
>>> logits = model(input_ids)

Demo Usage
----------
$ python -m bioplausible.equitile.lm.demo --task shakespeare --epochs 5
"""

from .components import (
    FastEquiTileLayer,
    FastLMConfig,
    MixtureOfTiles,
    SwiGLUFeedForward,
    TileLocalAttention,
)
from .data import (
    CharacterTokenizer,
    LMDataset,
    Tokenizer,
    create_python_dataset,
    create_shakespeare_dataset,
    create_tinystories_dataset,
)
from .data_advanced import (
    BPETokenizer,
    WordPieceTokenizer,
    create_tokenizer,
    load_shakespeare_tokenizer,
)
from .fast_lm import (
    FastLMEquiTile,
    create_fast_lm_medium,
    create_fast_lm_shakespeare,
    create_fast_lm_small,
    create_fast_lm_tiny,
)
from .training import LMTrainer, TrainingConfig, TrainingMetrics, train_model

__all__ = [
    # Model
    "FastLMEquiTile",
    "FastLMConfig",
    "FastEquiTileLayer",
    "MixtureOfTiles",
    "TileLocalAttention",
    "SwiGLUFeedForward",
    # Factory
    "create_fast_lm_tiny",
    "create_fast_lm_small",
    "create_fast_lm_medium",
    "create_fast_lm_shakespeare",
    # Data
    "LMDataset",
    "create_shakespeare_dataset",
    "create_tinystories_dataset",
    "create_python_dataset",
    "Tokenizer",
    "CharacterTokenizer",
    # Advanced Tokenizers
    "BPETokenizer",
    "WordPieceTokenizer",
    "create_tokenizer",
    "load_shakespeare_tokenizer",
    # Training
    "LMTrainer",
    "TrainingConfig",
    "TrainingMetrics",
    "train_model",
]
