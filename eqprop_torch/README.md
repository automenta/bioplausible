# EqProp-Torch: Equilibrium Propagation for PyTorch

> **The reference implementation of Equilibrium Propagation** with spectral normalization, torch.compile acceleration, and optional GPU kernels.

[![PyPI](https://img.shields.io/pypi/v/eqprop-torch)](https://pypi.org/project/eqprop-torch/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Installation

```bash
pip install eqprop-torch

# With GPU acceleration (NVIDIA)
pip install eqprop-torch[gpu]

# With HuggingFace datasets
pip install eqprop-torch[datasets]

# Everything
pip install eqprop-torch[all]
```

## Quick Start

```python
from eqprop_torch import EqPropTrainer, LoopedMLP
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Create model with spectral normalization (required for stability)
model = LoopedMLP(
    input_dim=784, 
    hidden_dim=256, 
    output_dim=10,
    use_spectral_norm=True  # Guarantees L < 1
)

# High-level trainer with torch.compile for 2x speedup
trainer = EqPropTrainer(model, use_compile=True)

# Standard PyTorch training loop
train_data = datasets.MNIST('.', train=True, download=True, 
                            transform=transforms.ToTensor())
trainer.fit(DataLoader(train_data, batch_size=64), epochs=10)

# Evaluate
print(f"Accuracy: {trainer.evaluate(test_loader)['accuracy']:.1%}")
```

## Language Modeling

```python
from eqprop_torch import EqPropTrainer, TransformerEqProp
from eqprop_torch.datasets import get_lm_dataset

# Character-level LM on tiny_shakespeare
model = TransformerEqProp(
    vocab_size=65,
    hidden_dim=256,
    num_layers=4,
    max_seq_len=128,
)

train_loader, vocab = get_lm_dataset("tiny_shakespeare", seq_len=128)
trainer = EqPropTrainer(model, use_compile=True)
trainer.fit(train_loader, epochs=50)

# Generate text
print(model.generate("ROMEO:", max_new_tokens=100))
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Spectral Normalization** | Guarantees Lipschitz L < 1 for stable dynamics |
| **torch.compile** | 2-3x speedup, portable across CPU/CUDA/MPS |
| **O(1) Memory** | Kernel mode uses constant memory regardless of depth |
| **5 LM Variants** | Full, attention-only, recurrent-core, hybrid, looped-MLP |
| **ONNX Export** | Deploy to edge devices |

## Models

- `LoopedMLP` — Core EqProp model with fixed-point dynamics
- `ConvEqProp` — Convolutional variant for vision
- `TransformerEqProp` — Attention with equilibrium settling
- `EqPropLM` — Language model variants (full, attention_only, hybrid, etc.)
- `BackpropMLP` — Baseline for comparison

## Acceleration

```python
# Default: torch.compile (portable, 2-3x speedup)
trainer = EqPropTrainer(model, use_compile=True)

# Optional: CuPy kernel (NVIDIA only, O(1) memory)
trainer = EqPropTrainer(model, use_kernel=True)
```

## References

1. Scellier & Bengio (2017). *Equilibrium Propagation*. Frontiers in Computational Neuroscience.
2. Miyato et al. (2018). *Spectral Normalization for GANs*. ICLR.

## License

MIT
