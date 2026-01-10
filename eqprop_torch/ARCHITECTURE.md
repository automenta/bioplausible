# Architecture Guide

## Overview

The `eqprop-torch` library contains **two categories** of models:

### 1. Native PyTorch Models (`models.py`)

These are **production-ready, bio-plausible** models implemented as clean PyTorch `nn.Module` classes:

| Model | Description | Bio-Plausible? |
|-------|-------------|----------------|
| **LoopedMLP** | Recurrent EqProp with fixed-point dynamics | ✅ Yes |
| **ConvEqProp** | Convolutional EqProp for vision | ✅ Yes |
| **TransformerEqProp** | Transformer with equilibrium settling | ✅ Yes |
| **BackpropMLP** | Standard backprop baseline | ❌ No (baseline) |

**Usage:**
```python
from eqprop_torch import LoopedMLP, EqPropTrainer

model = LoopedMLP(784, 256, 10, use_spectral_norm=True)
trainer = EqPropTrainer(model)
trainer.fit(train_loader, epochs=10)
```

### 2. Research Algorithm Wrappers (`bioplausible.py`)

These wrap **13 experimental algorithms** from the `algorithms/` research directory. They use a different API (custom `train_step()` methods) so we provide a **compatibility wrapper** called `BioplausibleModel`:

**The 13 Research Algorithms:**
- backprop, eqprop, feedback_alignment
- eq_align, ada_fa, cf_align, leq_fa
- pc_hybrid, eg_fa, sparse_eq, mom_eq, sto_fa, em_fa

**Usage:**
```python
from eqprop_torch import BioplausibleModel

# Wraps research algorithm to be PyTorch-compatible
model = BioplausibleModel('ada_fa', 784, [512], 10)
```

## Key Distinction

- **Native models** (LoopedMLP, etc.): Direct PyTorch implementations, recommended for production
- **Wrapped algorithms**: Research code made compatible, useful for experiments/comparisons

**Important**: LoopedMLP and ConvEqProp ARE bio-plausible! They're just implemented directly in PyTorch rather than using the research codebase's custom training interface.

## Recommendation

For most users:
- Use `LoopedMLP` or `ConvEqProp` from `models.py` for production
- Use `BioplausibleModel` wrapper only for experimental comparison with research algorithms

## Future Direction

Consider migrating popular research algorithms (e.g., `ada_fa`, `eq_align`) to native PyTorch implementations in `models.py` for better integration.
