# Strategy Permutation Guide

## Overview

REFACTOR8 introduces **generic strategy permutation factories** that decouple update/constraint/feedback strategies from specific algorithm implementations. This enables rapid experimentation with different combinations of:

- **Gradient strategies**: How gradients are computed (backprop, FA, target_prop, PC, hebbian, EP)
- **Update strategies**: How weights are updated (plain SGD, Muon, Dion, etc.)
- **Constraint strategies**: Spectral normalization, weight decay, etc.
- **Feedback strategies**: Error feedback, direct feedback, etc.

## Core Factory: `make_strategy_optimizer()`

Located in `computronium/core/optimization/factory.py`.

```python
from computronium.core.optimization.factory import make_strategy_optimizer

optimizer = make_strategy_optimizer(
    model=model,
    gradient="backprop",  # or "fa", "target_prop", "pc", "hebbian", "ep"
    update="plain",  # or "muon", "dion"
    constraint="none",  # or "spectral"
    feedback="none",  # or "error_feedback"
    lr=0.01,
    beta=0.5,  # for EP/contrastive
    **kwargs,
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `nn.Module` | Model with `.layers` (list of `nn.Linear`) or `.transition_modules()` |
| `gradient` | `str` | Gradient computation strategy |
| `update` | `str` | Weight update strategy |
| `constraint` | `str` | Constraint/regularization strategy |
| `feedback` | `str` | Feedback/error routing strategy |
| `lr` | `float` | Base learning rate |
| `beta` | `float` | Nudge strength for contrastive/EP methods |
| `**kwargs` | `dict` | Strategy-specific parameters |

### Gradient Strategies

| Strategy | Description | Compatible Models |
|----------|-------------|-------------------|
| `backprop` | Standard BPTT | Any model with Linear stack |
| `fa` | Feedback Alignment | Models with `feedback_weights` (e.g., `standard_fa`) |
| `target_prop` | Target Propagation | `diff_target_prop` (has inverse nets) |
| `pc` | Predictive Coding | `predictive_coding_hybrid` (has layers + top_down) |
| `hebbian` | Hebbian/3-factor | Models with `transition_modules()` + `hebbian_lr` |
| `ep` | Equilibrium Prop | `eqprop_mlp` / `LoopedMLP` (settling dynamics) |

### Update Strategies

| Strategy | Description | Requirements |
|----------|-------------|--------------|
| `plain` | Standard SGD/Adam | None |
| `muon` | Muon orthogonalization (Newton-Schulz) | Triton or CuPy for GPU |
| `dion` | Low-rank SVD (DIon) | PyTorch `svd_lowrank` |

### Constraint Strategies

| Strategy | Description |
|----------|-------------|
| `none` | No constraint |
| `spectral` | Spectral normalization on weight matrices |

### Feedback Strategies

| Strategy | Description |
|----------|-------------|
| `none` | Standard error propagation |
| `error_feedback` | Error feedback for EP variants |

## Preset Configurations

The factory includes 8 curated presets for common research permutations:

```python
from computronium.core.optimization.factory import STRATEGY_PRESETS

# List available presets
print(STRATEGY_PRESETS.keys())
# ['backprop_plain', 'backprop_muon', 'plain_tp', 'muon_tp',
#  'plain_pc', 'muon_pc', 'plain_hebbian', 'muon_hebbian',
#  'smep', 'sdmep']  # MEP-style (need energy_fn)
```

| Preset | Gradient | Update | Constraint | Feedback | Use Case |
|--------|----------|--------|------------|----------|----------|
| `backprop_plain` | backprop | plain | none | none | Baseline BPTT |
| `backprop_muon` | backprop | muon | spectral | none | BPTT + Muon |
| `plain_tp` | target_prop | plain | none | none | Target Prop baseline |
| `muon_tp` | target_prop | muon | spectral | none | Target Prop + Muon |
| `plain_pc` | pc | plain | none | none | PC baseline |
| `muon_pc` | pc | muon | spectral | none | PC + Muon |
| `plain_hebbian` | hebbian | plain | none | none | Hebbian baseline |
| `muon_hebbian` | hebbian | muon | spectral | none | Hebbian + Muon |
| `smep` | ep | muon | spectral | none | Simplified MEP |
| `sdmep` | ep | dion | spectral | error_feedback | Structured MEP |

### Using Presets

```python
from computronium.core.optimization.factory import (
    make_strategy_optimizer,
    STRATEGY_PRESETS,
)

# Use a preset
config = STRATEGY_PRESETS["muon_pc"]
optimizer = make_strategy_optimizer(model=model, **config, lr=0.01)
```

## Integration with CoreTrainer

The `CoreTrainer` supports strategy optimizers via the `optimizer` config field:

```python
from computronium.core.trainer import CoreTrainer, TrainerConfig

config = TrainerConfig(
    model="predictive_coding_hybrid",
    optimizer="muon_pc",  # Use preset name
    optimizer_kwargs={"lr": 0.01, "beta": 0.5},
    task="mnist",
    epochs=10,
    use_kernel=True,  # Optional: use kernel backend
    kernel_backend="triton",
)
trainer = CoreTrainer(config)
trainer.fit()
```

## Benchmarking Strategy Permutations

Use the benchmark tool to sweep permutations across models/datasets:

```bash
uv run python tools/benchmark_strategy_permutations.py \
    --models backprop_mlp standard_fa pepita diff_target_prop predictive_coding_hybrid eqprop \
    --datasets digits mnist fashion_mnist \
    --precisions fp32 fp16 bf16 \
    --epochs 20 \
    --output artifacts/strategy_benchmark_report.json
```

### Benchmark Output Schema (v1)

```json
{
  "schema_version": "v1",
  "timestamp": "2024-01-15T10:30:00Z",
  "results": [
    {
      "model": "predictive_coding_hybrid",
      "dataset": "mnist",
      "permutation": "muon_pc",
      "precision": "fp32",
      "epochs": 20,
      "final_accuracy": 0.923,
      "backprop_baseline": 0.951,
      "parity_ratio": 0.971,
      "passes_gate": true,
      "time_per_epoch_sec": 12.4,
      "peak_memory_mb": 1024
    }
  ]
}
```

### Gate Criteria

Each permutation must reach **≥90% of `backprop_plain` accuracy** on digits within 20 epochs.

## Model Compatibility Matrix

| Model | backprop_plain | backprop_muon | plain_tp | muon_tp | plain_pc | muon_pc | plain_hebbian | muon_hebbian |
|-------|----------------|---------------|----------|---------|----------|---------|---------------|--------------|
| `backprop_mlp` | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `standard_fa` | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅* | ✅* |
| `pepita` | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `diff_target_prop` | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| `predictive_coding_hybrid` | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ |
| `eqprop` | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

*Requires `hebbian_lr` in model config.

## Custom Strategy Registration

For advanced research, register custom strategies:

```python
from computronium.core.optimization.factory import (
    register_gradient_strategy,
    register_update_strategy,
    GradientStrategy,
    UpdateStrategy,
)

class MyCustomGradient(GradientStrategy):
    def compute_gradients(self, model, x, y):
        # Custom gradient computation
        pass

register_gradient_strategy("my_custom", MyCustomGradient)

# Now use it
optimizer = make_strategy_optimizer(model, gradient="my_custom", ...)
```

## Performance Notes

- **Muon update**: Requires Triton or CuPy for GPU acceleration. CPU fallback uses PyTorch SVD (slower).
- **Spectral constraint**: Applied per-layer during update. Adds ~10-15% overhead.
- **Kernel backend**: When `use_kernel=True` in `CoreTrainer`, the kernel backend handles forward/backward; the strategy optimizer only manages the update step.

## Migration from MEP Presets

REFACTOR7 MEP presets (`smep`, `sdmep`, `local_ep`, `natural_ep`, `muon_backprop`) are still available but now implemented as compositions of the generic factory. They require an `energy_fn` parameter:

```python
from computronium.core.optimization.factory import STRATEGY_PRESETS

# MEP presets (require energy_fn)
config = STRATEGY_PRESETS["smep"]
optimizer = make_strategy_optimizer(
    model=model,
    energy_fn=my_energy_fn,  # Required for EP-based presets
    **config,
)
```

## Troubleshooting

### "Model does not have required attributes"

Ensure your model exposes:
- `model.layers` — list/ModuleList of `nn.Linear` (for backprop, FA, TP, PC)
- `model.transition_modules()` — returns list of `nn.Linear` (for Hebbian, EP)
- `model.feedback_weights` — for FA gradient strategy
- `model.hebbian_lr` — for Hebbian gradient strategy

### "Muon update requires Triton/CuPy"

Install Triton: `uv add triton`
Or ensure CuPy is available: `uv add cupy`

### Parity gate fails on synthetic data

The 90% gate on `digits` (5-10 epochs) is stringent for non-backprop variants. Use:
- Real MNIST/Fashion-MNIST (`--datasets mnist fashion_mnist`)
- More epochs (`--epochs 20+`)
- FP16/BF16 with 15+ epochs for parity