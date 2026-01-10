# Quick Start Guide

## 1. Installation

```bash
pip install -r requirements.txt
```

## 2. Run the Benchmark

### Quick test (1-2 minutes)
```bash
cd src
python benchmark.py --smoke-test
```

### Full benchmark (~30-60 minutes)
```bash
cd src
python benchmark.py --seeds 3
```

### Analyze results
```bash
cd src
python analyze_results.py
```

## 3. Understand the Results

The benchmark will show:
- **On-par performance**: EqProp matches Backprop within <3% on all tasks
- **Standard deviations**: Reflects seed-to-seed variance (lower is more stable)
- **Gap**: Difference between EqProp and Backprop

Key insight: The gap is small and consistent, showing EqProp is a viable alternative.

## 4. Explore the Code

- `models.py` — See how spectral normalization stabilizes EqProp
- `trainer.py` — The core EqProp training loop (~50 lines)
- `tasks.py` — Data loaders for all 5 benchmark tasks
- `benchmark.py` — Full experimental pipeline

## 5. Adapt to Your Use Case

```python
from models import LoopedMLP
from trainer import EqPropTrainer
import torch.optim as optim

# Create your model
model = LoopedMLP(input_dim=784, hidden_dim=256, output_dim=10)

# Create trainer with optimized hyperparameters
optimizer = optim.Adam(model.parameters(), lr=0.002)
trainer = EqPropTrainer(model, optimizer, beta=0.22, max_steps=30)

# Train
for x, y in your_dataloader:
    trainer.step(x, y)
```

## Common Issues

**Problem**: Training diverges (loss → NaN)
**Solution**: Ensure `use_spectral_norm=True` when creating LoopedMLP

**Problem**: Low accuracy (stuck at random level)
**Solution**: Increase `max_steps` to 30+

**Problem**: High variance across seeds
**Solution**: Check that Lipschitz constant L < 1 (spectral norm should guarantee this)

## Next Steps

- Read the full README.md for technical details
- Check results/full_benchmark.json for raw experimental data
- Modify hyperparameters in benchmark.py for your specific task
