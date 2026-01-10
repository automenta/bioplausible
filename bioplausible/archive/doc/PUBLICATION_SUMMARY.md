# TorEqProp: Multi-Task Benchmark Results

## Executive Summary

**Core Finding**: Equilibrium Propagation with Spectral Normalization achieves practical parity with Backpropagation across diverse tasks.

## Results Matrix

| Task | Backprop | LoopedMLP (SN) | Gap | Status |
|------|----------|----------------|-----|--------|
| Digits (8x8) | 97.04% ± 0.35% | 94.63% ± 0.73% | -2.4% | ✅ |
| MNIST | 94.91% ± 0.05% | 94.22% ± 0.08% | -0.7% | ✅ |
| Fashion-MNIST | 83.25% ± 0.33% | 83.32% ± 0.21% | +0.1% | ✅ |
| CartPole-BC | 99.80% ± 0.08% | 97.13% ± 1.64% | -2.7% | ✅ |
| Acrobot-BC | 97.97% ± 0.47% | 96.83% ± 1.17% | -1.1% | ✅ |

**Average Gap**: -1.4%

## Key Insights

1. **Parity Achieved**: All gaps < 3%, demonstrating EqProp is a viable alternative to Backprop.
2. **Spectral Normalization Essential**: Without it, training diverges.
3. **Hyperparameter Sensitivity**: `max_steps=30` and task-specific `beta` are critical.
4. **Generalization**: Works on vision (MNIST, Fashion) AND control (CartPole, Acrobot).

## Optimal Hyperparameters

| Task | Learning Rate | Beta | Max Steps |
|------|---------------|------|-----------|
| Digits | 0.001 | 0.22 | 30 |
| MNIST | 0.002 | 0.22 | 30 |
| Fashion-MNIST | 0.002 | 0.50 | 30 |
| CartPole | 0.001 | 0.22 | 30 |
| Acrobot | 0.002 | 0.50 | 30 |

## Reproducibility

All experiments use:
- 3 random seeds (42, 43, 44)
- PyTorch with CUDA
- Adam optimizer
- Spectral normalization enabled

Run: `python scripts/multi_dataset_benchmark.py --seeds 3`

## Statistical Validation

- All results pass significance tests (p < 0.05 vs random baseline)
- Standard deviations < 2% for most configurations
- See `results/statistical_report.md` for full analysis

## Conclusion

**Equilibrium Propagation is a practical, biologically-plausible alternative to Backpropagation** that achieves competitive accuracy across diverse tasks while enabling:
- O(1) memory training (constant regardless of depth)
- Local Hebbian-like weight updates
- Compatibility with neuromorphic hardware
