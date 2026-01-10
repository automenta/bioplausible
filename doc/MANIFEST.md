# TorEqProp Release Package

## Contents

This directory contains a complete, self-contained implementation demonstrating that Equilibrium Propagation achieves on-par performance with Backpropagation.

### Documentation
- `README.md` — Comprehensive technical documentation for researchers
- `QUICKSTART.md` — Quick start guide to run experiments in minutes
- `LICENSE` — MIT License

### Source Code
- `src/models.py` — LoopedMLP (EqProp) and BackpropMLP implementations
- `src/trainer.py` — EqProp training algorithm
- `src/tasks.py` — Data loaders for all 5 benchmark tasks
- `src/benchmark.py` — Main benchmark script
- `src/analyze_results.py` — Results analysis and visualization

### High-Performance Kernel (Optional)
- `kernel/eqprop_kernel.py` — Pure NumPy/CuPy implementation
- `kernel/README.md` — Kernel documentation and usage guide

**Kernel advantages**: 1.2-1.5x faster, O(1) memory, FPGA-ready, no PyTorch dependency

### Data
- `results/full_benchmark.json` — Complete experimental results (3 seeds × 5 tasks)
- `requirements.txt` — Python dependencies

### Key Statistics

| Metric | Value |
|--------|-------|
| Total code | ~400 lines |
| Tasks tested | 5 (vision + control) |
| Average gap | -1.4% |
| Max gap | -2.7% (CartPole) |
| Min gap  | +0.1% (Fashion-MNIST) |

All gaps are well within 3%, demonstrating practical parity.

## Quick Reference

### Run experiments
```bash
pip install -r requirements.txt
cd src && python benchmark.py --seeds 3
```

### View results
```bash
cd src && python analyze_results.py
```

### Read documentation
- Start with `QUICKSTART.md` for a 5-minute introduction
- Then `README.md` for full technical details

## For EqProp Researchers

This package answers the key question: **"Can EqProp match Backprop accuracy?"**

Answer: **Yes**, when properly stabilized with spectral normalization. The performance gap is consistently small (<3%) across diverse tasks.

What this means:
- EqProp is not fundamentally limited
- Biological plausibility is achievable without sacrificing performance
- Neuromorphic deployment becomes practical

## Citation

If you use this work, please cite:

```
@misc{toreqprop2026,
  title={Spectral Normalization Enables Practical Equilibrium Propagation},
  year={2026},
  note={Available at: [URL]}
}
```
