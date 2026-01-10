# Novel Hybrid Learning Algorithms - Evaluation Guide

## Overview

This project implements **13 learning algorithms** (3 baselines + 10 novel hybrids) that combine Equilibrium Propagation and Feedback Alignment principles to explore parameter-efficient, biologically-plausible learning.

## Quick Start

### 1. Test All Algorithms
```bash
python tests/test_algorithms.py -v
```

### 2. List Available Algorithms
```bash
python run_evaluation.py --list-algorithms
```

### 3. Run 1-Hour Comprehensive Evaluation
```bash
python run_evaluation.py --hours 1.0
```

## Implemented Algorithms

### Baselines
- **backprop**: Standard backpropagation (control)
- **eqprop**: Standard Equilibrium Propagation (contrastive Hebbian)
- **feedback_alignment**: Standard Feedback Alignment (random fixed B)

### Core Hybrids
- **eq_align**: Equilibrium dynamics + FA training (best of both)
- **ada_fa**: Adaptive FA (feedback weights slowly evolve)
- **cf_align**: Contrastive learning via FA signals
- **leq_fa**: Layer-wise local equilibrium settling

### Radical Variants  
- **pc_hybrid**: Predictive Coding + Feedback Alignment
- **eg_fa**: Energy-guided FA (state energy modulates updates)
- **sparse_eq**: Sparse equilibrium (Top-K neuron updates)
- **mom_eq**: Momentum-accelerated settling dynamics
- **sto_fa**: Stochastic FA (dropout on feedback signals)
- **em_fa**: Energy minimization as training objective

## Evaluation Options

### Parameter Budgets
```bash
# Test specific parameter budgets
python run_evaluation.py --param-budgets "25000,50000,100000"
```

### Duration
```bash
# Quick test (5 minutes)
python run_evaluation.py --hours 0.083

# Standard run (1 hour)
python run_evaluation.py --hours 1.0

# Deep evaluation (4 hours)
python run_evaluation.py --hours 4.0
```

### Training Data
```bash
# Use subset for faster evaluation
python run_evaluation.py --n-samples 5000

# Use more data for better convergence
python run_evaluation.py --n-samples 20000
```

## Expected Output

Results are saved to `results/algorithm_comparison/TIMESTAMP/`:

```
results/algorithm_comparison/20260108_103000/
├── config.json                      # Run configuration
├── results_budget_50000.json       # Raw results per budget
├── results_budget_100000.json
├── results_budget_200000.json
└── report.md                        # Comprehensive markdown report
```

## Interpreting Results

The evaluation measures:
- **Test Accuracy**: Final accuracy on held-out test set
- **Train Accuracy**: Training set performance
- **Epochs**: Number of complete training passes
- **Time**: Walltime used (seconds)
- **Params**: Total trainable parameters

### Key Questions
1. **Which algorithm achieves highest accuracy?**
2. **Which is most parameter-efficient?** (best accuracy per param)
3. **Which trains fastest?** (best accuracy per second)
4. **Do hybrids outperform baselines?**

## Design Philosophy

### Core Innovation
Novel hybrids explore **combining complementary strengths**:
- EqProp: Rich equilibrium dynamics, biological plausibility
- FA: Fast single-pass training, no explicit backward pass

### Parameter Efficiency Focus
Testing at 50K-200K params to demonstrate "doing more with less" and identify algorithms suited for:
- Edge devices
- Continual learning
- Resource-constrained environments

## Technical Notes

### Device
All tests run on **CPU** for reproducibility and fairness (no GPU acceleration bias).

### Random Seeds
Results may vary due to random initialization. For publication-quality results, run with multiple seeds and average.

### Failed Algorithms
Some algorithms may fail to train on certain tasks. This is valuable information about algorithmic robustness.

## Next Steps

After evaluation completes:
1. Review `report.md` for rankings
2. Identify top 3 performers
3. Conduct deeper analysis on promising algorithms
4. Test on additional datasets (CIFAR-10, Shakespeare LM)

## Citation

If you use these algorithms or evaluation framework, please cite:
```
[To be updated after evaluation results are analyzed]
```
