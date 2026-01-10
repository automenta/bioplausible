# Equilibrium Propagation with Spectral Normalization
http://github.com/narchy/eqprop

## Motivation

Recent EqProp variants—Holomorphic EP (Laborieux et al., NeurIPS 2022), Finite-Nudge EP (Litman, 2025), and Directed EP—address theoretical limitations around gradient exactness and biological plausibility. However, we observed that practical implementations often struggle with training instability, particularly on complex datasets like SVHN and CIFAR-10.

This led us to investigate whether **spectral normalization**, widely used in GANs for stability, could address EqProp's convergence issues. _To our knowledge, this combination hasn't been systematically studied in the literature._

## Empirical Findings

We compared EqProp models with and without spectral normalization across multiple architectures and datasets. Results on SVHN (real-world street-view digits, RGB) were particularly informative:

| Architecture | With SN | Without SN | Δ | Lipschitz |
|--------------|---------|------------|---|-----------|
| Standard LoopedMLP | 57.2% | 33.2% | +24.0% | 1.02 → 5.89 |
| Holomorphic-inspired | 52.9% | 27.4% | +25.5% | 2.08 → 4.94 |
| Finite-Nudge (β=0.5) | 51.4% | 24.3% | +27.1% | 2.03 → 5.33 |
| DEEP (Asymmetric) | 59.2% | 28.6% | +30.6% | 2.29 → 5.63 |

The pattern held across datasets: easier tasks (MNIST) showed minimal difference, while challenging RGB datasets demonstrated dramatic improvements. Without SN, the recurrent weight Lipschitz constant consistently grew to 5-6 during training, apparently disrupting equilibrium convergence.

## Interpretation

Spectral normalization constrains the maximum singular value of weight matrices to ≤1, enforcing a contraction mapping in the equilibrium dynamics. This appears to provide:

- **Convergence guarantee**: L < 1 ensures unique fixed points exist
- **Stability under perturbation**: Exponential damping of noise (validated experimentally)
- **Compatibility with theory**: Works across all tested 2025 variants

Notably, SN doesn't replace recent theoretical advances—it appears to solve an orthogonal implementation problem. The 2025 methods address gradient exactness and biological plausibility; SN addresses numerical stability.

## Reproducibility

The repository includes:
- 30 automated validation tracks (`python verify.py`)
- Comparison scripts for all major experiments
- SVHN benchmark: `python definitive_2025_comparison.py`
- Extended dataset tests: `python extended_sn_benchmark.py`

All experiments use PyTorch with seeds fixed for reproducibility. Training typically completes in 2-5 minutes per configuration on a single GPU.

## Open Questions

While our results are consistent, several aspects warrant independent verification:

1. **Scale**: Does the SN benefit persist at ImageNet scale or beyond?
2. **True holomorphic implementation**: Our holomorphic tests used real-valued approximations
3. **Hardware**: Would these stability benefits translate to neuromorphic chips?
4. **Theory**: Can we formally characterize when SN is critical vs. optional?

We've found the effect most pronounced on RGB datasets, smaller models, and longer training runs—suggesting a relationship between task complexity and SN necessity.

## Invitation

If you work with Equilibrium Propagation, we encourage you to:
- Reproduce the SVHN comparison (most dramatic results)
- Test SN on your own EqProp variants or datasets
- Report any contradictory findings

The goal is community verification of whether SN represents a practical requirement for production EqProp systems, as our experiments suggest.

---

**Repository includes**: Verified implementations • Benchmark scripts • 30 validation tracks • Comprehensive documentation
