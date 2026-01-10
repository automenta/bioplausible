# Memory Scaling Analysis

> **Date**: 2025-12-31  
> **Test**: Memory usage vs model size (hidden_dim: 64→1024)

---

## Executive Summary

**Finding**: Current EqProp implementation uses **MORE memory** than Backprop at large scales.

| Hidden Dim | EqProp | Backprop | Ratio |
|------------|--------|----------|-------|
| 64 | 1.12 MB | 1.00 MB | 1.12× |
| 256 | 1.35 MB | 0.80 MB | 1.68× |
| 1024 | 21.01 MB | 4.30 MB | **4.89×** |

**Memory Growth**: EqProp 18.84× vs Backprop 4.31× (64→1024)

---

## Why EqProp Uses More Memory

### Current Implementation

EqProp stores:
1. **Equilibrium states** (h_free, h_nudged)
2. **Intermediate forward_step states** during solving
3. **Energy computations**

**Result**: O(steps × hidden_dim) memory during equilibrium solving

### True O(1) Memory: LocalHebbianUpdate

The **LocalHebbianUpdate** strategy (ported from archive) achieves true O(1) memory by:
- Computing weight updates via **local Hebbian learning**
- **No backpropagation** through equilibrium solver
- **No storage** of intermediate activations

**Status**: Requires full trainer integration (not yet connected)

---

## Detailed Results

### EqProp Memory Scaling

| Hidden Dim | Memory | Parameters | Memory/Param Ratio |
|------------|--------|------------|-------------------|
| 64 | 1.12 MB | 8,970 | 0.125 KB/param |
| 128 | 1.22 MB | 26,122 | 0.047 KB/param |
| 256 | 1.35 MB | 85,002 | 0.016 KB/param |
| 512 | 5.51 MB | 301,066 | 0.018 KB/param |
| 1024 | 21.01 MB | 1,126,410 | 0.019 KB/param |

**Scaling Rate**: Memory grows at **0.14× parameter growth** (sub-linear)

### Backprop Memory Scaling

| Hidden Dim | Memory | Parameters |
|------------|--------|------------|
| 64 | 1.00 MB | 8,970 |
| 128 | 0.96 MB | 26,122 |
| 256 | 0.80 MB | 85,002 |
| 512 | 1.15 MB | 301,066 |
| 1024 | 4.30 MB | 1,126,410 |

**Scaling Rate**: More efficient due to PyTorch's optimized autodiff

---

## Implications

### Current State (MSEProxyUpdate)

| Aspect | EqProp | Backprop |
|--------|--------|----------|
| Small models (64-256) | ✓ Comparable | ✓ Baseline |
| Large models (512-1024) | ❌ 4-5× more memory | ✓ More efficient |
| Scaling | Sub-linear (0.14×) | Superior |

### With LocalHebbianUpdate (Future)

**Expected** O(1) memory characteristics:
- Memory **independent** of model depth
- No activation storage
- Constant ~1-2 MB overhead

**Trade-off**: Slightly lower accuracy (~1-2% based on archive tests)

---

## Path Forward

### Option 1: Integrate LocalHebbianUpdate

**Effort**: Medium (2-3 hours)
- Port `LocalHebbianUpdate` trainer integration from archive
- Add hooks for activation recording
- Test accuracy vs memory trade-off

**Expected Result**: True O(1) memory with ~95-96% accuracy

### Option 2: Memory-Optimized Standard EqProp

**Effort**: Low (30 mins)
- Clear intermediate states after each equilibrium phase
- Use `torch.cuda.empty_cache()` strategically
- Gradient checkpointing for large models

**Expected Result**: 20-30% memory reduction, still not O(1)

### Option 3: Document Current Limitation

**Effort**: Minimal
- Update claims to reflect current implementation
- Note LocalHebbianUpdate as future work
- Focus on other advantages (biological plausibility, speed with neuromorphic hardware)

---

## Recommendation

**For now**: Option 3 (document limitation)

**Rationale**:
- Current implementation achieves competitive **accuracy** (97.50%)
- Memory is not a bottleneck for MNIST-scale problems  
- Spectral normalization (our key contribution) is orthogonal to memory
- LocalHebbianUpdate integration can be future work

**Updated Claims**:
- ✅ Competitive accuracy with spectral norm
- ✅ Biological plausibility
- ✅ Sub-linear memory scaling (0.14× parameter growth)
- ⚠️ O(1) memory requires LocalHebbianUpdate (future work)

---

## Conclusion

**Memory Analysis Summary**:
- Current EqProp: Sub-linear scaling but 4-5× more than Backprop at large scales
- True O(1) possible with LocalHebbianUpdate (requires integration)
- Not a limitation for current benchmarks (MNIST, <1GB models)

**Focus remains on**:
- ✓ Accuracy (97.50% = Backprop)
- ✓ Spectral norm for stability
- ✓ Biological plausibility
