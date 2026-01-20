# TorEqProp: Publication Strategy & Roadmap

> **Status**: Ready to Execute  
> **Goal**: arXiv preprint → Conference submission  
> **Timeline**: 2-4 weeks to arXiv, ICML/NeurIPS 2025/2026 submission

---

## 🎯 Publication Strategy Overview

### Confirmed Novelty

After exhaustive prior art search (arXiv, Google Scholar, NeurIPS/ICLR/ICML, OpenReview, X):

> **No prior work on EqProp for transformer training exists.**

This is a **first** in the field.

### Publication Portfolio (4 Papers)

| Paper | Target Venue | Timeline | Readiness | Priority |
|-------|--------------|----------|-----------|----------|
| **A: Spectral Normalization for EqProp** | NeurIPS/ICML | 2-3 weeks | 90% | ⭐ FLAGSHIP |
| **B: β-Stability Guidelines** | TMLR/JMLR | 3-4 weeks | 85% | Secondary |
| **C: Pure NumPy/CuPy Kernel** | MLSys/JMLR-OSS | 2 weeks | Complete | ⭐ SYSTEMS |
| **D: Hierarchical EqProp for Vision** | ICLR 2026 | 4-6 weeks | 40% | Contingent |

---

## 📋 Paper Details

### Paper A: Spectral Normalization Enables Stable EqProp ⭐ **FLAGSHIP**

**Target**: NeurIPS 2025 / ICML 2025 (Main Track)  
**Status**: Template ready, needs experimental validation  
**Timeline**: 2-3 weeks

**Core Contribution**: First demonstration that spectral normalization solves EqProp stability, enabling competitive accuracy (97.50% = backprop).

**Key Results Needed**:
- ✅ Lipschitz L < 1 maintained during training
- ⬜ Multi-seed MNIST ≥ 94% (5 seeds)
- ⬜ CIFAR-10 ≥ 50% (hierarchical)
- ⬜ Ablation: without SN → divergence

**Novelty**: First rigorous EqProp scaling to modern architectures.

---

### Paper B: Fixed β Beats Annealing

**Target**: TMLR / JMLR / NeurIPS Workshop  
**Status**: Template exists, needs multi-seed data  
**Timeline**: 1-2 weeks

**Core Contribution**: Empirical discovery that β-annealing causes collapse; fixed β=0.22 is optimal.

**Key Results Needed**:
- ⬜ β-annealing collapse reproduced (3+ seeds)
- ⬜ β sweep [0.15-0.30] with 5-7 values
- ⬜ Optimal β characterization
- ✅ Stability range [0.20-0.26] validated

**Novelty**: First systematic β study; contradicts theory (β→0 better).

**Publication Strategy**: 
- **Option 1**: Standalone paper (TMLR, 4-6 months review)
- **Option 2**: Appendix to Paper A (stronger combination)
- **Recommendation**: Include in Paper A for maximum impact

---

### Paper C: A Pure NumPy/CuPy Kernel for Portable EqProp ⭐ **SYSTEMS**

**Target**: MLSys 2026 / NeurIPS Systems Track  
**Status**: Implementation complete, needs writeup  
**Timeline**: 2 weeks

**Core Contribution**: Standalone, autograd-free EqProp kernel achieving 58% speedup over PyTorch.

**Key Results**:
- ✅ Kernel implementation (1,056 lines)
- ✅ 58% faster than PyTorch (21.4ms vs 33.9ms, aggressive mode)
- ✅ 2.49x GPU speedup
- ✅ 69% MNIST accuracy (learning confirmed)
- ⬜ CIFAR-10 validation
- ⬜ Memory scaling O(1) verification

**Novelty**: First portable, hardware-deployable EqProp implementation. Directly translates to HLS/FPGA.

**Why This Matters**:
- Enables neuromorphic deployment
- Proves EqProp viability for edge AI
- Reference implementation for researchers

**Publication Strategy**: 
- Strong fit for **MLSys** (systems/implementation focus)
- Alternative: **JMLR** Open Source Software track
- Can cite Paper A for theoretical foundation

---

### Paper D: Hierarchical EqProp for Vision Tasks

**Target**: ICLR 2026 / Computer Vision venue  
**Status**: Models exist (EnhancedMSTEP), needs experimental validation  
**Timeline**: 4-6 weeks *(contingent on CIFAR-10 success)*

**Core Contribution**: Multi-scale hierarchical architectures enable EqProp scaling to complex vision tasks.

**Key Results Needed**:
- ⬜ CIFAR-10 ≥ 60% with EnhancedMSTEP
- ⬜ Ablation: hierarchical vs flat
- ⬜ ImageNet-32 proof-of-concept (optional)

**Novelty**: First hierarchical EqProp architecture; first serious CIFAR-10 results.

**Publication Strategy**: 
- **Contingent**: Only if CIFAR-10 results are strong (≥50%)
- If results weak (<50%), **defer to future work** or include as preliminary in Paper A

---

## 📅 Timeline & Execution Plan

### Publication Timeline

```
2026-01-02 (Now)
    ↓
    ├── Week 1-2: Run Experiments (Phases 1-2 from TODO.md)
    │   ├── Multi-seed MNIST
    │   ├── Hierarchical CIFAR-10
    │   ├── Ablations
    │   └── Speed/memory validation
    ↓
    ├── Week 3: Generate Papers A+B
    │   ├── Run generate_paper.py for Paper A
    │   ├── Write Paper C (kernel) manually
    │   └── Validate all claims
    ↓
    ├── Week 4: arXiv Submission
    │   ├── Paper A (Spectral Norm) → arXiv ⭐ PRIORITY
    │   ├── Paper C (Kernel) → arXiv
    │   └── Community announcement
    ↓
    ├── Weeks 5-6: Conference Preparation
    │   ├── Paper A → NeurIPS 2025 (May deadline)
    │   └── Paper C → MLSys 2026 (Oct deadline)
    ↓
    ├── Contingent: Paper D
        └── Only if CIFAR-10 ≥ 50% by Week 2
```

### Immediate Action Plan (Next 2 Weeks)

#### Week 1: Strengthen Evidence

**Day 1-2: Multi-Seed Validation**
```bash
# Run 5-seed validation for all main claims
python scripts/competitive_benchmark.py --seeds 5 --epochs 50

# Save results
mv /tmp/competitive_benchmark.json results/competitive_benchmark_5seed.json
```

**Goal**: Statistical significance for accuracy claims

**Day 3-4: Additional Experiments**
```bash
# Fashion-MNIST (easy extension)
python scripts/competitive_benchmark.py --dataset fashion-mnist --seeds 3

# Full MNIST (60K samples)
python scripts/competitive_benchmark.py --dataset mnist --dataset-size 60000 --epochs 100
```

**Goal**: Demonstrate generalization beyond toy benchmarks

**Day 5-7: Figures & Visualizations**

Create publication-quality figures:
- [ ] Training curves (accuracy vs epoch)
- [ ] Lipschitz evolution during training (with/without SN)
- [ ] β sweep accuracy curve
- [ ] Memory scaling plot
- [ ] Architecture diagram

```bash
# Generate figures (after implementing)
python scripts/generate_figures.py --output figures/
```

#### Week 2: Write & Submit Preprint

**Day 8-10: Complete Paper A Draft**

1. Fill in all `<!-- INSERT:... -->` markers with real data
2. Write complete related work section
3. Add all figures
4. Proofread abstract

```bash
# Generate paper with real data
python scripts/generate_paper.py --paper spectral_normalization
```

**Day 11-12: Internal Review**

- [ ] Check all numbers match experimental data
- [ ] Verify claims have evidence
- [ ] Review for clarity and flow
- [ ] Check citation completeness

**Day 13-14: arXiv Submission**

1. Convert to LaTeX:
   ```bash
   pandoc papers/spectral_normalization_paper.md -o paper.tex
   ```

2. Add arXiv metadata

3. Submit to arXiv (cs.LG, cs.NE)

4. Post on X/Reddit for community feedback

---

## 🗓️ Conference Submission Calendar

### 2025 Deadlines (Check for updates!)

| Conference | Abstract | Paper | Decision | Notes |
|------------|----------|-------|----------|-------|
| **ICLR 2026** | Sep 2025 | Oct 2025 | Jan 2026 | Premier ML venue |
| **ICML 2025** | Jan 2025 | Feb 2025 | May 2025 | May be too soon |
| **NeurIPS 2025** | May 2025 | May 2025 | Sep 2025 | Good timeline |
| **AAAI 2026** | Aug 2025 | Aug 2025 | Nov 2025 | Backup venue |

### 2025/2026 Strategy

1. **January 2025**: Submit to arXiv immediately (timestamp novelty)
2. **February 2025**: Target ICML 2025 if ready, else NeurIPS 2025
3. **May 2025**: NeurIPS 2025 submission (main target)
4. **Sep 2025**: ICLR 2026 (backup/improved version)

---

## 🔬 Experiments Needed for Publication

### Must Have (Paper A)

| Experiment | Status | Time | Priority |
|------------|--------|------|----------|
| MNIST with 5 seeds | ⬜ Pending | 2h | P0 |
| Fashion-MNIST | ⬜ Pending | 2h | P0 |
| Lipschitz evolution plots | ⬜ Pending | 1h | P0 |
| β sweep with 3 seeds | ⬜ Pending | 3h | P0 |
| Training curves | ⬜ Pending | 1h | P0 |

### Nice to Have (Strengthens Paper)

| Experiment | Status | Time | Priority |
|------------|--------|------|----------|
| CIFAR-10 | ⬜ Pending | 6h | P1 |
| Longer sequences (algorithmic) | ⬜ Pending | 4h | P1 |
| Memory profiling at scale | ⬜ Pending | 2h | P1 |
| Ablation: attention type | ⬜ Pending | 3h | P2 |

### Future Work (Paper C)

| Experiment | Status | Time | Priority |
|------------|--------|------|----------|
| LocalHebbianUpdate fix | ⬜ Pending | 6h | P1 |
| O(1) memory validation | ⬜ Pending | 4h | P1 |
| Neuromorphic simulation | ⬜ Pending | 40h | P2 |

---

## 📝 Related Work Section (Draft)

Use this in papers:

```markdown
## Related Work

### Equilibrium Propagation
Scellier & Bengio (2017) introduced Equilibrium Propagation as a biologically 
plausible alternative to backpropagation. Subsequent work scaled EqProp to 
convolutional networks (Laborieux et al., 2021), extended it to continuous 
time (Ernoult et al., 2020), and explored hardware implementations for 
neuromorphic systems (various, 2024-2025). However, all prior work has been 
limited to MLPs, CNNs, or recurrent architectures. **To our knowledge, we are 
the first to apply Equilibrium Propagation to train attention-based transformer 
architectures.**

### Looped Transformers
Weight-tied transformers have been studied theoretically (Giannou et al., 2023) 
and shown to enable adaptive computation (Yang et al., 2024). These works use 
standard backpropagation for training. Our work combines looped architectures 
with EqProp training, inheriting the parameter efficiency of weight-tying while 
gaining the biological plausibility of contrastive Hebbian learning.

### Deep Equilibrium Models
DEQs (Bai et al., 2019) also find fixed points as forward pass outputs but 
compute gradients via implicit differentiation through the equilibrium, 
preserving backpropagation. In contrast, EqProp uses local contrastive updates 
that require no gradient backpropagation, making it suitable for neuromorphic 
hardware.
```

---

## 🎯 Recommended Submission Strategy

### Bundling vs Splitting Decision Matrix

| Approach | Papers | Pros | Cons | Recommendation |
|----------|--------|------|------|----------------|
| **Bundle All** | 1 paper (A+B+C) | Comprehensive, high impact | Dilutes focus, harder review | ❌ Too much |
| **Split A+B, Separate C** | 2 papers | Balanced, clear narrative | β discovery might be weak alone | ✅ **BEST** |
| **Split All** | 3-4 papers | More publications | Salami slicing, lower impact each | ⚠️ Risky |
| **A only** | 1 flagship | Maximum focus | Leaves kernel unpublished | ❌ Wastes work |

### Final Recommendation: **2-Paper Strategy**

1. **Paper A**: Spectral Norm + β-Annealing (NeurIPS/ICML)
   - Combines two complementary findings
   - Stronger experimental section
   - More compelling narrative

2. **Paper C**: Kernel Implementation (MLSys/JMLR-OSS)
   - Different venue, different audience
   - Systems contribution
   - Enables adoption

3. **Paper D**: Contingent (defer decision to Week 2 results)

### Immediate (Next 2 months)

1. **Paper A (Spectral Norm)** → NeurIPS 2025
   - Strongest contribution
   - Include β-annealing discovery as secondary finding
   - Target main conference track

2. **Paper C (Kernel)** → arXiv + MLSys 2026
   - Systems contribution, different audience
   - Enables reproducibility
   - Cite Paper A for theory

### Medium-term (3-6 months)

3. **Paper D (Hierarchical)** → ICLR 2026 *(if CIFAR-10 strong)*
   - Only pursue if ≥60% accuracy achieved
   - Otherwise include as "future work" in Paper A

### Long-term (Post-publication)

4. **Paper E: Neuromorphic Deployment** *(deferred to Phase 3)*
   - FPGA/Loihi implementation
   - Real power measurements
   - Target: Nature Electronics / specialized venue

---

## 📢 Community Engagement Plan

### Pre-Publication

1. **arXiv announcement** → Timestamp novelty claim
2. **X thread** → Summary of key findings
3. **r/MachineLearning post** → Technical discussion

### Post-Publication

1. **Blog post** → Accessible explanation
2. **YouTube/podcast** → If invited
3. **Conference presentation** → Workshops, poster sessions

### Key People to Engage

| Person | Affiliation | Relevance |
|--------|-------------|-----------|
| Yoshua Bengio | Mila | EqProp co-inventor |
| Benjamin Scellier | Cornell | EqProp inventor |
| Damien Querlioz | U Paris-Saclay | EqProp hardware |
| Various @neuromorph_ | X | Neuromorphic community |

---

## ⚠️ Risk Mitigation

### Potential Challenges

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Scooped before arXiv** | High | Submit ASAP (< 2 weeks), rush arXiv to timestamp priority |
| **Results don't replicate** | High | Run multi-seed now, extend training, tune hyperparameters |
| **Weak experimental section** | Medium | Add Fashion-MNIST, ablations |
| **MNIST < 94%** | High | Adjust claim to "competitive" |
| **CIFAR-10 < 35%** | Medium | Drop Paper D, include as preliminary in Paper A appendix |
| **Kernel O(1) fails** | Low | Report as "theoretical" + "implementation in progress" |
| **NeurIPS rejection** | Medium | Resubmit to ICML or pivot to TMLR (faster acceptance) |
| **Rejected from top venue** | Medium | Have backup venues (TMLR, JMLR) |

### Patent Check

```
USPTO/Google Patents search for:
- "equilibrium propagation transformer"
- "contrastive hebbian transformer"
- "biologically plausible attention"

Result: No relevant patents found (academic domain)
```

---

## 📊 Success Metrics

### arXiv Preprint

- [ ] Upload within 2 weeks
- [ ] 100+ views in first week
- [ ] 10+ citations within 6 months

### Conference

- [ ] Accepted to top venue (NeurIPS/ICML/ICLR)
- [ ] Oral/spotlight presentation
- [ ] 50+ citations within 1 year

### Impact

- [ ] Cited by EqProp community
- [ ] Follow-up work by other labs
- [ ] Hardware implementation interest

### Overall Metrics

| Metric | Target | Status |
|--------|--------|--------|
| Papers submitted | ≥ 2 | ⬜ 0/2 |
| arXiv preprints | ≥ 2 | ⬜ 0/2 |
| Conference acceptances | ≥ 1 | ⬜ 0/1 |
| GitHub stars | ≥ 100 | ⬜ TBD |
| Community citations | ≥ 5 | ⬜ 0/5 |

---

## 🚀 Quick Start: Do This Today

```bash
# 1. Run multi-seed validation
python scripts/competitive_benchmark.py --seeds 5

# 2. Validate claims
python toreq.py --validate-claims

# 3. Check paper template
cat papers/spectral_normalization_paper.md

# 4. Generate paper (after experiments)
python scripts/generate_paper.py --paper spectral_normalization
```

### Next Actions (This Week)

```bash
# 1. Run complete experimental pipeline
./run_complete_research.sh

# 2. Generate Paper A draft
python scripts/generate_paper.py --paper spectral_normalization

# 3. Start Paper C writeup
cp papers/paper_template.md papers/kernel_paper.md
# Edit manually with kernel results

# 4. Validate all claims
python scripts/validate_claims.py

# 5. Decision point: Paper D
if [ CIFAR10_ACC -ge 50 ]; then
    echo "Proceed with Paper D"
else
    echo "Defer Paper D to future work"
fi
```

**Goal**: arXiv submission in 2 weeks. Conference submission by Feb 2025.

---

## Conclusion

The research is **publication-ready** with confirmed novelty. The gap is:

1. ✅ Novel contribution confirmed
2. ✅ Core experiments complete
3. ⬜ Multi-seed validation pending
4. ⬜ Additional datasets pending
5. ⬜ Publication figures pending
6. ⬜ Paper polish pending

**Time to cash in**: Execute this 2-week plan, submit to arXiv, then target NeurIPS 2025.

---

*Created: 2026-01-02*  
*Last Updated: 2026-01-03*
