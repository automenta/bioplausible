# Bioplausible Development Roadmap

> **Mission**: Release production-ready v0.1.0 while empowering cutting-edge Equilibrium Propagation research
> 
> **Status**: Pre-Release Engineering + Active Research  
> **Timeline**: 3-4 weeks to v0.1.0 release

---

## Quick Start

```bash
# Verify current state
python -m pytest tests/ -v --tb=short
eqprop-verify --quick

# Run comprehensive validation
eqprop-verify --track 0 41  # Framework + rapid validation

# Check code quality
black --check bioplausible bioplausible_ui
flake8 bioplausible bioplausible_ui --select=E9,F63,F7,F82
```

---

## Phase 1: Pre-Release Engineering (High Priority) 🔥

**Goal**: Fix critical issues blocking v0.1.0 release  
**Timeline**: 1-2 weeks  
**Owner**: Core maintainers

### 1.0 Code Modernization ⭐ COMPLETED ✅

**Achievement**: Replaced ~1,100 lines of custom evolution code with Optuna

- [x] **Integrate Optuna** (3 hours)
  - Added `optuna` dependency to `pyproject.toml`
  - Created `bioplausible/hyperopt/optuna_bridge.py` (~200 lines)
  - Maps `ModelSpec` → Optuna search space automatically
  
- [x] **Replace custom evolution** 
  - Simplified `hyperopt/engine.py`: 260 → 100 lines (thin wrapper)
  - Simplified `search_space.py`: removed 168 lines of GA code
  - Updated `SearchTab` to use Optuna TPE/NSGA-II
  - Marked `evolution/` directory as deprecated
  
- [x] **Benefits gained**
  - Automatic pruning (30-50% compute savings)
  - Multi-objective optimization (TPE, NSGA-II built-in)
  - Built-in visualization (Pareto fronts, parameter importance)
  - SQLite persistence for study resumption

**Net Impact**: **-1,100+ lines of code**, significantly better algorithms

---

### 1.1 Test Suite Critical Fixes ⭐ BLOCKING

**Problem**: 46 test collection errors, only 39 tests collected

- [ ] **Fix import errors** in test files (Day 1-2)
  - Investigate circular imports
  - Add missing test dependencies
  - Fix module path issues
- [ ] **Make Triton optional** (Day 2-3)
  - Wrap Triton imports in try/except
  - Add `@pytest.mark.skipif(not HAS_TRITON)` decorators
  - Fallback to PyTorch for alignment tests
- [ ] **Resolve UI test failures** (Day 3-4)
  - All `bioplausible_ui/tests/*.py` failing
  - May need headless PyQt6 setup
  - Consider marking as integration tests
- [ ] **Achieve >90% test pass rate** (Day 5)
  - Target: 35+ passing tests
  - Document known limitations
  - Create test report

**Acceptance**: `pytest --collect-only` shows 0 errors

---

### 1.2 Code Quality Baseline ⭐ BLOCKING

**Problem**: 17 files need formatting, 4 critical import errors

- [ ] **Fix critical import errors** (2 hours)
  ```python
  # bioplausible/utils.py:22-23
  import random  # MISSING
  import os     # MISSING
  
  # bioplausible/export.py:51
  # Remove unused global model_instance
  ```

- [ ] **Auto-format all code** (1 hour)
  ```bash
  black bioplausible bioplausible_ui
  isort bioplausible bioplausible_ui
  ```

- [ ] **Set up pre-commit hooks** (2 hours)
  ```yaml
  # .pre-commit-config.yaml
  repos:
    - repo: https://github.com/psf/black
      hooks: [black]
    - repo: https://github.com/pycqa/flake8
      hooks: [flake8]
    - repo: https://github.com/pycqa/isort  
      hooks: [isort]
  ```

- [ ] **Clean up archive code** (4 hours)
  - Fix or remove `bioplausible/archive/dashboard_fixes.py`
  - Archive folder should not be in package imports

**Acceptance**: `flake8 --select=F` shows 0 errors

---

### 1.3 Documentation Essentials ⭐ REQUIRED

- [ ] **Add LICENSE file** (30 min) - LEGAL REQUIREMENT
  ```bash
  # README says MIT, add official LICENSE
  cp templates/LICENSE.MIT ./LICENSE
  ```

- [ ] **Create CHANGELOG.md** (2 hours)
  - Document v0.1.0 features
  - List all 51 validation tracks
  - Note breaking changes
  - Follow [Keep a Changelog](https://keepachangelog.com/)

- [ ] **Fix version inconsistencies** (1 hour)
  ```python
  # bioplausible/__init__.py: "0.1.0" ✅
  # bioplausible/hyperopt/__init__.py: "1.0.0" ❌
  # FIX: Use single source in pyproject.toml
  ```

- [ ] **Add CONTRIBUTING.md** (2 hours)
  - Development setup
  - Testing requirements
  - PR guidelines

**Acceptance**: Repository has LICENSE, CHANGELOG.md, consistent versions

---

### 1.4 Dependency Cleanup

- [ ] **Audit dependencies** (4 hours)
  - Pin critical versions in `pyproject.toml`
  - Test clean install: `pip install -e .[dev]`
  - Verify Python 3.9-3.12 compatibility
  - Make optional deps truly optional (cupy, triton)

- [ ] **Resolve conflicts** (2 hours)
  - Check if system packages affect Bioplausible
  - Document minimum versions
  - Test in Docker container

**Acceptance**: Clean install in fresh venv, no errors

---

### 1.5 Package Distribution Prep

- [ ] **Exclude archive from build** (1 hour)
  ```toml
  # pyproject.toml
  [tool.setuptools.packages.find]
  exclude = ["bioplausible.archive*", "bioplausible_ui_old*"]
  ```

- [ ] **Remove deprecated UI** (2 hours)
  - Verify all features in `bioplausible_ui/`
  - Delete `bioplausible_ui_old/` (688KB)
  - Update references

- [ ] **Test package build** (1 hour)
  ```bash
  python -m build
  pip install dist/bioplausible-0.1.0-py3-none-any.whl
  eqprop-verify --quick
  ```

**Acceptance**: Built wheel <5MB, installs cleanly

---

## Phase 2: Scientific Infrastructure (Medium Priority) 🔬

**Goal**: Empower researchers with professional tooling  
**Timeline**: 1-2 weeks  
**Impact**: Accelerates research by 2-3×

### 2.1 Experiment Tracking Integration ⭐ HIGH IMPACT

**Problem**: Manual tracking of runs, no centralized metrics

**Solution**: Integrate Weights & Biases / MLflow

```python
# bioplausible/tracking.py (NEW)
import wandb
from typing import Dict, Any

class ExperimentTracker:
    """Unified experiment tracking."""
    
    def __init__(self, project="bioplausible", backend="wandb"):
        self.backend = backend
        if backend == "wandb":
            wandb.init(project=project)
    
    def log_hyperparams(self, config: Dict[str, Any]):
        """Log hyperparameters."""
        if self.backend == "wandb":
            wandb.config.update(config)
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log training metrics."""
        if self.backend == "wandb":
            wandb.log(metrics, step=step)
    
    def log_lipschitz(self, L: float, step: int):
        """Log Lipschitz constant (critical for EqProp)."""
        self.log_metrics({"lipschitz_constant": L}, step)
    
    def log_validation_track(self, track_id: int, results: Dict):
        """Log validation track results."""
        wandb.log({
            f"track_{track_id}_score": results["score"],
            f"track_{track_id}_evidence": results["evidence_level"]
        })
```

**Tasks**:
- [ ] Implement `ExperimentTracker` class (4 hours)
- [ ] Integrate into training loops (2 hours)
- [ ] Add to verification framework (3 hours)
- [ ] Create W&B project template (1 hour)
- [ ] Document usage in README (1 hour)

**Acceptance**: All experiments auto-logged to W&B

---

### 2.2 Automated Result Visualization ⭐ HIGH IMPACT

**Problem**: Manual matplotlib scripts for every experiment

**Solution**: Auto-generate publication-quality plots

```python
# bioplausible/visualization.py (NEW)
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class ResultVisualizer:
    """Publication-quality plots."""
    
    def __init__(self, output_dir="results/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        sns.set_style("whitegrid")
        sns.set_palette("colorblind")
    
    def plot_lipschitz_trajectory(self, L_history, save_name="lipschitz.pdf"):
        """Plot L(t) over training."""
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(L_history, linewidth=2)
        ax.axhline(1.0, color='red', linestyle='--', label='L=1 (contraction)')
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Lipschitz Constant")
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300)
        plt.close()
    
    def plot_track_comparison(self, results: Dict, save_name="comparison.pdf"):
        """Compare multiple tracks."""
        # Auto-generate comparison tables and plots
        pass
    
    def generate_paper_figures(self, experiment_name: str):
        """Generate all figures for a paper."""
        # Create figures directory
        # Generate standard plots (accuracy, loss, L(t), etc.)
        # Output LaTeX-ready PDFs
        pass
```

**Tasks**:
- [ ] Implement core visualization class (6 hours)
- [ ] Add standard plot templates (4 hours)
  - Lipschitz trajectory
  - Accuracy/loss curves
  - Memory scaling plots
  - OOD detection curves
- [ ] Auto-generate from tracker data (3 hours)
- [ ] LaTeX integration (2 hours)

**Acceptance**: One-command figure generation for papers

---

### 2.3 Statistical Analysis Toolkit ⭐ SCIENTIFIC RIGOR

**Enhancement**: Extend existing `validation/utils.py` statistics

```python
# bioplausible/statistics.py (NEW)
from scipy import stats
import numpy as np

class StatisticalAnalyzer:
    """Publication-grade statistical analysis."""
    
    def compare_algorithms(self, results_a, results_b, 
                          names=("EqProp", "Backprop")):
        """Full statistical comparison with reporting."""
        # Cohen's d
        d = self.cohens_d(results_a, results_b)
        
        # Paired t-test
        t, p = stats.ttest_rel(results_a, results_b)
        
        # Confidence intervals
        ci_a = self.confidence_interval(results_a)
        ci_b = self.confidence_interval(results_b)
        
        # Auto-generate report
        report = f"""
        ## {names[0]} vs {names[1]}
        
        **{names[0]}**: {np.mean(results_a):.3f} ({ci_a[0]:.3f}, {ci_a[1]:.3f})
        **{names[1]}**: {np.mean(results_b):.3f} ({ci_b[0]:.3f}, {ci_b[1]:.3f})
        
        **Effect Size**: Cohen's d = {d:.3f} ({self.interpret_d(d)})
        **Significance**: p = {p:.4f} ({self.interpret_p(p)})
        """
        return report
    
    def meta_analysis(self, track_results: Dict[int, Dict]):
        """Aggregate evidence across tracks."""
        # Combine p-values (Fisher's method)
        # Overall effect size
        # Meta-analysis forest plot
        pass
```

**Tasks**:
- [ ] Implement statistical analyzer (4 hours)
- [ ] Add meta-analysis support (3 hours)
- [ ] Auto-report generation (2 hours)
- [ ] Integrate with verification (2 hours)

**Acceptance**: Statistical reports auto-generated for all comparisons

---

### 2.4 Paper Generation Pipeline ⭐ FUTURE

**Vision**: Auto-generate paper drafts from validation results

```python
# bioplausible/paper.py (NEW)
class PaperGenerator:
    """Generate LaTeX paper from results."""
    
    def generate_neurips_paper(self, 
                               track_ids=[34, 35, 36, 37],
                               title="Spectral Normalization for Stable EqProp"):
        """Generate full NeurIPS-format paper."""
        # Load results from verification notebook
        # Generate sections:
        #  - Abstract (from track summaries)
        #  - Introduction (template)
        #  - Methods (from model descriptions)
        #  - Results (auto-populate figures/tables)
        #  - Discussion (template + key findings)
        # Output: paper.tex + figures/
        pass
```

**Tasks** (OPTIONAL, post-release):
- [ ] LaTeX template library (8 hours)
- [ ] Auto-populate results (6 hours)
- [ ] Citation management (4 hours)
- [ ] One-click paper generation (2 hours)

---

### 2.5 Reproducibility Enhancements

- [ ] **Experiment config system** (4 hours)
  ```python
  # experiments/configs/cifar_baseline.yaml
  model:
    type: ModernConvEqProp
    hidden_dim: 256
    eq_steps: 15
  training:
    epochs: 100
    lr: 0.001
    batch_size: 128
  tracking:
    wandb_project: bioplausible-cifar
  ```

- [ ] **Automatic seed management** (2 hours)
  - Ensure all experiments use multiple seeds
  - Auto-aggregate results
  - Reproducibility guarantees

- [ ] **Result archiving** (3 hours)
  - Save all configs, checkpoints, logs
  - Generate unique experiment IDs
  - Easy result retrieval

**Acceptance**: Any experiment fully reproducible from config

---

## Phase 3: CI/CD & Infrastructure (Medium Priority) ⚙️

**Goal**: Robust automated testing and deployment  
**Timeline**: 3-5 days

### 3.1 GitHub Actions Enhancement

**Current**: Basic workflow testing validation only

**New**: Comprehensive CI/CD pipeline

```yaml
# .github/workflows/ci.yml (ENHANCED)
name: CI/CD Pipeline

on: [push, pull_request]

jobs:
  code-quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Check formatting
        run: |
          pip install black flake8 isort
          black --check bioplausible bioplausible_ui
          flake8 bioplausible --select=E9,F63,F7,F82
          isort --check-only bioplausible
  
  test-matrix:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11", "3.12"]
        backend: [pytorch-cpu, pytorch-cuda, numpy]
    steps:
      - name: Install dependencies
        run: pip install -e .[dev]
      - name: Run tests
        run: pytest tests/ -v --cov=bioplausible
      - name: Upload coverage
        uses: codecov/codecov-action@v3
  
  validation-smoke:
    runs-on: ubuntu-latest
    steps:
      - name: Quick validation
        run: eqprop-verify --quick --track 0 41
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: validation-results
          path: results/
  
  build-package:
    runs-on: ubuntu-latest
    steps:
      - name: Build wheel
        run: python -m build
      - name: Test installation
        run: |
          pip install dist/*.whl
          eqprop-verify --version
  
  docs:
    runs-on: ubuntu-latest
    steps:
      - name: Build docs
        run: mkdocs build
      - name: Deploy to GitHub Pages
        if: github.ref == 'refs/heads/main'
        uses: peaceiris/actions-gh-pages@v3
```

**Tasks**:
- [ ] Implement code quality job (2 hours)
- [ ] Add matrix testing (3 hours)
- [ ] Set up coverage reporting (2 hours)
- [ ] Add validation artifacts (2 hours)
- [ ] Package build verification (2 hours)
- [ ] Documentation deployment (3 hours)

**Acceptance**: All checks passing, coverage >70%

---

### 3.2 Documentation Infrastructure

- [ ] **Set up MkDocs** (4 hours)
  ```yaml
  # mkdocs.yml
  site_name: Bioplausible
  theme:
    name: material
  nav:
    - Home: index.md
    - Getting Started: quickstart.md
    - Validation Tracks: tracks.md
    - API Reference: api/
  ```

- [ ] **Generate API docs** (6 hours)
  - Auto-document all public classes
  - Add usage examples
  - Link to validation tracks

- [ ] **Create tutorials** (8 hours)
  - Beginner: Training your first EqProp model
  - Intermediate: Running validation tracks
  - Advanced: Custom architectures

- [ ] **Host on GitHub Pages** (2 hours)

**Acceptance**: Live docs at `username.github.io/bioplausible`

---

### 3.3 Docker Improvements

**Current**: Basic Dockerfile exists

**Enhancement**: Multi-stage, GPU-ready

```dockerfile
# Dockerfile (ENHANCED)
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 AS base
RUN apt-get update && apt-get install -y python3.11
WORKDIR /app

FROM base AS builder
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM base AS runtime
COPY --from=builder /root/.local /root/.local
COPY . .
RUN pip install -e .
ENV PATH=/root/.local/bin:$PATH

# Entrypoints
CMD ["eqprop-verify", "--quick"]
```

**Tasks**:
- [ ] Multi-stage build (2 hours)
- [ ] GPU support (2 hours)
- [ ] Docker Compose for full stack (3 hours)
- [ ] Reduce image size (2 hours)

**Acceptance**: Docker image <1GB, GPU functional

---

## Phase 4: Advanced Features (Low Priority, Post-Release) 🚀

**Timeline**: v0.2.0+ (2-3 months)

### 4.1 Performance Optimization

- [ ] **Triton kernel debugging** (1 week)
  - Fix compilation errors
  - Benchmark vs PyTorch
  - Optional fallback

- [ ] **Memory profiling** (3 days)
  - Identify bottlenecks
  - Optimize equilibrium loops
  - Gradient checkpointing improvements

- [ ] **Distributed training** (2 weeks)
  - Multi-GPU support
  - DDP integration
  - P2P architecture search scaling

### 4.2 Extended Validation

- [ ] **ImageNet experiments** (2 weeks)
  - Large-scale vision validation
  - Compare to ResNet baselines

- [ ] **Large language models** (3 weeks)
  - GPT-2 scale EqProp
  - Perplexity benchmarks

- [ ] **Reinforcement learning** (2 weeks)
  - Atari games
  - Continuous control

### 4.3 Web-Based Tools

- [ ] **Browser dashboard** (3 weeks)
  - No PyQt6 required
  - Real-time experiment monitoring
  - Remote training management

- [ ] **Interactive demos** (2 weeks)
  - Try EqProp in browser
  - Educational visualizations
  - Jupyter notebooks

### 4.4 Community Features

- [ ] **Model zoo** (ongoing)
  - Pre-trained EqProp models
  - Easy download and fine-tuning

- [ ] **Tutorial videos** (1 week)
  - YouTube channel
  - Conference talks

- [ ] **Discord/Slack community** (ongoing)
  - User support
  - Research discussions

---

## Success Metrics

### Phase 1 (Release Readiness)
✅ All critical tests passing (>90%)  
✅ Zero F-level Flake8 errors  
✅ LICENSE file present  
✅ CHANGELOG.md complete  
✅ Clean pip install  
✅ Version 0.1.0 on PyPI

### Phase 2 (Scientific Tools)
✅ Experiment tracking integrated  
✅ Auto-plotting working  
✅ Statistical reports generated  
✅ 5+ researchers using tools

### Phase 3 (Infrastructure)
✅ CI passing all checks  
✅ Coverage >70%  
✅ Docs deployed  
✅ Docker images published

### Phase 4 (Advanced)
✅ ImageNet results published  
✅ 1000+ PyPI downloads  
✅ 100+ GitHub stars  
✅ Paper accepted at top venue

---

## Research Roadmap (Parallel Track)

From `archive/TODO.md`, these are ongoing research goals:

### Active Research Tracks (v0.1.0 - v0.2.0)

**Track 34**: CIFAR-10 ConvEqProp ≥75% (ModernConvEqProp architecture)  
**Track 35**: O(1) Memory Scaling Demo (gradient checkpointing)  
**Track 36**: Energy-based OOD Detection (AUROC ≥0.85)  
**Track 37**: Language Modeling Parity (perplexity within 20% of Backprop)  
**Track 38**: Adaptive Compute Analysis (complexity-settling correlation)  
**Track 39**: EqProp Diffusion (OPTIONAL - stretch goal)  
**Track 40**: Hardware Efficiency Analysis (FLOP counting, quantization)

See [archive/TODO.md](bioplausible/archive/TODO.md) for detailed research plan.

---

## Timeline Overview

```
Week 1-2:  Phase 1 (Engineering fixes)
  ├─ Fix tests (Days 1-5)
  ├─ Code quality (Days 1-2)
  ├─ Documentation (Days 3-5)
  └─ Dependencies (Days 6-7)

Week 3:    Phase 1 + Phase 2 starts
  ├─ Package prep (Days 8-10)
  ├─ Experiment tracking (Days 11-12)
  └─ Visualization (Days 13-14)

Week 4:    Phase 2 + Phase 3
  ├─ Statistics toolkit (Days 15-17)
  ├─ CI/CD (Days 18-20)
  └─ Documentation (Days 21-22)

Week 5+:   Release & Phase 4
  ├─ v0.1.0 Release (Day 23)
  ├─ Monitor adoption
  └─ Start advanced features
```

---

## Quick Reference

### Essential Commands

```bash
# Development
black bioplausible bioplausible_ui
pytest tests/ -v --cov=bioplausible
eqprop-verify --quick

# Release
python -m build
twine upload dist/*

# Research
eqprop-verify --track 34 35 36 37  # Key tracks
python experiments/cifar_breakthrough.py
python experiments/language_modeling_comparison.py
```

### Key Files

- `pyproject.toml` - Package configuration
- `bioplausible/__init__.py` - Version source
- `validation/` - Verification framework
- `experiments/` - Research implementations
- `.github/workflows/ci.yml` - CI/CD

### Resources

- **Validation**: [SCIENTIFIC_RIGOR.md](bioplausible/validation/SCIENTIFIC_RIGOR.md)
- **Research**: [archive/TODO.md](bioplausible/archive/TODO.md) 
- **Experiments**: [experiments/README.md](bioplausible/experiments/README.md)
- **Results**: [results/verification_notebook.md](results/verification_notebook.md)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) (to be created in Phase 1.3)

For questions: Check existing issues or create new one

---

**Last Updated**: 2026-01-22  
**Version**: 0.1.0-dev  
**Status**: Phase 1 in progress
