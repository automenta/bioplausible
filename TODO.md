# Bioplausible Development Roadmap (Refined)

> **Last Updated**: 2026-01-20  
> **Status**: 51/51 validation tracks passing, 28 models in registry, production-ready P2P mesh  
> **Key Insight**: Project is research-mature but has critical performance gaps (O(1) memory, Triton, CuPy)

---

## 🚀 Quick Wins (< 1 Day Each)

### Core Infrastructure

- [ ] **Fix CuPy CUDA_PATH detection** (4 hours) 🔥
  - File: `bioplausible/kernel.py` L32-140
  - **Current issue**: Complex detection logic still fails
  - **Fix**: Simplify to 4-source fallback (env > torch > nvcc > ldconfig)
  - **Test**: `pytest bioplausible/tests/test_kernel.py::test_cupy_gpu -v`
  - *Impact*: Unlocks 28KB kernel for GPU acceleration

- [ ] **Add @compile decorators to models** (3 hours) 🔥
  - **Discovery**: `acceleration.py` has `compile_settling_loop` decorator but NO usage found
  - **Files**: `models/looped_mlp.py`, `models/eqprop_base.py`, `models/modern_conv_eqprop.py`
  - **Pattern**:
  ```python
  from bioplausible.acceleration import compile_settling_loop
  
  @compile_settling_loop  # Add this!
  def forward_step(self, h, x_emb):
      ...
  ```
  - **Target**: 2-3× speedup from torch.compile
  - *Impact*: Free performance boost across all models

- [ ] **Enable TF32 by default** (30 min)
  - File: `bioplausible/__init__.py`
  - Add at module load:
  ```python
  from bioplausible.acceleration import enable_tf32
  enable_tf32()  # 2-3× speedup on Ampere+ GPUs
  ```
  - *Impact*: Ampere/Ada GPU users get instant 2× speedup

- [ ] **Fix pytest environment** (15 min)
  - `pip install -e .` + verify all deps
  - *Impact*: CI/CD readiness

### UI Polish

- [ ] **Add GPU/RAM monitoring** (2 hours)
  - File: `bioplausible_ui/dashboard.py`
  - Use `psutil` + `torch.cuda` for live stats
  - Add status bar to main window
  - *Impact*: Transparency during long runs

- [ ] **Add Recent Models menu** (1 hour)
  - Store last 5 in `~/.bioplausible/recent.json`
  - *Impact*: Workflow efficiency

- [ ] **Tooltip blitz** (2 hours)
  - Add `setToolTip()` to all controls in 11 tabs
  - *Impact*: Onboarding friction reduction

---

## 💎 High-Value Medium Efforts (1-3 Days)

### Core Library (Priority Order)

#### 1. **O(1) Memory in User API** (2-3 days) 🔥🔥🔥
**Rationale**: README claims O(1), but `SupervisedTrainer` uses PyTorch autograd = O(N) memory.

**Evidence**: 
- Track 35 validates kernel has O(1)
- But `models/looped_mlp.py` never calls kernel
- `training/supervised.py` L95 has `use_kernel` param but limited integration

**Implementation**:
```python
# In bioplausible/models/__init__.py
from bioplausible.kernel import EqPropKernel

class LoopedMLP:
    def __init__(self, ..., backend='auto'):  # NEW param
        if backend == 'kernel' or (backend == 'auto' and torch.cuda.is_available()):
            self._engine = EqPropKernel(...)
            self._backend = 'kernel'
        else:
            self._backend = 'pytorch'
```

**Files to modify**:
- `bioplausible/models/looped_mlp.py` (add kernel fallback)
- `bioplausible/models/eqprop_base.py` (base class support)
- `bioplausible/training/supervised.py` (wire kernel mode properly)

**Verification**:
- Extend Track 35 to test user-facing API
- Memory profile at depth 100: should be ~1× not 19×

**Effort**: 2-3 days | **Impact**: Critical - closes marketing gap

---

#### 2. **Activate Triton Kernel in Model Forward Loop** (3-5 days) 🔥🔥
**Rationale**: `models/ triton_kernel.py` (8KB) exists but unused in actual models.

**Discovery**: 
- `modern_conv_eqprop.py` L11 imports `TritonEqPropOps`
- But never actually calls it
- `acceleration.py` checks `TRITON_AVAILABLE` but has commented example

**Implementation**:
1. Audit `triton_kernel.py` for missing ops
2. Wire into `eqprop_base.py::forward_step`:
```python
if HAS_TRITON and self.use_triton:
    from bioplausible.models.triton_kernel import TritonEqPropOps
    h_next = TritonEqPropOps.fused_settling_step(h, W, x_emb)
else:
    h_next = tanh(x_emb + W @ spectral_norm(h))
```

**Target**: 10× wall-clock speedup vs PyTorch loop

**Effort**: 3-5 days | **Impact**: Major performance unlock

---

#### 3. **1000-Layer Signal Probe** (3-4 days) 🔬
**Rationale**: README has "vanishing gradient disclaimer" for 10K+ layers

**From TODO7.md Track 2.1**: "Inject perturbation at Layer 1000, measure at Layer 1"

**Implementation**:
```python
# experiments/deep_signal_probe.py
def measure_signal_propagation(depths=[10, 100, 500, 1000]):
    for depth in depths:
        model = LoopedMLP(..., num_layers=depth)
        
        # Inject perturbation at output
        with torch.no_grad():
            h_base = model.forward(x)
            h_perturbed = h_base + 0.1 * torch.randn_like(h_base)
            
        # Backpropagate perturbation via equilibrium
        # Measure Euclidean distance at each layer
        signals = measure_layer_signals(model, h_perturbed)
        
        # Compare with skip connections
        model_skip = LoopedMLP(..., num_layers=depth, use_residual=True)
        signals_skip = measure_layer_signals(model_skip, h_perturbed)
```

**Add as Track 42**:
- File: `bioplausible/validation/tracks/signal_tracks.py` (NEW)
- Success: Signal > 1% at depth 1000

**Effort**: 3-4 days | **Value**: Research publication potential

---

### UI Enhancements

#### 4. **Unified Theme System** (1 day)
**Discovery**: `themes.py` has `PLOT_COLORS` but tabs apply inconsistently

**Implementation**:
```python
# bioplausible_ui/themes.py
THEME_REGISTRY = {
    'dark': {'ui': DARK_THEME, 'plots': DARK_PLOT_COLORS},
    'light': {'ui': LIGHT_THEME, 'plots': LIGHT_PLOT_COLORS},
    'cyberpunk': {'ui': CYBER_THEME, 'plots': NEON_PLOT_COLORS},  # NEW
    'nord': {'ui': NORD_THEME, 'plots': NORD_PLOT_COLORS},       # NEW
}

# In dashboard.py
def _toggle_theme(self):
    current = self.current_theme
    next_theme = THEME_REGISTRY[(current + 1) % len(THEME_REGISTRY)]
    self._apply_theme(next_theme)
    for tab in self.tabs:
        tab.update_theme(next_theme)
```

**Effort**: 1 day | **Impact**: Visual polish

---

#### 5. **Global Keyboard Shortcuts** (2 hours)
**Files**: `bioplausible_ui/dashboard.py:_show_shortcuts()`

**Add**:
- `Ctrl+T`: Start Training
- `Ctrl+S`: Stop
- `Ctrl+G`: Generate  
- `Ctrl+R`: Reset plots
- `Ctrl+E`: Export
- `Ctrl+M`: Microscope one-click capture
- `Ctrl+?`: Show shortcuts

**Effort**: 2 hours | **Impact**: Power user efficiency

---

#### 6. **Diffusion Tab Completion** (2 days)
**Current**: `tabs/diffusion_tab.py` only 241 lines vs Vision's 1,402

**Missing**:
- Training progress plots (loss, FID)
- Live sample preview during training
- Noise schedule visualization
- CIFAR-10 support (L40 hardcoded MNIST)

**Reference**: `experiments/diffusion_mnist.py` + Track 39

**Effort**: 2 days | **Impact**: Feature parity

---

#### 7. **Lazy Plot Rendering** (1 day)
**Rationale**: 11 tabs init all plots at startup (slow)

**Pattern** (apply to all tabs):
```python
def _setup_ui(self):
    self._plots_initialized = False
    # Create placeholder widgets
    
def showEvent(self, event):
    if not self._plots_initialized:
        self._init_plots()  # Deferred creation
        self._plots_initialized = True
```

**Effort**: 1 day | **Impact**: Sub-second startup

---

## 🏗️ Deep Architecture (1-2 Weeks)

### Model Registry Expansion

**Discovery**: `models/registry.py` has 28 models but many under-tested

#### 8. **Holomorphic EP Validation Track** (3-4 days) 🔬
**Context**: `models/holomorphic_ep.py` (8KB) implements complex-valued EqProp (Laborieux NeurIPS 2024)

**Current status**: Registered but no dedicated track

**Implementation**:
- Add Track 55: `validation/tracks/advanced_eqprop_tracks.py`
- Compare gradient accuracy: Holomorphic vs Standard EP
- Benchmark convergence speed
- Test on CIFAR-10 (should beat standard EP)

**Effort**: 3-4 days | **Value**: Cutting-edge feature validation

---

#### 9. **Neural Cube 3D Visualization** (2-3 days) 🎨
**Context**: `models/neural_cube.py` implements 3D lattice with 26-neighbor connectivity

**Current**: Track 5 validates training  
**Missing**: No visualization of 3D structure

**Implementation**:
- Extend Cube Visualizer in `tabs/vision_tab.py` L215
- Add interactive 3D plot with PyQtGraph 3D
- Show activation propagation through cube
- Export to GIF/MP4

**Effort**: 2-3 days | **Value**: Compelling demo

---

### Training Infrastructure

#### 10. **P2P Evolution Improvements** (2-3 days)
**Discovery**: `p2p/evolution.py` has proof-of-accuracy (L181) but only 10% verification rate

**Enhancements**:
1. **Adaptive verification**: Verify more often when detecting anomalies
2. **Lineage visualization**: Graph of genome evolution tree
3. **Multi-task evolution**: Evolve single model for both CIFAR + MNIST
4. **Reputation system**: Track nodes that submit false results

**Files**:
- `p2p/evolution.py:_verify_model()` (increase verification rate)
- `tabs/discovery_tab.py` (add lineage graph)

**Effort**: 2-3 days | **Value**: Research network integrity

---

#### 11. **Sleep Phase Negative Learning** (2-3 days) 🔬
**From**: TODO7.md Track 4.2

**Implementation**:
```python
# In models/eqprop_base.py
def sleep_phase(self, steps=100, anti_hebbian_lr=0.01):
    """Unlearn spurious equilibria via negative phase."""
    for _ in range(steps):
        # Random noise input
        noise = torch.randn_like(self.input_buffer)
        
        # Free-run to equilibrium (no nudge)
        h_spurious = self.solve_equilibrium(noise, beta=0.0)
        
        # Anti-Hebbian update (subtract correlations)
        with torch.no_grad():
            self.W -= anti_hebbian_lr * (h_spurious @ h_spurious.T)
```

**Test on**: 1000-layer signal probe (should improve stability)

**Add as Track 43**

**Effort**: 2-3 days | **Value**: Novel contribution

---

### UI Advanced Features

#### 12. **Discovery Tab: Visual Architecture Builder** (3-4 days)
**Current**: Genome editor is JSON text edit (L110)

**Vision**: Drag-and-drop NN builder
- Palette of layer types (MLP, Conv, Attention, Equilibrium)
- Drag to canvas
- Connect with arrows
- Auto-generate config JSON
- Export to Python code (not just JSON)

**Tech**: Qt Graphics View or integrate NN-SVG API

**Effort**: 3-4 days | **Value**: Accessibility

---

#### 13. **Export to Standalone HTML Dashboard** (3 days)
**Use case**: Share results without Python

**Implementation**:
- Replace pyqtgraph with Plotly.js (web-compatible)
- Generate `results/dashboard_{timestamp}.html`
- Embed training data as inline JSON
- Include interactive plots
- No server needed (local file open)

**Files**:
- `bioplausible_ui/export.py` (NEW)
- `bioplausible_ui/templates/dashboard.html` (NEW)

**Effort**: 3 days | **Impact**: Shareability

---

## 🔬 Long-Term Research (Weeks-Months)

### Ambitious Research Directions

#### 14. **Multi-Objective NAS with Pareto Frontiers** (1-2 weeks)
**Discovery**: Evolution system has crossover/mutation but no explicit Pareto optimization

**Vision**: Optimize for accuracy + speed + memory simultaneously

**Files**:
- `evolution/pareto.py` already exists (11KB)!
- Wire into `p2p/evolution.py`

**Metrics**:
- Accuracy
- Inference time
- Training memory
- Model FLOPs

**Effort**: 1-2 weeks | **Value**: Publication-grade NAS

---

#### 15. **Hardware HDL Generation** (2-4 weeks) 🔥
**From**: TODO7.md Track 3

**Vision**: Generate Verilog for ternary neuron cells

**Prerequisites**: 
- ✅ Track 4 (Ternary Weights) passes
- Spectral Norm ternary version

**Implementation**:
1. MyHDL or direct Verilog emission
2. Target: Zero multipliers (Add/Sub/Mux only)  
3. Synthesis report vs FP32 MAC
4. FPGA deployment (PYNQ board)

**Effort**: 2-4 weeks | **Value**: Unique capability (no other library has this)

---

#### 16. **Real-Time Collaborative Training** (1-2 weeks)
**Vision**: Google Docs for neural networks

**Tech Stack**:
- WebRTC for parameter sync
- Shared session tokens via P2P DHT
- Live cursor positions on plots
- Conflict resolution for concurrent updates

**Dependencies**: 
- ✅ P2P infrastructure exists

**Effort**: 1-2 weeks | **Value**: Novel research tool

---

#### 17. **Mobile Monitoring App** (2-3 weeks)
**Platform**: iOS + Android

**Features**:
- WebSocket to dashboard
- Read-only plots
- Push notifications
- Remote stop/pause training

**Tech**: React Native

**Effort**: 2-3 weeks | **ROI**: High for overnight runs

---

#### 18. **Differentiable Architecture Search (DARTS) Variant** (2 weeks)
**Vision**: Continuous relaxation of architecture search space

**Innovation**: Use EqProp's equilibrium as relaxation mechanism
- Architecture parameters α are equilibrium states
- Bi-level optimization: inner loop (weights), outer loop (α)

**Effort**: 2 weeks | **Value**: Research paper

---

#### 19. **Curriculum Learning for EqProp** (1 week)
**Hypothesis**: EqProp benefits more from curriculum than backprop

**Experiment**:
- Start with easy examples (high confidence)
- Progress to hard examples
- Compare EqProp vs Backprop gain

**Files**: `training/curriculum.py` (NEW)

**Effort**: 1 week | **Value**: Training efficiency

---

## 🎯 Recommended 4-Week Sprint

### Week 1: Foundation (Performance Core)
**Day 1**:
- [ ] Fix CuPy CUDA_PATH (4h)
- [ ] Add pytest (15min)
- [ ] Enable TF32 default (30min)
- [ ] Add @compile decorators (3h)

**Day 2-3**:
- [ ] O(1) Memory in User API (2 days)

**Day 4-5**:
- [ ] Triton Kernel Activation (2 days)

**Target**: 10× faster training, O(1) memory proven

---

### Week 2: Research (Deep Architecture)
**Day 1-3**:
- [ ] 1000-Layer Signal Probe (3 days)
  - Track 42
  - Test with/without skip connections

**Day 4-5**:
- [ ] Sleep Phase implementation (2 days)
  - Track 43
  - Integrate with signal probe

**Target**: Address vanishing gradient disclaimer

---

### Week 3: UI Polish
**Day 1**:
- [ ] Unified theme system (1 day)

**Day 2**:
- [ ] Diffusion tab completion (1 day, part 1)
  - Training plots

**Day 3**:
- [ ] Diffusion tab completion (1 day, part 2)
  - CIFAR-10 support

**Day 4**:
- [ ] Lazy plot rendering (1 day)

**Day 5**: 
- [ ] Quick wins blitz:
  - GPU/RAM monitoring
  - Recent models menu
  - Keyboard shortcuts
  - Tooltips

**Target**: Professional-grade UI

---

### Week 4: Advanced Features (Choose 1-2)
**Option A**: P2P Evolution improvements
**Option B**: Holomorphic EP track + Neural Cube viz
**Option C**: Discovery tab visual builder

---

## 📊 Success Metrics

| Metric | Current | Target | Timeline |
|--------|---------|--------|----------|
| **Performance** | | | |
| Memory at depth 100 | ~19× | 1× | Week 1 |
| GPU kernel speed | 3× slower | 10× faster | Week 1 |
| Compile speedup | 1× (no compile) | 2-3× | Week 1, Day 1 |
| **Research** | | | |
| Signal at depth 1000 | Unknown | \u003e 1% | Week 2 |
| Tracks validated | 51 | 55+ | Week 2-4 |
| **UI** | | | |
| Startup time | ~2s | \u003c 0.5s | Week 3 |
| Themes available | 2 | 4+ | Week 3 |
| Keyboard shortcuts | 2 | 10+ | Week 3 |

---

## 🔗 References

- **Model Registry**: 28 models in `models/registry.py`
- **Acceleration Utils**: `acceleration.py` (TF32, torch.compile, backend detection)
- **P2P Evolution**: `p2p/evolution.py` (362 lines with crossover/mutation)
- **Hyperopt Tasks**: `hyperopt/tasks.py` (LM/Vision/RL abstraction)
- **Kernel**: `kernel.py` (870 lines NumPy/CuPy with O(1) memory)

---

## 🚨 Critical Path Dependencies

```mermaid
graph TD
    A[Fix CuPy] --> B[O(1) Memory API]
    A --> C[Triton Kernel]
    B --> D[Track 35 Extension]
    C --> E[10× Speedup]
    D --> F[Production Ready]
    E --> F
    
    G[Signal Probe] --> H[Sleep Phase]
    H --> I[Deep Scaling Solved]
    
    J[Theme System] --> K[UI Polish]
    L[Lazy Plots] --> K
    M[Diffusion Tab] --> K
    K --> N[Professional Product]
```

---

## Notes

- **All estimates**: Single developer, full-time
- **Parallel work**: Core vs UI tracks independent
- **Testing**: Continuous, not end-of-sprint
- **README**: Update as features complete
- **Priorities**: Can shift based on research discoveries

**Key Insight**: Project has excellent scientific foundation (51/51 tracks) but needs performance infrastructure (O(1), Triton, compile) to match the vision.
