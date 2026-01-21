# Bioplausible UI - User Experience Flows

## Primary User Journeys

### 1. Quick Training (3 clicks)
**Goal**: Train a model on MNIST with minimal friction

**Flow**:
1. Launch `biopl` → Train tab loads (default)
2. Dataset selector shows MNIST (default)
3. Click "▶ Start Training"

**Result**: Training begins immediately with sensible defaults

---

### 2. Custom Training (6 clicks)
**Goal**: Train specific model with custom hyperparameters

**Flow**:
1. Select task (Vision/LM/RL/Diffusion)
2. Select dataset
3. Select algorithm family
4. Select specific model
5. Click "🔧 Advanced" → Adjust hyperparams
6. Click "▶ Start Training"

**Result**: Full control over training configuration

---

### 3. Algorithm Comparison (5 clicks)
**Goal**: Compare EqProp vs FA vs Hebbian on same dataset

**Flow**:
1. Click "Compare" tab
2. Select dataset
3. Multi-select algorithms (Ctrl+click)
4. Click "Run All"
5. View side-by-side metrics table

**Result**: Direct performance comparison with statistical significance

---

### 4. Hyperparameter Search (3 clicks)
**Goal**: Find optimal hyperparameters for model

**Flow**:
1. Click "Search" tab
2. Select dataset + algorithm
3. Click "Start Search" (uses default search space)

**Option**: Click "Configure Search Space" for custom ranges

**Result**: Leaderboard of best configurations

---

### 5. Experimental Analysis (2 clicks)
**Goal**: Understand model dynamics

**Flow**:
1. After training in biopl, click Tools → "Open Lab"
2. Model auto-loads into biopl-lab

**Alternative**:
```bash
biopl-lab --model results/trained_model.pt
```

**Result**: Access to Microscope, Dreaming, Oracle, Alignment tools

---

## Navigation Principles

### Minimal Click Paths
- **Fastest to training**: 3 clicks
- **Cross-app workflow**: 2 clicks (Tools → Open Lab)
- **Most common actions**: ≤ 5 clicks

### Progressive Disclosure
- **Level 1** (Beginner): Dataset + Start button
- **Level 2** (Intermediate): Advanced button → Hyperparams
- **Level 3** (Expert): Lab app → Analysis tools

### Smart Defaults
- MNIST for vision, tiny_shakespeare for LM
- EqProp MLP as default model
- Standard preset (10 epochs, lr=0.001)
- Batch size auto-adjusted for GPU memory

---

## Error Handling

### Graceful Degradation
- Missing CUDA → Auto-switch to CPU with notification
- Dataset not downloaded → One-click download prompt
- Model incompatible with task → Disable in dropdown

### Clear Messages
❌ **Bad**: `RuntimeError: Expected 3D tensor`  
✅ **Good**: `Model needs 28x28 images. MNIST is preprocessed correctly.`

❌ **Bad**: `FileNotFoundError: data/mnist`  
✅ **Good**: `MNIST dataset not found. [Download Now]`

---

## Key UX Principles

1. **No frozen UI** - Training runs in QThread, UI stays responsive
2. **Immediate feedback** - Progress bar updates every batch
3. **Undo-friendly** - Can stop training, all training runs saved to Results tab
4. **Contextual help** - Tooltips on every control
5. **Smart filtering** - Model dropdown only shows task-compatible models
