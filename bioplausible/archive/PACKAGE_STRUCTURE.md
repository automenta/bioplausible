# EqProp-Torch Package Structure

This document describes the self-contained package structure for easy extraction.

## Directory Layout

```
bioplausible/                    # Main library package
├── __init__.py                  # Public API exports
├── core.py                      # EqPropTrainer (320 lines)
├── models.py                    # Native PyTorch models (380 lines)
├── lm_models.py                 # 5 LM variants (578 lines)
├── kernel.py                    # NumPy/CuPy kernel (340 lines)
├── acceleration.py              # torch.compile utilities (130 lines)
├── datasets.py                  # Data loaders (200 lines)
├── utils.py                     # ONNX, verification, presets (220 lines)
├── config.py                    # Centralized defaults (160 lines)
├── bioplausible.py              # Research algorithm wrapper (182 lines)
├── tests/                       # Unit tests
│   ├── __init__.py
│   ├── README.md
│   ├── test_library.py          # Core tests (9 tests)
│   └── test_bioplausible.py     # Algorithm tests (12 tests)
├── README.md                    # User documentation
└── ARCHITECTURE.md              # Design documentation

bioplausible_ui/                  # Dashboard application
├── __init__.py                  # Package initialization
├── main.py                      # Entry point (40 lines)
├── dashboard.py                 # Main UI (577 lines)
├── worker.py                    # Background threads (140 lines)
├── themes.py                    # QSS styling (250 lines)
├── tests/                       # UI tests (if needed)
│   └── __init__.py
└── README.md                    # App documentation

examples/                        # Usage examples (separate)
├── train_mnist.py               # Basic training example
└── verify_library.py            # Verification script
```

## Self-Containment Checklist

### bioplausible Package ✅
- [x] All code in package directory
- [x] Tests in `bioplausible/tests/`
- [x] README.md with installation & usage
- [x] ARCHITECTURE.md explaining design
- [x] No dependencies on parent repo structure
- [x] Can be extracted and published to PyPI

### bioplausible_ui Package ✅
- [x] All code in package directory
- [x] README.md with standalone usage
- [x] Graceful import handling
- [x] Can run independently if eqprop-torch installed
- [x] Can be extracted to separate repo

## Extraction Instructions  

### To extract eqprop-torch:
```bash
mkdir eqprop-torch-repo
cp -r bioplausible/ eqprop-torch-repo/
cp pyproject.toml eqprop-torch-repo/
cp examples/ eqprop-torch-repo/
```

### To extract eqprop-trainer:
```bash
mkdir eqprop-trainer-repo
cp -r bioplausible_ui/ eqprop-trainer-repo/
# Add separate pyproject.toml if publishing separately
```

## Dependencies

**eqprop-torch:**
- torch >= 2.0.0
- numpy >= 1.20.0
- Optional: cupy, datasets, tokenizers

**eqprop-trainer:**
- eqprop-torch
- PyQt6 >= 6.0
- pyqtgraph >= 0.13

Both packages are designed to work independently when properly installed.
