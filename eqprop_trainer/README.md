# EqProp-Trainer Dashboard

**Self-contained PyQt6 application for training Equilibrium Propagation models**

## Installation

This package is designed to be standalone. To use:

```bash
# If extracting to separate repo
pip install PyQt6 pyqtgraph

# Or install with parent package
pip install -e ".[app]"
```

## Usage

```bash
python -m eqprop_trainer
```

## Structure

```
eqprop_trainer/
├── __init__.py          # Package initialization
├── main.py              # Entry point (GUI setup)
├── dashboard.py         # Main window UI (LM + Vision tabs)
├── worker.py            # Background training threads
├── themes.py            # Dark cyberpunk QSS theme
├── tests/               # Unit tests (if needed)
└── README.md            # This file
```

## Dependencies

**Required:**
- PyQt6 >= 6.0
- pyqtgraph >= 0.13
- eqprop-torch (parent library)

**Optional:**
- torch (for training, provided by eqprop-torch)

## Features

- 🎨 Dark cyberpunk theme with neon accents
- 📊 Live pyqtgraph plots (loss, accuracy, Lipschitz)
- 🔤 Language modeling tab (5 EqProp LM variants)
- 📷 Vision tab (16 models: 3 core + 13 bio-plausible)
- ⚙️ Threaded training (non-blocking UI)
- 🎯 Model presets and hyperparameter controls

## Standalone Usage

If this package is extracted to a separate repository:

1. Install dependencies: `pip install PyQt6 pyqtgraph`
2. Ensure `eqprop-torch` is installed
3. Run: `python -m eqprop_trainer`

## Development

The dashboard is designed to work with or without the parent repo structure. Models are imported dynamically, so it gracefully handles missing dependencies.
