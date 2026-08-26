# computronium-stability v0.1.0

Calibrated stability guard for dynamical neural systems — extracted from the Computronium research codebase.

## Scope Statement (Mandatory v1)

**This guard is calibrated on:**
- Settling/energy-based dynamics (energy minimization, predictive settling)
- Non-normal linear dynamics (Ginibre ensemble)
- 16 real substrate × settling-dynamics coordinates (windowed growth = 1.000, FKR 0% at τ=1.029)

**Not a v1 claim:** General-transformer collapse detection is future calibration work.

## Installation

```bash
pip install -e .
```

## Quick Start (20 lines)

```python
import torch
from computronium_stability import attach, StabilityVerdict

# Your vanilla PyTorch model
model = torch.nn.Sequential(
    torch.nn.Linear(64, 128),
    torch.nn.Tanh(),
    torch.nn.Linear(128, 64),
)

# Attach the guard (default τ=1.029, windowed_growth statistic)
guard = attach(model)

# Training loop with stability monitoring
for step in range(1000):
    x = torch.randn(32, 64)  # Batch of 32, dim 64
    y = model(x)
    loss = y.pow(2).mean()
    loss.backward()
    # ... optimizer step ...

    # Check stability
    verdict = guard.check({"x": x, "y": y, "loss": loss}, step=step)
    if verdict.kill:
        print(f"Instability detected at step {step}: max_stat={verdict.max_statistic:.4f}")
        print(f"  Decisions: {[(d.statistic_kind, d.statistic, d.kill) for d in verdict.decisions]}")
        break
else:
    print("Training completed without instability.")
```

## Core API

### `attach(model, threshold=1.029, statistic="windowed_growth", window=10, transition_fn=None)`
Returns a `GuardHandle` with a `check(state, step)` method.

### Statistics
- **`windowed_growth`** (recommended): Peak activity growth over a settling window. Tracks asymptotic divergence directly.
- **`fast_proxy`**: One-step Jacobian-vector gain. Cheaper (~10× step cost) but blind to non-normal transients.

### `StabilityVerdict`
- `kill`: True if any statistic exceeds threshold
- `decisions`: Tuple of `GuardDecision` per statistic
- `max_statistic`: Maximum statistic value observed
- `threshold`: The threshold used
- `step`: Step number

## Calibration Data

Included in the package:
- `calibration.json`: ROC calibration on Ginibre ensemble (good/bad linear dynamics)
- `family_sweep.json`: 16 real settling coordinates with windowed_growth=1.000, proxy disagreement metrics

```python
import json
from pathlib import Path

pkg_dir = Path(__file__).parent
with open(pkg_dir / "calibration.json") as f:
    calib = json.load(f)
# calib["calibration"]["windowed_growth"]["threshold"] == 1.02895...
```

## Advanced Usage

### Custom Transition Function
```python
def my_transition(state):
    x = state["x"]
    with torch.no_grad():
        y = model(x)
    return {"x": y, "hidden": y}  # Recurrent: output feeds back

guard = attach(model, transition_fn=my_transition)
```

### Using Individual Estimators
```python
from computronium_stability import (
    SpectralRadiusEstimator,
    LyapunovEstimator,
    SettlingMonitor,
    BasinStabilityEstimator,
)

rho_est = SpectralRadiusEstimator(fast_mode=True)
lyap_est = LyapunovEstimator(fast_mode=True)
settle = SettlingMonitor(tolerance=1e-4)
basin = BasinStabilityEstimator(fast_mode=True)

rho = rho_est(my_transition, state)
lyap = lyap_est(my_transition, state)
steps, norms = settle(my_transition, state)
basin_stab = basin(my_transition, settled_state)
```

## Requirements
- Python ≥3.12
- PyTorch ≥2.5
- NumPy ≥2.0

## License
MIT

## Known Duplication / DRY Debt (Post-v1 Work)

This standalone package intentionally duplicates the core algorithms that also exist in `computronium/core/stability/` to remain framework-agnostic and pip-installable without the full computronium stack.

**Remaining deduplication work:**

| Algorithm | Standalone Location | Internal Location | Deduplication Strategy |
|-----------|---------------------|-------------------|------------------------|
| Spectral radius (power iteration) | `spectral_radius.py` | `computronium/core/stability/spectral_radius.py` | Internal wraps standalone via adapter (CompositeState→dict) |
| Lyapunov exponent | `lyapunov.py` | `computronium/core/stability/lyapunov.py` | Internal wraps standalone via adapter |
| Settling time | `settling.py` | `computronium/core/stability/settling.py` | Internal wraps standalone; internal keeps `measure_settling_time_full_state` (joint-specific) |
| Basin stability | `basin.py` | `computronium/core/stability/basin.py` | Internal wraps standalone via adapter |
| Guard (`StabilityGuard`) | `guard.py` | `computronium/core/stability/guard.py` | Internal wraps standalone; internal keeps calibration utilities (`calibrate_threshold`, `quantify_proxy_disagreement`, `measure_guard_overhead`) |

**To fully DRY (future):**
1. Make `computronium/core/stability/` a thin wrapper that only handles `CompositeState`/`SystemContext` ↔ dict adaptation
2. Move calibration utilities to standalone (or make them pluggable)
3. Use `computronium-stability` as a `pyproject.toml` dependency in computronium (currently blocked by local path issues)
4. Delete duplicated algorithm implementations from `computronium/core/stability/`

**Why not done in v1:** The internal package has computronium-specific integrations (full-Jacobian spectral radius, full-state settling, calibration CLI) that don't belong in the standalone package. The wrapper approach was attempted but introduced import/runtime complexity. v1 prioritizes working, tested code over perfect deduplication.