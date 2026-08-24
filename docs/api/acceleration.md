# Acceleration API Reference

This document provides a complete reference for the `computronium.acceleration` module, which contains all kernel backends, export utilities, and hardware target facades.

## Module Structure

```
computronium.acceleration
├── kernel_backend.py      # KernelBackend protocol & registry
├── fa_kernels.py          # Feedback Alignment kernels
├── hebbian_kernels.py     # Hebbian / 3-factor kernels
├── ff_kernels.py          # Forward-Forward / PEPITA kernels
├── tp_kernels.py          # Target Propagation kernels
├── pc_kernels.py          # Predictive Coding kernels
├── snn_kernels.py         # Spiking STDP kernels
├── tile_kernels.py        # Tile substrate kernels
├── mep_kernels.py         # MEP (Muon/Equilibrium) kernels
├── backprop_kernels.py    # Backprop baseline kernels
├── contrastive_kernels.py # Contrastive kernels (O(1) memory)
├── eqprop_kernel_backend.py # EQPROP kernel backend adapter
├── export.py              # HLS/Verilog/NxSDK/SPICE export
├── triton_kernels.py      # Triton ops for MEP
└── __init__.py            # Public API exports
```

---

## KernelBackend Protocol

### `KernelBackend` (Protocol)

```python
class KernelBackend(Protocol):
    """Protocol for all kernel backends."""

    name: str
    supported_dtypes: tuple[torch.dtype, ...]
    supports_autograd: bool
    requires_settle: bool

    def initialize(self, config: KernelConfig) -> None: ...
    def forward(self, *args, **kwargs) -> tuple[Tensor, ...]: ...
    def backward(self, *args, **kwargs) -> dict[str, Tensor]: ...
    def update_weights(self, *args, **kwargs) -> None: ...
    def get_memory_stats(self) -> dict[str, float]: ...
    def get_settle_telemetry(self) -> SettleTelemetry | None: ...
```

**Required attributes:**
- `name`: Unique identifier for the backend
- `supported_dtypes`: Tuple of supported torch dtypes
- `supports_autograd`: Whether the backend supports PyTorch autograd
- `requires_settle`: Whether the backend requires settling iterations

**Required methods:**
- `initialize(config)`: Initialize backend with kernel config
- `forward(...)`: Forward pass returning output tensors
- `backward(...)`: Backward pass returning gradient dict
- `update_weights(...)`: Apply weight updates
- `get_memory_stats()`: Return memory usage statistics
- `get_settle_telemetry()`: Return settling telemetry (optional)

---

### `KernelConfig` (dataclass)

```python
@dataclass(frozen=True, slots=True)
class KernelConfig:
    algorithm: AlgorithmFamily
    hardware: HardwareTarget
    dtype: torch.dtype = torch.float32
    use_autograd: bool = False
    settle_steps: int = 0
    beta: float = 0.0
    gamma: float = 1.0
    spectral_norm: bool = False
    # Algorithm-specific extras via **kwargs
```

---

### `AlgorithmFamily` (StrEnum)

```python
class AlgorithmFamily(StrEnum):
    BACKPROP = "backprop"
    FEEDBACK_ALIGNMENT = "fa"
    HEBBIAN = "hebbian"
    FORWARD_FORWARD = "ff"
    TARGET_PROP = "tp"
    PREDICTIVE_CODING = "pc"
    SPIKING = "snn"
    TILE = "tile"
    MEP = "mep"
    O1MEMORY = "o1memory"
    EQPROP = "eqprop"
    CONTRASTIVE = "contrastive"
```

---

### `HardwareTarget` (StrEnum)

```python
class HardwareTarget(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"
    TRITON = "triton"
    CUPY = "cupy"
    METAL = "metal"
    FPGA = "fpga"
    NEUROMORPHIC = "neuromorphic"
    ANALOG = "analog"
```

---

## KernelRegistry

```python
class KernelRegistry:
    """Global registry for kernel backends."""

    @classmethod
    def register(cls, backend: type[KernelBackend]) -> None: ...
    @classmethod
    def get(
        cls, family: AlgorithmFamily, hardware: HardwareTarget
    ) -> KernelBackend | None: ...
    @classmethod
    def get_best(
        cls, family: AlgorithmFamily, hardware: HardwareTarget
    ) -> KernelBackend | None: ...
    @classmethod
    def list_families(cls) -> list[AlgorithmFamily]: ...
    @classmethod
    def list_backends(cls, family: AlgorithmFamily) -> list[HardwareTarget]: ...
```

**Usage:**
```python
from computronium.acceleration import KernelRegistry, AlgorithmFamily, HardwareTarget

# Get best backend for a family/hardware combination
backend = KernelRegistry.get_best(
    AlgorithmFamily.FEEDBACK_ALIGNMENT, HardwareTarget.TRITON
)

# List all available backends for a family
backends = KernelRegistry.list_backends(AlgorithmFamily.MEP)
```

---

## Kernel Backends

### FA Kernels (`fa_kernels.py`)

| Backend | Hardware | Description |
|---------|----------|-------------|
| `FAKernelBackend` | CPU, CUDA, Triton | Fused FA forward/backward |

**Features:**
- Fused forward + feedback projection
- Supports spectral norm
- Triton acceleration on GPU

---

### Hebbian Kernels (`hebbian_kernels.py`)

| Backend | Hardware | Description |
|---------|----------|-------------|
| `HebbianKernelBackend` | CPU, CUDA, Triton | 3-factor Hebbian outer products |

**Features:**
- Pre/Post/Modulator outer products
- Batched outer product kernels
- Supports eligibility traces

---

### Forward-Forward Kernels (`ff_kernels.py`)

| Backend | Hardware | Description |
|---------|----------|-------------|
| `FFKernelBackend` | CPU, CUDA, Triton | FF/PEPITA fused updates |

**Features:**
- Positive/negative pass fusion
- Goodness computation
- Layer-wise updates

---

### Target Prop Kernels (`tp_kernels.py`)

| Backend | Hardware | Description |
|---------|----------|-------------|
| `TPKernelBackend` | CPU, CUDA | Inverse network + target kernels |

**Features:**
- Inverse forward pass
- Target computation
- Layer-wise contrastive updates

---

### Predictive Coding Kernels (`pc_kernels.py`)

| Backend | Hardware | Description |
|---------|----------|-------------|
| `PCKernelBackend` | CPU, CUDA | Graph inference + error updates |

**Features:**
- Multi-layer error propagation
- Prediction error computation
- Configurable inference steps

---

### Spiking Kernels (`snn_kernels.py`)

| Backend | Hardware | Description |
|---------|----------|-------------|
| `SNNKernelBackend` | CPU, CUDA, Triton | LIF + 3-factor STDP |

**Features:**
- Leaky integrate-and-fire neurons
- Event-driven simulation
- STDP with 3-factor modulation

---

### Tile Kernels (`tile_kernels.py`)

| Backend | Hardware | Description |
|---------|----------|-------------|
| `TileKernelBackend` | CPU, CUDA, Triton | Tile substrate parallel kernels |

**Features:**
- Parallel tile updates
- Graph-structured computation
- Multiple algorithm modes (EP, FA, TP, PC, SNN)

---

### MEP Kernels (`mep_kernels.py`)

| Backend | Hardware | Description |
|---------|----------|-------------|
| `MEPKernelBackend` | CPU, CUDA, Triton | Muon/Dion/Fisher + EP settle |

**Features:**
- Newton-Schulz orthogonalization (Muon)
- Low-rank SVD (Dion)
- Fisher whitening
- Fused EP settle (Triton)

---

### Backprop Kernels (`backprop_kernels.py`)

| Backend | Hardware | Description |
|---------|----------|-------------|
| `BackpropKernelBackend` | CPU, CUDA, Triton | Fused BPTT baseline |

**Features:**
- Fused linear + activation
- Gradient accumulation
- Mixed precision support

---

### Contrastive Kernels (`contrastive_kernels.py`)

| Backend | Hardware | Description |
|---------|----------|-------------|
| `ContrastiveHebbianKernel` | CPU, CUDA | O(1) memory contrastive |

**Features:**
- Free/nudged phase separation
- O(1) activation memory
- Multiple algorithm variants

---

### EQPROP Kernel Backend (`eqprop_kernel_backend.py`)

```python
class EqPropKernelBackend:
    """Thin adapter wrapping EqPropKernel for KernelRegistry."""

    name = AlgorithmFamily.EQPROP
    supported_dtypes = (torch.float32, torch.float16, torch.bfloat16)
    supports_autograd = False
    requires_settle = True
```

**Usage:**
```python
from computronium.acceleration import (
    EqPropKernelBackend,
    KernelRegistry,
    AlgorithmFamily,
    HardwareTarget,
)

# Register (done automatically on import)
backend = KernelRegistry.get(AlgorithmFamily.EQPROP, HardwareTarget.CPU)
```

---

## Export Pipeline (`export.py`)

### High-Level Functions

```python
def export_kernel_to_hls(kernel, config) -> Path: ...
def export_kernel_to_verilog(kernel, config) -> Path: ...
def export_kernel_to_nxsdk(kernel) -> Path: ...
def export_kernel_to_spice(kernel, config) -> Path: ...
def export_kernel_to_onnx(kernel, config) -> Path: ...
```

### PyTorch Export (NEW)

```python
def export_with_torch_export(model, config, output_dir) -> Path:
    """Export using torch.export.export() + torch.onnx.export_from_ep()."""
```

**CLI:**
```bash
# Legacy export (manifest only)
uv run biopl-export-kernel --algorithm fa --target triton --output ./fa_kernel

# Trained weight export (NEW)
uv run biopl-export-trained-kernel --algorithm backprop --target cpu --epochs 20 --output ./trained_bp
```

### Export Output Structure

```
output_dir/
├── manifest.json          # Kernel metadata & config
├── state_dict.pt          # Trained weights (trained export only)
├── model.onnx             # ONNX model (optional)
└── export_summary.json    # Export statistics
```

---

## Triton Kernels (`triton_kernels.py`)

```python
class MEP_TritonOps:
    """Triton-accelerated operations for MEP."""

    @staticmethod
    def muon_orthogonalize(W: Tensor, steps: int = 5) -> Tensor: ...
    @staticmethod
    def fisher_whiten(grad: Tensor, fisher: Tensor, damping: float) -> Tensor: ...
    @staticmethod
    def ep_settle(
        x: Tensor,
        W1: Tensor,
        W2: Tensor,
        b1: Tensor,
        b2: Tensor,
        steps: int,
        beta: float,
        target: Tensor | None,
    ) -> Tensor: ...
```

---

## Usage with CoreTrainer

```python
from computronium.core.trainer import CoreTrainer, TrainerConfig

config = TrainerConfig(
    model="standard_fa",
    task="mnist",
    use_kernel=True,
    kernel_backend="triton",  # or "cuda", "cpu"
    epochs=10,
    learning_rate=0.01,
)
trainer = CoreTrainer(config)
trainer.fit()
```

---

## Mixed Precision Support

All backends declare `supported_dtypes`. Use with `CoreTrainer`:

```python
config = TrainerConfig(
    model="standard_fa",
    task="mnist",
    use_kernel=True,
    kernel_backend="triton",
    dtype=torch.float16,  # or torch.bfloat16
    epochs=10,
)
```

**Accuracy Parity Gates (REFACTOR8):**
- FP16/BF16: within 2% of FP32 on digits (15+ epochs)
- INT8: within 5% (requires quantization-aware training)

---

## Settle Telemetry

Backends with `requires_settle=True` expose telemetry:

```python
telemetry = backend.get_settle_telemetry()
# Returns SettleTelemetry with:
# - algorithm, family, steps_taken, max_steps
# - converged, final_delta, deltas
# - settle_time_ms, memory_mb
# - hardware, backend
```

This integrates with `TrainingMetrics.extra["settle_telemetry"]`.

---

## Adding Custom Kernels

1. Implement `KernelBackend` protocol
2. Register with `KernelRegistry.register(YourBackend)`
3. Add to `AlgorithmFamily` if new family

```python
from computronium.acceleration import KernelBackend, KernelRegistry, KernelConfig
from computronium.acceleration.kernel_backend import AlgorithmFamily, HardwareTarget
import torch
from torch import Tensor


class MyCustomKernel:
    name = "my_custom"
    supported_dtypes = (torch.float32, torch.float16)
    supports_autograd = True
    requires_settle = False

    def initialize(self, config: KernelConfig) -> None:
        self.config = config

    def forward(self, x: Tensor, weight: Tensor) -> tuple[Tensor, ...]:
        return (x @ weight.T,)

    def backward(
        self, grad_output: Tensor, x: Tensor, weight: Tensor
    ) -> dict[str, Tensor]:
        return {"weight": grad_output.T @ x, "input": grad_output @ weight}

    def update_weights(self, **kwargs) -> None:
        pass

    def get_memory_stats(self) -> dict[str, float]:
        return {}


KernelRegistry.register(MyCustomKernel)
```