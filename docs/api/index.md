# API Reference

## Kernel Backend Infrastructure

- [Kernel Backend Development Guide](kernel_backend_guide.md) — Protocol, registry, configuration, and implementation patterns for kernel backends
- [Hardware Target Guide](hardware_targets.md) — Neuromorphic, Optical, Analog Crossbar, Quantum, and FPGA target specifications

## Strategy Permutations

- [Strategy Permutation Guide](../strategy_permutations.md) — Generic `make_strategy_optimizer()`, presets, compatibility matrix, and benchmarking

## Core Modules

- `bioplausible.acceleration` — Multi-algorithm kernel acceleration layer
- `bioplausible.core.trainer` — Core training loop with kernel dispatch
- `bioplausible.core.registry` — Component registry with kernel backend category
- `bioplausible.core.local_learning` — Local learning rules and settling primitives
- `bioplausible.core.optimization.factory` — Strategy permutation factory (`make_strategy_optimizer`)

## Algorithm Families

| Family | Kernel Backend | Contrastive Kernel |
|--------|---------------|-------------------|
| EqProp | `EqPropKernelBackend` (uniform), `EqPropKernel` (NumPy/CuPy/Triton) | — |
| Feedback Alignment | `FAKernelBackend` | `FAContrastiveKernel` |
| Hebbian / 3-Factor | `HebbianKernelBackend`, `ThreeFactorKernelBackend` | `HebbianContrastiveKernel` |
| Forward-Forward | `FFKernelBackend` | `FFContrastiveKernel` |
| PEPITA | `PEPITAKernelBackend` | `PEPITAContrastiveKernel` |
| Target Propagation | `TPKernelBackend` | `TPContrastiveKernel` |
| Predictive Coding | `PCKernelBackend` | `PCContrastiveKernel` |
| Spiking STDP | `SNNKernelBackend` | `SNNContrastiveKernel` |
| Tile Substrate | `TileKernelBackend` | `TileContrastiveKernel` |
| MEP Presets | `MEPKernelBackend` | `MEPContrastiveKernel` |
| O1MemoryEPv2 | `O1MemoryEPv2KernelBackend` | `O1MemoryContrastiveKernel` |
| Backprop Baseline | `BackpropKernelBackend` | — |

## CLI Tools

- `biopl-export-kernel` — Export kernel backends to hardware targets (ONNX, HLS, NxSDK, SPICE)
- `biopl-export-trained-kernel` — Train kernel-backed model and export trained weights
- `tools/benchmark_all_kernels.py` — Automated multi-family kernel benchmarking
- `tools/benchmark_strategy_permutations.py` — Strategy permutation benchmarking

## Configuration

See `bioplausible.config.unified.TrainerConfig` for kernel-related options:
- `use_kernel: bool` — Enable kernel backend
- `target_hardware: HardwareTarget` — Target hardware (cpu, cuda, triton, fpga, neuromorphic, optical, crossbar, quantum)
- `kernel_backend: str` — Backend selection
- `kernel_dtype: torch.dtype` — Computation precision

Strategy optimizer options (via `optimizer` and `optimizer_kwargs`):
- `optimizer: str` — Preset name (e.g., `muon_pc`, `backprop_plain`, `smep`) or custom strategy
- `optimizer_kwargs.lr` — Base learning rate
- `optimizer_kwargs.beta` — Nudge strength for contrastive/EP methods
- `optimizer_kwargs.energy_fn` — Required for EP-based presets (`smep`, `sdmep`, etc.)