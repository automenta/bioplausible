# API Reference

## Kernel Backend Infrastructure

- [Kernel Backend Development Guide](kernel_backend_guide.md) — Protocol, registry, configuration, and implementation patterns for kernel backends
- [Hardware Target Guide](hardware_targets.md) — Neuromorphic, Optical, Analog Crossbar, Quantum, and FPGA target specifications

## Core Modules

- `bioplausible.acceleration` — Multi-algorithm kernel acceleration layer
- `bioplausible.core.trainer` — Core training loop with kernel dispatch
- `bioplausible.core.registry` — Component registry with kernel backend category
- `bioplausible.core.local_learning` — Local learning rules and settling primitives

## Algorithm Families

| Family | Kernel Backend | Contrastive Kernel |
|--------|---------------|-------------------|
| EqProp | `EqPropKernel` (NumPy/CuPy/Triton) | — |
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
- `tools/benchmark_all_kernels.py` — Automated multi-family kernel benchmarking

## Configuration

See `bioplausible.config.unified.TrainerConfig` for kernel-related options:
- `use_kernel: bool` — Enable kernel backend
- `target_hardware: HardwareTarget` — Target hardware (cpu, cuda, triton, fpga, neuromorphic, optical, crossbar, quantum)
- `kernel_backend: str` — Backend selection
- `kernel_dtype: torch.dtype` — Computation precision