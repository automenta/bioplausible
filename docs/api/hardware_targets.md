# Hardware Target Guide

Status: 2026-08-16 (REFACTOR7 Phase 11)

How the seven hardware targets map to facades, kernel backends, and export.

## 1. The targets

`TrainerConfig.target_hardware` accepts:
`None`/`"gpu"`, `"fpga"`, `"analog"`, `"neuromorphic"`, `"optical"`,
`"crossbar"`, `"quantum"`.

| Target | Facade model | Kernel mapping | Key params |
|--------|-------------|----------------|-----------|
| GPU / digital | none (reference) | CUDA/Triton/CuPy kernels | — |
| FPGA | `QuantizedLoopedMLP` | 8-bit quantisation | `bits` |
| Analog | `NoisyLoopedMLP` | additive device noise | `noise_level` |
| Neuromorphic | `SpikingLoopedMLP` | LIF spike-and-reset (via SNN kernel style) | `tau_mem`, `tau_syn`, `spike_threshold`, `refractory_period` |
| Optical | `OpticalLoopedMLP` | phase/detector noise | `phase_noise`, `detector_noise` |
| Crossbar | `CrossbarLoopedMLP` | conductance + ADC/DAC + IR-drop | `conductance_range`, `adc_bits`, `dac_bits`, `ir_drop_factor` |
| Quantum | `QuantumLoopedMLP` | shot-noise | `n_qubits`, `ansatz_depth`, `shot_noise` |

## 2. Facades

All non-reference facades live in
`computronium/zoo/models/eqprop/hardware_variants.py` and extend `LoopedMLP`.
They follow the `forward_dynamics` override pattern: call
`super().forward_dynamics(...)` then transform hidden activations (layers
`1..len-1`), keeping the output layer clean.

```
class NeuromorphicLoopedMLP(LoopedMLP):
    def forward_dynamics(self, activations, beta=0.0, target=None):
        activations = super().forward_dynamics(activations, beta, target)
        activations[1:-1] = self.lif_step(activations[1:-1])   # LIF dynamics
        return activations
```

`SpikingLoopedMLP` needs **batch-shaped** refractory counters (a 1-D vector
fails when batch exceeds neuron count); the `_refractory_for` helper lazily
re-allocates on batch/shape change.

## 3. Trainer wiring

`core/trainer.py::_apply_hardware` maps `TrainerConfig.target_hardware` to a
facade and `_hardware_meta_for` records the hardware meta (target, dtype,
noise descriptor). The facade replaces the configured model at
construction time; the `ModelCache` key includes `target_hardware` so facades
are cached separately from the digital reference.

Kernel acceleration is orthogonal: `TrainerConfig.use_kernel` +
`kernel_backend` (`pytorch|cupy|triton`) attaches a family `KernelBackend` to
the model; the backend computes on the corresponding hardware target, so a
`use_kernel=True` run is the digital reference's accelerated twin.

## 4. Export

`tools/export_kernel.py` (CLI `biopl-export-kernel`) writes a per-target
hardware manifest via `acceleration/export.py::export_kernel`:

| Target | Manifest descriptor | Best-effort artifact |
|--------|--------------------|---------------------|
| FPGA | `hls` | ONNX of the bound stack |
| Neuromorphic | `nxsdk` | ONNX |
| Crossbar | `spice` | ONNX |
| Optical | `dsl` | ONNX |
| Quantum | `qasm` | ONNX |
| CUDA / TRITON | `triton` | ONNX |
| CPU | `onnx` | ONNX |

The manifest is authoritative and always written; HLS/NxSDK/SPICE generators
from §6.3 of the plan are the aspirational plug-in points. `export_kernel`
currently emits a trained-state dict only when a bound model is provided;
construct via `CoreTrainer(use_kernel=True)` for trained weights.

## 5. Benchmarking targets

`tools/benchmark_all_kernels.py` sweeps every registered
`(AlgorithmFamily, HardwareTarget)` pair, checking the finite/well-shaped
contract and reporting wall time + peak memory into
`artifacts/kernel_benchmark_report.json`. Accuracy parity per family is gated
by the pytest suites, not this tool.

## 6. Gotchas

- Keep facades minimal: share `LoopedMLP` logic, use composition over
  inheritance.
- Facade noise is expected to degrade accuracy; benchmark it against the
  digital reference, don't tune it away silently.
- Quantum/optical/crossbar are *simulators*, not real backends; parameters
  are physical-motivated, not physically calibrated.
- The reference path stays digital: `target_hardware=None`/`"gpu"` never
  substitutes a facade.