# Kernel Backend Development Guide

Status: 2026-08-16 (REFACTOR7 Phase 11)

This guide documents how to add or extend a `KernelBackend` for a
bio-plausible algorithm family, and how the infra consumes it at runtime.

## 1. The protocol surface

All kernels live under `bioplausible/acceleration/`. One module per family
(`fa_kernels.py`, `hebbian_kernels.py`, `ff_kernels.py`, `tp_kernels.py`,
`pc_kernels.py`, `snn_kernels.py`, `tile_kernels.py`, `mep_kernels.py`,
`backprop_kernels.py`), each downloading a `KernelBackend` implementation and
registering it lazily at import end:

```python
from bioplausible.acceleration.kernel_backend import (
    AlgorithmFamily,
    HardwareTarget,
    KernelRegistry,
)
from bioplausible.acceleration.triton_kernels import MEP_TritonOps


class MyKernelBackend:
    name = AlgorithmFamily.MYFAMILY
    supported_dtypes = (torch.float32, torch.float16, torch.bfloat16)
    supports_autograd = False
    requires_settle = True
    memory_complexity = "O(1)"
    locality_level = LocalityLevel.EQUILIBRIUM

    def initialize(self, config: KernelConfig) -> None: ...
    def forward(self, *args, **kwargs) -> tuple[Tensor, ...]: ...
    def backward(self, *args, **kwargs) -> dict[str, Tensor]: ...
    def update_weights(self, *args, **kwargs) -> None: ...
    def get_memory_stats(self) -> dict[str, float]: ...


KernelRegistry.register(AlgorithmFamily.MYFAMILY, HardwareTarget.CPU, MyKernelBackend)
```

`KernelRegistry` is a self-registering dispatch table keyed by
`(AlgorithmFamily, HardwareTarget)`; `KernelRegistry.get_best(family,
hardware)` finds a backend for the requested target and falls back across
targets. Registration happens lazily — the accel module only imports a kernel
module when `get_algorithm_kernels()` is called (tests do this in an autouse
fixture).

## 2. The two consumption routes

A backend attached via `TrainerConfig.use_kernel=True` is driven through one of
two seams in `core/trainer.py`:

- **Uniform-interface families** (FA, Backprop, Hebbian): the generic consumer
  `_run_kernel_train_step(model, backend, config, x, y, optimizer)` binds the
  backend to the model's `nn.Linear` stack and calls
  `forward -> backward -> update_weights`. It requires `forward(x)` returning
  `(out, activations)` and `backward(activations, error) -> {name: grad}`. The
  gradients are routed through the model's own optimizer (or the trainer's),
  and the backend activation is synced to the model via
  `_resolve_model_activation`.
- **Bespoke-dynamics kernels** (FF, PEPITA, TP, PC, SNN, Tile, MEP, O1): expose
  a `kernel_train_step(model, config, x, y, optimizer)` method. When present it
  is authoritative — `dispatch_train_step` prefers it and falls through to the
  model's own `train_step` only if it returns `None`. PEPITA
  (`acceleration/ff_kernels.py`) is the working template: mirror the model's
  reference `train_step` dynamics exactly, read the model's `output_dim` / `lr`
  / layer stacks, apply in-place updates, and return `{"loss", "accuracy",
  "logits"}`.

Rule of thumb: **derive internal matrices from the bound model, never from the
`extra` dim hints alone.** FA feedback weights are rebuilt from the bound
layers' `in_features`/`out_features`; config `num_layers` can disagree with
resolved hidden dims and cause `IndexError`s.

## 3. Settle/telemetry surface

Families with settling dynamics must record and expose their settle loop via
`get_settle_telemetry()`. The unified protocol is
`bioplausible/core/local_learning/settling.py::SettleProtocol` with
`settle_universal()`, `SettleConfig`, `SettleTelemetry`. The shared primitives
`settle_state` / `settle_activations_list` / `energy_gradient_descent` /
`settle_manual_o1` are reused; a backend simply stores the last loop's
telemetry dict in `self._last_settle_telemetry` and returns it.

## 4. Parity gates

Every backend must pass the DRY multi-family harness
`tests/unit/validation/test_family_kernel_parity.py`. Adding a backend = add a
`_make_*`/`_run_*` pair and register it in the `HARNESSES` dict. The harness
checks registry contract, `initialize` + finite memory stats, finite
forward/backward/update, and (for settling families) non-empty settle
telemetry.

The consolidated accuracy gates for families with model-side learners live in
`tests/integration/test_kernel_accuracy_parity.py`. Bespoke families proven to
*learn* through the dispatch seam: FA, Backprop, PEPITA, TP.

## 5. MEP `backend="triton"` toggle

MEP update strategies accept `backend: Literal["pytorch", "triton"]` (default
"pytorch"). When "triton", the heavy ops route through `MEP_TritonOps`
(`acceleration/triton_kernels.py`): muon NS orthogonalization, Dion low-rank
SVD, Fisher whitening, EP settling. The Triton kernels auto-fallback to their
own PyTorch reference implementations, and those fallbacks reproduce the
strategy references exactly (muon/dion/fisher parity gates are in
`tests/unit/acceleration/test_mep_backend_triton.py`). This is the opt-in
kernel path for the MEP presets (`zoo/mep/presets/__init__.py`).

## 6. Debugging / gotchas

- Use `torch.isfinite(x).all()` liberally in probes; the most common backend bug
  class is division by an unnormalized quantity (e.g. Newton-Schulz without an
  up-front norm clamp).
- `except E, F` (comma-separated tuple) is valid Python 3.14+ per PEP 758 and
  parses as `except (E, F)`; it is not the old Python-2 exception binding.
- Do not import `tests/` modules from production code.