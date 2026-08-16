"""Kernel export pipeline (REFACTOR7 Phase 11).

Exports a trained :class:`~bioplausible.acceleration.kernel_backend.KernelBackend`
to deployment targets:

- **ONNX** — edge inference of the kernel's bound Linear stack.
- **State dict** (``.pt``) — the kernel's trained weights for re-import.
- **Hardware manifest** (``.json``) — kernel metadata plus a per-target hardware
  mapping descriptor (FPGA/HLS, Neuromorphic/NxSDK, Analog crossbar/SPICE,
  Optical/DSL, Quantum/QASM) encoding the parameters a target toolchain would
  need to synthesize the kernel.

The manifest is always written and is the authoritative artifact; ONNX export is
best-effort (wrapped so a backend without a resolvable Linear stack or an
onnx/symbolic hiccup never fails the export).
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from bioplausible.acceleration.kernel_backend import (
    AlgorithmFamily,
    HardwareTarget,
    KernelConfig,
)
from bioplausible.core.logging import get_logger

logger = get_logger()

# Per-target descriptor: the hardware mapping a kernel export should encode.
_TARGET_SPECS: dict[HardwareTarget, str] = {
    HardwareTarget.FPGA: "hls",
    HardwareTarget.NEUROMORPHIC: "nxsdk",
    HardwareTarget.OPTICAL: "dsl",
    HardwareTarget.CROSSBAR: "spice",
    HardwareTarget.CUDA: "triton",
    HardwareTarget.TRITON: "triton",
    HardwareTarget.CPU: "onnx",
    HardwareTarget.QUANTUM: "qasm",
}


@dataclass(frozen=True, slots=True)
class KernelExport:
    """Serialized kernel export artifacts."""

    manifest_path: str
    state_dict_path: str
    onnx_path: str | None


def _json_safe(value: object) -> object:
    """Recursively coerce a value into a JSON-serializable form."""
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return str(value)


def _resolve_stack(kernel: object) -> list[nn.Linear] | None:
    """Return the kernel's bound Linear stack in forward order, if any."""
    for attr in ("_layers", "_transition_modules"):
        stack = getattr(kernel, attr, None)
        if (
            isinstance(stack, (list, nn.ModuleList))
            and stack
            and all(isinstance(m, nn.Linear) for m in stack)
        ):
            resolved: list[nn.Linear] = []
            for module in stack:
                if isinstance(module, nn.Linear):
                    resolved.append(module)
            return resolved
    return None


def _activation_of(kernel: object) -> nn.Module:
    """Resolve the backend's activation into an ``nn.Module``."""
    act = getattr(kernel, "_activation", None)
    if isinstance(act, nn.Module):
        return act
    if isinstance(act, str):
        from bioplausible.acceleration.backprop_kernels import _ACTIVATIONS

        return _ACTIVATIONS.get(act, nn.Identity())
    return nn.Identity()


def _build_export_module(
    stack: list[nn.Linear], activation: nn.Module
) -> nn.Sequential:
    """Wrap the Linear stack with inter-layer activations for ONNX export."""
    layers: list[nn.Module] = []
    for i, layer in enumerate(stack):
        layers.append(layer)
        if i < len(stack) - 1 and not isinstance(activation, nn.Identity):
            layers.append(activation)
    return nn.Sequential(*layers)


def _onnx_export(module: nn.Module, input_tensor: torch.Tensor, path: Path) -> None:
    """Best-effort ONNX export, suppressing the legacy-exporter deprecation."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*TorchScript-based ONNX export.*")
        warnings.filterwarnings("ignore", message=r".*The feature will be removed.*")
        torch.onnx.export(
            module,
            (input_tensor,),
            str(path),
            export_params=True,
            opset_version=11,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            dynamo=False,
        )


def _build_manifest(
    kernel: object,
    config: KernelConfig,
    target: HardwareTarget,
    state_dict_name: str,
    onnx_name: str | None,
) -> dict[str, object]:
    """Assemble the hardware manifest for a kernel export."""
    family = getattr(kernel, "name", config.algorithm)
    supported = getattr(kernel, "supported_dtypes", (torch.float32,))
    family_name = family.value if isinstance(family, AlgorithmFamily) else str(family)
    return {
        "algorithm": family_name,
        "family": config.algorithm.value,
        "hardware_target": config.hardware.value,
        "target_spec": _TARGET_SPECS.get(config.hardware, target.value),
        "dtype": str(config.dtype),
        "supported_dtypes": [str(d) for d in supported],
        "supports_autograd": bool(getattr(kernel, "supports_autograd", False)),
        "requires_settle": bool(getattr(kernel, "requires_settle", False)),
        "memory_complexity": str(getattr(kernel, "memory_complexity", "O(L)")),
        "locality_level": str(getattr(kernel, "locality_level", "")),
        "settle_steps": config.settle_steps,
        "beta": config.beta,
        "gamma": config.gamma,
        "use_autograd": config.use_autograd,
        "spectral_norm": config.spectral_norm,
        "extra": _json_safe(config.extra),
        "state_dict": state_dict_name,
        "onnx": onnx_name,
    }


def export_kernel(
    kernel: object,
    config: KernelConfig,
    target: HardwareTarget | None = None,
    output_dir: str = "./exports",
    include_onnx: bool = True,
) -> KernelExport:
    """Export a kernel backend's weights + hardware manifest to ``output_dir``.

    Args:
        kernel: A bound ``KernelBackend`` instance (post ``initialize`` +
            ``set_model_ref``).
        config: The ``KernelConfig`` the backend was initialized with.
        target: Hardware target for the manifest descriptor (defaults to
            ``config.hardware``).
        output_dir: Directory to write artifacts into.
        include_onnx: Whether to attempt ONNX export of the Linear stack.

    Returns:
        A :class:`KernelExport` describing the written artifacts.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = target or config.hardware

    base = f"{config.algorithm.value}_{config.hardware.value}"
    state_dict_name = f"{base}_state.pt"
    state_dict_path = out / state_dict_name

    stack = _resolve_stack(kernel)
    onnx_name: str | None = None

    if stack is not None:
        # Serialize trained weights (the source of truth for the target).
        module = _build_export_module(stack, _activation_of(kernel))
        torch.save({"state_dict": module.state_dict()}, state_dict_path)

        if include_onnx:
            try:
                onnx_name = f"{base}.onnx"
                module.eval()
                sample = torch.zeros(
                    1, stack[0].in_features, dtype=config.dtype, device="cpu"
                )
                _onnx_export(module, sample, out / onnx_name)
            except (RuntimeError, TypeError, ValueError) as exc:  # pragma: no cover
                logger.warning("ONNX export skipped for %s: %s", config.algorithm, exc)
                onnx_name = None
    else:
        logger.warning(
            "Kernel %s has no bound Linear stack; writing manifest only",
            config.algorithm,
        )

    manifest_path = out / f"{base}_manifest.json"
    manifest_path.write_text(
        json.dumps(
            _build_manifest(kernel, config, target, state_dict_name, onnx_name),
            indent=2,
        )
    )

    logger.info(
        "Exported %s kernel to %s (onnx=%s)",
        config.algorithm.value,
        out,
        onnx_name or "none",
    )
    return KernelExport(
        manifest_path=str(manifest_path),
        state_dict_path=str(state_dict_path),
        onnx_path=str(out / onnx_name) if onnx_name else None,
    )


__all__ = [
    "_TARGET_SPECS",
    "KernelExport",
    "export_kernel",
]
