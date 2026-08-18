#!/usr/bin/env python3
"""REFACTOR7 automated multi-family kernel benchmark harness (§7.1).

Runs a finite/well-shaped probe over every registered ``KernelBackend`` pair of
(algorithm family, hardware target) and emits
``artifacts/kernel_benchmark_report.json`` with per-entry status, wall time,
peak memory, and the backend's own memory/telemetry stats.

The probe dispatch mirrors ``tests/unit/validation/test_family_kernel_parity.py``
(the DRY multi-family harness): uniform-interface families use ``forward →
backward``; bespoke families exercise their documented entry points
(``forward_positive/negative``, ``simulate``, ``settle``, ``tile_forward``,
``kernel_ops``). Accuracy *gates* live in the pytest parity suites; this tool
only automates the finite/well-shaped contract sweep the plan's ``§7.1``
describes.

Usage:
    uv run python tools/benchmark_all_kernels.py
    [--output artifacts/kernel_benchmark_report.json]
    [--kernel-type standard|contrastive|both]
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import torch
from torch import nn

from bioplausible.acceleration import (
    get_algorithm_kernels,
    get_contrastive_kernels,
)
from bioplausible.acceleration.contrastive_kernels import (
    ContrastiveConfig,
    get_contrastive_kernel,
)
from bioplausible.acceleration.kernel_backend import (
    AlgorithmFamily,
    HardwareTarget,
    KernelConfig,
    KernelRegistry,
)


def _linear_stack(
    dims: tuple[int, ...],
    device: torch.device,
    seed: int = 0,
) -> list[nn.Linear]:
    torch.manual_seed(seed)
    return [nn.Linear(dims[i], dims[i + 1]).to(device) for i in range(len(dims) - 1)]


def _rand(*shape: int, b: object, normal: bool = False) -> torch.Tensor:
    device = getattr(b, "_device", torch.device("cpu"))
    if normal:
        return torch.randn(*shape, device=device)
    return torch.rand(*shape, device=device)


# ============================================================
# Standard KernelBackend runners
# ============================================================
def _run_fa(b: object) -> dict[str, object]:
    x = _rand(6, 8, b=b, normal=True)
    out, acts = b.forward(x)
    err = _rand(6, 4, b=b, normal=True)
    grads = b.backward(acts, err)
    return {"finite": torch.isfinite(out).all().item(), "grads": grads}


def _run_ff(b: object) -> dict[str, object]:
    torch.manual_seed(0)
    x = _rand(6, 8, b=b, normal=True)
    y = torch.randint(0, 4, (6,), device=getattr(b, "_device", torch.device("cpu")))
    pos_out, pos_acts = b.forward_positive(x, y)
    neg_acts = b.forward_negative(x, y)[1]
    grads = b.backward(pos_acts, neg_acts)
    return {"out": pos_out, "grads": grads}


def _run_pepita(b: object) -> dict[str, object]:
    torch.manual_seed(0)
    x = _rand(6, 8, b=b, normal=True)
    err = _rand(6, 4, b=b, normal=True)
    std_out, std_acts = b.forward_standard(x)
    err_acts = b.forward_error_modulated(x, err)[1]
    grads = b.backward(std_acts, err_acts, err)
    return {"out": std_out, "grads": grads}


def _run_tp(b: object) -> dict[str, object]:
    torch.manual_seed(0)
    x = _rand(6, 8, b=b, normal=True)
    out, acts = b.forward_forward(x)
    targets = b.compute_targets(acts, _rand(6, 4, b=b, normal=True))
    grads = b.backward(acts, targets)
    return {"out": out, "grads": grads}


def _run_pc(b: object) -> dict[str, object]:
    torch.manual_seed(0)
    x = _rand(6, 8, b=b, normal=True)
    b.init_states(x)
    free_mu, _ = b.settle(x, y=None, steps=4)
    nudged_mu = b.settle(
        x,
        y=torch.randint(0, 4, (6,), device=getattr(b, "_device", torch.device("cpu"))),
        steps=4,
    )[0]
    grads = b.backward(x, free_mu, nudged_mu)
    return {"grads": grads}


def _run_snn(b: object) -> dict[str, object]:
    torch.manual_seed(0)
    x = _rand(6, 8, b=b)
    spike_trains, telemetry = b.simulate(x)[0], b.simulate(x)[2]
    grads = b.backward_contrastive(spike_trains, spike_trains, beta=0.5)
    return {"spike_trains": spike_trains, "grads": grads, "telemetry": telemetry}


def _run_tile(b: object) -> dict[str, object]:
    torch.manual_seed(0)
    x = _rand(6, 16, b=b, normal=True)
    out = b.tile_forward(x)[0]
    free_states, _ = b.settle(x, beta=0.0, steps=4)
    nudged_states, _ = b.settle(x, beta=0.5, steps=4)
    grads = b.backward_contrastive(free_states, nudged_states)
    return {"out": out, "grads": grads}


def _run_mep(b: object) -> dict[str, object]:
    torch.manual_seed(0)
    w = torch.randn(16, 16)
    ortho = b.muon_orthogonalize(w)
    grad = torch.randn(16, 16)
    whitened = b.fisher_whiten(grad, torch.rand(16, 16))
    settled, telemetry = b.ep_settle(
        torch.randn(6, 16),
        torch.randn(6, 16),
        torch.randn(16, 16),
        torch.zeros(16),
        torch.randn(16, 16),
        torch.zeros(16),
        steps=4,
    )
    return {
        "ortho": ortho,
        "whitened": whitened,
        "settled": settled,
        "telemetry": telemetry,
    }


def _run_o1memory(b: object) -> dict[str, object]:
    torch.manual_seed(0)
    states = [
        _rand(6, 16, b=b, normal=True),
        _rand(6, 16, b=b, normal=True),
        _rand(6, 4, b=b, normal=True),
    ]
    states, telemetry = b.settle_manual_o1(
        states, _rand(6, 4, b=b, normal=True), steps=4
    )
    return {"states": states, "telemetry": telemetry}


def _run_backprop(b: object) -> dict[str, object]:
    x = _rand(6, 8, b=b, normal=True)
    out, acts = b.forward(x)
    err = _rand(6, 4, b=b, normal=True)
    grads = b.backward(acts, err)
    return {"out": out, "grads": grads}


def _run_hebbian(b: object) -> dict[str, object]:
    x = _rand(6, 8, b=b, normal=True)
    out, acts = b.forward(x)
    grads = b.backward(acts)
    return {"out": out, "grads": grads}


# ============================================================
# Contrastive Kernel runners
# ============================================================
def _run_contrastive(b: object, family: AlgorithmFamily) -> dict[str, object]:
    """Run a contrastive step for any ContrastiveKernel."""
    torch.manual_seed(0)
    device = getattr(b, "_device", torch.device("cpu"))

    # Per-family input dimensions (matching parity harness)
    if family == AlgorithmFamily.FF:
        x = torch.randn(6, 12, device=device)  # input_dim + output_dim
        y = torch.randint(0, 4, (6,), device=device)
    elif family == AlgorithmFamily.TILE:
        x = torch.randn(6, 16, device=device)  # input_dim = 16
        y = torch.randint(0, 4, (6,), device=device)
    elif family == AlgorithmFamily.MEP or family == AlgorithmFamily.O1MEMORY:
        x = torch.randn(6, 16, device=device)  # input_dim = 16 (hidden_dim)
        y = torch.randint(0, 4, (6,), device=device)
    elif family == AlgorithmFamily.TP:
        x = torch.randn(6, 8, device=device)  # input_dim = 8
        y = torch.randn(6, 4, device=device)  # target is continuous for TP
    else:
        x = torch.randn(6, 8, device=device)
        y = torch.randint(0, 4, (6,), device=device)

    metrics = b.contrastive_step(x, y)
    logits = b.predict(x)
    acc = (logits.argmax(dim=1) == y).float().mean().item() if y.dim() == 1 else 0.0
    return {
        "finite": torch.isfinite(logits).all().item(),
        "metrics": metrics,
        "accuracy": acc,
    }


# Runner map for standard KernelBackend
_STANDARD_RUNNERS: dict[AlgorithmFamily, object] = {
    AlgorithmFamily.FA: _run_fa,
    AlgorithmFamily.FF: _run_ff,
    AlgorithmFamily.PEPITA: _run_pepita,
    AlgorithmFamily.TP: _run_tp,
    AlgorithmFamily.PC: _run_pc,
    AlgorithmFamily.SNN: _run_snn,
    AlgorithmFamily.TILE: _run_tile,
    AlgorithmFamily.MEP: _run_mep,
    AlgorithmFamily.O1MEMORY: _run_o1memory,
    AlgorithmFamily.BACKPROP: _run_backprop,
    AlgorithmFamily.HEBBIAN: _run_hebbian,
}

# Runner map for ContrastiveKernel (all families use same interface)
_CONTRASTIVE_RUNNERS: dict[AlgorithmFamily, object] = {
    AlgorithmFamily.FA: _run_contrastive,
    AlgorithmFamily.HEBBIAN: _run_contrastive,
    AlgorithmFamily.FF: _run_contrastive,
    AlgorithmFamily.PEPITA: _run_contrastive,
    AlgorithmFamily.TP: _run_contrastive,
    AlgorithmFamily.PC: _run_contrastive,
    AlgorithmFamily.SNN: _run_contrastive,
    AlgorithmFamily.TILE: _run_contrastive,
    AlgorithmFamily.MEP: _run_contrastive,
    AlgorithmFamily.O1MEMORY: _run_contrastive,
}

# Bind configuration for standard KernelBackend
_STANDARD_BIND: dict[AlgorithmFamily, tuple[dict[str, object], object]] = {
    AlgorithmFamily.FA: (
        {"input_dim": 8, "hidden_dim": 16, "output_dim": 4, "num_layers": 2},
        lambda b, d: b.set_model_ref(_linear_stack((8, 16, 4), d), nn.ReLU()),
    ),
    AlgorithmFamily.FF: (
        {"input_dim": 8, "hidden_dim": 16, "output_dim": 4, "num_layers": 2},
        lambda b, d: b.set_model_ref(_linear_stack((8 + 4, 16, 4), d), nn.ReLU()),
    ),
    AlgorithmFamily.PEPITA: (
        {
            "input_dim": 8,
            "hidden_dim": 16,
            "output_dim": 4,
            "feedback_matrix_scale": 0.1,
        },
        lambda b, d: b.set_model_ref(_linear_stack((8, 16, 4), d), nn.ReLU()),
    ),
    AlgorithmFamily.TP: (
        {"input_dim": 8, "hidden_dim": 16, "output_dim": 4, "activation": "tanh"},
        lambda b, d: b.set_model_ref(
            _linear_stack((8, 16, 4), d, seed=1),
            _linear_stack((4, 16), d, seed=2),
            nn.Tanh(),
        ),
    ),
    AlgorithmFamily.PC: (
        {
            "input_dim": 8,
            "hidden_dim": 16,
            "output_dim": 4,
            "activation": "tanh",
            "infer_steps": 4,
        },
        lambda b, d: b.set_model_ref(_linear_stack((8, 16, 4), d), "tanh"),
    ),
    AlgorithmFamily.SNN: (
        {"input_dim": 8, "hidden_dim": 16, "output_dim": 4, "num_steps": 5},
        lambda b, d: b.set_model_ref(_linear_stack((8, 16, 4), d)),
    ),
    AlgorithmFamily.TILE: (
        {
            "input_dim": 16,
            "neurons_per_tile": 8,
            "tiles_per_layer": 2,
            "num_hidden_layers": 2,
        },
        None,
    ),
    AlgorithmFamily.MEP: ({"ns_steps": 3}, None),
    AlgorithmFamily.O1MEMORY: (
        {"loss_type": "mse"},
        lambda b, d: b.set_model_ref([nn.Linear(16, 16).to(d), nn.Linear(16, 4).to(d)]),
    ),
    AlgorithmFamily.BACKPROP: (
        {"input_dim": 8, "hidden_dim": 16, "output_dim": 4, "num_layers": 2},
        lambda b, d: b.set_model_ref(_linear_stack((8, 16, 4), d)),
    ),
    AlgorithmFamily.HEBBIAN: (
        {"input_dim": 8, "hidden_dim": 16, "output_dim": 4, "num_layers": 2},
        lambda b, d: b.set_model_ref(_linear_stack((8, 16, 4), d)),
    ),
}

# Bind configuration for ContrastiveKernel
_CONTRASTIVE_BIND: dict[AlgorithmFamily, tuple[dict[str, object], object]] = {
    AlgorithmFamily.FA: (
        {"input_dim": 8, "hidden_dim": 16, "output_dim": 4, "num_layers": 2},
        lambda b, d: b.set_model_ref(_linear_stack((8, 16, 4), d), nn.ReLU()),
    ),
    AlgorithmFamily.HEBBIAN: (
        {"input_dim": 8, "hidden_dim": 16, "output_dim": 4, "num_layers": 2},
        lambda b, d: b.set_model_ref(_linear_stack((8, 16, 4), d), nn.ReLU()),
    ),
    AlgorithmFamily.FF: (
        {"input_dim": 8, "hidden_dim": 16, "output_dim": 4, "num_layers": 2},
        lambda b, d: b.set_model_ref(_linear_stack((8 + 4, 16, 4), d), nn.ReLU()),
    ),
    AlgorithmFamily.PEPITA: (
        {
            "input_dim": 8,
            "hidden_dim": 16,
            "output_dim": 4,
            "feedback_matrix_scale": 0.1,
        },
        lambda b, d: b.set_model_ref(_linear_stack((8, 16, 4), d), nn.ReLU()),
    ),
    AlgorithmFamily.TP: (
        {"input_dim": 8, "hidden_dim": 16, "output_dim": 4, "activation": "tanh"},
        lambda b, d: b.set_model_ref(
            _linear_stack((8, 16, 4), d, seed=1),
            _linear_stack((4, 16), d, seed=2),
            nn.Tanh(),
        ),
    ),
    AlgorithmFamily.PC: (
        {
            "input_dim": 8,
            "hidden_dim": 16,
            "output_dim": 4,
            "activation": "tanh",
            "infer_steps": 4,
        },
        lambda b, d: b.set_model_ref(_linear_stack((8, 16, 4), d), nn.Tanh()),
    ),
    AlgorithmFamily.SNN: (
        {"input_dim": 8, "hidden_dim": 16, "output_dim": 4, "num_steps": 5},
        lambda b, d: b.set_model_ref(_linear_stack((8, 16, 4), d)),
    ),
    AlgorithmFamily.TILE: (
        {
            "input_dim": 16,
            "neurons_per_tile": 8,
            "tiles_per_layer": 2,
            "num_hidden_layers": 2,
        },
        lambda b, d: b.set_model_ref(_linear_stack((16, 16, 16, 4), d)),
    ),
    AlgorithmFamily.MEP: (
        {"ns_steps": 3},
        lambda b, d: b.set_model_ref(_linear_stack((16, 16, 4), d)),
    ),
    AlgorithmFamily.O1MEMORY: (
        {"loss_type": "mse"},
        lambda b, d: b.set_model_ref(_linear_stack((16, 16, 4), d)),
    ),
}

# HardwareTarget → device mapping for simulation
_DEVICE_FOR_HW: dict[HardwareTarget, torch.device] = {
    HardwareTarget.CPU: torch.device("cpu"),
    HardwareTarget.CUDA: torch.device("cuda"),
    HardwareTarget.TRITON: torch.device("cuda"),
    HardwareTarget.FPGA: torch.device("cpu"),  # HLS simulation
    HardwareTarget.NEUROMORPHIC: torch.device("cpu"),  # Event sim (Loihi/NxSDK)
    HardwareTarget.OPTICAL: torch.device("cpu"),  # Wave optics sim
    HardwareTarget.CROSSBAR: torch.device("cpu"),  # SPICE/circuit sim on CPU
    HardwareTarget.QUANTUM: torch.device("cpu"),  # State vector sim
}


def _peak_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value)
        return value.detach().cpu().tolist()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _make_kernel_config(
    family: AlgorithmFamily, hw: HardwareTarget, extra: dict
) -> KernelConfig:
    return KernelConfig(
        algorithm=family,
        hardware=hw,
        settle_steps=extra.get("settle_steps", 4),
        beta=extra.get("beta", 0.5),
        gamma=extra.get("gamma", 0.95),
        extra={
            k: v for k, v in extra.items() if k not in ("settle_steps", "beta", "gamma")
        },
    )


def _make_contrastive_config(
    family: AlgorithmFamily, hw: HardwareTarget, extra: dict
) -> ContrastiveConfig:
    return ContrastiveConfig(
        algorithm=family,
        hardware=hw,
        beta=extra.get("beta", 0.5),
        lr=extra.get("lr", 0.01),
        settle_steps=extra.get("settle_steps", 30),
        gamma=extra.get("gamma", 1.0),
        extra={
            k: v
            for k, v in extra.items()
            if k not in ("beta", "lr", "settle_steps", "gamma")
        },
    )


def _probe_standard(family: AlgorithmFamily, hw: HardwareTarget) -> dict[str, object]:
    backend = KernelRegistry.get_best(family, hw)
    if backend is None:
        raise ValueError(f"No standard KernelBackend for {family.value} on {hw.value}")
    extra, bind = _STANDARD_BIND[family]
    config = _make_kernel_config(family, hw, extra)
    backend.initialize(config)
    if bind is not None:
        bind(backend, _DEVICE_FOR_HW[hw])
    result = _STANDARD_RUNNERS[family](backend)
    result["memory_stats"] = backend.get_memory_stats()
    result["telemetry"] = backend.get_settle_telemetry()
    return result


def _probe_contrastive(
    family: AlgorithmFamily, hw: HardwareTarget
) -> dict[str, object]:
    backend = get_contrastive_kernel(family)
    if backend is None:
        raise ValueError(f"No contrastive kernel for {family.value}")
    extra, bind = _CONTRASTIVE_BIND[family]
    config = _make_contrastive_config(family, hw, extra)
    backend.initialize(config)
    if bind is not None:
        bind(backend, _DEVICE_FOR_HW[hw])
    result = _CONTRASTIVE_RUNNERS[family](backend, family)
    result["memory_stats"] = backend.get_memory_stats()
    result["telemetry"] = backend.get_settle_telemetry()
    return result


def benchmark(kernel_type: str = "both") -> dict[str, object]:
    """Run benchmark for specified kernel type(s).

    Args:
        kernel_type: "standard", "contrastive", or "both"
    """
    get_algorithm_kernels()
    get_contrastive_kernels()

    report: dict[str, object] = {"schema": "benchmark_all_kernels/v2", "families": {}}
    families: dict[str, object] = {}

    for family in AlgorithmFamily:
        if family == AlgorithmFamily.EQPROP:
            continue  # EQPROP uses standalone EqPropKernel engine

        # Standard KernelBackend
        if kernel_type in ("standard", "both") and family in _STANDARD_RUNNERS:
            for hw in HardwareTarget:
                if not KernelRegistry.has(family, hw):
                    continue
                t0 = time.perf_counter()
                try:
                    result = _probe_standard(family, hw)
                    status = "ok"
                except Exception as e:
                    result, status = {}, f"error: {type(e).__name__}: {e}"
                elapsed_ms = (time.perf_counter() - t0) * 1e3
                key = f"{family.value}:standard"
                families.setdefault(key, {})[hw.value] = {
                    "status": status,
                    "time_ms": round(elapsed_ms, 3),
                    "peak_memory_mb": round(_peak_memory_mb(), 3),
                    "result": _jsonable(result),
                }

        # ContrastiveKernel
        if kernel_type in ("contrastive", "both") and family in _CONTRASTIVE_RUNNERS:
            for hw in HardwareTarget:
                if not KernelRegistry.has(family, hw):
                    continue
                t0 = time.perf_counter()
                try:
                    result = _probe_contrastive(family, hw)
                    status = "ok"
                except Exception as e:
                    result, status = {}, f"error: {type(e).__name__}: {e}"
                elapsed_ms = (time.perf_counter() - t0) * 1e3
                key = f"{family.value}:contrastive"
                families.setdefault(key, {})[hw.value] = {
                    "status": status,
                    "time_ms": round(elapsed_ms, 3),
                    "peak_memory_mb": round(_peak_memory_mb(), 3),
                    "result": _jsonable(result),
                }

    report["families"] = families
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/kernel_benchmark_report.json"),
    )
    parser.add_argument(
        "--kernel-type",
        choices=["standard", "contrastive", "both"],
        default="both",
        help="Which kernel type(s) to benchmark",
    )
    args = parser.parse_args()
    report = benchmark(args.kernel_type)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    total = sum(len(hw) for hw in report["families"].values())
    print(
        f"wrote {args.output} | {len(report['families'])} family:kernel entries, "
        f"{total} pair-entries"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
