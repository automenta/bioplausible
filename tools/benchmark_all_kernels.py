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


def _run_fa(b: object) -> dict[str, object]:
    x = _rand(6, 8, b=b, normal=True)
    out, acts = b.forward(x)  # type: ignore[attr-defined]
    err = _rand(6, 4, b=b, normal=True)
    grads = b.backward(acts, err)  # type: ignore[attr-defined]
    return {"finite": torch.isfinite(out).all().item(), "grads": grads}


def _run_ff(b: object) -> dict[str, object]:
    torch.manual_seed(0)
    x = _rand(6, 8, b=b, normal=True)
    y = torch.randint(0, 4, (6,), device=getattr(b, "_device", torch.device("cpu")))
    pos_out, pos_acts = b.forward_positive(x, y)  # type: ignore[attr-defined]
    neg_acts = b.forward_negative(x, y)[1]  # type: ignore[attr-defined]
    grads = b.backward(pos_acts, neg_acts)  # type: ignore[attr-defined]
    return {"out": pos_out, "grads": grads}


def _run_pepita(b: object) -> dict[str, object]:
    torch.manual_seed(0)
    x = _rand(6, 8, b=b, normal=True)
    err = _rand(6, 4, b=b, normal=True)
    std_out, std_acts = b.forward_standard(x)  # type: ignore[attr-defined]
    err_acts = b.forward_error_modulated(x, err)[1]  # type: ignore[attr-defined]
    grads = b.backward(std_acts, err_acts, err)  # type: ignore[attr-defined]
    return {"out": std_out, "grads": grads}


def _run_tp(b: object) -> dict[str, object]:
    torch.manual_seed(0)
    x = _rand(6, 8, b=b, normal=True)
    out, acts = b.forward_forward(x)  # type: ignore[attr-defined]
    targets = b.compute_targets(acts, _rand(6, 4, b=b, normal=True))  # type: ignore[attr-defined]
    grads = b.backward(acts, targets)  # type: ignore[attr-defined]
    return {"out": out, "grads": grads}


def _run_pc(b: object) -> dict[str, object]:
    torch.manual_seed(0)
    x = _rand(6, 8, b=b, normal=True)
    b.init_states(x)  # type: ignore[attr-defined]
    free_mu, _ = b.settle(x, y=None, steps=4)  # type: ignore[attr-defined]
    nudged_mu = b.settle(
        x,
        y=torch.randint(0, 4, (6,), device=getattr(b, "_device", torch.device("cpu"))),
        steps=4,
    )[0]  # type: ignore[attr-defined]
    grads = b.backward(x, free_mu, nudged_mu)  # type: ignore[attr-defined]
    return {"grads": grads}


def _run_snn(b: object) -> dict[str, object]:
    torch.manual_seed(0)
    x = _rand(6, 8, b=b)
    spike_trains, telemetry = b.simulate(x)[0], b.simulate(x)[2]  # type: ignore[attr-defined]
    grads = b.backward_contrastive(spike_trains, spike_trains, beta=0.5)  # type: ignore[attr-defined]
    return {"spike_trains": spike_trains, "grads": grads, "telemetry": telemetry}


def _run_tile(b: object) -> dict[str, object]:
    torch.manual_seed(0)
    x = _rand(6, 16, b=b, normal=True)
    out = b.tile_forward(x)[0]  # type: ignore[attr-defined]
    free_states, _ = b.settle(x, beta=0.0, steps=4)  # type: ignore[attr-defined]
    nudged_states, _ = b.settle(x, beta=0.5, steps=4)  # type: ignore[attr-defined]
    grads = b.backward_contrastive(free_states, nudged_states)  # type: ignore[attr-defined]
    return {"out": out, "grads": grads}


def _run_mep(b: object) -> dict[str, object]:
    torch.manual_seed(0)
    w = torch.randn(16, 16)
    ortho = b.muon_orthogonalize(w)  # type: ignore[attr-defined]
    grad = torch.randn(16, 16)
    whitened = b.fisher_whiten(grad, torch.rand(16, 16))  # type: ignore[attr-defined]
    settled, telemetry = b.ep_settle(  # type: ignore[attr-defined]
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
    states, telemetry = b.settle_manual_o1(  # type: ignore[attr-defined]
        states, _rand(6, 4, b=b, normal=True), steps=4
    )
    return {"states": states, "telemetry": telemetry}


def _run_backprop(b: object) -> dict[str, object]:
    x = _rand(6, 8, b=b, normal=True)
    out, acts = b.forward(x)  # type: ignore[attr-defined]
    err = _rand(6, 4, b=b, normal=True)
    grads = b.backward(acts, err)  # type: ignore[attr-defined]
    return {"out": out, "grads": grads}


def _run_hebbian(b: object) -> dict[str, object]:
    x = _rand(6, 8, b=b, normal=True)
    out, acts = b.forward(x)  # type: ignore[attr-defined]
    grads = b.backward(acts)  # type: ignore[attr-defined]
    return {"out": out, "grads": grads}


_RUNNERS: dict[AlgorithmFamily, object] = {
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

# (family -> extra config + optional set_model_ref builder). Mirrors the parity
# harness: backends that build internal matrices need matching dims. Layer
# stacks are placed on the backend's compute device (hardware target).
_BIND: dict[AlgorithmFamily, tuple[dict[str, object], object]] = {
    AlgorithmFamily.FA: (
        {"input_dim": 8, "hidden_dim": 16, "output_dim": 4, "num_layers": 2},
        lambda b, d: b.set_model_ref(_linear_stack((8, 16, 4), d), nn.ReLU()),  # type: ignore[attr-defined]
    ),
    AlgorithmFamily.FF: (
        {"input_dim": 8, "hidden_dim": 16, "output_dim": 4, "num_layers": 2},
        lambda b, d: b.set_model_ref(  # type: ignore[attr-defined]
            _linear_stack((8 + 4, 16, 4), d), nn.ReLU()
        ),
    ),
    AlgorithmFamily.PEPITA: (
        {
            "input_dim": 8,
            "hidden_dim": 16,
            "output_dim": 4,
            "feedback_matrix_scale": 0.1,
        },
        lambda b, d: b.set_model_ref(_linear_stack((8, 16, 4), d), nn.ReLU()),  # type: ignore[attr-defined]
    ),
    AlgorithmFamily.TP: (
        {"input_dim": 8, "hidden_dim": 16, "output_dim": 4, "activation": "tanh"},
        lambda b, d: b.set_model_ref(  # type: ignore[attr-defined]
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
        lambda b, d: b.set_model_ref(_linear_stack((8, 16, 4), d), "tanh"),  # type: ignore[attr-defined]
    ),
    AlgorithmFamily.SNN: (
        {"input_dim": 8, "hidden_dim": 16, "output_dim": 4, "num_steps": 5},
        lambda b, d: b.set_model_ref(_linear_stack((8, 16, 4), d)),  # type: ignore[attr-defined]
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
        lambda b, d: b.set_model_ref(  # type: ignore[attr-defined]
            [nn.Linear(16, 16).to(d), nn.Linear(16, 4).to(d)]
        ),
    ),
    AlgorithmFamily.BACKPROP: (
        {"input_dim": 8, "hidden_dim": 16, "output_dim": 4, "num_layers": 2},
        lambda b, d: b.set_model_ref(_linear_stack((8, 16, 4), d)),  # type: ignore[attr-defined]
    ),
    AlgorithmFamily.HEBBIAN: (
        {"input_dim": 8, "hidden_dim": 16, "output_dim": 4, "num_layers": 2},
        lambda b, d: b.set_model_ref(_linear_stack((8, 16, 4), d)),  # type: ignore[attr-defined]
    ),
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


def _make_config(family: AlgorithmFamily, hw: HardwareTarget) -> KernelConfig:
    extra, _ = _BIND[family]
    return KernelConfig(
        algorithm=family,
        hardware=hw,
        settle_steps=4,
        beta=0.5,
        gamma=0.95,
        extra=dict(extra),
    )


def _device_for(hw: HardwareTarget) -> torch.device:
    return torch.device(
        "cuda" if hw in {HardwareTarget.CUDA, HardwareTarget.TRITON} else "cpu"
    )


def _probe(family: AlgorithmFamily, hw: HardwareTarget) -> dict[str, object]:
    backend = KernelRegistry.get_best(family, hw)
    config = _make_config(family, hw)
    backend.initialize(config)  # type: ignore[attr-defined]
    _, bind = _BIND[family]
    if bind is not None:
        bind(backend, _device_for(hw))
    result = _RUNNERS[family](backend)
    result["memory_stats"] = backend.get_memory_stats()  # type: ignore[attr-defined]
    result["telemetry"] = backend.get_settle_telemetry()  # type: ignore[attr-defined]
    return result


def benchmark() -> dict[str, object]:
    get_algorithm_kernels()  # trigger lazy self-registration
    report: dict[str, object] = {"schema": "benchmark_all_kernels/v1", "families": {}}
    families: dict[str, object] = {}
    for family in AlgorithmFamily:
        if family not in _RUNNERS:
            continue
        for hw in HardwareTarget:
            if not KernelRegistry.has(family, hw):
                continue
            t0 = time.perf_counter()
            try:
                result = _probe(family, hw)
                status = "ok"
            except Exception as e:
                result, status = {}, f"error: {type(e).__name__}: {e}"
            elapsed_ms = (time.perf_counter() - t0) * 1e3
            families.setdefault(family.value, {})[hw.value] = {
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
    args = parser.parse_args()
    report = benchmark()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    covered = sum(len(hw) for hw in report["families"].values())
    print(
        f"wrote {args.output} | {len(report['families'])} families, "
        f"{covered} pair-entries"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
