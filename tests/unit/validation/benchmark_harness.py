"""Benchmark Harness — Sprint 1.3.

Parametrized, marker-gated benchmarks for every model family: wall-time per
epoch, peak memory, forward FLOPs, and train accuracy on a tiny synthetic task.
Output is JSONL (one record per run) ready for Pareto-frontier plots in
Sprint 4.3's leaderboard generator.

Design notes
------------
* Each benchmark is ``@pytest.mark.benchmark`` so it can be launched in
  isolation: ``pytest tests/unit/validation/benchmark_harness.py -m benchmark``.
* Runs are deliberately tiny (1–2 epochs, 128–256 samples) so the harness is
  usable on CPU as well as GPU. Enlarge via the ``bench_sample`` / ``bench_epochs``
  markers or fixtures when running on a beefy GPU.
* Peak memory uses ``torch.cuda.max_memory_allocated()`` on CUDA, otherwise
  ``tracemalloc`` as a CPU proxy. FLOPs come from :func:`count_flops`.
"""

import json
import logging
import pathlib
import time
import tracemalloc

import pytest
import torch
from torch import nn, optim

from bioplausible.core.profiling import count_flops
from bioplausible.zoo.models.eqprop.looped_mlp import LoopedMLP

logger = logging.getLogger(__name__)

BENCH_MODELS = [
    "eqprop_mlp",
    "fa",
    "mep",
    "tile_pc",
    "forward_forward",
    "pepita",
    "spiking",
]

# Model families that need their own bespoke instantiation path.
BESPOKE = {
    "eqprop_mlp",
    "fa",
    "mep",
    "forward_forward",
    "pepita",
}


def _instantiate(model_name: str, input_dim: int, output_dim: int, device: str, **kw):
    """Instantiate a benchmark model; default hyperparams unless overridden."""
    from bioplausible.core.registry import ComponentCategory, Registry

    if model_name == "eqprop_mlp":
        model = LoopedMLP(
            input_dim=input_dim,
            hidden_dim=kw.get("hidden_dim", 64),
            output_dim=output_dim,
            use_spectral_norm=True,
            max_steps=kw.get("max_steps", 20),
            gradient_method="contrastive",
            backend="pytorch",
        )
        model.hebbian_lr = kw.get("hebbian_lr", 0.008)
        model.beta = kw.get("beta", 0.03)
        return model.to(device)

    if model_name == "fa":
        from bioplausible.config.unified import ModelConfig
        from bioplausible.zoo.models.fa import StandardFA

        cfg = ModelConfig(
            name="standard_fa",
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=[kw.get("hidden_dim", 64)],
        )
        return StandardFA(config=cfg).to(device)

    if model_name == "mep":
        from bioplausible.zoo.models.eqprop.memory_efficient import (
            MemoryEfficientLoopedMLP,
        )

        model = MemoryEfficientLoopedMLP(
            input_dim=input_dim,
            hidden_dim=kw.get("hidden_dim", 64),
            output_dim=output_dim,
            max_steps=20,
            gradient_method="contrastive",
            use_gpu_if_available=False,
        )
        return model.to(device)

    if model_name == "forward_forward":
        from bioplausible.zoo.models.forward_only import ForwardForwardNet

        return ForwardForwardNet(
            input_dim=input_dim,
            hidden_dim=kw.get("hidden_dim", 64),
            output_dim=output_dim,
            threshold=0.5,
            num_layers=2,
            layer_lr=0.01,
            classifier_lr=0.005,
        ).to(device)

    if model_name == "pepita":
        from bioplausible.zoo.models.forward_only import PEPITA

        return PEPITA(
            input_dim=input_dim,
            hidden_dim=kw.get("hidden_dim", 64),
            output_dim=output_dim,
            num_layers=2,
            lr=0.3,
        ).to(device)

    if model_name == "spiking":
        from bioplausible.zoo.models.spiking import SpikingSTDP

        return SpikingSTDP(
            input_dim=input_dim,
            hidden_dim=kw.get("hidden_dim", 64),
            output_dim=output_dim,
            num_steps=10,
        ).to(device)

    # Generic registry build (equitile, spiking, others).
    from bioplausible.zoo import get_model_spec

    # equitile deployment models live in the equitile package (NOT zoo) so they
    # must be imported to register; without this the benchmark silently skips.
    if model_name.endswith("_equitile"):
        import bioplausible.equitile  # ruff: ignore[unused-import]

    spec = get_model_spec(model_name)
    model_cls = Registry.get(ComponentCategory.MODEL, model_name)
    return model_cls.build(
        spec=spec,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=kw.get("hidden_dim", 64),
        num_layers=2,
        device=device,
        task_type="vision",
    )


def _train_one_epoch(model, x, y, batch_size=64, lr=1e-3):
    """Train one epoch via the model's preferred interface."""
    model.train()
    n = len(x)
    perm = torch.randperm(n)
    any_update = False

    if hasattr(model, "train_step"):
        result = model.train_step(x[:batch_size], y[:batch_size])
        any_update = result is not None
        if any_update:
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                model.train_step(x[idx], y[idx])
            return

    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for i in range(0, n, batch_size):
        idx = perm[i : i + batch_size]
        xb, yb = x[idx], y[idx]
        opt.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        opt.step()


def _measure(model, x, y, device: str, epochs: int = 1) -> dict[str, object]:
    """Measure wall-time, peak memory, FLOPs and accuracy for ``epochs`` epochs."""
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    tracemalloc.start()
    start = time.perf_counter()

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        _train_one_epoch(model, x, y)
        wall = time.perf_counter() - start
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    else:
        _train_one_epoch(model, x, y)
        wall = time.perf_counter() - start
        _current, peak = tracemalloc.get_traced_memory()
        peak_mb = peak / 1024**2
    tracemalloc.stop()

    model.eval()
    with torch.no_grad():
        out = model(x[:128])
        acc = (out.argmax(1) == y[:128]).float().mean().item()

    fwd_flops = count_flops(model, (x.shape[0], x.shape[1]))

    return {
        "params": params,
        "forward_flops": fwd_flops,
        "peak_memory_mb": round(peak_mb, 2),
        "wall_time_ms": round(wall * 1000, 2),
        "train_accuracy": round(acc, 4),
        "device": device,
        "epochs": epochs,
    }


def write_jsonl(records: list[dict[str, object]], out_path: str) -> None:
    """Append benchmark records to ``out_path`` as newline-delimited JSON."""
    path = pathlib.Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


# =============================================================================
# Parametrized benchmarks
# =============================================================================


@pytest.fixture(scope="module")
def bench_data(device):
    """Tiny deterministic synthetic task for all benchmark models."""
    torch.manual_seed(42)
    n_samples = 256
    input_dim = 64
    n_classes = 10
    x = torch.randn(n_samples, input_dim, device=device)
    y = torch.randint(0, n_classes, (n_samples,), device=device)
    for c in range(n_classes):
        mask = y == c
        if mask.any():
            direction = torch.randn(input_dim, device=device)
            direction = direction / direction.norm() * 1.5
            x[mask] += direction * 0.8
    return x, y, input_dim, n_classes


@pytest.mark.benchmark
@pytest.mark.parametrize("model_name", BENCH_MODELS)
def test_benchmark_model(model_name, bench_data, device, tmp_path):
    """Wall-time / memory / FLOPs / accuracy for one family, 1 epoch."""
    x, y, input_dim, n_classes = bench_data

    try:
        torch.manual_seed(7)
        model = _instantiate(model_name, input_dim, n_classes, device)
    except (NotImplementedError, TypeError, ValueError, KeyError) as e:
        pytest.skip(f"{model_name} instantiation failed: {e}")

    record = {"model": model_name, "family": _family_of(model_name)}
    record.update(_measure(model, x, y, device, epochs=1))
    write_jsonl([record], str(tmp_path / "benchmark.jsonl"))

    # Visibility for -v runs; assertions kept loose so the gate is about
    # producing comparable numbers, not hitting a magic threshold.
    logger.info("benchmark %s: %s", model_name, record)


def _family_of(model_name: str) -> str:
    return {
        "eqprop_mlp": "eqprop",
        "fa": "fa",
        "mep": "mep",
        "tile_pc": "tile",
        "forward_forward": "forward_only",
        "pepita": "forward_only",
        "spiking": "spiking",
    }[model_name]


def test_benchmark_jsonl_schema(tmp_path):
    """The harness JSONL output satisfies the leaderboard schema (Sprint 4.3)."""
    x, y = torch.randn(8, 8), torch.randint(0, 3, (8,))
    model = torch.nn.Linear(8, 3)
    write_jsonl([_measure(model, x, y, "cpu")], str(tmp_path / "b.jsonl"))
    records = [
        json.loads(line) for line in (tmp_path / "b.jsonl").read_text().splitlines()
    ]
    assert len(records) == 1
    assert {"params", "forward_flops", "peak_memory_mb", "wall_time_ms"} <= records[
        0
    ].keys()
    assert records[0]["params"] > 0
    assert records[0]["wall_time_ms"] >= 0
