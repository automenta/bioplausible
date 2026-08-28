"""§7 controlled experiment: adaptive early-stopping threshold for eqprop.

Tests combinations of ``convergence_threshold`` and ``convergence_start`` on
``StandardEqProp`` (MNIST, 5 epochs) to find the Pareto of (time, accuracy).

Hypothesis: looser thresholds / earlier checks reduce settling iterations
without hurting accuracy. A pure compute win if time drops at matched acc.

Usage::

    uv run python scripts/equil_adaptive_stop.py --max-steps 30 --probes 40 --epochs 2
"""

from __future__ import annotations

import argparse
import itertools
import time

import torch
from computronium.zoo.models.eqprop.standard_eqprop import StandardEqProp

# Test grid: (threshold, start) pairs
THRESHOLDS = [1e-2, 1e-3, 1e-4]
STARTS = [2, 5]
GRID = list(itertools.product(THRESHOLDS, STARTS))

SPEEDUP_MIN: float = 1.05
SPEEDUP_NEUTRAL_LO: float = 0.9
SPEEDUP_NEUTRAL_HI: float = 1.1
ACC_DRIFT: float = 0.02


def _model(
    seed: int,
    max_steps: int,
    threshold: float,
    start: int,
) -> StandardEqProp:
    torch.manual_seed(seed)
    return StandardEqProp(
        config=None,
        input_dim=784,
        output_dim=10,
        hidden_dim=256,
        num_layers=2,
        use_spectral_norm=False,
        max_steps=max_steps,
        learning_rate=1e-3,
        convergence_threshold=threshold,
        convergence_start=start,
    )


def _mnist_batches(batch: int, probes: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    from torch.utils.data import DataLoader, TensorDataset
    from torchvision import datasets, transforms

    ds = datasets.MNIST(
        root="data", train=True, download=True, transform=transforms.ToTensor()
    )
    xs = torch.stack([t[0].flatten() for t in ds])
    ys = torch.tensor([t[1] for t in ds])
    loader = DataLoader(TensorDataset(xs, ys), batch_size=batch, shuffle=True)
    return [(x, y) for x, y in loader][:probes]


def _timed_epoch(
    model: StandardEqProp, batches: list[tuple[torch.Tensor, torch.Tensor]]
) -> tuple[float, float]:
    """Time one epoch of train_step calls, returning (elapsed_s, mean_acc)."""
    model.train()
    start = time.time()
    acc_sum = 0.0
    for x, y in batches:
        metrics = model.train_step(x.clone(), y)
        acc_sum += metrics["accuracy"]
    elapsed = time.time() - start
    return elapsed, acc_sum / len(batches)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--probes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batches = [
        (x.to(device), y.to(device)) for x, y in _mnist_batches(args.batch, args.probes)
    ]

    results = _run_grid(args, batches, device)
    _report(args, device, results)
    return 0


def _run_grid(
    args: argparse.Namespace,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: str,
) -> dict[tuple[float, int], tuple[float, float]]:
    """Run all (threshold, start) combos, return {(thresh, start): (time, acc)}."""
    results: dict[tuple[float, int], tuple[float, float]] = {}
    for threshold, start in GRID:
        m = _model(args.seed, args.max_steps, threshold, start).to(device)
        _timed_epoch(m, batches)  # warmup
        total_time = 0.0
        final_acc = 0.0
        for _ in range(args.epochs - 1):
            e, a = _timed_epoch(m, batches)
            total_time += e
            final_acc = a
        results[threshold, start] = (total_time, final_acc)
    return results


def _report(
    args: argparse.Namespace,
    device: str,
    results: dict[tuple[float, int], tuple[float, float]],
) -> None:
    baseline_time, baseline_acc = results[1e-3, 5]
    print(
        f"device={device} max_steps={args.max_steps} probes={args.probes} epochs={args.epochs}"
    )
    print(f"baseline (1e-3, 5): time={baseline_time:.2f}s acc={baseline_acc:.4f}")
    print()

    best: tuple[float, int] | None = None
    best_speedup = 0.0

    for (threshold, start), (t, a) in results.items():
        speedup = baseline_time / t
        dacc = a - baseline_acc
        verdict = ""
        if speedup > SPEEDUP_MIN and abs(dacc) < ACC_DRIFT:
            verdict = "WIN"
            if speedup > best_speedup:
                best_speedup = speedup
                best = (threshold, start)
        elif abs(dacc) >= ACC_DRIFT:
            verdict = "ACC_DRIFT"
        else:
            verdict = "no win"
        print(
            f"  thresh={threshold:.0e}, start={start}: time={t:.2f}s ({speedup:.2f}x) "
            f"acc={a:.4f} ({dacc:+.4f})  [{verdict}]"
        )

    print()
    if best:
        print(
            f"BEST: threshold={best[0]:.0e}, start={best[1]}  (speedup={best_speedup:.2f}x)"
        )
    else:
        print("NO WINNING CONFIGURATION")


if __name__ == "__main__":
    raise SystemExit(main())
