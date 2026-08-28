"""§7 controlled experiment: equilibrium warm-start vs cold-start settling.

Hypothesis: initialising the nudged (beta>0) equilibrium settle from the
settled free-phase state — rather than re-settling from the raw feedforward
init — converges in fewer inner iterations, cutting per-step wall time at
matched accuracy.

This isolates the settling cost of the ``StandardEqProp`` contrastive
train_step (the dominant §7 time wall) by timing ``train_step`` over the same
MNIST batches for two identical models (same seed, same init): one with
``use_equilibrium_warm_start=True``, one with it False. It reports total wall
time and final train accuracy for each variant.

Usage::

    uv run python scripts/equil_warmstart_experiment.py [--max-steps 20]
        [--epochs 1] [--batch 128] [--probes 4]
"""

from __future__ import annotations

import argparse
import time

import torch
from computronium.zoo.models.eqprop.standard_eqprop import StandardEqProp

_DEFAULT_MAX_STEPS = 20

SPEEDUP_MIN: float = 1.05
SPEEDUP_NEUTRAL_LO: float = 0.9
SPEEDUP_NEUTRAL_HI: float = 1.1
ACC_DRIFT: float = 0.02


def _model(seed: int, warm_start: bool, max_steps: int) -> StandardEqProp:
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
        use_equilibrium_warm_start=warm_start,
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
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--probes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    batches = [
        (x.to(device), y.to(device)) for x, y in _mnist_batches(args.batch, args.probes)
    ]

    # Drop the first CUDA epoch to flush one-time warmup/autotune (§14), and
    # counterbalance the variant order so GPU-warmup does not confound timing.
    # ``epochs`` must be >= 3 to discard 1 warmup + time >= 2 processed epochs.
    variants = [False, True] if args.seed % 2 == 0 else [True, False]
    results: dict[str, float] = {}
    for ws in variants:
        m = _model(args.seed, warm_start=ws, max_steps=args.max_steps).to(device)
        _timed_epoch(m, batches)  # warmup epoch, not counted
        for _ in range(args.epochs - 1):
            e, a = _timed_epoch(m, batches)
        results[str(ws)] = (e, a)

    _report(args, device, results["False"], results["True"])
    return 0


def _report(
    args: argparse.Namespace,
    device: str,
    cold: tuple[float, float],
    warm: tuple[float, float],
) -> None:
    """Compare cold- vs warm-start timings/accuracy and print the verdict."""
    (t0, a0) = cold
    (t1, a1) = warm
    speedup = t0 / t1
    dacc = a1 - a0
    print(
        f"device={device} max_steps={args.max_steps} probes={args.probes} epochs={args.epochs}"
    )
    print(f"cold-start (warm_start=False): time={t0:.2f}s acc={a0:.4f}")
    print(f"warm-start (warm_start=True) : time={t1:.2f}s acc={a1:.4f}")
    print(f"speedup={speedup:.2f}x  dAcc={dacc:+.4f}")
    print("RESULT:", "WARM_START_HELPS" if speedup > SPEEDUP_MIN else "NO_SPEEDUP")
    if (
        not (SPEEDUP_NEUTRAL_LO <= speedup <= SPEEDUP_NEUTRAL_HI)
        and abs(dacc) < ACC_DRIFT
    ):
        print("Note: time differs but accuracy matched -> a pure compute win.")
    if abs(dacc) >= ACC_DRIFT:
        print(
            "Note: accuracy drifted between variants -> check convergence, not just time."
        )


if __name__ == "__main__":
    raise SystemExit(main())
