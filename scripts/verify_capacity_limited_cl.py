"""Phase 3.5.2 — Capacity-Limited Continual Learning Forgetting Probe.

Run full 5-task Split-MNIST with small hidden_dim (32) to induce catastrophic
forgetting and discriminate between arms. Expected ranges per TODO5:
- backprop ~0.15
- EWC ~0.05
- replay ~0.01
- fast_weights target <=0.1
- LwF/SI should differ from backprop

This probe validates that continual learning arms actually differ in forgetting
behavior under capacity constraints.

Usage:
    uv run python scripts/verify_capacity_limited_cl.py [--arms ...] [--seeds 3] [--epochs 5] [--hidden_dim 32]
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from computronium.core.continual.runner import CLConfig, run_continual_learning_suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", default="fast_weights,ewc,backprop,replay,lwf,si")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=32,
        help="Hidden dimension (32-64 for capacity-limited probe)",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--protocol",
        default="task_incremental",
        choices=["task_incremental", "task_free"],
    )
    parser.add_argument(
        "--out", default="benchmark_results/arm_verification/capacity_limited_cl.json"
    )
    args = parser.parse_args(argv)

    from pathlib import Path

    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    config = CLConfig(
        epochs_per_task=args.epochs,
        batch_size=64,
        seed=0,
        device=device,
        hidden_dim=args.hidden_dim,
    )

    print(
        f"Running capacity-limited CL probe: hidden_dim={args.hidden_dim}, epochs={args.epochs}, seeds={args.seeds}"
    )
    print(f"Arms: {arms}")
    print(f"Protocol: {args.protocol}")
    print(f"Device: {device}")
    print()

    t0 = time.perf_counter()
    results = run_continual_learning_suite(
        arms=arms,
        protocols=[args.protocol],
        output_dir=args.out.replace(".json", ""),
        config=config,
        seeds=args.seeds,
    )
    elapsed = time.perf_counter() - t0

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"CAPACITY-LIMITED CL PROBE RESULTS (hidden_dim={args.hidden_dim})")
    print(f"{'=' * 60}")
    print(f"{'Arm':<15} {'Avg Forgetting':>15} {'BWT':>10} {'FWT':>10}")
    print(f"{'-' * 60}")

    summary = {}
    for arm, proto_results in results.items():
        for proto, res in proto_results.items():
            mean_forgetting = res.get("mean_avg_forgetting", 0.0)
            mean_bwt = res.get("mean_backward_transfer", 0.0)
            mean_fwt = res.get("mean_forward_transfer", 0.0)
            print(
                f"{arm:<15} {mean_forgetting:>15.4f} {mean_bwt:>10.4f} {mean_fwt:>10.4f}"
            )
            summary[arm] = {
                "mean_forgetting": mean_forgetting,
                "mean_bwt": mean_bwt,
                "mean_fwt": mean_fwt,
                "seeds": res.get("seeds", []),
            }

    print(f"{'-' * 60}")
    print(f"Total time: {elapsed:.1f}s")

    # Save detailed results
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(
            {
                "config": {
                    "hidden_dim": args.hidden_dim,
                    "epochs_per_task": args.epochs,
                    "seeds": args.seeds,
                    "protocol": args.protocol,
                    "device": device,
                },
                "results": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    import torch

    sys.exit(main())
