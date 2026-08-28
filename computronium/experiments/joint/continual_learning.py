"""Continual Learning Flagship (Phase 2).

The scientific centerpiece: ψ/θ decoupling prevents catastrophic forgetting
without a replay buffer.

Arms (from computronium.core.system_trainer):
- FastWeightPlasticity (ψ/θ decoupling via fast weights)
- ElasticConsolidationUpdate (EWC - θ regularization)
- Backprop+SGD (baseline)
- Replay buffer (matched total memory)
- LwF (Learning without Forgetting)
- Synaptic Intelligence

Protocols:
- Task-incremental (task boundaries signaled)
- Task-free (no boundaries, gradual shift)

Metrics:
- Backward transfer matrix
- Forgetting measure per boundary
- Memory footprint (replay storage vs ψ state)
- Stability rider (ρ(J_F), windowed growth during ψ-adaptation)
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Re-export constants from continual module
# Re-export all public API from system_trainer for backward compatibility
from computronium.core.system_trainer import (
    CLConfig,
    run_continual_learning_suite,
)


def main():
    parser = argparse.ArgumentParser(
        description="Continual Learning Flagship (Phase 2)"
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["fast_weights", "ewc", "backprop", "replay", "lwf", "si"],
    )
    parser.add_argument(
        "--protocols", nargs="+", default=["task_incremental", "task_free"]
    )
    parser.add_argument("--output-dir", default="benchmark_results/continual_learning")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.epochs = 1
        args.seeds = 1

    config = CLConfig(
        epochs_per_task=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        hidden_dim=args.hidden_dim,
    )

    run_continual_learning_suite(
        arms=args.arms,
        protocols=args.protocols,
        output_dir=Path(args.output_dir),
        config=config,
        seeds=args.seeds,
    )


if __name__ == "__main__":
    main()
