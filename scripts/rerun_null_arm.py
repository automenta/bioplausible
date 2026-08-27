#!/usr/bin/env python
"""Re-run B2 null arm (directed_ep, feedback_gain=0) with config_extras tag.

Outputs to a separate directory tree so null-arm runs don't mix with
regular directed_ep runs. This gives a clean comparison.
"""

import subprocess
import sys
from itertools import product
from pathlib import Path


def run_profile(
    model: str,
    task: str,
    num_layers: int,
    hidden_dim: int,
    beta: float,
    lr: float,
    epochs: int,
    batch_size: int,
    seed: int,
    device: str,
    output_dir: Path,
    feedback_gain: float | None = None,
) -> subprocess.CompletedProcess:
    """Run a single contrastive profile."""
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/contrastive_profile.py",
        "--model",
        model,
        "--task",
        task,
        "--num-layers",
        str(num_layers),
        "--hidden-dim",
        str(hidden_dim),
        "--beta",
        str(beta),
        "--learning-rate",
        str(lr),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--seed",
        str(seed),
        "--device",
        device,
        "--output-dir",
        str(output_dir),
    ]
    if feedback_gain is not None:
        cmd.extend(["--feedback-gain", str(feedback_gain)])
    print(f"Running: {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300)


def main():
    output_base = Path("runs/contrastive_profile/b2_autopsy/null_arm")
    output_base.mkdir(parents=True, exist_ok=True)

    depths = [1, 2, 3, 4]
    seeds = [0, 1, 2]
    betas = [0.01, 0.03, 0.1]
    hidden_dim = 256
    batch_size = 128
    lr = 0.05
    epochs = 1
    device = "cuda"
    task = "digits"

    all_configs = []
    for depth, seed, beta in product(depths, seeds, betas):
        all_configs.append((depth, seed, beta))

    print(f"Total null arm runs: {len(all_configs)}", flush=True)

    failed = []
    for i, (depth, seed, beta) in enumerate(all_configs):
        print(
            f"\n=== Run {i + 1}/{len(all_configs)}: null_arm depth={depth} seed={seed} beta={beta} ===",
            flush=True,
        )
        run_dir = output_base / f"beta{beta}" / f"depth{depth}_seed{seed}"
        result = run_profile(
            model="directed_ep",
            task=task,
            num_layers=depth,
            hidden_dim=hidden_dim,
            beta=beta,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            device=device,
            output_dir=run_dir,
            feedback_gain=0.0,
        )
        if result.returncode != 0:
            print(f"FAILED: {result.stderr[-300:]}", flush=True)
            failed.append((depth, seed, beta, result.stderr[-300:]))
        else:
            print("OK", flush=True)

    print("\n=== SUMMARY ===")
    print(f"Total: {len(all_configs)}")
    print(f"Failed: {len(failed)}")
    if failed:
        for depth, seed, beta, err in failed:
            print(f"  depth={depth} seed={seed} beta={beta}: {err[:200]}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
