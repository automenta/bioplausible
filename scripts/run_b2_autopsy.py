#!/usr/bin/env python
"""Run B2 Autopsy (Plan 8 Track B2) — pre-registered protocol.

Pre-registration (Session 15.4):
- models: eqprop, directed_ep, directed_ep(feedback_gain=0) [null arm]
- depths 1–4; slope fit on 2–4 only; depth 1 reported, excluded from slope
- hidden_dim 256, batch 128, seeds {0,1,2}
- beta ∈ {0.01, 0.03, 0.1}; lr 0.05
- ratio = mean over steps 1–9 (step-0 transient excluded),
  early = first hidden layer
- report per-depth median ratio + bootstrap CI;
  OLS slope of log(ratio) vs depth with CI
- G1 tripwire unchanged; the slope is the evidence
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
    output_base = Path("runs/contrastive_profile/b2_autopsy")
    output_base.mkdir(parents=True, exist_ok=True)

    # Pre-registered parameters
    models = ["eqprop", "directed_ep"]
    depths = [1, 2, 3, 4]
    seeds = [0, 1, 2]
    betas = [0.01, 0.03, 0.1]
    hidden_dim = 256
    batch_size = 128
    lr = 0.05
    epochs = 1  # Profile first epoch (10 steps)
    device = "cuda"
    task = "digits"

    all_configs = []

    # Main models
    for model, depth, seed, beta in product(models, depths, seeds, betas):
        all_configs.append((model, depth, seed, beta, None))

    # Null arm: directed_ep with feedback_gain=0
    for depth, seed, beta in product(depths, seeds, betas):
        all_configs.append(("directed_ep", depth, seed, beta, 0.0))

    print(f"Total runs: {len(all_configs)}", flush=True)

    failed = []
    for i, (model, depth, seed, beta, fb_gain) in enumerate(all_configs):
        print(
            f"\n=== Run {i + 1}/{len(all_configs)}: {model} depth={depth} seed={seed} beta={beta} fb_gain={fb_gain} ===",
            flush=True,
        )
        # Organize output by model/beta so analyze-depths can find them
        run_dir = output_base / f"{model}_beta{beta}" / f"depth{depth}_seed{seed}"
        result = run_profile(
            model=model,
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
            feedback_gain=fb_gain,
        )
        if result.returncode != 0:
            print(f"FAILED: {result.stderr}", flush=True)
            failed.append((model, depth, seed, beta, fb_gain, result.stderr))
        else:
            print("OK", flush=True)

    print("\n=== SUMMARY ===")
    print(f"Total: {len(all_configs)}")
    print(f"Failed: {len(failed)}")
    if failed:
        for model, depth, seed, beta, fb_gain, err in failed:
            print(
                f"  {model} depth={depth} seed={seed} beta={beta} fb_gain={fb_gain}: {err[:200]}"
            )
        sys.exit(1)

    # Now run depth-scale analysis per model/beta
    print("\n=== Running depth-scale analysis ===")
    for model in ["eqprop", "directed_ep"]:
        for beta in betas:
            model_dir = output_base / f"{model}_beta{beta}"
            if model_dir.exists():
                print(f"Analyzing {model_dir}...")
                result = subprocess.run(
                    [
                        "uv",
                        "run",
                        "python",
                        "scripts/contrastive_profile.py",
                        "analyze-depths",
                        "--output-dir",
                        str(model_dir),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if result.returncode == 0:
                    print(result.stdout)
                else:
                    print(f"Analysis failed: {result.stderr}")

    # For null arm, the runs are mixed in the same directories
    # We'll need a separate analysis script for that


if __name__ == "__main__":
    main()
