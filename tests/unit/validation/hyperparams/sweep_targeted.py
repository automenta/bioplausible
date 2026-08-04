#!/usr/bin/env python3
"""Targeted sweep for eqprop_mlp and pepita."""

import itertools
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def _instantiate_model_with_config(
    model_name: str, input_dim: int, output_dim: int, config: dict
):
    if model_name == "eqprop_mlp":
        from bioplausible.zoo.models.eqprop.looped_mlp import LoopedMLP

        model = LoopedMLP(
            input_dim=input_dim,
            hidden_dim=config.get("hidden_dim", 64),
            output_dim=output_dim,
            use_spectral_norm=config.get("use_spectral_norm", True),
            max_steps=config.get("max_steps", 20),
            gradient_method=config.get("gradient_method", "contrastive"),
            backend="pytorch",
        )
        model.hebbian_lr = config.get("hebbian_lr", 0.001)
        return model

    elif model_name == "pepita":
        from bioplausible.zoo.models.forward_only import PEPITA

        return PEPITA(
            input_dim=input_dim,
            hidden_dim=config.get("hidden_dim", 64),
            output_dim=output_dim,
            num_layers=config.get("num_layers", 2),
            lr=config.get("lr", 0.01),
        )


def _train_model(model, x, y, epochs=3, batch_size=32):
    model.train()
    if hasattr(model, "train_step"):
        xb, yb = x[:batch_size], y[:batch_size]
        result = model.train_step(xb, yb)
        has_custom_train = result is not None
    else:
        has_custom_train = False

    if has_custom_train:
        for _ in range(epochs):
            perm = torch.randperm(len(x))
            for i in range(0, len(x), batch_size):
                idx = perm[i : i + batch_size]
                xb, yb = x[idx], y[idx]
                model.train_step(xb, yb)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        for _ in range(epochs):
            perm = torch.randperm(len(x))
            for i in range(0, len(x), batch_size):
                idx = perm[i : i + batch_size]
                xb, yb = x[idx], y[idx]
                opt.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()


def evaluate_config(
    model_name: str, config: dict, x, y, input_dim, n_classes, backprop_baseline
):
    try:
        torch.manual_seed(456)
        model = _instantiate_model_with_config(model_name, input_dim, n_classes, config)
        _train_model(model, x, y, epochs=3)

        model.eval()
        with torch.no_grad():
            logits = model(x)
            bio_acc = (logits.argmax(1) == y).float().mean().item()

        diff = backprop_baseline - bio_acc
        return {
            "config": config,
            "bio_acc": bio_acc,
            "backprop_baseline": backprop_baseline,
            "diff": diff,
            "passed": diff <= 0.05,
        }
    except Exception as e:
        return {
            "config": config,
            "bio_acc": 0.0,
            "backprop_baseline": backprop_baseline,
            "diff": 1.0,
            "passed": False,
            "error": str(e),
        }


def generate_grid(param_grid: dict[str, list]) -> list[dict]:
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def run_targeted_sweep():
    torch.manual_seed(42)
    n_samples = 500
    input_dim = 64
    n_classes = 10
    x = torch.randn(n_samples, input_dim)
    y = torch.randint(0, n_classes, (n_samples,))
    for c in range(n_classes):
        mask = y == c
        if mask.any():
            direction = torch.randn(input_dim)
            direction = direction / direction.norm() * 2.0
            x[mask] += direction * 0.8

    torch.manual_seed(123)
    bp_model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, n_classes),
    )
    opt = torch.optim.Adam(bp_model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    bp_model.train()
    for _ in range(3):
        perm = torch.randperm(len(x))
        for i in range(0, len(x), 32):
            idx = perm[i : i + 32]
            xb, yb = x[idx], y[idx]
            opt.zero_grad()
            logits = bp_model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

    bp_model.eval()
    with torch.no_grad():
        logits = bp_model(x)
        backprop_baseline = (logits.argmax(1) == y).float().mean().item()

    print(f"Backprop baseline: {backprop_baseline:.3f}")
    print("=" * 60)

    # eqprop_mlp - more targeted search near the best config
    eqprop_grid = {
        "lr": [1e-3, 3e-3],
        "beta": [0.05, 0.1, 0.2],
        "max_steps": [20, 30, 40],
        "hebbian_lr": [0.005, 0.008, 0.01, 0.015, 0.02],
    }
    eqprop_fixed = {
        "hidden_dim": 64,
        "num_layers": 2,
        "gradient_method": "contrastive",
        "use_spectral_norm": True,
    }

    # pepita - try higher LRs and maybe more layers
    pepita_grid = {
        "lr": [0.05, 0.1, 0.2, 0.3, 0.5],
        "num_layers": [2, 3],
    }
    pepita_fixed = {"hidden_dim": 64}

    models = [
        ("eqprop_mlp", eqprop_grid, eqprop_fixed),
        ("pepita", pepita_grid, pepita_fixed),
    ]

    best_configs = {}

    for model_name, param_grid, fixed_params in models:
        print(f"\n=== Targeted sweep for {model_name} ===")
        grid = generate_grid(param_grid)
        print(f"Testing {len(grid)} configurations...")

        best_result = None
        best_diff = float("inf")

        for i, params in enumerate(grid):
            config = {**fixed_params, **params}
            result = evaluate_config(
                model_name, config, x, y, input_dim, n_classes, backprop_baseline
            )

            status = "✓ PASS" if result["passed"] else "✗ FAIL"
            error = f" ({result.get('error', '')})" if "error" in result else ""
            print(
                f"  [{i + 1}/{len(grid)}] {status} diff={result['diff']:.3f} bio={result['bio_acc']:.3f} config={params}{error}"
            )

            if result["diff"] < best_diff:
                best_diff = result["diff"]
                best_result = result

            if result["passed"]:
                print("    *** FOUND PASSING CONFIG! ***")
                break

        best_configs[model_name] = best_result
        print(
            f"\nBest for {model_name}: diff={best_diff:.3f}, config={best_result['config']}"
        )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for model_name, best in best_configs.items():
        status = "✓ CAN PASS" if best["passed"] else "✗ NEEDS WORK"
        print(f"  {model_name}: {status} (best diff={best['diff']:.3f})")
        print(f"    Best config: {best['config']}")

    return best_configs


if __name__ == "__main__":
    run_targeted_sweep()
