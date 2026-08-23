#!/usr/bin/env python3
"""Hyperparameter sweep for backprop parity tests.

Runs test_backprop_parity.py with different hyperparameters for each model
to find configs that achieve 5% parity target.
"""

import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from bioplausible.core.registry import ComponentCategory, Registry
from bioplausible.zoo import get_model_spec


@dataclass
class SweepConfig:
    model_name: str
    param_grid: dict[str, list]
    fixed_params: dict


# Define sweep configurations for each model
SWEEPS = [
    SweepConfig(
        model_name="eqprop_mlp",
        param_grid={
            "lr": [1e-3, 3e-3, 1e-2],
            "beta": [0.1, 0.3, 0.5],
            "max_steps": [20, 30, 50],
            "gradient_method": ["contrastive"],
            "use_spectral_norm": [True],
            "hebbian_lr": [0.005, 0.01, 0.02, 0.05],
        },
        fixed_params={"hidden_dim": 64, "num_layers": 2},
    ),
    SweepConfig(
        model_name="directed_ep",
        param_grid={
            "lr": [1e-3, 3e-3, 1e-2, 3e-2],
            "beta": [0.1, 0.3, 0.5, 1.0],
            "eq_steps": [20, 30, 50],
        },
        fixed_params={"hidden_dim": 64, "num_layers": 2},
    ),
    SweepConfig(
        model_name="forward_forward",
        param_grid={
            "threshold": [0.5, 1.0, 1.5, 2.0],
            "layer_lr": [0.01, 0.03, 0.05, 0.1],
            "classifier_lr": [0.005, 0.01, 0.02],
        },
        fixed_params={"hidden_dim": 64, "num_layers": 2},
    ),
    SweepConfig(
        model_name="pepita",
        param_grid={
            "lr": [0.005, 0.01, 0.02, 0.05, 0.1],
        },
        fixed_params={"hidden_dim": 64, "num_layers": 2},
    ),
]


def _instantiate_model_with_config(
    model_name: str, input_dim: int, output_dim: int, config: dict
):
    """Instantiate a model with specific hyperparameters."""
    spec = get_model_spec(model_name)
    model_cls = Registry.get(ComponentCategory.MODEL, model_name)

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
        # Set hebbian_lr after instantiation (via parent EqPropModel)
        model.hebbian_lr = config.get("hebbian_lr", 0.001)
        return model

    elif model_name == "directed_ep":
        from bioplausible.zoo.models.eqprop.deep_ep import DirectedEP

        from bioplausible.config.unified import ModelConfig

        model_config = ModelConfig(
            name="directed_ep",
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=[config.get("hidden_dim", 64)] * config.get("num_layers", 2),
            learning_rate=config.get("lr", 1e-3),
            beta=config.get("beta", 0.3),
            max_steps=config.get("eq_steps", 20),
        )
        return DirectedEP(config=model_config, device="cpu")

    elif model_name == "forward_forward":
        from bioplausible.zoo.models.forward_only import ForwardForwardNet

        return ForwardForwardNet(
            input_dim=input_dim,
            hidden_dim=config.get("hidden_dim", 64),
            output_dim=output_dim,
            threshold=config.get("threshold", 2.0),
            num_layers=config.get("num_layers", 2),
            layer_lr=config.get("layer_lr", 0.03),
            classifier_lr=config.get("classifier_lr", 0.01),
        )

    elif model_name == "pepita":
        from bioplausible.zoo.models.forward_only import PEPITA

        return PEPITA(
            input_dim=input_dim,
            hidden_dim=config.get("hidden_dim", 64),
            output_dim=output_dim,
            num_layers=config.get("num_layers", 2),
            lr=config.get("lr", 0.01),
        )

    # Fallback to build()
    return model_cls.build(
        spec=spec,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=config.get("hidden_dim", 64),
        num_layers=config.get("num_layers", 2),
        device="cpu",
        task_type="vision",
        **{k: v for k, v in config.items() if k not in ["hidden_dim", "num_layers"]},
    )


def _train_model(model, x, y, epochs=3, batch_size=32):
    """Train a model using its preferred training method."""
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
    """Evaluate a single hyperparameter config."""
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
            "passed": diff <= 0.05,  # 5% target
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
    """Generate all combinations from param grid."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def run_sweep():
    """Run hyperparameter sweep for all models."""
    # Create synthetic task
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

    # Train backprop baseline
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

    results = {}
    best_configs = {}

    for sweep in SWEEPS:
        print(f"\n=== Sweeping {sweep.model_name} ===")
        grid = generate_grid(sweep.param_grid)
        print(f"Testing {len(grid)} configurations...")

        best_result = None
        best_diff = float("inf")
        found_passing = False

        for i, params in enumerate(grid):
            config = {**sweep.fixed_params, **params}
            result = evaluate_config(
                sweep.model_name, config, x, y, input_dim, n_classes, backprop_baseline
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
                found_passing = True
                break  # Early exit for this model

        if not found_passing:
            # Continue to check remaining configs for best diff
            for i, params in enumerate(grid):
                if (
                    i < len(grid)
                    and best_result
                    and best_result["config"] == {**sweep.fixed_params, **params}
                ):
                    continue  # Already evaluated
                config = {**sweep.fixed_params, **params}
                result = evaluate_config(
                    sweep.model_name,
                    config,
                    x,
                    y,
                    input_dim,
                    n_classes,
                    backprop_baseline,
                )
                if result["diff"] < best_diff:
                    best_diff = result["diff"]
                    best_result = result

        results[sweep.model_name] = {"grid": grid, "results": []}
        best_configs[sweep.model_name] = best_result

        print(
            f"\nBest for {sweep.model_name}: diff={best_diff:.3f}, config={best_result['config']}"
        )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for model_name, best in best_configs.items():
        status = "✓ CAN PASS" if best["passed"] else "✗ NEEDS WORK"
        print(f"  {model_name}: {status} (best diff={best['diff']:.3f})")
        print(f"    Best config: {best['config']}")

    # Save results
    output_path = Path(__file__).parent / "sweep_results.json"
    with Path(output_path).open("w") as f:
        json.dump(
            {
                "backprop_baseline": backprop_baseline,
                "best_configs": {
                    k: {"config": v["config"], "diff": v["diff"], "passed": v["passed"]}
                    for k, v in best_configs.items()
                },
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {output_path}")

    return best_configs


if __name__ == "__main__":
    run_sweep()
