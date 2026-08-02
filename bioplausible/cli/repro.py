"""``biopl-repro-check`` — Sprint 1.4 deterministic-seeding gate.

Runs a one-epoch training pass for every registered model family twice under
an identical global seed and asserts the resulting parameter state dicts are
bitwise identical. Any model that is not bitwise reproducible fails the gate
with a non-zero exit code.

This is the CI-enforceable expression of the Sprint 1.4 validation: same seed
→ bitwise identical weights. It guards against RNG leaks (unseeded sources
creeping in), non-deterministic kernels, and any future PR that breaks
reproducibility without noticing.

Usage::

    uv run biopl-repro-check --seed 42 --device cpu
    uv run biopl-repro-check --models eqprop_mlp,equitile --device cuda
"""

import argparse
import json
import logging

import torch

from bioplausible.utils import capture_environment, deps_hash, set_global_seed

logger = logging.getLogger(__name__)

# Model families exercised by the gate. Keep this aligned with the benchmark
# harness (Sprint 1.3) so one tiny synthetic task covers every learning rule.
REPRO_MODELS = [
    "eqprop_mlp",
    "fa",
    "mep",
    "equitile",
    "forward_forward",
    "pepita",
    "spiking",
]


def _instantiate(
    model_name: str, input_dim: int, output_dim: int, device: str
) -> torch.nn.Module:
    """Instantiate a model; mirrors the benchmark harness instantiation paths."""
    from bioplausible.core.registry import ComponentCategory, Registry

    if model_name == "eqprop_mlp":
        from bioplausible.zoo.models.eqprop.looped_mlp import LoopedMLP

        model = LoopedMLP(
            input_dim=input_dim,
            hidden_dim=64,
            output_dim=output_dim,
            use_spectral_norm=True,
            max_steps=20,
            gradient_method="contrastive",
            backend="pytorch",
        )
        model.hebbian_lr = 0.008
        model.beta = 0.03
        return model.to(device)

    if model_name == "fa":
        from bioplausible.core.config import ModelConfig
        from bioplausible.zoo.models.fa import StandardFA

        cfg = ModelConfig(
            name="standard_fa",
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=[64],
        )
        return StandardFA(config=cfg).to(device)

    if model_name == "mep":
        from bioplausible.zoo.models.eqprop.memory_efficient import (
            MemoryEfficientLoopedMLP,
        )

        return MemoryEfficientLoopedMLP(
            input_dim=input_dim,
            hidden_dim=64,
            output_dim=output_dim,
            max_steps=20,
            gradient_method="contrastive",
            use_gpu_if_available=False,
        ).to(device)

    if model_name == "forward_forward":
        from bioplausible.zoo.models.forward_only import ForwardForwardNet

        return ForwardForwardNet(
            input_dim=input_dim,
            hidden_dim=64,
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
            hidden_dim=64,
            output_dim=output_dim,
            num_layers=2,
            lr=0.3,
        ).to(device)

    if model_name == "spiking":
        from bioplausible.zoo.models.spiking import SpikingSTDP

        return SpikingSTDP(
            input_dim=input_dim,
            hidden_dim=64,
            output_dim=output_dim,
            num_steps=10,
        ).to(device)

    from bioplausible.zoo import get_model_spec

    spec = get_model_spec(model_name)
    model_cls = Registry.get(ComponentCategory.MODEL, model_name)

    model = model_cls.build(
        spec=spec,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=64,
        num_layers=2,
        device=device,
        task_type="vision",
    )
    return model


def _train_one_epoch(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> None:
    """Train one epoch via the model's preferred interface (fallback Adam)."""
    from torch import nn, optim

    model.train()
    n = len(x)
    batch_size = 64
    perm = torch.randperm(n)

    if hasattr(model, "train_step"):
        result = model.train_step(x[:batch_size], y[:batch_size])
        if result is not None:
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                model.train_step(x[idx], y[idx])
            return

    opt = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for i in range(0, n, batch_size):
        idx = perm[i : i + batch_size]
        xb, yb = x[idx], y[idx]
        opt.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        opt.step()


def _synthetic_data(seed: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic tiny classification task sized for all families.

    The same seed yields identical data regardless of device: generation is
    performed on CPU, then the tensors are moved to ``device`` iff requested.
    """
    n_samples = 256
    input_dim = 64
    n_classes = 10
    torch.manual_seed(seed)
    x = torch.randn(n_samples, input_dim)
    y = torch.randint(0, n_classes, (n_samples,))
    if device != "cpu":
        x = x.to(device)
        y = y.to(device)
    return x, y


def _states_equal(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    if set(a) != set(b):
        return False
    return all(
        torch.equal(a[k].detach().cpu(), b[k].detach().cpu())
        for k in a
    )


def run_one_model(
    model_name: str,
    seed: int,
    device: str,
) -> bool:
    """Train ``model_name`` twice under ``seed``; return bitwise identical."""
    x, y = _synthetic_data(seed, device)

    set_global_seed(seed, device)
    m1 = _instantiate(model_name, x.shape[1], 10, device)
    _train_one_epoch(m1, x, y)
    s1 = {k: v.detach().clone() for k, v in m1.state_dict().items()}

    set_global_seed(seed, device)
    m2 = _instantiate(model_name, x.shape[1], 10, device)
    _train_one_epoch(m2, x, y)

    return _states_equal(s1, m2.state_dict())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="biopl-repro-check",
        description="Verify bitwise reproducibility of every model family "
        "under a fixed global seed (Sprint 1.4).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Master seed")
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu (default) or cuda/gpu (requires CUDA availability)",
    )
    parser.add_argument(
        "--models",
        default=",".join(REPRO_MODELS),
        help="Comma-separated model names (default: all families)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON report instead of line output",
    )
    args = parser.parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        logger.error("No models requested")
        return 2

    env = capture_environment()
    logger.info("environment: %s (hash %s)", env, deps_hash(env))

    results: dict[str, bool] = {}
    for model in models:
        try:
            ok = run_one_model(model, args.seed, args.device)
        except Exception:  # noqa: BLE001 - report any family failure
            logger.exception("model %s failed during reproducibility pass", model)
            results[model] = False
            continue
        results[model] = ok
        status = "OK" if ok else "DIFF"
        logger.info("[%s]  %s", status, model)

    failed = [m for m, ok in results.items() if not ok]
    exit_code = 1 if failed else 0

    if args.json:
        print(
            json.dumps(
                {
                    "seed": args.seed,
                    "device": args.device,
                    "environment": env,
                    "deps_hash": deps_hash(env),
                    "results": results,
                    "exit_code": exit_code,
                },
                indent=2,
            )
        )
    else:
        logger.info(
            "repro check: %d/%d reproducible%s",
            len(results) - len(failed),
            len(results),
            f" — FAILED: {', '.join(failed)}" if failed else "",
        )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
