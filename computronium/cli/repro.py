"""``comp repro`` — Sprint 1.4 deterministic-seeding gate.

Runs a one-epoch training pass for every registered model family twice under
an identical global seed and asserts the resulting parameter state dicts are
bitwise identical. Any model that is not bitwise reproducible fails the gate
with a non-zero exit code.

This is the CI-enforceable expression of the Sprint 1.4 validation: same seed
→ bitwise identical weights. It guards against RNG leaks (unseeded sources
creeping in), non-deterministic kernels, and any future PR that breaks
reproducibility without noticing.

Migrated to native compositions after legacy zoo removal.

Usage::

    uv run comp repro --seed 42 --device cpu
    uv run comp repro --models native_eqprop_mlp,native_fa_mlp --device cuda
"""

import argparse
import json
import logging

import torch

from computronium.core.logging import get_logger
from computronium.utils import capture_environment, deps_hash, seed_everything

logger = get_logger()

# Native model families exercised by the gate. Keep this aligned with the benchmark
# harness so one tiny synthetic task covers every learning rule.
# native_tile_ep is excluded (R11.1.3): EnergyMinimizationDynamics has no
# tile-mesh settle kernel — the coordinate raises before settling, so
# bitwise reproducibility is undefined for it. It is a permanent strict
# xfail in tests/property/test_native_smoke.py.
REPRO_MODELS = [
    "native_backprop_mlp",
    "native_eqprop_mlp",
    "native_fa_mlp",
    "native_pepita_mlp",
    "native_tile_fa",
    "native_tile_hebbian",
    "native_tile_tp",
]


def _instantiate(model_name: str, input_dim: int, output_dim: int, device: str):  # ruff: ignore[too-many-return-statements]
    """Instantiate a native model; mirrors the benchmark harness instantiation paths."""
    if model_name == "native_backprop_mlp":
        from computronium.models.native.backprop_native import (
            create_native_backprop_mlp,
        )

        return create_native_backprop_mlp(
            input_dim, 64, output_dim, num_layers=2, lr=1e-3
        )

    if model_name == "native_eqprop_mlp":
        from computronium.models.native.eqprop_native import create_native_eqprop_mlp

        return create_native_eqprop_mlp(
            input_dim=input_dim,
            hidden_dim=64,
            output_dim=output_dim,
            num_layers=2,
            beta=0.5,
            settle_steps=20,
            lr=1e-3,
        )

    if model_name == "native_fa_mlp":
        from computronium.models.native.fa_native import create_native_fa_mlp

        return create_native_fa_mlp(input_dim, 64, output_dim, num_layers=2, lr=1e-3)

    if model_name == "native_pepita_mlp":
        from computronium.models.native.pepita_native import create_native_pepita_mlp

        return create_native_pepita_mlp(
            input_dim, 64, output_dim, num_layers=2, lr=1e-3
        )

    if model_name == "native_tile_fa":
        from computronium.models.native.tile_native import create_native_tile_fa

        return create_native_tile_fa(
            input_dim=input_dim,
            hidden_dim=64,
            output_dim=output_dim,
            num_layers=2,
            neurons_per_tile=16,
            tiles_per_layer=2,
            lr=1e-3,
        )

    if model_name == "native_tile_hebbian":
        from computronium.models.native.tile_native import create_native_tile_hebbian

        return create_native_tile_hebbian(
            input_dim=input_dim,
            hidden_dim=64,
            output_dim=output_dim,
            num_layers=2,
            neurons_per_tile=16,
            tiles_per_layer=2,
            lr=1e-3,
        )

    if model_name == "native_tile_tp":
        from computronium.models.native.tile_native import create_native_tile_tp

        return create_native_tile_tp(
            input_dim=input_dim,
            hidden_dim=64,
            output_dim=output_dim,
            num_layers=2,
            neurons_per_tile=16,
            tiles_per_layer=2,
            lr=1e-3,
            beta=0.1,
        )

    raise ValueError(f"Unknown model: {model_name}")


def _train_one_epoch(model, x: torch.Tensor, y: torch.Tensor) -> None:
    """Train one epoch via the native System train_step."""
    model.train()  # type: ignore[attr-defined]
    n = len(x)
    batch_size = 64
    perm = torch.randperm(n)

    for i in range(0, n, batch_size):
        idx = perm[i : i + batch_size]
        model.train_step(x[idx], y[idx])


def _synthetic_data(seed: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic tiny classification task sized for all families.

    The same seed yields identical data regardless of device: generation is
    performed on CPU, then the tensors are moved to ``device`` iff requested.
    """
    n_samples = 256
    input_dim = 64
    n_classes = 10
    seed_everything(seed, device)
    x = torch.randn(n_samples, input_dim)
    y = torch.randint(0, n_classes, (n_samples,))
    if device != "cpu":
        x = x.to(device)
        y = y.to(device)
    return x, y


def _states_equal(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    if set(a) != set(b):
        return False
    return all(torch.equal(a[k].detach().cpu(), b[k].detach().cpu()) for k in a)


def run_one_model(
    model_name: str,
    seed: int,
    device: str,
) -> bool:
    """Train ``model_name`` twice under ``seed``; return bitwise identical."""
    x, y = _synthetic_data(seed, device)

    seed_everything(seed, device)
    m1 = _instantiate(model_name, x.shape[1], 10, device)
    _train_one_epoch(m1, x, y)
    # Get state from geometry params
    s1 = {k: v.detach().clone() for k, v in m1.geometry.params.items()}

    seed_everything(seed, device)
    m2 = _instantiate(model_name, x.shape[1], 10, device)
    _train_one_epoch(m2, x, y)

    s2 = {k: v.detach().clone() for k, v in m2.geometry.params.items()}

    return _states_equal(s1, s2)


def _gradient_gate() -> dict[str, bool]:
    """Run the gradient-equivalence gate.

    Returns family -> passed.
    """
    # Native models use different credit assignments - skip legacy gradient gate
    # This would need to be redesigned for native compositions
    logger.info("Gradient gate skipped for native models (legacy models deleted)")
    return {}


def _report_resume_noop(path: str) -> int:
    """Verify every probe in an experiment Report is a resume no-op.

    Reloads the Report (resume index) and checks that each recorded ok probe's
    key is present in the finished set — i.e. a re-run would skip it.
    """
    from computronium.experiment.report import Report, probe_index_key

    report = Report(path)
    finished = report.finished_keys()
    recorded = 0
    missing: list[str] = []
    for stage in sorted(set(_report_stages(path))):
        for result in report.stage_results(stage):
            if result.status == "error":
                continue
            recorded += 1
            if probe_index_key(stage, result) not in finished:
                missing.append(probe_index_key(stage, result))
    if missing:
        logger.error("resume no-op verification failed: %d keys absent", len(missing))
        return 1
    logger.info("resume no-op verified: %d finished probes", recorded)
    return 0 if recorded > 0 else 1


def _report_stages(path: str) -> list[str]:
    import json
    from pathlib import Path

    stages: list[str] = []
    p = Path(path)
    if not p.exists():
        return stages
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        stage = json.loads(line).get("stage", "")
        if stage and stage not in stages:
            stages.append(stage)
    return stages


def main(argv: list[str] | None = None) -> int:  # ruff: ignore[complex-structure]
    parser = argparse.ArgumentParser(
        prog="comp repro",
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
        help="Comma-separated model names (default: all native families)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON report instead of line output",
    )
    parser.add_argument(
        "--gradient",
        action="store_true",
        help="Run the gradient-equivalence gate on native models (currently skipped)",
    )
    parser.add_argument(
        "--resume-check",
        default=None,
        metavar="REPORT",
        help="Verify an experiment Report is a resume no-op",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
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
        except Exception:
            logger.exception("model %s failed during reproducibility pass", model)
            results[model] = False
            continue
        results[model] = ok
        status = "OK" if ok else "DIFF"
        logger.info("[%s]  %s", status, model)

    failed = [m for m, ok in results.items() if not ok]
    exit_code = 1 if failed else 0

    gradient_results: dict[str, bool] = {}
    if args.gradient:
        gradient_results = _gradient_gate()
        grad_failed = [m for m, ok in gradient_results.items() if not ok]
        for model, ok in gradient_results.items():
            logger.info("[%s]  gradient %s", "OK" if ok else "FAIL", model)
        if grad_failed:
            exit_code = 1

    if args.resume_check and _report_resume_noop(args.resume_check) != 0:
        exit_code = 1

    if args.json:
        report: dict[str, object] = {
            "seed": args.seed,
            "device": args.device,
            "environment": env,
            "deps_hash": deps_hash(env),
            "results": results,
            "exit_code": exit_code,
        }
        if gradient_results:
            report["gradient_gate"] = gradient_results
        if args.resume_check:
            report["resume_check"] = args.resume_check
        print(json.dumps(report, indent=2))
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
