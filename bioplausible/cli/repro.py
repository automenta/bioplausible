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

from bioplausible.core.logging import get_logger
from bioplausible.utils import capture_environment, deps_hash, seed_everything

logger = get_logger()

# Model families exercised by the gate. Keep this aligned with the benchmark
# harness (Sprint 1.3) so one tiny synthetic task covers every learning rule.
REPRO_MODELS = [
    "eqprop_mlp",
    "fa",
    "mep",
    "tile_pc",
    "forward_forward",
    "pepita",
    "spiking",
]


def _instantiate(  # ruff: ignore[too-many-return-statements]  (one return per registered family is the readable form)
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
        from bioplausible.config.unified import ModelConfig
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

    # Every remaining registered family (tile substrate like ``tile_pc`` and the
    # deployment ``BioModel`` family like ``conv_equitile``) constructs through
    # its canonical ``build`` classmethod, routed via the single construction
    # funnel so registry validation + device placement stay uniform.
    from bioplausible.core.construction import construct_model

    model = construct_model(
        Registry.get(ComponentCategory.MODEL, model_name),
        {
            "hidden_dim": 64,
            "num_layers": 2,
            "device": device,
            "task_type": "vision",
        },
        input_dim=input_dim,
        output_dim=output_dim,
        model_name=model_name,
    )
    return model  # type: ignore[return-value]  # construct_model returns object; every branch yields an nn.Module


def _train_one_epoch(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> None:
    """Train one epoch via the shared train-step dispatcher (Pillar A).

    Routes each batch through the canonical ``dispatch_train_step`` seam, so
    models that own a ``train_step`` use it and the rest fall back to the Adam
    BPTT path — no hand-rolled copy of the training loop.
    """
    from torch import optim

    from bioplausible.core.trainer import dispatch_train_step

    model.train()
    n = len(x)
    batch_size = 64
    perm = torch.randperm(n)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for i in range(0, n, batch_size):
        idx = perm[i : i + batch_size]
        dispatch_train_step(
            model=model,
            x=x[idx],
            y=y[idx],
            adapt_input=lambda t: t,  # repro data is already flat 2D
            optimizer=optimizer,
        )


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
    s1 = {k: v.detach().clone() for k, v in m1.state_dict().items()}

    seed_everything(seed, device)
    m2 = _instantiate(model_name, x.shape[1], 10, device)
    _train_one_epoch(m2, x, y)

    return _states_equal(s1, m2.state_dict())


def _gradient_gate() -> dict[str, bool]:
    """Run the gradient-equivalence gate (architecture §7#2) per model family.

    Only gradient-aligned propagators have a defined update direction vs. the
    task loss; forward-only (FF/PEPITA) and spiking families are excluded by
    design. Returns family -> passed.
    """
    from bioplausible.validation.gradient_check import (
        check_gradient_equivalence,
        loss_ce,
        loss_mse,
    )
    from bioplausible.zoo.mep.presets import smep as _smep
    from bioplausible.core.local_learning.rules.backprop import Backprop as _Backprop
    from bioplausible.core.local_learning.rules.eqprop import EqProp as _EqProp
    from bioplausible.core.local_learning.rules.fa import (
        DirectFA as _DirectFA,
    )
    from bioplausible.core.local_learning.rules.fa import (
        FeedbackAlignment as _FeedbackAlignment,
    )
    from bioplausible.core.local_learning.rules.fa import (
        StochasticFA as _StochasticFA,
    )
    from bioplausible.core.local_learning.rules.hebbian import ContrastiveHebbianLearning

    def _lro_driver(opt, model, x, y) -> None:  # ruff: ignore[unused-function-argument]  (driver protocol fixes the signature)
        opt.step(x=x, target=y)

    def _bptt_driver(opt, model, x, y) -> None:
        model.zero_grad()
        torch.nn.functional.cross_entropy(model(x), y).backward()
        opt.step()

    families: list[tuple[str, object, object, object, float]] = [
        ("backprop", _Backprop, _lro_driver, loss_ce, 0.9),
        ("feedback_alignment", _FeedbackAlignment, _lro_driver, loss_ce, 0.9),
        ("direct_fa", _DirectFA, _lro_driver, loss_ce, 0.9),
        ("stochastic_fa", _StochasticFA, _lro_driver, loss_ce, 0.9),
        (
            "smep_backprop",
            lambda p, m: _smep(p, m, mode="backprop", ns_steps=0),
            _bptt_driver,
            loss_ce,
            0.9,
        ),
        (
            "eq_prop",
            lambda p, m: _EqProp(p, m, beta=0.5, settle_steps=30, settle_lr=0.15),
            _lro_driver,
            loss_mse,
            0.6,
        ),
        (
            "smep_ep",
            lambda p, m: _smep(
                p, m, mode="ep", settle_steps=30, ns_steps=0, settle_lr=0.15
            ),
            _lro_driver,
            loss_mse,
            0.6,
        ),
        (
            "contrastive_hebbian_learning",
            ContrastiveHebbianLearning,
            _lro_driver,
            loss_mse,
            0.6,
        ),
    ]

    results: dict[str, bool] = {}
    for name, build, driver, loss, threshold in families:
        try:
            check_gradient_equivalence(name, build, driver, loss, threshold)
            results[name] = True
        except Exception:  # broad: a failing family must not kill the gate
            logger.exception("family %s failed the gradient gate", name)
            results[name] = False
    return results


def _report_resume_noop(path: str) -> int:
    """Verify every probe in an experiment Report is a resume no-op.

    Reloads the Report (resume index) and checks that each recorded ok probe's
    key is present in the finished set — i.e. a re-run would skip it.
    """
    from bioplausible.experiment.report import Report, probe_index_key

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


def main(argv: list[str] | None = None) -> int:  # ruff: ignore[complex-structure]  (CLI orchestrates repro + gradient + resume gates)
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
    parser.add_argument(
        "--gradient",
        action="store_true",
        help="Run the gradient-equivalence gate (architecture §7#2) on "
        "gradient-aligned families",
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
