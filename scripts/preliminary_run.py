"""Multi-family frontier run (plan §4D/§8/§9/§11) — guidance answer generator.

Executes the fair-comparison pipeline on MNIST across **several bio rule
families** against one shared ideal-backprop frontier, time-boxed for
~15 min - 1 hr:

1. Loads (or derives, if absent) the **ideal backprop** frontier over
   ``RULE_SPACES["backprop"]`` — cached, so it is derived once per task and
   reused across runs and across all bio rules.
2. For each requested bio rule (default ``eqprop,neural_cube,pepita,
   forward_forward,feedback_alignment``), searches over its *own* continuous
   space (incl. eq-specific params) via ``RuleFrontierFinder``.
3. Compares each bio frontier to the shared backprop frontier
   (``compare_frontiers``) → ``cost_of_plausibility``.
4. Fits the ``accuracy ~ log(FLOPs)`` scaling law for each rule and predicts
   the FLOPs to reach a target accuracy with a 95% CI (§8 resource-allocation).

Emits one JSON report with a per-rule table answering the guidance questions:
"which rules dominate under which resource constraints, and what is the cost of
bio-plausibility per family".

Usage::

    uv run python scripts/preliminary_run.py --device cuda --bio eqprop,pepita --bp-probes 10 --bio-probes 3
    uv run python scripts/preliminary_run.py --device cuda --bio neural_cube,feedback_alignment --target-acc 0.9
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import computronium.zoo  # ruff: ignore[unused-import]  (registration side effect)
from computronium.hyperopt.comparator import compare_frontiers
from computronium.hyperopt.ideal_backprop import IdealBackpropFinder
from computronium.hyperopt.rule_frontier import RuleFrontierFinder
from computronium.hyperopt.scaling_law import (
    fit_accuracy_scaling,
    predict_flops_for_accuracy,
)

logger = logging.getLogger("preliminary_run")

_DEFAULT_TASK = "mnist"
_DEFAULT_TARGET_ACC = 0.95
_DEFAULT_BIO = "eqprop,neural_cube,pepita,forward_forward,feedback_alignment"


def _build_driver(num_workers: int, target_hardware: str | None = None) -> object:
    """Construct the real CoreTrainerDriver with tracking enabled."""
    from computronium.experiment.probe import CoreTrainerDriver

    return CoreTrainerDriver(
        num_workers=num_workers,
        batch_size=128,
        track_energy=True,
        track_flops=True,
        track_memory=True,
        target_hardware=target_hardware,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", default=_DEFAULT_TASK)
    parser.add_argument(
        "--bio",
        default=_DEFAULT_BIO,
        help="Comma-separated RULE_SPACES keys of bio families to run",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--bp-probes", type=int, default=10)
    parser.add_argument("--bio-probes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-acc", type=float, default=_DEFAULT_TARGET_ACC)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-dir", default="logs")
    parser.add_argument(
        "--target-hardware",
        choices=["gpu", "fpga", "analog"],
        default=None,
        help="Substrate facade for BOTH backprop reference and bio rules (plan §17); "
        "is part of the frontier cache identity, so GPU vs FPGA frontiers never mix.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    bio_rules = [r.strip() for r in args.bio.split(",") if r.strip()]
    driver = _build_driver(args.num_workers, args.target_hardware)
    start = time.time()

    # 1. Ideal backprop frontier — reference, derived once then cached.
    backprop = IdealBackpropFinder(
        driver,
        task=args.task,
        backprop="backprop_mlp",
        budget_probes=args.bp_probes,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        cache_dir=args.cache_dir,
        target_hardware=args.target_hardware,
    ).find(force=False)  # reuse cached backprop frontier across runs/rules
    logger.info(
        "ideal backprop: %d points, %d frontier (cached)",
        len(backprop.points),
        len(backprop.frontier),
    )
    bp_law = fit_accuracy_scaling(
        list(backprop.points), rule="backprop", task=args.task
    )
    bp_pred = _flops_summary(bp_law, args.target_acc)
    if bp_pred is not None:
        logger.info("backprop FLOPs for %.3f acc: %s", args.target_acc, bp_pred)

    rules_out: dict[str, object] = {}
    for rule in bio_rules:
        logger.info("=== searching bio family '%s' ===", rule)
        bio = RuleFrontierFinder(
            driver,
            rule=rule,
            model=rule,  # RULE_SPACES keys match registered model names
            task=args.task,
            budget_probes=args.bio_probes,
            epochs=args.epochs,
            seed=args.seed,
            device=args.device,
            cache_dir=args.cache_dir,
            target_hardware=args.target_hardware,
        ).find(force=True)
        logger.info(
            "rule '%s': %d points, %d frontier",
            rule,
            len(bio.points),
            len(bio.frontier),
        )

        comparison = compare_frontiers(
            list(bio.frontier),
            list(backprop.frontier),
            rule=rule,
            backprop="backprop_mlp",
            task=args.task,
        )
        logger.info(
            "cost_of_plausibility(%s) = %.3f (n_dominating=%d)",
            rule,
            comparison.cost_of_plausibility,
            comparison.n_dominating_points,
        )

        bio_law = fit_accuracy_scaling(list(bio.points), rule=rule, task=args.task)
        bio_pred = _flops_summary(bio_law, args.target_acc)
        if bio_pred is not None:
            logger.info(
                "rule '%s' FLOPs for %.3f acc: %s", rule, args.target_acc, bio_pred
            )

        rules_out[rule] = {
            "n_points": len(bio.points),
            "frontier": [_p(p) for p in bio.frontier],
            "comparison": comparison.to_dict(),
            "scaling_law": _law_dict(bio_law, bio_pred, args.target_acc),
        }

    report = {
        "task": args.task,
        "device": args.device,
        "seed": args.seed,
        "elapsed_s": round(time.time() - start, 1),
        "bp_probes": args.bp_probes,
        "bio_probes": args.bio_probes,
        "ideal_backprop": {
            "n_points": len(backprop.points),
            "frontier": [_p(p) for p in backprop.frontier],
            "scaling_law": _law_dict(bp_law, bp_pred, args.target_acc),
        },
        "bio_rules": rules_out,
    }

    out_path = Path(args.cache_dir) / f"multi_family_{args.task}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        logger.info("report written to %s", out_path)
    return 0


def _p(p) -> dict[str, object]:
    return {
        "accuracy": p.accuracy,
        "total_flops": p.total_flops,
        "peak_memory_mb": p.peak_memory_mb,
        "wall_time_s": p.wall_time_s,
        "config": dict(p.config),
    }


def _law_dict(law, pred, target: float) -> dict[str, object]:
    if law is None:
        return {"n": 0}
    return {
        **law.to_dict(),
        "target_accuracy": target,
        "predicted_flops": pred or {},
    }


def _flops_summary(law, target: float) -> dict[str, object] | None:
    if law is None:
        return None
    mean, lo, hi = predict_flops_for_accuracy(law, target)
    return {"flops_mean": mean, "flops_ci_low": lo, "flops_ci_high": hi}


if __name__ == "__main__":
    raise SystemExit(main())
