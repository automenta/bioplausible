"""Z3 confirmatory run against the capability registrations (E-1).

Runs the promoted ``wu60_hot`` recipe (+ registered adaptation entropy
floor) over >= registration.min_seeds matched seeds, then evaluates the v4
order-robustness registration ``configs/preregistrations/
z3_capability_order_robust.json``: PRIMARY = demonstration gate — every seed
must reach >= 0.95 final hard-selection accuracy on ALL THREE tasks, hit the
registered 100-step-window criterion on all three, and hold exact Δθ under
randomized per-seed task order. The random-controller control runs the
identical protocol; its failure proportion is compared descriptively
(secondary exact one-sided Fisher), alongside paired-gap CI, order-broken
stats, and per-step gate histories for every arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import time
from pathlib import Path

import torch

from computronium.experiments.joint.z3_fixed_weights import (
    MetaRecipe,
    evaluate_z3,
)
from computronium.validation.preregistration import (
    ThresholdRegistration,
    paired_comparison,
)
from computronium.validation.statistics import fisher_exact_p_one_sided

logger = logging.getLogger(__name__)

COORDINATE = (
    "digital/recurrent/energy_minimization/rule_state/thermodynamic_contrast/euclidean"
)
TASKS = ("parity", "last_symbol", "threshold")
REGISTRATION = Path("configs/preregistrations/z3_capability_order_robust.json")
_ACCURACY_FLOOR = 0.95

PROMOTED_RECIPE = MetaRecipe(
    entropy_beta=0.2,
    episode_len=16,
    warmup_fraction=0.6,
    adapt_temp=2.0,
    adapt_temp_end=0.5,
    adapt_entropy_beta=0.1,
)


def _seed_task_order(seed: int) -> tuple[str, ...]:
    order = list(TASKS)
    random.Random(seed).shuffle(order)  # ruff: ignore[suspicious-non-cryptographic-random-usage]
    return tuple(order)


def _speed_report(row: dict) -> dict:
    """Descriptive z3-vs-finetune log step ratios at candidate windows."""
    from computronium.experiments.joint.z3_fixed_weights import (
        _windowed_criterion_step,
    )

    curves = {
        "z3": {t: row["tasks"][t]["accuracy_curve"] for t in TASKS},
        "finetune": {
            t: row["baselines"]["finetune_forgetting"]["accuracy_curves"][t]
            for t in TASKS
        },
    }
    budget = len(curves["z3"][TASKS[0]])
    report: dict[str, dict[str, float | None]] = {}
    for window in (20, 50, 100):
        per_task: dict[str, float | None] = {}
        for task in TASKS:
            steps = {}
            for arm in ("z3", "finetune"):
                raw = _windowed_criterion_step(curves[arm][task], window=window)
                steps[arm] = budget if raw is None else raw
            per_task[task] = math.log(steps["finetune"] / steps["z3"])
        report[f"w{window}"] = per_task
    return report


def _order_broken_stats(rows: list[dict]) -> dict:
    """Final accuracy per adaptation position (order sensitivity, both arms)."""
    positions: dict[int, dict[str, list[float]]] = {
        i: {"z3": [], "random": []} for i in range(len(TASKS))
    }
    for row in rows:
        for position, task in enumerate(row["task_order"]):
            positions[position]["z3"].append(row["accuracies"][task])
            positions[position]["random"].append(row["random_accuracies"][task])
    return {
        str(position): {
            arm: {
                "mean": sum(vals) / len(vals),
                "min": min(vals),
                "tasks": sorted({
                    task
                    for row in rows
                    for p, task in enumerate(row["task_order"])
                    if p == position
                }),
            }
            for arm, vals in arms.items()
        }
        for position, arms in positions.items()
    }


def _proportion_analysis(rows: list[dict], alpha: float) -> dict:
    failures_z3 = [r for r in rows if min(r["accuracies"].values()) < _ACCURACY_FLOOR]
    failures_random = [
        r for r in rows if min(r["random_accuracies"].values()) < _ACCURACY_FLOOR
    ]
    n = len(rows)
    f_z3, f_rand = len(failures_z3), len(failures_random)
    return {
        "arm_size": n,
        "failures_z3": f_z3,
        "failures_random": f_rand,
        "p_value": fisher_exact_p_one_sided(f_z3, f_rand, n),
        "alpha": alpha,
        "failing_seeds_z3": [r["seed"] for r in failures_z3],
        "failing_seeds_random": [r["seed"] for r in failures_random],
    }


def run(seeds: int, meta_epochs: int, eval_epochs: int, device: str) -> dict:
    registration = ThresholdRegistration.load(REGISTRATION)
    if seeds < registration.min_seeds:
        raise ValueError(
            f"seed budget {seeds} below registered floor {registration.min_seeds}"
        )

    rows: list[dict] = []
    for seed in range(seeds):
        order = _seed_task_order(seed)
        started = time.perf_counter()
        result = evaluate_z3(
            COORDINATE,
            meta_train_epochs=meta_epochs,
            eval_epochs_per_task=eval_epochs,
            device=device,
            seed=seed,
            with_baselines=True,
            recipe=PROMOTED_RECIPE,
            task_order=order,
        )
        elapsed = time.perf_counter() - started
        if not result["theta_invariant"]:
            raise RuntimeError(f"theta drift at seed {seed}")
        worst_z3 = min(result["tasks"][t]["accuracy"] for t in TASKS)
        random_tasks = result["baselines"]["random_psi"]["tasks"]
        worst_random = min(random_tasks[t]["accuracy"] for t in TASKS)
        criterion_ok = all(
            result["tasks"][t]["steps_to_criterion"] is not None for t in TASKS
        )
        rows.append({
            "seed": seed,
            "task_order": result["task_order"],
            "worst_task_accuracy_z3": worst_z3,
            "worst_task_accuracy_random": worst_random,
            "all_tasks_criterion": criterion_ok,
            "accuracies": {t: result["tasks"][t]["accuracy"] for t in TASKS},
            "random_accuracies": {t: random_tasks[t]["accuracy"] for t in TASKS},
            "operator_diversity": result["operator_diversity"],
            "pre_adapt_accuracy": {
                t: result["tasks"][t].get("pre_adapt_accuracy") for t in TASKS
            },
            "gate_histories": {t: result["tasks"][t]["gate_history"] for t in TASKS},
            "random_gate_histories": {
                t: random_tasks[t]["gate_history"] for t in TASKS
            },
            "speed_log_ratios": _speed_report(result),
            "wall_clock_s": elapsed,
        })
        logger.info(
            "seed %s: order=%s z3=%.4f random=%.4f criterion_all=%s (%.0fs)",
            seed,
            "+".join(t[:4] for t in order),
            worst_z3,
            worst_random,
            criterion_ok,
            elapsed,
        )

    comparison = paired_comparison(
        treatment=[r["worst_task_accuracy_z3"] for r in rows],
        control=[r["worst_task_accuracy_random"] for r in rows],
        seed=42,
    )
    proportion = _proportion_analysis(rows, alpha=registration.alpha)
    seed_gates = [
        {
            "seed": r["seed"],
            "accuracy_floor_all_tasks": r["worst_task_accuracy_z3"] >= _ACCURACY_FLOOR,
            "criterion_all_tasks": r["all_tasks_criterion"],
            # Δθ drift raises inside the loop, so reaching a row ⇒ gate held.
            "theta_invariant": True,
        }
        for r in rows
    ]
    gates = {
        "accuracy_floor_all_seeds": all(
            g["accuracy_floor_all_tasks"] for g in seed_gates
        ),
        "criterion_all_seeds": all(g["criterion_all_tasks"] for g in seed_gates),
        "theta_invariant_all_seeds": True,
    }
    confirmed = all(gates.values())
    return {
        "registration": registration.to_dict(),
        "recipe": {
            f: getattr(PROMOTED_RECIPE, f)
            for f in (
                "entropy_beta",
                "episode_len",
                "warmup_fraction",
                "adapt_temp",
                "adapt_entropy_beta",
            )
        }
        | {"meta_train_epochs": meta_epochs, "eval_epochs_per_task": eval_epochs},
        "seeds": rows,
        "seed_gates": seed_gates,
        "primary_coverage_gate": gates,
        "secondary_fisher_descriptive": proportion,
        "descriptive_paired_mean_gap": {
            "n": comparison.n,
            "mean_diff": comparison.mean_diff,
            "ci_lower": comparison.ci_lower,
            "ci_upper": comparison.ci_upper,
            "p_value": comparison.p_value,
            "cohens_dz": comparison.cohens_dz,
        },
        "order_broken_stats": _order_broken_stats(rows),
        "confirmed": confirmed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--meta-epochs", type=int, default=300)
    parser.add_argument("--eval-epochs", type=int, default=240)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmark_results/z3_order_robust")
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    payload = run(args.seeds, args.meta_epochs, args.eval_epochs, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "z3_order_robust_results.json"
    out.write_text(json.dumps(payload, indent=2))
    from computronium.experiments.joint.z3_fixed_weights import _git_commit

    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "registration_sha256": hashlib.sha256(
                    REGISTRATION.read_bytes()
                ).hexdigest(),
                "git_commit": _git_commit(),
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
        )
    )
    logger.info("confirmed=%s → %s", payload["confirmed"], out)


if __name__ == "__main__":
    main()
