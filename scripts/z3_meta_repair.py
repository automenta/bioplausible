"""E-2 repair rounds for Z3 meta-training (TODO4 queue item 1).

Runs the ranked attack configurations from the 2026-08-26 pilot autopsy —
(a) loss/gate feedback into ψ over per-task episodes, (b) temperature
annealing + gate-entropy bonus, (c) forced-operator θ warm-up — scoring each
by post-meta ψ-only adaptation accuracy under hard selection.

Gate: a configuration promotes to the pilot rerun only when ALL three tasks
sit materially above chance (seed-mean accuracy ≥ 0.7).
Writes ``benchmark_results/z3_meta_repair/round{N}.json``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from computronium.experiments.joint.z3_fixed_weights import (
    MetaRecipe,
    evaluate_z3,
)

logger = logging.getLogger(__name__)

COORDINATE = (
    "digital/recurrent/energy_minimization/rule_state/thermodynamic_contrast/euclidean"
)
GATE_ACCURACY = 0.7
TASKS = ("parity", "last_symbol", "threshold")
_PAIRED_ARMS_FROM_ROUND = 4

# Round 1: isolate each attack, then compose. ≤8 configs per E-2 round.
ROUND_1: dict[str, MetaRecipe] = {
    # control ≈ pre-repair recipe (stepwise tasks, constant T=1, dead ψ)
    "base": MetaRecipe(
        episode_len=1,
        feedback=False,
        entropy_beta=0.0,
        temp_start=1.0,
        temp_end=1.0,
        warmup_fraction=0.0,
    ),
    # episode structure alone (ψ stays dead)
    "episodes": MetaRecipe(
        feedback=False,
        entropy_beta=0.0,
        warmup_fraction=0.0,
    ),
    # attack (a): feedback ψ channel + episodes
    "fb": MetaRecipe(entropy_beta=0.0, warmup_fraction=0.0),
    # attacks (a)+(b): + temperature anneal & entropy bonus
    "fb_anneal": MetaRecipe(warmup_fraction=0.0),
    # attack (c) alone: forced-operator θ warm-up, no ψ feedback
    "warmup": MetaRecipe(feedback=False, entropy_beta=0.0),
    # attacks (a)+(b)+(c) composed
    "full": MetaRecipe(),
}

# Round 2: corrected solver map (threshold→Identity) + live ψ dynamics
# (decay/scale). Isolates the map-fix contribution and tunes around it.
ROUND_2: dict[str, MetaRecipe] = {
    "full": MetaRecipe(),  # attacks (a)+(b)+(c), validated defaults
    "fb_only": MetaRecipe(warmup_fraction=0.0),  # no forced warm-up phase
    "full_wu60": MetaRecipe(warmup_fraction=0.6),
    "full_b02": MetaRecipe(entropy_beta=0.2),  # stronger anti-collapse bonus
    "full_longep": MetaRecipe(episode_len=16),
    "full_slow_anneal": MetaRecipe(temp_end=1.0),
}

# Round 3: compose round-2 winners (entropy bonus × episode length) at a
# doubled meta budget; threshold needs the most adaptation headroom.
ROUND_3: dict[str, MetaRecipe] = {
    "full": MetaRecipe(),  # round-2 baseline at m100
    "b02": MetaRecipe(entropy_beta=0.2),
    "b02_longep": MetaRecipe(entropy_beta=0.2, episode_len=16),
    "b02_longep_wu60": MetaRecipe(
        entropy_beta=0.2, episode_len=16, warmup_fraction=0.6
    ),
    "b03_longep": MetaRecipe(entropy_beta=0.3, episode_len=16),
    "b02_longep_slow": MetaRecipe(entropy_beta=0.2, episode_len=16, temp_end=1.0),
}

# Round 4 (differential): close the meta-vs-random-ψ gap. Base = promoted
# b02_longep_wu60. Attacks per TODO4 queue item 1:
#   (a) much longer controller phase (meta 300, warmup share down to 0.4);
#   (b) entropy curriculum high→low so routing locks instead of exploring;
#   (c) supervised replay distillation of episode-best operators.
# Every config runs WITH baselines so the random-ψ control is paired on the
# same meta-trained trunk.
ROUND_4: dict[str, MetaRecipe] = {
    "m300_wu40": MetaRecipe(entropy_beta=0.2, episode_len=16, warmup_fraction=0.4),
    "m300_curric": MetaRecipe(
        entropy_beta=0.5,
        entropy_end=0.0,
        episode_len=16,
        warmup_fraction=0.4,
    ),
    "curric_wu60": MetaRecipe(
        entropy_beta=0.5,
        entropy_end=0.0,
        episode_len=16,
        warmup_fraction=0.6,
    ),
    "replay": MetaRecipe(
        entropy_beta=0.2,
        episode_len=16,
        warmup_fraction=0.6,
        replay_steps=4,
    ),
}

ROUNDS: dict[int, dict[str, MetaRecipe]] = {
    1: ROUND_1,
    2: ROUND_2,
    3: ROUND_3,
    4: ROUND_4,
}

# Round 5 (differential, informed by the R4 probe autopsy):
#   - Pre-adaptation routing cannot converge for ALL tasks at ψ=0 (identical
#     input distributions ⇒ one shared default routing), so the differential
#     must come from faster WITHIN-episode re-routing once ψ carries history.
#   - Parity is self-revealing (solver op emits the label as a feature): a
#     fresh random controller solves it instantly via broad sampling while a
#     meta-trained controller at cold gating T locks wrong routing forever.
#   - Replay distillation (attack c) directly supervises (ψ, x) → best-op,
#     which should install both parity routing and fast re-routing.
# Axis under test: replay × adaptation temperature (hot explores, cold
# preserves priors); plus curriculum-only and promoted-recipe references.
ROUND_5: dict[str, MetaRecipe] = {
    "replay_hot": MetaRecipe(
        entropy_beta=0.5,
        entropy_end=0.0,
        episode_len=16,
        warmup_fraction=0.4,
        replay_steps=4,
        adapt_temp=2.0,
    ),
    "replay_mid": MetaRecipe(
        entropy_beta=0.5,
        entropy_end=0.0,
        episode_len=16,
        warmup_fraction=0.4,
        replay_steps=4,
        adapt_temp=1.25,
    ),
    "replay_cold": MetaRecipe(
        entropy_beta=0.5,
        entropy_end=0.0,
        episode_len=16,
        warmup_fraction=0.4,
        replay_steps=4,
        adapt_temp=0.75,
    ),
    "noreplay_mid": MetaRecipe(
        entropy_beta=0.5,
        entropy_end=0.0,
        episode_len=16,
        warmup_fraction=0.4,
        adapt_temp=1.25,
    ),
    "wu60_hot": MetaRecipe(
        entropy_beta=0.2,
        episode_len=16,
        warmup_fraction=0.6,
        adapt_temp=2.0,
    ),
    "lock_mid": MetaRecipe(
        entropy_beta=0.5,
        entropy_end=0.0,
        episode_len=16,
        warmup_fraction=0.4,
        temp_end=0.25,
        adapt_temp=1.25,
    ),
}
ROUNDS[5] = ROUND_5


@dataclass(frozen=True, slots=True)
class RoundBudget:
    """Compute budget and output location for one repair round."""

    seeds: int = 2
    meta_epochs: int = 50
    eval_epochs: int = 100
    device: str = "cpu"
    output_dir: Path = Path("benchmark_results/z3_meta_repair")


def _differential_verdict(
    seed_rows: list[dict], budget_steps: int
) -> dict[str, object]:
    """Meta-vs-random-ψ paired verdict over censored steps-to-criterion.

    Censored arms score at the budget. The differential closes only when the
    meta arm is strictly faster than the random-ψ control on the WORST task
    AND reaches the registered criterion on threshold inside the budget.
    """
    per_task: dict[str, dict[str, float]] = {}
    for task in TASKS:
        arms = {
            "meta": [r["tasks"][task]["steps_to_criterion"] for r in seed_rows],
            "random": [
                r["baselines"]["random_psi"]["tasks"][task]["steps_to_criterion"]
                for r in seed_rows
            ],
            "finetune": [
                r["baselines"]["finetune_forgetting"]["steps_to_criterion"][task]
                for r in seed_rows
            ],
        }
        per_task[task] = {
            name: sum(budget_steps if s is None else s for s in steps) / len(steps)
            for name, steps in arms.items()
        }
    margin = min(v["random"] - v["meta"] for v in per_task.values())
    threshold_solved = all(
        r["tasks"]["threshold"]["steps_to_criterion"] is not None for r in seed_rows
    )
    return {
        "arms_mean_steps_to_criterion": per_task,
        "worst_margin_random_minus_meta": margin,
        "threshold_criterion_reached": threshold_solved,
        "pass": bool(margin > 0 and threshold_solved),
    }


def _seed_curves(row: dict, tasks: tuple[str, ...]) -> dict:
    """Per-arm accuracy curves for one seed (absent arms map to None)."""
    curves: dict[str, dict[str, list[float] | None]] = {
        "meta": {t: row["tasks"][t]["accuracy_curve"] for t in tasks}
    }
    baselines = row.get("baselines")
    if baselines:
        curves["random"] = {
            t: baselines["random_psi"]["tasks"][t].get("accuracy_curve") for t in tasks
        }
        ft_curves = baselines["finetune_forgetting"].get("accuracy_curves", {})
        curves["finetune"] = {t: ft_curves.get(t) for t in tasks}
    return curves


def _config_row(
    name: str,
    recipe: MetaRecipe,
    seed_rows: list[dict],
    elapsed: float,
    *,
    paired_arms: bool,
    eval_epochs: int,
) -> tuple[dict, bool]:
    """Summarize one config's seed rows; returns (row, passed_all_gates)."""
    means = {
        task: sum(row["tasks"][task]["accuracy"] for row in seed_rows) / len(seed_rows)
        for task in TASKS
    }
    passed = min(means.values()) >= GATE_ACCURACY
    row: dict = {
        "config": name,
        "recipe": dataclasses.asdict(recipe),
        "task_mean_accuracy": means,
        "min_task_mean_accuracy": min(means.values()),
        "gate_passed": passed,
        "diversity": [r["operator_diversity"] for r in seed_rows],
        "steps_to_criterion": [
            {t: r["tasks"][t]["steps_to_criterion"] for t in TASKS} for r in seed_rows
        ],
        "pre_adapt_accuracy": [
            {t: r["tasks"][t]["pre_adapt_accuracy"] for t in TASKS} for r in seed_rows
        ],
        "seeds": [
            {
                "accuracy": {t: r["tasks"][t]["accuracy"] for t in TASKS},
                "theta_change": r["theta_change"],
                "wall_clock_s": r["wall_clock_s"],
                "curves": _seed_curves(r, TASKS),
            }
            for r in seed_rows
        ],
        "wall_clock_s": elapsed,
    }
    differential = (
        _differential_verdict(seed_rows, eval_epochs) if paired_arms else None
    )
    if differential is not None:
        row["differential"] = differential
        passed = passed and bool(differential["pass"])
    return row, passed


def run_round(
    round_number: int,
    configs: list[str] | None,
    budget: RoundBudget,
) -> Path | None:
    """Run one E-2 repair round; returns the artifact path when written."""
    catalog = ROUNDS.get(round_number)
    if catalog is None:
        logger.error("Unknown round %s (defined: %s)", round_number, sorted(ROUNDS))
        return None
    selected = configs or list(catalog)
    unknown = set(selected) - set(catalog)
    if unknown:
        logger.error(
            "Unknown configs %s (round has %s)", sorted(unknown), sorted(catalog)
        )
        return None

    rows: list[dict] = []
    paired_arms = round_number >= _PAIRED_ARMS_FROM_ROUND
    for name in selected:
        recipe = catalog[name]
        seed_rows = []
        started = time.perf_counter()
        for seed in range(budget.seeds):
            result = evaluate_z3(
                COORDINATE,
                meta_train_epochs=budget.meta_epochs,
                eval_epochs_per_task=budget.eval_epochs,
                device=budget.device,
                seed=seed,
                with_baselines=paired_arms,
                recipe=recipe,
            )
            if not result["theta_invariant"]:
                raise RuntimeError(f"θ drifted under config {name}, seed {seed}")
            seed_rows.append(result)
        elapsed = time.perf_counter() - started
        row, passed = _config_row(
            name,
            recipe,
            seed_rows,
            elapsed,
            paired_arms=paired_arms,
            eval_epochs=budget.eval_epochs,
        )
        rows.append(row)
        logger.info(
            "%s [%s] parity=%.3f last_symbol=%.3f threshold=%.3f (%.1fs)",
            "PASS" if passed else "fail",
            name,
            row["task_mean_accuracy"]["parity"],
            row["task_mean_accuracy"]["last_symbol"],
            row["task_mean_accuracy"]["threshold"],
            elapsed,
        )
        differential = row.get("differential")
        if not isinstance(differential, dict):
            continue
        logger.info(
            "    differential: margin(random-meta)=%.1f steps, threshold@%s",
            differential["worst_margin_random_minus_meta"],
            "criterion" if differential["threshold_criterion_reached"] else "censored",
        )

    budget.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = budget.output_dir / f"round{round_number}.json"
    payload = {
        "round": round_number,
        "gate_accuracy": GATE_ACCURACY,
        **{
            k: str(v) if isinstance(v, Path) else v
            for k, v in dataclasses.asdict(budget).items()
        },
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": rows,
    }
    artifact.write_text(json.dumps(payload, indent=2))
    logger.info("Round %s artifacts → %s", round_number, artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--configs", nargs="*", default=None)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--meta-epochs", type=int, default=50)
    parser.add_argument("--eval-epochs", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmark_results/z3_meta_repair")
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    run_round(
        args.round,
        args.configs,
        RoundBudget(
            seeds=args.seeds,
            meta_epochs=args.meta_epochs,
            eval_epochs=args.eval_epochs,
            device=device,
            output_dir=args.output_dir,
        ),
    )


if __name__ == "__main__":
    main()
