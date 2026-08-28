"""Phase 3.6.7 — Z3 Re-verification Audit.

Four checks after fixing RuleStatePlasticity truncation and in-place op bugs:

1. RuleStatePlasticity in Z3: ψ evolves during adaptation (norm > 0).
2. Z3 v2 canonical-order: 5-seed confirmatory, all tasks ≥ 0.95, Δθ = 0.
3. Z3 v4 order-robust: per-seed permuted order, document sensitivity.
4. Gate-history instrumentation: complete per-step gate records.

Artifacts: audit_results/z3_reverification.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from computronium.core.plasticity.theta_audit import ThetaInvarianceAudit
from computronium.experiments.joint.z3_fixed_weights import (
    MetaRecipe,
    TaskShape,
    Z3Model,
    _adapt_all_tasks,
    _fixed_probe,
    _is_theta_param,
    _meta_train_phases,
    _run_adaptation,
    create_last_symbol_task,
    create_parity_task,
    create_threshold_task,
)

TASKS = [
    ("parity", create_parity_task),
    ("last_symbol", create_last_symbol_task),
    ("threshold", create_threshold_task),
]
TASK_NAMES = ["parity", "last_symbol", "threshold"]

RECIPE = MetaRecipe(
    entropy_beta=0.2,
    episode_len=16,
    warmup_fraction=0.6,
    adapt_temp=2.0,
    adapt_temp_end=0.5,
    adapt_entropy_beta=0.1,
)

_ACCURACY_FLOOR = 0.95
_SEEDS = 5
_META_EPOCHS = 300
_EVAL_EPOCHS = 240
_BATCH_SIZE = 64
_SEQ_LEN = 10
_INPUT_DIM = 32
_PROBE_BATCHES = 16


def _build_model(seed: int, device: torch.device) -> Z3Model:
    torch.manual_seed(seed)
    return Z3Model(
        num_operators=8,
        operator_dim=_INPUT_DIM,
        controller_hidden=128,
        temperature=RECIPE.temp_start,
    ).to(device)


def _meta_train_model(
    model: Z3Model,
    tasks: list,
    criterion: torch.nn.Module,
    shape: TaskShape,
) -> dict[str, torch.Tensor]:
    _meta_train_phases(
        model,
        tasks,
        criterion,
        shape,
        recipe=RECIPE,
        meta_train_epochs=_META_EPOCHS,
    )
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _check_psi_evolution(
    model: Z3Model,
    tasks: list,
    criterion: torch.nn.Module,
    shape: TaskShape,
) -> dict:
    """Check 1: ψ evolves during adaptation (norm > 0 after steps)."""
    model.eval()
    model.freeze_theta()
    psi_before = model.psi_state.norm().item()

    probe = _fixed_probe(shape, tasks[0][1], batches=_PROBE_BATCHES)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=0.001
    )
    _run_adaptation(
        model,
        optimizer,
        criterion,
        shape,
        tasks[0][1],
        epochs=30,
        probe=probe,
        feedback=True,
        adapt_entropy_beta=0.1,
        adapt_temp_end=0.5,
    )

    psi_after = model.psi_state.norm().item()
    psi_evolved = psi_after > psi_before
    psi_nonzero = psi_after > 1e-6

    return {
        "psi_norm_before": psi_before,
        "psi_norm_after": psi_after,
        "psi_evolved": psi_evolved,
        "psi_nonzero": psi_nonzero,
        "pass": psi_evolved and psi_nonzero,
    }


def _check_canonical_order(
    seed: int,
    model: Z3Model,
    meta_state: dict[str, torch.Tensor],
    tasks: list,
    criterion: torch.nn.Module,
    shape: TaskShape,
) -> dict:
    """Check 2: canonical-order confirmatory at one seed."""
    model.load_state_dict(meta_state)
    model.freeze_theta()
    with ThetaInvarianceAudit(model, selector=_is_theta_param) as audit:
        rows, wall = _adapt_all_tasks(
            model,
            tasks,
            criterion,
            shape,
            epochs=_EVAL_EPOCHS,
            probe_batches=_PROBE_BATCHES,
            feedback=True,
            adapt_entropy_beta=0.1,
            adapt_temp_end=0.5,
        )
    report = audit.report
    assert report is not None
    task_accs = {name: rows[name]["accuracy"] for name in TASK_NAMES}
    all_pass = all(acc >= _ACCURACY_FLOOR for acc in task_accs.values())
    criterion_ok = all(
        rows[name]["steps_to_criterion"] is not None for name in TASK_NAMES
    )

    return {
        "seed": seed,
        "task_accuracies": task_accs,
        "all_above_floor": all_pass,
        "criterion_reached_all": criterion_ok,
        "theta_change": report.max_abs_change,
        "theta_invariant": report.is_within(1e-6),
        "wall_clock_s": wall,
        "pass": all_pass and report.is_within(1e-6),
    }


def _check_order_robustness(
    seed: int,
    model: Z3Model,
    meta_state: dict[str, torch.Tensor],
    tasks: list,
    criterion: torch.nn.Module,
    shape: TaskShape,
    task_order: list[str],
) -> dict:
    """Check 3: order-robustness at one seed with permuted task order."""
    ordered = sorted(tasks, key=lambda t: task_order.index(t[0]))
    model.load_state_dict(meta_state)
    model.freeze_theta()
    with ThetaInvarianceAudit(model, selector=_is_theta_param) as audit:
        rows, wall = _adapt_all_tasks(
            model,
            ordered,
            criterion,
            shape,
            epochs=_EVAL_EPOCHS,
            probe_batches=_PROBE_BATCHES,
            feedback=True,
            adapt_entropy_beta=0.1,
            adapt_temp_end=0.5,
        )
    report = audit.report
    assert report is not None
    task_accs = {name: rows[name]["accuracy"] for name in TASK_NAMES}
    all_pass = all(acc >= _ACCURACY_FLOOR for acc in task_accs.values())
    return {
        "seed": seed,
        "task_order": task_order,
        "task_accuracies": task_accs,
        "all_above_floor": all_pass,
        "theta_change": report.max_abs_change,
        "wall_clock_s": wall,
        "pass": all_pass and report.is_within(1e-6),
    }


def _check_gate_history(rows: dict) -> dict:
    """Check 4: gate history completeness for all adaptation steps."""
    completeness: dict[str, dict] = {}
    for name in TASK_NAMES:
        gh = rows[name]["gate_history"]
        n_steps = len(gh["entropy"])
        all_keys = all(
            len(gh[k]) == n_steps for k in ("mean_gates", "hard_op_fraction", "entropy")
        )
        completeness[name] = {
            "n_steps": n_steps,
            "mean_gates_len": len(gh["mean_gates"]),
            "hard_op_fraction_len": len(gh["hard_op_fraction"]),
            "entropy_len": len(gh["entropy"]),
            "complete": all_keys and n_steps > 0,
        }
    all_complete = all(c["complete"] for c in completeness.values())
    return {
        "per_task": completeness,
        "all_complete": all_complete,
        "pass": all_complete,
    }


def run_audit(device: str = "cpu") -> dict:
    dev = torch.device(device)
    tasks = list(TASKS)
    shape = TaskShape(
        batch_size=_BATCH_SIZE, seq_len=_SEQ_LEN, input_dim=_INPUT_DIM, device=dev
    )
    criterion = torch.nn.CrossEntropyLoss()

    results: dict = {
        "checks": {},
        "meta": {
            "device": str(dev),
            "seeds": _SEEDS,
            "meta_epochs": _META_EPOCHS,
            "eval_epochs": _EVAL_EPOCHS,
        },
    }
    all_pass = True

    # --- Check 1: ψ evolution ---
    print("Check 1: RuleStatePlasticity ψ evolution...")
    model = _build_model(seed=0, device=dev)
    meta_state = _meta_train_model(model, tasks, criterion, shape)
    psi_check = _check_psi_evolution(model, tasks, criterion, shape)
    results["checks"]["psi_evolution"] = psi_check
    all_pass &= psi_check["pass"]
    print(
        f"  ψ norm: {psi_check['psi_norm_before']:.6f} → {psi_check['psi_norm_after']:.6f} | pass={psi_check['pass']}"
    )

    # --- Check 2: canonical-order confirmatory (5 seeds) ---
    print("Check 2: Z3 v2 canonical-order confirmatory...")
    canonical_results = []
    for seed in range(_SEEDS):
        print(f"  Seed {seed}...")
        m = _build_model(seed=seed, device=dev)
        ms = _meta_train_model(m, tasks, criterion, shape)
        cr = _check_canonical_order(seed, m, ms, tasks, criterion, shape)
        canonical_results.append(cr)
        print(
            f"    accs: {cr['task_accuracies']} | θ-change: {cr['theta_change']:.8f} | pass={cr['pass']}"
        )
    canonical_all_pass = all(r["pass"] for r in canonical_results)
    all_pass &= canonical_all_pass
    results["checks"]["canonical_order"] = {
        "seeds": canonical_results,
        "all_seeds_pass": canonical_all_pass,
        "pass": canonical_all_pass,
    }

    # --- Check 3: order-robustness (permutations) ---
    print("Check 3: Z3 v4 order-robustness...")
    order_results = []
    permuted_orders = [
        ["threshold", "parity", "last_symbol"],
        ["last_symbol", "threshold", "parity"],
    ]
    for i, order in enumerate(permuted_orders):
        print(f"  Order {i}: {order}...")
        m = _build_model(seed=i + 100, device=dev)
        ms = _meta_train_model(m, tasks, criterion, shape)
        or_result = _check_order_robustness(
            i + 100, m, ms, tasks, criterion, shape, order
        )
        order_results.append(or_result)
        print(f"    accs: {or_result['task_accuracies']} | pass={or_result['pass']}")
    order_sensitivity_detected = not all(r["pass"] for r in order_results)
    results["checks"]["order_robustness"] = {
        "permutations_tested": permuted_orders,
        "results": order_results,
        "order_sensitivity_detected": order_sensitivity_detected,
        "all_pass": all(r["pass"] for r in order_results),
        "pass": True,  # Document sensitivity, not a failure
    }

    # --- Check 4: gate history completeness ---
    print("Check 4: Gate history instrumentation...")
    m = _build_model(seed=0, device=dev)
    ms = _meta_train_model(m, tasks, criterion, shape)
    m.load_state_dict(ms)
    m.freeze_theta()
    with ThetaInvarianceAudit(m, selector=_is_theta_param):
        adapt_rows, _ = _adapt_all_tasks(
            m,
            tasks,
            criterion,
            shape,
            epochs=_EVAL_EPOCHS,
            probe_batches=_PROBE_BATCHES,
            feedback=True,
            adapt_entropy_beta=0.1,
            adapt_temp_end=0.5,
        )
    gate_check = _check_gate_history(adapt_rows)
    results["checks"]["gate_history"] = gate_check
    all_pass &= gate_check["pass"]
    print(
        f"  Gate history complete: {gate_check['all_complete']} | pass={gate_check['pass']}"
    )

    results["all_pass"] = all_pass
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("audit_results"))
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("Phase 3.6.7 — Z3 Re-verification Audit")
    print("=" * 80)

    started = time.perf_counter()
    results = run_audit(device=device)
    elapsed = time.perf_counter() - started
    results["wall_clock_s"] = elapsed

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for name, check in results["checks"].items():
        status = "✅ PASS" if check["pass"] else "❌ FAIL"
        print(f"  {name}: {status}")
    overall = "✅ PASS" if results["all_pass"] else "❌ FAIL"
    print(f"\nOverall: {overall}")
    print(f"Elapsed: {elapsed:.1f}s")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "z3_reverification.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
