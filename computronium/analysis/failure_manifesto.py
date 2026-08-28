from pathlib import Path

import pandas as pd

from computronium.core._paths import db_path
from computronium.execution._state import FailureTracker

__all__ = [
    "FailureManifestoGenerator",
    "main",
    "write_continual_learning_null_memo",
    "write_z3_boundary_memo",
]


class FailureManifestoGenerator:
    """
    Auto-generates reports/failure_manifesto.md from experiment DB.
    """

    def __init__(self, db_path: str):
        self.tracker = FailureTracker(db_path)

    def generate(
        self,
        output_path: str = "reports/failure_manifesto.md",
        model: str | None = None,
    ):
        """
        Extracts failures from DB and groups them by algorithm and FailureCategory.
        Outputs a markdown manifesto report.
        """
        _ = self.tracker.get_failure_stats()
        recent_failures = self.tracker.get_recent_failures(limit=1000)

        # Build DataFrame for easier cross-tabulation
        fail_data = []
        for r in recent_failures:
            if model is not None and r.model_name != model:
                continue
            fail_data.append({
                "model": r.model_name,
                "task": r.task_name,
                "type": r.failure_type,
                "epoch": r.failure_epoch,
            })

        df = pd.DataFrame(fail_data)

        Path(output_path).parent.mkdir(exist_ok=True, parents=True)

        with Path(output_path).open("w") as f:
            f.write("# Failure Modes Manifesto\n\n")
            f.write(
                "This document tracks the explicit failure modes encountered "
                "across different computronium algorithms.\n\n"
            )
            if model is not None:
                f.write(f"### Scope: `{model}`\n\n")
            if df.empty:
                if model is not None:
                    f.write(f"No failures logged for `{model}` yet.\n")
                else:
                    f.write("No failures logged yet.\n")
                return output_path

            _write_distribution(f, df)
            _write_crosstab(f, df)
            _write_diagnostics(f, self.tracker)

        return output_path

    def generate_with_memos(
        self,
        output_path: str = "reports/failure_manifesto.md",
        model: str | None = None,
        z3: bool = False,
        continual_learning: bool = False,
    ):
        """Generate the standard manifesto plus the requested memo appendices."""
        self.generate(output_path, model)
        if not (z3 or continual_learning):
            return output_path
        with Path(output_path).open("a") as f:
            if z3:
                f.write("\n\n")
                f.write(write_z3_boundary_memo())
            if continual_learning:
                f.write("\n\n")
                f.write(write_continual_learning_null_memo())
        return output_path


def write_z3_boundary_memo() -> str:
    """Generate the Z3 boundary memo as a standalone string for inclusion in the failure manifesto.

    Cites sessions 9–14 evidence; no new runs.
    """
    return """# Appendix: Z3 Boundary Memo

## Executive Summary

The Z3 benchmark (frozen-θ algorithm switching via ψ-mediated operator selection) reached its honest endpoint across sessions 9–14. The capability is **real but order-scoped**; the speed-vs-finetune endpoint is a **null**; order-randomization was **not confirmed**; and two v4 confirmatory attempts triaged with a residual stochastic tail. This memo records the final epistemic state for citation.

---

## Session-by-Session Evidence Trail

### Session 9 (Meta-Training Repair)
- **Defects corrected:** (1) ψ-logit integrator removed (unbounded random walk, ‖ψ‖: 1→157); (2) train/eval consistency: straight-through Gumbel replaces soft mixture; (3) `TASK_OPERATOR_MAP` corrected (threshold→Identity, not Threshold — linear-probe falsified original).
- **Two-phase recipe established:** forced-operator θ warm-up → controller-only straight-through phase.
- **Promoted config:** `b02_longep_wu60` (entropy_beta=0.2, episode_len=16, warmup_fraction=0.6): 1.000 / 0.988 / 0.808 (parity/lastsym/threshold).

### Session 10 (Pilot Rerun, Post-Repair)
- **Outcome:** Δθ exact both seeds; diversity H=1.42 (>log 2, no collapse); ψ-only reaches 100-step-window criterion on parity (@107–112) and last_symbol (@107–130); threshold materially above chance (0.83–0.85) but censored at 240-step budget.
- **Critical caveat:** `random_psi` control adapts essentially as well (≈1.0 / 0.99 / 0.82–0.83) — mechanism is in-episode bandit exploration over warmed-up θ trunk, NOT meta-learned routing. Frozen floor shows threshold prior (~0.99 fresh-ψ) erodes under sequential adaptation (final 0.84).

### Session 11 (Differential Rounds R4/R5)
- **R4 finding:** Cold adaptation (T≈0.5) preserves priors but starves discovery — parity dies. Pre-adaptation routing CANNOT converge for all tasks (ψ=0 ⇒ identical inputs ⇒ one task matches default).
- **Parity trilemma:** Parity operator emits label as feature ⇒ ANY broad sampler sits at metric floor (~window size). No adapter beats floor ⇒ worst-task SPEED margins ≤0 structurally.
- **R5 promoted:** `wu60_hot` (adapt_temp=2.0): 1.000/0.996/0.992, all tasks criterion-reached, Δθ exact.
- **Speed-vs-finetune: NULL** — log step ratios cluster around 0 (±0.1 at w50) across all windows {20,50,100}; identical optimizer/step budget over same trainable surface.
- **Differential vs random-ψ: RELIABILITY, not speed** — meta-training buys task coverage (100% seeds solve all 3) vs random (~60% at hot T).

### Session 12 (v2 Confirmatory, 5 Seeds, Fixed Canonical Order)
- **Gates 3/3 PASS all seeds:** Δθ exact; all-task criterion coverage; final accuracy ≥0.9789 worst-task.
- **Primary endpoint INCONCLUSIVE (E-7 null):** mean gap = 0.2577 (>0.25 margin) but bootstrap CI [0.0764, 0.4389] straddles margin; permutation p=0.1297.
- **Autopsy:** Control is bimodal — random solves all in seeds {2,3} (≈0.99), fails last_symbol in {0,1,4} (0.52–0.60). ~40% per-seed "luck rate" pins mean gap at margin. Instrument's null is Bernoulli-mixture, not Gaussian.

### Session 13 (v3 Re-registration: Proportion Endpoint + Order Randomization)
- **New design:** Exact Fisher on failure proportions (10 seeds/arm); randomized task order per seed.
- **Result NOT CONFIRMED:** Fisher p=0.5 (z3 fails 3/10 {1,2,3}; random fails 4/10 {4,6,7,8}).
- **Load-bearing finding:** ALL 7/7 failures across both arms on PARITY alone. Coverage structure:
  - Parity first ⇒ both arms solve everything (seeds 0,5,9).
  - Z3 fails iff order = (lastsym → threshold → parity), deterministically 3/3 seeds.
  - Random fails parity in 4/6 non-parity-first cells; SOLVES parity exactly where z3 fails.
- **Interpretation:** Parity carries no installed prior (pre-adapt ≈0.49–0.51 vs threshold 0.61–0.97); bandit lock-in fragile to routing basin from preceding phases.

### Session 14 (v4 System Redesign + Confirmatory Attempts)
- **Registered changes:** (1) Per-task Adam rebuild in `_adapt_all_tasks` (identical both arms); (2) Entropy floor β=0.1; (3) Gate-history rider (per-step gates/entropy persisted).
- **Coded parity REVOKED** after E-2 triage: coded quadrature/antipodal made lastsym/threshold worse; per-task Adam rebuild alone solved the starvation mechanism.
- **Budget amended:** eval_epochs_per_task 240→400 (covers max discovery latency 239 + 100-step window).
- **First confirmatory attempt (10 seeds): NOT CONFIRMED** — z3 fails seeds {2,3} (parity never discovered within 400); random 10/10 clean.
- **Gate-history census:** All 30 solved task-phases show solver-discovery latencies 1–239 steps; every failure is budget-truncation, not acquisition failure. Protocol fix works; budget was sized before latencies known.

---

## Final Citable Claims (Scoped Precisely)

| Claim | Status | Scope |
|-------|--------|-------|
| **θ-free ψ-mediated switching** (100–400 steps/task) | ✅ CONFIRMED | Canonical order (parity→last_symbol→threshold), 5 seeds, Δθ exact, all-task criterion coverage |
| **Speed advantage vs θ-fine-tune** | ❌ NULL | Offline re-analysis at windows {20,50,100}; no config shows stable ≥1.25× ratio |
| **Order-robust coverage** | ❌ NOT CONFIRMED | v3/v4 designs: deterministic parity failures under specific orders |
| **Meta-training buys routing skill** | ⚠️ PARTIAL | Buys task COVERAGE reliability (100% vs ~60% seeds at hot T), not raw adaptation speed |

**Do NOT claim:** Zero-shot adaptation, order-robust parity, speed advantage over fine-tuning, general transformer applicability.

---

## Root-Cause Taxonomy (for Future Work)

1. **Parity self-revelation (metric flooring):** T_4 emits label as feature → random control sits at metric floor. Fix: coded parity emission (quadrature/antipodal) — REJECTED in v4 (made other tasks worse, exclusive basin deepens).
2. **Controller basin inheritance:** Shared controller (non-θ params) trained per phase; parity has no prior → lock-in depends on routing basin left by preceding phases. Fix: per-task Adam rebuild (works for acquisition, not for within-phase exploration).
3. **Exploration temperature governs everything:** Hot (T=2.0) enables discovery but sacrifices prior retention; cold preserves priors but starves discovery. No single temperature solves all three tasks robustly across orders.
4. **Metric window floors speed differences:** 100-step registered window ≥ convergence time (~30–80 steps) ⇒ ratio compression makes speed differences inexpressible.

---

## Artifacts Released (Citable, Versioned)

- **Operator library:** 8 ψ-operators (`Identity`, `Threshold`, `Accumulate`, `LastSymbol`, `Parity`, `SparseTopKRoute`, `SignFlip`, `Delay`) in `core/plasticity/rule_state.py`
- **θ-invariance audit harness:** `ThetaInvarianceAudit` in `core/plasticity/theta_audit.py` — exact-diff context manager
- **Gate-history schema:** Per-step mean gates / hard-selection histogram / entropy recorded for every adaptation arm
- **Calibration data:** `benchmark_results/z3_full/` (v2, 5 seeds), `benchmark_results/z3_proportion/` (v3, 10 seeds), `benchmark_results/z3_order_robust/` (v4 attempts)

---

## Closure Statement

Z3 is closed as a research line. The operator library, audit harness, and gate-history instrumentation are released as versioned artifacts for citation. The ψ/θ decoupling machinery, guard, and campaign stack are redirected to Phase 2 (continual learning flagship) where the same structural advantage — ψ-adaptation without θ-change — is tested against replay-based baselines with explicit memory accounting, on a problem (catastrophic forgetting) where backprop is structurally disadvantaged by design.
"""


def write_continual_learning_null_memo() -> str:
    """Return the Phase 2 continual-learning null result memo.

    Documents the E-7 kill of the pre-registered claim that FastWeightPlasticity
    (psi/theta decoupling) beats replay at matched memory on Split-MNIST. Includes
    the arm-calibration caveat that motivated Phase 3.5.
    """
    return """# Appendix: Phase 2 Continual-Learning Null Memo

## Executive Summary

The pre-registered claim — *psi/theta decoupling prevents catastrophic forgetting
better than replay at matched total memory* — was **REJECTED (E-7 null, kill
confirmed)** on Split-MNIST task-incremental, 5 seeds, paired. This memo records
the result, the arm-calibration caveat that tempers its interpretation, and the
deferred follow-ups.

---

## Design (Pre-registered via PR-4, prior to the full run)

- **Task:** Split-MNIST, 5 binary tasks (0/1, 2/3, 4/5, 6/7, 8/9), single 10-class
  output with task masking, task-incremental protocol (boundaries signaled).
- **Comparison:** FastWeightPlasticity (psi/theta decoupling, state memory) vs.
  matched-total-memory replay buffer. n=5 seeds, paired.
- **Endpoint:** backward transfer at matched memory; required superiority margin
  +0.10 (pre-registered) for the claim to survive.
- **Second comparator (informative, not pre-registered):** per-boundary forgetting.

## Result (E-7 Triage)

| Endpoint | mean_diff (fast_weights − replay) | 95% CI | p | Verdict |
|---|---|---|---|---|
| Backward transfer | **−0.062** | [−0.082, −0.039] | 0.0068 | fast_weights WORSE |
| Forgetting | **+0.081** | [0.073, 0.089] | 0.0034 | fast_weights forgets MORE |

Pre-registration required +0.10 superiority; the CI excludes the margin **in the
wrong direction**. Kill criterion honored; the claim is REJECTED.

## Arm-Calibration Caveat (drives Phase 3.5)

Inspection of `benchmark_results/continual_learning_full_rerun_v2/` shows the
non-replay arms were **not correctly calibrated** at the time of this run:

- **fast_weights:** final per-task accuracy ≈ 0.45–0.60 on the 10-class head —
  chance-level on every task after task 0. It was not learning subsequent tasks,
  so its high measured forgetting is confounded by failed acquisition, not a
  clean forgetting signature.
- **ewc:** tasks 1–4 also sit at chance (~0.31–0.62); only task 0 is learned.
- **backprop / lwf / si:** produce **bit-identical** accuracy matrices and
  forgetting curves — the LwF distillation and SI regularization paths were not
  taking effect (they collapse to the plain backprop loop).

**Interpretation:** the paired null reflects a broken fast_weights arm against a
working replay arm. It is a valid *negative result about the as-built pipeline*,
but it is **not** a clean test of the psi/theta-decoupling hypothesis. Phase 3.5
(single-task MNIST >=95%, two-task forgetting probe, credit-correctness, and
plasticity-state audit) is the gate that must pass before any scaling or any
re-tested CL comparison is trusted. See TODO5 §3.5.

## Artifacts

- `benchmark_results/continual_learning_full_rerun_v2/continual_learning_results.json`
  (5 seeds × 6 arms × task_incremental; accuracy matrix, BWT, forgetting,
  spectral-radius rider, stability kills).
- Pre-registration + kill decision logged in `DECISIONS.md`.

## Follow-ups (deferred / blocked)

- Permuted-MNIST 50-task stretch: **deferred** until Phase 3.5 arm verification
  passes (TODO5 §2.5).
- Re-test of the psi/theta hypothesis on verified arms: only legitimate after the
  Phase 3.5 gate, and only with a re-registration.
"""


def _write_distribution(f, df: pd.DataFrame) -> None:
    """Write the overall failure-type distribution table."""
    f.write("## Overall Failure Distribution\n\n")
    type_counts = df["type"].value_counts()
    f.write("| Failure Type | Count |\n")
    f.write("|--------------|-------|\n")
    for t, c in type_counts.items():
        f.write(f"| `{t}` | {c} |\n")
    f.write("\n")


def _write_crosstab(f, df: pd.DataFrame) -> None:
    """Write failures-by-model-and-type as a markdown table."""
    f.write("## Failures by Model and Type\n\n")
    cross_tab = pd.crosstab(df["model"], df["type"])
    cols = ["Model"] + list(cross_tab.columns)
    f.write("| " + " | ".join(cols) + " |\n")
    f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
    for index, row in cross_tab.iterrows():
        row_vals = [str(index)] + [str(v) for v in row.values]
        f.write("| " + " | ".join(row_vals) + " |\n")
    f.write("\n")


def _write_diagnostics(f, tracker: FailureTracker) -> None:
    """Write the advanced failure-pattern diagnostics section."""
    f.write("## Advanced Diagnostics\n\n")
    analysis = tracker.analyze_failure_patterns()
    if not analysis.get("recommendations"):
        f.write(
            "No critical failure patterns detected requiring immediate intervention.\n"
        )
        return
    for rec in analysis["recommendations"]:
        sev = rec.get("severity", "info")
        f.write(
            "### [Severity: {}] {}\n".format(
                sev.upper(), rec.get("issue", "Unknown Issue")
            )
        )
        f.write(f"- **Recommendation**: {rec.get('suggestion')}\n")
        if "affected_models" in rec:
            f.write(
                "- **Affected Models**: {}\n".format(", ".join(rec["affected_models"]))
            )
        if "details" in rec:
            f.write(f"- **Details**: {rec['details']}\n")
        f.write("\n")


def main(argv: list[str] | None = None) -> int:
    """``biopl-failure-manifesto`` entry point (Sprint 2.4).

    Generates a markdown failure-mode manifesto from the experiment failure DB,
    optionally scoped to a single model. Can also append the Z3 boundary memo.
    """
    import argparse
    import logging

    parser = argparse.ArgumentParser(
        description="Generate a markdown failure-mode manifesto from the DB."
    )
    parser.add_argument(
        "--db", default=db_path("computronium.db"), help="Path to the DB."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Only include failures for this model (e.g. eqprop_mlp).",
    )
    parser.add_argument(
        "--output",
        default="reports/failure_manifesto.md",
        help="Output markdown path.",
    )
    parser.add_argument(
        "--z3-memo",
        action="store_true",
        help="Append the Z3 boundary memo to the manifesto.",
    )
    parser.add_argument(
        "--cl-memo",
        action="store_true",
        help="Append the Phase 2 continual-learning null memo to the manifesto.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    generator = FailureManifestoGenerator(args.db)
    if args.z3_memo or args.cl_memo:
        out = generator.generate_with_memos(
            args.output,
            model=args.model,
            z3=args.z3_memo,
            continual_learning=args.cl_memo,
        )
    else:
        out = generator.generate(args.output, model=args.model)
    logging.info("wrote failure manifesto to %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
