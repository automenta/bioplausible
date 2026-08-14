#!/usr/bin/env python3
"""
Static seam-checker for REFACTOR4: enforces that the consolidated seams hold.

Each criterion is encoded as ``violator-set ⊆ allowlist``. The allowlist is the
explicit, committed home for legitimate exceptions (Ground rule 14) AND today's
baseline debt. On creation each allowlist = today's violator set, so the gate
passes immediately; completing a stream step *removes* its entries, so the
allowlist ratchets down monotonically. A file NOT in the allowlist that starts
violating a seam fails CI fast — that is the regression guard.

Usage:
    python tools/check_seams.py

Exit codes:
    0 - All seam gates pass
    1 - A gate found a violator outside its allowlist
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "bioplausible"

# ── seam:loop-backward ──────────────────────────────────────────────────────
# Criterion #1 proxy: every training loop dispatches through
# `core/trainer.dispatch_train_step`, measured by zero `loss.backward()` outside
# the legitimate-measurement set below (Ground rules 2 & 3). These exclusions are
# PERMANENT (measurement / RL / RULE-scope propagators), not allowlist debt.
LOOP_EXCLUSIONS: tuple[str, ...] = (
    "core/",
    "training_mixin",
    "validation/tracks/",
    "analysis/",
    "execution/robustness",
    "execution/interpretability",
    "execution/_guards",
    "benchmarks/",
    "zoo/mep/benchmarks/",
    "training/rl.py",
    "zoo/propagators/",
)
# Baseline convertible debt (verified 2026-08-14). Remove entries as LOOP steps
# clear them: step 3 -> {cli/repro,validation/utils,lightning_/module,sklearn_interface};
# step 4 -> graph/training.py (or exempt + note); step 5 -> ewc,nebc_base; 6-8 -> rest.
LOOP_ALLOW: set[str] = {
    "cli/repro.py",
    "validation/utils.py",
    "lightning_/module.py",
    "sklearn_interface.py",
    "zoo/models/eqprop/eqprop_diffusion.py",
    "zoo/models/forward_only.py",
    "zoo/models/target_prop.py",
    "zoo/optimizers/ewc.py",
    "zoo/mep/__init__.py",
    "zoo/mep/optimizers/__init__.py",
    "zoo/nebc_base.py",
    "graph/training.py",
}

# ── seam:model-cls ───────────────────────────────────────────────────────────
# Criterion #3: zero `model_cls(` outside construction.py. Allowlist is empty.
MODEL_CLS_SKIP: tuple[str, ...] = ("construction.py", "__pycache__")

# ── seam:propagators ─────────────────────────────────────────────────────────
# Criterion #6 (RULE): zoo/propagators/ = mep.py + pure gradient transformers.
# Baseline debt = everything RULE steps 1-3 will delete or convert.
PROPAGATORS_ALLOW: set[str] = {
    "backprop.py",
    "base.py",
    "eqprop.py",
    "fa.py",
    "hebbian.py",
    "spiking.py",
}

# ── seam:result-sink ─────────────────────────────────────────────────────────
# Criterion #3b (FUNNEL + MEASURE): all outcome writes go through result_sink.
# The gate tracks the sanctioned writer set: every file that calls
# record_experiment_result must be listed here, so a NEW outcome-write site is a
# visible diff requiring review. The plan's "Already routed (don't touch)" list.
# Deep backend-bypass detection (no writer outside the sink) is finalized when
# FUNNEL lands.
RESULT_SINK_ALLOW: set[str] = {
    "core/trainer.py",
    "experiment/probe.py",
    "hyperopt/experiment.py",
    "validation/tracks/hardware_tracks.py",
}


def _grep(pattern: str, skip: tuple[str, ...] = ()) -> list[str]:
    """Return source-relative paths whose text matches ``pattern``."""
    out = subprocess.run(
        ["grep", "-rln", pattern, str(SRC)], capture_output=True, text=True
    )
    hits: list[str] = []
    for line in out.stdout.splitlines():
        path = Path(line).relative_to(SRC).as_posix()
        if "__pycache__" in path or path.endswith(".pyc"):
            continue
        if any(part in path for part in skip):
            continue
        hits.append(path)
    return hits


def _check(name: str, violators: list[str], allow: set[str], note: str = "") -> int:
    """Report a gate: sorted violators outside the allowlist fail."""
    print(f"\n=== {name} ===")
    if note:
        print(note)
    new = [v for v in violators if v not in allow]
    if new:
        print("VIOLATORS OUTSIDE ALLOWLIST:")
        for v in sorted(new):
            print(f"  {v}")
        return 1
    if violators:
        print(f"OK: {len(violators)} violator(s), all within allowlist.")
    else:
        print("OK: no violators.")
    return 0


def main() -> int:
    """Run all seam gates."""
    failures = 0

    loop_hits = _grep("loss.backward()")
    loop_violators = [p for p in loop_hits if not any(e in p for e in LOOP_EXCLUSIONS)]
    failures += _check(
        "seam:loop-backward",
        loop_violators,
        LOOP_ALLOW,
        "Criterion #1 proxy: loss.backward() outside measurement/RL/RULE set.",
    )

    model_hits = _grep("model_cls(", MODEL_CLS_SKIP)
    failures += _check(
        "seam:model-cls",
        model_hits,
        set(),
        "Criterion #3: model_cls( must live only in construction.py.",
    )

    # Criterion #6: zoo/propagators/ must contain only mep.py + pure gradient
    # transformers (rule 14; the import-consumer grep is RULE step 1's job).
    prop_violators = sorted(
        p.name
        for p in (SRC / "zoo/propagators").glob("*.py")
        if p.name not in ("mep.py", "__init__.py")
    )
    failures += _check(
        "seam:propagators",
        prop_violators,
        PROPAGATORS_ALLOW,
        "Criterion #6: files in zoo/propagators/ beyond mep.py + gradient transformers.",
    )

    sink_hits = _grep("record_experiment_result", ())
    sink_violators = [
        p for p in sink_hits if "result_sink" not in p
    ]
    failures += _check(
        "seam:result-sink",
        sink_violators,
        RESULT_SINK_ALLOW,
        "Criterion #3b: outcome writes via result_sink (finalize when FUNNEL lands).",
    )

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED: {failures} seam gate(s) have violators outside their allowlist.")
        return 1
    print("PASSED: all seam gates hold (violators within allowlists).")
    return 0


if __name__ == "__main__":
    sys.exit(main())