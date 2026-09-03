"""R7 probe #9 (imp-51): statistical specification audit for the r5b_b null.

Question: is the 0.05-claimable-floor test powered for the effect sizes the
campaign actually exhibits? A mis-specified/underpowered test converts "no
evidence of effect" into "evidence of no effect" — the exact error the
fidelity policy forbids.

Output shape (per comparison): observed Cohen's d (task_loss, arm vs null)
→ min detectable effect at 80% power for the observed group sizes →
verdict: powered / underpowered. Uses ``power_for_two_sample`` from the
PR-4 statistics kit.

Scope note: task_loss is claim-grade per the imp-46 census (free-settle
reads); the resource axes of these pre-fix records remain quarantined and
are not audited here. The registered null itself stays uninterpreted.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from computronium.validation.power_preregistration import (
    DEFAULT_TARGET_POWER as TARGET_POWER,
)
from computronium.validation.power_preregistration import min_detectable_effect
from computronium.validation.statistics import cohens_d, power_for_two_sample


def audit(episodes_path: Path, alpha: float) -> list[dict[str, object]]:
    records = json.loads(episodes_path.read_text(encoding="utf-8"))
    arms: dict[str, list[float]] = defaultdict(list)
    for record in records:
        arms[record["coordinate"].split("/")[3]].append(record["task_loss"])
    null = arms["null"]
    n = len(null)

    comparisons: list[tuple[str, list[float]]] = [
        (f"null vs {k}", v) for k, v in sorted(arms.items()) if k != "null"
    ]
    pooled = [v for k, v in arms.items() if k != "null" for v in v]
    comparisons.append(("null vs plasticity-pooled", pooled))

    mde = min_detectable_effect(n, alpha)
    rows: list[dict[str, object]] = []
    for label, group in comparisons:
        d = cohens_d(null, group)
        rows.append({
            "comparison": label,
            "n_null": n,
            "n_arm": len(group),
            "cohens_d": round(d, 4),
            "mde_80": round(mde, 4),
            "power_at_observed_d": round(power_for_two_sample(abs(d), n, alpha), 4),
            "verdict": "powered" if abs(d) >= mde else "underpowered",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes",
        type=Path,
        default=Path("autoscientist_campaigns/r5b_b/records/episodes.json"),
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    rows = audit(args.episodes, args.alpha)
    powered = all(row["verdict"] == "powered" for row in rows)
    print(f"statistical specification audit — {args.episodes}")
    print(f"alpha={args.alpha}  target_power={TARGET_POWER}")
    for row in rows:
        print(
            f"  {row['comparison']:<28} d={row['cohens_d']:+.4f} "
            f"n={row['n_arm']:<4} MDE@80%={row['mde_80']:.4f} "
            f"power@d={row['power_at_observed_d']:.3f}  -> {row['verdict']}"
        )
    mde = float(rows[0]["mde_80"]) if rows else float("inf")
    print(
        f"VERDICT: {'powered' if powered else 'UNDERPOWERED'} — "
        + (
            "the 0.05 floor is interpretable at this scale"
            if powered
            else "the null is not evidence of no effect; scale or design must change first"
        )
    )
    if not powered and not math.isfinite(mde):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
