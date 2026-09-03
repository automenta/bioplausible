#!/usr/bin/env python
"""Analyze null arm (directed_ep with feedback_gain=0) separately.

Reads diagnostics.json files, identifies null-arm runs by
``config_extras.feedback_gain == 0.0``, fits depth-scale slope.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


def _slope_vs_depth(points: list[tuple[int, float]]) -> dict[str, float]:
    """Fit log(metric) vs depth by OLS; returns slope/intercept/R²/n."""
    xs = [float(d) for d, _ in points]
    ys = [math.log(v) for _, v in points]
    n = len(xs)
    if n < 3:
        return {"slope": 0.0, "intercept": 0.0, "r2": 0.0, "n": n}
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    sxx = sum((x - x_mean) ** 2 for x in xs)
    if sxx == 0:
        return {"slope": 0.0, "intercept": 0.0, "r2": 0.0, "n": n}
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / sxx
    intercept = y_mean - slope * x_mean
    y_hat = [slope * x + intercept for x in xs]
    ss_res = sum((y - yh) ** 2 for y, yh in zip(ys, y_hat))
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"slope": slope, "intercept": intercept, "r2": r2, "n": n}


def _depth_scale(summaries: list[dict]) -> dict:
    """Per-depth mean delta ratios + OLS slope fit (depths 2+, per protocol)."""
    points_all: list[tuple[int, float]] = []
    points_24: list[tuple[int, float]] = []
    per_depth: dict[int, list[float]] = {}
    for s in summaries:
        ratios = [r["delta_ratio"] for r in s.get("signal_ratios", [])]
        mean_ratio = sum(ratios) / len(ratios) if ratios else float("nan")
        depth = int(s["num_layers"])
        if mean_ratio > 0 and math.isfinite(mean_ratio):
            points_all.append((depth, mean_ratio))
            if depth >= 2:
                points_24.append((depth, mean_ratio))
            per_depth.setdefault(depth, []).append(mean_ratio)
    return {
        "fit_all": _slope_vs_depth(points_all),
        "fit_2plus": _slope_vs_depth(points_24),
        "per_depth": {d: v for d, v in sorted(per_depth.items())},  # ruff: ignore[unnecessary-comprehension]
        "n": len(summaries),
    }


def main():
    output_base = Path("runs/contrastive_profile/b2_autopsy")

    print("=" * 72)
    print("B2 AUTOPSY — NULL ARM ANALYSIS (feedback_gain=0)")
    print("=" * 72)

    for beta in [0.01, 0.03, 0.1]:
        model_dir = output_base / f"directed_ep_beta{beta}"
        if not model_dir.exists():
            continue

        diag_paths = list(model_dir.rglob("diagnostics.json"))
        null_runs: list[dict] = []
        regular_runs: list[dict] = []
        untagged: list[dict] = []

        for diag_path in diag_paths:
            summary = json.loads(diag_path.read_text(encoding="utf-8"))
            extras = summary.get("config_extras", {})
            fb = extras.get("feedback_gain")
            if fb == 0.0:
                null_runs.append(summary)
            elif fb is None:
                # Pre-fix runs: assume regular (the first run timestamp)
                untagged.append(summary)
            else:
                regular_runs.append(summary)

        print(f"\n--- Beta={beta} ---")
        print(f"  Total diagnostics.json: {len(diag_paths)}")
        print(f"  Tagged null arm: {len(null_runs)}")
        print(f"  Tagged regular: {len(regular_runs)}")
        print(f"  Untagged (pre-fix): {len(untagged)}")

        # The untagged runs are from before we added config_extras.
        # Since null arm runs were done AFTER the regular runs (later timestamps),
        # and some failed, the untagged ones are a mix. We can't reliably separate
        # them without the tag. Report what we can.

        if null_runs:
            analysis = _depth_scale(null_runs)
            print("\n  NULL ARM (tagged):")
            print(
                f"    Slope (all depths): {analysis['fit_all']['slope']:.4f}, R²={analysis['fit_all']['r2']:.4f}"
            )
            print(
                f"    Slope (depth 2+):   {analysis['fit_2plus']['slope']:.4f}, R²={analysis['fit_2plus']['r2']:.4f}"
            )
            for depth, ratios in sorted(analysis["per_depth"].items()):
                median = sorted(ratios)[len(ratios) // 2]
                print(
                    f"    Depth {depth}: median={median:.4g}, n={len(ratios)}, all={[f'{r:.3g}' for r in sorted(ratios)]}"
                )

        if regular_runs:
            analysis = _depth_scale(regular_runs)
            print("\n  REGULAR (tagged):")
            print(
                f"    Slope (all depths): {analysis['fit_all']['slope']:.4f}, R²={analysis['fit_all']['r2']:.4f}"
            )
            print(
                f"    Slope (depth 2+):   {analysis['fit_2plus']['slope']:.4f}, R²={analysis['fit_2plus']['r2']:.4f}"
            )
            for depth, ratios in sorted(analysis["per_depth"].items()):
                median = sorted(ratios)[len(ratios) // 2]
                print(
                    f"    Depth {depth}: median={median:.4g}, n={len(ratios)}, all={[f'{r:.3g}' for r in sorted(ratios)]}"
                )

    print("\n" + "=" * 72)
    print("NOTE: Untagged runs (pre-fix) cannot be reliably separated.")
    print("Re-running null arm with the config_extras tag is needed for")
    print("a clean null-arm comparison.")
    print("=" * 72)


if __name__ == "__main__":
    main()
