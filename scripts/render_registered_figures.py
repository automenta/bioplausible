"""Render registered-scale figures for RESULTS.md's corroboration appendix.

Historical figures drawn from preregistered artifacts under
``benchmark_results/`` — labeled history, never live claims (TODO10 R10.1.5:
the front page carries only live demonstrations). Each render writes the PNG
plus a sidecar ``<name>.json`` recording the source artifact, its checksum,
the rendering commit, and the source run's provenance status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess  # ruff: ignore[suspicious-subprocess-import]
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from collections.abc import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "docs" / "figures" / "registered"

ARM_STYLE = {
    "gradient": ("gradient (BPTT-profiled)", "#1f77b4"),
    "thermodynamic_contrast": ("thermodynamic contrast (BPTT-profiled)", "#9467bd"),
    "random_projections": ("random projections (O(1)-memory)", "#2ca02c"),
    "control": ("control (lr=0)", "#7f7f7f"),
}


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            [  # ruff: ignore[start-process-with-partial-path] git is on PATH
                "git",
                "rev-parse",
                "HEAD",
            ],
            text=True,
            timeout=10,
        ).strip()
    except OSError, subprocess.SubprocessError:
        return "unknown"


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def deep_credit_cliff(artifact: Path, out_png: Path, out_meta: Path) -> None:
    """The depth cliff: probe accuracy vs depth per credit arm, 16 seeds."""
    data = json.loads(artifact.read_text(encoding="utf-8"))
    envs = data["envs"]
    depths = [env["depth"] for env in envs]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for arm, (label, color) in ARM_STYLE.items():
        means = [_mean(data["arms"][arm]["probe_by_env"][env["name"]]) for env in envs]
        ax.plot(depths, means, marker="o", color=color, label=label)
    ax.set_xscale("log")
    ax.set_xticks(depths, [str(d) for d in depths])
    ax.set_xlabel("network depth")
    ax.set_ylabel("mean probe accuracy (16 seeds)")
    ax.set_title("The depth cliff — registered study (historical)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    out_meta.write_text(
        json.dumps(
            {
                "figure": out_png.name,
                "source_artifact": str(artifact.relative_to(REPO_ROOT)),
                "source_artifact_sha256": hashlib.sha256(
                    artifact.read_bytes()
                ).hexdigest(),
                "source_run_provenance": "unknown (artifact records no git commit)",
                "rendered_by": str(Path(__file__).relative_to(REPO_ROOT)),
                "rendered_at_commit": _git_commit(),
                "scope_label": "registered research scale (historical, not a live claim)",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


FIGURES: dict[str, Callable[[Path, Path, Path], None]] = {
    "deep_credit_cliff": deep_credit_cliff,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", choices=sorted(FIGURES), help="render a single registered figure"
    )
    args = parser.parse_args(argv)

    names = [args.only] if args.only else sorted(FIGURES)
    rendered: list[str] = []
    for name in names:
        artifact = (
            REPO_ROOT
            / "benchmark_results"
            / {
                "deep_credit_cliff": "deep_credit_registered.json",
            }[name]
        )
        if not artifact.exists():
            print(f"registered-figure: missing artifact {artifact}", flush=True)
            return 1
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        png = OUT_DIR / f"{name}.png"
        FIGURES[name](artifact, png, OUT_DIR / f"{name}.json")
        rendered.append(png.name)
        print(f"registered-figure: {png}", flush=True)
    print(f"registered-figures: rendered {len(rendered)} -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
