"""CP-6 Path B, E-1 smoke (TODO11): the P-axis Pareto — settling/latency
cost vs basin stability per plasticity primitive.

The doctrine deliverable: "the settling-time (latency) vs basin-stability
trade-off curve per plasticity primitive — the figure that justifies the 6-D
ontology to hardware researchers." This smoke measures whether the axes
DISCRIMINATE at demo scale before any campaign is commissioned (RESEARCH3
risk note: "proxy metrics may not discriminate primitives at small scale").

Design: each M arm (null / fast_weights / routing) walks the R9.1 segmented
stream A→B per seed (forgetting-trial machinery, persistent θ, segment-keyed
stationary teachers). Per episode the walk collects the frontier resource
vector (wall-clock latency, work-derived compute/energy, ψ-capacity). After
the walk, two stability scalars on held-out segment-A batches:
  - retention: the standard R9.1 held-out A-probe accuracy (memory stability);
  - basin stability: fraction of input perturbations (radius r) that leave
    the unperturbed prediction unchanged — the width of the decision basins
    the learned solution carves.

Pareto point per primitive: (mean episode latency, basin stability), with
retention and ψ-capacity as annotations. Multi-seed mean ± sd throughout.

Run: uv run python scripts/probes/p_axis_pareto.py
"""

import json
import time
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch

from computronium.core.campaign.evaluation import episode_batch, resolve_device
from computronium.experiments.joint.forgetting_trial import (
    PROBE_EPISODE_BASE,
    PROBED_SEGMENT,
    TRIAL_CAMPAIGN_ID,
    TrialConfig,
    _compose,
    _probe,
)

ARMS = ("null", "fast_weights", "routing")
SEEDS = (0, 1, 2, 3, 4)
SEGMENTS = (("A", 40), ("B", 40))
BASIN_RADII = (0.25, 0.5, 1.0, 2.0, 4.0)
BASIN_DRAWS = 8  # perturbation draws per probe episode
BASIN_EPISODES = 8


def _basin_curve(
    joint, coordinate: str, config: TrialConfig, seed: int
) -> tuple[list[float], float]:
    """Agreement-vs-radius curve on held-out segment-A batches (probe
    episode space — same keying discipline as the R9.1 probe), plus the
    r50 basin-width scalar: the smallest swept radius where agreement
    drops to/below 0.5 (the radius at which half the input perturbations
    leave the learned decision basin)."""
    curves: dict[float, list[float]] = {r: [] for r in BASIN_RADII}
    device = joint.device
    with torch.no_grad():
        for i in range(BASIN_EPISODES):
            x, _ = episode_batch(
                PROBE_EPISODE_BASE + 100 + i,
                task_name="synthetic",
                batch_size=config.batch_size,
                input_dim=config.input_dim,
                num_classes=config.num_classes,
                teacher_key=(TRIAL_CAMPAIGN_ID, coordinate, seed, PROBED_SEGMENT),
                teacher_noise=config.teacher_noise,
            )
            x = x.to(device)
            base_pred = joint.forward(x).argmax(dim=-1)
            gen = torch.Generator().manual_seed(seed * 10_000 + i)
            for _ in range(BASIN_DRAWS):
                direction = torch.nn.functional.normalize(
                    torch.randn(x.shape, generator=gen).to(device), dim=-1
                )
                for radius in BASIN_RADII:
                    pert_pred = joint.forward(x + radius * direction).argmax(dim=-1)
                    curves[radius].append(
                        (pert_pred == base_pred).float().mean().item()
                    )
    means = [mean(curves[r]) for r in BASIN_RADII]
    r50 = next((r for r, m in zip(BASIN_RADII, means, strict=True) if m <= 0.5), None)
    return means, r50


def _walk(arm: str, config: TrialConfig) -> dict:  # noqa: PLR0914
    """One arm: persistent-θ walk + resource vector + stability scalars."""
    coordinate = f"digital/feedforward/instantaneous/{arm}/gradient/euclidean"
    latencies: list[float] = []
    computes: list[float] = []
    energies: list[float] = []
    a_mastery: list[float] = []
    a_retained: list[float] = []
    basin_curves: list[list[float]] = []
    r50s: list[float | None] = []
    for seed in SEEDS:
        joint = _compose(coordinate, config)
        episode = 0
        boundary: list[float] = []
        for seg, n in config.segments:
            for _ in range(n):
                started = time.perf_counter()
                record, _ = _episode(joint, coordinate, config, seed, episode, seg)
                latencies.append(time.perf_counter() - started)
                computes.append(record.resources.compute)
                energies.append(record.resources.energy)
                episode += 1
            boundary.append(_probe(joint, coordinate, config, seed))
        a_mastery.append(boundary[0])
        a_retained.append(boundary[-1])
        means, r50 = _basin_curve(joint, coordinate, config, seed)
        basin_curves.append(means)
        r50s.append(r50)
    r50_vals = [r for r in r50s if r is not None]
    return {
        "arm": arm,
        "latency_ms_median": float(np.median(latencies)) * 1e3,
        "compute_mean": mean(computes),
        "energy_j_mean": mean(energies),
        "psi_capacity": float(
            sum(
                (
                    _compose(coordinate, config).plasticity.config.plastic_state_dims
                    or {}
                ).values()
            )
        ),
        "a_mastery": a_mastery,
        "a_retained": a_retained,
        "retention_mean": mean(a_retained),
        "retention_sd": stdev(a_retained) if len(a_retained) > 1 else 0.0,
        "basin_radii": list(BASIN_RADII),
        "basin_curve_mean": [
            float(np.mean(col)) for col in zip(*basin_curves, strict=True)
        ],
        "r50": r50_vals,
        "r50_median": float(np.median(r50_vals)) if r50_vals else None,
    }


def _episode(joint, coordinate, config, seed, episode, segment):
    from computronium.core.campaign.evaluation import evaluate_episode

    return evaluate_episode(
        joint,
        coordinate=coordinate,
        task_name="synthetic",
        campaign_id=TRIAL_CAMPAIGN_ID,
        episode=episode,
        batch_size=config.batch_size,
        input_dim=config.input_dim,
        num_classes=config.num_classes,
        guard_threshold=None,
        seed=seed,
        stationary_teacher=True,
        teacher_noise=config.teacher_noise,
        segment=segment,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = TrialConfig(
        segments=SEGMENTS,
        seeds=SEEDS,
        device=args.device,
    )
    device = resolve_device(config.device)
    rows = []
    for arm in ARMS:
        started = time.perf_counter()
        row = _walk(arm, config)
        row["wall_s"] = time.perf_counter() - started
        rows.append(row)
        print(
            f"{arm:>13}: latency {row['latency_ms_median']:.2f} ms/ep "
            f"compute {row['compute_mean']:.3g} psi {row['psi_capacity']:.0f} | "
            f"mastery {mean(row['a_mastery']):.3f} "
            f"retained {row['retention_mean']:.3f}±{row['retention_sd']:.3f} | "
            f"basin@r {[format(m, '.2f') for m in row['basin_curve_mean']]} "
            f"r50 {row['r50_median']} "
            f"[{row['wall_s']:.0f} s, {device}]"
        )
    out = Path("benchmark_results/p_axis_pareto_smoke.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
