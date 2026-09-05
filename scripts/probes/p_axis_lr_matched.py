"""P-axis lr-matched control pilot (E-1, 2026-09-05).

The F3 realization audit left one OPEN item: routing's retention
advantage over null (0.273 vs 0.194) could be an effective-lr effect —
the per-unit masks average ≈ 0.5, so routing's gradients are ~half of
null's at the same nominal lr. This pilot quantifies the confound at
the registered A40/B40 regime:

1. Measure per-episode θ displacement ||Δθ|| for each arm from identical
   inits (one episode, same data) and for null across an lr grid.
2. Interpolate each arm's lr-matched null (the null lr whose single-
   episode displacement equals the arm's).
3. Run the full persistent-θ walk (5 seeds) with null@matched-lr per
   arm and compare retention against the arm's own retention.

Outcome: routing's advantage DISSOLVES at matched lr (null@0.0154
retains 0.294 vs routing 0.273); fast-weights' deficit is real.
Promoted into the F3 test as record["mechanism_audit"]["lr_matched_audit"].
"""

import itertools

import numpy as np
import torch

from computronium.core.campaign.evaluation import (
    episode_batch,
    evaluate_episode,
    probe_episode,
)
from computronium.experiments.joint.forgetting_trial import (
    PROBE_EPISODE_BASE,
    PROBED_SEGMENT,
    TRIAL_CAMPAIGN_ID,
    TrialConfig,
    _compose,
)

ARMS = ("null", "fast_weights", "routing")
SEEDS = tuple(range(5))
SEGMENTS = (("A", 40), ("B", 40))
LR = 0.03

_CONFIG = TrialConfig(segments=SEGMENTS, seeds=SEEDS, device="cpu", lr=LR)


def _coordinate(arm: str) -> str:
    return f"digital/feedforward/instantaneous/{arm}/gradient/euclidean"


def _theta_displacement(coordinate: str, lr: float) -> float:
    """Mean relative per-parameter ||Δθ|| over 3 episodes from identical
    fixed init (seed 11), same episode data — the arm's effective step
    size, independent of the walk trajectory."""
    config = TrialConfig(segments=SEGMENTS, seeds=SEEDS, device="cpu", lr=lr)
    displacements = []
    for episode in range(3):
        torch.manual_seed(11)
        joint = _compose(coordinate, config)
        before = {n: p.detach().clone() for n, p in joint.geometry.params.items()}
        x, y = episode_batch(
            episode,
            task_name="synthetic",
            batch_size=_CONFIG.batch_size,
            input_dim=_CONFIG.input_dim,
            num_classes=_CONFIG.num_classes,
            teacher_key=(TRIAL_CAMPAIGN_ID, coordinate, 0, PROBED_SEGMENT),
            teacher_noise=_CONFIG.teacher_noise,
        )
        joint.train_step(x, y)
        disp = float(
            np.mean([
                (joint.geometry.params[n] - before[n]).norm().item()
                / before[n].norm().item()
                for n in before
            ])
        )
        displacements.append(disp)
    return float(np.mean(displacements))


def _matched_lr(arm: str, grid: tuple[float, ...]) -> tuple[float, float, float]:
    """(matched_lr, arm_disp, null_disp_at_matched) via log-linear
    interpolation on the null displacement grid."""
    null_coord = _coordinate("null")
    arm_disp = _theta_displacement(_coordinate(arm), LR)
    grid_disp = [(lr, _theta_displacement(null_coord, lr)) for lr in grid]
    matched = grid[0]
    for (lr0, d0), (lr1, d1) in itertools.pairwise(grid_disp):
        if d0 <= arm_disp <= d1:  # displacement is monotone in lr
            t = (np.log(arm_disp) - np.log(d0)) / (np.log(d1) - np.log(d0))
            matched = float(lr0 * (lr1 / lr0) ** t)
            break
    else:
        matched = grid_disp[-1][0] if arm_disp > grid_disp[-1][1] else grid_disp[0][0]
    return matched, arm_disp, _theta_displacement(null_coord, matched)


def _walk_retention(coordinate: str, lr: float) -> tuple[float, float]:
    """(mastery_mean, retention_mean) over 5 seeds."""
    config = TrialConfig(segments=SEGMENTS, seeds=SEEDS, device="cpu", lr=lr)
    a_mastery, a_retained = [], []
    for seed in SEEDS:
        joint = _compose(coordinate, config)
        episode = 0
        boundary = []
        for segment, n in config.segments:
            for _ in range(n):
                evaluate_episode(
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
                episode += 1
            boundary.append(
                float(
                    np.mean([
                        probe_episode(
                            joint,
                            coordinate=coordinate,
                            task_name="synthetic",
                            campaign_id=TRIAL_CAMPAIGN_ID,
                            episode=PROBE_EPISODE_BASE + i,
                            batch_size=config.batch_size,
                            input_dim=config.input_dim,
                            num_classes=config.num_classes,
                            seed=seed,
                            stationary_teacher=True,
                            teacher_noise=config.teacher_noise,
                            segment=PROBED_SEGMENT,
                        )
                        for i in range(8)
                    ])
                )
            )
        a_mastery.append(boundary[0])
        a_retained.append(boundary[-1])
    return float(np.mean(a_mastery)), float(np.mean(a_retained))


if __name__ == "__main__":
    grid = (0.0075, 0.015, 0.03, 0.06)
    matched: dict[str, float] = {}
    for arm in ("fast_weights", "routing"):
        lr_m, arm_d, null_d = _matched_lr(arm, grid)
        matched[arm] = lr_m
        print(
            f"{arm:>13}: displacement {arm_d:.4f} @lr{LR} "
            f"-> matched null lr {lr_m:.4f} (displacement {null_d:.4f})"
        )
    print(f"{'null':>13}: nominal walk @lr{LR}")
    m, r = _walk_retention(_coordinate("null"), LR)
    print(f"{'null':>13}: mastery {m:.3f} retained {r:.3f}")
    for arm in ("fast_weights", "routing"):
        m_arm, r_arm = _walk_retention(_coordinate(arm), LR)
        m_ctl, r_ctl = _walk_retention(_coordinate("null"), matched[arm])
        print(
            f"{arm:>13}: mastery {m_arm:.3f} retained {r_arm:.3f} | "
            f"null@lr{matched[arm]:.4f}: mastery {m_ctl:.3f} retained {r_ctl:.3f} "
            f"-> arm-advantage {r_arm - r_ctl:+.3f}"
        )
