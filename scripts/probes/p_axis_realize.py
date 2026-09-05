"""P-axis primitive realization probe (F3 audit follow-up, 2026-09-05).

Measures the REALIZED primitives at the F3 registered regime (A40/B40,
5 seeds, lr 0.03, synthetic 8-class):
  - routing: gate drive is now a fixed per-gate projection (gates
    differentiate) and modulate is per-unit sigmoid masks — is the gate
    trace still flat? does per-unit mask variance > 0? does retention
    still beat null, and is it still an effective-lr effect?
  - fast_weights: ψ steps on the settled FREE activity (not the raw
    target) — is the modulation still gradient-live (θ gap monotone from
    identical inits)? retention vs null?
  - null control at lr 0.015 (the old effective-lr brake check).

Reuses the F3 test harness pieces. Throwaway: informs the F3 test
re-audit; numbers recorded in its docstring, not here.
"""

import itertools
import time

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
from computronium.state import CompositeState

ARMS = ("null", "fast_weights", "routing")
SEEDS = tuple(range(5))
SEGMENTS = (("A", 40), ("B", 40))
CHANCE = 1 / 8
LR = 0.03
LR_CONTROL = 0.015

_CONFIG = TrialConfig(segments=SEGMENTS, seeds=SEEDS, device="cpu", lr=LR)


def _coordinate(arm: str) -> str:
    return f"digital/feedforward/instantaneous/{arm}/gradient/euclidean"


def _walk_arm(arm: str, *, lr: float = LR) -> tuple[float, dict]:
    config = (
        _CONFIG
        if lr == LR
        else TrialConfig(segments=SEGMENTS, seeds=SEEDS, device="cpu", lr=lr)
    )
    coordinate = _coordinate(arm)
    latencies = []
    a_mastery, a_retained = [], []
    for seed in SEEDS:
        joint = _compose(coordinate, config)
        episode = 0
        boundary = []
        for segment, n in config.segments:
            for _ in range(n):
                started = time.perf_counter()
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
                latencies.append(time.perf_counter() - started)
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
    return (
        float(np.median(latencies)) * 1e3,
        {
            "a_mastery": a_mastery,
            "a_retained": a_retained,
            "retention_mean": float(np.mean(a_retained)),
            "retention_sd": float(np.std(a_retained, ddof=1)),
        },
    )


def _routing_instruments() -> dict:
    coordinate = _coordinate("routing")
    joint = _compose(coordinate, _CONFIG)
    plasticity = joint.plasticity
    x, _ = episode_batch(
        0,
        task_name="synthetic",
        batch_size=_CONFIG.batch_size,
        input_dim=_CONFIG.input_dim,
        num_classes=_CONFIG.num_classes,
        teacher_key=(TRIAL_CAMPAIGN_ID, coordinate, 0, PROBED_SEGMENT),
        teacher_noise=_CONFIG.teacher_noise,
    )
    x = x.to(joint.device)
    psi = plasticity.initial_psi(joint.context, batch_size=x.shape[0])
    trace = []
    mask_spreads = []
    with torch.no_grad():
        for episode in range(80):
            psi = plasticity.step(
                psi,
                CompositeState(activity={"x": x, "y": x}, plastic=psi, substrate={}),
                joint.context,
            )
            trace.append(float(torch.sigmoid(psi["gate_logits"]).mean()))
            proj = plasticity._unit_projection(0, 16, psi["gate_logits"].device)
            mask = torch.sigmoid(psi["gate_logits"] @ proj)
            mask_spreads.append(float(mask.std(dim=0).mean()))
    return {
        "gate_mean": float(np.mean(trace)),
        "gate_spread": float(np.ptp(trace)),
        "mask_unit_std": float(np.mean(mask_spreads)),
    }


def _theta_divergence() -> tuple[float, bool]:
    coords = {arm: _coordinate(arm) for arm in ("null", "fast_weights")}
    joints = {arm: _compose(c, _CONFIG) for arm, c in coords.items()}
    with torch.no_grad():
        for name, param in joints["null"].geometry.params.items():
            joints["fast_weights"].geometry.params[name].data.copy_(param.data)

    def gap() -> float:
        return float(
            np.mean([
                (
                    joints["fast_weights"].geometry.params[n]
                    - joints["null"].geometry.params[n]
                )
                .norm()
                .item()
                / joints["null"].geometry.params[n].norm().item()
                for n in joints["null"].geometry.params
            ])
        )

    gaps = []
    for episode in range(5):
        for arm, joint in joints.items():
            x, y = episode_batch(
                episode,
                task_name="synthetic",
                batch_size=_CONFIG.batch_size,
                input_dim=_CONFIG.input_dim,
                num_classes=_CONFIG.num_classes,
                teacher_key=(TRIAL_CAMPAIGN_ID, coords[arm], 0, PROBED_SEGMENT),
                teacher_noise=_CONFIG.teacher_noise,
            )
            joint.train_step(x, y)
        gaps.append(gap())
    return gaps[-1], all(b > a for a, b in itertools.pairwise(gaps))


if __name__ == "__main__":
    inst = _routing_instruments()
    print(
        f"routing: gate mean {inst['gate_mean']:.4f} "
        f"spread {inst['gate_spread']:.4f} "
        f"per-unit mask std {inst['mask_unit_std']:.4f}"
    )
    fw_gap, fw_monotone = _theta_divergence()
    print(f"fw: theta gap ep5 {fw_gap:.4f} monotone {fw_monotone}")
    walks = {}
    for arm in ARMS:
        walks[arm] = _walk_arm(arm)
    walks["null_lr015"] = _walk_arm("null", lr=LR_CONTROL)
    for arm, (lat, data) in walks.items():
        wins_vs_null = (
            int(
                np.sum(
                    np.array(data["a_retained"])
                    >= np.array(walks["null"][1]["a_retained"])
                )
            )
            if arm != "null"
            else -1
        )
        print(
            f"{arm:>13}: latency {lat:.2f} ms/ep | "
            f"mastery {np.mean(data['a_mastery']):.3f} "
            f"retained {data['retention_mean']:.3f}±{data['retention_sd']:.3f} "
            f"(wins vs null {wins_vs_null}/5)"
        )
