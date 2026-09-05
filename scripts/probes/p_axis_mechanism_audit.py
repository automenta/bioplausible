"""F3 mechanism audit (R11.5.5a discipline): before the P-axis Pareto's
attribution story stands, check the instruments for what the primitives
ACTUALLY do.

Three suspicions from reading the implementations:
1. RoutingPlasticity.modulate applies mean(sigmoid(gate_logits)) as a
   per-sample SCALAR GAIN to every layer — with gate drive |x|bar and
   lr 0.01 the logits may stay ~0 for the whole walk, making routing a
   constant ~0.5 gain = a disguised learning-rate brake. Control: null at
   halved lr — does it retain like routing?
2. FastWeightPlasticity.modulate ADDS proj(outer(x, y_target)) to every
   layer's activations — target-correlated bias injection, not
   associative memory. Control: measure the actual theta-trajectory
   divergence vs null and whether fw's modulation changes gradients
   (constant shift through downstream layers does).
3. Gate-strength trajectory: log mean sigmoid(gate_logits) across the
   walk — if ~0.5 throughout, the "routing" primitive implements gain
   control, not routing (a primitive-level implementation fact).

Run: uv run python scripts/probes/p_axis_mechanism_audit.py
"""

import json
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from computronium.core.campaign.evaluation import evaluate_episode, probe_episode
from computronium.experiments.joint.forgetting_trial import (
    PROBE_EPISODE_BASE,
    PROBED_SEGMENT,
    TRIAL_CAMPAIGN_ID,
    TrialConfig,
    _compose,
)

SEEDS = (0, 1, 2, 3, 4)
SEGMENTS = (("A", 40), ("B", 40))
LR_GRID = (0.03, 0.015, 0.01)


def _probe(joint, coordinate: str, seed: int) -> float:
    return float(
        np.mean(
            [
                probe_episode(
                    joint,
                    coordinate=coordinate,
                    task_name="synthetic",
                    campaign_id=TRIAL_CAMPAIGN_ID,
                    episode=PROBE_EPISODE_BASE + i,
                    batch_size=16,
                    input_dim=8,
                    num_classes=8,
                    seed=seed,
                    stationary_teacher=True,
                    teacher_noise=0.1,
                    segment=PROBED_SEGMENT,
                )
                for i in range(8)
            ]
        )
    )


def _walk(coordinate: str, config: TrialConfig, *, trace_gates: bool = False):
    lat_retained, gate_trace = [], []
    for seed in config.seeds:
        joint = _compose(coordinate, config)
        plasticity = joint.plasticity
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
                if trace_gates and hasattr(plasticity, "_config"):
                    psi_gate = getattr(joint.plasticity, "_last_gate_strength", None)
                    episode += 1
                episode += 1
            boundary.append(_probe(joint, coordinate, seed))
        lat_retained.append(boundary[-1])
    return lat_retained


def _gate_strength_trace(config: TrialConfig) -> dict:
    """Direct instrument: walk routing one seed and read gate strength each
    step by calling step() on the same input the episode sees."""
    coordinate = "digital/feedforward/instantaneous/routing/gradient/euclidean"
    trace = []
    for seed in config.seeds[:1]:
        joint = _compose(coordinate, config)
        for episode in range(80):
            x, _ = _episode_input(episode, seed)
            x = x.to(joint.device)
            psi = joint.plasticity.initial_psi(joint.context, batch_size=16)
            z = _composite(x)
            psi = joint.plasticity.step(psi, z, joint.context)
            strength = torch.sigmoid(psi["gate_logits"]).mean(dim=-1)
            trace.append(float(strength.mean()))
    return {"mean": float(np.mean(trace)), "first5": trace[:5], "last5": trace[-5:]}


def _episode_input(episode: int, seed: int):
    from computronium.core.campaign.evaluation import episode_batch

    return episode_batch(
        episode,
        task_name="synthetic",
        batch_size=16,
        input_dim=8,
        num_classes=8,
        teacher_key=(TRIAL_CAMPAIGN_ID, "routing", seed, PROBED_SEGMENT),
        teacher_noise=0.1,
    )


def _composite(x):
    from computronium.state import CompositeState

    return CompositeState(activity={"x": x}, plastic={}, substrate={})


def _theta_divergence(config: TrialConfig) -> dict:
    """How fast does fw's theta trajectory separate from null's?"""
    coords = {
        arm: f"digital/feedforward/instantaneous/{arm}/gradient/euclidean"
        for arm in ("null", "fast_weights")
    }
    joints = {arm: _compose(c, config) for arm, c in coords.items()}
    divs = []
    for episode in range(10):
        for arm, joint in joints.items():
            x, y = _episode_input(episode, 0)
            joint.train_step(x.to(joint.device), y.to(joint.device))
        names = list(joints["null"].geometry.params)
        gaps = [
            (
                joints["fast_weights"].geometry.params[n]
                - joints["null"].geometry.params[n]
            )
            .norm()
            .item()
            / joints["null"].geometry.params[n].norm().item()
            for n in names
        ]
        divs.append(float(np.mean(gaps)))
    return {"relative_param_gap_mean": mean(divs), "per_episode": divs}


def main() -> None:
    base = TrialConfig(segments=SEGMENTS, seeds=SEEDS)
    results: dict = {}

    print("== 1. routing gate-strength trajectory (seed 0, 80 steps) ==")
    results["gate_trace"] = _gate_strength_trace(base)
    print(results["gate_trace"])

    print("== 2. theta divergence: fw vs null (10 paired episodes, seed 0) ==")
    results["theta_divergence"] = _theta_divergence(base)
    print(results["theta_divergence"])

    print("== 3. null at reduced lr — the effective-lr control ==")
    results["lr_control"] = {}
    for lr in LR_GRID:
        config = TrialConfig(segments=SEGMENTS, seeds=SEEDS, lr=lr)
        retained = _walk(
            "digital/feedforward/instantaneous/null/gradient/euclidean", config
        )
        results["lr_control"][f"null_lr{lr}"] = {
            "retention_mean": float(np.mean(retained)),
            "retention_sd": float(np.std(retained, ddof=1)),
            "per_seed": [round(r, 3) for r in retained],
        }
        print(f"null lr={lr}: {results['lr_control'][f'null_lr{lr}']}")

    routing = TrialConfig(segments=SEGMENTS, seeds=SEEDS)
    retained = _walk(
        "digital/feedforward/instantaneous/routing/gradient/euclidean", routing
    )
    results["routing_reference"] = {
        "retention_mean": float(np.mean(retained)),
        "per_seed": [round(r, 3) for r in retained],
    }
    print(f"routing lr=0.03: {results['routing_reference']}")

    out = Path("benchmark_results/p_axis_mechanism_audit.json")
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
