"""D3 — The M-axis swap matters, seen.

Null vs RoutingPlasticity walk the same segmented switching stream
(A→B, segment-keyed stationary teachers): θ accumulates, ψ adapts. The
runner watches routing visibly retain what null forgets: after segment B,
routing holds segment-A accuracy where null has collapsed toward chance.

Demonstrated regime (pinned 2026-09-02, 10-seed calibration): the registered
A40/B40 schedule at toy dims (8/8, hidden 16, lr 0.03) — routing retained
0.302 vs null 0.195 (routing ≥ null at 9/10 seeds), with routing mastering
A slower than null in every seed. The mastery precondition is asserted
BEFORE reading retention: below mastery the comparison is unreadable (at
A20/B20 the effect reverses — see Watch, TODO10).
"""

import numpy as np

from computronium.experiments.joint.forgetting_trial import (
    PROBED_SEGMENT,
    TrialConfig,
    _arm_coordinate,
    _walk_arm,
)

ARMS = ("null", "routing")
SEEDS = tuple(range(10))
CHANCE = 1 / 8


def test_demo_swap_plasticity(emit_run_record) -> None:
    config = TrialConfig(seeds=SEEDS, device="cpu")
    arms = {
        label: _walk_arm(label, _arm_coordinate(label), config)[0] for label in ARMS
    }
    record: dict = {
        "segments": [["A", 40], ["B", 40]],
        "seeds": list(SEEDS),
        "chance": CHANCE,
        "probed_segment": PROBED_SEGMENT,
        "arms": {
            label: {
                "a_mastery": list(arm.a_mastery),
                "a_retained": list(arm.a_retained),
            }
            for label, arm in arms.items()
        },
    }

    # Mastery precondition (read before retention): null is competent on A
    # at the switch, and routing masters A slower in every seed.
    assert min(arms["null"].a_mastery) >= 0.45
    assert all(
        routing < null
        for routing, null in zip(
            arms["routing"].a_mastery, arms["null"].a_mastery, strict=True
        )
    )

    # The visible effect: routing retains what null forgets.
    gap = float(np.mean(arms["routing"].a_retained)) - float(
        np.mean(arms["null"].a_retained)
    )
    record["mean_retained_gap"] = gap
    wins = int(
        np.sum(
            np.array(arms["routing"].a_retained) >= np.array(arms["null"].a_retained)
        )
    )
    record["seeds_routing_retains"] = wins
    assert gap >= 0.05, "routing's retention advantage must be visible in means"
    assert wins >= 8, "routing must retain at least as well as null per-seed"

    emit_run_record("D3", "swap_plasticity", record)
