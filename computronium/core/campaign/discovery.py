"""R5b-D discovery locks: pre-registered claims enforced as code.

A campaign delta becomes a discovery only when it survives four locks:
winner replication across seeds and families (①), attribution rank
stability (②), replay from the episode seed (③), and implementation
fidelity — the winner's behavior must be explained by its axes, not by a
defect (④). Locks are spec-driven: :class:`DiscoverySpec` states the
claimed axis transition and threshold, or the registered null result. A
verdict that flips after an instrument repair forces re-evaluation instead
of silently preserving a stale claim; per TODO8 policy a failed fidelity
check is *inconclusive*, never a refutation of the hypothesis.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypedDict

from computronium.analysis.counterfactual import attribute_axis_effects

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import torch

    from computronium.core.campaign.fidelity import CoordinateFidelity
    from computronium.core.campaign.frontier_record import FrontierRecord

Stratum = tuple[int, str]  # (seed, task family)
Transition = tuple[str, str, str]  # (axis, from_value, to_value)


class SpecTransitionError(ValueError):
    """Discovery spec asserts an effect without naming its transition."""

    def __init__(self, spec_name: str) -> None:
        super().__init__(f"spec {spec_name!r} does not name a transition")
        self.spec_name = spec_name


class BuildKwargs(TypedDict, total=False):
    """Coordinate composition overrides (the shape triple)."""

    input_dim: int
    output_dim: int
    hidden_dims: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DiscoverySpec:
    """A registered discovery claim (or the registered null result).

    ``claim="effect"`` asserts the ``from_value -> to_value`` transition on
    ``axis`` moves ``metric`` by at least ``min_delta`` in every (seed,
    family) stratum and ranks first in attribution. ``claim="null"``
    asserts no transition is stratified-stable — the honest state of a
    flywheel that has not yet earned a claim.
    """

    name: str
    metric: str = "task_accuracy"
    axis: str | None = None
    from_value: str | None = None
    to_value: str | None = None
    min_delta: float = 0.0
    task_name: str = "synthetic"
    claim: Literal["effect", "null"] = "effect"

    @property
    def transition(self) -> Transition:
        if self.axis is None or self.from_value is None or self.to_value is None:
            raise SpecTransitionError(self.name)
        return (self.axis, self.from_value, self.to_value)


@dataclass(frozen=True, slots=True)
class LockVerdict:
    """Outcome of one discovery lock over a record set."""

    lock: str
    holds: bool
    detail: str


def _strata(records: Sequence[FrontierRecord]) -> dict[Stratum, list[FrontierRecord]]:
    groups: dict[Stratum, list[FrontierRecord]] = defaultdict(list)
    for record in records:
        groups[record.seed, record.task_name].append(record)
    return dict(groups)


def _canonical(transition: Transition) -> Transition:
    """Direction-independent key for a transition.

    Minimal pairs occur in both orders (records from different seeds/families
    carry the same axis swap), so attribution splits one logical effect into
    direction-split keys; every lock compares transitions canonically.
    """
    axis, src, dst = transition
    return (axis, src, dst) if src <= dst else (axis, dst, src)


def _oriented_delta(attributions: Sequence, spec: DiscoverySpec) -> float | None:
    """Merged mean delta of the claimed transition in the spec's orientation.

    Both observed directions contribute (the reversed one negated), so the
    result is the pooled effect of the claimed swap regardless of which
    order the record stream happened to produce.
    """
    axis, src, dst = spec.transition
    total = 0.0
    n = 0
    for attribution in attributions:
        if attribution.axis != axis:
            continue
        if (attribution.from_value, attribution.to_value) == (src, dst):
            total += attribution.mean_delta
            n += 1
        elif (attribution.from_value, attribution.to_value) == (dst, src):
            total -= attribution.mean_delta
            n += 1
    return total / n if n else None


@dataclass(frozen=True, slots=True)
class CanonicalAttribution:
    """One logical axis effect after direction merging.

    Minimal pairs occur in both record orders (cross-seed/cross-family), so
    raw attribution fragments one effect into direction-split keys; reports
    and locks rank the canonical rows, never the raw halves.
    """

    axis: str
    from_value: str
    to_value: str
    mean_delta: float
    n_pairs: int


def canonical_attributions(attributions: Sequence) -> list[CanonicalAttribution]:
    """Merge direction-split attributions into canonical signed rows.

    The reversed half of each minimal pair contributes its negated delta, so
    the merged mean is the pooled effect of the logical swap regardless of
    which order the record stream happened to emit. Rows are sorted by
    absolute mean delta (most influential first); equal magnitudes keep
    first-encounter order.
    """
    grouped: dict[Transition, list[tuple[float, int]]] = defaultdict(list)
    for attribution in attributions:
        orientation = (
            attribution.axis,
            attribution.from_value,
            attribution.to_value,
        )
        key = _canonical(orientation)
        grouped[key].append((
            -attribution.mean_delta if key != orientation else attribution.mean_delta,
            attribution.n_pairs,
        ))
    rows = [
        CanonicalAttribution(
            axis=key[0],
            from_value=key[1],
            to_value=key[2],
            mean_delta=sum(delta for delta, _ in members) / len(members),
            n_pairs=sum(count for _, count in members),
        )
        for key, members in grouped.items()
    ]
    return sorted(rows, key=lambda row: abs(row.mean_delta), reverse=True)


def _merged_attributions(
    attributions: Sequence,
) -> list[tuple[Transition, float]]:
    """Collapse direction-split attributions into canonical signed effects."""
    return [
        ((row.axis, row.from_value, row.to_value), row.mean_delta)
        for row in canonical_attributions(attributions)
    ]


def _top_transition(attributions: Sequence) -> Transition | None:
    if not attributions:
        return None
    merged = _merged_attributions(attributions)
    return max(merged, key=lambda item: abs(item[1]))[0]


def _null_replication_verdict(
    strata: dict[Stratum, list[FrontierRecord]], spec: DiscoverySpec
) -> LockVerdict:
    """Null lock ①: no transition may be stratified-stable.

    Stable = present in every stratum, consistent sign, |delta| ≥
    min_delta. The moment one becomes stable, the null claim fails and a
    pre-registered effect claim must take its place.
    """
    per_stratum: dict[Stratum, dict[Transition, float]] = {
        stratum: dict(
            _merged_attributions(attribute_axis_effects(group, metric=spec.metric))
        )
        for stratum, group in strata.items()
    }
    shared: set[Transition] = set.intersection(
        *(set(deltas) for deltas in per_stratum.values())
    )
    stable: list[Transition] = []
    for transition in shared:
        values = [deltas[transition] for deltas in per_stratum.values()]
        same_sign = all(v > 0 for v in values) or all(v < 0 for v in values)
        if same_sign and min(map(abs, values)) >= spec.min_delta:
            stable.append(transition)
    if stable:
        detail = "; ".join(f"{t[0]}: {t[1]}→{t[2]}" for t in sorted(stable))
        return LockVerdict(
            "winner-replication", False, f"stratified-stable effect: {detail}"
        )
    return LockVerdict(
        "winner-replication",
        True,
        f"no stratified-stable transition over {len(strata)} strata "
        f"(threshold {spec.min_delta})",
    )


def verify_winner_replication(
    records: Sequence[FrontierRecord], spec: DiscoverySpec
) -> LockVerdict:
    """Lock ①: the claimed gap must replicate in every (seed, family) stratum.

    For a null claim the lock inverts (see ``_null_replication_verdict``).
    """
    strata = _strata(records)
    if not strata:
        return LockVerdict("winner-replication", False, "no records")

    if spec.claim == "null":
        return _null_replication_verdict(strata, spec)

    transitions = {
        stratum: _oriented_delta(
            attribute_axis_effects(group, metric=spec.metric), spec
        )
        for stratum, group in strata.items()
    }
    missing = [stratum for stratum, delta in transitions.items() if delta is None]
    if missing:
        return LockVerdict(
            "winner-replication",
            False,
            f"transition absent in strata {sorted(missing)}",
        )
    failed = {
        stratum: delta
        for stratum, delta in transitions.items()
        if delta is not None and delta < spec.min_delta
    }
    if failed:
        detail = ", ".join(f"{s}: {d:+.4f}" for s, d in sorted(failed.items()))
        return LockVerdict(
            "winner-replication",
            False,
            f"below min_delta={spec.min_delta} in: {detail}",
        )
    detail = ", ".join(f"{s}: {d:+.4f}" for s, d in sorted(transitions.items()))
    return LockVerdict("winner-replication", True, f"replicated: {detail}")


def _null_rank_verdict(
    pooled_top: Transition,
    pooled_top_delta: float,
    stratum_tops: dict[Stratum, Transition | None],
    spec: DiscoverySpec,
) -> LockVerdict:
    """Null lock ②: the pooled top must not be a stratified-stable effect.

    Agreement across all strata at or above the claimable threshold forces
    an effect claim to be registered; anything else is the honest no-effect
    state.
    """
    agreeing = [s for s, top in stratum_tops.items() if top == pooled_top]
    unanimous = len(agreeing) == len(stratum_tops)
    if unanimous and abs(pooled_top_delta) >= spec.min_delta:
        return LockVerdict(
            "attribution-rank",
            False,
            f"top transition {pooled_top} ({pooled_top_delta:+.4f}) stable "
            "across all strata — register an effect claim",
        )
    if unanimous:
        return LockVerdict(
            "attribution-rank",
            True,
            f"pooled top {pooled_top} ({pooled_top_delta:+.4f}) stable but "
            f"below threshold {spec.min_delta} — no claimable effect",
        )
    return LockVerdict(
        "attribution-rank",
        True,
        f"pooled top {pooled_top} not stratified-stable "
        f"({len(agreeing)}/{len(stratum_tops)} strata agree) — no claimable "
        "effect",
    )


def verify_attribution_rank(
    records: Sequence[FrontierRecord],
    spec: DiscoverySpec,
    *,
    manifest: Mapping[str, CoordinateFidelity] | None = None,
) -> LockVerdict:
    """Lock ②: attribution ranks the discovered transition first, stably.

    Pooled (defect-filtered when ``manifest`` is given) attribution must
    place the claimed transition first by |mean delta|, and every (seed,
    family) stratum must agree. Null claims hold while strata disagree on
    the top transition — the registered R5b-B/C state.
    """
    passing = [
        r
        for r in records
        if manifest is None
        or ((verdict := manifest.get(r.coordinate)) is not None and verdict.passed)
    ]
    merged_pooled = _merged_attributions(
        attribute_axis_effects(passing, metric=spec.metric)
    )
    if not merged_pooled:
        return LockVerdict("attribution-rank", False, "no minimal pairs")
    pooled_top, pooled_top_delta = max(merged_pooled, key=lambda item: abs(item[1]))

    stratum_tops = {
        stratum: _top_transition(attribute_axis_effects(group, metric=spec.metric))
        for stratum, group in _strata(passing).items()
    }

    if spec.claim == "null":
        return _null_rank_verdict(pooled_top, pooled_top_delta, stratum_tops, spec)

    claimed = _canonical(spec.transition)
    if pooled_top != claimed:
        return LockVerdict(
            "attribution-rank",
            False,
            f"pooled top is {pooled_top}, not the claimed {claimed}",
        )
    disagreeing = {s: top for s, top in stratum_tops.items() if top != claimed}
    if disagreeing:
        return LockVerdict(
            "attribution-rank",
            False,
            f"strata not ranking the claim first: {sorted(disagreeing)}",
        )
    return LockVerdict(
        "attribution-rank",
        True,
        f"{spec.transition} ranks first pooled and in all {len(stratum_tops)} strata",
    )


def verify_fidelity_standing(
    manifest: Mapping[str, CoordinateFidelity], coordinates: Sequence[str]
) -> LockVerdict:
    """Lock ④: winner coordinates must pass implementation fidelity.

    A "winner" whose behavior is better explained by a defect (leaked
    metric, inert instrument, unwired nudge) fails its fidelity probe; this
    lock fails the discovery with it instead of preserving a false result.
    """
    failing = [
        coordinate
        for coordinate in coordinates
        if (verdict := manifest.get(coordinate)) is None or not verdict.passed
    ]
    if failing:
        return LockVerdict(
            "fidelity-standing",
            False,
            f"coordinates fail (or lack) fidelity: {sorted(failing)}",
        )
    return LockVerdict(
        "fidelity-standing",
        True,
        f"{len(coordinates)} coordinates pass implementation fidelity",
    )


def verify_replay(
    coordinate: str,
    *,
    seed: int,
    campaign_id: str,
    iteration: int,
    task_name: str = "synthetic",
    device: str | torch.device = "cuda",
    tolerance: float = 1e-6,
    build_kwargs: BuildKwargs | None = None,
) -> LockVerdict:
    """Lock ③: same (seed, campaign_id, iteration) re-derives the episode.

    Two independent constructions from the same episode seed must agree
    within tight tolerance — on GPU per the R5b-D spec; bitwise equality is
    explicitly *not* required (TODO8 determinism policy: tolerance +
    environment-locked manifests, bitwise is an opt-in extra).
    """
    import torch

    from computronium.core.campaign.evaluation import (
        build_coordinate_system,
        episode_seed,
        evaluate_episode,
    )

    target = torch.device(device) if isinstance(device, str) else device
    runs: list[tuple[float, float]] = []
    for _ in range(2):
        torch.manual_seed(episode_seed(seed, campaign_id, iteration, coordinate))
        joint = build_coordinate_system(
            coordinate, device=target, **(build_kwargs or {})
        )
        record, _ = evaluate_episode(
            joint,
            coordinate=coordinate,
            task_name=task_name,
            campaign_id=campaign_id,
            episode=iteration,
            guard_threshold=None,
            seed=seed,
        )
        runs.append((record.task_loss, record.task_accuracy))
    (loss_a, acc_a), (loss_b, acc_b) = runs
    ok = (
        abs(loss_a - loss_b) <= tolerance * max(1.0, abs(loss_a))
        and abs(acc_a - acc_b) <= tolerance
    )
    detail = (
        f"loss {loss_a:.6g} vs {loss_b:.6g}, acc {acc_a:.6g} vs {acc_b:.6g} "
        f"on {target} (tolerance {tolerance})"
    )
    return LockVerdict("replay", ok, detail)
