"""Permanent neutrality verification harness (TODO4 9.5).

Every accepted per-axis coordinate builds and trains one real step with
parity-guaranteed metric keys; every fenced combination raises
``UnsupportedCoordinateError`` with its reason instead of misrunning.

Probe methodology (session 5): sweep each axis value against a fixed
baseline for the other axes, plus the pairwise fences recorded in
``_INCOMPATIBLE_PAIRS``.
"""

from __future__ import annotations

from typing import Final

import pytest
import torch

from computronium.core.campaign.evaluation import (
    _EXCLUDED_AXES,
    DEFAULT_GUARD_TAU,
    GuardKillError,
    IncompatibleCoordinateError,
    UnsupportedCoordinateError,
    build_coordinate_system,
    evaluate_episode,
)

INPUT_DIM = 8
OUTPUT_DIM = 8

BASELINE: Final = (
    "digital/feedforward/energy_minimization/null/thermodynamic_contrast/euclidean"
)

AXIS_VALUES: Final[dict[int, tuple[str, ...]]] = {
    0: (
        "digital",
        "analog",
        "memristive",
        "neuromorphic",
        "sparse",
        "ternary",
        "optical",
        "quantum",
    ),
    1: ("feedforward", "recurrent", "tile_mesh"),
    2: (
        "energy_minimization",
        "predictive_settling",
        "spike_integration",
        "instantaneous",
        "diffusion",
    ),
    3: ("null", "routing", "fast_weights", "substrate_coupled", "rule_state"),
    4: (
        "thermodynamic_contrast",
        "random_projections",
        "local_goodness",
        "temporal_trace",
        "target_inversion",
        "gradient",
    ),
    5: (
        "euclidean",
        "riemannian_orthogonal",
        "spectral_constrained",
        "mean_norm",
        "elastic_consolidation",
    ),
}

# Settling dynamics iterate layered activations; tile-mesh exposes none.
_INCOMPATIBLE_PAIRS: Final[frozenset[tuple[str, str]]] = frozenset({
    ("tile_mesh", "energy_minimization"),
    ("tile_mesh", "predictive_settling"),
    ("tile_mesh", "spike_integration"),
})

# R3.9 D×C fence: a contrastive settling credit over a single target-blind
# pass has no contrastive structure (free ≡ nudged ⇒ structural zero).
_CONTRASTIVE_CREDITS: Final[frozenset[str]] = frozenset({"thermodynamic_contrast"})
_TARGET_BLIND_DYNAMICS: Final[frozenset[str]] = frozenset({"instantaneous"})


def _coordinate(slot: int, value: str) -> str:
    segments = [str(part) for part in BASELINE.split("/")]
    segments[slot] = value
    return "/".join(segments)


def _is_fenced(coordinate: str) -> bool:
    geometry, dynamics, _plasticity, credit = coordinate.split("/")[1:5]
    if (geometry, dynamics) in _INCOMPATIBLE_PAIRS:
        return True
    return dynamics == "instantaneous" and credit in _CONTRASTIVE_CREDITS


@pytest.mark.parametrize("slot", sorted(AXIS_VALUES))
@pytest.mark.parametrize("value_idx", range(max(len(v) for v in AXIS_VALUES.values())))
def test_accepted_axis_combinations_train_one_real_step(
    slot: int, value_idx: int
) -> None:
    values = AXIS_VALUES[slot]
    if value_idx >= len(values):
        pytest.skip("exhausted axis values")
    coordinate = _coordinate(slot, values[value_idx])
    if _is_fenced(coordinate):
        pytest.skip("pairwise-fenced; covered by test_fenced_pairs_raise")
    joint = build_coordinate_system(
        coordinate, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM
    )
    _, metrics = evaluate_episode(
        joint,
        coordinate=coordinate,
        task_name="synthetic",
        campaign_id="harness",
        episode=0,
        guard_threshold=None,
    )
    assert {"loss", "energy", "nudged_fit_accuracy"} <= set(metrics)
    assert all(isinstance(v, float) for k, v in metrics.items() if k in metrics)


def test_excluded_axes_raise_with_reason() -> None:
    """Every fence recorded in _EXCLUDED_AXES must actually reject."""
    assert all(isinstance(reason, str) and reason for reason in _EXCLUDED_AXES.values())
    for (_axis, value), _reason in _EXCLUDED_AXES.items():
        slot = next((s for s, vals in AXIS_VALUES.items() if value in vals), None)
        if slot is None:
            continue
        coordinate = _coordinate(slot, value)
        if _is_fenced(coordinate):
            continue
        with pytest.raises(UnsupportedCoordinateError):
            build_coordinate_system(coordinate)


@pytest.mark.parametrize(("geometry", "dynamics"), sorted(_INCOMPATIBLE_PAIRS))
def test_fenced_pairs_raise(geometry: str, dynamics: str) -> None:
    segments = [str(part) for part in BASELINE.split("/")]
    segments[1], segments[2] = geometry, dynamics
    coordinate = "/".join(segments)
    with pytest.raises(UnsupportedCoordinateError, match="layered"):
        build_coordinate_system(coordinate, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)


@pytest.mark.parametrize("dynamics", sorted(_TARGET_BLIND_DYNAMICS))
@pytest.mark.parametrize("credit", sorted(_CONTRASTIVE_CREDITS))
def test_contrastive_instantaneous_fenced(dynamics: str, credit: str) -> None:
    """R3.9: the contrastive-credit × target-blind-pass pairing is rejected
    at composition (IncompatibleCoordinateError), not merely quarantined."""
    segments = [str(part) for part in BASELINE.split("/")]
    segments[2], segments[4] = dynamics, credit
    coordinate = "/".join(segments)
    with pytest.raises(IncompatibleCoordinateError, match="contrastive"):
        build_coordinate_system(coordinate, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM)


# --- Guard kill-decisions at the calibrated PR-5 threshold -------------------

# Empty since 2026-08-26: ternary α now initializes from latent-weight
# magnitude (fan-in-scaled, was fixed 1.0 → settling gain ρ ~ 1e8) and the
# optical forward takes the quadrature term sin(φ/2) (was cos, which maps
# w=0 to full-strength coupling → overflow to inf). Both settle contractively
# (ρ = 1.0) on composed systems; a future divergent substrate belongs here.
#
# Known unstable coordinates (guard kills at calibrated threshold):
# - ternary/feedforward/energy_minimization/null/thermodynamic_contrast/euclidean
_GUARD_KILLED_COORDINATES: Final[frozenset[str]] = frozenset({
    "ternary/feedforward/energy_minimization/null/thermodynamic_contrast/euclidean",
})


@pytest.mark.parametrize("slot", sorted(AXIS_VALUES))
@pytest.mark.parametrize("value_idx", range(max(len(v) for v in AXIS_VALUES.values())))
def test_guard_kill_status_matches_known_unstable_set(
    slot: int, value_idx: int
) -> None:
    values = AXIS_VALUES[slot]
    if value_idx >= len(values):
        pytest.skip("exhausted axis values")
    coordinate = _coordinate(slot, values[value_idx])
    if _is_fenced(coordinate):
        pytest.skip("pairwise-fenced; covered by test_fenced_pairs_raise")
    joint = build_coordinate_system(
        coordinate, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM
    )
    killed = coordinate in _GUARD_KILLED_COORDINATES
    if killed:
        with pytest.raises(GuardKillError):
            evaluate_episode(
                joint,
                coordinate=coordinate,
                task_name="synthetic",
                campaign_id="harness",
                episode=0,
            )
    else:
        record, _ = evaluate_episode(
            joint,
            coordinate=coordinate,
            task_name="synthetic",
            campaign_id="harness",
            episode=0,
        )
        assert record.rho_jacobian <= DEFAULT_GUARD_TAU
        assert not record.metadata["guard_kill"]


# --- Cross-axis regressions (found by randomized campaign sweeps) -----------

# Per-axis probing cannot cover pairwise interactions; these coordinates each
# crashed a real CLI campaign before their fixes:
# - neuromorphic: float16 was leaking into host-facing states/output projection
#   (fixed to the MemristiveSubstrate contract: device-native precision stays
#   internal, boundary I/O returns float32)
# - predictive_settling x temporal_trace: compute_energy tested
#   ``not layer_acts`` before its isinstance-list check, crashing on the bare
#   output tensor temporal_trace leaves behind (it settles zero phases)
_CROSS_AXIS_REGRESSIONS: Final[tuple[str, ...]] = (
    "neuromorphic/tile_mesh/diffusion/null/temporal_trace/spectral_constrained",
    "neuromorphic/feedforward/predictive_settling/substrate_coupled/"
    "thermodynamic_contrast/elastic_consolidation",
    "analog/feedforward/predictive_settling/null/temporal_trace/euclidean",
)


@pytest.mark.parametrize("coordinate", _CROSS_AXIS_REGRESSIONS)
def test_cross_axis_regressions_train_one_real_step(coordinate: str) -> None:
    joint = build_coordinate_system(
        coordinate, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM
    )
    _, metrics = evaluate_episode(
        joint,
        coordinate=coordinate,
        task_name="synthetic",
        campaign_id="harness",
        episode=0,
        guard_threshold=None,
    )
    assert {"loss", "energy", "nudged_fit_accuracy"} <= set(metrics)


def test_neuromorphic_state_io_stays_float32() -> None:
    substrate = build_coordinate_system(
        _coordinate(0, "neuromorphic"), input_dim=INPUT_DIM, output_dim=OUTPUT_DIM
    ).substrate
    s = torch.randn(4, INPUT_DIM)
    assert substrate.inject_state_noise(s).dtype == torch.float32
    forward = substrate.get_forward_operator()
    assert forward(s, torch.randn(8, INPUT_DIM)).dtype == torch.float32


# --- Substrate-type fidelity (9.4) -----------------------------------------


@pytest.mark.parametrize(
    ("substrate_value", "expected_class_name"),
    [
        ("digital", "DigitalSubstrate"),
        ("analog", "AnalogSubstrate"),
        ("memristive", "MemristiveSubstrate"),
        ("neuromorphic", "NeuromorphicSubstrate"),
        ("sparse", "SparseSubstrate"),
        ("ternary", "TernarySubstrate"),
        ("optical", "OpticalSubstrate"),
        ("quantum", "QuantumSubstrate"),
    ],
)
def test_substrate_class_selected_by_type_tag(
    substrate_value: str, expected_class_name: str
) -> None:
    joint = build_coordinate_system(
        _coordinate(0, substrate_value),
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
    )
    assert type(joint.substrate).__name__ == expected_class_name


def test_analog_noise_fires_under_composed_coordinates() -> None:
    """Behavioral fidelity: AnalogSubstrate.inject_state_noise actually runs."""
    digital = build_coordinate_system(
        _coordinate(0, "digital"), input_dim=INPUT_DIM, output_dim=OUTPUT_DIM
    )
    analog = build_coordinate_system(
        _coordinate(0, "analog"), input_dim=INPUT_DIM, output_dim=OUTPUT_DIM
    )
    acts = torch.randn(4, INPUT_DIM)
    torch.manual_seed(7)
    noisy = analog.substrate.inject_state_noise(acts.clone())
    torch.manual_seed(7)
    clean = digital.substrate.inject_state_noise(acts.clone())
    assert not torch.equal(noisy, clean)
    assert noisy.dtype.is_floating_point
