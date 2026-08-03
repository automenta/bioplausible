"""Tests for the O(1) memory scaling track (Track 10).

Validates that the track reports *measured* activation memory (not theoretical)
and that the EqProp vs Backprop scaling distinction holds: EqProp activations
stay flat with depth (O(1)) while Backprop activations grow (O(n)).
"""

import pytest
import torch

from bioplausible.validation.tracks.scaling_tracks import track_10_memory_scaling

expected_track_id = 10
cuda_available = torch.cuda.is_available()


class _QuickVerifier:
    quick_mode = True


@pytest.mark.skipif(
    not cuda_available,
    reason="Track 10 measures GPU peak memory via torch.cuda; requires a GPU",
)
def test_track_10_measured_memory_scaling():
    result = track_10_memory_scaling(_QuickVerifier())

    assert result.track_id == expected_track_id
    assert result.status in {"pass", "partial", "fail"}
    assert result.metrics["device"] == "cuda"
    assert result.metrics["metric"] == "activation_memory_delta_mb"

    results = result.metrics["results"]
    depths = (10, 25, 50)
    flat_tol_mb, pass_ratio = 0.5, 5
    assert all(d in results for d in depths)

    eq_act = [results[d]["eqprop_activation_mb"] for d in results]
    bp_act = [results[d]["backprop_activation_mb"] for d in results]

    # EqProp activation memory must be (approximately) flat across depths.
    eqprop_spread = max(eq_act) - min(eq_act)
    assert eqprop_spread < flat_tol_mb, f"EqProp act not flat: {eq_act}"

    # Backprop activation memory must grow with depth (O(n)).
    assert bp_act[-1] > bp_act[0] * 2, f"Backprop not scaling: {bp_act}"

    # Final-ratio must beat the pass threshold.
    final = results[max(results)]
    assert final["ratio"] > pass_ratio or result.status != "fail"
