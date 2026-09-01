"""Weight-viz transforms tests (Sprint 3.5) — browser-free."""

import torch
from runner import DemoPanel, default_trainer_config
from weight_viz import (
    align_length,
    diff_frame,
    matrix_frame,
    weight_layers,
)


def _fake_panel(shape=(4, 3), n=5) -> DemoPanel:
    p = DemoPanel(trainer_config=default_trainer_config(model="eqprop_mlp"))
    p.weight_history["in.weight"] = [
        torch.arange(n * 4 * 3, dtype=torch.float32).view(n, 4, 3)[i].contiguous()
        for i in range(n)
    ]
    return p


class TestWeightViz:
    def test_weight_layers_names(self):
        p = _fake_panel()
        assert weight_layers(p) == ["in.weight"]

    def test_matrix_frame_normalizes_to_unit_range(self):
        p = _fake_panel()
        f = matrix_frame(p, "in.weight", 2)
        assert f is not None
        assert f.rows == 4 and f.cols == 3
        assert min(f.values) == 0.0
        assert max(f.values) == 1.0
        assert len(f.as_grid(4, 3)) == 4

    def test_matrix_frame_clamps_out_of_range(self):
        p = _fake_panel(n=5)
        assert matrix_frame(p, "in.weight", 99).frame == 4

    def test_missing_layer_returns_none(self):
        assert matrix_frame(_fake_panel(), "nope", 0) is None

    def test_diff_frame(self):
        a = _fake_panel()
        b = _fake_panel()
        d = diff_frame(a, b, "in.weight", 1)
        assert d is not None
        assert d.rows == 4 and d.cols == 3
        # a - b == 0 for identical histories -> all-zero (span falls back) frame
        assert all(v == 0.0 for v in d.values)

    def test_align_length_min(self):
        a = _fake_panel(n=5)
        b = _fake_panel(n=3)
        assert align_length(a, b, "in.weight") == 3

    def test_no_panels_empty_history(self):
        p = DemoPanel(trainer_config=default_trainer_config())
        assert weight_layers(p) == []
        assert matrix_frame(p, "x", 0) is None
