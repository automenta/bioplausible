"""Chart transform tests (Sprint 3.4)."""

import pytest

pytest.importorskip("charts", reason="demo-local module; run from the demo venv")
pytest.importorskip("runner", reason="demo-local module; run from the demo venv")

from charts import parity_gap, rolling_mean
from runner import DemoPanel, default_trainer_config


class TestRollingMean:
    def test_none_until_window_fills(self):
        out = rolling_mean([1.0, 2.0, 3.0, 4.0], window=2)
        assert out[0] is None
        assert out[1] == 1.5
        assert out[-1] == 3.5


class TestParityGap:
    def test_none_when_not_finished(self):
        a = DemoPanel(trainer_config=_cfg(), finished=False)
        b = DemoPanel(trainer_config=_cfg(), finished=False)
        assert parity_gap(a, b) is None

    def test_gap_computation(self):
        a = DemoPanel(trainer_config=_cfg(), finished=True, accuracies=[0.8])
        b = DemoPanel(trainer_config=_cfg(), finished=True, accuracies=[0.9])
        assert parity_gap(a, b) == 10.0


def _cfg():
    return default_trainer_config(model="backprop_mlp", epochs=1)
