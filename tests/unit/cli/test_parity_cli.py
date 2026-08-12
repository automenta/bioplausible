"""Sprint 3.7 ``biopl-parity`` CLI tests + Sprint 0.5 lazy-import regression guards.

Covers: (a) the parity gap arithmetic matches the demo's ``(B - A) * 100`` pp
definition, (b) task validation, and (c) the circular-import / lazy ``cli``
package-init fixes so the console scripts import in any order.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from bioplausible.cli.parity import _per_epoch_accuracy, run_parity

ROOT = Path(__file__).resolve().parents[3]


class _FakeMetrics:
    def __init__(self, train=None, val=None):
        self.train_accuracy = train
        self.val_accuracy = val


def test_per_epoch_accuracy_prefers_train():
    h = [_FakeMetrics(train=0.91, val=0.88), _FakeMetrics(train=0.95, val=0.93)]
    assert _per_epoch_accuracy(h) == [0.91, 0.95]


def test_per_epoch_accuracy_falls_back_to_val():
    h = [_FakeMetrics(train=None, val=0.77)]
    assert _per_epoch_accuracy(h) == [0.77]


def test_task_validation_rejects_unknown_tasks():
    with pytest.raises(ValueError):
        run_parity("backprop_mlp", "backprop_mlp", "not_a_task", 1, 0.01, 32, 0)


@pytest.mark.parametrize(
    "name,module_stmt",
    [
        (
            "cli_parity_imports_standalone",
            "from bioplausible.cli.parity import main",
        ),
        (
            "cli_main_resolves_after_lazy_import",
            "import bioplausible.cli as c; _ = c.main",
        ),
        (
            "execution_guards_imports_standalone",
            "from bioplausible.execution._guards "
            "import create_constrained_optuna_config",
        ),
    ],
)
def test_lazy_imports_break_circular_dependency(name, module_stmt):
    """The Sprint 0.5 lazy package-init fixes work in a fresh interpreter."""
    out = subprocess.run(
        [sys.executable, "-c", module_stmt],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert out.returncode == 0, out.stderr


def test_cli_parity_end_to_end():
    """biopl-parity trains two models and reports a valid parity gap (JSON)."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "bioplausible.cli.parity",
            "--config-a",
            "tile_pc",
            "--config-b",
            "backprop_mlp",
            "--task",
            "digits",
            "--epochs",
            "1",
            "--lr",
            "0.01",
            "--hidden",
            "32",
            "--seed",
            "0",
            "--json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stderr
    import json

    report = json.loads(result.stdout)
    assert report["config_a"] == "tile_pc"
    assert report["config_b"] == "backprop_mlp"
    # gap_pp == (B - A) * 100, mirroring demo/charts.py parity_gap (within
    # display-rounding tolerance: gap is computed pre-rounding of accuracies).
    assert report["gap_pp"] == pytest.approx(
        (report["accuracy_b"] - report["accuracy_a"]) * 100, abs=0.011
    )
