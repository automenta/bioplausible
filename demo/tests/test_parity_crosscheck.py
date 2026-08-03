"""Sprint 3.7 — demo one-click parity is provably CLI-consistent.

Both the NiceGUI dashboard (via ``charts.parity_gap`` on two ``DemoPanel``s
trained headlessly) and the ``biopl-parity`` CLI (``run_parity``) express the
parity gap as ``(val_acc_B - val_acc_A) * 100`` pp and use the same train-first
accuracy rule. This test locks that the two surfaces agree under an identical
seed/configuration, so a number shown in the UI is guaranteed to match what
``biopl-parity --json`` would report for the same run.
"""

import pytest

from bioplausible.cli.parity import run_parity as cli_run_parity

from charts import parity_gap
from runner import DemoPanel, default_trainer_config, run_headless

TASK = "digits"
EPOCHS = 1
LR = 0.001
HIDDEN = 16
SEED = 7


def _train_panel(model: str, seed: int) -> DemoPanel:
    cfg = default_trainer_config(
        model=model, task=TASK, epochs=EPOCHS, lr=LR, hidden_dim=HIDDEN
    )
    panel = DemoPanel(trainer_config=cfg, seed=seed)
    run_headless(panel)
    assert panel.error is None, f"{model} failed: {panel.error}"
    return panel


def test_demo_gap_matches_cli_gap_pp() -> None:
    """The demo two-panel gap equals the CLI gap under one shared seed."""
    model_a, model_b = "equitile", "backprop_mlp"
    a, b = _train_panel(model_a, SEED), _train_panel(model_b, SEED)
    demo_gap = parity_gap(a, b)
    assert demo_gap is not None

    cli = cli_run_parity(
        model_a, model_b, TASK, EPOCHS, LR, HIDDEN, SEED
    )
    assert demo_gap == pytest.approx(cli["gap_pp"], abs=1e-6)


def test_demo_gap_is_deterministic_under_seed() -> None:
    """Re-running both panels at the same seed reproduces the same gap."""
    def _gap() -> float | None:
        a = _train_panel("equitile", SEED)
        b = _train_panel("backprop_mlp", SEED)
        return parity_gap(a, b)

    assert _gap() == pytest.approx(_gap(), abs=1e-6)
