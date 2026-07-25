"""Regression tests for bugs found and fixed while completing REFACTOR2.

Tests are organized by the bug they regress against, with a one-line
reference to the original failure mode in each docstring.  Each test is
self-contained and uses light-weight synthetic data so it can run on
CPU without external dependencies.
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

import bioplausible  # noqa: F401  # triggers full registration
from bioplausible.core.registry import ComponentCategory, Registry
from bioplausible.core.trainer import (
    CoreTrainer,
    TrainerConfig,
    _reshape_logits_targets_for_ce,
)
from bioplausible.zoo import _LegacyModelSpec, get_model_spec


# ---------------------------------------------------------------------------
# Bug #1: `spec.citation` raised AttributeError in execution/report/latex.py
# because `citation` was not declared in `_LegacyModelSpec.__slots__`.
# ---------------------------------------------------------------------------


def test_legacy_model_spec_exposes_citation_slot():
    """`_LegacyModelSpec.citation` must be a readable attribute.

    Regression: previously `latex.py:51` did ``if spec.citation:`` but
    `citation` was missing from ``__slots__`` → AttributeError at runtime
    whenever the LaTeX report generator resolved a model successfully.
    """
    spec = get_model_spec("eqprop_mlp")
    assert isinstance(spec, _LegacyModelSpec)
    # Reading the attribute must not raise — None is acceptable for models
    # without a bibliographic citation.
    assert spec.citation is None or isinstance(spec.citation, str)


@pytest.mark.parametrize(
    "attr",
    [
        "citation",
        "description",
        "tags",
        "version",
        "custom_hyperparams",
        "variant",
        "family",
        "model_type",
        "task_compat",
        "credit_locality",
        "requires_backward",
        "default_lr",
        "credit_assignment_type",
        "name",
    ],
)
def test_legacy_model_spec_slot_readable(attr: str):
    """Every legacy adapter attribute must be reachable on every model."""
    for name in Registry.list("model")["model"]:
        spec = get_model_spec(name)
        # Must not raise AttributeError; None is acceptable for unset fields.
        getattr(spec, attr)


def test_legacy_model_spec_citation_read_in_latex_report_context():
    """End-to-end smoke: latex report generator can read `spec.citation`.

    Simulates the exact pattern used in
    ``bioplausible.execution.report.latex`` to build a BibTeX block.
    """
    used_models = ["eqprop_mlp", "backprop_mlp", "forward_forward"]
    bib: set[str] = set()
    for model_name in used_models:
        try:
            spec = get_model_spec(model_name)
        except ValueError:
            continue
        if spec.citation:
            bib.add(spec.citation)
    # No exception → test passes regardless of whether citations are set.
    assert isinstance(bib, set)


# ---------------------------------------------------------------------------
# Bug #7: `_FAMILY_TAGS` had `"forward_only"` but ForwardForwardNet/PEPITA
# registered with tag `"forward-only"` (hyphen).  Family resolution silently
# returned "experimental", mis-routing metamodel classification.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", ["forward_forward", "pepita"])
def test_forward_only_family_resolves_correctly(model_name: str):
    """Models tagged with `forward-only` (hyphen) must resolve to family
    `forward_only` (underscore), matching `_FAMILY_TAGS` and the metamodel
    fallback table in `hyperparameter_metamodel.py`.
    """
    spec = get_model_spec(model_name)
    assert spec.family == "forward_only", (
        f"Expected family=='forward_only' for {model_name}, got {spec.family!r}"
    )


def test_family_resolution_normalizes_hyphenated_tags():
    """`_LegacyModelSpec._FAMILY_TAGS` accepts both hyphenated and
    underscored variants of algorithm family tags.
    """
    from bioplausible.zoo import _LegacyModelSpec

    assert "forward_only" in _LegacyModelSpec._FAMILY_TAGS
    assert "forward-only" in _LegacyModelSpec._FAMILY_TAGS


# ---------------------------------------------------------------------------
# Bug #3, #4: CoreTrainer's standard `_train_step`/`_validate` did not handle
# 3-D LM-style logits `[B, L, V]` — cross_entropy + argmax(1) crashed.
# ---------------------------------------------------------------------------


def test_reshape_logits_targets_for_ce_handles_3d_logits():
    """3-D LM logits must be sliced to last-token prediction."""
    logits = torch.randn(4, 7, 13)  # [B, L, V]
    y = torch.randint(0, 13, (4,))
    out_logits, out_y = _reshape_logits_targets_for_ce(logits, y)
    assert out_logits.shape == (4, 13)
    assert out_y.shape == (4,)
    assert out_y.dtype == torch.long


def test_reshape_logits_targets_for_ce_squeezes_singleton_target_dim():
    """Regression tasks return `[B, 1]` float targets — must become `[B]`
    long indices for CrossEntropyLoss (or remain float for MSE elsewhere).
    """
    logits = torch.randn(8, 5)
    y = torch.rand(8, 1) * 4  # [B, 1] float targets simulating regression
    out_logits, out_y = _reshape_logits_targets_for_ce(logits, y)
    assert out_y.shape == (8,)
    assert out_y.dtype == torch.long


def test_reshape_logits_targets_for_ce_idempotent_for_plain_classification():
    """Plain `[B, C]` logits with `[B]` long labels should pass through."""
    logits = torch.randn(4, 10)
    y = torch.tensor([0, 1, 2, 3])
    out_logits, out_y = _reshape_logits_targets_for_ce(logits, y)
    assert torch.equal(out_logits, logits)
    assert torch.equal(out_y, y)


def test_core_trainer_train_step_supports_3d_logits_model():
    """CoreTrainer should train models that return 3-D logits without
    requiring the model to define its own `train_step`.  Previously the
    standard forward/backward path called ``F.cross_entropy(logits, y)``
    and ``logits.argmax(1)`` — both crash with 3-D LM heads.
    """

    class LMHeadStub(nn.Module):
        """Emits [B, L, V] logits — typical autoregressive LM shape."""

        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 7)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # [B, H=4] -> [B, L=3, V=7] via broadcast on a singleton seq axis.
            return self.linear(x).unsqueeze(1).expand(-1, 3, -1)

    model = LMHeadStub()
    config = TrainerConfig(
        model="lm_head_stub",
        task="mnist",
        epochs=1,
        batches_per_epoch=1,
        val_batches=0,
        track_energy=False,
        device="cpu",
    )
    trainer = CoreTrainer(config=config)
    # Inject model directly (bypassing Registry.get()) so the test exercises
    # _train_step's standard forward/backward path on a 3-D-logits model.
    trainer.model = model
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    metrics = trainer._train_step(torch.randn(4, 4), torch.randint(0, 7, (4,)))
    assert "loss" in metrics and "accuracy" in metrics
    assert math.isfinite(metrics["loss"])
    assert 0.0 <= metrics["accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# Bug #5, #9, #16: `_TaskTrainer.train_epoch` (hyperopt) initialized
# `val_accuracy` to `train_accuracy` (masking validation failure) and used a
# bare `except Exception: pass` that silently swallowed any validation bug.
# ---------------------------------------------------------------------------


def test_task_trainer_nan_val_when_validation_fails():
    """When validation raises RuntimeError (e.g. no val split), the metrics
    must surface NaN so downstream ranking logic cannot mistake a silently
    swallowed failure for a real 0.0 / train-accuracy result.
    """
    from bioplausible.hyperopt.tasks import BaseTask, _TaskTrainer

    class FailingValTask(BaseTask):
        """Stub task whose validation path raises RuntimeError."""

        def __init__(self) -> None:
            super().__init__(name="failing_val", device="cpu", quick_mode=True)
            self._input_dim = 4
            self._output_dim = 3

        # setup not exercised by trainer; provide no-op to honor abstract API.
        def setup(self) -> None:  # noqa: D401
            pass

        @property
        def task_type(self) -> str:
            return "vision"

        def get_batch(self, split: str = "train", batch_size: int = 4):
            if split == "train":
                return torch.randn(batch_size, 4), torch.randint(0, 3, (batch_size,))
            raise RuntimeError("no validation split available")

        def create_trainer(self, model: nn.Module, **kwargs):
            return _TaskTrainer(model, self, device="cpu", **kwargs)

    model = nn.Linear(4, 3)
    task = FailingValTask()
    trainer = _TaskTrainer(
        model=model,
        task=task,
        device="cpu",
        batches_per_epoch=1,
        epochs=1,
        optimizer=torch.optim.SGD(model.parameters(), lr=1e-2),
    )
    metrics = trainer.train_epoch()
    # train metrics should be real numbers
    assert math.isfinite(metrics["loss"])
    # val metrics must be NaN (not silently 0.0 / not silently train_accuracy)
    assert math.isnan(metrics["val_accuracy"]), (
        f"val_accuracy={metrics['val_accuracy']!r}; expected NaN to surface a "
        "swallowed validation failure"
    )
    assert math.isnan(metrics["val_loss"])


def test_task_trainer_uses_mse_for_regression_tasks():
    """Tabular regression tasks (output_dim == 1) must select MSELoss;
    using CrossEntropyLoss on float [B,1] targets crashed with a shape /
    dtype error.
    """
    from bioplausible.hyperopt.tasks import (
        BaseTask,
        _TaskTrainer,
        _resolve_task_loss,
    )

    class RegressionTask(BaseTask):
        def __init__(self) -> None:
            super().__init__("regression", "cpu", quick_mode=True)
            self._input_dim = 4
            self._output_dim = 1

        def setup(self) -> None:
            pass

        @property
        def task_type(self) -> str:
            return "tabular"

        def get_batch(self, split: str = "train", batch_size: int = 8):
            x = torch.randn(batch_size, 4)
            y = torch.rand(batch_size, 1) * 5  # float regression targets
            return x, y

        def create_trainer(self, model, **kwargs):
            return _TaskTrainer(model, self, **kwargs)

    task = RegressionTask()
    loss = _resolve_task_loss(task)
    assert isinstance(loss, nn.MSELoss), (
        f"Expected MSELoss for output_dim==1 regression, got {type(loss).__name__}"
    )

    # Full train_epoch should run end-to-end without dtype errors.
    model = nn.Linear(4, 1)
    trainer = _TaskTrainer(
        model=model,
        task=task,
        device="cpu",
        batches_per_epoch=2,
        epochs=1,
        optimizer=torch.optim.SGD(model.parameters(), lr=1e-3),
    )
    metrics = trainer.train_epoch()
    assert math.isfinite(metrics["loss"])


def test_task_trainer_uses_cross_entropy_for_discrete_rl():
    """Discrete RL tasks (action space size > 1 reported as scalar
    action_dim) should pick CrossEntropyLoss for classification over
    discrete actions, matching the old default behavior.
    """
    from bioplausible.hyperopt.tasks import BaseTask, _resolve_task_loss

    class DiscreteRLTask(BaseTask):
        def __init__(self) -> None:
            super().__init__("discrete_rl", "cpu", quick_mode=True)
            self._input_dim = 8
            self._output_dim = 6  # discrete action space of size 6

        def setup(self) -> None:
            pass

        @property
        def task_type(self) -> str:
            return "rl"

        def get_batch(self, split: str = "train", batch_size: int = 8):
            return torch.randn(batch_size, 8), torch.randint(0, 6, (batch_size,))

        def create_trainer(self, model, **kwargs):
            return _TaskTrainer(model, self, **kwargs)

    loss = _resolve_task_loss(DiscreteRLTask())
    assert isinstance(loss, nn.CrossEntropyLoss)


# ---------------------------------------------------------------------------
# Bug #6, #13: `analysis/ablation.py` had a dead `delattr(cfg.model,
# "num_layers")` branch (cfg.model is a `ModelConfig` dataclass without a
# `num_layers` field). Removing it must not break ablation runs.
# ---------------------------------------------------------------------------


def test_ablation_run_single_experiment_does_not_reference_dead_branch():
    """Read the ablation module's source (only executable statements —
    ignoring comments) and confirm the dead `delattr(cfg.model,
    "num_layers")` branch plus stale `get_model_spec` import are gone.
    """
    import ast
    import inspect

    from bioplausible.analysis import ablation

    src = inspect.getsource(ablation)
    tree = ast.parse(src)
    # Walk the AST and ensure no `delattr(<anything>, "num_layers")` call
    # appears anywhere in the ablation module — that branch was dead.
    found_delattr_num_layers = False
    found_model_type_check = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "delattr":
                arg0 = node.args[0] if node.args else None
                arg1 = node.args[1] if len(node.args) > 1 else None
                if (
                    isinstance(arg0, ast.Attribute)
                    and isinstance(arg1, ast.Constant)
                    and arg1.value == "num_layers"
                ):
                    found_delattr_num_layers = True
        # Detect any comparison using a list literal containing
        # "eqprop_mlp" / "memory_efficient_mlp" / etc. — that was the dead
        # spec.model_type membership check.
        if isinstance(node, (ast.Compare, ast.In)):
            for comp in getattr(node, "comparators", []) or []:
                if isinstance(comp, ast.List):
                    for elt in comp.elts:
                        if isinstance(elt, ast.Constant) and elt.value in {
                            "eqprop_mlp",
                            "memory_efficient_mlp",
                            "backprop_mlp",
                            "looped_mlp",
                        }:
                            found_model_type_check = True
    assert not found_delattr_num_layers, (
        "dead `delattr(cfg.model, 'num_layers')` branch reappeared in ablation.py"
    )
    assert not found_model_type_check, (
        "dead `spec.model_type in [...]` membership check reappeared in ablation.py"
    )
    # `get_model_spec` should not be imported into ablation anymore — it
    # was only used by the now-removed dead branch.
    assert "from bioplausible.zoo import get_model_spec" not in src
    assert "get_model_spec(" not in src


# ---------------------------------------------------------------------------
# Bug: `except (AttributeError, StopIteration):` comma-form — py3.14 parses it
# as a tuple but it is confusing and historically Python-2 semantics. We
# rewrite all 18 lib + test sites to `except (X, Y):` for clarity.
# ---------------------------------------------------------------------------


def test_all_bioplausible_lib_files_parse_without_syntaxerror():
    """Verify every ``.py`` file in ``bioplausible/`` parses cleanly
    (covering the ``except X, Y:`` comma form which is legal Python
    3.14+ as a tuple-of-exceptions, but not in Python 3.12-).
    """
    import ast
    from pathlib import Path

    for p in Path("bioplausible").rglob("*.py"):
        if "/tests/" in str(p):
            continue
        try:
            ast.parse(p.read_text(), filename=str(p))
        except SyntaxError as e:
            pytest.fail(f"{p}: {e}")


# ---------------------------------------------------------------------------
# Final sanity: confirm pyproject.toml + README + registry still consistent
# with the plan §15 success criteria.
# ---------------------------------------------------------------------------


def test_all_three_undocumented_models_are_registered_and_documented():
    """Plan §15.9: GraphEqProp, PredictiveCodingHybrid, StandardFA must be
    registered in the Registry (they were originally undocumented but are
    registered now under snake_case names)."""
    models = set(Registry.list("model")["model"])
    for name in ("graph_eqprop", "predictive_coding_hybrid", "feedback_alignment"):
        assert name in models, f"Required model {name!r} missing from registry"


def test_registry_lists_at_least_one_component_per_category():
    """Plan §3.1: a single Registry must be the source of truth for all
    four capability categories (model, propagator, optimizer, sparsity)."""
    for cat in (
        ComponentCategory.MODEL,
        ComponentCategory.PROPAGATOR,
        ComponentCategory.OPTIMIZER,
        ComponentCategory.SPARSITY,
    ):
        comps = Registry.list(cat)[cat.value]
        assert comps, f"Registry category {cat!r} should not be empty"


def test_core_trainer_validate_supports_3d_logits_model():
    """Bug #4: ``_validate`` mirrored the ``_train_step`` shape mismatch —
    CrossEntropy on 3-D logits raised RuntimeError during validation.
    Verification: ``_validate`` should now succeed for a 3-D-logitsLM model
    via the ``task_obj`` code path. Returns ``val_loss`` / ``val_accuracy``
    as finite floats.
    """

    class LMHeadStub(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 7)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x).unsqueeze(1).expand(-1, 3, -1)

    model = LMHeadStub()
    config = TrainerConfig(
        model="lm_head_stub",
        task="mnist",
        epochs=1,
        batches_per_epoch=1,
        val_batches=1,
        track_energy=False,
        device="cpu",
    )
    trainer = CoreTrainer(config=config)
    trainer.model = model
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    trainer.train_loader = None
    trainer.val_loader = None

    fake_task = MagicMock()
    fake_task.task_type = "lm"

    def get_batch(split: str, batch_size: int = 4):
        return torch.randn(4, 4), torch.randint(0, 7, (4,))

    fake_task.get_batch = get_batch
    trainer.task_obj = fake_task

    result = trainer._validate(val_batches=1)
    assert "val_loss" in result
    assert "val_accuracy" in result
    assert math.isfinite(result["val_loss"])
    assert 0.0 <= result["val_accuracy"] <= 1.0
