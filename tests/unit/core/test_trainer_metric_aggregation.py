"""Metric aggregation contract (TODO11 R11.2.23) + GradientCredit fail-loud.

Epoch metrics are sample-weighted running sums — a ragged final batch must
not over-weight per-batch means — and validation reports
``val_ppl = exp(mean CE)`` from the same per-sample normalization.
``GradientCredit`` raises when the loss graph fails to reach a learnable
weight instead of silently zero-filling.
"""

import math

import pytest
import torch

from computronium import (
    DigitalSubstrate,
    EuclideanUpdate,
    FeedforwardGeometry,
    GeometryConfig,
    InstantaneousDynamics,
    ParameterUpdateConfig,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemTrainer,
    SystemTrainerConfig,
    compose_system,
)
from computronium.ontology import GradientCredit
from computronium.ontology.credit import BackpropCredit

_DIM_IN, _DIM_OUT, _HIDDEN = 12, 4, 8


def _make_system():
    return compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device="cpu")),
        geometry=FeedforwardGeometry(
            GeometryConfig.feedforward(
                input_dim=_DIM_IN, output_dim=_DIM_OUT, hidden_dims=(_HIDDEN,)
            )
        ),
        dynamics=InstantaneousDynamics(StateDynamicsConfig.instantaneous()),
        credit=BackpropCredit(),
        update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.05)),
    )


def _stream(
    batch_sizes: tuple[int, ...], seed: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    g = torch.Generator().manual_seed(seed)
    return [
        (
            torch.randn(n, _DIM_IN, generator=g),
            torch.randint(0, _DIM_OUT, (n,), generator=g),
        )
        for n in batch_sizes
    ]


def _flatten(batches):
    return torch.cat([x for x, _ in batches]), torch.cat([y for _, y in batches])


class _SpySystem:
    """Delegating instrument: records per-batch metrics, freezes nothing."""

    def __init__(self, inner):
        self._inner = inner
        self.calls: list[tuple[dict[str, float], int]] = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def train_step(self, x, y):
        m = self._inner.train_step(x, y)
        self.calls.append((m, x.size(0)))
        return m


def test_train_epoch_metrics_are_sample_weighted() -> None:
    torch.manual_seed(0)
    batches = _stream((3, 5), seed=3)
    system = compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device="cpu")),
        geometry=FeedforwardGeometry(
            GeometryConfig.feedforward(
                input_dim=_DIM_IN, output_dim=_DIM_OUT, hidden_dims=(_HIDDEN,)
            )
        ),
        dynamics=InstantaneousDynamics(StateDynamicsConfig.instantaneous()),
        credit=BackpropCredit(),
        update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.0)),
    )
    spy = _SpySystem(system)
    trainer = SystemTrainer(
        system=spy,  # type: ignore[arg-type]
        config=SystemTrainerConfig(max_epochs=1, device="cpu", seed=0),
        train_data=batches,
    )
    metrics = trainer.train_epoch()

    per_batch = spy.calls
    n = sum(size for _, size in per_batch)
    for key, epoch_key in (
        ("loss", "train_loss"),
        ("energy", "train_energy"),
    ):
        weighted = sum(m.get(key, 0.0) * size for m, size in per_batch) / n
        assert metrics[epoch_key] == pytest.approx(weighted)
    acc_key = (
        "free_accuracy" if "free_accuracy" in per_batch[0][0] else "nudged_fit_accuracy"
    )
    weighted_acc = sum(m.get(acc_key, 0.0) * size for m, size in per_batch) / n
    assert metrics["train_acc"] == pytest.approx(weighted_acc)

    unweighted = sum(m.get("loss", 0.0) for m, _ in per_batch) / len(per_batch)
    assert metrics["train_loss"] != pytest.approx(unweighted, abs=1e-9)


def test_validate_reports_weighted_loss_acc_and_ppl() -> None:
    torch.manual_seed(0)
    val_batches = _stream((3, 5), seed=11)
    trainer = SystemTrainer(
        system=_make_system(),
        config=SystemTrainerConfig(max_epochs=1, device="cpu", seed=0),
        train_data=_stream((4, 4), seed=3),
        val_data=val_batches,
    )
    trainer.train_epoch()
    val = trainer.validate()

    x, y = _flatten(val_batches)
    logits = trainer.system.forward(x)
    ce = torch.nn.functional.cross_entropy(logits, y, reduction="sum")
    expected_loss = (ce / 8).item()
    expected_acc = (logits.argmax(-1) == y).sum().item() / 8

    assert val["val_loss"] == pytest.approx(expected_loss, abs=1e-4)
    assert val["val_acc"] == pytest.approx(expected_acc, abs=1e-4)
    assert val["val_ppl"] == pytest.approx(math.exp(expected_loss), rel=1e-6)


def _geometry() -> FeedforwardGeometry:
    torch.manual_seed(0)
    return FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=_DIM_IN, output_dim=_DIM_OUT, hidden_dims=(_HIDDEN,)
        )
    )


def test_gradient_credit_raises_on_detached_weights() -> None:
    credit = BackpropCredit()
    orphan = torch.randn((), requires_grad=True)
    with pytest.raises(RuntimeError, match="no gradient reached"):
        credit.compute_pseudo_gradient({}, orphan, _geometry())


def test_gradient_credit_returns_grads_on_full_graph() -> None:
    credit = BackpropCredit()
    geometry = _geometry()
    x = torch.randn(4, _DIM_IN)
    y = torch.randint(0, _DIM_OUT, (4,))
    loss = torch.nn.functional.cross_entropy(geometry.forward(x), y)
    grads = credit.compute_pseudo_gradient({}, loss, geometry)
    assert grads
    assert all(isinstance(g, torch.Tensor) for g in grads)
    assert all(g.abs().sum() > 0 for g in grads)


def test_backprop_alias_shares_fail_loud() -> None:
    assert BackpropCredit is GradientCredit
