"""Hypothesis property-based tests for energy functions in core/energies.py."""

import torch
from hypothesis import assume, given
from hypothesis import strategies as st

from bioplausible.core.energies import (
    contrastive_energy,
    hybrid_energy,
    mse_energy,
    node_energy,
    prediction_error_energy,
    supervised_energy,
)


def _tensor_strat(
    min_dim: int = 1, max_dim: int = 4, batch: int = 4, features: int = 10
):
    """Strategy for random tensors."""
    return st.just(torch.randn(batch, features))


@given(
    acts_0=st.lists(_tensor_strat(), min_size=2, max_size=4),
    preds=st.lists(_tensor_strat(), min_size=1, max_size=3),
)
def test_prediction_error_non_negative(acts_0, preds):
    """Energy is always non-negative."""
    # Ensure compatible shapes: acts[i+1] and preds[i] must match in trailing dim
    assume(len(acts_0) >= 2 and len(preds) >= 1)
    # Trim to ensure acts and preds alignment is possible
    n = min(len(acts_0) - 1, len(preds))
    acts = acts_0[: n + 1]
    preds_sub = preds[:n]
    # Align last dim of preds with next act
    for i in range(n):
        assume(acts[i + 1].shape[-1] == preds_sub[i].shape[-1])
    energy = prediction_error_energy(acts, preds_sub)
    assert energy >= 0


@given(
    acts=st.lists(_tensor_strat(), min_size=2, max_size=4),
    preds=st.lists(_tensor_strat(), min_size=1, max_size=3),
)
def test_prediction_error_zero_for_exact_match(acts, preds):
    """Energy is zero when activities exactly equal predictions."""
    assume(len(acts) >= 2 and len(preds) >= 1)
    n = min(len(acts) - 1, len(preds))
    # Force exact match: copy act into pred
    preds_sub = [acts[i + 1].clone() for i in range(n)]
    acts_sub = acts[: n + 1]
    energy = prediction_error_energy(acts_sub, preds_sub)
    assert energy <= 1e-6


@given(
    pred=st.lists(_tensor_strat(features=5), min_size=1, max_size=1).map(
        lambda x: x[0]
    ),
    target=st.lists(_tensor_strat(features=5), min_size=1, max_size=1).map(
        lambda x: x[0]
    ),
)
def test_mse_energy_non_negative(pred, target):
    """MSE energy is always non-negative."""
    assume(pred.shape == target.shape)
    energy = mse_energy(pred, target)
    assert energy >= 0


@given(
    x=st.lists(_tensor_strat(), min_size=1, max_size=1).map(lambda x: x[0]),
)
def test_mse_energy_zero_for_exact_match(x):
    """MSE energy is zero for exact match."""
    assert mse_energy(x, x) <= 1e-6


@given(
    activity=_tensor_strat(),
    reg_weight=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
)
def test_node_energy_non_negative(activity, reg_weight):
    """Node energy is always non-negative with zero bias."""
    energy = node_energy(activity, reg_weight=reg_weight)
    assert energy >= 0


def test_node_energy_zero_activity_zero_reg():
    """Node energy is zero when activity is zero."""
    assert node_energy(torch.zeros(4, 10), reg_weight=0.0) <= 1e-6


@given(
    free=st.floats(min_value=0.0, max_value=5.0, allow_nan=False),
    nudged=st.floats(min_value=0.0, max_value=5.0, allow_nan=False),
    beta=st.floats(min_value=0.01, max_value=5.0, allow_nan=False),
)
def test_contrastive_energy_sign(free, nudged, beta):
    """Contrastive energy sign matches (nudged - free) / beta."""
    fe = torch.tensor(free)
    ne = torch.tensor(nudged)
    energy = contrastive_energy(fe, ne, beta)
    expected = (ne - fe) / beta
    assert abs(energy - expected) < 1e-6


@given(
    free=st.floats(min_value=0.0, max_value=5.0, allow_nan=False),
    beta=st.floats(min_value=0.01, max_value=5.0, allow_nan=False),
)
def test_contrastive_energy_equal_is_zero(free, beta):
    """Contrastive energy is zero when free == nudged."""
    val = torch.tensor(free)
    assert abs(contrastive_energy(val, val, beta)) <= 1e-6


@given(
    logits=_tensor_strat(features=10),
    targets=st.integers(min_value=0, max_value=9),
    supervised_weight=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
)
def test_hybrid_energy_decomposes(logits, targets, supervised_weight):
    """Hybrid energy = prediction_error + supervised_weight * cross_entropy."""
    assume(supervised_weight >= 0)
    batch = logits.shape[0]
    tgt = torch.full((batch,), targets, dtype=torch.long)
    h = torch.randn(batch, 20)
    out = torch.randn(batch, 10)
    acts = [torch.randn(batch, 10), h, out]
    preds = [torch.randn(batch, 20), torch.randn(batch, 10)]
    pe = prediction_error_energy(acts, preds)
    se = supervised_energy(logits, tgt)
    he = hybrid_energy(acts, preds, logits, tgt, supervised_weight=supervised_weight)
    assert abs(he - (pe + supervised_weight * se)) < 1e-5
