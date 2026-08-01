"""Hypothesis property-based tests for acceleration kernels (Sprint 5.3).

Verifies numerical equivalence of the pure NumPy acceleration primitives
against their PyTorch references (and analytic expressions):
  - softmax outputs match torch.softmax
  - cross_entropy matches torch.nn.functional.cross_entropy
  - tanh_deriv matches the analytic 1 - tanh(x)^2
  - spectral_normalize recovers W/sigma and yields spectral norm ~= 1
  - shape invariants hold
"""

import numpy as np
import torch
from hypothesis import given
from hypothesis import strategies as st

from bioplausible.acceleration.kernels import (
    cross_entropy,
    softmax,
    spectral_normalize,
    tanh_deriv,
)

ATOL = 1e-5
SN_ATOL = 1e-2


@given(
    batch=st.integers(min_value=1, max_value=16),
    classes=st.integers(min_value=2, max_value=32),
)
def test_softmax_matches_torch(batch, classes):
    """softmax logits match torch.softmax and preserve shape."""
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((batch, classes)).astype(np.float32)
    out = softmax(logits)
    assert out.shape == logits.shape
    expected = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    np.testing.assert_allclose(out, expected, atol=ATOL)


@given(
    batch=st.integers(min_value=1, max_value=16),
    classes=st.integers(min_value=2, max_value=32),
)
def test_softmax_rows_sum_to_one(batch, classes):
    """Each logits row maps to a valid probability distribution."""
    rng = np.random.default_rng(1)
    logits = rng.standard_normal((batch, classes)).astype(np.float32)
    out = softmax(logits)
    np.testing.assert_allclose(out.sum(axis=-1), np.ones(batch), atol=ATOL)


@given(
    batch=st.integers(min_value=1, max_value=16),
    classes=st.integers(min_value=2, max_value=32),
)
def test_cross_entropy_matches_torch(batch, classes):
    """cross_entropy matches torch's cross-entropy for integer targets."""
    rng = np.random.default_rng(2)
    logits = rng.standard_normal((batch, classes)).astype(np.float32)
    targets = rng.integers(0, classes, size=batch)
    loss = cross_entropy(logits, targets)
    expected = torch.nn.functional.cross_entropy(
        torch.tensor(logits), torch.tensor(targets)
    ).item()
    np.testing.assert_allclose(loss, expected, atol=ATOL)


@given(
    batch=st.integers(min_value=1, max_value=8),
    dim=st.integers(min_value=1, max_value=64),
)
def test_tanh_deriv_matches_analytic(batch, dim):
    """tanh derivative matches the analytic expression."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal((batch, dim)).astype(np.float32)
    got = tanh_deriv(x)
    expected = 1.0 - np.tanh(x) ** 2
    np.testing.assert_allclose(got, expected, atol=ATOL)


@given(
    out=st.integers(min_value=1, max_value=16),
    inn=st.integers(min_value=1, max_value=16),
)
def test_spectral_normalize_recovers_scaled_matrix(out, inn):
    """W_normalized ~= W / sigma (the largest singular value)."""
    rng = np.random.default_rng(4)
    w = rng.standard_normal((out, inn)).astype(np.float32)
    w_norm, _u, sigma = spectral_normalize(w, num_iters=200)
    expected = w / (sigma + 1e-12)
    np.testing.assert_allclose(w_norm, expected, atol=1e-4)


@given(
    out=st.integers(min_value=1, max_value=16),
    inn=st.integers(min_value=1, max_value=16),
)
def test_spectral_normalize_unit_spectral_norm(out, inn):
    """The normalized matrix has spectral norm ~= 1 (Lipschitz <= 1)."""
    rng = np.random.default_rng(5)
    w = rng.standard_normal((out, inn)).astype(np.float32)
    w_norm, _u, _sigma = spectral_normalize(w, num_iters=200)
    s = torch.linalg.svdvals(torch.tensor(w_norm)).max().item()
    assert abs(s - 1.0) < SN_ATOL


@given(
    out=st.integers(min_value=2, max_value=16),
    inn=st.integers(min_value=2, max_value=16),
)
def test_spectral_norm_approximates_svd(out, inn):
    """Estimated sigma approximates the true largest singular value."""
    rng = np.random.default_rng(6)
    w = rng.standard_normal((out, inn)).astype(np.float32)
    _w_norm, _u, sigma = spectral_normalize(w, num_iters=200)
    true_sigma = torch.linalg.svdvals(torch.tensor(w)).max().item()
    assert abs(sigma - true_sigma) <= 0.05 * abs(true_sigma) + 1e-4


@given(dim=st.integers(min_value=1, max_value=64))
def test_softmax_shape_preserved_1d(dim):
    """softmax on a 1-D vector preserves its length."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal(dim).astype(np.float32)
    out = softmax(x)
    assert out.shape == (dim,)
