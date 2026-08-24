"""Hypothesis property-based tests for domains.base value objects
(Sprint 5.7).

Laws:
  - Metrics.from_dict(to_dict(m)) == m (round-trip, reserved keys preserved).
  - Batch.to(device) preserves metadata, batch_size, and input/output tensors.
  - DomainSpec is immutable and default metrics are preserved.
"""

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from computronium.domains.base import Batch, DomainSpec, DomainType, Metrics

RESERVED = {"loss", "accuracy", "perplexity"}
CUSTOM_VALUE = 0.5
ACC = 0.9
PPL = 5.0
DEFAULT_BATCH = 32
DEFAULT_LR = 1e-3
custom_key_strat = st.text(min_size=1, max_size=8, alphabet="abcdef").filter(
    lambda s: s not in RESERVED
)


@given(
    loss=st.floats(min_value=1e-6, max_value=1e3, allow_nan=False),
    acc=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
    ppl=st.one_of(st.none(), st.floats(min_value=1.0, max_value=1e3, allow_nan=False)),
)
def test_metrics_round_trip(loss, acc, ppl):
    """from_dict(to_dict(m)) reproduces the Metrics value object."""
    m = Metrics(loss=loss, accuracy=acc, perplexity=ppl)
    assert Metrics.from_dict(m.to_dict()) == m


@given(
    loss=st.floats(min_value=1e-6, max_value=1e3, allow_nan=False),
    custom_keys=st.lists(custom_key_strat, min_size=0, max_size=4, unique=True),
)
def test_metrics_custom_round_trip(loss, custom_keys):
    """Custom metric dict round-trips through to_dict/from_dict."""
    custom = dict.fromkeys(custom_keys, CUSTOM_VALUE)
    m = Metrics(loss=loss, custom=custom)
    restored = Metrics.from_dict(m.to_dict())
    assert restored.loss == loss
    for k in custom_keys:
        assert restored.custom[k] == pytest.approx(CUSTOM_VALUE)


@pytest.mark.parametrize("acc,ppl", [(ACC, PPL), (None, None)])
def test_metrics_to_dict_reserved_presence(acc, ppl):
    """accuracy/perplexity appear in to_dict only when not None."""
    d = Metrics(loss=1.0, accuracy=acc, perplexity=ppl).to_dict()
    if acc is None:
        assert "accuracy" not in d
    else:
        assert d["accuracy"] == pytest.approx(ACC)
    if ppl is None:
        assert "perplexity" not in d
    else:
        assert d["perplexity"] == pytest.approx(PPL)


@given(
    batch=st.integers(min_value=1, max_value=16),
    feats=st.integers(min_value=1, max_value=16),
    nclasses=st.integers(min_value=1, max_value=8),
)
def test_batch_to_preserves_metadata_and_shape(batch, feats, nclasses):
    """Batch.to(device) preserves tensors, metadata, and batch_size."""
    b = Batch(
        inputs=torch.randn(batch, feats, 3, 3),
        targets=torch.randint(0, nclasses, (batch,)),
        metadata={"task": "vision", "split": "train"},
    )
    moved = b.to(torch.device("cpu"))
    assert torch.equal(moved.inputs, b.inputs)
    assert torch.equal(moved.targets, b.targets)
    assert moved.metadata == {"task": "vision", "split": "train"}
    assert moved.batch_size == batch


@given(
    n=st.integers(min_value=1, max_value=64),
    d=st.integers(min_value=1, max_value=64),
    batch=st.integers(min_value=1, max_value=16),
)
def test_batch_to_recomputes_batch_size(n, d, batch):
    """Moving to device preserves the number of examples."""
    b = Batch(inputs=torch.randn(batch, n, d), targets=torch.randn(batch, d))
    assert b.to(torch.device("cpu")).batch_size == batch


@given(name=st.text(min_size=1, max_size=20, alphabet="abcdef"))
def test_domain_spec_defaults(name):
    """DomainSpec uses expected defaults and keeps provided metrics."""
    spec = DomainSpec(name=name, domain_type=DomainType.VISION, default_metrics=["acc"])
    assert spec.description == ""
    assert spec.default_batch_size == DEFAULT_BATCH
    assert spec.default_lr == pytest.approx(DEFAULT_LR)
    assert spec.default_metrics == ["acc"]
    assert spec.supported_tasks == []
