"""Smoke tests for zoo/models/wrappers.py (RecurrentWrapper, StackedRecurrentWrapper, TransformerEqPropWrapper)."""

import pytest
import torch
from bioplausible.zoo.models.wrappers import (
    RecurrentWrapper,
    StackedRecurrentWrapper,
    TransformerEqPropWrapper,
    create_rnn_eqprop,
    create_transformer_eqprop,
)


def test_recurrent_wrapper_forward():
    """Basic forward pass through RecurrentWrapper with RNNCell."""
    model = RecurrentWrapper(
        cell=torch.nn.RNNCell(8, 16),
        input_dim=8,
        hidden_dim=16,
        output_dim=4,
        use_spectral_norm=False,
        max_steps=3,
    )
    x = torch.randn(2, 8)
    logits = model(x)
    assert logits.shape == (2, 4)
    assert not logits.isnan().any()


def test_recurrent_wrapper_spectral_norm():
    """RecurrentWrapper with spectral norm enabled."""
    model = RecurrentWrapper(
        cell=torch.nn.RNNCell(8, 16),
        input_dim=8,
        hidden_dim=16,
        output_dim=4,
        use_spectral_norm=True,
        max_steps=3,
    )
    x = torch.randn(2, 8)
    logits = model(x)
    assert logits.shape == (2, 4)
    assert not logits.isnan().any()


def test_stacked_recurrent_wrapper_rnn():
    """StackedRecurrentWrapper with RNN cells."""
    model = StackedRecurrentWrapper(
        cell_type="rnn",
        input_dim=8,
        hidden_dim=16,
        output_dim=4,
        num_layers=2,
        use_spectral_norm=False,
        max_steps=3,
    )
    x = torch.randn(2, 8)
    logits = model(x)
    assert logits.shape == (2, 4)
    assert not logits.isnan().any()


def test_stacked_recurrent_wrapper_gru():
    """StackedRecurrentWrapper with GRU cells."""
    model = StackedRecurrentWrapper(
        cell_type="gru",
        input_dim=8,
        hidden_dim=16,
        output_dim=4,
        num_layers=2,
        use_spectral_norm=False,
        max_steps=3,
    )
    x = torch.randn(2, 8)
    logits = model(x)
    assert logits.shape == (2, 4)
    assert not logits.isnan().any()


@pytest.mark.slow
def test_stacked_recurrent_wrapper_lstm():
    """StackedRecurrentWrapper with LSTM cells (marked slow due to LSTM overhead)."""
    model = StackedRecurrentWrapper(
        cell_type="lstm",
        input_dim=8,
        hidden_dim=16,
        output_dim=4,
        num_layers=2,
        use_spectral_norm=False,
        max_steps=3,
    )
    x = torch.randn(2, 8)
    logits = model(x)
    assert logits.shape == (2, 4)
    assert not logits.isnan().any()


def test_transformer_wrapper_forward():
    """Basic forward pass through TransformerEqPropWrapper."""
    model = TransformerEqPropWrapper(
        input_dim=8,
        hidden_dim=16,
        output_dim=4,
        num_heads=2,
        num_layers=2,
        dim_feedforward=32,
        use_spectral_norm=False,
        max_steps=3,
    )
    x = torch.randn(2, 8)
    logits = model(x)
    assert logits.shape == (2, 4)
    assert not logits.isnan().any()


def test_transformer_wrapper_spectral_norm():
    """Transformer wrapper with spectral norm enabled."""
    model = TransformerEqPropWrapper(
        input_dim=8,
        hidden_dim=16,
        output_dim=4,
        num_heads=2,
        num_layers=2,
        dim_feedforward=32,
        use_spectral_norm=True,
        max_steps=3,
    )
    x = torch.randn(2, 8)
    logits = model(x)
    assert logits.shape == (2, 4)
    assert not logits.isnan().any()


def test_create_rnn_eqprop_single_layer():
    """create_rnn_eqprop with single layer returns RecurrentWrapper."""
    model = create_rnn_eqprop(input_dim=8, hidden_dim=16, output_dim=4, num_layers=1)
    assert isinstance(model, RecurrentWrapper)
    x = torch.randn(2, 8)
    logits = model(x)
    assert logits.shape == (2, 4)


def test_create_rnn_eqprop_stacked():
    """create_rnn_eqprop with multiple layers returns StackedRecurrentWrapper."""
    model = create_rnn_eqprop(input_dim=8, hidden_dim=16, output_dim=4, num_layers=2)
    assert isinstance(model, StackedRecurrentWrapper)
    x = torch.randn(2, 8)
    logits = model(x)
    assert logits.shape == (2, 4)


def test_create_transformer_eqprop():
    """create_transformer_eqprop returns a working TransformerEqPropWrapper."""
    model = create_transformer_eqprop(
        input_dim=8, hidden_dim=16, output_dim=4, num_heads=2, num_layers=2
    )
    assert isinstance(model, TransformerEqPropWrapper)
    x = torch.randn(2, 8)
    logits = model(x)
    assert logits.shape == (2, 4)
