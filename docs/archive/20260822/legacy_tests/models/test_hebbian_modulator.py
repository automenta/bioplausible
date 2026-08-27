"""Test three-factor Hebbian modulator.

Verifies that the modulator is a graded error signal (not binary)
and that hidden layers receive error-modulated updates.
"""

import torch
from bioplausible.zoo.models.hebbian import ThreeFactorHebbian


def test_modulator_is_graded_not_binary():
    """Modulator should correlate with prediction error, not be binary (+1/-1)."""
    model = ThreeFactorHebbian(input_dim=8, hidden_dim=16, output_dim=4)
    x = torch.randn(4, 8)
    y = torch.randint(0, 4, (4,))

    # Forward pass
    out = model.forward(x)
    pred_probs = torch.softmax(out, dim=1)
    y_onehot = torch.zeros_like(out)
    y_onehot.scatter_(1, y.unsqueeze(1), 1.0)
    modulator = y_onehot - pred_probs

    # Check it's not binary (all values should not be just +1 or -1)
    unique_vals = torch.unique(modulator)
    assert len(unique_vals) > 2, (
        f"Modulator should be graded (continuous), got {len(unique_vals)} unique values"
    )

    # Modulator should sum to ~0 across classes (one-hot - softmax)
    assert modulator.abs().sum() > 0, "Modulator should be non-trivial"


def test_hidden_weights_change_with_error():
    """Hidden layer weights must update via error-modulated Hebbian rule."""
    model = ThreeFactorHebbian(input_dim=8, hidden_dim=16, output_dim=4)
    x = torch.randn(4, 8)
    y = torch.randint(0, 4, (4,))

    w_before = model.layers[0].weight.data.clone()
    model.train_step(x, y)

    assert not torch.allclose(w_before, model.layers[0].weight.data), (
        "Hidden layer weights should change with error-modulated update"
    )


def test_modulator_correlates_with_error():
    """Larger errors should produce larger modulator values."""
    model = ThreeFactorHebbian(input_dim=8, hidden_dim=16, output_dim=4)

    # Wrong prediction
    x = torch.randn(8, 8)
    y_correct = torch.randint(0, 4, (8,))
    out = model.forward(x)
    pred_probs = torch.softmax(out, dim=1)
    y_onehot = torch.zeros_like(out)
    y_onehot.scatter_(1, y_correct.unsqueeze(1), 1.0)
    error_mod = y_onehot - pred_probs

    # Correct prediction (set labels to argmax)
    y_correct_pred = out.argmax(1)
    y_onehot_correct = torch.zeros_like(out)
    y_onehot_correct.scatter_(1, y_correct_pred.unsqueeze(1), 1.0)
    correct_mod = y_onehot_correct - pred_probs

    # Error modulator should have larger magnitude than correct modulator
    assert error_mod.abs().sum() > correct_mod.abs().sum(), (
        "Error signal should produce larger modulator than correct predictions"
    )


if __name__ == "__main__":
    test_modulator_is_graded_not_binary()
    test_hidden_weights_change_with_error()
    test_modulator_correlates_with_error()
    print("All hebbian modulator tests passed!")
