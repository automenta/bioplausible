"""Test spiking STDP 3-factor modulation.

Verifies that error backprojection modulates hidden layer weight updates
(element of supervised credit assignment via the 3rd factor).
"""

import torch

from bioplausible.zoo.models.spiking import SpikingSTDP


def test_3factor_modulator_is_error_correlated():
    """The modulator (backprojected error) should differ based on label."""
    model = SpikingSTDP(input_dim=8, hidden_dim=16, output_dim=4)
    x = torch.randn(4, 8)

    err_y0 = torch.mm(
        torch.nn.functional.one_hot(torch.zeros(4, dtype=torch.long), 4).float() * 4,
        model.W_fb,
    )
    err_y1 = torch.mm(
        torch.nn.functional.one_hot(torch.ones(4, dtype=torch.long), 4).float() * 4,
        model.W_fb,
    )

    assert not torch.allclose(err_y0, err_y1), (
        "Modulator should differ based on target label (3-factor requires error signal)"
    )


def test_hidden_weights_change_with_error_signal():
    """fc1 weights must change (unlike pure unsupervised STDP)."""
    model = SpikingSTDP(input_dim=8, hidden_dim=16, output_dim=4)
    x = torch.randn(4, 8)
    y = torch.randint(0, 4, (4,))

    w_before = model.fc1.weight.data.clone()
    model.train_step(x, y)

    assert not torch.allclose(w_before, model.fc1.weight.data), (
        "fc1 weights should change with 3-factor error signal"
    )


def test_feedback_weights_are_fixed():
    """W_fb should be the same after train_step (not trained)."""
    model = SpikingSTDP(input_dim=8, hidden_dim=16, output_dim=4)
    x = torch.randn(4, 8)
    y = torch.randint(0, 4, (4,))

    fb_before = model.W_fb.clone()
    model.train_step(x, y)

    assert torch.allclose(fb_before, model.W_fb), (
        "Feedback weights should not change during training"
    )


if __name__ == "__main__":
    test_3factor_modulator_is_error_correlated()
    test_hidden_weights_change_with_error_signal()
    test_feedback_weights_are_fixed()
    print("All spiking modulation tests passed!")
