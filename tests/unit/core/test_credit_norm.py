"""TODO12 A4 — credit_norm per-layer credit-signal normalization locks.

Zeros stay zeros (no fabricated signal), each mode's identity holds,
and "none" is bit-exact passthrough.
"""

import torch

from computronium.ontology.credit import _apply_credit_norm


def _grads() -> list[torch.Tensor]:
    torch.manual_seed(0)
    return [torch.randn(6, 4) * 3.0, torch.randn(4, 5), torch.zeros(3, 3)]


def test_none_is_passthrough():
    g = _grads()
    out = _apply_credit_norm(g, "none", None)
    for a, b in zip(g, out):
        assert torch.equal(a, b)


def test_rms_unit_per_layer_and_zeros_stay_zeros():
    g = _grads()
    out = _apply_credit_norm(g, "rms", None)
    for i in (0, 1):
        assert torch.isclose(
            out[i].square().mean().sqrt(), torch.tensor(1.0), rtol=1e-3
        )
    assert torch.equal(out[2], torch.zeros(3, 3))


def test_relative_uses_error_reference():
    g = _grads()
    refs = [torch.full((8,), 2.0), torch.ones(5), torch.zeros(3)]
    out = _apply_credit_norm(g, "relative", refs)
    expected0 = g[0] / (2.0 * 8**0.5)
    assert torch.allclose(out[0], expected0, rtol=1e-4)
    assert torch.equal(out[2], torch.zeros(3, 3))


def test_beta_adaptive_is_unit_rms_error_reference():
    g = _grads()
    refs = [torch.randn(10), torch.randn(4), torch.zeros(2)]
    out = _apply_credit_norm(g, "beta_adaptive", refs)
    for i in (0, 1):
        scale = refs[i].square().mean().sqrt()
        assert torch.allclose(out[i], g[i] / (scale + 1e-8), rtol=1e-4)


def test_spectral_radius_one():
    g = _grads()
    out = _apply_credit_norm(g, "spectral", None)
    for i in (0, 1):
        radius = torch.linalg.matrix_norm(out[i], ord=2)
        assert torch.isclose(radius, torch.tensor(1.0), rtol=1e-3)
    # vector-shaped grads pass through untouched
    vec = [torch.randn(6)]
    assert torch.equal(_apply_credit_norm(vec, "spectral", None)[0], vec[0])


def test_all_zero_grad_untouched_in_every_mode():
    g = [torch.zeros(4, 4)]
    for mode in ("relative", "rms", "beta_adaptive", "spectral"):
        assert torch.equal(_apply_credit_norm(g, mode, [torch.zeros(4)])[0], g[0])
