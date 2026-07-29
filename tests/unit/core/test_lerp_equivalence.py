import torch


def test_lerp_equivalence():
    """Verify torch.lerp matches manual interpolation."""
    torch.manual_seed(42)

    batch_size = 100
    dim = 50

    h = torch.randn(batch_size, dim)
    target = torch.randn(batch_size, dim)
    alpha = 0.5

    manual = (1 - alpha) * h + alpha * target
    lerp_out = torch.lerp(h, target, alpha)

    diff = (manual - lerp_out).abs().max()
    assert diff <= 1e-6, f"torch.lerp deviates: max diff={diff.item()}"


def test_max_norm_equivalence():
    """Verify max norm behavior."""
    torch.manual_seed(42)

    h_new = torch.randn(10, 10)
    h = torch.randn(10, 10)

    delta_manual = (h_new - h).abs().max()
    delta_dist = torch.dist(h_new, h, p=float("inf"))

    diff = (delta_manual - delta_dist).abs().item()
    assert diff <= 1e-6, f"torch.dist(p=inf) deviates: diff={diff}"
