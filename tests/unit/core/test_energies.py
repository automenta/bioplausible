"""Tests for shared energy functions in core/energies.py."""

import torch

from computronium.core.energies import (
    contrastive_energy,
    hybrid_energy,
    mse_energy,
    node_energy,
    prediction_error_energy,
    supervised_energy,
)


class TestPredictionErrorEnergy:
    def test_non_negative(self) -> None:
        h = torch.randn(4, 20)
        acts = [torch.randn(4, 10), h, torch.randn(4, 5)]
        preds = [torch.randn(4, 20), torch.randn(4, 5)]
        energy = prediction_error_energy(acts, preds)
        assert energy >= 0

    def test_zero_for_exact_match(self) -> None:
        h = torch.randn(4, 20)
        out = torch.randn(4, 5)
        acts = [torch.randn(4, 10), h, out]
        preds = [h.clone(), out.clone()]
        energy = prediction_error_energy(acts, preds)
        assert energy <= 1e-6

    def test_single_layer(self) -> None:
        acts = [torch.randn(4, 8), torch.randn(4, 5)]
        preds = [torch.randn(4, 5)]
        energy = prediction_error_energy(acts, preds)
        assert energy >= 0

    def test_with_weights(self) -> None:
        h = torch.randn(4, 20)
        out = torch.randn(4, 5)
        acts = [torch.randn(4, 10), h, out]
        preds = [torch.randn(4, 20), torch.randn(4, 5)]
        weights = [torch.tensor(0.5), torch.tensor(1.0)]
        energy = prediction_error_energy(acts, preds, weights=weights)
        assert energy >= 0


class TestSupervisedEnergy:
    def test_non_negative(self) -> None:
        logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        energy = supervised_energy(logits, targets)
        assert energy >= 0

    def test_default_loss_is_ce(self) -> None:
        logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        energy = supervised_energy(logits, targets)
        expected = torch.nn.functional.cross_entropy(logits, targets)
        assert abs(energy - expected) < 1e-6


class TestHybridEnergy:
    def test_non_negative(self) -> None:
        h = torch.randn(4, 20)
        acts = [torch.randn(4, 10), h, torch.randn(4, 5)]
        preds = [torch.randn(4, 20), torch.randn(4, 5)]
        logits = torch.randn(4, 5)
        targets = torch.randint(0, 5, (4,))
        energy = hybrid_energy(acts, preds, logits, targets)
        assert energy >= 0

    def test_supervised_weight_zero(self) -> None:
        h = torch.randn(4, 20)
        out = torch.randn(4, 5)
        acts = [torch.randn(4, 10), h, out]
        preds = [h.clone(), out.clone()]
        logits = torch.randn(4, 5)
        targets = torch.randint(0, 5, (4,))
        energy = hybrid_energy(acts, preds, logits, targets, supervised_weight=0.0)
        assert energy <= 1e-6

    def test_supervised_weight_scales(self) -> None:
        h = torch.randn(4, 20)
        acts = [torch.randn(4, 10), h, torch.randn(4, 5)]
        preds = [torch.randn(4, 20), torch.randn(4, 5)]
        logits = torch.randn(4, 5)
        targets = torch.randint(0, 5, (4,))
        e1 = hybrid_energy(acts, preds, logits, targets, supervised_weight=1.0)
        e2 = hybrid_energy(acts, preds, logits, targets, supervised_weight=2.0)
        assert e2 > e1


class TestContrastiveEnergy:
    def test_non_negative(self) -> None:
        fe = torch.tensor(0.5)
        ne = torch.tensor(1.2)
        assert contrastive_energy(fe, ne, beta=0.1) >= 0

    def test_negative_when_nudged_lower(self) -> None:
        assert contrastive_energy(torch.tensor(1.0), torch.tensor(0.3), beta=0.1) < 0

    def test_inverse_beta(self) -> None:
        fe = torch.tensor(1.0)
        ne = torch.tensor(2.0)
        e1 = contrastive_energy(fe, ne, beta=0.5)
        e2 = contrastive_energy(fe, ne, beta=1.0)
        assert e2 < e1

    def test_zero_equal(self) -> None:
        e = torch.tensor(0.5)
        assert abs(contrastive_energy(e, e, beta=0.1)) <= 1e-6


class TestMSEEnergy:
    def test_non_negative(self) -> None:
        pred = torch.randn(4, 10)
        target = torch.randn(4, 10)
        energy = mse_energy(pred, target)
        assert energy >= 0

    def test_zero_for_perfect_match(self) -> None:
        x = torch.randn(4, 10)
        assert mse_energy(x, x) <= 1e-6


class TestNodeEnergy:
    def test_non_negative(self) -> None:
        assert node_energy(torch.randn(4, 20), reg_weight=1.0) >= 0

    def test_zero_for_zero_activity(self) -> None:
        assert node_energy(torch.zeros(4, 20), reg_weight=1.0) <= 1e-6

    def test_zero_reg_weight(self) -> None:
        assert node_energy(torch.randn(4, 20), reg_weight=0.0) <= 1e-6

    def test_scales_with_reg_weight(self) -> None:
        x = torch.randn(4, 20)
        e1 = node_energy(x, reg_weight=1.0)
        e2 = node_energy(x, reg_weight=2.0)
        assert abs(e2 - 2 * e1) < 1e-6
