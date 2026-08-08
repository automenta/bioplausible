"""Tests for zoo/models/hebbian.py (HebbianLayer, DeepHebbianChain, HebbianCube, ThreeFactorHebbian)."""

import pytest
import torch

from bioplausible.zoo.models.hebbian import (
    DeepHebbianChain,
    HebbianCube,
    HebbianLayer,
    ThreeFactorHebbian,
)

NUM_CLASSES = 4


# ============================================================================
# HebbianLayer
# ============================================================================


class TestHebbianLayer:
    """Single Hebbian layer with Oja's rule."""

    def test_forward_shape(self):
        layer = HebbianLayer(8, 16)
        x = torch.randn(2, 8)
        out = layer(x)
        assert out.shape == (2, 16)
        assert not out.isnan().any()

    def test_weight_orthonormal_init(self):
        layer = HebbianLayer(8, 16)
        W = layer.weight.detach()
        # Orthogonal init: each out-feature column has norm ~gain=1.5

        col_norms = W.pow(2).sum(dim=0)
        assert abs(col_norms.mean().item() - 2.25) < 0.3

    def test_hebbian_update_with_oja(self):
        layer = HebbianLayer(8, 16, learning_rate=0.01, use_oja=True)
        x = torch.randn(4, 8)
        y = layer(x)
        w_before = layer.weight.data.clone()
        layer.hebbian_update(x, y)
        assert not (layer.weight.data == w_before).all()

    def test_hebbian_update_without_oja(self):
        layer = HebbianLayer(8, 16, learning_rate=0.01, use_oja=False)
        x = torch.randn(4, 8)
        y = layer(x)
        w_before = layer.weight.data.clone()
        layer.hebbian_update(x, y)
        assert not (layer.weight.data == w_before).all()

    def test_forward_after_hebbian_update(self):
        layer = HebbianLayer(8, 16, learning_rate=0.01, use_oja=True)
        x = torch.randn(2, 8)
        y0 = layer(x)
        layer.hebbian_update(x, y0)
        y1 = layer(x)
        assert y1.shape == (2, 16)

    def test_weight_norm_does_not_explode_many_updates(self):
        """After many Hebbian updates with Oja, weight norm should stay bounded."""
        layer = HebbianLayer(8, 16, learning_rate=0.005, use_oja=True)
        norms = []
        for _ in range(50):
            x = torch.randn(8, 8)
            y = layer(x)
            layer.hebbian_update(x, y)
            norms.append(layer.weight.data.pow(2).sum().item())

        # Weight norm should not grow unbounded — Oja's rule enforces stability
        W_norm = layer.weight.data.pow(2).sum().item()
        assert W_norm < 500.0, f"Weight norm exploded to {W_norm}"
        assert max(norms) < 500.0

    def test_batch_size_one(self):
        layer = HebbianLayer(8, 16)
        x = torch.randn(1, 8)
        out = layer(x)
        assert out.shape == (1, 16)
        layer.hebbian_update(x, out)

    @pytest.mark.parametrize("use_oja", [True, False])
    def test_oja_flag_respected(self, use_oja):
        layer = HebbianLayer(8, 16, use_oja=use_oja)
        assert layer.use_oja == use_oja


# ============================================================================
# DeepHebbianChain
# ============================================================================


class TestDeepHebbianChain:
    """Deep Hebbian chain with spectral normalization."""

    def test_forward_shape(self):
        model = DeepHebbianChain(
            input_dim=8,
            hidden_dim=16,
            output_dim=NUM_CLASSES,
            num_layers=3,
            use_spectral_norm=False,
            max_steps=1,
        )
        x = torch.randn(2, 8)
        out = model(x)
        assert out.shape == (2, NUM_CLASSES)
        assert not out.isnan().any()

    def test_forward_with_spectral_norm(self):
        model = DeepHebbianChain(
            input_dim=8,
            hidden_dim=16,
            output_dim=NUM_CLASSES,
            num_layers=3,
            use_spectral_norm=True,
            max_steps=1,
        )
        x = torch.randn(2, 8)
        out = model(x)
        assert out.shape == (2, NUM_CLASSES)

    def test_forward_eval_with_spectral_norm(self):
        """Eval mode with spectral norm uses cached normalized weights."""
        model = DeepHebbianChain(
            input_dim=8,
            hidden_dim=16,
            output_dim=NUM_CLASSES,
            num_layers=3,
            use_spectral_norm=True,
            max_steps=1,
        )
        model.eval()
        x = torch.randn(2, 8)
        out = model(x)
        assert out.shape == (2, NUM_CLASSES)

    def test_forward_return_signal_norms(self):
        model = DeepHebbianChain(
            input_dim=8,
            hidden_dim=16,
            output_dim=NUM_CLASSES,
            num_layers=3,
            use_spectral_norm=False,
            max_steps=1,
        )
        x = torch.randn(2, 8)
        out, norms = model(x, return_signal_norms=True)
        assert out.shape == (2, NUM_CLASSES)
        assert isinstance(norms, list)
        assert len(norms) == model.num_layers + 1

    def test_forward_return_signal_norms_spectral_norm(self):
        model = DeepHebbianChain(
            input_dim=8,
            hidden_dim=16,
            output_dim=NUM_CLASSES,
            num_layers=2,
            use_spectral_norm=True,
            max_steps=1,
        )
        model.eval()
        x = torch.randn(2, 8)
        out, norms = model(x, return_signal_norms=True)
        assert out.shape == (2, NUM_CLASSES)
        assert len(norms) == 3

    def test_measure_signal_propagation(self):
        model = DeepHebbianChain(
            input_dim=8,
            hidden_dim=16,
            output_dim=NUM_CLASSES,
            num_layers=3,
            use_spectral_norm=False,
            max_steps=1,
        )
        x = torch.randn(2, 8)
        result = model.measure_signal_propagation(x)
        assert isinstance(result, dict)
        assert "initial_norm" in result
        assert "final_norm" in result
        assert "decay_ratio" in result
        assert "norms" in result
        assert result["initial_norm"] > 0

    def test_get_stats_includes_hebbian_params(self):
        model = DeepHebbianChain(
            input_dim=8,
            hidden_dim=16,
            output_dim=NUM_CLASSES,
            num_layers=3,
            use_spectral_norm=True,
            max_steps=1,
            hebbian_lr=0.005,
            use_oja=False,
        )
        stats = model.get_stats()
        assert stats["hebbian_lr"] == pytest.approx(0.005)
        assert stats["use_oja"] is False

    def test_get_stats_includes_base_keys(self):
        model = DeepHebbianChain(
            input_dim=8,
            hidden_dim=16,
            output_dim=NUM_CLASSES,
            num_layers=3,
            use_spectral_norm=True,
            max_steps=1,
        )
        stats = model.get_stats()
        assert "num_layers" in stats
        assert "lipschitz" in stats
        assert stats["num_layers"] == 3

    def test_train_step_returns_metrics(self):
        model = DeepHebbianChain(
            input_dim=8,
            hidden_dim=16,
            output_dim=NUM_CLASSES,
            num_layers=3,
            use_spectral_norm=False,
            max_steps=1,
        )
        result = model.train_step(
            torch.randn(4, 8), torch.randint(0, NUM_CLASSES, (4,))
        )
        assert isinstance(result, dict)
        assert "loss" in result
        assert "accuracy" in result

    def test_build_classmethod(self):
        from types import SimpleNamespace

        spec = SimpleNamespace(name="deep_hebbian")
        model = DeepHebbianChain.build(
            spec=spec,
            input_dim=8,
            output_dim=NUM_CLASSES,
            hidden_dim=16,
            num_layers=3,
            device="cpu",
            task_type="vision",
        )
        assert isinstance(model, DeepHebbianChain)
        assert model.hebbian_lr == pytest.approx(0.01)
        assert model.use_oja is True

    def test_build_passes_lr_from_kwargs(self):
        """build() must pass learning_rate/lr from kwargs, not hardcode it."""
        from types import SimpleNamespace

        spec = SimpleNamespace(name="deep_hebbian")
        model = DeepHebbianChain.build(
            spec=spec,
            input_dim=8,
            output_dim=NUM_CLASSES,
            hidden_dim=16,
            num_layers=3,
            device="cpu",
            task_type="vision",
            lr=0.123,
        )
        assert model.hebbian_lr == pytest.approx(0.123)

    def test_construct_model_threads_learning_rate(self):
        """construct_model (the sweep's actual path) must thread the sampled
        ``learning_rate`` into ``hebbian_lr``. Without an accepted alias the
        construction layer silently drops it (``hebbian_lr`` falls back to
        default), so two probes with different sampled LRs produce identical
        results — the original silent-LR-discard bug."""
        from bioplausible.core.construction import construct_model

        m = construct_model(
            DeepHebbianChain,
            {
                "input_dim": 8,
                "output_dim": NUM_CLASSES,
                "hidden_dim": 16,
                "num_layers": 3,
                "learning_rate": 0.42,
            },
            input_dim=8,
            output_dim=NUM_CLASSES,
            model_name="deep_hebbian",
        )
        assert m.hebbian_lr == pytest.approx(0.42)

    def test_train_step_updates_spectral_normed_head(self):
        """train_step must update the *original* parameter behind spectral
        norm, not the computed ``weight`` property (which silently discards
        in-place writes)."""
        model = DeepHebbianChain(
            input_dim=8,
            hidden_dim=16,
            output_dim=NUM_CLASSES,
            num_layers=2,
            use_spectral_norm=True,
            hebbian_lr=0.5,
        )
        head_orig = dict(model.head.named_parameters())[
            "parametrizations.weight.original"
        ]
        w_before = head_orig.clone()
        model.train_step(torch.randn(4, 8), torch.randint(0, NUM_CLASSES, (4,)))
        assert not torch.allclose(w_before, head_orig), \
            "Spectral-normed head must receive in-place weight updates"

    def test_train_step_learns_separable_task(self):
        """End-to-end: a shallow DeepHebbianChain should learn a linearly
        separable task to >80% accuracy within 50 steps. This guards against
        silent update-discarding bugs (e.g. writing to a parametrized
        ``weight`` property)."""
        torch.manual_seed(42)
        model = DeepHebbianChain(
            input_dim=32,
            hidden_dim=64,
            output_dim=4,
            num_layers=2,
            hebbian_lr=0.5,
        )
        protos = torch.randn(4, 32) * 3
        y = torch.arange(4).repeat(8)
        x = protos[y] + torch.randn(32, 32) * 0.3
        accs = [model.train_step(x, y)["accuracy"] for _ in range(50)]
        assert accs[-1] > 0.8, f"Did not learn separable task: final acc {accs[-1]:.3f}"

    def test_forward_no_spectral_norm_train(self):
        model = DeepHebbianChain(
            input_dim=8,
            hidden_dim=16,
            output_dim=NUM_CLASSES,
            num_layers=2,
            use_spectral_norm=False,
            max_steps=1,
        )
        model.train()
        x = torch.randn(2, 8)
        out = model(x)
        assert out.shape == (2, NUM_CLASSES)

    def test_forward_multiple_steps(self):
        model = DeepHebbianChain(
            input_dim=8,
            hidden_dim=16,
            output_dim=NUM_CLASSES,
            num_layers=2,
            use_spectral_norm=False,
            max_steps=3,
        )
        x = torch.randn(2, 8)
        out = model(x, steps=2)
        assert out.shape == (2, NUM_CLASSES)

    @pytest.mark.parametrize("num_layers", [1, 3, 5])
    def test_varying_depth(self, num_layers):
        model = DeepHebbianChain(
            input_dim=8,
            hidden_dim=16,
            output_dim=NUM_CLASSES,
            num_layers=num_layers,
            use_spectral_norm=False,
            max_steps=1,
        )
        x = torch.randn(2, 8)
        out = model(x)
        assert out.shape == (2, NUM_CLASSES)

    def test_signal_decay_ratio(self):
        model = DeepHebbianChain(
            input_dim=8,
            hidden_dim=16,
            output_dim=NUM_CLASSES,
            num_layers=5,
            use_spectral_norm=True,
            max_steps=1,
        )
        torch.manual_seed(42)
        x = torch.randn(4, 8)
        result = model.measure_signal_propagation(x)
        assert 0 < result["decay_ratio"] < 1.5


# ============================================================================
# HebbianCube
# ============================================================================


class TestHebbianCube:
    """3D Hebbian lattice with Conv3d."""

    def test_forward_shape(self):
        model = HebbianCube(
            input_dim=8,
            hidden_dim=64,
            output_dim=NUM_CLASSES,
            num_layers=2,
            cube_size=4,
            use_spectral_norm=False,
            max_steps=1,
        )
        x = torch.randn(2, 8)
        out = model(x)
        assert out.shape == (2, NUM_CLASSES)
        assert not out.isnan().any()

    def test_forward_with_spectral_norm(self):
        model = HebbianCube(
            input_dim=8,
            hidden_dim=64,
            output_dim=NUM_CLASSES,
            num_layers=2,
            cube_size=4,
            use_spectral_norm=True,
            max_steps=1,
        )
        x = torch.randn(2, 8)
        out = model(x)
        assert out.shape == (2, NUM_CLASSES)

    def test_forward_larger_batch(self):
        model = HebbianCube(
            input_dim=8,
            hidden_dim=64,
            output_dim=NUM_CLASSES,
            num_layers=2,
            cube_size=4,
            use_spectral_norm=False,
            max_steps=1,
        )
        x = torch.randn(8, 8)
        out = model(x)
        assert out.shape == (8, NUM_CLASSES)

    def test_get_stats_includes_num_layers(self):
        model = HebbianCube(
            input_dim=8,
            hidden_dim=64,
            output_dim=NUM_CLASSES,
            num_layers=3,
            cube_size=4,
            use_spectral_norm=False,
            max_steps=1,
        )
        stats = model.get_stats()
        assert stats["num_layers"] == 3

    def test_cube_architecture(self):
        model = HebbianCube(
            input_dim=8,
            hidden_dim=64,
            output_dim=NUM_CLASSES,
            num_layers=2,
            cube_size=4,
            use_spectral_norm=False,
            max_steps=1,
        )
        assert hasattr(model, "conv_layers")
        assert len(model.conv_layers) == 2
        assert model.cube_size == 4


# ============================================================================
# ThreeFactorHebbian
# ============================================================================


class TestThreeFactorHebbian:
    """Three-factor learning with neuromodulatory signal."""

    def test_forward_shape(self):
        model = ThreeFactorHebbian(
            input_dim=8, hidden_dim=16, output_dim=NUM_CLASSES, num_layers=2
        )
        x = torch.randn(2, 8)
        out = model(x)
        assert out.shape == (2, NUM_CLASSES)
        assert not out.isnan().any()

    def test_train_step_returns_dict(self):
        model = ThreeFactorHebbian(
            input_dim=8, hidden_dim=16, output_dim=NUM_CLASSES, num_layers=2
        )
        x = torch.randn(4, 8)
        y = torch.randint(0, NUM_CLASSES, (4,))
        result = model.train_step(x, y)
        assert isinstance(result, dict)
        assert "loss" in result
        assert "accuracy" in result
        assert isinstance(result["loss"], float)
        assert isinstance(result["accuracy"], float)

    def test_train_step_updates_weights(self):
        model = ThreeFactorHebbian(
            input_dim=8, hidden_dim=16, output_dim=NUM_CLASSES, num_layers=2
        )
        x = torch.randn(4, 8)
        y = torch.randint(0, NUM_CLASSES, (4,))
        w_in_before = model.layers[0].weight.data.clone()
        w_out_before = model.out_layer.weight.data.clone()
        model.train_step(x, y)
        assert not (model.layers[0].weight.data == w_in_before).all()
        assert not (model.out_layer.weight.data == w_out_before).all()

    def test_build_classmethod(self):
        from types import SimpleNamespace

        spec = SimpleNamespace(name="three_factor_hebbian")
        model = ThreeFactorHebbian.build(
            spec=spec,
            input_dim=8,
            output_dim=NUM_CLASSES,
            hidden_dim=16,
            num_layers=2,
            device="cpu",
            task_type="vision",
        )
        assert isinstance(model, ThreeFactorHebbian)

    def test_build_passes_lr_from_kwargs(self):
        """build() must pass learning_rate/lr from kwargs, not hardcode it."""
        from types import SimpleNamespace

        spec = SimpleNamespace(name="three_factor_hebbian")
        model = ThreeFactorHebbian.build(
            spec=spec,
            input_dim=8,
            output_dim=NUM_CLASSES,
            hidden_dim=16,
            num_layers=2,
            device="cpu",
            task_type="vision",
            lr=0.321,
        )
        assert model.lr == pytest.approx(0.321)

    def test_modulator_is_graded_not_binary(self):
        """The three-factor modulator must be a graded (continuous) signal
        ``(y_onehot - softmax(logits))``, not a binary correct/incorrect
        value. A binary modulator gives no error-magnitude information to
        hidden layers."""
        model = ThreeFactorHebbian(
            input_dim=8, hidden_dim=16, output_dim=NUM_CLASSES, num_layers=2
        )
        torch.manual_seed(0)
        x = torch.randn(8, 8)
        y = torch.randint(0, NUM_CLASSES, (8,))
        with torch.no_grad():
            out = model.forward(x)
            pred_probs = torch.softmax(out, dim=1)
            y_onehot = torch.zeros_like(out)
            y_onehot.scatter_(1, y.unsqueeze(1), 1.0)
            modulator = y_onehot - pred_probs

        # Binary modulator would have exactly 2 unique values per sample.
        # A graded one has at least 3 distinct intermediate values.
        n_unique = len(torch.unique(modulator[0].round(decimals=4)))
        assert n_unique >= 3, (
            f"Modulator appears binary (n_unique={n_unique}); must be graded"
        )

    def test_deeper_network(self):
        model = ThreeFactorHebbian(
            input_dim=8, hidden_dim=16, output_dim=NUM_CLASSES, num_layers=4
        )
        # 1 input layer + (num_layers - 1) hidden layers = num_layers total
        assert len(model.layers) == 4
        x = torch.randn(2, 8)
        out = model(x)
        assert out.shape == (2, NUM_CLASSES)

    def test_tuple_input_dim(self):
        model = ThreeFactorHebbian(
            input_dim=(2, 4), hidden_dim=16, output_dim=NUM_CLASSES, num_layers=2
        )
        assert model.layers[0].in_features == 8
        x = torch.randn(2, 8)
        out = model(x)
        assert out.shape == (2, NUM_CLASSES)

    def test_accuracy_tracking(self):
        model = ThreeFactorHebbian(
            input_dim=8, hidden_dim=16, output_dim=NUM_CLASSES, num_layers=2
        )
        x = torch.randn(8, 8)
        y = torch.randint(0, NUM_CLASSES, (8,))
        result = model.train_step(x, y)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_loss_not_nan(self):
        model = ThreeFactorHebbian(
            input_dim=8, hidden_dim=16, output_dim=NUM_CLASSES, num_layers=2
        )
        x = torch.randn(4, 8)
        y = torch.randint(0, NUM_CLASSES, (4,))
        result = model.train_step(x, y)
        assert result["loss"] == result["loss"]  # not NaN
        assert result["loss"] > 0
