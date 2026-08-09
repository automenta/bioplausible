"""DirectedEP feedback-pathway tests (Plan 8 Track D3).

DirectedEP (``variant="feedback"``) adds output→hidden feedback layers
``W_fb``. These tests verify the feedback pathway exists for depth ≥ 2, that
feedback layers are trainable (receive contrastive updates), and that feedback
influences the deep-layer contrastive signal (the Plan 8 salvage hypothesis).
"""


import torch

from bioplausible.core.config import ModelConfig
from bioplausible.zoo.models.eqprop import DirectedEP, StandardEqProp


def _make_config(num_hidden: int = 3, **overrides) -> ModelConfig:
    defaults = {
        "name": "test",
        "input_dim": 10,
        "output_dim": 5,
        "hidden_dims": [20] * num_hidden,
        "max_steps": 3,
        "extra": {"contrastive_diagnostics": True, "gradient_method": "contrastive"},
    }
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _get_weight_grad_norms(model) -> list[tuple[str, float]]:
    """Collect (param_name, grad_norm) for parameters with a grad set."""
    out = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            out.append((name, p.grad.norm().item()))
    return out


def test_feedback_layers_exist_for_depth_two():
    """DirectedEP builds one feedback layer per hidden layer (depth ≥ 2)."""
    model = DirectedEP(config=_make_config(num_hidden=3))
    fb_layers = list(model.feedback_layers)
    assert len(fb_layers) == 3
    # Each feedback layer maps output -> hidden width.
    for layer in fb_layers:
        assert layer.in_features == model.config.output_dim
        assert layer.out_features in model.config.hidden_dims


def test_plain_eqprop_has_no_feedback_layers():
    """Vanilla EqProp must NOT have feedback layers (clean contrast)."""
    model = StandardEqProp(config=_make_config(num_hidden=2))
    assert not hasattr(model, "feedback_layers")


def test_feedback_layers_receive_gradients():
    """Feedback weights get a contrastive update during train_step."""
    torch.manual_seed(0)
    model = DirectedEP(config=_make_config(num_hidden=2))
    x = torch.randn(4, 10)
    y = torch.randint(0, 5, (4,))
    model.train_step(x, y)

    trained_fb = [
        name for name, _ in _get_weight_grad_norms(model) if "feedback" in name
    ]
    assert any("feedback_layers" in n for n in trained_fb), (
        "feedback_layers must receive gradients during the contrastive update"
    )


def test_feedback_keeps_early_layer_signal_alive():
    """DirectedEP early-layer post-state deltas exceed vanilla EqProp's.

    The Plan 8 salvage hypothesis: an explicit output→hidden feedback drive
    prevents the vanishing contrastive signal seen in deep vanilla EqProp.
    This is a mechanism check, not a precision claim: with a profiler-like
    configuration (spectral norm, convergence detection, 10 training steps) the
    feedback pathway must move early-layer states measurably more than the
    vanilla rule on a depth-4 model.
    """
    torch.manual_seed(1)
    cfg = ModelConfig(
        name="test",
        input_dim=10,
        output_dim=5,
        hidden_dims=[20] * 4,
        max_steps=10,
        use_spectral_norm=True,
        beta=0.1,
        learning_rate=0.05,
        extra={
            "contrastive_diagnostics": True,
            "gradient_method": "contrastive",
            "convergence_threshold": 1e-3,
            "convergence_start": 5,
        },
    )
    van = StandardEqProp(config=cfg)
    dir_ = DirectedEP(config=cfg)

    x = torch.randn(8, 10)
    y = torch.randint(0, 5, (8,))

    van_delta = dir_delta = 0.0
    for _ in range(10):
        van_res = van.train_step(x, y)
        dir_res = dir_.train_step(x, y)
        van_delta += van_res["layer_diagnostics"][0]["post_state_delta_norm"]
        dir_delta += dir_res["layer_diagnostics"][0]["post_state_delta_norm"]
    van_layer0_delta = van_delta / 10
    dir_layer0_delta = dir_delta / 10
    # Loose margin (>1.5x): the exact factor depends on feedback gain/beta, but
    # the feedback pathway should move early-layer states measurably more than
    # the vanilla rule at depth ≥ 4.
    assert dir_layer0_delta > van_layer0_delta * 1.5, (
        f"feedback layer0 delta={dir_layer0_delta:.4g} "
        f"vs vanilla {van_layer0_delta:.4g} — feedback not restoring signal?"
    )


def test_feedback_gain_scales_hidden_state_drive():
    """feedback_gain scales the feedback drive injected during the nudge.

    The drive on hidden layer i is ``beta * feedback_gain * fb_i``; a larger
    ``feedback_gain`` must produce a larger nudged-vs-free hidden-state delta
    at a fixed beta. Measure the *first hidden layer* delta — the output layer
    is dominated by the direct ``beta * (target - out)`` nudge, while the
    hidden layer delta is the clean readout of the feedback pathway.
    """
    torch.manual_seed(7)
    x = torch.randn(4, 10)
    y = torch.randint(0, 5, (4,))

    def _hidden0_delta(feedback_gain: float) -> float:
        cfg = ModelConfig(
            name="fb",
            input_dim=10,
            output_dim=5,
            hidden_dims=[20, 20, 20],
            max_steps=5,
            beta=0.1,
            extra={"gradient_method": "contrastive", "feedback_gain": feedback_gain},
        )
        model = DirectedEP(config=cfg)
        from bioplausible.zoo.models.eqprop._contrastive import _make_onehot_target

        target = _make_onehot_target(y, model.config.output_dim)
        acts = model._initial_activations(x)
        with torch.no_grad():
            free = model.forward_dynamics(acts, beta=0.0, target=None)
            nudged = model.forward_dynamics(acts, beta=0.1, target=target)
        # Layer index 1 == first hidden state h_0.
        return (nudged[1] - free[1]).norm().item()

    low = _hidden0_delta(0.1)
    high = _hidden0_delta(10.0)
    # 100x gain must move the first-hidden-layer state measurably more (the
    # exact factor is bounded by tanh saturation of hidden activations).
    assert high > low * 2, (
        f"feedback_gain should scale the hidden drive: low={low:.4g} high={high:.4g}"
    )


def test_directed_ep_diagnostics_include_feedback_bounds():
    """Feedback diagnostics do not crash stepping for all depths."""
    for depth in (1, 2, 4):
        model = DirectedEP(config=_make_config(num_hidden=depth))
        x = torch.randn(4, 10)
        y = torch.randint(0, 5, (4,))
        res = model.train_step(x, y)
        assert len(res["layer_diagnostics"]) == depth + 1  # hidden + output
