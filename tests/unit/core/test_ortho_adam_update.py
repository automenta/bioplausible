"""OrthoAdamUpdate locks: hybrid semantics, dispatch, distinctness.

The learning-algorithm hunt (2026-09-05) measured OrthoAdam — Adam
moments with Muon's SVD-polar orthogonalization applied to matrix-shaped
first-moment directions — beating BOTH parents on three of the four D16
geometries. These locks pin the hybrid's semantics before any coverage
claim uses it.
"""

import pytest
import torch

from computronium import (
    AdamUpdate,
    OrthoAdamUpdate,
    ParameterUpdateConfig,
)
from computronium.ontology.geometry import FeedforwardGeometry, GeometryConfig
from computronium.ontology.utils import _learnable_weight_names


def _geometry():
    return FeedforwardGeometry(
        GeometryConfig.feedforward(input_dim=4, output_dim=2, hidden_dims=(8,))
    )


def test_ortho_adam_matrix_direction_is_orthogonal():
    """The matrix-parameter step must be proportional to the polar factor
    of the first moment — orthogonal rows/cols, rescaled to Adam's step
    magnitude — not to the raw moment."""
    torch.manual_seed(0)
    upd = OrthoAdamUpdate(
        ParameterUpdateConfig.ortho_adam(step_size=1e-3, ortho_lr=3e-3)
    )
    params = {"layer_0_weight": torch.zeros(4, 4)}
    grads = [torch.randn(4, 4) * 0.1]
    out = upd.step(dict(params), grads, _geometry())["layer_0_weight"]
    delta = out - params["layer_0_weight"]
    # rows of an orthogonal matrix are unit vectors (after rescaling, all
    # rows share the same norm): row norms must be equal
    row_norms = delta.norm(dim=1)
    assert torch.allclose(row_norms, row_norms[0].expand_as(row_norms), rtol=1e-4), (
        "OrthoAdam's matrix step must be a rescaled orthogonal direction"
    )
    # rows must be mutually orthogonal
    gram = delta @ delta.T
    off = gram - torch.diag(torch.diag(gram))
    assert off.abs().max() < 1e-5, "orthogonalization must decorrelate rows"


def test_ortho_adam_vector_params_take_plain_adam():
    """Vector parameters (biases, 1-D tensors) must follow plain Adam
    semantics — identical to AdamUpdate at the same step_size."""
    torch.manual_seed(0)
    params = {"layer_0_bias": torch.zeros(8)}
    grads = [torch.randn(8) * 0.1]
    hybrid = OrthoAdamUpdate(ParameterUpdateConfig.ortho_adam(step_size=1e-3))
    adam = AdamUpdate(ParameterUpdateConfig.adam(step_size=1e-3))
    out_h = hybrid.step(dict(params), list(grads), _geometry())["layer_0_bias"]
    out_a = adam.step(dict(params), list(grads), _geometry())["layer_0_bias"]
    assert torch.allclose(out_h, out_a), (
        "vector params must take plain Adam under OrthoAdam"
    )


def test_ortho_adam_state_reuse_fails_loud():
    """Optimizer state is system-scoped (inherited from AdamUpdate)."""
    upd = OrthoAdamUpdate(ParameterUpdateConfig.ortho_adam())
    upd.step({"layer_0_weight": torch.zeros(1, 4)}, [torch.ones(1, 4)], _geometry())
    with pytest.raises(RuntimeError, match="reused across different geometries"):
        upd.step({"layer_0_weight": torch.zeros(1, 8)}, [torch.ones(1, 8)], _geometry())


def test_ortho_adam_is_distinct_from_adam_on_matrices():
    """Same gradient sequence, different trajectories on matrix params —
    the hybrid must never silently alias plain Adam."""
    torch.manual_seed(1)
    names = _learnable_weight_names(_geometry().params)
    wname = next(n for n in names if "weight" in n)
    shape = _geometry().params[wname].shape
    hybrid = OrthoAdamUpdate(ParameterUpdateConfig.ortho_adam(step_size=1e-3))
    adam = AdamUpdate(ParameterUpdateConfig.adam(step_size=1e-3))
    p = {wname: torch.zeros(shape)}
    g = [torch.randn(shape) * 0.1]
    out_h = hybrid.step(dict(p), g, _geometry())[wname]
    out_a = adam.step(dict(p), g, _geometry())[wname]
    assert not torch.allclose(out_h, out_a), (
        "OrthoAdam and Adam must be distinct update rules on matrices"
    )


def test_ortho_adam_dispatch_round_trip():
    """The update must resolve through the spec dispatcher (factory/spec/
    joint wiring) from its config."""
    from computronium.core.system_trainer.spec import _update_from_config

    cfg = ParameterUpdateConfig.ortho_adam(step_size=1e-3, ortho_lr=3e-3)
    upd = _update_from_config(cfg)
    assert isinstance(upd, OrthoAdamUpdate)
    assert upd.config.step_size == pytest.approx(1e-3)
    assert upd.config.ortho_lr == pytest.approx(3e-3)
