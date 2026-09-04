"""SystemModule drop-in facade (TODO11 R11.4.1).

A composed 5-D System wrapped as an nn.Module: plain-PyTorch inference
(``forward`` under ``no_grad``/``eval``), internal credit assignment via
``fit_step`` (no optimizer), and ``parameters()`` delegating to geometry.
"""

import torch

from computronium import (
    BackpropCredit,
    DigitalSubstrate,
    EuclideanUpdate,
    FeedforwardGeometry,
    GeometryConfig,
    InstantaneousDynamics,
    ParameterUpdateConfig,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemModule,
    compose_system,
)

_DIM_IN, _DIM_OUT, _HIDDEN = 8, 4, 6


def _module() -> SystemModule:
    torch.manual_seed(0)
    return SystemModule(
        compose_system(
            substrate=DigitalSubstrate(SubstrateConfig.digital(device="cpu")),
            geometry=FeedforwardGeometry(
                GeometryConfig.feedforward(
                    input_dim=_DIM_IN, output_dim=_DIM_OUT, hidden_dims=(_HIDDEN,)
                )
            ),
            dynamics=InstantaneousDynamics(StateDynamicsConfig.instantaneous()),
            credit=BackpropCredit(),
            update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.1)),
        )
    )


def _batch(n: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(n, _DIM_IN)
    y = torch.randint(0, _DIM_OUT, (n,))
    return x, y


def test_forward_is_plain_pytorch_inference() -> None:
    model = _module()
    model.eval()
    x, _ = _batch()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, _DIM_OUT)
    assert model.training is False


def test_parameters_delegate_to_geometry() -> None:
    model = _module()
    params = list(model.parameters())
    geometry_params = list(model.system.geometry.params.values())
    assert len(params) == len(geometry_params) > 0
    assert all(p.requires_grad for p in params)
    assert params[0] is geometry_params[0]


def test_fit_step_updates_theta_and_reports_metrics() -> None:
    model = _module()
    x, y = _batch(8)
    before = {k: v.detach().clone() for k, v in model.system.geometry.params.items()}
    metrics = model.fit_step(x, y)
    assert {"loss", "energy"} <= set(metrics)
    assert any(
        not torch.equal(before[k], v.detach())
        for k, v in model.system.geometry.params.items()
    )


def test_train_mode_propagates_to_geometry() -> None:
    model = _module()
    model.eval()
    assert model.system.geometry.training is False
    model.train()
    assert model.system.geometry.training is True
