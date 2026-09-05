"""D13 — The U-axis is a swap, and local credit × Muon is real.

One coordinate — Feedforward × Instantaneous on the digital substrate —
one swapped ``update`` argument (Euclidean vs Muon-class orthogonalized),
across three credit rules (Backprop, Forward-Forward, PEPITA). Claims:

1. **The headline (user-hypothesis pull, single seed at demo scale):**
   FF×Muon and PEPITA×Muon train to ≈ 0.85 where the same credits on
   Euclidean reach ≈ 0.26 — orthogonalizing the update rescues the local
   rules to BP-grade at this budget. Not studied elsewhere as far as we
   know; multi-seed verification is pending (treat as demo-scale).
2. **The instrument history is part of the claim:** this only works after
   two defects were fixed — (a) the update now orthogonalizes the
   MOMENTUM buffer (raw single-batch orthogonalization amplifies the
   noise floor), and (b) the polar factor comes from SVD (``U @ Vh``):
   reduced QR has a sign-arbitrary R-diagonal, its "orthogonalized"
   direction measured cos ≈ 0 with the gradient and trained at chance.
   Both fixes are locked here (ratchets).

The mechanism ratchet (3) is the load-bearing one: if anyone reverts the
polar factor to QR, or drops the momentum accumulation, this demo goes
flat and the lock fires.
"""

from itertools import islice

import pytest
import torch

from computronium import (
    BackpropCredit,
    CreditAssignmentConfig,
    DigitalSubstrate,
    EuclideanUpdate,
    FeedforwardGeometry,
    GeometryConfig,
    InstantaneousDynamics,
    LocalGoodnessCredit,
    ParameterUpdateConfig,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemState,
    SystemTrainer,
    SystemTrainerConfig,
    compose_system,
    create_task,
)
from computronium.core.pipeline import forward_pass, task_loss
from computronium.ontology.credit import Phase, _learnable_weight_names
from computronium.ontology.update import RiemannianOrthogonalUpdate

WIDTH = 32
DEPTH = 2
BATCH_CAP = 150
LR_EUCLID = 0.2
LR_MUON = 0.02
LOCAL_FLOOR = 0.7
EUCLID_LOCAL_CEILING = (
    0.65  # loader-draw order moves this baseline; the lift is the claim
)


def _flatten(loader, cap):
    for x, y in islice(loader, cap):
        yield x.view(x.size(0), -1), y


_CREDITS = {
    "bp": BackpropCredit,
    "ff": lambda: LocalGoodnessCredit(
        CreditAssignmentConfig.local_goodness(feedback_scale=0.01)
    ),
    "pepita": lambda: LocalGoodnessCredit(
        CreditAssignmentConfig.local_goodness(feedback_scale=0.01)
    ),
}


def _loader():
    task = create_task("mnist", device="cpu", quick_mode=True, num_workers=0)
    task.setup()
    return task.get_dataloader("train")


def test_demo_uaxis_muon_swap(emit_run_record) -> None:
    loader = _loader()
    config = SystemTrainerConfig(max_epochs=1, device="cpu", seed=42)
    torch.manual_seed(0)  # seed BEFORE the loader draw (D8 trap)
    train_data = list(_flatten(loader, BATCH_CAP))

    record: dict = {"arms": {}, "lr_euclid": LR_EUCLID, "lr_muon": LR_MUON}
    accs: dict[str, float] = {}
    for credit_name, make_credit in _CREDITS.items():
        for update_name, update in (
            (
                "euclidean",
                EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=LR_EUCLID)),
            ),
            (
                "muon",
                RiemannianOrthogonalUpdate(
                    ParameterUpdateConfig.riemannian_orthogonal(
                        step_size=LR_MUON, momentum=0.9
                    )
                ),
            ),
        ):
            torch.manual_seed(0)
            geometry = FeedforwardGeometry(
                GeometryConfig.feedforward(
                    input_dim=784, output_dim=10, hidden_dims=(WIDTH,) * DEPTH
                )
            )
            system = compose_system(
                substrate=DigitalSubstrate(SubstrateConfig.digital(device="cpu")),
                geometry=geometry,
                dynamics=InstantaneousDynamics(StateDynamicsConfig.instantaneous()),
                credit=make_credit(),
                update=update,
            )
            acc = SystemTrainer(
                system=system, config=config, train_data=train_data
            ).fit()[-1]["train_acc"]
            key = f"{credit_name}/{update_name}"
            accs[key] = acc
            record["arms"][key] = acc
            print(f"{key:>16}: {acc:.3f}")

    emit_run_record("D13", "uaxis_muon_swap", record)

    assert accs["bp/euclidean"] > 0.5, "BP×Euclidean baseline must learn"
    for local in ("ff", "pepita"):
        assert accs[f"{local}/euclidean"] < EUCLID_LOCAL_CEILING, (
            "local credit on Euclidean must be the weak baseline at this budget"
        )
        assert accs[f"{local}/muon"] > LOCAL_FLOOR, (
            "local credit × Muon must train to BP-grade (the headline)"
        )
        assert accs[f"{local}/muon"] > accs[f"{local}/euclidean"] + 0.2, (
            "the Muon lift over the same credit's Euclidean baseline is the claim"
        )
    assert accs["bp/muon"] > 0.6, "BP×Muon must also learn"


def test_muon_polar_factor_is_descent_aligned() -> None:
    """Ratchet for fix (b): the orthogonalized direction must stay
    positively aligned with the gradient (SVD polar factor). A QR revert
    measures cos ≈ 0 and fires this lock."""
    task = create_task("mnist", device="cpu", quick_mode=True, num_workers=0)
    task.setup()
    loader = task.get_dataloader("train")
    x, y = next(iter(loader))
    x = x.view(x.size(0), -1)
    torch.manual_seed(0)
    geometry = FeedforwardGeometry(
        GeometryConfig.feedforward(input_dim=784, output_dim=10, hidden_dims=(32, 32))
    )
    substrate = DigitalSubstrate(SubstrateConfig.digital(device="cpu"))
    upd = RiemannianOrthogonalUpdate(
        ParameterUpdateConfig.riemannian_orthogonal(step_size=0.1, momentum=0.9)
    )
    with torch.enable_grad():
        state = SystemState(x=x, y=y)
        state.activations = forward_pass(substrate, geometry, x)
        state.loss = task_loss(state, y)
        grads = BackpropCredit().compute_pseudo_gradient(
            {Phase.FREE: state}, state.loss, geometry
        )
    assert len(grads) == len(_learnable_weight_names(geometry.params))
    for name, grad in zip(_learnable_weight_names(geometry.params), grads):
        ortho = upd._orthogonalize(grad.detach())
        cos = torch.nn.functional.cosine_similarity(
            ortho.flatten(), grad.flatten(), dim=0
        ).item()
        assert cos > 0.4, (
            f"{name}: polar factor misaligned with gradient (cos {cos:.3f})"
        )


def test_momentum_buffer_reuse_fails_loud() -> None:
    """Ratchet for fix (a): reusing an update instance across geometries
    with mismatched shapes must raise, not silently corrupt state."""
    torch.manual_seed(0)
    g2 = FeedforwardGeometry(
        GeometryConfig.feedforward(input_dim=784, output_dim=10, hidden_dims=(32, 32))
    )
    g8 = FeedforwardGeometry(
        GeometryConfig.feedforward(input_dim=784, output_dim=10, hidden_dims=(32,) * 8)
    )
    update = EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.1))
    grads2 = [
        torch.randn_like(g2.params[n]) for n in _learnable_weight_names(g2.params)
    ]
    update.step(g2.params, grads2, g2)
    with pytest.raises(RuntimeError, match="reused across different geometries"):
        update.step(
            g8.params,
            [
                torch.randn_like(g8.params[n])
                for n in _learnable_weight_names(g8.params)
            ],
            g8,
        )
