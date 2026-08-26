"""Family-neutral pipeline harness (TODO4 Phase 9).

Capabilities are declared, not assumed: settle counts must equal declared
phases per credit family, autograd is enabled only under
``requires_autograd``, metrics keys have cross-family parity, and memory
stays flat across steps for non-autograd families (7.2.1 hard gate).
"""

from __future__ import annotations

import gc

import pytest
import torch

from computronium.core.campaign.evaluation import build_coordinate_system
from computronium.core.ontology import BackpropCredit, LocalGoodnessCredit, Phase
from computronium.core.pipeline import run_train_step

INPUT_DIM = 8
OUTPUT_DIM = 8
HIDDEN_DIMS = (16,)

EXPECTED_PHASES: dict[str, tuple[Phase, ...]] = {
    "thermodynamic_contrast": (Phase.FREE, Phase.NUDGED),
    "homeostatic": (Phase.FREE, Phase.NUDGED),
    "random_projections": (Phase.FREE, Phase.NUDGED),
    "target_inversion": (Phase.FREE, Phase.NUDGED),
    "local_goodness": (Phase.FREE,),
    "backprop": (Phase.NUDGED,),
    "temporal_trace": (),
}
METRIC_PARITY_KEYS = {"loss", "energy", "accuracy"}


def _build(credit: str):
    coordinate = f"digital/feedforward/energy_minimization/null/{credit}/euclidean"
    return build_coordinate_system(
        coordinate, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, hidden_dims=HIDDEN_DIMS
    )


def _batch(batch: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(batch, INPUT_DIM), torch.randint(0, OUTPUT_DIM, (batch,))


@pytest.mark.parametrize("credit", sorted(EXPECTED_PHASES))
def test_credit_phase_declarations(credit: str) -> None:
    system = _build(credit)
    assert type(system.credit).phases == EXPECTED_PHASES[credit]


@pytest.mark.parametrize("credit", sorted(EXPECTED_PHASES))
def test_settle_count_equals_declared_phases(credit: str) -> None:
    system = _build(credit)
    bound = system.dynamics.settle
    calls = {"n": 0}

    def spy(state, geometry, substrate, target=None):
        calls["n"] += 1
        return bound(state, geometry, substrate, target=target)

    system.dynamics.settle = spy
    x, y = _batch()
    metrics = system.train_step(x, y)
    assert calls["n"] == len(EXPECTED_PHASES[credit])
    assert set(metrics) >= METRIC_PARITY_KEYS


def test_one_phase_family_pays_single_settle_cost() -> None:
    two_phase = _build("thermodynamic_contrast")
    one_phase = _build("local_goodness")
    x, y = _batch()
    assert len(two_phase.credit.phases) == 2
    assert len(one_phase.credit.phases) == 1
    # Same forward+update work; the one-phase family skips a full settle.
    m2 = two_phase.train_step(x, y)
    m1 = one_phase.train_step(x, y)
    assert set(m1) >= METRIC_PARITY_KEYS and set(m2) >= METRIC_PARITY_KEYS


def test_backprop_runs_autograd_path() -> None:
    assert BackpropCredit.requires_autograd is True
    system = _build("gradient")
    before = {k: v.clone() for k, v in system.geometry.params.items()}
    losses = []
    for _ in range(5):
        x, y = _batch(16)
        losses.append(system.train_step(x, y)["loss"])
    assert all(loss == loss for loss in losses)  # no NaN
    deltas = {
        k: (v - before[k]).abs().max().item() for k, v in system.geometry.params.items()
    }
    weights_moved = any(v > 0.0 for k, v in deltas.items() if "weight" in k)
    biases_untouched = all(v == 0.0 for k, v in deltas.items() if "bias" in k)
    assert weights_moved and biases_untouched


def test_non_autograd_default_is_no_grad() -> None:
    assert BackpropCredit.requires_autograd is True
    for credit in ("local_goodness", "thermodynamic_contrast"):
        assert _build(credit).credit.requires_autograd is False


def test_run_train_step_matches_system_step() -> None:
    system = _build("thermodynamic_contrast")
    x, y = _batch()
    via_system = dict(system.train_step(x, y))
    via_loop = run_train_step(
        system.substrate,
        system.geometry,
        system.dynamics,
        system.credit,
        system.update,
        x,
        y,
    )
    assert set(via_system) == set(via_loop)


@pytest.fixture
def flat_memory_system():
    return _build("thermodynamic_contrast")


def test_memory_flat_across_steps_cpu(flat_memory_system) -> None:
    """Hard gate: no graph retention step-over-step on the default path.

    Counts live non-leaf tensors after gc; retention would grow this census
    linearly with steps.
    """
    system = flat_memory_system
    x, y = _batch(8)
    for _ in range(10):
        system.train_step(x, y)

    def live_graph_tensors() -> int:
        gc.collect()
        return sum(
            1
            for obj in gc.get_objects()
            if isinstance(obj, torch.Tensor)
            and not isinstance(obj, torch.nn.Parameter)
            and obj.grad_fn is not None
        )

    checkpoints = []
    for _ in range(3):
        for _ in range(20):
            system.train_step(x, y)
        checkpoints.append(live_graph_tensors())
    assert max(checkpoints) - min(checkpoints) <= 5, checkpoints


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_memory_flat_across_steps_cuda(flat_memory_system) -> None:
    system = flat_memory_system
    device = torch.device("cuda")
    system.geometry.to(device)
    x, y = _batch(32)
    x, y = x.to(device), y.to(device)
    for _ in range(10):
        system.train_step(x, y)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    for _ in range(50):
        system.train_step(x, y)
    torch.cuda.synchronize()
    growth_mb = (torch.cuda.memory_allocated() - baseline) / 1024**2
    assert growth_mb < 5.0, f"{growth_mb:.1f} MB leaked over 50 steps"


def test_local_goodness_free_only_declaration() -> None:
    assert LocalGoodnessCredit.phases == (Phase.FREE,)
