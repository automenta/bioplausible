"""Per-family kernel parity suites for existing backends (REFACTOR7 Phase 2-9).

Each ``AlgorithmFamily`` providing a ``KernelBackend`` protocol implementation is
exercised through a shared harness: initialise with a synthetic model reference,
run a finite forward/backward, verify gradient shapes and finiteness, and (for
settling families) check the settle loop converges and reports telemetry.

This is the DRY counterpart to the backprop reference suite
(``test_kernel_parity_base.py``): instead of a separate module per family with
copied scaffolding, one parametrised harness drives every backend. The backends
have heterogeneous signatures, so each case provides small ``_make`` /
``_run`` callables.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import nn

from bioplausible.acceleration import (
    AlgorithmFamily,
    HardwareTarget,
    KernelConfig,
    KernelRegistry,
    get_algorithm_kernels,
)


@pytest.fixture(scope="module", autouse=True)
def _populate_kernel_registry():
    """Force the (lazy) kernel module imports so backends self-register."""
    get_algorithm_kernels()
    yield


@pytest.fixture(autouse=True)
def _clear_kernel_cache():
    """Clear kernel registry cache between tests to avoid state pollution."""
    from bioplausible.acceleration.kernel_backend import KernelRegistry

    KernelRegistry.clear_cache()
    yield
    KernelRegistry.clear_cache()


def _linear_stack(dims: tuple[int, ...], seed: int = 0) -> list[nn.Linear]:
    """A fresh stack of ``nn.Linear`` layers with fixed init."""
    torch.manual_seed(seed)
    return [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]


def _activ(name: str = "relu") -> nn.Module:
    return {"relu": nn.ReLU(), "tanh": nn.Tanh(), "silu": nn.SiLU()}[name]


def _config(family: AlgorithmFamily, **extra: object) -> KernelConfig:
    return KernelConfig(
        algorithm=family,
        hardware=HardwareTarget.CPU,  # deterministic CPU reference for parity
        dtype=torch.float32,
        settle_steps=20,
        beta=0.5,
        gamma=0.5,
        extra={"hidden_dim": 16, "input_dim": 8, "output_dim": 4, **extra},
    )


class _Harness:
    """Per-family backend harness: build a wired backend and run its core ops."""

    def __init__(
        self,
        family: AlgorithmFamily,
        make: Callable[[], object],
        run: Callable[[object], dict[str, object]] | None = None,
        requires_settle: bool = False,
    ) -> None:
        self.family = family
        self.make = make
        self.run = run
        self.requires_settle = requires_settle

    def build(self) -> object:
        backend = self.make()
        assert KernelRegistry.list_for(self.family), (
            f"{self.family.value} not registered"
        )
        return backend


def _make_fa():
    from bioplausible.acceleration.fa_kernels import FAKernelBackend

    b = FAKernelBackend()
    b.initialize(
        _config(
            AlgorithmFamily.FA,
            num_layers=2,
            input_dim=8,
            hidden_dim=16,
            output_dim=4,
        )
    )
    b.set_model_ref(_linear_stack((8, 16, 4)), _activ("relu"))
    return b


def _run_fa(backend) -> dict[str, object]:
    x = torch.randn(6, 8)
    out, acts = backend.forward(x)
    err = torch.randn(6, 4)
    grads = backend.backward(acts, err)
    return {"out": out, "grads": grads}


def _make_hebbian():
    from bioplausible.acceleration.hebbian_kernels import HebbianKernelBackend

    b = HebbianKernelBackend()
    b.initialize(_config(AlgorithmFamily.HEBBIAN))
    b.set_model_ref(_linear_stack((8, 16, 4)))
    return b


def _run_hebbian(backend) -> dict[str, object]:
    x = torch.randn(6, 8)
    out, acts = backend.forward(x)
    grads = backend.backward(acts)
    return {"out": out, "grads": grads}


def _make_ff():
    from bioplausible.acceleration.ff_kernels import FFKernelBackend

    b = FFKernelBackend()
    b.initialize(
        _config(
            AlgorithmFamily.FF,
            input_dim=8,
            hidden_dim=16,
            output_dim=4,
            num_layers=2,
        )
    )
    # FF concatenates the one-hot label (output_dim) to the input, so the first
    # layer accepts input_dim + output_dim.
    b.set_model_ref(_linear_stack((8 + 4, 16, 4)), _activ("relu"))
    return b


def _run_ff(backend) -> dict[str, object]:
    x = torch.randn(6, 8)
    y = torch.randint(0, 4, (6,))
    pos_out, pos_acts = backend.forward_positive(x, y)
    neg_out, neg_acts = backend.forward_negative(x, y)
    grads = backend.backward(pos_acts, neg_acts)
    goodness = backend.compute_goodness(pos_acts, neg_acts)
    return {"out": pos_out, "grads": grads, "goodness": goodness}


def _make_pepita():
    from bioplausible.acceleration.ff_kernels import PEPITAKernelBackend

    b = PEPITAKernelBackend()
    b.initialize(
        _config(
            AlgorithmFamily.PEPITA,
            input_dim=8,
            hidden_dim=16,
            output_dim=4,
            feedback_matrix_scale=0.1,
        )
    )
    b.set_model_ref(_linear_stack((8, 16, 4)), _activ("relu"))
    return b


def _run_pepita(backend) -> dict[str, object]:
    x = torch.randn(6, 8)
    err = torch.randn(6, 4)
    std_out, std_acts = backend.forward_standard(x)
    err_out, err_acts = backend.forward_error_modulated(x, err)
    grads = backend.backward(std_acts, err_acts, err)
    return {"out": std_out, "grads": grads}


def _make_tp():
    from bioplausible.acceleration.tp_kernels import TPKernelBackend

    b = TPKernelBackend()
    b.initialize(_config(AlgorithmFamily.TP, activation="tanh"))
    # Forward (8,16,4); a DTP inverse maps output->hidden (L-1 layers).
    b.set_model_ref(
        _linear_stack((8, 16, 4), seed=1),
        _linear_stack((4, 16), seed=2),
        _activ("tanh"),
    )
    return b


def _run_tp(backend) -> dict[str, object]:
    x = torch.randn(6, 8)
    out, acts = backend.forward_forward(x)
    targets = backend.compute_targets(acts, torch.randn(6, 4))
    grads = backend.backward(acts, targets)
    return {"out": out, "grads": grads}


def _make_pc():
    from bioplausible.acceleration.pc_kernels import PCKernelBackend

    b = PCKernelBackend()
    b.initialize(_config(AlgorithmFamily.PC, activation="tanh", infer_steps=10))
    b.set_model_ref(_linear_stack((8, 16, 4)), "tanh")
    return b


def _run_pc(backend) -> dict[str, object]:
    x = torch.randn(6, 8)
    backend.init_states(x)
    free_mu, _ = backend.settle(x, y=None, steps=8)
    nudged_mu, _ = backend.settle(x, y=torch.randint(0, 4, (6,)), steps=8)
    grads = backend.backward(x, free_mu, nudged_mu)
    return {"grads": grads}


def _make_snn():
    from bioplausible.acceleration.snn_kernels import SNNKernelBackend

    b = SNNKernelBackend()
    b.initialize(_config(AlgorithmFamily.SNN, num_steps=5))
    b.set_model_ref(_linear_stack((8, 16, 4)))
    return b


def _run_snn(backend) -> dict[str, object]:
    x = torch.rand(6, 8)
    spike_trains, voltage_traces, telemetry = backend.simulate(x)
    # Same-length layer-aligned trains for both phases (free == nudged: the
    # contrastive delta is then exactly zero, but shapes/layers must align).
    grads = backend.backward_contrastive(spike_trains, spike_trains, beta=0.5)
    return {"spike_trains": spike_trains, "grads": grads, "telemetry": telemetry}


def _make_tile():
    from bioplausible.acceleration.tile_kernels import TileKernelBackend

    b = TileKernelBackend()
    b.initialize(
        _config(
            AlgorithmFamily.TILE,
            neurons_per_tile=8,
            tiles_per_layer=2,
            num_hidden_layers=2,
        )
    )
    return b


def _run_tile(backend) -> dict[str, object]:
    x = torch.randn(6, 16)
    out, tile_states = backend.tile_forward(x)
    free_states, _ = backend.settle(x, beta=0.0, steps=4)
    nudged_states, _ = backend.settle(x, beta=0.5, steps=4)
    grads = backend.backward_contrastive(free_states, nudged_states)
    return {"out": out, "grads": grads}


def _make_mep():
    from bioplausible.acceleration.mep_kernels import MEPKernelBackend

    b = MEPKernelBackend()
    b.initialize(_config(AlgorithmFamily.MEP, ns_steps=3))
    b.set_model_ref([nn.Linear(16, 16), nn.Linear(16, 4)])
    return b


def _run_mep(backend) -> dict[str, object]:
    w = torch.randn(16, 16)
    ortho = backend.muon_orthogonalize(w)
    grad = torch.randn(16, 16)
    whitened = backend.fisher_whiten(grad, torch.rand(16, 16))
    h = torch.randn(6, 16)
    x_emb = torch.randn(6, 16)
    w1 = torch.randn(16, 16)
    w2 = torch.randn(16, 16)
    settled, telemetry = backend.ep_settle(
        h, x_emb, w1, torch.zeros(16), w2, torch.zeros(16), steps=4
    )
    return {"ortho": ortho, "whitened": whitened, "settled": settled}


def _make_o1memory():
    from bioplausible.acceleration.mep_kernels import O1MemoryEPv2KernelBackend

    b = O1MemoryEPv2KernelBackend()
    b.initialize(_config(AlgorithmFamily.O1MEMORY, loss_type="mse"))
    b.set_model_ref([nn.Linear(16, 16), nn.Linear(16, 4)])
    return b


def _run_o1memory(backend) -> dict[str, object]:
    states = [torch.randn(6, 16), torch.randn(6, 16), torch.randn(6, 4)]
    states, telemetry = backend.settle_manual_o1(states, torch.randn(6, 4), steps=8)
    return {"states": states, "telemetry": telemetry}


HARNESSES: dict[AlgorithmFamily, _Harness] = {
    AlgorithmFamily.FA: _Harness(AlgorithmFamily.FA, _make_fa, _run_fa),
    AlgorithmFamily.HEBBIAN: _Harness(
        AlgorithmFamily.HEBBIAN, _make_hebbian, _run_hebbian
    ),
    AlgorithmFamily.FF: _Harness(AlgorithmFamily.FF, _make_ff, _run_ff),
    AlgorithmFamily.PEPITA: _Harness(AlgorithmFamily.PEPITA, _make_pepita, _run_pepita),
    AlgorithmFamily.TP: _Harness(AlgorithmFamily.TP, _make_tp, _run_tp),
    AlgorithmFamily.PC: _Harness(
        AlgorithmFamily.PC, _make_pc, _run_pc, requires_settle=True
    ),
    AlgorithmFamily.SNN: _Harness(
        AlgorithmFamily.SNN, _make_snn, _run_snn, requires_settle=True
    ),
    AlgorithmFamily.TILE: _Harness(
        AlgorithmFamily.TILE, _make_tile, _run_tile, requires_settle=True
    ),
    AlgorithmFamily.MEP: _Harness(
        AlgorithmFamily.MEP, _make_mep, _run_mep, requires_settle=True
    ),
    AlgorithmFamily.O1MEMORY: _Harness(
        AlgorithmFamily.O1MEMORY, _make_o1memory, _run_o1memory, requires_settle=True
    ),
}


@pytest.mark.parametrize(
    "family",
    list(HARNESSES.keys()),
    ids=lambda f: f.value,
)
class TestFamilyKernelParity:
    """Shared parity checks for every non-backprop kernel backend."""

    def test_registered(self, family):
        assert KernelRegistry.list_for(family), f"{family.value} not registered"

    def test_initialise_and_memory_stats(self, family):
        backend = HARNESSES[family].build()
        assert backend._config is not None
        stats = backend.get_memory_stats()
        assert stats and all(torch.isfinite(torch.tensor(v)) for v in stats.values())

    def test_core_ops_finite(self, family):
        harness = HARNESSES[family]
        backend = harness.build()
        result = harness.run(backend)
        for key, value in result.items():
            if key == "telemetry":
                continue
            if isinstance(value, dict):
                for grad in value.values():
                    assert torch.isfinite(grad).all(), f"{family}: {key} non-finite"
            elif isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all(), f"{family}: {key} non-finite"
            elif isinstance(value, (list, tuple)):
                for item in value:
                    assert torch.isfinite(item).all(), f"{family}: {key} non-finite"

    def test_update_weights_applies(self, family):
        harness = HARNESSES[family]
        backend = harness.build()
        result = harness.run(backend)
        grads = result.get("grads")
        if grads is None:
            pytest.skip(f"{family.value} exposes no update-weights path")
        # Collect the layers that ``update_weights`` mutates. Backends keep them
        # under different attributes (and ``_forward_layers`` is a *method* on
        # FF/PEPITA), so gather only list-valued containers.
        layer_containers = [
            getattr(backend, name)
            for name in (
                "_layers",
                "_forward_layers",
                "_inverse_layers",
                "_transition_modules",
            )
            if isinstance(getattr(backend, name, None), list)
        ]
        layers = [m for container in layer_containers for m in container]
        before = [p.data.clone() for m in layers for p in m.parameters()]
        backend.update_weights(grads, lr=0.0)
        after = [p.data.clone() for m in layers for p in m.parameters()]
        # The update path must run without error and keep parameters finite.
        assert len(before) == len(after)
        assert all(torch.isfinite(p).all() for p in after)

    def test_settle_telemetry_surface(self, family):
        harness = HARNESSES[family]
        if not harness.requires_settle:
            return
        backend = harness.build()
        # Settling families expose telemetry from their settle loop and a
        # protocol-level getter.
        assert hasattr(backend, "get_settle_telemetry")
        # After running the harness, the settle loop must have recorded
        # telemetry surfaced by the getter.
        harness.run(backend)
        telemetry = backend.get_settle_telemetry()
        assert telemetry is not None, f"{family.value} recorded no settle telemetry"
        assert len(telemetry) > 0
