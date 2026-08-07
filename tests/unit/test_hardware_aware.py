"""Tests for hardware-aware learning-rule measurement (plan §17).

Covers the substrate-faithful model facades (``QuantizedLoopedMLP`` /
``NoisyLoopedMLP``), the ``TrainerConfig.target_hardware`` knob that swaps them
into ``CoreTrainer``, and the result-sink wiring for the validation hardware
tracks.
"""

import tempfile

import pytest
import torch

import bioplausible.zoo  # ruff: ignore[unused-import]  (populates the model registry)
from bioplausible.core.registry import ComponentCategory, Registry
from bioplausible.core.trainer import CoreTrainer, TrainerConfig
from bioplausible.experiment.result_sink import configure as sink_configure
from bioplausible.zoo.models.eqprop import (
    LoopedMLP,
    NoisyLoopedMLP,
    QuantizedLoopedMLP,
)


def _kpw(**kwargs):
    """Constructor kwargs a LoopedMLP family needs to build."""
    return {"input_dim": 32, "hidden_dim": 16, "output_dim": 4, **kwargs}


@pytest.fixture
def sink_paths():
    with tempfile.TemporaryDirectory() as tmpdir:
        sink_configure(kb_path=f"{tmpdir}/kb.db", failure_path=f"{tmpdir}/fail.db")
        yield tmpdir
        sink_configure()


def test_hardware_variants_are_registered_models():
    """The facades are first-class registry members, not track-local hacks."""
    q = Registry.get(ComponentCategory.MODEL, "quantized_looped_mlp")
    n = Registry.get(ComponentCategory.MODEL, "noisy_looped_mlp")
    assert issubclass(q, QuantizedLoopedMLP) and issubclass(q, LoopedMLP)
    assert issubclass(n, NoisyLoopedMLP) and issubclass(n, LoopedMLP)
    # The validation-tracks module re-exports the SAME registered classes.
    from bioplausible.validation.tracks.hardware_tracks import (
        NoisyLoopedMLP as TrackN,
    )
    from bioplausible.validation.tracks.hardware_tracks import (
        QuantizedLoopedMLP as TrackQ,
    )

    assert TrackQ is QuantizedLoopedMLP and TrackN is NoisyLoopedMLP


def test_quantized_step_bounds_state_to_unit_scale():
    """FPGA facade's quantized step keeps the hidden state in [-1, 1]."""
    model = QuantizedLoopedMLP(**_kpw())
    x = model._transform_input(torch.randn(8, 32))
    h = torch.randn(8, 16) * 3.0
    out = model._forward_step_impl(h, x)
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_noisy_step_injects_stochastic_noise():
    """Analog facade's step is stochastic (noise differs every call)."""
    model = NoisyLoopedMLP(**_kpw(), noise_level=0.05)
    x = model._transform_input(torch.randn(8, 32))
    h = torch.randn(8, 16)
    outs = {model._forward_step_impl(h, x) for _ in range(3)}
    assert len(outs) > 1  # runs with fresh noise differ


@pytest.mark.parametrize(
    ("target", "expected_cls", "meta_key"),
    [
        ("fpga", QuantizedLoopedMLP, "bits"),
        ("analog", NoisyLoopedMLP, "noise_level"),
    ],
)
def test_target_hardware_swaps_eqprop_model(target, expected_cls, meta_key):
    """The knob swaps an eqprop LoopedMLP for the substrate facade."""
    cfg = TrainerConfig(
        model="eqprop_mlp",
        model_kwargs=_kpw(),
        task="mnist",
        device="cpu",
        target_hardware=target,
    )
    trainer = CoreTrainer(cfg)
    trainer._create_model()
    assert isinstance(trainer.model, expected_cls)
    assert trainer._hardware_meta["target_hardware"] == target
    assert meta_key in trainer._hardware_meta


def test_target_hardware_none_is_inert():
    """No knob / gpu target leaves the base model untouched."""
    for target in (None, "gpu"):
        cfg = TrainerConfig(
            model="eqprop_mlp",
            model_kwargs=_kpw(),
            task="mnist",
            device="cpu",
            target_hardware=target,
        )
        trainer = CoreTrainer(cfg)
        trainer._create_model()
        assert type(trainer.model) is LoopedMLP
        assert trainer._hardware_meta == {}


def test_target_hardware_inert_for_non_looped_model():
    """A non-equilibrium model is not swapped (no substrate defined)."""
    cfg = TrainerConfig(
        model="backprop_mlp",
        model_kwargs=_kpw(),
        task="mnist",
        device="cpu",
        target_hardware="fpga",
    )
    trainer = CoreTrainer(cfg)
    trainer._create_model()
    # backprop_mlp is BackpropMLP (a plain MLP, not a LoopedMLP) → no swap.
    assert not isinstance(trainer.model, LoopedMLP)


def test_sink_wires_hardware_track_into_kb(sink_paths):
    """A completed hardware-track result lands in the KnowledgeBase."""
    from bioplausible.knowledge.kb import KnowledgeBase
    from bioplausible.validation.notebook import TrackResult
    from bioplausible.validation.tracks.hardware_tracks import _sink_hardware_track

    result = TrackResult(
        track_id=16,
        name="FPGA Bit Precision",
        status="pass",
        score=90.0,
        metrics={"accuracy": 0.98, "bits": 8},
        evidence="x",
        time_seconds=1.0,
    )
    _sink_hardware_track(
        result=result, model="quantized_looped_mlp", task="synthetic", hardware="fpga"
    )

    kb = KnowledgeBase(db_path=f"{sink_paths}/kb.db", auto_embed=False)
    exp = list(kb.query(model_family="quantized_looped_mlp"))
    assert len(exp) == 1
    assert exp[0].metrics["accuracy"] == pytest.approx(0.98)


def test_sink_wires_failed_hardware_track_into_failures(sink_paths):
    """A failed hardware-track result lands in the FailureTracker."""
    from bioplausible.execution._state import FailureTracker
    from bioplausible.validation.notebook import TrackResult
    from bioplausible.validation.tracks.hardware_tracks import _sink_hardware_track

    result = TrackResult(
        track_id=17,
        name="Analog Noise",
        status="fail",
        score=0.0,
        metrics={"accuracy": 0.1},
        evidence="x",
        time_seconds=1.0,
    )
    _sink_hardware_track(
        result=result, model="noisy_looped_mlp", task="synthetic", hardware="analog"
    )

    ft = FailureTracker(db_path=f"{sink_paths}/fail.db")
    assert ft.get_failure_stats()["total_failures"] == 1


class _FakeDriver:
    """Records calls; returns a fixed, config-independent metric."""

    def __init__(self):
        self.calls = 0

    def train(self, *, model, task, config, seed, epochs, device):
        self.calls += 1
        return {
            "final_acc": 0.95,
            "forward_flops": 100,
            "backward_flops": 50,
            "peak_memory_mb": 20.0,
            "wall_time_s": 3.0,
        }


def test_target_hardware_is_part_of_frontier_cache_identity(tmp_path):
    """GPU and FPGA frontiers cache under distinct keys (plan §17 / §16.3)."""
    from bioplausible.hyperopt.ideal_backprop import IdealBackpropFinder

    gpu = IdealBackpropFinder(
        _FakeDriver(),
        task="mnist",
        budget_probes=4,
        epochs=1,
        seed=0,
        cache_dir=str(tmp_path),
        target_hardware=None,
    )
    fpga = IdealBackpropFinder(
        _FakeDriver(),
        task="mnist",
        budget_probes=4,
        epochs=1,
        seed=0,
        cache_dir=str(tmp_path),
        target_hardware="fpga",
    )
    assert gpu._cache_name() != fpga._cache_name()
    assert "hwfpga" in fpga._cache_name()

    gpu.find()
    fpga.find()
    files = {p.name for p in tmp_path.iterdir()}
    assert len([f for f in files if f.startswith("ideal_backprop_mnist_")]) >= 2
    # A fresh FPGA finder must NOT reuse the GPU cache (distinct keys → retrain).
    fresh_fpga = IdealBackpropFinder(
        _FakeDriver(),
        task="mnist",
        budget_probes=4,
        epochs=1,
        seed=0,
        cache_dir=str(tmp_path),
        target_hardware="fpga",
    )
    cached = fresh_fpga.load_cache()
    assert cached is not None and cached.target_hardware == "fpga"
