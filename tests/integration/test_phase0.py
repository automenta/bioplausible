import warnings

import pytest
import torch
from omegaconf import OmegaConf

from bioplausible.config.schema import RunConfig
from bioplausible.core.energy import EnergyTracker
from bioplausible.core.registry import ComponentCategory, Registry
from bioplausible.core.trainer import run_from_runconfig as run_from_config


# --- 1. Config Loading ---
def test_config_load():
    cfg = OmegaConf.create({
        "seed": 42,
        "device": "cpu",
        "output_dir": "test_results",
        "data": {"task": "mnist", "batch_size": 32, "augment": False},
        "model": {"name": "backprop_mlp", "hidden_dim": 64, "num_layers": 2},
        "optimizer": {"name": "adam", "lr": 0.001},
        "trainer": {"epochs": 1, "batches_per_epoch": 10, "track_energy": True},
    })

    # Validate against schema
    conf = OmegaConf.merge(OmegaConf.structured(RunConfig), cfg)
    assert conf.seed == 42
    assert conf.data.task == "mnist"


# --- 2. Forward-Forward Model ---
def test_forward_forward_train_step():
    from bioplausible.zoo.models.forward_only import ForwardForwardNet

    model = ForwardForwardNet(input_dim=10, hidden_dim=20, output_dim=2, num_layers=2)
    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))

    # Test forward
    out = model(x)
    assert out.shape == (4, 2)

    # Test train_step
    metrics = model.train_step(x, y)
    assert "loss" in metrics
    assert "accuracy" in metrics

    # Check requires_backward metadata
    spec = Registry.get_metadata(ComponentCategory.MODEL, "forward_forward")
    assert not spec.requires_backward


# --- 3. PEPITA Model ---
def test_pepita_train_step():
    from bioplausible.zoo.models.forward_only import PEPITA

    model = PEPITA(input_dim=10, hidden_dim=20, output_dim=2, num_layers=2)
    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))

    # Test forward
    out = model(x)
    assert out.shape == (4, 2)

    # Test train_step
    metrics = model.train_step(x, y)
    assert "loss" in metrics
    assert "accuracy" in metrics

    # Check requires_backward metadata
    spec = Registry.get_metadata(ComponentCategory.MODEL, "pepita")
    assert not spec.requires_backward


# --- 4. Energy Tracking ---
def test_energy_tracking():
    model = torch.nn.Linear(10, 2)
    x = torch.randn(4, 10)

    with EnergyTracker(model, requires_backward=True) as et:
        out = model(x)
        out.sum().backward()

    prof = et.profile
    assert prof is not None
    assert prof.forward_flops > 0
    assert prof.backward_flops > 0
    assert prof.energy_proxy > 0
    assert prof.requires_backward

    # Test backward-free model
    model_nobwd = torch.nn.Linear(10, 2)
    with EnergyTracker(model_nobwd, requires_backward=False) as et_nobwd:
        out = model_nobwd(x)
        # No backward pass here in reality for FF/PEPITA,
        # but tracker just calculates proxy based on flag

    prof_nobwd = et_nobwd.profile
    assert prof_nobwd.backward_flops == 0
    assert not prof_nobwd.requires_backward
    assert (
        prof_nobwd.energy_proxy < prof.energy_proxy
    )  # Should be roughly half (ignoring sparsity)


# --- 5. Run from Config (Integration) ---
@pytest.mark.skipif(
    not torch.cuda.is_available() and False, reason="Run CPU test if needed"
)
def test_integration_run(tmp_path):
    # Use CharNGram for speed/no-download
    cfg = OmegaConf.create({
        "seed": 42,
        "device": "cpu",
        "output_dir": str(tmp_path / "run"),
        "data": {
            "task": "char_ngram",
            "batch_size": 16,
        },
        "model": {
            "name": "backprop_mlp",
            "hidden_dim": 32,
            "num_layers": 1,
        },
        "optimizer": {"name": "adam", "lr": 0.01},
        "trainer": {
            "epochs": 1,
            "batches_per_epoch": 5,
            "track_energy": True,
            "use_compile": False,  # slower for tiny tests
        },
    })

    conf = OmegaConf.merge(OmegaConf.structured(RunConfig), cfg)

    # Isolate the sink so the revert record lands in tmp, not the repo DB.
    from bioplausible.experiment import result_sink

    result_sink.configure(
        kb_path=str(tmp_path / "kb.db"), failure_path=str(tmp_path / "fail.db")
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = run_from_config(conf)

    # The clinical guard prunes this config (constant high-confidence), so the
    # standalone runner must swallow the prune into a failure record rather than
    # propagate optuna.TrialPruned (PLAN4 S0d).
    assert "history" in res
    assert "status" in res
    assert res["status"] in {"completed", "error", "failed", "expensive"}
    if res["status"] != "completed":
        from bioplausible.execution._state import FailureTracker

        tracker = FailureTracker(db_path=str(tmp_path / "fail.db"))
        failures = tracker.get_recent_failures(limit=10)
        assert len(failures) == 1
        assert failures[0].model_name == "backprop_mlp"
    else:
        assert len(res["history"]) == 1
        assert "loss" in res["history"][0]
        assert "energy_proxy" in res["history"][0]
