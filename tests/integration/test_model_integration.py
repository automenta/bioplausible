"""Model-in-task integration tests for Phase 1 HPO.

These catch constructor/signature mismatches and input-format defects that unit
tests (which exercise models in isolation with flat inputs) cannot. Each model
registered for a vision task is built via its ``build`` classmethod with the
parameters the HPO search space samples, then driven through the real
``CoreTrainer.from_task`` path (as ``TrialRunner`` does) with 4D image batches.

The trainer's ``_adapt_input`` handles the spatial→flat reshuffle per each
model's declared ``input_format``, so conv models keep 4D while MLP/equilibrium
models receive the flattened ``(B, input_dim)`` they were designed for.

See FIX.md#22.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from bioplausible.cli.run import FAMILY_MAP, _model_compatible
from bioplausible.core.registry import ComponentCategory, Registry
from bioplausible.core.trainer import CoreTrainer
from bioplausible.domains.base import DomainType
from bioplausible.zoo import get_model_spec

BATCH = 8


# The full suite otherwise accumulates cross-test torch/numpy/random state into
# these parametrized cases, making failures order-dependent (PLAN4 S0a). Reset
# all generators before every test so each case is reproducible in isolation.
@pytest.fixture(autouse=True)
def _reset_global_rng():
    import random

    random.seed(0)
    import numpy as np

    np.random.seed(0)
    torch.manual_seed(0)


# Models with non-generic forward interfaces or intentionally excluded types.
EXCLUDED_BUILD = {
    "backprop_transformer_lm",  # LM only (Domain.LM) - not in vision runs
    "custom_stacked_model",  # domains=[] -> excluded everywhere (FIX.md#4b)
    "eqprop_diffusion",  # diffusion denoiser; needs t via its own train_step
    # EquiTile vision/head models expose a specialized constructor (per-tile
    # config) that the generic build()/flat-int input semantics cannot reach; the
    # tiled forward is covered by dedicated fixtures in test_registry_audit.
    "conv_equitile",
    "enhanced_equitile",
}

# Models that do not converge on this toy synthetic task under the standard
# optimizer protocol, by design rather than by breakage (PLAN4 S0a). Each drives
# its own specialized training path or requires a task-specific recipe, so the
# synthetic loss-reduction signal is inapplicable. Deduplicated against the
# EXCLUDED_BUILD set so a name can never appear in both.
NON_CONVERGING_LEARNS = {
    "fabricpc_graph_pcn": "decoupled internal FabricPC trainer, not the test optimizer",
    "spiking_stdp": "LIF+STDP only learns under its own spiking recipe",
    "stochastic_fa": "noisy feedback facade nondeterministic under seed (cf. S0b)",
    "three_factor_hebbian": "neuromodulated STDP requires its specialized protocol",
    "equitile_ep": "tile equilibrium rule does not converge on the toy linear task",
}

REG_FAMILIES = list(FAMILY_MAP.keys())

INPUT_CH = (1, 8, 8)  # digits spatial input (C, H, W)


class _MiniVisionTask:
    """Minimal vision task satisfying the TaskProtocol surface for training."""

    name = "digits"
    quick_mode = True
    task_type = DomainType.VISION
    output_dim = 10
    input_dim = INPUT_CH

    def __init__(self, device: str = "cpu"):
        self.device = device

    def setup(self):
        return None

    def get_batch(self, split: str = "train", batch_size: int = 32):
        dev = self.device
        return torch.randn(batch_size, *INPUT_CH, device=dev), torch.randint(
            0, self.output_dim, (batch_size,), device=dev
        )

    def compute_metrics(self, logits, y, loss):
        acc = (logits.argmax(1) == y).float().mean().item()
        return {"accuracy": acc}

    def create_trainer(self, model, **kwargs):
        return CoreTrainer.from_task(model=model, task=self, **kwargs)


def _vision_models() -> list[str]:
    """All models compatible with the digits task and buildable via build()."""
    out: list[str] = []
    for name in Registry.list(ComponentCategory.MODEL).get("model", []):
        if name in EXCLUDED_BUILD:
            continue
        if _model_compatible(name, "digits"):
            out.append(name)
    return sorted(out)


MODELS = _vision_models()


def _build_model(model_name: str, device: str = "cpu") -> nn.Module:
    spec = get_model_spec(model_name)
    model_cls = Registry.get(ComponentCategory.MODEL, model_name)
    build = getattr(model_cls, "build", None)
    assert build is not None, f"{model_name} has no build()"
    return build(
        spec=spec,
        input_dim=INPUT_CH,
        output_dim=10,
        hidden_dim=32,
        num_layers=2,
        device=device,
        task_type="vision",
    )


@pytest.mark.parametrize("model_name", MODELS)
def test_trainer_forward_vision(model_name, device):
    """A model built from the search-space config runs a training + validation
    pass through the real CoreTrainer path on 4D image batches."""
    model = _build_model(model_name, device).to(device)
    model.train()
    task = _MiniVisionTask(device)
    trainer = CoreTrainer.from_task(
        model=model, task=task, epochs=1, batches_per_epoch=2, track_energy=False
    )
    result = trainer.train_epoch()
    assert isinstance(result, dict), f"{model_name}: train_epoch returned non-dict"
    assert "loss" in result or "val_loss" in result, f"{model_name}: no loss metric"


@pytest.mark.parametrize("model_name", MODELS)
def test_build_and_adapted_forward_vision(model_name, device):
    """Every vision-compatible model produces a [B, 10] output from a 4D batch
    when the trainer's input adapter is applied (mirrors _adapt_input)."""
    model = _build_model(model_name, device).to(device)
    model.eval()
    x = torch.randn(BATCH, *INPUT_CH, device=device)
    # Flatten for non-spatial models exactly as the trainer does.
    if getattr(model, "input_format", "flat") != "spatial":
        x = x.view(x.size(0), -1)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (BATCH, 10), (
        f"{model_name}: expected output {(BATCH, 10)}, got {tuple(out.shape)}"
    )


# Models known to not converge without specialized protocols (see FIX.md#31).
# The test below catches "stuck at random" — a model whose loss does NOT
# decrease over training (signalling missing train_step, broken gradient flow,
# or weight-symmetry issues in eqprop).


@pytest.mark.parametrize("model_name", MODELS)
def test_model_learns_synthetic(model_name, device):
    """Each model must reduce training loss over epochs (i.e., is learning).

    A model that fails to lower its loss at all is structurally broken:
    - eqprop: asymmetric weights block error backpropagation
    - FA variants: feedback alignment requires layer-wise training protocol
    - missing train_step that silently falls to BPTT

    Models in ``NON_CONVERGING_LEARNS`` are *marginal* learners on this toy task
    (their own trainer is decoupled or they need a specialized recipe), not
    broken. They still run the full training here — demonstrating they execute,
    produce finite losses and don't crash — but a missing improvement is allowed
    to soft-fail (xfail) instead of skipping, so they remain visible and counted
    in every run instead of silently vanishing (PLAN4 S0a).
    """
    model = _build_model(model_name, device).to(device)
    task = _LearnableTask(device)
    # Production (hyperopt/experiment.py) resolves a string optimizer name to an
    # Optimizer instance before handing it to CoreTrainer.from_task. The test must
    # mirror that resolution: a bare string would make _bptt_step crash with
    # "'str' object has no attribute 'zero_grad'" (FIX.md#13), falsifying any model.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = CoreTrainer.from_task(
        model=model,
        task=task,
        epochs=5,
        batches_per_epoch=10,
        val_batch_size=32,
        val_batches=2,
        device=device,
        optimizer=optimizer,
        optimizer_kwargs={"lr": 1e-3},
        track_energy=False,
    )
    # Ensure optimizer on model (eqprop's train_step expects it)
    model.optimizer = trainer.optimizer

    model.train()
    first_loss = trainer.train_epoch()["loss"]
    # Train 4 more epochs
    last_loss = first_loss
    for _ in range(4):
        result = trainer.train_epoch()
        last_loss = result["loss"]

    loss_reduction = first_loss - last_loss
    try:
        assert loss_reduction > 1e-4, (
            f"{model_name}: loss did not improve (first={first_loss:.4f}, "
            f"last={last_loss:.4f}). The model is structurally broken — "
            "check train_step / weight symmetry / optimizer attachment."
        )
    except AssertionError:
        if model_name not in NON_CONVERGING_LEARNS:
            raise
        # Known marginal learner: runs, but doesn't reliably reduce loss on this
        # toy task. Tracked so a future improvement flips it to XPASS (signal),
        # while remaining visible/counted in every suite run (PLAN4 S0a note).
        pytest.xfail(
            f"{model_name}: marginal/non-converging on the toy task — "
            f"{NON_CONVERGING_LEARNS[model_name]}"
        )


def test_excluded_models_still_registered():
    """Excluded models must never silently disappear from the registry.

    Every name in ``EXCLUDED_BUILD`` / ``NON_CONVERGING_LEARNS`` is imported to
    guarantee registration (some live behind modules only loaded by other test
    files), then asserted present. If a model is later deleted, this fails — so
    relaxing a test can never double as dropping coverage of a real model.
    """
    from bioplausible.zoo.models.eqprop.looped_mlp import LoopedMLP  # noqa: F401
    from bioplausible.zoo.models.fa import (  # noqa: F401
        FeedbackAlignmentEqProp,
        StochasticFA,
    )
    from bioplausible.zoo.models.hebbian import HebbianCube, ThreeFactorHebbian  # noqa: F401
    from bioplausible.zoo.models.predictive_coding import (  # noqa: F401
        FabricPCGraphPCN,
        PredictiveCodingHybrid,
    )
    from bioplausible.zoo.models.spiking import SpikingSTDP  # noqa: F401
    from bioplausible.equitile.deployments.vision import ConvEquiTile  # noqa: F401
    from bioplausible.equitile._internal.enhanced import EnhancedEquiTile  # noqa: F401
    from bioplausible.equitile.core.model import EquiTileEP  # noqa: F401

    registered = set(Registry.list(ComponentCategory.MODEL).get("model", []))
    everyone = set(EXCLUDED_BUILD) | set(NON_CONVERGING_LEARNS)
    missing = everyone - registered
    assert not missing, (
        "excluded-from-learns models are no longer registered — either restore "
        f"or remove the exclusion explicitly: {sorted(missing)}"
    )


class _LearnableTask:
    """Deterministic learnable task: y = argmax of pixel block (perfectly learnable)."""

    task_type = DomainType.VISION
    output_dim = 10
    input_dim = INPUT_CH
    name = "learnable"

    def __init__(self, device: str = "cpu"):
        self.device = device

    def setup(self):
        return None

    def get_batch(self, split="train", batch_size=64):
        dev = self.device
        x = torch.randn(batch_size, *INPUT_CH, device=dev)
        flat = x.view(batch_size, -1)
        # Target = argmax over 10 disjoint groups of the input's mean activation.
        # Each group mean is a linear function of x, so a linear readout (which
        # every equilibrium/FA model learns) can recover the argmax exactly.
        # A random projection (y = argmax(x @ W_rand)) yields non-linearly-
        # separable class polytopes, so convergence was marginal and RNG-offset
        # dependent (PLAN4 S0a). Group partition is fixed and deterministic.
        group_mean = flat[:, :60].reshape(batch_size, 10, 6).mean(dim=2)
        y = group_mean.argmax(dim=1)
        return x, y

    def compute_metrics(self, logits, y, loss):
        acc = (logits.argmax(1) == y).float().mean().item()
        return {"accuracy": acc}

    def create_trainer(self, model, **kwargs):
        return CoreTrainer.from_task(model=model, task=self, **kwargs)
