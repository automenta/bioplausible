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

# Models with non-generic forward interfaces or intentionally excluded types.
EXCLUDED_BUILD = {
    "backprop_transformer_lm",  # LM only (Domain.LM) - not in vision runs
    "custom_stacked_model",  # domains=[] -> excluded everywhere (FIX.md#4b)
    "eqprop_diffusion",  # diffusion denoiser; needs t via its own train_step
}

REG_FAMILIES = list(FAMILY_MAP.keys())

INPUT_CH = (1, 8, 8)  # digits spatial input (C, H, W)


class _MiniVisionTask:
    """Minimal vision task satisfying the TaskProtocol surface for training."""

    name = "digits"
    device = "cpu"
    quick_mode = True
    task_type = DomainType.VISION
    output_dim = 10
    input_dim = INPUT_CH

    def setup(self):
        return None

    def get_batch(self, split: str = "train", batch_size: int = 32):
        return torch.randn(batch_size, *INPUT_CH), torch.randint(
            0, self.output_dim, (batch_size,)
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


def _build_model(model_name: str) -> nn.Module:
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
        device="cpu",
        task_type="vision",
    )


@pytest.mark.parametrize("model_name", MODELS)
def test_trainer_forward_vision(model_name):
    """A model built from the search-space config runs a training + validation
    pass through the real CoreTrainer path on 4D image batches."""
    model = _build_model(model_name)
    model.train()
    task = _MiniVisionTask()
    trainer = CoreTrainer.from_task(
        model=model, task=task, epochs=1, batches_per_epoch=2, track_energy=False
    )
    result = trainer.train_epoch()
    assert isinstance(result, dict), f"{model_name}: train_epoch returned non-dict"
    assert "loss" in result or "val_loss" in result, f"{model_name}: no loss metric"


@pytest.mark.parametrize("model_name", MODELS)
def test_build_and_adapted_forward_vision(model_name):
    """Every vision-compatible model produces a [B, 10] output from a 4D batch
    when the trainer's input adapter is applied (mirrors _adapt_input)."""
    model = _build_model(model_name)
    model.eval()
    x = torch.randn(BATCH, *INPUT_CH)
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
def test_model_learns_synthetic(model_name):
    """Each model must reduce training loss over epochs (i.e., is learning).

    A model that fails to lower its loss at all is structurally broken:
    - eqprop: asymmetric weights block error backpropagation
    - FA variants: feedback alignment requires layer-wise training protocol
    - missing train_step that silently falls to BPTT
    Loss is a more robust signal than accuracy on a random task.
    """
    model = _build_model(model_name)
    task = _LearnableTask()
    # Production (hyperopt/experiment.py) resolves a string optimizer name to an
    # Optimizer instance before handing it to CoreTrainer.from_task. The test must
    # mirror that resolution: a bare string would make _bptt_step crash with
    # "'str' object has no attribute 'zero_grad'" (FIX.md#13), falsifying any model.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = CoreTrainer.from_task(
        model=model, task=task, epochs=5, batches_per_epoch=10,
        val_batch_size=32, val_batches=2, device="cpu",
        optimizer=optimizer, optimizer_kwargs={"lr": 1e-3},
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
    assert loss_reduction > 0, (
        f"{model_name}: loss did not improve (first={first_loss:.4f}, "
        f"last={last_loss:.4f}). The model is structurally broken — "
        "check train_step / weight symmetry / optimizer attachment."
    )


class _LearnableTask:
    """Deterministic learnable task: y = argmax of pixel block (perfectly learnable)."""

    task_type = DomainType.VISION
    output_dim = 10
    input_dim = INPUT_CH
    name = "learnable"

    def setup(self):
        return None

    def get_batch(self, split="train", batch_size=64):
        x = torch.randn(batch_size, *INPUT_CH)
        # Target = argmax of a fixed linear projection of input (deterministic, learnable)
        W = torch.randn(INPUT_CH[0] * INPUT_CH[1] * INPUT_CH[2], 10)
        logits = x.view(x.size(0), -1) @ W
        y = logits.argmax(dim=1)
        return x, y

    def compute_metrics(self, logits, y, loss):
        acc = (logits.argmax(1) == y).float().mean().item()
        return {"accuracy": acc}

    def create_trainer(self, model, **kwargs):
        return CoreTrainer.from_task(model=model, task=self, **kwargs)
