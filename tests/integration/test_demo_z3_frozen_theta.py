"""D5 — Frozen θ is a guarantee, bitwise.

The Z3 lifecycle — freeze θ → adapt ψ on task A → switch to task B on a
fresh ψ → restore the stage-A ψ-system → probe — reuses the registered
retention-arm machinery so the demonstration and the registered study run
the same instrument. Two categorical guarantees are watched directly:

1. θ's SHA-256 hash is identical before and after the whole run — the
   freeze is bitwise, not a tolerance band (J2, demonstrated, not just
   locked).
2. The restored ψ-system reproduces stage-A accuracy *exactly* on the fixed
   probe set — a lossless snapshot/restore.

Demonstrated regime (pinned 2026-09-02): compressed Z3 (dims 16, controller
64, 4 meta epochs, 4 adaptation epochs, batch 32) — stage A ≈ 1.0 (chance
0.5 + 0.1 margin), restored == stage A, fresh-ψ floor ≈ 0.68.
"""

import hashlib
import random

import torch
from torch import nn

from computronium.core.utils.device import get_device
from computronium.experiments.joint.z3_fixed_weights import (
    MetaRecipe,
    TaskShape,
    Z3Model,
    _meta_train_phases,
    _run_retention_arm,
    _snapshot,
    create_last_symbol_task,
    create_parity_task,
)
from computronium.visualization import bars_panel, figure_spec

META_EPOCHS = 4
ADAPT_EPOCHS = 4
PROBE_BATCHES = 4


def _theta_sha256(model: Z3Model) -> str:
    raw = (
        model.operator_embeddings.detach().cpu().contiguous().view(-1).view(torch.uint8)
    )
    return hashlib.sha256(bytes(raw.tolist())).hexdigest()


def test_demo_z3_frozen_theta(emit_run_record) -> None:
    torch.manual_seed(42)
    random.seed(42)
    device = get_device("cpu")
    recipe = MetaRecipe()
    model = Z3Model(
        num_operators=8,
        operator_dim=16,
        controller_hidden=64,
        temperature=recipe.temp_start,
    ).to(device)
    tasks = [("parity", create_parity_task), ("last_symbol", create_last_symbol_task)]
    shape = TaskShape(batch_size=32, seq_len=8, input_dim=16, device=device)

    # Freeze → adapt → switch → restore → probe, on the registered machinery.
    _meta_train_phases(
        model,
        tasks,
        nn.CrossEntropyLoss(),
        shape,
        recipe=recipe,
        meta_train_epochs=META_EPOCHS,
    )
    meta_state = _snapshot(model)
    model.freeze_theta()
    hash_before = _theta_sha256(model)
    result = _run_retention_arm(
        model,
        meta_state,
        tasks,
        shape,
        epochs=ADAPT_EPOCHS,
        probe_batches=PROBE_BATCHES,
        recipe=recipe,
        entry_temperature=model.temperature,
    )
    hash_after = _theta_sha256(model)

    record: dict = {
        "theta_sha256_before": hash_before,
        "theta_sha256_after": hash_after,
        **result["retention"],
    }

    # Guarantee 1: bitwise θ invariance across the whole lifecycle.
    assert hash_before == hash_after, "frozen θ moved — bitwise guarantee broken"
    assert result["retention_gate"]["items"]["theta_exact_invariant"]
    # Guarantee 2: the restored ψ-system reproduces stage A exactly.
    retention = result["retention"]
    assert retention["restored"]["task_a_accuracy"] == retention["stage_a"]["accuracy"]
    # The demonstration is visible: stage A is acquired, restoration beats
    # the fresh-ψ floor by the registered margin.
    assert result["retention_gate"]["passed"], result["retention_gate"]["failed"]

    record["figure"] = figure_spec(
        f"D5 — frozen θ is bitwise "
        f"({'identical' if hash_before == hash_after else 'CHANGED'})",
        bars_panel(
            {
                stage: {"fixed-probe accuracy": acc}
                for stage, acc in (
                    ("stage A (adapted)", retention["stage_a"]["accuracy"]),
                    ("restored ψ", retention["restored"]["task_a_accuracy"]),
                    ("fresh-ψ floor", retention["restored"]["fresh_psi_floor"]),
                )
            },
            chance=0.5,
            chance_label="chance (0.5)",
            ylabel="fixed-probe accuracy",
            ylim=(0, 1),
        ),
        figsize=[6, 4],
    )

    emit_run_record("D5", "z3_frozen_theta", record)
