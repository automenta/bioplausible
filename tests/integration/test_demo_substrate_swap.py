"""D6 — The substrate axis is physical.

The same coordinate — Feedforward × Instantaneous × Null × Backprop ×
Euclidean — is trained five times on MNIST through identical
``SystemTrainer`` wiring; the only difference between arms is the substrate.
``create_backprop_mlp`` runs on the digital substrate (native execution);
``create_memristive_mlp`` swaps the S-axis to memristive physics — signed
weights realized as differential-pair conductances (every device bounded
in [0, 1] with int8 straight-through quantization, the pair difference
carrying the sign) and activations carrying IR-drop noise of
``noise_level`` standard deviations; ``create_neuromorphic_mlp`` swaps it
to neuromorphic physics — the state thinned to the active spike set each
forward step (each element survives with probability ``1 - sparsity``,
functional spike dropout off the ambient seeded stream). The runner watches
the physics, not a metric table: at mild IR-drop the pair's signed range
learns nearly as well as digital, at severe IR-drop the noise walls
learning at chance; at mild spike dropout the surviving spikes still carry
the signal, at the default 0.95 sparsity the thinning walls it — the
physical constraint, not the algorithm, decides. Each arm's substrate is
probed post-training: the recorded zeros fraction of a ones state is the
dial itself.

Demonstrated regime (re-pinned 2026-09-02 under differential-pair
conductance + functional spike-dropout semantics): MNIST quick-mode train
stream capped at 1000 batches (batch 32), 1 epoch, hidden ``(32,)``,
Euclidean step 0.05, IR-drop noise 1.5 (mild) vs 8.0 (severe), spike
dropout 0.5 (mild) vs 0.95 (severe, the config default) -> accuracy ≈ 0.91
digital / 0.78 memristive-mild / 0.12 memristive-severe / 0.70
neuromorphic-mild / 0.11 neuromorphic-severe (chance 0.1). Live IR-drop
sweep 2026-09-02: noise 0.5 -> 0.89, 3.0 -> 0.56, 6.0 -> 0.22 — the
staircase is monotone; 8.0 walls the arm, so the severe arm's contrast is
categorical by regime choice, not by guard band. (Under the pre-pair
unsigned-clamp semantics the same dial walled at 3.0; the signed pair
doubles the signal path and moved the wall.)
"""

from itertools import islice

import torch

from computronium import (
    SystemTrainer,
    SystemTrainerConfig,
    create_backprop_mlp,
    create_memristive_mlp,
    create_neuromorphic_mlp,
    create_task,
)
from computronium.visualization import bars_panel, figure_spec

BATCH_CAP = 800  # loader cap (Register C): suite walltime, regime re-pinned 2026-09-03 (five arms)
LEARN_FLOOR = 0.5  # constrained arm must learn (5x chance)
WALLED_CEILING = 0.4  # severe arms must visibly fall short of the mild arms


def _flatten(loader, cap):
    for x, y in islice(loader, cap):
        yield x.view(x.size(0), -1), y


def test_demo_substrate_swap(emit_run_record) -> None:
    task = create_task("mnist", device="cpu", quick_mode=True)
    task.setup()
    train_loader = task.get_dataloader("train")
    config = SystemTrainerConfig(max_epochs=1, device="cpu", seed=42)

    arms = (
        ("digital", lambda: create_backprop_mlp(784, (32,), 10, lr=0.05, device="cpu")),
        (
            "memristive_mild",
            lambda: create_memristive_mlp(
                784, (32,), 10, lr=0.05, noise_level=1.5, device="cpu"
            ),
        ),
        (
            "memristive_severe",
            lambda: create_memristive_mlp(
                784, (32,), 10, lr=0.05, noise_level=8.0, device="cpu"
            ),
        ),
        (
            "neuromorphic_mild",
            lambda: create_neuromorphic_mlp(
                784, (32,), 10, lr=0.05, sparsity=0.5, device="cpu"
            ),
        ),
        (
            "neuromorphic_severe",
            lambda: create_neuromorphic_mlp(784, (32,), 10, lr=0.05, device="cpu"),
        ),
    )

    record: dict = {"arms": {}}
    accs: dict[str, float] = {}
    probe = torch.ones(64, 784)
    for name, factory in arms:
        torch.manual_seed(0)
        system = factory()  # the one swapped argument is inside the factory
        metrics = SystemTrainer(
            system=system, config=config, train_data=_flatten(train_loader, BATCH_CAP)
        ).fit()[-1]
        accs[name] = metrics["train_acc"]
        probe_zeros = system.substrate.inject_state_noise(probe).eq(0).float().mean()
        print(
            f"{name}: {metrics['train_acc']:.1%} (probe state zeros {probe_zeros:.2f})"
        )
        record["arms"][name] = {
            "train_acc": metrics["train_acc"],
            "probe_state_zeros": probe_zeros.item(),
        }

    record["figure"] = figure_spec(
        "D6 — one wiring, one swapped substrate (mild physics learns, severe walls)",
        bars_panel(
            {
                name: {"train accuracy": a["train_acc"]}
                for name, a in record["arms"].items()
            },
            chance=1 / 10,
            chance_label="chance (0.1)",
            ylabel="train accuracy",
            ylim=(0, 1),
        ),
        bars_panel(
            {
                name: {"probe state zeros": a["probe_state_zeros"]}
                for name, a in record["arms"].items()
            },
            ylabel="probe state zeros (fraction)",
            title="the dial itself: dropout thins the state, noise does not",
            ylim=(0, 1),
        ),
        figsize=[11, 4],
    )

    emit_run_record("D6", "substrate_swap", record)

    assert accs["digital"] > LEARN_FLOOR, "digital baseline must learn"
    assert accs["memristive_mild"] > LEARN_FLOOR, (
        "memristive physics at mild IR-drop must still learn"
    )
    assert accs["memristive_mild"] < accs["digital"], (
        "the memristive substrate's physics must cost visible accuracy"
    )
    assert accs["memristive_severe"] < WALLED_CEILING, (
        "severe IR-drop must visibly wall learning well below the mild arm"
    )
    assert accs["neuromorphic_mild"] > LEARN_FLOOR, (
        "neuromorphic physics at mild spike dropout must still learn"
    )
    assert accs["neuromorphic_mild"] < accs["digital"], (
        "spike dropout must cost visible accuracy"
    )
    assert accs["neuromorphic_severe"] < WALLED_CEILING, (
        "default spike sparsity must visibly wall learning well below the mild arm"
    )
    assert record["arms"]["neuromorphic_mild"]["probe_state_zeros"] > 0.3, (
        "the neuromorphic substrate's spike dropout must be functional, not cosmetic"
    )
    assert record["arms"]["neuromorphic_severe"]["probe_state_zeros"] > 0.9, (
        "the default sparsity (0.95) must thin the state to the active spike set"
    )
    assert record["arms"]["digital"]["probe_state_zeros"] == 0.0, (
        "the digital substrate must not thin the state"
    )
