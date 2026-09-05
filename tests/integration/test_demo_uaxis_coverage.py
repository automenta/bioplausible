"""D16 — The U-axis coverage map: Muon's lift generalizes across geometry.

The U-axis was demonstrated on feedforward (D13) and on depth-frontier
MLPs (D15). This demo completes the map the ontology claims: the same
capacity-identical update swap (same geometry, swapped update rule —
parameter counts identical within each geometry) across FOUR geometries
× FOUR update rules, mnist quick, 150 batches, 1 epoch, seeds 0–2, TEST
accuracy:

1. **Muon ≥ Euclidean on every geometry** (mean lift asserted per
   geometry): ff +0.068, attention +0.038, graph +0.102, lattice +0.050.
   The orthogonalized-update lift is not an MLP artifact.
2. **The optimizer family is a first-class coordinate (adam column,
   2026-09-05):** Adam BEATS Muon on attention (0.900 ± 0.001 vs 0.874)
   while Muon keeps ff, graph, and lattice — NO update rule dominates
   the map. The earlier "Muon is the robust default" reading was drawn
   inside the SGD family only; Euclidean, Adam, and Muon are three
   distinct optimizer families, not step-size variants. Adam also lifts
   the local rule to parity on ff (ff×Adam 0.899 vs ff×Muon 0.896).
3. **Spectral-constrained is geometry-conditioned:** it matches Muon on
   attention (0.879 vs 0.874) but never beats it and clearly lags on
   ff, graph, and lattice.
4. **Natural gradient is at chance at this budget on ff/graph/lattice**
   and unstable on attention (0.38 ± 0.11) — asserted as an upper bound,
   an OPEN budget/regime question, not a physics verdict (R11.5.5a).
5. **The local reference trails the global rule here:** ff-credit ×
   Muon < bp × Muon on all four geometries at this budget (contrast
   D13/D15's parity at their regimes) — except on Adam × ff, where the
   local rule reaches the global rule (0.899 vs 0.892).

Capacity-matched per the D8–D12 convention: 47.7k–57.5k params across
geometries (max/min < 1.25, asserted); identical within each geometry.
LR note: each family uses its own working lr (SGD 0.1, Muon 0.02, Adam
1e-3) — the comparison is per-family best-effort, not equal-lr.

Naming convention: "ff/" prefix = Forward-Forward credit (the repo's
D13 convention, `local_objective="ff"`); "mlp_d2_w64" = the two-layer
width-64 feedforward geometry ("mlp" avoids colliding with the credit
abbreviation — both were "ff" in the first draft of this map).
GraphGeometry contract: node dim == batch dim (D9 semantics) — each
32-sample batch is one graph over an 8×4 grid of batch positions.
"""

from itertools import islice
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

import pytest
import torch

from computronium import (
    AdamUpdate,
    AttentionGeometry,
    BackpropCredit,
    CreditAssignmentConfig,
    DigitalSubstrate,
    EuclideanUpdate,
    FeedforwardGeometry,
    GeometryConfig,
    GraphGeometry,
    InstantaneousDynamics,
    LocalGoodnessCredit,
    NaturalGradientUpdate,
    ParameterUpdateConfig,
    SpatialLattice3DGeometry,
    SpectralConstrainedUpdate,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemState,
    SystemTrainer,
    SystemTrainerConfig,
    compose_system,
    create_task,
)
from computronium.ontology.update import RiemannianOrthogonalUpdate
from computronium.visualization._demo_api import figure_spec, heatmap_panel

BATCH_CAP = 150
SEEDS = (0, 1, 2)
CHANCE = 0.1


def _grid_edges(h: int = 8, w: int = 4) -> list[list[int]]:
    src: list[int] = []
    dst: list[int] = []
    for r in range(h):
        for c in range(w):
            i = r * w + c
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    j = nr * w + nc
                    src.extend((i, j))
                    dst.extend((j, i))
    return [src, dst]


def _geometries() -> dict[str, Callable]:
    return {
        "mlp_d2_w64": lambda: FeedforwardGeometry(
            GeometryConfig.feedforward(
                input_dim=784, output_dim=10, hidden_dims=(64, 64)
            )
        ),
        "attention": lambda: AttentionGeometry(
            GeometryConfig.attention(
                input_dim=784, output_dim=10, hidden_dim=32, num_layers=2, num_heads=4
            )
        ),
        "graph_grid8x4": lambda: GraphGeometry(
            GeometryConfig.graph(
                input_dim=784,
                output_dim=10,
                edge_index=_grid_edges(),
                hidden_dims=(56, 56),
            )
        ),
        "lattice3d": lambda: SpatialLattice3DGeometry(
            GeometryConfig.spatial_lattice(
                input_dim=784,
                output_dim=10,
                lattice_dims=(4, 3, 3),
                hidden_dims=(2,),
                connectivity_radius=1,
            )
        ),
    }


def _updates() -> dict[str, Callable]:
    return {
        "euclid": lambda: EuclideanUpdate(
            ParameterUpdateConfig.euclidean(step_size=0.1)
        ),
        "muon": lambda: RiemannianOrthogonalUpdate(
            ParameterUpdateConfig.riemannian_orthogonal(step_size=0.02, momentum=0.9)
        ),
        "spectral": lambda: SpectralConstrainedUpdate(
            ParameterUpdateConfig.spectral_constrained(step_size=0.1)
        ),
        "natural": lambda: NaturalGradientUpdate(
            ParameterUpdateConfig.natural_gradient(step_size=0.1)
        ),
        "adam": lambda: AdamUpdate(ParameterUpdateConfig.adam(step_size=1e-3)),
    }


def _run(credit: str, update: str, geometry_fn: Callable, seed: int) -> float:
    torch.manual_seed(seed)
    if credit == "bp":
        credit_obj = BackpropCredit()
    else:
        credit_obj = LocalGoodnessCredit(
            CreditAssignmentConfig.local_goodness(
                feedback_scale=0.01, local_objective="ff"
            )
        )
    system = compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device="cpu")),
        geometry=geometry_fn(),
        dynamics=InstantaneousDynamics(StateDynamicsConfig.instantaneous()),
        credit=credit_obj,
        update=_updates()["muon" if update == "ff_muon" else update](),
    )
    SystemTrainer(
        system=system,
        config=SystemTrainerConfig(max_epochs=1, device="cpu", seed=42),
        train_data=_TRAIN_DATA,
    ).fit()
    correct = total = 0
    with torch.no_grad():
        for batch_x, batch_y in _TEST_BATCHES:
            state = system.dynamics.settle(
                SystemState(x=batch_x), system.geometry, system.substrate, None
            )
            acts = state.activations
            out = acts[-1] if isinstance(acts, list) else acts
            correct += (out.argmax(1) == batch_y).sum().item()
            total += batch_y.size(0)
    return correct / total


def _n_params(geometry_fn: Callable) -> int:
    return sum(p.numel() for p in geometry_fn().params.values())


_TASK = create_task("mnist", device="cpu", quick_mode=True, num_workers=0)
_TASK.setup()
torch.manual_seed(0)  # seed BEFORE the loader draw (D8 trap)
_TRAIN_DATA = [
    (xb.view(xb.size(0), -1), yb)
    for xb, yb in islice(_TASK.get_dataloader("train"), BATCH_CAP)
]
# GraphGeometry aggregates over dim-0 node indices — partial final test
# batches (16 samples) cannot host the 32-node graph; drop them.
_TEST_BATCHES = [
    (xb.view(xb.size(0), -1), yb)
    for xb, yb in _TASK.get_dataloader("test")
    if xb.size(0) == 32
]


# Slow tier: ~265 s (4 geometries × 7 arms × 3 seeds at capacity-match).
@pytest.mark.slow
@pytest.mark.timeout(600)
def test_demo_uaxis_coverage(emit_run_record) -> None:
    geometries = _geometries()
    updates = list(_updates())
    record: dict = {"arms": {}, "seeds": list(SEEDS), "params": {}}
    grid: dict[str, dict[str, float]] = {}

    for name, geometry_fn in geometries.items():
        params = _n_params(geometry_fn)
        record["params"][name] = params
        grid[name] = {}
        for update in [*updates, "ff_muon", "ff_adam"]:
            credit = "bp"
            u = update
            if update == "ff_muon":
                credit, u = "ff", "muon"
            elif update == "ff_adam":
                credit, u = "ff", "adam"
            accs = [_run(credit, u, geometry_fn, s) for s in SEEDS]
            mean = sum(accs) / len(accs)
            record["arms"][f"{name}/{update}"] = {
                "mean": mean,
                "std": (sum((a - mean) ** 2 for a in accs) / len(accs)) ** 0.5,
                "seeds": accs,
            }
            grid[name][update] = mean
            print(f"{name:>12} {update:>8}: {mean:.3f} {accs} params={params}")

    arms = record["arms"]
    record["figure"] = figure_spec(
        "D16 — the U-axis coverage map: no update rule dominates (Muon, Adam, Euclid are distinct families)",
        heatmap_panel(
            grid=[
                [grid[g][u] for u in [*updates, "ff_muon", "ff_adam"]]
                for g in geometries
            ],
            row_labels=list(geometries),
            col_labels=[*updates, "ff/muon", "ff/adam"],
            cmap="viridis",
            annotate=True,
            chance=CHANCE,
            title="test accuracy, mnist quick 150 batches, seeds 0–2 (params 47.7k–57.5k)",
        ),
        figsize=[8.5, 4.5],
    )

    emit_run_record("D16", "uaxis_coverage", record)

    param_values = list(record["params"].values())
    assert max(param_values) / min(param_values) < 1.25, (
        "geometries must be capacity-matched (D8–D12 convention)"
    )
    for name in geometries:
        muon = arms[f"{name}/muon"]["mean"]
        euclid = arms[f"{name}/euclid"]["mean"]
        assert muon - euclid > 0.02, (
            f"{name}: Muon lift over Euclidean must hold (muon {muon:.3f} "
            f"vs euclid {euclid:.3f})"
        )
        spectral = arms[f"{name}/spectral"]["mean"]
        assert muon >= spectral - 0.01, (
            f"{name}: Muon must not clearly trail spectral ({muon:.3f} vs "
            f"{spectral:.3f})"
        )
        assert arms[f"{name}/natural"]["mean"] < 0.6, (
            f"{name}: natural gradient stays far below the others at this "
            "budget (OPEN regime question, R11.5.5a)"
        )
        assert arms[f"{name}/ff_muon"]["mean"] < muon, (
            f"{name}: local ff×Muon trails bp×Muon at this budget"
        )
    for name in ("mlp_d2_w64", "graph_grid8x4"):
        muon = arms[f"{name}/muon"]["mean"]
        spectral = arms[f"{name}/spectral"]["mean"]
        assert muon > spectral + 0.02, (
            f"{name}: Muon must clearly beat spectral ({muon:.3f} vs {spectral:.3f})"
        )
