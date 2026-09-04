"""The Gallery: figures rendered from the demo suite's own deterministic
run records at HEAD (TODO10 R10.1).

Every figure is what a demo test shows, drawn: fixed seeds, CPU, current
code. One pure figure factory per capability, each consuming the run record
its demo test emitted (``docs/figures/run_records/``). Nothing frozen,
nothing to re-verify — the figure lock (R10.1.4) regenerates each figure and
compares data-layer checksums so the gallery cannot silently drift from what
the code actually demonstrates.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from computronium.visualization._style import (
    COLOR_ARM,
    COLOR_CONTRAST,
    COLOR_FEASIBLE,
    COLOR_WALLED,
    apply_style,
    chance_line,
    save,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from matplotlib.figure import Figure

SCOPE_LABEL = "live demo scale (HEAD, CPU, fixed seeds)"
RECORDS_DIRNAME = "run_records"

# For D9 graph geometry swap
NUM_CLASSES = 4


@dataclass(frozen=True, slots=True)
class FigureMeta:
    """Provenance and scope of one gallery figure."""

    capability_id: str
    capability_name: str
    demo_test: str
    provenance: dict[str, str]
    scope_label: str
    data_sha256: str
    figure_png: str


def _records(records_dir: Path) -> Iterator[dict]:
    for path in sorted(records_dir.glob("*.json")):
        record: dict = json.loads(path.read_text(encoding="utf-8"))
        record["_path"] = path
        yield record


def _sha256_data(record: dict) -> str:
    payload = json.dumps(record["data"], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _fig_compose_train(record: dict) -> Figure:
    import matplotlib.pyplot as plt

    data = record["data"]
    history = data["six_axis"]["history"]
    epochs = range(1, len(history) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [h["train_acc"] for h in history], marker="o", color=COLOR_ARM)
    ax.set_ylim(0, 1)
    chance = 1 / 10
    chance_line(ax, chance, "chance (0.1)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("train accuracy")
    ax.set_title(
        "D1 — six-axis composition trains "
        f"(J1 θ-bitwise-equal: {data['j1']['theta_bitwise_equal']}, "
        f"round-trip: {data['round_trip']})"
    )
    apply_style(fig)
    return fig


def _fig_credit_swap(record: dict) -> Figure:
    import matplotlib.pyplot as plt

    arms = record["data"]["arms"]
    names = list(arms)
    accs = [arms[name]["train_acc"] for name in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, accs, color=[COLOR_ARM, COLOR_CONTRAST, COLOR_FEASIBLE])
    ax.set_ylim(0, 1)
    chance = 1 / 10
    chance_line(ax, chance, "chance (0.1)")
    ax.set_ylabel("train accuracy")
    ax.set_title("D2 — one trainer, three credit rules (wiring identical)")
    for i, acc in enumerate(accs):
        ax.text(i, acc, f"{acc:.2f}", ha="center", va="bottom", fontsize=9)
    apply_style(fig)
    return fig


def _fig_plasticity_swap(record: dict) -> Figure:
    import matplotlib.pyplot as plt

    arms = record["data"]["arms"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, color in (("null", COLOR_WALLED), ("routing", COLOR_ARM)):
        mastery = arms[label]["a_mastery"]
        retained = arms[label]["a_retained"]
        for m, r in zip(mastery, retained, strict=True):
            ax.plot([0, 1], [m, r], color=color, alpha=0.35, linewidth=1)
        ax.plot(
            [0, 1],
            [sum(mastery) / len(mastery), sum(retained) / len(retained)],
            color=color,
            linewidth=3,
            label=label,
        )
    chance_line(ax, record["data"]["chance"], "chance")
    ax.set_xticks([0, 1], ["after segment A", "after segment B"])
    ax.set_ylabel("segment-A probe accuracy")
    ax.set_title("D3 — the M-axis swap: routing retains what null forgets")
    ax.legend()
    apply_style(fig)
    return fig


def _fig_memory_wall(record: dict) -> Figure:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    data = record["data"]
    budget_bytes = data["budget_mib"] * 1024 * 1024
    cells = {
        k: v for k, v in data["saved_bytes"].items() if not k.startswith("control")
    }
    envs = sorted({k.split("@")[1] for k in cells})
    arms = sorted({k.split("@")[0] for k in cells})
    fig, ax = plt.subplots(figsize=(6, 4))
    for row, arm in enumerate(arms):
        for col, env in enumerate(envs):
            saved = cells[f"{arm}@{env}"]
            feasible = saved <= budget_bytes
            ax.add_patch(
                Rectangle(
                    (col, row),
                    0.9,
                    0.9,
                    facecolor=COLOR_FEASIBLE if feasible else COLOR_WALLED,
                    alpha=0.75,
                )
            )
            ax.text(
                col + 0.45,
                row + 0.45,
                "runs" if feasible else "walled",
                ha="center",
                va="center",
                fontsize=9,
            )
    ax.set_xlim(0, len(envs))
    ax.set_ylim(0, len(arms))
    ax.set_xticks([c + 0.45 for c in range(len(envs))], envs)
    ax.set_yticks([r + 0.45 for r in range(len(arms))], arms)
    ax.set_xlabel(f"depth environment (budget {data['budget_mib']} MiB)")
    ax.set_title("D4 — the memory profiler decides before training")
    apply_style(fig)
    return fig


def _fig_frozen_theta(record: dict) -> Figure:
    import matplotlib.pyplot as plt

    r = record["data"]
    stages = ["stage A (adapted)", "restored ψ", "fresh-ψ floor"]
    accs = [
        r["stage_a"]["accuracy"],
        r["restored"]["task_a_accuracy"],
        r["restored"]["fresh_psi_floor"],
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(stages, accs, color=[COLOR_ARM, COLOR_FEASIBLE, COLOR_WALLED])
    ax.set_ylim(0, 1)
    chance_line(ax, 0.5, "chance (0.5)")
    ax.set_ylabel("fixed-probe accuracy")
    badge = (
        "identical"
        if r["theta_sha256_before"] == r["theta_sha256_after"]
        else "CHANGED"
    )
    ax.set_title(
        f"D5 — frozen θ is bitwise ({badge}: θ sha256 {r['theta_sha256_before'][:12]}…)"
    )
    apply_style(fig)
    return fig


def _fig_substrate_swap(record: dict) -> Figure:
    import matplotlib.pyplot as plt

    arms = record["data"]["arms"]
    names = list(arms)
    accs = [arms[name]["train_acc"] for name in names]
    zeros = [arms[name].get("probe_state_zeros", 0.0) for name in names]
    labels = [
        name.replace("memristive_", "memristive\nIR-drop ").replace(
            "neuromorphic_", "neuromorphic\nspike-drop "
        )
        for name in names
    ]
    fig, (ax_acc, ax_probe) = plt.subplots(1, 2, figsize=(11, 4))
    acc_colors = [
        COLOR_FEASIBLE
        if name == "digital"
        else COLOR_WALLED
        if "severe" in name
        else COLOR_ARM
        for name in names
    ]
    ax_acc.bar(labels, accs, color=acc_colors)
    ax_acc.set_ylim(0, 1)
    chance_line(ax_acc, 1 / 10, "chance (0.1)")
    ax_acc.set_ylabel("train accuracy")
    ax_acc.set_title(
        "D6 — one wiring, one swapped substrate (mild physics learns, severe walls)"
    )
    for i, acc in enumerate(accs):
        ax_acc.text(i, acc, f"{acc:.2f}", ha="center", va="bottom", fontsize=9)
    ax_probe.bar(labels, zeros, color=acc_colors)
    ax_probe.set_ylim(0, 1)
    ax_probe.set_ylabel("probe state zeros (fraction)")
    ax_probe.set_title("the dial itself: dropout thins the state, noise does not")
    for i, z in enumerate(zeros):
        ax_probe.text(i, z, f"{z:.2f}", ha="center", va="bottom", fontsize=9)
    apply_style(fig)
    return fig


def _fig_spike_settle(record: dict) -> Figure:
    import matplotlib.pyplot as plt

    data = record["data"]
    arms = data["arms"]
    obs = data["spike_observation"]
    fig, (ax_acc, ax_spikes) = plt.subplots(1, 2, figsize=(9, 4))
    names = list(arms)
    accs = [arms[name]["train_acc"] for name in names]
    ax_acc.bar(names, accs, color=[COLOR_CONTRAST, COLOR_ARM])
    ax_acc.set_ylim(0, 1)
    chance_line(ax_acc, 1 / 10, "chance (0.1)")
    ax_acc.set_ylabel("train accuracy")
    ax_acc.set_title("D7 — one wiring, one swapped D-axis")
    for i, acc in enumerate(accs):
        ax_acc.text(i, acc, f"{acc:.2f}", ha="center", va="bottom", fontsize=9)
    totals = obs["spike_totals"]
    half = len(totals) // 2
    steps = range(len(totals))
    ax_spikes.bar(
        steps, totals, color=[COLOR_ARM] * half + [COLOR_WALLED] * (len(totals) - half)
    )
    ax_spikes.set_xlabel("settle step (hidden | output)")
    ax_spikes.set_ylabel("spikes per step")
    ax_spikes.set_title(
        f"LIF settle: {obs['total_spikes']:.0f} spikes, "
        f"membrane max {obs['membrane_max']:.2f} ≤ {data['threshold']}"
    )
    apply_style(fig)
    return fig


def _fig_geometry_swap(record: dict) -> Figure:
    import matplotlib.pyplot as plt

    data = record["data"]
    arms = data["arms"]
    names = list(arms)
    shifts = data["probe_shifts"]
    fig, (ax_train, ax_probe) = plt.subplots(1, 2, figsize=(9, 4))
    accs = [arms[name]["train_acc"] for name in names]
    params = [arms[name]["param_count"] / 1000 for name in names]
    colors = [COLOR_CONTRAST, COLOR_ARM]
    labels = [
        f"{name}\n({p:.1f}k params)" for name, p in zip(names, params, strict=True)
    ]
    ax_train.bar(labels, accs, color=colors)
    ax_train.set_ylim(0, 1)
    chance_line(ax_train, 1 / 10, "chance (0.1)")
    ax_train.set_ylabel("train accuracy")
    ax_train.set_title("D8 — one wiring, one swapped G-axis (capacity-fair)")
    for i, acc in enumerate(accs):
        ax_train.text(i, acc, f"{acc:.2f}", ha="center", va="bottom", fontsize=9)
    for name, color, p in zip(names, colors, params, strict=True):
        probe = arms[name]["probe"]
        ax_probe.plot(
            shifts,
            [probe[str(s)] for s in shifts],
            marker="o",
            color=color,
            label=f"{name} ({p:.1f}k params)",
        )
    chance_line(ax_probe, 1 / 10, "chance (0.1)")
    ax_probe.set_xlabel("probe digit shift (px)")
    ax_probe.set_ylabel("probe accuracy")
    ax_probe.set_title("the smaller conv arm retains the shifted digits")
    ax_probe.legend()
    apply_style(fig)
    return fig


def _fig_graph_geometry_swap(record: dict) -> Figure:
    import matplotlib.pyplot as plt

    data = record["data"]
    arms = data["arms"]
    names = list(arms)
    fig, (ax_train, ax_probe) = plt.subplots(1, 2, figsize=(9, 4))
    accs = [arms[name]["train_acc"] for name in names]
    params = [arms[name]["param_count"] / 1000 for name in names]
    colors = [COLOR_CONTRAST, COLOR_ARM]
    labels = [
        f"{name}\n({p:.1f}k params)" for name, p in zip(names, params, strict=True)
    ]
    ax_train.bar(labels, accs, color=colors)
    ax_train.set_ylim(0, 1)
    chance = 1 / NUM_CLASSES
    chance_line(ax_train, chance, f"chance ({chance:.2f})")
    ax_train.set_ylabel("train accuracy")
    ax_train.set_title("D9 — one wiring, one swapped G-axis (graph structure)")
    for i, acc in enumerate(accs):
        ax_train.text(i, acc, f"{acc:.2f}", ha="center", va="bottom", fontsize=9)

    # Probe comparison: edge perturbation robustness
    probe_key = "probe_perturb_02"
    probe_vals = [arms[name][probe_key] for name in names]
    ax_probe.bar(labels, probe_vals, color=colors)
    ax_probe.set_ylim(0, 1)
    chance_line(ax_probe, chance, f"chance ({chance:.2f})")
    ax_probe.set_ylabel("probe accuracy (20% edge dropout)")
    ax_probe.set_title("graph arm more robust to edge perturbation")
    for i, p in enumerate(probe_vals):
        ax_probe.text(i, p, f"{p:.2f}", ha="center", va="bottom", fontsize=9)
    apply_style(fig)
    return fig


def _fig_attention_geometry_swap(record: dict) -> Figure:
    import matplotlib.pyplot as plt

    data = record["data"]
    arms = data["arms"]
    names = list(arms)
    fig, (ax_train, ax_probe) = plt.subplots(1, 2, figsize=(9, 4))
    accs = [arms[name]["train_acc"] for name in names]
    params = [arms[name]["param_count"] / 1000 for name in names]
    colors = [COLOR_CONTRAST, COLOR_ARM]
    labels = [
        f"{name}\n({p:.1f}k params)" for name, p in zip(names, params, strict=True)
    ]
    ax_train.bar(labels, accs, color=colors)
    ax_train.set_ylim(0, 1)
    chance_line(ax_train, 1 / 10, "chance (0.1)")
    ax_train.set_ylabel("train accuracy")
    ax_train.set_title("D10 — one wiring, one swapped G-axis (attention)")
    for i, acc in enumerate(accs):
        ax_train.text(i, acc, f"{acc:.2f}", ha="center", va="bottom", fontsize=9)

    probe_vals = [arms[name]["probe_normal"] for name in names]
    perm_vals = [arms[name]["probe_permuted"] for name in names]
    ax_probe.bar(labels, probe_vals, color=colors, label="unpermuted probe")
    ax_probe.bar(labels, perm_vals, color=colors, alpha=0.45, label="permuted probe")
    ax_probe.set_ylim(0, 1)
    chance_line(ax_probe, 1 / 10, "chance (0.1)")
    ax_probe.set_ylabel("probe accuracy")
    ax_probe.set_title("probe vs pixel-permuted probe")
    ax_probe.legend()
    apply_style(fig)
    return fig


def _fig_spatial_lattice_geometry_swap(record: dict) -> Figure:
    import matplotlib.pyplot as plt

    data = record["data"]
    arms = data["arms"]
    names = list(arms)
    fig, (ax_train, ax_probe) = plt.subplots(1, 2, figsize=(9, 4))
    accs = [arms[name]["train_acc"] for name in names]
    params = [arms[name]["param_count"] / 1000 for name in names]
    colors = [COLOR_CONTRAST, COLOR_ARM]
    labels = [
        f"{name}\n({p:.1f}k params)" for name, p in zip(names, params, strict=True)
    ]
    ax_train.bar(labels, accs, color=colors)
    ax_train.set_ylim(0, 1)
    chance_line(ax_train, 1 / 10, "chance (0.1)")
    ax_train.set_ylabel("train accuracy")
    ax_train.set_title("D11 — one wiring, one swapped G-axis (3D lattice)")
    for i, acc in enumerate(accs):
        ax_train.text(i, acc, f"{acc:.2f}", ha="center", va="bottom", fontsize=9)

    probe_vals = [arms[name]["probe_normal"] for name in names]
    noisy_vals = [arms[name]["probe_noisy"] for name in names]
    ax_probe.bar(labels, probe_vals, color=colors, label="clean probe")
    ax_probe.bar(labels, noisy_vals, color=colors, alpha=0.45, label="noisy probe")
    ax_probe.set_ylim(0, 1)
    chance_line(ax_probe, 1 / 10, "chance (0.1)")
    ax_probe.set_ylabel("probe accuracy")
    ax_probe.set_title("probe vs additive-noise probe")
    ax_probe.legend()
    apply_style(fig)
    return fig


def _fig_epc_fast_settle(record: dict) -> Figure:
    import matplotlib.pyplot as plt

    data = record["data"]
    arms = data["arms"]
    names = list(arms)
    fig, (ax_acc, ax_dev) = plt.subplots(1, 2, figsize=(9, 4))
    accs = [arms[name]["train_acc"] for name in names]
    ax_acc.bar(names, accs, color=[COLOR_CONTRAST, COLOR_ARM])
    ax_acc.set_ylim(0, 1)
    chance_line(ax_acc, 1 / 10, "chance (0.1)")
    ax_acc.set_ylabel("train accuracy")
    ax_acc.set_title("D12 — one wiring, one swapped D-axis (ePC)")
    for i, (name, acc) in enumerate(zip(names, accs, strict=True)):
        budget = arms[name]["settle_budget"]
        ax_acc.text(
            i, acc, f"{acc:.2f}\n({budget} steps)", ha="center", va="bottom", fontsize=9
        )
    for name, color in zip(names, [COLOR_CONTRAST, COLOR_ARM], strict=True):
        devs = arms[name]["nudged_layer_deviations"]
        ax_dev.plot(
            range(len(devs)),
            devs,
            marker="o",
            color=color,
            label=name,
        )
    ax_dev.set_yscale("symlog", linthresh=1e-4)
    ax_dev.set_xlabel("layer (input → hidden → output)")
    ax_dev.set_ylabel("|nudged − free| (max, per layer)")
    ax_dev.set_title("the output-error signal reaches every layer in ePC")
    ax_dev.legend()
    apply_style(fig)
    return fig


_FACTORIES: dict[str, Callable[[dict], Figure]] = {
    "compose_6axis": _fig_compose_train,
    "swap_credit": _fig_credit_swap,
    "swap_plasticity": _fig_plasticity_swap,
    "memory_budget": _fig_memory_wall,
    "substrate_swap": _fig_substrate_swap,
    "spike_settle": _fig_spike_settle,
    "z3_frozen_theta": _fig_frozen_theta,
    "geometry_swap": _fig_geometry_swap,
    "graph_geometry_swap": _fig_graph_geometry_swap,
    "attention_geometry_swap": _fig_attention_geometry_swap,
    "spatial_lattice_geometry_swap": _fig_spatial_lattice_geometry_swap,
    "epc_fast_settle": _fig_epc_fast_settle,
}


def render_gallery(records_dir: Path, out_dir: Path) -> list[FigureMeta]:
    """Render one figure per demo run record and write the manifest.

    A record whose demo test no longer exists produces no figure — no
    orphaned claims (R10.3.2). Returns the rendered figures' metadata.
    """
    metas: list[FigureMeta] = []
    for record in _records(records_dir):
        capability_name = record["capability_name"]
        factory = _FACTORIES.get(capability_name)
        if factory is None or not Path(record["demo_test"]).exists():
            continue
        fig = factory(record)
        png = out_dir / f"{record['capability'].lower()}_{capability_name}.png"
        save(fig, png)
        plt_close(fig)
        metas.append(
            FigureMeta(
                capability_id=record["capability"],
                capability_name=capability_name,
                demo_test=record["demo_test"],
                provenance=record["provenance"],
                scope_label=SCOPE_LABEL,
                data_sha256=_sha256_data(record),
                figure_png=png.name,
            )
        )
    manifest = {"figures": [asdict(m) for m in metas]}
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metas


def plt_close(fig: Figure) -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)


__all__ = ["FigureMeta", "render_gallery"]
