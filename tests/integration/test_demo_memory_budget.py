"""D4 — The memory profiler is honest.

Before any training runs, the memory profile decides feasibility. The
backprop-profiled arm (gradient credit — BPTT saves activations for
backward) is walled under the tight 0.015 MiB budget at every depth, and
the O(1)-memory arm (ThermodynamicContrast — nothing saved) runs. The
demonstration is categorical by construction: verdicts are memory-profile
arithmetic, and a walled cell produces no walk.

Demonstrated regime (pinned 2026-09-02, from the registered grid): the wall
is a depth phenomenon — saved bytes grow with depth (26.5 KiB at depth 4,
134.5 KiB at depth 16, width 16, batch 16) while thermo stays at exactly 0.
The feasible arm then walks at the competence tier (depth 4, 50 episodes)
to a held-out probe ≈ 0.3 (chance 0.125).
"""

from computronium.experiments.joint.memory_budget_trial import (
    BUDGETS_MIB,
    MemoryBudgetConfig,
    _arm_coordinate,
    _environments,
    _feasibility_grid,
    _measure_saved_bytes,
    _walk_seed,
)
from computronium.visualization import figure_spec, heatmap_panel

ARMS = ("gradient", "random_projections", "thermodynamic_contrast")
CONTROL_CREDIT = "thermodynamic_contrast"  # the only credit feasible at every budget
BUDGET_MIB = BUDGETS_MIB[0]  # 0.015 MiB: walls every O(depth) arm at every depth
COMPETENCE_FLOOR = 0.225  # chance 0.125 + registered margin 0.1


def test_demo_memory_budget(emit_run_record) -> None:
    config = MemoryBudgetConfig(episodes=50, seeds=(0,), depths=(4, 16), device="cpu")
    envs = _environments(config)
    saved_bytes = {
        f"{arm}@{env.name}": _measure_saved_bytes(arm, env, config)
        for env in envs
        for arm in ARMS
    }
    # The frozen control shares thermo's profile: same credit, lr=0.
    for env in envs:
        saved_bytes[f"control@{env.name}"] = saved_bytes[f"{CONTROL_CREDIT}@{env.name}"]
    record: dict = {
        "budget_mib": BUDGET_MIB,
        "saved_bytes": saved_bytes,
    }

    # The profile separates the arms: O(1)-memory vs O(depth)-in-BPTT.
    thermo_bytes = saved_bytes["thermodynamic_contrast@depth_16"]
    assert thermo_bytes <= 0.0, "thermo must save nothing (O(1) memory)"
    assert saved_bytes["gradient@depth_16"] > saved_bytes["gradient@depth_4"] > 0.0

    # The feasibility grid: a walled cell is never commissioned.
    grid, never = _feasibility_grid(saved_bytes, envs, (BUDGET_MIB,))
    record["never_commissionable"] = list(never)
    for arm in ("gradient", "random_projections"):
        for env in envs:
            assert f"{arm}@{env.name}" in never, "the O(depth) arm must be walled"
    walled_env = next(env for env in envs if env.depth == 16)
    assert grid[f"{BUDGET_MIB}"][walled_env.name]["gradient"] is False
    assert grid[f"{BUDGET_MIB}"][walled_env.name]["thermodynamic_contrast"] is True

    # The feasible arm runs: O(1) memory walks to competence at depth 4.
    probe, late, _records = _walk_seed(
        "thermodynamic_contrast",
        False,
        envs[0],
        _arm_coordinate("thermodynamic_contrast"),
        0,
        config=config,
    )
    record["feasible_walk"] = {"probe": probe, "late_window_acc": late}
    assert probe >= COMPETENCE_FLOOR, "the O(1)-memory arm must demonstrably run"

    budget_bytes = BUDGET_MIB * 1024 * 1024
    arm_names = sorted({k.split("@")[0] for k in saved_bytes})
    env_names = sorted({k.split("@")[1] for k in saved_bytes})
    record["figure"] = figure_spec(
        "D4 — the memory profiler decides before training",
        heatmap_panel(
            [
                [
                    1.0 if saved_bytes[f"{arm}@{env}"] <= budget_bytes else 0.0
                    for env in env_names
                ]
                for arm in arm_names
            ],
            row_labels=arm_names,
            col_labels=env_names,
            vmin=0.0,
            vmax=1.0,
            fmt=".0f",
        ),
        figsize=[6, 4],
    )

    emit_run_record("D4", "memory_budget", record)
