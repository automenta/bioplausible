"""Shared CLI utilities and constants."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from computronium.core._paths import db_path
from computronium.core.logging import get_logger
from computronium.core.registry import ComponentCategory, Registry
from computronium.hyperopt.eval_tiers import (
    EvaluationConfig,
    PatientLevel,
)

if TYPE_CHECKING:
    import optuna

__all__ = [
    "FAMILY_MAP",
    "_BASELINE_MODELS",
    "_DB_PATH",
    "_STORAGE_URL",
    "_TrialContext",
    "_family_target",
    "_make_objective",
    "_query_registry_models",
    "_resolve_family_models",
    "_resolve_survivors",
    "_resolve_targets",
    "_set_storage",
    "_tier_for_args",
]

logger = get_logger()

# Optuna stores studies in SQLite via SQLAlchemy-style URLs; the same file is
# also read by HyperoptStorage (``trial_id`` PK matches Optuna's trial number).
_DB_PATH = db_path("computronium.db")
_STORAGE_URL = f"sqlite:///{_DB_PATH}"


def _set_storage(db_path_str: str | None = None) -> tuple[str, str]:
    """Resolve the SQLite storage backend, defaulting to the hardcoded file.

    Returns ``(db_path, storage_url)``.  ``--db`` on any HPO subcommand lets
    parallel/long runs isolate artifacts in a dedicated file.
    """
    path = db_path_str or _DB_PATH
    if not str(path).endswith(".db"):
        path = f"{path}.db"
    return path, f"sqlite:///{path}"


# Maps the CLI family label to the canonical ``family`` value used on the
# component registry metadata.  ``feedback_alignment`` → ``fa`` (matching the
# registry convention used by the native model registrations).
FAMILY_MAP: dict[str, str] = {
    "eqprop": "eqprop",
    "forward_only": "forward_only",
    "feedback_alignment": "fa",
    "tile": "tile",
    "hebbian": "hebbian",
    "predictive_coding": "predictive_coding",
    "target_prop": "target_prop",
    "spiking": "spiking",
    "mep": "mep",
    "backprop": "backprop",
}

# Models documented as intentional baselines (fail learns-gate; excluded from Phase 1 HPO)
# See FIX.md §37 for rationale.
_BASELINE_MODELS = frozenset({
    # EqProp contrastive variants (nudge dies in deep layers)
    "eqprop",
    "directed_ep",
    "finite_nudge_ep",
    "momentum_equilibrium",
    "sparse_equilibrium",
    "equilibrium_alignment",
    "layerwise_equilibrium_fa",
    # FA hybrids (credit assignment fails end-to-end)
    "contrastive_feedback_alignment",
    "energy_guided_fa",
    "energy_minimizing_fa",
    # Hebbian (update rule plateaus)
    "hebbian_3d",
    # Predictive Coding (requires per-graph propagator)
    "fabricpc_graph_pcn",
    # Spiking (requires surrogate-gradient BPTT)
    "spiking_stdp",
    # Diffusion (needs timestep in forward)
    "eqprop_diffusion",
})


def _query_registry_models(reg_family: str) -> list[str]:
    """Return registered model names for a registry family value."""
    entries = Registry.query(category=ComponentCategory.MODEL, family=reg_family)
    return [str(m["name"]) for m in entries]


def _resolve_family_models(cli_family: str) -> tuple[str, list[str]]:
    """Resolve a CLI family label to (registry_family, [model_names]).

    Native tile deployments register under ``family="tile"``; registration
    is triggered lazily by the ``Registry.query`` below.
    """
    reg_family = FAMILY_MAP.get(cli_family, cli_family)
    models = _query_registry_models(reg_family)
    # Exclude documented baselines that fail the learns-gate
    models = [m for m in models if m not in _BASELINE_MODELS]
    return reg_family, models


def _resolve_targets(args) -> list[tuple[str, str, str | None, list[str]]]:
    """Return ``[(study_name, reg_family, cli_family, [model_names]), ...]``.

    ``cli_family`` is ``None`` in the legacy per-model (``--models``) path.
    """
    targets: list[tuple[str, str, str | None, list[str]]] = []
    if args.family:
        if args.family == "survivors":
            return _resolve_survivors(args)
        families = list(FAMILY_MAP) if args.family == "all" else [args.family]
        for cli_family in families:
            target = _family_target(cli_family, args.task)
            if target is not None:
                targets.append(target)
        return targets

    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        for m in models:
            targets.append((f"{m}_{args.task}", m, None, [m]))
        return targets

    return targets


def _family_target(
    cli_family: str, task: str
) -> tuple[str, str, str | None, list[str]] | None:
    """Resolve one CLI family label to a (study, family, cli, models) target."""
    reg_family, models = _resolve_family_models(cli_family)
    if not models:
        logger.warning("No models registered for family '%s'; skipping", cli_family)
        return None
    for m in models:
        logger.warning("Skipping %s: incompatible with task '%s'", m, task)
    return (f"{reg_family}_{task}", reg_family, cli_family, models)


def _resolve_survivors(args) -> list[tuple[str, str, str | None, list[str]]]:
    """Resolve ``--family survivors``: auto-expand to surviving CLI families.

    Reads the portfolio CSV we emit (default ``results/portfolio.csv``) and
    keeps only rows whose status is ``Scale`` or ``Hold`` (Phase 1.2 gate:
    only families that survived the digits decision advance to CIFAR-10).
    """
    import csv as _csv

    path = Path(getattr(args, "survivors_csv", "") or "results/portfolio.csv")
    if not path.exists():
        logger.warning(
            "Survivors CSV '%s' not found; treating all families as survivors", path
        )
        families = list(FAMILY_MAP)
    else:
        families: list[str] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(_csv.DictReader(f))
        header = list(rows[0].keys()) if rows else []
        status_col = next((c for c in header if c.lower() == "status"), "status")
        for r in rows:
            fam = (r.get("family") or "").strip()
            if not fam or fam == "backprop":
                continue
            status = (r.get(status_col) or "").strip()
            if status in {"Scale", "Hold"}:
                families.append(fam)
        if not families:
            logger.warning("No survivors in '%s'; defaulting to all families", path)
            families = list(FAMILY_MAP)

    # Map registry family values back to CLI labels where possible.
    cli_by_reg = {rf: cli for cli, rf in FAMILY_MAP.items()}
    resolved: list[tuple[str, str, str | None, list[str]]] = []
    for family in families:
        cli = cli_by_reg.get(family, family)
        if cli not in FAMILY_MAP:
            logger.warning("Unknown survivor family '%s'; skipping", family)
            continue
        reg_family, models = _resolve_family_models(cli)
        if models:
            resolved.append((f"{reg_family}_{args.task}", reg_family, cli, models))
    return resolved


def _tier_for_args(args) -> PatientLevel:
    """Pick a patience tier from args (explicit ``--tier`` wins)."""
    if getattr(args, "tier", None):
        return PatientLevel(args.tier.lower())
    return PatientLevel.STANDARD


@dataclass(frozen=True, slots=True)
class _TrialContext:
    """Captured state for building an Optuna trial objective."""

    model: str
    family: str
    task: str
    eval_cfg: EvaluationConfig
    quick_mode: bool
    device: str
    tier_name: str


def _make_objective(
    ctx: _TrialContext,
    objectives: list[str] | None = None,
    directions: list[str] | None = None,
    max_params: int | None = None,
    search_space: dict[str, object] | None = None,
):
    """Build an Optuna objective closure for a single model.

    Args:
        ctx: Trial context
        objectives: List of objective names to optimize (default: ["accuracy", "loss"])
        directions: List of directions ("maximize" or "minimize") for each objective
        max_params: Hard constraint on maximum parameter count; trials exceeding this are pruned
        search_space: Experiment-owned search-space overrides passed to Optuna sampling
    """
    if objectives is None:
        objectives = ["accuracy", "loss"]
    if directions is None:
        directions = ["maximize", "minimize"]

    def objective(trial: optuna.Trial):
        trial.set_user_attr("model_name", ctx.model)
        trial.set_user_attr("family", ctx.family)
        trial.set_user_attr("task", ctx.task)
        trial.set_user_attr("tier", ctx.tier_name)

        # Sample hyperparameters
        from computronium.hyperopt import sample_config

        config = sample_config(trial, ctx.model, ctx.task, ctx.eval_cfg, ctx.quick_mode)

        # Run single trial
        from computronium.hyperopt.experiment import run_single_trial

        result = run_single_trial(
            model_name=ctx.model,
            task_name=ctx.task,
            config=config,
            eval_cfg=ctx.eval_cfg,
            device=ctx.device,
            seed=trial.number,
        )

        # Return objectives
        if len(objectives) == 1:
            return result.get(objectives[0], 0.0)
        return tuple(result.get(obj, 0.0) for obj in objectives)

    return objective
