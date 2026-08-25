"""
CLI Runner for Bioplausible Experiments

Commands:
    train        Run a single training session (``--config`` for YAML).
    core-train   Train via the new CoreTrainer API.
    from-config  Train from a YAML config file.
    search       Compute-matched HPO across a propagator family.
    compare      Rank families from completed HPO studies into a CSV.
    verify       Re-run the top-k configs of a study with n seeds.
    pareto       Emit Pareto frontier artefacts for a study.
    benchmark    Cross-domain benchmark suite.
    list         List registered models.
"""

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import optuna

from computronium.core._paths import db_path
from computronium.core.logging import get_logger
from computronium.core.ontology import PredictiveSettlingDynamics
from computronium.core.registry import ComponentCategory, Registry
from computronium.hyperopt import (
    create_optuna_space,
    create_study,
)
from computronium.hyperopt.comparison import (
    ComparisonMetric,
    compute_algorithm_rankings,
    generate_comparison_summary,
    group_trials_by_family,
)
from computronium.hyperopt.eval_tiers import (
    EvaluationConfig,
    PatientLevel,
    get_evaluation_config,
)
from computronium.hyperopt.experiment import run_single_trial_task
from computronium.hyperopt.portfolio import (
    PortfolioRow,
    regime_advantage_label,
)
from computronium.zoo import get_model_spec

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "FAMILY_MAP",
    "list_models",
    "logger",
    "main",
    "run_benchmark",
    "run_compare",
    "run_core_train",
    "run_from_yaml",
    "run_pareto",
    "run_portfolio",
    "run_search",
    "run_training",
    "run_verify",
]
logger = get_logger()

# ---------------------------------------------------------------------------
# HPO infrastructure constants
# ---------------------------------------------------------------------------

# Optuna stores studies in SQLite via SQLAlchemy-style URLs; the same file is
# also read by HyperoptStorage (``trial_id`` PK matches Optuna's trial number).
_DB_PATH = db_path("computronium.db")
_STORAGE_URL = f"sqlite:///{_DB_PATH}"


def _set_storage(db_path: str | None = None) -> tuple[str, str]:
    """Resolve the SQLite storage backend, defaulting to the hardcoded file.

    Returns ``(db_path, storage_url)``.  ``--db`` on any HPO subcommand lets
    parallel/long runs isolate artifacts in a dedicated file.
    """
    path = db_path or _DB_PATH
    if not str(path).endswith(".db"):
        path = f"{path}.db"
    return path, f"sqlite:///{path}"


# Maps the CLI family label to the canonical ``family`` value used on the
# component registry metadata.  ``feedback_alignment`` → ``fa`` (matching the
# registry convention seen in ``zoo/models/fa.py``).
FAMILY_MAP: dict[str, str] = {
    "eqprop": "eqprop",
    "forward_only": "forward_only",
    "feedback_alignment": "fa",
    "equitile": "equitile",
    "hebbian": "hebbian",
    "predictive_coding": "predictive_coding",
    "target_prop": "target_prop",
    "spiking": "spiking",
    "mep": "mep",
    "backprop": "backprop",
}

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _set_seeds(seed: int) -> None:
    """Seed every RNG used downstream for reproducible single-trial runs."""
    from computronium.core.utils.seeds import set_all_seeds

    set_all_seeds(seed)


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

    The deployment models (``*_equitile``) register under ``family="equitile"``
    when the zoo package is imported, which the top of this module already does
    via ``computronium.zoo.get_model_spec``.
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
    objectives: list[str] = None,
    directions: list[str] = None,
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

        trial_config = create_optuna_space(
            trial, ctx.model, task_name=ctx.task, search_space=search_space
        )
        trial_config["epochs"] = ctx.eval_cfg.epochs
        trial_config["batch_size"] = ctx.eval_cfg.batch_size
        trial_config["tier"] = ctx.tier_name
        trial_config.setdefault("device", ctx.device)

        try:
            metrics = run_single_trial_task(
                task=ctx.task,
                model_name=ctx.model,
                config=trial_config,
                storage_path=_DB_PATH,
                quick_mode=ctx.quick_mode,
                verbose=False,
            )
        except optuna.TrialPruned:
            raise
        except Exception:
            logger.exception("Trial failed for %s", ctx.model)
            raise optuna.TrialPruned

        if metrics is None:
            raise optuna.TrialPruned

        param_count = float(metrics["param_count"])
        if max_params is not None and param_count > max_params:
            logger.info(
                "   [PRUNED] %s: params=%.0f exceeds max_params=%d",
                ctx.model,
                param_count,
                max_params,
            )
            raise optuna.TrialPruned

        trial.set_user_attr("param_count", param_count)
        trial.set_user_attr("iteration_time", float(metrics["time"]))
        trial.set_user_attr("loss", float(metrics["loss"]))
        trial.set_user_attr("epochs", ctx.eval_cfg.epochs)
        trial.set_user_attr("batch_size", ctx.eval_cfg.batch_size)

        # Build return values based on requested objectives
        values = []
        for obj in objectives:
            if obj == "accuracy":
                values.append(float(metrics["accuracy"]))
            elif obj == "loss":
                values.append(float(metrics["loss"]))
            elif obj == "param_count":
                values.append(param_count)
            elif obj == "epoch_time_s" or obj == "time":
                values.append(float(metrics["time"]))
            else:
                logger.warning("Unknown objective '%s', defaulting to 0.0", obj)
                values.append(0.0)

        acc = float(metrics["accuracy"])
        loss = float(metrics["loss"])
        logger.info(
            "   [OK] %s: Acc=%.4f Loss=%.4f Params=%.2fM Time=%.2fs",
            ctx.model,
            acc,
            loss,
            param_count,
            metrics.get("time", 0),
        )
        return tuple(values)

    return objective


def _safe_sampler_name(
    study_name: str,
    requested: str,
    n_startup_trials: int,
    storage_url: str,
) -> str:
    """Return a safe sampler name for ``study_name``.

    Optuna 4.9's multi-objective TPE crashes with ``TypeError`` when a study
    has accumulated >= ``n_startup_trials`` trials but every one is PRUNED
    (``values is None``) — e.g. an eqprop model whose forward pass raises a
    shape bug on every draw.  TPE then builds its Parzen estimator from
    ``None`` objective values and dies.  That state is not recoverable by TPE,
    so fall back to a RandomSampler (still seeded/reproducible) instead of
    crashing the whole family run.

    For NSGA-II, we need at least a few completed trials to form a population;
    otherwise fall back to TPE.
    """
    if requested == "nsga2":
        try:
            study = optuna.load_study(study_name=study_name, storage=storage_url)
        except KeyError, OSError:
            return "tpe"  # Study doesn't exist yet, start with TPE
        completed = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if len(completed) < 4:  # NSGA-II needs a small population to start
            return "tpe"
        return "nsga2"
    if requested != "tpe":
        return requested
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
    except KeyError, OSError:
        return "tpe"
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    total = len(study.trials)
    if total >= n_startup_trials and not completed:
        logger.warning(
            "[SAMPLER] study '%s' has %d trials but no COMPLETE values; "
            "MOTPE cannot build an estimator — falling back to random",
            study_name,
            total,
        )
        return "random"
    return "tpe"


def _fail_stale_running(study_name: str, storage_url: str) -> None:
    """Best-effort cleanup of stale RUNNING trials left by a killed process.

    Runs in isolation: if the study can't be read for any reason (including a
    corrupt trial that Optuna's RDB layer cannot reconstruct), we return
    silently rather than aborting the whole experiment. Stale RUNNING trials
    simply continue to be ignored downstream (runs skipped, not repeated).
    """
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        for t in study.trials:
            if t.state == optuna.trial.TrialState.RUNNING:
                logger.warning(
                    "[CLEAN] marking stale RUNNING trial %d as FAILED", t.number
                )
                try:
                    study._storage.set_trial_state_values(
                        t._trial_id,
                        optuna.trial.TrialState.FAIL,
                        (float("nan"),) * len(study.directions),
                    )
                except Exception:
                    pass
    except KeyError, ValueError, AssertionError:
        # Study doesn't exist yet, or a reusable-study header differs — normal on first run.
        return
    except Exception:
        logger.warning("[CLEAN] could not read study '%s' (skipped)", study_name)


def _run_hpo_family(
    study_name: str,
    reg_family: str,
    cli_family: str | None,
    model_list: list[str],
    args,
) -> list[tuple[str, optuna.Study | None]]:
    """Run ``--budget`` trials for each model, one Optuna study per model."""
    results: list[tuple[str, optuna.Study | None]] = []
    tier = _tier_for_args(args)
    eval_cfg = get_evaluation_config(tier)
    n_trials = args.budget if args.budget else eval_cfg.n_trials
    quick_mode = tier == PatientLevel.SMOKE
    device = getattr(args, "device", "auto")
    method = getattr(args, "method", None)
    sampler_name = "random" if method == "random" else "tpe"
    seed = getattr(args, "seed", None)
    n_startup = getattr(eval_cfg, "n_startup_trials", 10)

    if device == "auto":
        from computronium.core.utils.device import get_device

        resolved = str(get_device())
        logger.info("[DEVICE] auto -> %s", resolved)

    logger.info(
        "[SEARCH] family=%s study=%s models=%s trials/model=%s (%s, %s) n_startup=%d",
        cli_family or reg_family,
        study_name,
        model_list,
        n_trials,
        tier.value,
        sampler_name,
        n_startup,
    )

    for model in model_list:
        model_start = time.time()
        model_study_name = f"{reg_family}_{model}_{args.task}"
        logger.info("[MODEL] %s", model)
        _fail_stale_running(model_study_name, _STORAGE_URL)
        effective_sampler = _safe_sampler_name(
            model_study_name, sampler_name, n_startup, _STORAGE_URL
        )
        if effective_sampler != sampler_name:
            logger.info("[MODEL] %s -> sampler %s", model, effective_sampler)
        study = create_study(
            model_names=[model],
            n_objectives=2,
            storage=_STORAGE_URL,
            study_name=model_study_name,
            use_pruning=eval_cfg.use_pruning,
            sampler_name=effective_sampler,
            mode="pareto",
            seed=seed,
        )

        try:
            family_for_tag = get_model_spec(model).family or reg_family
        except ValueError:
            family_for_tag = reg_family
        objective = _make_objective(
            _TrialContext(
                model,
                family_for_tag,
                args.task,
                eval_cfg,
                quick_mode,
                device,
                tier.value,
            )
        )
        try:
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        except KeyboardInterrupt:
            logger.warning("Search interrupted for %s", model)
        except RuntimeError, ValueError, OSError, TypeError:
            logger.exception("[FAIL] Optimizing %s", model)
        logger.info("[OK]    %s done in %.1fs", model, time.time() - model_start)
        results.append((model, study))
    return results


def _write_study_jsonl(study: optuna.Study, stem: str, out_dir: str) -> None:
    """Append a study's complete trials to ``<out_dir>/<stem>.jsonl``."""
    out_path = Path(out_dir) / f"{stem}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            record = {
                "study": stem,
                "trial_id": trial.number,
                "model_name": trial.user_attrs.get("model_name"),
                "family": trial.user_attrs.get("family"),
                "params": trial.params,
                "accuracy": trial.values[0] if trial.values else 0.0,
                "loss": trial.values[1] if len(trial.values) > 1 else None,
                "param_count": trial.user_attrs.get("param_count"),
                "iteration_time": trial.user_attrs.get("iteration_time"),
            }
            f.write(json.dumps(record) + "\n")
    logger.info("[FILE]  Wrote trials for %s -> %s", stem, out_path)


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def run_training(args):
    """Run a single training session or training from YAML config."""
    if args.config:
        run_from_yaml(args)
        return

    logger.info("[START]  Starting Headless Training: %s on %s", args.model, args.task)

    from computronium.core.system_trainer import SystemTrainer, SystemTrainerConfig
    from computronium.domains.factory import create_task

    task = create_task(args.task, device=args.device, quick_mode=False)
    task.batch_size = args.batch_size
    task.setup()

    trainer_config = SystemTrainerConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        grad_clip=1.0,
        track_energy=True,
        track_flops=True,
        track_memory=True,
        log_every_n_steps=100,
        seed=42,
    )

    # Create system based on model name
    from computronium.core.registry import Registry

    system = Registry.to_system(
        args.model,
        input_dim=task.input_dim or 0,
        hidden_dim=args.hidden_dim if args.hidden_dim else 256,
        output_dim=task.output_dim,
        num_layers=2,
    )

    trainer = SystemTrainer(
        system=system,
        config=trainer_config,
        train_data=task.get_dataloader("train"),
        val_data=task.get_dataloader("val"),
    )

    try:
        from tqdm import tqdm

        pbar = tqdm(range(args.epochs), desc="Epochs")

        history = trainer.fit()
        for epoch_metric in history:
            pbar.update(1)
            pbar.set_postfix({
                "loss": epoch_metric.get("train_loss", 0.0),
                "acc": epoch_metric.get("train_acc", 0.0),
            })

        pbar.close()
        logger.info("[OK]  Training Complete")

    except KeyboardInterrupt:
        logger.warning("Training Interrupted")


def run_search(args):
    """Run a hyperparameter search (discovery protocol).

    One Optuna study is created per model (``{family}_{model}_{task}``) so that
    each study's categorical hyperparameter space is homogeneous (Optuna rejects
    per-trial choice-list changes within a study).  Trials per model = ``--budget``
    (or the tier default) — i.e. compute-matched within a family.  The legacy
    ``--models`` flag still works for ad-hoc per-model studies.
    """
    targets = _resolve_targets(args)
    if not targets:
        logger.error("No targets resolved. Use --family or --models.")
        return

    output = getattr(args, "output", None)
    for study_name, reg_family, cli_family, model_list in targets:
        start = time.time()
        studies = _run_hpo_family(study_name, reg_family, cli_family, model_list, args)
        logger.info("[DONE] %s in %.1fs", study_name, time.time() - start)
        if output:
            for model, study in studies:
                if study is not None:
                    _write_study_jsonl(study, f"{study_name}_{model}", output)


def run_core_train(args):
    """Run a single training session using SystemTrainer (unified interface)."""
    from computronium.core.registry import Registry
    from computronium.core.system_trainer import SystemTrainer, SystemTrainerConfig
    from computronium.domains.factory import create_task

    task = create_task(args.task, device=args.device, quick_mode=False)
    task.batch_size = args.batch_size
    task.setup()

    trainer_config = SystemTrainerConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        grad_clip=1.0,
        track_energy=not args.no_track_energy,
        track_flops=True,
        track_memory=True,
        log_every_n_steps=100,
        seed=42,
    )

    system = Registry.to_system(
        args.model,
        input_dim=task.input_dim or 0,
        hidden_dim=args.hidden_dim if args.hidden_dim else 256,
        output_dim=task.output_dim,
        num_layers=2,
    )

    trainer = SystemTrainer(
        system=system,
        config=trainer_config,
        train_data=task.get_dataloader("train"),
        val_data=task.get_dataloader("val"),
    )

    history = trainer.fit()

    if history:
        final = history[-1]
        logger.info(
            "Results: Train Acc=%.4f, Val Acc=%.4f",
            final.get("train_acc", 0.0),
            final.get("val_acc", 0.0),
        )


def run_from_yaml(args):
    """Run training from a YAML config file (flat preset format)."""
    import torch
    from omegaconf import OmegaConf

    from computronium.core.system_trainer import SystemTrainer, SystemTrainerConfig
    from computronium.domains.factory import create_task

    # Load YAML config
    cfg = OmegaConf.load(args.config)
    config = OmegaConf.to_container(cfg, resolve=True)

    # Extract components from flat YAML
    substrate_cfg = config.get("substrate", {})
    geometry_cfg = config.get("geometry", {})
    dynamics_cfg = config.get("dynamics", {})
    plasticity_cfg = config.get("plasticity", {})
    credit_cfg = config.get("credit", {})
    update_cfg = config.get("update", {})
    training_cfg = config.get("training", {})

    # Get training params - CLI --device overrides config
    device = getattr(args, "device", "auto")
    if device == "auto":
        device = training_cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = training_cfg.get("max_epochs", 10)
    batch_size = training_cfg.get("batch_size", 64)
    task_name = training_cfg.get("task", "mnist")

    # Create task and data loaders
    task = create_task(task_name, device=device, quick_mode=False)
    task.batch_size = batch_size
    task.setup()

    # Wrap data loaders to flatten input
    from torch.utils.data import DataLoader

    class _FlattenLoader:
        """Wrapper that flattens input tensors from a DataLoader."""

        def __init__(self, loader: DataLoader):
            self.loader = loader

        def __iter__(self):
            for x, y in self.loader:
                if x.dim() > 2:
                    x = x.view(x.size(0), -1)
                yield x, y

        def __len__(self) -> int:
            return len(self.loader)

    train_loader = _FlattenLoader(task.get_dataloader("train"))
    val_loader = _FlattenLoader(task.get_dataloader("val"))

    # Build system from flat config
    system = _build_system_from_flat_config(
        substrate_cfg,
        geometry_cfg,
        dynamics_cfg,
        plasticity_cfg,
        credit_cfg,
        update_cfg,
        device,
    )

    # Create trainer
    trainer_config = SystemTrainerConfig(
        max_epochs=epochs,
        batch_size=batch_size,
        device=device,
        grad_clip=training_cfg.get("grad_clip", 1.0),
        track_energy=training_cfg.get("track_energy", True),
        track_flops=training_cfg.get("track_flops", True),
        track_memory=training_cfg.get("track_memory", True),
        log_every_n_steps=training_cfg.get("log_every_n_steps", 100),
        seed=training_cfg.get("seed", 42),
        deterministic=training_cfg.get("deterministic", False),
    )

    trainer = SystemTrainer(
        system=system,
        config=trainer_config,
        train_data=train_loader,
        val_data=val_loader,
    )

    history = trainer.fit()

    if history:
        final = history[-1]
        logger.info(
            "Results: Train Acc=%.4f, Val Acc=%.4f",
            final.get("train_acc", final.get("val_acc", 0.0)),
            final.get("val_acc", 0.0),
        )


def _build_system_from_flat_config(
    substrate_cfg,
    geometry_cfg,
    dynamics_cfg,
    plasticity_cfg,
    credit_cfg,
    update_cfg,
    device,
):
    """Build a System or JointSystem from flat YAML config."""

    from computronium.core.ontology import (
        BackpropCredit,
        CreditAssignmentConfig,
        DigitalSubstrate,
        EnergyMinimizationDynamics,
        EuclideanUpdate,
        FeedforwardGeometry,
        GeometryConfig,
        InstantaneousDynamics,
        LocalGoodnessCredit,
        ParameterUpdateConfig,
        PredictiveSettlingDynamics,
        RandomProjectionsCredit,
        RecurrentGeometry,
        SpikeIntegrationDynamics,
        StateDynamicsConfig,
        SubstrateConfig,
        TargetInversionCredit,
        ThermodynamicContrast,
        TileGeometry,
        TemporalTraceCredit,
    )
    from computronium.core.plasticity import (
        FastWeightPlasticity,
        FastWeightPlasticityConfig,
        NullPlasticity,
        RoutingPlasticity,
        RoutingPlasticityConfig,
    )
    from computronium.core.system_trainer import compose_joint_system, compose_system

    # Build substrate
    substrate_precision = substrate_cfg.get("precision", "float32")
    substrate = DigitalSubstrate(
        SubstrateConfig(
            precision=substrate_precision,
            noise_level=substrate_cfg.get("noise_level", 0.0),
            weight_bounds=substrate_cfg.get("weight_bounds"),
            sparsity=substrate_cfg.get("sparsity", 0.0),
            device=device,
        )
    )

    # Build geometry
    geo_type = geometry_cfg.get("type", "feedforward")
    if geo_type == "recurrent":
        geometry = RecurrentGeometry(
            GeometryConfig.recurrent(
                input_dim=geometry_cfg["input_dim"],
                output_dim=geometry_cfg["output_dim"],
                hidden_dims=tuple(geometry_cfg["hidden_dims"]),
                init_scale=geometry_cfg.get("init_scale", 0.1),
            ),
            hidden_dim=geometry_cfg["hidden_dims"][-1]
            if geometry_cfg["hidden_dims"]
            else geometry_cfg["output_dim"],
        )
    elif geo_type == "tile_mesh":
        geometry = TileGeometry(
            GeometryConfig(
                input_dim=geometry_cfg["input_dim"],
                output_dim=geometry_cfg["output_dim"],
                hidden_dims=tuple(geometry_cfg["hidden_dims"]),
                num_layers=len(geometry_cfg["hidden_dims"]) + 1,
                topology_type="tile_mesh",
                connectivity=None,
                recurrent_weight=None,
                init_scale=geometry_cfg.get("init_scale", 0.1),
            ),
            neurons_per_tile=geometry_cfg.get("neurons_per_tile", 8),
            tiles_per_layer=geometry_cfg.get("tiles_per_layer", 2),
        )
    else:
        geometry = FeedforwardGeometry(
            GeometryConfig.feedforward(
                input_dim=geometry_cfg["input_dim"],
                output_dim=geometry_cfg["output_dim"],
                hidden_dims=tuple(geometry_cfg["hidden_dims"]),
                init_scale=geometry_cfg.get("init_scale", 0.1),
            )
        )

    # Build dynamics
    dyn_type = dynamics_cfg.get("type", "instantaneous")
    if dyn_type == "energy_minimization":
        dynamics = EnergyMinimizationDynamics(
            StateDynamicsConfig.energy_minimization(
                max_steps=dynamics_cfg.get("max_steps", 20),
                convergence_threshold=dynamics_cfg.get("convergence_threshold", 1e-4),
                convergence_start=dynamics_cfg.get("convergence_start", 5),
                step_size=dynamics_cfg.get("step_size", 0.1),
                beta=dynamics_cfg.get("beta", 0.5),
                momentum=dynamics_cfg.get("momentum", 0.0),
                track_free_energy_per_iter=dynamics_cfg.get(
                    "track_free_energy_per_iter", False
                ),
            )
        )
    elif dyn_type == "predictive_settling":
        dynamics = PredictiveSettlingDynamics(
            StateDynamicsConfig.predictive_settling(
                max_steps=dynamics_cfg.get("max_steps", 20),
                convergence_threshold=dynamics_cfg.get("convergence_threshold", 1e-4),
                convergence_start=dynamics_cfg.get("convergence_start", 5),
                step_size=dynamics_cfg.get("step_size", 0.1),
                beta=dynamics_cfg.get("beta", 0.5),
                momentum=dynamics_cfg.get("momentum", 0.0),
                track_free_energy_per_iter=dynamics_cfg.get(
                    "track_free_energy_per_iter", False
                ),
            )
        )
    elif dyn_type == "spike_integration":
        dynamics = SpikeIntegrationDynamics(
            StateDynamicsConfig.spike_integration(
                max_steps=dynamics_cfg.get("max_steps", 30),
                beta=dynamics_cfg.get("beta", 0.1),
            )
        )
    else:
        dynamics = InstantaneousDynamics(StateDynamicsConfig.instantaneous())

    # Build credit
    credit_type = credit_cfg.get("type", "backprop")
    if credit_type == "thermodynamic_contrast":
        credit = ThermodynamicContrast(
            CreditAssignmentConfig(
                credit_type="thermodynamic_contrast",
                beta=credit_cfg.get("beta", 0.5),
                feedback_matrix=None,
                local_objective=credit_cfg.get("local_objective", "mse"),
                orthogonal_init=credit_cfg.get("orthogonal_init", False),
                feedback_scale=credit_cfg.get("feedback_scale", 0.01),
            )
        )
    elif credit_type == "random_projections":
        credit = RandomProjectionsCredit(
            CreditAssignmentConfig(
                credit_type="random_projections",
                beta=credit_cfg.get("beta", 0.5),
                feedback_matrix=None,
                local_objective=credit_cfg.get("local_objective", "mse"),
                orthogonal_init=credit_cfg.get("orthogonal_init", False),
                feedback_scale=credit_cfg.get("feedback_scale", 0.01),
            )
        )
    elif credit_type == "target_inversion":
        credit = TargetInversionCredit(
            CreditAssignmentConfig.target_inversion(
                beta=credit_cfg.get("beta", 0.1),
                feedback_scale=credit_cfg.get("feedback_scale", 0.01),
            )
        )
    elif credit_type == "local_goodness":
        credit = LocalGoodnessCredit(
            CreditAssignmentConfig.local_goodness(
                feedback_scale=credit_cfg.get("feedback_scale", 0.01),
            )
        )
    elif credit_type == "temporal_trace":
        credit = TemporalTraceCredit(
            CreditAssignmentConfig.temporal_trace(
                feedback_scale=credit_cfg.get("feedback_scale", 0.01),
            )
        )
    else:
        credit = BackpropCredit(
            CreditAssignmentConfig(
                credit_type="gradient",
                beta=0.5,
                feedback_matrix=None,
                local_objective="mse",
                orthogonal_init=False,
                feedback_scale=0.01,
            )
        )

    # Build update
    update_type = update_cfg.get("type", "euclidean")
    if update_type == "euclidean":
        update = EuclideanUpdate(
            ParameterUpdateConfig.euclidean(
                step_size=update_cfg.get("step_size", 0.01),
                momentum=update_cfg.get("momentum", 0.9),
                ortho_steps=update_cfg.get("ortho_steps", 5),
                spectral_norm=update_cfg.get("spectral_norm", 1.0),
                fisher_damping=update_cfg.get("fisher_damping", 1e-3),
                ewc_lambda=update_cfg.get("ewc_lambda", 1000.0),
                grad_clip=update_cfg.get("grad_clip", 1.0),
            )
        )
    else:
        update = EuclideanUpdate(
            ParameterUpdateConfig.euclidean(
                step_size=update_cfg.get("step_size", 0.01),
                momentum=update_cfg.get("momentum", 0.9),
                ortho_steps=update_cfg.get("ortho_steps", 5),
                spectral_norm=update_cfg.get("spectral_norm", 1.0),
                fisher_damping=update_cfg.get("fisher_damping", 1e-3),
                ewc_lambda=update_cfg.get("ewc_lambda", 1000.0),
                grad_clip=update_cfg.get("grad_clip", 1.0),
            )
        )

    # Build plasticity
    plast_type = plasticity_cfg.get("type", "null")
    if plast_type == "routing":
        plasticity = RoutingPlasticity(
            gate_dim=plasticity_cfg.get("gate_dim", 64),
            temperature=plasticity_cfg.get("temperature", 1.0),
            top_k=plasticity_cfg.get("top_k"),
            decay=plasticity_cfg.get("decay", 0.99),
            learning_rate=plasticity_cfg.get("learning_rate", 0.01),
        )
    elif plast_type == "fast_weights":
        plasticity = FastWeightPlasticity(
            fast_weight_dim=plasticity_cfg.get("fast_weight_dim", 512),
            decay=plasticity_cfg.get("decay", 0.9),
            learning_rate=plasticity_cfg.get("learning_rate", 0.1),
            outer_product_scale=plasticity_cfg.get("outer_product_scale", 1.0),
        )
    else:
        plasticity = NullPlasticity()

    # Check if plasticity is null - use 5-D system, else use 6-D joint system
    if isinstance(plasticity, NullPlasticity):
        return compose_system(substrate, geometry, dynamics, credit, update)
    else:
        return compose_joint_system(
            substrate, geometry, dynamics, plasticity, credit, update
        )


def list_models(_args):
    """List all registered zoo models with metadata."""
    from computronium.core.registry import ComponentCategory, Registry

    models = Registry.list(ComponentCategory.MODEL)
    model_names = models.get("model", [])
    logger.info("Available Models (Zoo Registry):")
    for name in sorted(model_names):
        meta = Registry.get_metadata(ComponentCategory.MODEL, name)
        score = meta.bio_plausibility_score
        logger.info("  %-25s bio=%.1f", name, score)


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def _load_study_trials(study_name: str) -> list[dict]:
    """Load COMPLETE trials from an Optuna study as plain dicts.

    Uses Optuna's own storage tables (the study database) and enriches each
    trial with ``model_name`` / ``family`` / ``param_count`` /
    ``iteration_time`` from trial user attributes, falling back to the
    ``hyperopt_logs`` table when necessary.
    """
    try:
        study = optuna.load_study(study_name=study_name, storage=_STORAGE_URL)
    except (KeyError, OSError) as exc:
        logger.warning("Study '%s' not found in %s: %s", study_name, _STORAGE_URL, exc)
        return []

    trials: list[dict] = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        values = t.values or ()
        model_name = t.user_attrs.get("model_name", "unknown")
        family = t.user_attrs.get("family")
        if not family:
            try:
                family = get_model_spec(model_name).family or model_name
            except ValueError:
                family = model_name

        param_count = t.user_attrs.get("param_count")
        iteration_time = t.user_attrs.get("iteration_time")
        # Fall back to the hyperopt_logs table for legacy trials.
        if param_count is None or iteration_time is None:
            fallback = _read_hyperopt_logs(t.number)
            if param_count is None and fallback["param_count"] is not None:
                param_count = fallback["param_count"]
            if iteration_time is None and fallback["iteration_time"] is not None:
                iteration_time = fallback["iteration_time"]

        trials.append({
            "trial_id": t.number,
            "model_name": model_name,
            "family": family,
            "study_name": study_name,
            "accuracy": values[0] if len(values) > 0 else 0.0,
            "loss": values[1] if len(values) > 1 else 0.0,
            "param_count": param_count or 0.0,
            "iteration_time": iteration_time or 0.0,
            "config": dict(t.params),
        })
    logger.info("Loaded %d trials from study '%s'", len(trials), study_name)
    return trials


def _read_hyperopt_logs(trial_id: int) -> dict[str, float | None]:
    """Best-effort read of param_count/iteration_time from the legacy table."""
    import sqlite3

    try:
        conn = sqlite3.connect(_DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT param_count, iteration_time FROM hyperopt_logs WHERE trial_id = ?",
            (trial_id,),
        ).fetchone()
        conn.close()
    except sqlite3.Error:
        return {"param_count": None, "iteration_time": None}
    if row is None:
        return {"param_count": None, "iteration_time": None}
    return {"param_count": row["param_count"], "iteration_time": row["iteration_time"]}


def _discover_family_studies(family: str, task: str) -> list[str]:
    """Return all study names matching ``{family}_%_{task}`` in the SQLite store."""
    import sqlite3

    try:
        conn = sqlite3.connect(_DB_PATH)
        conn.row_factory = sqlite3.Row
        pattern = f"{family}_%_{task}"
        rows = conn.execute(
            "SELECT study_name FROM studies WHERE study_name LIKE ? ORDER BY study_name",
            (pattern,),
        ).fetchall()
        conn.close()
    except sqlite3.Error:
        return []
    return [r["study_name"] for r in rows]


def run_compare(args):
    """Rank families from completed HPO studies and emit a CSV."""
    if args.studies:
        study_names = [s.strip() for s in args.studies.split(",") if s.strip()]
    elif args.family and args.task:
        study_names = _discover_family_studies(args.family, args.task)
    else:
        study_names = []
    all_trials: list[dict] = []
    for sn in study_names:
        all_trials.extend(_load_study_trials(sn))

    if not all_trials:
        logger.error("No complete trials found in studies: %s", study_names)
        return

    grouped = group_trials_by_family(all_trials)
    metric = ComparisonMetric(args.metric)
    rankings = compute_algorithm_rankings(grouped, metric=metric)

    # Gap-to-baseline analysis (baseline = backprop family).
    baseline = next((r for r in rankings if "backprop" in r.family.lower()), None)
    if baseline is not None and baseline.best_value > 0:
        for r in rankings:
            r.gap_to_baseline = (
                (baseline.best_value - r.best_value) / baseline.best_value * 100
            )

    # Annotate pareto counts from the multi-objective frontier.
    pareto_ids = _study_pareto_ids(study_names)
    _annotate_pareto_counts(all_trials, rankings, pareto_ids)

    _write_rankings_csv(rankings, args.output)
    logger.info(generate_comparison_summary(rankings, baseline="backprop"))
    logger.info("[FILE]  Rankings written to %s", args.output)


def _study_pareto_ids(study_names: list[str]) -> set[int]:
    """Collect Pareto-frontier trial numbers across the given studies."""
    frontier_ids: set[int] = set()
    for sn in study_names:
        try:
            study = optuna.load_study(study_name=sn, storage=_STORAGE_URL)
        except KeyError, OSError:
            continue
        if len(study.directions) == 1:
            best = study.best_trial
            frontier_ids.add(best.number)
        else:
            for t in study.best_trials:
                frontier_ids.add(t.number)
    return frontier_ids


def _annotate_pareto_counts(
    all_trials: list[dict],
    rankings: list,
    pareto_ids: set[int],
) -> None:
    """Set ``pareto_count`` on each ranking from the frontier trial set."""
    family_pareto: dict[str, int] = {}
    for t in all_trials:
        if t["trial_id"] in pareto_ids:
            family_pareto[t["family"]] = family_pareto.get(t["family"], 0) + 1
    for r in rankings:
        r.pareto_count = family_pareto.get(r.family, 0)


def _write_rankings_csv(rankings: list, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank",
            "family",
            "best_value",
            "avg_value",
            "std_value",
            "n_trials",
            "best_trial_id",
            "gap_to_baseline_pct",
            "pareto_count",
        ])
        for r in rankings:
            writer.writerow([
                r.rank,
                r.family,
                f"{r.best_value:.6f}",
                f"{r.avg_value:.6f}",
                f"{r.std_value:.6f}",
                r.n_trials,
                r.best_trial_id,
                f"{r.gap_to_baseline:+.1f}",
                r.pareto_count,
            ])


# ---------------------------------------------------------------------------
# portfolio
# ---------------------------------------------------------------------------


def _family_localities(family: str) -> set[str]:
    """Distinct ``locality_level`` values across a registry family."""
    from computronium.core.registry import (
        ComponentCategory,
        ComponentMetadata,
        Registry,
    )

    metas: list[ComponentMetadata] = []
    for entry in Registry.query(category=ComponentCategory.MODEL, family=family):
        meta = entry.get("metadata")
        if isinstance(meta, ComponentMetadata):
            metas.append(meta)
    return {meta.locality_level.value for meta in metas}


def _family_task_trials(family: str, task: str) -> list[dict]:
    """Load all complete trials for every study of a family on a task."""
    trials: list[dict] = []
    for sn in _discover_family_studies(family, task):
        trials.extend(_load_study_trials(sn))
    return trials


def run_portfolio(  # ruff: ignore[too-many-locals]  (per-task/per-family bookkeeping)
    args,
):
    """Build a Phase 1 portfolio ranking table with survival decisions.

    Loads every family's tuned accuracy per task from the Optuna store, applies
    the registry-derived regime-advantage, and emits the ranking CSV with
    ``Scale`` / ``Hold`` / ``Eliminated`` statuses per the Phase 1.1 criterion.
    """
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not tasks:
        logger.error("--tasks must name at least one task")
        return

    per_task = {
        task: {rf: _family_task_trials(rf, task) for rf in set(FAMILY_MAP.values())}
        for task in tasks
    }
    # Drop empty slots; keep a stable family list from the union.
    for task in tasks:
        per_task[task] = {f: t for f, t in per_task[task].items() if t}

    baselines = {
        task: max(
            (t["accuracy"] for t in per_task[task].get("backprop", [])), default=0.0
        )
        for task in tasks
    }

    families: set[str] = set()
    families.add("backprop")
    for fams in per_task.values():
        families.update(fams)

    rows: list[dict[str, object]] = []
    for family in sorted(families):
        locality = _family_localities(family) if family != "backprop" else {"global"}
        wall_times = [
            t["iteration_time"]
            for task in tasks
            for t in per_task[task].get(family, [])
            if t.get("iteration_time")
        ]
        wall_time_s = float(sum(wall_times) / len(wall_times)) if wall_times else None
        best_accs = {
            task: max(
                (t["accuracy"] for t in per_task[task].get(family, [])), default=None
            )
            for task in tasks
        }

        # Phase 1.5: capture the param count of the best-accuracy trial so we can
        # rank families on efficiency (accuracy per 1k params), keeping the
        # comparison fair vs backprop's small models.
        _ACC_TOL = 1e-12

        def _best_trial_params(
            task_trials: list[dict], best: float | None
        ) -> float | None:
            if best is None:
                return None
            for t in task_trials:
                if abs(t["accuracy"] - best) < _ACC_TOL:
                    return t["param_count"]
            return None

        best_params = {
            task: _best_trial_params(per_task[task].get(family, []), best_accs[task])
            for task in tasks
        }
        primary = next((t for t in tasks if best_accs[t] is not None), tasks[0])
        primary_acc = best_accs[primary]
        if primary_acc is None:
            logger.warning("Family '%s' has no trials on any task; skipping", family)
            continue
        base = baselines[primary]
        row = PortfolioRow(
            family=family,
            best_acc=primary_acc,
            baseline_acc=base,
            locality=locality,
            wall_time_s=wall_time_s,
        )

        record: dict[str, object] = {
            "rank": 0,
            "family": family,
            "n_trials": sum(len(per_task[task].get(family, [])) for task in tasks),
            "wall_time_s": wall_time_s if wall_time_s is not None else "",
            "peak_mem": "O(1)"
            if locality & {"equilibrium", "forward-only", "local", "layerwise"}
            else "O(N)",
            "regime_advantage": (
                "baseline"
                if family == "backprop"
                else regime_advantage_label(family, locality)
            ),
            "status": "baseline" if family == "backprop" else row.status,
        }
        for task in tasks:
            task_acc = best_accs[task]
            record[f"acc_{task}"] = task_acc if task_acc is not None else ""
            record[f"parity_gap_{task}_pp"] = (
                round((baselines[task] - task_acc) * 100.0, 2)
                if task_acc is not None
                else ""
            )
            bp_params = best_params.get(task)
            record[f"params_at_best_{task}"] = (
                int(bp_params) if bp_params is not None else ""
            )
        # Efficiency: accuracy per 1k params at the best trial on the primary task.
        primary_params = best_params.get(primary)
        if primary_acc and primary_params:
            record["eff_acc_per_1k_params"] = round(
                primary_acc / (primary_params / 1000.0), 5
            )
        else:
            record["eff_acc_per_1k_params"] = ""
        rows.append(record)

    non_baseline = [r for r in rows if r["family"] != "backprop"]
    non_baseline.sort(key=lambda r: str(r["status"]) != "Scale")
    for idx, r in enumerate(non_baseline, 1):
        r["rank"] = idx
        _replace_in(rows, r)

    _write_portfolio_csv(rows, args.output)
    for r in rows:
        logger.info(
            "  %-4s %-18s status=%-10s regime=%-22s",
            r["rank"],
            r["family"],
            r["status"],
            r["regime_advantage"],
        )
    logger.info("[FILE]  Portfolio -> %s", args.output)


def _replace_in(rows: list[dict[str, object]], updated: dict[str, object]) -> None:
    """Mutate the matching row in-place so list identity is preserved."""
    for row in rows:
        if row["family"] == updated["family"]:
            row.update(updated)
            return


def _write_portfolio_csv(rows: list[dict[str, object]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def run_verify(  # ruff: ignore[too-many-locals]  (verification bookkeeping: seed, acc, CI + per-run metadata)
    args,
):
    """Re-run the top-k configs of a study with ``seeds`` different seeds."""
    try:
        study = optuna.load_study(study_name=args.study, storage=_STORAGE_URL)
    except (KeyError, OSError) as exc:
        logger.warning("Study '%s' not found: %s", args.study, exc)
        return

    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        logger.warning("No complete trials in study '%s'", args.study)
        return

    complete.sort(key=lambda t: (t.values or [0.0])[0], reverse=True)
    top_k = complete[: args.top_k]
    epochs = getattr(args, "epochs", None)
    base_seed = getattr(args, "seed", 42)

    records: list[dict] = []
    for t in top_k:
        model_name = t.user_attrs.get("model_name", "unknown")
        if not model_name or model_name == "unknown":
            logger.warning("Top trial #%d has no model_name attr; skipping", t.number)
            continue
        # Recover the original training protocol from attrs (set by the objective).
        trial_epochs = t.user_attrs.get("epochs", epochs or 5)
        trial_batch = t.user_attrs.get("batch_size", 32)
        task = t.user_attrs.get("task") or args.task

        logger.info(
            "[VERIFY] %s trial#%d  acc=%.4f  (re-running %d seeds)",
            model_name,
            t.number,
            (t.values or [0.0])[0],
            args.seeds,
        )
        seed_accs: list[float] = []
        for s in range(args.seeds):
            seed = base_seed + s * 1000
            _set_seeds(seed)
            config = dict(t.params)
            config["epochs"] = trial_epochs
            config["batch_size"] = trial_batch
            config["tier"] = "verify"
            config["is_verification"] = True
            config["verified_trial_id"] = t.number
            config["seed"] = seed
            quick = trial_epochs <= 3  # ruff: ignore[magic-value-comparison]  (smoke epochs threshold)
            metrics = run_single_trial_task(
                task=task,
                model_name=model_name,
                config=config,
                storage_path=_DB_PATH,
                quick_mode=quick,
                verbose=False,
            )
            acc = float(metrics["accuracy"]) if metrics else 0.0
            seed_accs.append(acc)
            records.append({
                "study": args.study,
                "original_trial": t.number,
                "model_name": model_name,
                "family": t.user_attrs.get("family", model_name),
                "seed": seed,
                "config": t.params,
                "accuracy": acc,
                "loss": float(metrics["loss"]) if metrics else None,
            })

        mean_acc = float(np.mean(seed_accs)) if seed_accs else 0.0
        std_acc = float(np.std(seed_accs)) if len(seed_accs) > 1 else 0.0
        ci95 = 1.96 * (std_acc / np.sqrt(len(seed_accs))) if len(seed_accs) > 1 else 0.0
        logger.info("   mean=%.4f std=%.4f 95%%CI=+/-%.4f", mean_acc, std_acc, ci95)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.output).open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        logger.info(
            "[FILE]  Verification results -> %s (%d runs)", args.output, len(records)
        )


# ---------------------------------------------------------------------------
# pareto
# ---------------------------------------------------------------------------


def run_pareto(args):
    """Generate Pareto frontier artefacts (plot + data) for a study."""
    try:
        study = optuna.load_study(study_name=args.study, storage=_STORAGE_URL)
    except (KeyError, OSError) as exc:
        logger.warning("Study '%s' not found: %s", args.study, exc)
        return

    trials = _load_study_trials(args.study)
    if not trials:
        logger.error("No trials to plot for study '%s'", args.study)
        return

    # Pareto frontier via Optuna (multi-objective best_trials).
    if len(study.directions) == 1:
        frontier = [study.best_trial]
    else:
        frontier = list(study.best_trials)
    frontier_ids = {t.number for t in frontier}
    logger.info("Pareto frontier: %d trials", len(frontier_ids))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd  # local: heavy import only when plotting

    rows = [
        {
            "model": t["model_name"],
            "param_count": t["param_count"],
            "val_accuracy": t["accuracy"],
            "val_loss": t["loss"],
            "is_pareto": t["trial_id"] in frontier_ids,
        }
        for t in trials
    ]
    df = pd.DataFrame(rows)

    metric = "val_accuracy" if "val_accuracy" in df.columns else "val_loss"
    try:
        fig = _plot_pareto(df, metric)
    except Exception:
        logger.exception("Plot generation failed; continuing with data export")
        fig = None

    fmt = getattr(args, "format", "html")
    stem = output_dir / args.study
    if fig is not None:
        png_path = Path(f"{stem}.png")
        fig.savefig(png_path, dpi=120, bbox_inches="tight", format="png")
        logger.info("[FILE]  Plot -> %s", png_path)
    frontier_data = [t for t in trials if t["trial_id"] in frontier_ids]
    json_path = Path(f"{stem}.json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "study": args.study,
                "n_trials": len(trials),
                "n_pareto": len(frontier_ids),
                "frontier": frontier_data,
            },
            f,
            indent=2,
            default=str,
        )
    logger.info("[FILE]  Pareto data -> %s", json_path)
    if fmt == "html":
        _write_pareto_html(frontier_data, stem)


def _plot_pareto(df: pd.DataFrame, metric: str):
    """Build an accuracy-vs-compute scatter with the Pareto frontier highlighted."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(df["model"].unique()), 1)))
    model_to_color = dict(zip(sorted(df["model"].unique()), colors))

    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model]
        ax.scatter(
            sub["param_count"],
            sub[metric],
            c=[model_to_color[model]],
            label=model,
            alpha=0.6,
            s=40,
        )

    front = df[df["is_pareto"]]
    if not front.empty:
        ax.scatter(
            front["param_count"],
            front[metric],
            facecolors="none",
            edgecolors="red",
            s=160,
            linewidths=2,
            label="Pareto frontier",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Parameter Count")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Pareto Frontier — {metric.replace('_', ' ').title()}")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    return fig


def _write_pareto_html(frontier_data, stem):
    """Write a standalone HTML page with the plot and a data table."""
    png_path = Path(f"{stem}.png")
    import base64

    if png_path.exists():
        b64 = base64.b64encode(png_path.read_bytes()).decode("ascii")
        img_tag = f'<img src="data:image/png;base64,{b64}" width="900"/>'
    else:
        img_tag = "<p>(plot unavailable)</p>"

    rows = "".join(
        f"<tr><td>{r['model_name']}</td><td>{r['accuracy']:.4f}</td>"
        f"<td>{r['loss']:.4f}</td><td>{r['param_count']:.0f}</td>"
        f"<td>{r['iteration_time']:.4f}</td></tr>"
        for r in frontier_data
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Pareto — {stem.name}</title></head>
<body>
<h1>Pareto Frontier — {stem.name}</h1>
<p>Frontier points: {len(frontier_data)}</p>
{img_tag}
<h2>Frontier trials</h2>
<table border="1" cellpadding="4"><tr>
<th>Model</th><th>Accuracy</th><th>Loss</th><th>Params</th><th>Time</th></tr>
{rows}
</table>
</body></html>"""
    html_path = Path(f"{stem}.html")
    html_path.write_text(html, encoding="utf-8")
    logger.info("[FILE]  HTML report -> %s", html_path)


def run_benchmark(args):
    """Run cross-domain benchmark suite."""
    from computronium.evaluation.cross_domain import CrossDomainBenchmarkSuite

    logger.info("[LAB]  Cross-Domain Benchmark Suite")

    models = None
    if args.models:
        models = [m.strip() for m in args.models.split(",")]

    from computronium.evaluation.cross_domain import BenchmarkSuiteConfig

    config = BenchmarkSuiteConfig(
        models=models,
        quick_mode=args.quick,
        intermediate_mode=args.intermediate,
        output_dir=args.output_dir,
        epochs=3 if args.quick else (10 if args.intermediate else 20),
        batch_size=64,
        track_energy=True,
    )

    suite = CrossDomainBenchmarkSuite(output_dir=args.output_dir)
    result = suite.run_suite(config)

    logger.info("Benchmark Results:")
    logger.info("   Total time: %.1fs", result.total_time_s)
    logger.info("   Results: %s benchmarks", len(result.results))

    if result.results:
        for r in result.results[:5]:
            logger.info("   - %s on %s: %s", r.model_name, r.task_name, r.metrics)

    suite.save_results(result)
    suite.generate_leaderboard()
    logger.info("[FILE]  Results saved to %s", args.output_dir)


def main():  # ruff: ignore[too-many-statements, complex-structure]  (argparse subparser registration is sequential by design)
    parser = argparse.ArgumentParser(description="Bioplausible Experiment Runner")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # ---- train ----
    train_parser = subparsers.add_parser(
        "train", help="Run training session or from YAML config"
    )
    train_parser.add_argument("--config", help="Path to YAML config file")
    train_parser.add_argument(
        "--model", help="Model name (required if not using --config)"
    )
    train_parser.add_argument(
        "--task", default="vision", choices=["vision", "lm", "rl"], help="Task type"
    )
    train_parser.add_argument("--dataset", help="Dataset name")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    # ---- core-train ----
    core_parser = subparsers.add_parser(
        "core-train", help="Train using CoreTrainer (new)"
    )
    core_parser.add_argument(
        "--model", default="backprop_mlp", help="Model name from Zoo registry"
    )
    core_parser.add_argument("--task", default="mnist", help="Task/dataset name")
    core_parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    core_parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    core_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    core_parser.add_argument("--optimizer", default="adam", help="Optimizer name")
    core_parser.add_argument(
        "--hidden-dim", type=int, default=256, help="Hidden dimension"
    )
    core_parser.add_argument(
        "--device", default="auto", help="Device (auto, cpu, cuda)"
    )
    core_parser.add_argument(
        "--no-track-energy", action="store_true", help="Disable energy tracking"
    )

    # ---- from-config ----
    config_parser = subparsers.add_parser(
        "from-config", help="Train from YAML config file"
    )
    config_parser.add_argument(
        "--config", required=True, help="Path to YAML config file"
    )
    config_parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Override device from config (auto, cpu, cuda)",
    )

    # ---- search ----
    search_parser = subparsers.add_parser(
        "search", help="Compute-matched HPO across a propagator family"
    )
    search_group = search_parser.add_mutually_exclusive_group()
    search_group.add_argument(
        "--family",
        choices=[*list(FAMILY_MAP), "all", "survivors"],
        help="Propagator family to search (one study per family).",
    )
    search_group.add_argument(
        "--models",
        help="Comma-separated model names (legacy per-model path)",
    )
    search_parser.add_argument(
        "--survivors-csv",
        default="results/portfolio.csv",
        help="Portfolio CSV read by --family survivors (Phase 1.2 gate)",
    )
    search_parser.add_argument(
        "--task",
        default="digits",
        choices=["digits", "cifar10", "tiny_shakespeare", "mnist"],
        help="Task/dataset name",
    )
    search_parser.add_argument(
        "--budget",
        type=int,
        default=0,
        help="Optuna trials per model (0 = use tier default)",
    )
    search_parser.add_argument(
        "--budget-tier",
        dest="tier",
        default="standard",
        choices=["smoke", "shallow", "standard", "deep"],
        help="Compute-matching tier (controls epochs, batch size, sampler warmup)",
    )
    search_parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Seeds for top-k verification (metadata only here; used by verify)",
    )
    search_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for the Optuna sampler + RNGs",
    )
    search_parser.add_argument(
        "--method",
        choices=["bayesian", "random"],
        default="bayesian",
        help="Optuna sampler: bayesian (TPE) or random",
    )
    search_parser.add_argument(
        "--device",
        default="auto",
        help="Device (auto, cpu, cuda)",
    )
    search_parser.add_argument(
        "--output", type=str, help="JSONL output path for trial records"
    )
    search_parser.add_argument(
        "--db",
        dest="db",
        help="SQLite DB file for this run (default: computronium.db)",
    )

    # ---- compare ----
    compare_parser = subparsers.add_parser(
        "compare", help="Rank families from completed HPO studies into a CSV"
    )
    compare_parser.add_argument(
        "--studies",
        required=False,
        default="",
        help="Comma-separated study names (e.g. eqprop_digits_eqprop_mlp,fa_digits_eqprop). "
        "If empty, --family/--task are used to discover studies.",
    )
    compare_parser.add_argument(
        "--family",
        help="Glob studies for this registry family (used when --studies omitted)",
    )
    compare_parser.add_argument(
        "--task",
        help="Task suffix to match when globbing via --family",
    )
    compare_parser.add_argument(
        "--metric",
        default="accuracy",
        choices=["accuracy", "loss", "param_efficiency"],
        help="Ranking metric",
    )
    compare_parser.add_argument("--output", required=True, help="Output CSV path")
    compare_parser.add_argument(
        "--db",
        dest="db",
        help="SQLite DB file containing the studies (default: computronium.db)",
    )

    # ---- verify ----
    verify_parser = subparsers.add_parser(
        "verify", help="Re-run top-k configs of a study with n seeds"
    )
    verify_parser.add_argument("--study", required=True, help="Study name to verify")
    verify_parser.add_argument(
        "--top-k", type=int, default=3, help="Number of top trials to re-run"
    )
    verify_parser.add_argument("--seeds", type=int, default=5, help="Seeds per config")
    verify_parser.add_argument(
        "--seed", type=int, default=42, help="Base seed for verification runs"
    )
    verify_parser.add_argument(
        "--epochs", type=int, default=None, help="Override epochs for verification"
    )
    verify_parser.add_argument(
        "--task", default="digits", help="Task name (fallback if not in study attrs)"
    )
    verify_parser.add_argument(
        "--output", type=str, help="JSONL output path for verified runs"
    )
    verify_parser.add_argument(
        "--db",
        dest="db",
        help="SQLite DB file containing the study (default: computronium.db)",
    )

    # ---- pareto ----
    pareto_parser = subparsers.add_parser(
        "pareto", help="Generate Pareto frontier plots/data for a study"
    )
    pareto_parser.add_argument("--study", required=True, help="Study name")
    pareto_parser.add_argument(
        "--output-dir", default="results/pareto", help="Output directory"
    )
    pareto_parser.add_argument(
        "--format",
        choices=["html", "png", "json"],
        default="html",
        help="Output format",
    )
    pareto_parser.add_argument(
        "--db",
        dest="db",
        help="SQLite DB file containing the study (default: computronium.db)",
    )

    # ---- portfolio ----
    portfolio_parser = subparsers.add_parser(
        "portfolio",
        help="Build Phase 1 portfolio ranking table (Scale/Hold/Eliminated)",
    )
    portfolio_parser.add_argument(
        "--tasks",
        default="digits,cifar10",
        help="Comma-separated task scopes to include (default: digits,cifar10)",
    )
    portfolio_parser.add_argument("--output", required=True, help="Output CSV path")
    portfolio_parser.add_argument(
        "--db",
        dest="db",
        help="SQLite DB file containing the studies (default: computronium.db)",
    )

    # ---- list ----
    subparsers.add_parser("list", help="List available models")

    # ---- benchmark ----
    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Run cross-domain benchmark suite"
    )
    benchmark_parser.add_argument(
        "--models",
        help="Comma-separated model names (default: all registered)",
    )
    benchmark_parser.add_argument(
        "--domains",
        help="Comma-separated domains (default: all)",
    )
    benchmark_parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode (3 epochs, smoke test)",
    )
    benchmark_parser.add_argument(
        "--intermediate",
        action="store_true",
        help="Intermediate mode (10 epochs)",
    )
    benchmark_parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Output directory for results",
    )

    args = parser.parse_args()

    if not getattr(args, "command", None):
        parser.print_help()
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )

    if getattr(args, "db", None):
        global _DB_PATH, _STORAGE_URL
        _DB_PATH, _STORAGE_URL = _set_storage(args.db)
        logger.info("Using storage: %s", _STORAGE_URL)

    if args.command == "train":
        run_training(args)
    elif args.command == "core-train":
        run_core_train(args)
    elif args.command == "from-config":
        run_from_yaml(args)
    elif args.command == "search":
        run_search(args)
    elif args.command == "compare":
        run_compare(args)
    elif args.command == "verify":
        run_verify(args)
    elif args.command == "pareto":
        run_pareto(args)
    elif args.command == "portfolio":
        run_portfolio(args)
    elif args.command == "benchmark":
        run_benchmark(args)
    elif args.command == "list":
        list_models(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
