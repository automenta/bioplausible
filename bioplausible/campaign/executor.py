"""Campaign execution: HPO loop, trial running, and checkpointing (FIX2a §5, §8).

The :class:`CampaignExecutor` runs the complete experiment pipeline:
1. TIER 0 / 0.5 gates (model triage)
2. For each surviving model x arm: Optuna HPO study
3. Per-trial training via :class:`CoreTrainer`
4. JSONL event logging + SQLite study persistence for resume
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import optuna

from bioplausible.campaign.logger import (
    Epoch,
    ExperimentLogger,
    GateOutcome,
    TrialEnd,
    TrialStart,
)
from bioplausible.campaign.runner import CampaignResult, CampaignRunner, run_gates
from bioplausible.core.trainer import CoreTrainer, TrainerConfig

if TYPE_CHECKING:
    from bioplausible.campaign.schema import Campaign

logger = logging.getLogger(__name__)

_OPTIMIZER_KWARGS: frozenset[str] = frozenset({"lr", "weight_decay", "beta", "momentum"})

__all__ = [
    "CampaignExecutor",
    "TrialContext",
    "run_campaign",
]


@dataclass(frozen=True, slots=True)
class TrialContext:
    """Context for a single trial execution."""

    trial_id: int
    arm_name: str
    model_name: str
    protocol: str
    seed: int
    device: str
    epochs: int
    output_dir: Path
    task: str
    tags: dict[str, object] = field(default_factory=dict)


class CampaignExecutor:
    """Execute a full campaign: gates → HPO → analysis artifacts."""

    def __init__(
        self,
        campaign: Campaign,
        output_dir: Path | None = None,
        study_name: str | None = None,
        storage_base: str | None = None,
    ) -> None:
        """
        Args:
            campaign: Validated campaign definition.
            output_dir: Base directory for artifacts (JSONL, checkpoints, studies).
                Defaults to ``campaign.output.artifacts_dir``.
            study_name: Base Optuna study name. Defaults to ``campaign.meta.name``.
            storage_base: Base directory for Optuna SQLite storage files.
                Defaults to ``{output_dir}/studies/``.
        """
        self.campaign = campaign
        self.output_dir = output_dir or Path(campaign.output.artifacts_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.study_name = study_name or campaign.meta.name
        self.storage_base = Path(storage_base or self.output_dir / "studies")
        self.storage_base.mkdir(parents=True, exist_ok=True)

        self._runner = CampaignRunner(campaign, self.output_dir)
        self._gate_result: CampaignResult | None = None
        self._studies: dict[str, optuna.Study] = {}

    @property
    def gate_result(self) -> CampaignResult:
        if self._gate_result is None:
            self._gate_result = self._run_gates()
        return self._gate_result

    def _run_gates(self) -> CampaignResult:
        """Run TIER 0 / 0.5 gates and log outcomes."""
        logger.info("Running gate tiers for campaign %s", self.campaign.meta.name)
        result = run_gates(
            self.campaign,
            device=self.campaign.compute.device,
            n_seeds=self.campaign.hpo.n_seeds,
            min_accuracy=getattr(self.campaign.hpo, "min_accuracy", 0.95),
        )

        # Log gate outcomes
        gates_log = self.output_dir / "gates.jsonl"
        with ExperimentLogger(gates_log) as log:
            for tier in ("tier0", "tier0.5"):
                for outcome in result.tiers.get(tier, []):
                    log.log(
                        GateOutcome(
                            tier=outcome.tier,
                            model=outcome.model,
                            task=outcome.task,
                            passed=outcome.passed,
                            reason=outcome.reason,
                            metrics=outcome.metrics,
                        )
                    )
        logger.info("Gate outcomes written to %s", gates_log)
        return result

    def get_surviving_models(self) -> list[tuple[str, str]]:
        """Return list of (arm_name, model_name) that passed all gates."""
        survivors: list[tuple[str, str]] = []
        tier0_passed = {o.model for o in self.gate_result.tiers.get("tier0", []) if o.passed}
        tier05_passed = {o.model for o in self.gate_result.tiers.get("tier0.5", []) if o.passed}

        for arm in self._runner.plan():
            for model in arm.models:
                if model in tier0_passed:
                    # If TIER 0.5 ran, model must also pass it
                    if self.gate_result.tiers.get("tier0.5") and model not in tier05_passed:
                        continue
                    survivors.append((arm.name, model))
        return survivors

    def _get_or_create_study(self, arm_name: str, model_name: str) -> optuna.Study:
        key = f"{arm_name}/{model_name}"
        if key not in self._studies:
            storage_url = f"sqlite:///{self.storage_base}/{key.replace('/', '_')}.db"
            self._studies[key] = optuna.create_study(
                study_name=f"{self.study_name}_{key}",
                storage=storage_url,
                load_if_exists=True,
                directions=["maximize", "minimize"],  # accuracy, param_count
                sampler=optuna.samplers.NSGAIISampler(
                    seed=self.campaign.reproducibility.seed
                ),
            )
        return self._studies[key]

    _OPTIMIZER_KWARGS: frozenset[str] = frozenset({"lr", "weight_decay", "beta", "momentum"})

    def _build_trial_config(
        self,
        trial: optuna.Trial,
        arm_name: str,
        model_name: str,
    ) -> TrainerConfig:
        """Build a TrainerConfig from a sampled trial."""
        from bioplausible.campaign.param_estimator import bound_estimator

        search_space = self.campaign.build_search_space()
        input_dim = self.campaign.arm_input_dim(arm_name)
        output_dim = self.campaign.arm_output_dim(arm_name)
        max_params = self.campaign.arms[arm_name].max_params

        config_dict = search_space.sample_feasible(
            trial,
            model_name,
            estimator=bound_estimator(model_name, input_dim, output_dim),
            max_params=max_params,
        )
        if config_dict is None:
            raise optuna.TrialPruned  # ruff: ignore[raise-vanilla-args]  # Optuna expects bare TrialPruned

        # Merge sampled config with fixed model_overrides and constants
        full_config = dict(search_space.defaults)
        full_config.update(search_space.constants.get(model_name, {}))
        full_config.update(config_dict)

        # Build model kwargs using the estimator's signature filter
        from bioplausible.campaign.param_estimator import build_model_kwargs
        from bioplausible.core.registry import ComponentCategory, Registry

        model_cls = Registry.get(ComponentCategory.MODEL, model_name)
        model_kwargs = build_model_kwargs(
            model_cls,
            full_config,
            input_dim=input_dim,
            output_dim=output_dim,
            model_name=model_name,
        )

        # Compute param count for logging
        param_count = sum(p.numel() for p in model_cls(**model_kwargs).parameters())

        protocol = self.campaign.protocols.resolve(model_name)
        propagator = protocol if protocol != "end2end" else None

        return TrainerConfig(
            model=model_name,
            model_kwargs=model_kwargs,
            propagator=propagator,
            optimizer=full_config.get("optimizer", "adam"),
            optimizer_kwargs={k: v for k, v in full_config.items() if k in _OPTIMIZER_KWARGS},
            task=self._resolve_task_for_arm(arm_name),
            epochs=self.campaign.tasks[0].epochs if self.campaign.tasks else 10,
            batch_size=full_config.get("batch_size", 128),
            num_workers=0,
            seed=self.campaign.reproducibility.seed + trial.number,
            device=self.campaign.compute.device,
            track_energy=getattr(self.campaign.compute, "track_energy", False),
            track_flops=getattr(self.campaign.compute, "track_flops", False),
            track_memory=getattr(self.campaign.compute, "track_memory", False),
            save_checkpoints=True,
            checkpoint_dir=str(self.output_dir / "checkpoints" / arm_name / model_name / f"trial_{trial.number}"),
            extra=full_config,
        ), param_count

    def _resolve_task_for_arm(self, arm_name: str) -> str:
        """Return the task name for the given arm.

        Uses the first task in the campaign that matches the arm's geometry,
        or the first task as fallback. The task name is passed to CoreTrainer
        which maps it to the appropriate dataset loader.
        """
        arm = self.campaign.arms[arm_name]
        arm_input_dim = self.campaign.arm_input_dim(arm_name)
        arm_output_dim = self.campaign.arm_output_dim(arm_name)

        for task in self.campaign.tasks:
            task_input_dim = task.input_dim or arm_input_dim
            task_num_classes = task.num_classes or arm_output_dim
            if task_input_dim == arm_input_dim and task_num_classes == arm_output_dim:
                return task.name

        # Fallback: use first task name, or generate one from geometry
        if self.campaign.tasks:
            return self.campaign.tasks[0].name
        return f"classification_{arm_input_dim}_{arm_output_dim}"

    def _objective(
        self,
        trial: optuna.Trial,
        arm_name: str,
        model_name: str,
        logger: ExperimentLogger,
        trial_id: int,
    ) -> tuple[float, float]:
        """Optuna objective function for a single trial.

        Returns:
            Tuple of (accuracy, -param_count) for multi-objective optimization.
        """
        ctx = TrialContext(
            trial_id=trial_id,
            arm_name=arm_name,
            model_name=model_name,
            protocol=self.campaign.protocols.resolve(model_name),
            seed=self.campaign.reproducibility.seed + trial.number,
            device=self.campaign.compute.device,
            epochs=self.campaign.tasks[0].epochs if self.campaign.tasks else 10,
            output_dir=self.output_dir / "trials" / arm_name / model_name / f"trial_{trial.number}",
            task=self._resolve_task_for_arm(arm_name),
        )

        # Build trial config
        try:
            config, param_count = self._build_trial_config(trial, arm_name, model_name)
        except optuna.TrialPruned:
            raise
        except Exception:
            logger.log(
                TrialEnd(
                    trial_id=trial_id,
                    status="config_error",
                    metrics={},
                    wall_time_s=0.0,
                )
            )
            raise

        # Log trial start
        logger.log(
            TrialStart(
                trial_id=trial_id,
                model=model_name,
                task=ctx.task,
                arm=arm_name,
                config=config.to_dict(),
                param_count=param_count,
                seed=ctx.seed,
            )
        )

        # Store param_count in trial for later analysis
        trial.set_user_attr("param_count", param_count)
        trial.set_user_attr("arm_name", arm_name)
        trial.set_user_attr("model_name", model_name)

        # Run training
        trainer = CoreTrainer(config)
        start_time = time.time()

        try:
            history = trainer.fit()
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.log(
                TrialEnd(
                    trial_id=trial_id,
                    status="failed",
                    metrics={"error": str(e)},
                    wall_time_s=time.time() - start_time,
                )
            )
            raise

        wall_time = time.time() - start_time

        # Log per-epoch events
        for epoch_metrics in history:
            logger.log(
                Epoch(
                    trial_id=trial_id,
                    epoch=epoch_metrics.epoch,
                    metrics=epoch_metrics.to_dict(),
                )
            )

        # Log trial end
        final_metrics = history[-1].to_dict() if history else {}
        logger.log(
            TrialEnd(
                trial_id=trial_id,
                status="completed",
                metrics=final_metrics,
                wall_time_s=wall_time,
            )
        )

        # Return objectives: (accuracy, -param_count) for maximization
        if history:
            accuracy = float(history[-1].val_accuracy or history[-1].train_accuracy)
        else:
            accuracy = 0.0
        return accuracy, -float(param_count)

    def run_arm_model(
        self,
        arm_name: str,
        model_name: str,
        n_trials: int | None = None,
    ) -> optuna.Study:
        """Run HPO for a single arm × model combination."""
        n_trials = n_trials or self.campaign.hpo.n_trials
        trial_log = self.output_dir / "trials" / arm_name / model_name / "trials.jsonl"
        trial_log.parent.mkdir(parents=True, exist_ok=True)

        with ExperimentLogger(trial_log) as logger:
            study = self._get_or_create_study(arm_name, model_name)

            def objective(trial: optuna.Trial) -> tuple[float, float]:
                trial_id = trial.number
                return self._objective(trial, arm_name, model_name, logger, trial_id)

            study.optimize(objective, n_trials=n_trials, catch=(Exception,), gc_after_trial=True)

        return study

    def run(self, n_trials: int | None = None) -> dict[str, optuna.Study]:
        """Run the full campaign: gates → HPO for each surviving model."""
        logger.info("Starting campaign %s", self.campaign.meta.name)

        # Step 1: Run gates (already logged in gate_result property)
        survivors = self.get_surviving_models()
        logger.info("Surviving models after gates: %s", survivors)

        if not survivors:
            logger.warning("No models survived gates; campaign complete")
            return {}

        # Step 2: Run HPO for each survivor
        studies: dict[str, optuna.Study] = {}
        for arm_name, model_name in survivors:
            key = f"{arm_name}/{model_name}"
            logger.info("Running HPO for %s (%d trials)", key, n_trials or self.campaign.hpo.n_trials)
            study = self.run_arm_model(arm_name, model_name, n_trials)
            studies[key] = study
            logger.info("Best value for %s: accuracy=%.4f", key, study.best_trials[0].values[0] if study.best_trials else 0.0)

        logger.info("Campaign %s complete", self.campaign.meta.name)
        return studies


def run_campaign(
    campaign: Campaign,
    output_dir: Path | None = None,
    n_trials: int | None = None,
    study_name: str | None = None,
    storage_base: str | None = None,
) -> dict[str, optuna.Study]:
    """Convenience function to run a complete campaign.

    Args:
        campaign: Validated campaign definition.
        output_dir: Base artifacts directory.
        n_trials: Override number of trials per model (default from campaign HPO config).
        study_name: Optuna study name.
        storage_base: Base directory for Optuna SQLite storage files.

    Returns:
        Mapping of ``"arm/model"`` to completed Optuna studies.
    """
    executor = CampaignExecutor(campaign, output_dir, study_name, storage_base)
    return executor.run(n_trials)
