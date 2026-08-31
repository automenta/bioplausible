"""Guard and validation logic for AutoScientist execution.

Consolidates safety and validation concerns:
- SafetyWrapper: NaN/Inf/gradient explosion protection during training
- Algorithm constraints: Hyperparameter bounds per algorithm family
- Experiment checks: Decision helpers for verification/CV/ablation/transfer/continual/low-data/robustness
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import torch

from computronium.core.logging import get_logger
from computronium.execution.task import ExperimentTask
from computronium.hyperopt.eval_tiers import PatientLevel
from computronium.core.model_spec import get_model_spec

__all__ = [
    "ALGORITHM_FAMILY_CONSTRAINTS",
    "SafetyConfig",
    "SafetyWrapper",
    "check_ablation_needed",
    "check_continual_learning_needed",
    "check_cv_needed",
    "check_low_data_needed",
    "check_robustness_needed",
    "check_transfer_needed",
    "check_verification_needed",
    "create_constrained_optuna_config",
    "get_constrained_search_space",
    "get_stats",
    "logger",
    "suggest_hyperparam",
]
logger = get_logger()

# =============================================================================
# Safety
# =============================================================================


@dataclass(frozen=True, slots=True)
class SafetyConfig:
    """Safety configuration for training."""

    max_grad_norm: float = 10.0
    nan_check_frequency: int = 10
    lr_reduction_on_nan: float = 0.5
    max_nan_retries: int = 3
    enable_anomaly_detection: bool = False


class SafetyWrapper:
    """Wraps training to catch and handle numerical instabilities."""

    def __init__(self, config: SafetyConfig | None = None):
        self.config = config or SafetyConfig()
        self.consecutive_failures = 0
        self.total_failures = 0
        self.step_count = 0

        if self.config.enable_anomaly_detection:
            torch.autograd.set_detect_anomaly(True)

    def safe_backward_and_step(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        clip_norm: float | None = None,
    ) -> tuple[bool, dict[str, object]]:
        self.step_count += 1

        if not torch.isfinite(loss):
            self.consecutive_failures += 1
            self.total_failures += 1
            return False, {
                "error": "loss_nan_or_inf",
                "loss_value": float(loss),
                "step": self.step_count,
            }

        try:
            loss.backward()
        except RuntimeError as e:
            self.consecutive_failures += 1
            self.total_failures += 1
            optimizer.zero_grad()
            return False, {
                "error": "backward_failed",
                "exception": str(e),
                "step": self.step_count,
            }

        total_norm = 0.0
        has_nan = False
        nan_param_names: list[str] = []

        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                if not torch.isfinite(param_norm):
                    has_nan = True
                    nan_param_names.append(name)
                    break
                total_norm += param_norm.item() ** 2

        if has_nan:
            self.consecutive_failures += 1
            self.total_failures += 1
            optimizer.zero_grad()
            logger.warning("NaN gradient detected in parameters: %s", nan_param_names)
            return False, {
                "error": "grad_nan",
                "grad_norm": float("nan"),
                "nan_params": nan_param_names,
                "step": self.step_count,
            }

        total_norm = total_norm**0.5

        clip_value = clip_norm if clip_norm is not None else self.config.max_grad_norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        try:
            optimizer.step()
            optimizer.zero_grad()
        except RuntimeError as e:
            self.consecutive_failures += 1
            self.total_failures += 1
            return False, {
                "error": "optimizer_step_failed",
                "exception": str(e),
                "step": self.step_count,
            }

        self.consecutive_failures = 0
        return True, {
            "grad_norm": total_norm,
            "loss": float(loss),
            "step": self.step_count,
        }

    def should_abort(self) -> bool:
        return self.consecutive_failures >= self.config.max_nan_retries

    def handle_failure(self, optimizer: torch.optim.Optimizer) -> None:
        for param_group in optimizer.param_groups:
            old_lr = param_group["lr"]
            new_lr = old_lr * self.config.lr_reduction_on_nan
            param_group["lr"] = new_lr
            logger.warning(
                f"Reduced LR from {old_lr:.2e} to {new_lr:.2e} (failure {self.consecutive_failures}/{self.config.max_nan_retries})"
            )

    def get_stats(self) -> dict[str, object]:
        return {
            "total_steps": self.step_count,
            "total_failures": self.total_failures,
            "consecutive_failures": self.consecutive_failures,
            "failure_rate": (
                self.total_failures / self.step_count if self.step_count > 0 else 0.0
            ),
        }


# =============================================================================
# Algorithm-Specific Hyperparameter Constraints
# =============================================================================

ALGORITHM_FAMILY_CONSTRAINTS: dict[str, dict[str, object]] = {
    "baseline": {
        "lr": (1e-5, 1e-2, "log"),
        "grad_clip": (0.5, 10.0, "linear"),
        "weight_decay": (0.0, 1e-2, "log"),
        "dropout": (0.0, 0.5, "linear"),
        "momentum": (0.0, 0.99, "linear"),
        "optimizer": ["sgd", "adam", "adamw", "rmsprop"],
        "hidden_dim": [32, 64, 128, 256, 512],
        "num_layers": (1, 4, "int"),
    },
    "eqprop": {
        "lr": (1e-6, 5e-4, "log"),
        "beta": (0.01, 0.5, "linear"),
        "steps": (10, 40, "int"),
        "grad_clip": (1.0, 5.0, "linear"),
        "nudge_type": ["output_clamping", "energy_based", "symmetric"],
        "hidden_dim": [32, 64, 128],
        "num_layers": (2, 6, "int"),
    },
    "hebbian": {
        "lr": (1e-5, 1e-3, "log"),
        "contrastive_steps": (5, 30, "int"),
        "grad_clip": (1.0, 10.0, "linear"),
        "hidden_dim": [64, 128],
        "num_layers": (2, 4, "int"),
    },
    "hybrid": {
        "lr": (1e-5, 5e-3, "log"),
        "grad_clip": (0.5, 10.0, "linear"),
        "fa_scale": (0.5, 2.0, "linear"),
        "adapt_rate": (1e-4, 1e-1, "log"),
        "hidden_dim": [64, 128, 256],
        "num_layers": (2, 5, "int"),
    },
}


def get_constrained_search_space(model_name: str) -> dict[str, object]:
    try:
        model_spec = get_model_spec(model_name)
        family = model_spec.family.lower()
    except KeyError, AttributeError, ValueError:
        logger.warning(
            "Could not determine family for %s, using baseline constraints",
            model_name,
        )
        family = "baseline"

    constraints = ALGORITHM_FAMILY_CONSTRAINTS.get(
        family, ALGORITHM_FAMILY_CONSTRAINTS["baseline"]
    )
    logger.info("Using %s constraints for %s", family, model_name)
    return constraints


def suggest_hyperparam(
    trial, param_name: str, constraint, prefix: str = ""
) -> float | int | str:
    full_name = f"{prefix}{param_name}" if prefix else param_name

    if isinstance(constraint, list):
        return trial.suggest_categorical(full_name, constraint)
    if isinstance(constraint, tuple) and len(constraint) == 3:
        min_val, max_val, scale = constraint
        if scale == "log":
            return trial.suggest_float(full_name, min_val, max_val, log=True)
        if scale == "int":
            return trial.suggest_int(full_name, int(min_val), int(max_val))
        return trial.suggest_float(full_name, min_val, max_val, log=False)
    raise ValueError(f"Invalid constraint format for {param_name}: {constraint}")


def create_constrained_optuna_config(
    trial,
    model_name: str,
    custom_constraints: dict[str, object] | None = None,
    task_name: str | None = None,
) -> dict[str, object]:
    from computronium.hyperopt.optuna_bridge import create_optuna_space

    final_constraints = custom_constraints.copy() if custom_constraints else {}

    if task_name == "cifar100":
        if "hidden_dim" not in final_constraints:
            final_constraints["hidden_dim"] = [128, 256, 512, 1024]
        if "num_layers" not in final_constraints:
            final_constraints["min_num_layers"] = 3
            final_constraints["max_num_layers"] = 8

    if "max_hidden_dim" in final_constraints:
        max_dim = final_constraints["max_hidden_dim"]
        if "hidden_dim" in final_constraints and isinstance(
            final_constraints["hidden_dim"], list
        ):
            filtered = [d for d in final_constraints["hidden_dim"] if d <= max_dim]
            if not filtered:
                filtered = [max_dim]
            final_constraints["hidden_dim"] = filtered

    return create_optuna_space(
        trial=trial,
        model_name=model_name,
        constraints=final_constraints,
        task_name=task_name,
    )


# =============================================================================
# Experiment Check Functions
# =============================================================================


def get_stats(
    progress: dict, model: str, task: str, tier: PatientLevel
) -> dict[str, object]:
    try:
        return progress[model][task][tier.value]
    except KeyError:
        return {"count": 0, "best_acc": 0.0, "trials": []}


def check_verification_needed(
    stats: dict,
    model: str,
    task: str,
    tier: PatientLevel,
    check_criterion_fn,
) -> ExperimentTask | None:
    """Check if verification trials are needed for a configuration."""
    trials = stats.get("trials", [])
    if not trials:
        return None

    trials.sort(key=lambda x: x.accuracy, reverse=True)
    best_trial = trials[0]

    if not check_criterion_fn(tier, task, best_trial.accuracy):
        return None

    repeats = 0
    target_config = {
        k: v
        for k, v in best_trial.config.items()
        if k not in ["tier", "task", "model", "epochs", "batch_size", "job_id", "fold"]
    }

    target_hash = hashlib.md5(
        json.dumps(target_config, sort_keys=True).encode()
    ).hexdigest()

    for t in trials:
        t_conf = {
            k: v
            for k, v in t.config.items()
            if k
            not in [
                "tier",
                "task",
                "model",
                "epochs",
                "batch_size",
                "job_id",
                "fold",
            ]
        }
        if (
            hashlib.md5(json.dumps(t_conf, sort_keys=True).encode()).hexdigest()
            == target_hash
        ):
            repeats += 1

    if repeats < 3:
        priority = 90.0 + best_trial.accuracy * 10.0
        config_copy = best_trial.config.copy()
        return ExperimentTask(
            model_name=model,
            task_name=task,
            tier=tier,
            study_name=f"{model}_{task}_{tier.value}",
            priority=priority,
            fixed_config=config_copy,
            verification_of_trial_id=best_trial.trial_id,
        )

    return None


def _compute_config_hash(config: dict) -> str:
    """Compute a hash of the experiment config for dedup."""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()


def check_cv_needed(
    std_stats: dict,
    progress: dict,
    model: str,
    task: str,
) -> ExperimentTask | None:
    """Check if 5-fold cross-validation is needed."""
    trials = std_stats.get("trials", [])
    if not trials:
        return None

    trials.sort(key=lambda x: x.accuracy, reverse=True)
    best_trial = trials[0]

    repeats = 0
    target_config = {
        k: v
        for k, v in best_trial.config.items()
        if k not in ["tier", "task", "model", "epochs", "batch_size", "job_id", "fold"]
    }
    target_hash = hashlib.md5(
        json.dumps(target_config, sort_keys=True).encode()
    ).hexdigest()

    for t in trials:
        t_conf = {
            k: v
            for k, v in t.config.items()
            if k
            not in ["tier", "task", "model", "epochs", "batch_size", "job_id", "fold"]
        }
        if (
            hashlib.md5(json.dumps(t_conf, sort_keys=True).encode()).hexdigest()
            == target_hash
        ):
            repeats += 1

    if repeats < 3:
        return None

    cv_stats = get_stats(progress, model, task, PatientLevel.CROSS_VAL)
    cv_trials = cv_stats.get("trials", [])

    completed_folds = set()
    for t in cv_trials:
        t_conf = {
            k: v
            for k, v in t.config.items()
            if k
            not in [
                "tier",
                "task",
                "model",
                "epochs",
                "batch_size",
                "job_id",
                "fold",
                "is_verification",
                "verified_trial_id",
            ]
        }
        if (
            hashlib.md5(json.dumps(t_conf, sort_keys=True).encode()).hexdigest()
            == target_hash
        ):
            fold = t.config.get("fold")
            if fold is not None:
                completed_folds.add(fold)

    for fold in range(5):
        if fold not in completed_folds:
            config_copy = best_trial.config.copy()
            priority = 95.0

            return ExperimentTask(
                model_name=model,
                task_name=task,
                tier=PatientLevel.CROSS_VAL,
                study_name=f"{model}_{task}_{PatientLevel.CROSS_VAL.value}",
                priority=priority,
                fixed_config=config_copy,
                verification_of_trial_id=best_trial.trial_id,
                fold_index=fold,
            )

    return None


def check_continual_learning_needed(
    stats: dict,
    progress: dict,
    model: str,
    task: str,
) -> ExperimentTask | None:
    """Schedule next steps in a Split-MNIST Continual Learning sequence."""
    if task != "mnist":
        return None

    if stats["count"] == 0 or stats["best_acc"] < 0.95:
        return None

    steps = [
        ("mnist_01", 0),
        ("mnist_23", 1),
        ("mnist_45", 2),
        ("mnist_67", 3),
        ("mnist_89", 4),
    ]

    previous_trial_id = None

    for i, (step_task, step_idx) in enumerate(steps):
        step_stats = get_stats(progress, model, step_task, PatientLevel.STANDARD)

        if step_stats["count"] == 0:
            config_copy: dict[str, object] = {}

            if step_idx > 0:
                if previous_trial_id is None:
                    return None

                prev_task_name = steps[i - 1][0]
                prev_stats = get_stats(
                    progress, model, prev_task_name, PatientLevel.STANDARD
                )
                best_prev = max(prev_stats["trials"], key=lambda t: t.accuracy)
                config_copy = best_prev.config.copy()
                config_copy["transfer_from"] = best_prev.trial_id
                config_copy["freeze_layers"] = False
            else:
                best_mnist = max(stats["trials"], key=lambda t: t.accuracy)
                config_copy = best_mnist.config.copy()

            config_copy["is_continual"] = True
            config_copy["continual_step"] = step_idx

            return ExperimentTask(
                model_name=model,
                task_name=step_task,
                tier=PatientLevel.STANDARD,
                study_name=f"{model}_mnist_cl_step{step_idx}",
                priority=98.0 + (step_idx * 0.1),
                fixed_config=config_copy,
                is_continual=True,
                continual_step=step_idx,
                transfer_from_trial=config_copy.get("transfer_from"),
            )

        best_step_trial = max(step_stats["trials"], key=lambda t: t.accuracy)
        if best_step_trial.accuracy < 0.80:
            return None

        previous_trial_id = best_step_trial.trial_id

    return None


def check_transfer_needed(
    stats: dict,
    progress: dict,
    model: str,
    task: str,
    curriculum,
) -> ExperimentTask | None:
    """Check if transfer learning experiment should be scheduled."""
    trials = stats.get("trials", [])
    if not trials:
        return None

    trials.sort(key=lambda x: x.accuracy, reverse=True)
    best_trial = trials[0]

    if best_trial.accuracy < 0.85:
        return None

    next_task = curriculum.get_next_task(model, task, success=True)

    if not next_task or next_task == "completed_track":
        return None

    target_stats = get_stats(progress, model, next_task, PatientLevel.STANDARD)

    already_done = False
    for t in target_stats.get("trials", []):
        if t.config.get("transfer_from") == best_trial.trial_id:
            already_done = True
            break

    if not already_done:
        config_copy = best_trial.config.copy()
        config_copy["transfer_from"] = best_trial.trial_id
        config_copy["freeze_layers"] = True

        return ExperimentTask(
            model_name=model,
            task_name=next_task,
            tier=PatientLevel.STANDARD,
            study_name=f"{model}_{next_task}_transfer",
            priority=92.0,
            fixed_config=config_copy,
            is_transfer=True,
            transfer_from_trial=best_trial.trial_id,
        )

    return None


def check_low_data_needed(
    stats: dict,
    progress: dict,
    model: str,
    task: str,
) -> ExperimentTask | None:
    """Check if low-data regime experiment should be scheduled."""
    if task not in ["mnist", "cifar10", "fashion_mnist"]:
        return None

    trials = stats.get("trials", [])
    if not trials:
        return None

    trials.sort(key=lambda x: x.accuracy, reverse=True)
    best_trial = trials[0]

    if best_trial.accuracy < 0.90:
        return None

    fractions = [0.1, 0.25]

    for frac in fractions:
        study_name = f"{model}_{task}_lowdata_{frac}"

        already_run = False
        for t in trials:
            if t.config.get("data_fraction") == frac:
                already_run = True
                break

        if not already_run:
            config_copy = best_trial.config.copy()
            config_copy["data_fraction"] = frac
            config_copy["epochs"] = 20

            return ExperimentTask(
                model_name=model,
                task_name=task,
                tier=PatientLevel.STANDARD,
                study_name=study_name,
                priority=85.0 - (frac * 10),
                fixed_config=config_copy,
                verification_of_trial_id=best_trial.trial_id,
            )

    return None


def check_ablation_needed(
    stats: dict,
    progress: dict,
    model: str,
    task: str,
    check_criterion_fn,
) -> ExperimentTask | None:
    """Check if ablation study should be scheduled."""
    trials = stats.get("trials", [])
    if not trials:
        return None

    trials.sort(key=lambda x: x.accuracy, reverse=True)
    best_trial = trials[0]

    if not check_criterion_fn(PatientLevel.STANDARD, task, best_trial.accuracy):
        return None

    ablations = []
    config = best_trial.config

    if "symmetric_weights" in config:
        ablations.append(("symmetric_weights", not config["symmetric_weights"]))

    if config.get("beta", 0.0) > 0.0:
        ablations.append(("beta", 0.0))

    if config.get("use_top_down", False):
        ablations.append(("use_top_down", False))

    if "eqprop" in model or "eq_prop" in model:
        current_nudge = config.get("nudge_factor", 1.0)
        if current_nudge != 0.1:
            ablations.append(("nudge_factor", 0.1))
        if current_nudge != 2.0:
            ablations.append(("nudge_factor", 2.0))

    if "hebbian" in model and "deep" in model:
        current_depth = config.get("num_layers", 100)
        if current_depth != 10:
            ablations.append(("num_layers", 10))
        if current_depth != 50:
            ablations.append(("num_layers", 50))

    if "transformer" in model:
        current_variant = config.get("variant", "full")
        if current_variant != "attention_only":
            ablations.append(("variant", "attention_only"))
        if current_variant != "recurrent_core":
            ablations.append(("variant", "recurrent_core"))

    for param, val in ablations:
        already_run = False
        for t in trials:
            if t.config.get("is_ablation") and t.config.get("ablation_param") == param:
                already_run = True
                break

        if not already_run:
            config_copy = config.copy()
            config_copy[param] = val
            config_copy["is_ablation"] = True
            config_copy["ablation_param"] = param

            return ExperimentTask(
                model_name=model,
                task_name=task,
                tier=PatientLevel.STANDARD,
                study_name=f"{model}_{task}_{PatientLevel.STANDARD.value}",
                priority=80.0,
                fixed_config=config_copy,
                verification_of_trial_id=best_trial.trial_id,
                is_ablation=True,
                ablation_param=param,
            )

    return None


def check_robustness_needed(
    deep_stats: dict,
    progress: dict,
    model: str,
    task: str,
    check_criterion_fn,
) -> ExperimentTask | None:
    """Check if robustness analysis should be scheduled."""
    trials = deep_stats.get("trials", [])
    if not trials:
        return None

    best_trial = max(trials, key=lambda t: t.accuracy)
    if not check_criterion_fn(PatientLevel.DEEP, task, best_trial.accuracy):
        return None

    for t in trials:
        if t.config.get("is_robustness_check"):
            return None

    priority = 85.0 + best_trial.accuracy * 10.0
    config_copy = best_trial.config.copy()

    return ExperimentTask(
        model_name=model,
        task_name=task,
        tier=PatientLevel.DEEP,
        study_name=f"{model}_{task}_{PatientLevel.DEEP.value}",
        priority=priority,
        fixed_config=config_copy,
        verification_of_trial_id=best_trial.trial_id,
        is_robustness_check=True,
    )
