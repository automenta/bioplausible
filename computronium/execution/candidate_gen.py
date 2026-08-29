"""
Candidate generation logic for execution strategy.
"""

import random
from dataclasses import dataclass

from computronium.core.logging import get_logger
from computronium.core.registry import ComponentCategory, Registry
from computronium.execution._guards import (
    check_ablation_needed,
    check_continual_learning_needed,
    check_cv_needed,
    check_low_data_needed,
    check_robustness_needed,
    check_transfer_needed,
    check_verification_needed,
    get_stats,
)
from computronium.execution._lifecycle import CurriculumManager, PromotionGate
from computronium.execution._state import DecisionLogger, ExperimentState
from computronium.execution.criteria import check_criterion
from computronium.execution.events import EventSink, NullEventSink
from computronium.execution.task import ExperimentTask
from computronium.hyperopt import PatientLevel

logger = get_logger("AutoScientist")


class _ModelSpec:
    """Lightweight model spec consumed by ExecutionStrategy.generate_candidates."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name


def _model_specs() -> list[_ModelSpec]:
    """Return the cached list of model specs, lazily built from Registry."""
    if getattr(_model_specs, "cache", None) is not None:
        return _model_specs.cache
    specs: list[_ModelSpec] = []
    try:
        for entry in Registry.query(category=ComponentCategory.MODEL):
            specs.append(_ModelSpec(entry["name"]))
    except (KeyError, AttributeError, ValueError):  # pragma: no cover - registry empty
        logger.exception("Failed to enumerate models from Registry")
    _model_specs.cache = specs
    return specs


@dataclass
class ExecutionStrategyConfig:
    """Configuration for ExecutionStrategy."""

    state: ExperimentState
    decision_logger: DecisionLogger | None = None
    task_filter: str | None = None
    tier_limit: str | None = None
    model_filter: str | None = None  # Comma-separated list of models to exclude
    event_sink: EventSink | None = None


class CandidateGenerator:
    """
    Generates experiment candidates based on current state.
    """

    def __init__(self, config: ExecutionStrategyConfig):
        self.state = config.state
        self.decision_logger = config.decision_logger
        self._events: EventSink = (
            config.event_sink if config.event_sink is not None else NullEventSink()
        )
        self.task_filter = config.task_filter
        self.tier_limit = config.tier_limit.lower() if config.tier_limit else None
        self.model_filter = set(config.model_filter.split(",")) if config.model_filter else set()
        self._logged_events: set[str] = set()
        self.curriculum = CurriculumManager()

    def _log(
        self,
        key: str,
        event_type: str,
        desc: str,
        meta: dict[str, object] | None = None,
    ) -> None:
        if key not in self._logged_events:
            if self.decision_logger:
                self.decision_logger.log_decision(event_type, desc, meta)
            self._logged_events.add(key)

        self._events.set_insight(desc)

    def generate_candidates(self) -> list[ExperimentTask]:
        """
        Generates a list of all possible valid experiments based on current state.
        """
        progress = self.state.get_progress()
        candidates: list[ExperimentTask] = []

        saturated_tasks = self._analyze_saturation(progress)

        evolve_task = self._check_evolution_needed(progress, saturated_tasks)
        if evolve_task:
            candidates.append(evolve_task)

        # Analyze failures to generate constraints
        failure_constraints = self._analyze_failures(progress)
        self._apply_failure_logging(failure_constraints)

        # Analyze fragility (High Accuracy but Low Robustness)
        fragility_constraints = self._analyze_fragility()
        if fragility_constraints:
            # Merge constraints
            for m, c in fragility_constraints.items():
                if m not in failure_constraints:
                    failure_constraints[m] = {}
                failure_constraints[m].update(c)
                self._log(
                    f"fragile_constraint_{m}",
                    "ROBUSTNESS_ENFORCED",
                    f"Model {m} is fragile. Enforcing regularization.",
                    c,
                )

        # Analyze saturation (Tasks that are "solved")
        saturated_tasks = self._analyze_saturation(progress)
        self._apply_saturation_logging(saturated_tasks)

        for spec in _model_specs():
            tasks = self._resolve_tasks(spec.name)

            for task in tasks:
                if not self._should_consider_task(
                    spec.name, task, progress, saturated_tasks
                ):
                    continue

                candidates.extend(
                    self._generate_candidates_for_task(
                        spec.name, task, progress, failure_constraints
                    )
                )

        self._filter_by_tier_limit(candidates)
        self._apply_prioritization(candidates)

        return candidates

    def _should_consider_task(
        self, model_name: str, task: str, progress: dict, saturated_tasks: dict
    ) -> bool:
        """Check if a task should be considered for candidate generation."""
        if model_name in self.model_filter:
            return False

        if not self._matches_filter(task):
            return False

        if model_name in saturated_tasks and task in saturated_tasks[model_name]:
            return False

        if not self._check_curriculum(progress, model_name, task):
            return False

        return True

    def _generate_candidates_for_task(
        self, model: str, task: str, progress: dict, failure_constraints: dict
    ) -> list[ExperimentTask]:
        """Generate candidates for a specific model/task pair across tiers."""
        candidates = []

        # 1. SMOKE
        smoke_task = self._check_smoke_tier(model, task, progress, failure_constraints)
        if smoke_task:
            candidates.append(smoke_task)
            return candidates

        smoke_stats = self._get_stats(progress, model, task, PatientLevel.SMOKE)
        if not check_criterion(PatientLevel.SMOKE, task, smoke_stats["best_acc"]):
            # Retry chance for failed smoke
            if random.random() < 0.01:
                retry_task = self._make_task(model, task, PatientLevel.SMOKE, 10.0)
                if model in failure_constraints:
                    retry_task.constraints = failure_constraints[model]
                candidates.append(retry_task)
            return candidates

        # 2. SHALLOW
        shallow_task = self._check_shallow_tier(
            model, task, progress, smoke_stats, failure_constraints
        )
        if shallow_task:
            candidates.append(shallow_task)
            return candidates

        shallow_stats = self._get_stats(progress, model, task, PatientLevel.SHALLOW)
        if not check_criterion(
            PatientLevel.SHALLOW, task, shallow_stats["best_acc"]
        ):
            self._log(
                f"stagnated_shallow_{model}_{task}",
                "STAGNATION",
                (
                    f"Model {model} failed Shallow Tier on {task}"
                    f" (Acc: {shallow_stats['best_acc']:.2%}). Stopping."
                ),
                {"best_acc": shallow_stats["best_acc"]},
            )
            return candidates

        # 3. STANDARD
        candidates.extend(
            self._generate_standard_candidates(
                model, task, progress, shallow_stats, failure_constraints
            )
        )

        # If we just generated a standard exploration task,
        # we might stop here? Original code had `continue` if std_stats["count"] < 20.
        # Let's check if we generated a main standard task.
        std_stats = self._get_stats(progress, model, task, PatientLevel.STANDARD)
        if std_stats["count"] < 20:
            return candidates

        if not check_criterion(
            PatientLevel.STANDARD, task, std_stats["best_acc"]
        ):
            return candidates

        # 4. DEEP
        candidates.extend(
            self._generate_deep_candidates(
                model, task, progress, std_stats, failure_constraints
            )
        )

        return candidates

    def _check_smoke_tier(
        self,
        model: str,
        task: str,
        progress: dict,
        failure_constraints: dict,
    ) -> ExperimentTask | None:
        smoke_stats = self._get_stats(progress, model, task, PatientLevel.SMOKE)
        if smoke_stats["count"] < 3:
            if smoke_stats["count"] == 0:
                self._log(
                    f"smoke_{model}_{task}",
                    "NEW_HYPOTHESIS",
                    (
                        f"Starting initial investigation"
                        f" (Smoke Test) for {model} on {task}."
                    ),
                )

            p = 100.0 if smoke_stats["count"] == 0 else 80.0
            task_obj = self._make_task(model, task, PatientLevel.SMOKE, p)
            if model in failure_constraints:
                task_obj.constraints = failure_constraints[model]
            return task_obj
        return None

    def _check_shallow_tier(
        self,
        model: str,
        task: str,
        progress: dict,
        smoke_stats: dict,
        failure_constraints: dict,
    ) -> ExperimentTask | None:
        shallow_stats = self._get_stats(progress, model, task, PatientLevel.SHALLOW)
        if shallow_stats["count"] < 10:
            # Tuned Priority
            base_p = 50.0 + (smoke_stats["best_acc"] * 30.0)
            if shallow_stats["count"] == 0:
                self._log(
                    f"shallow_{model}_{task}",
                    "PROMOTION",
                    (
                        f"Promoting {model} to Shallow Tier"
                        f" (Passed Smoke Test with"
                        f" {smoke_stats['best_acc']:.2%})."
                    ),
                )
                base_p += 10.0

            model_constraints = failure_constraints.get(model, {})
            task_obj = self._make_task(model, task, PatientLevel.SHALLOW, base_p)
            if model_constraints:
                task_obj.constraints = model_constraints
            return task_obj
        return None

    def _generate_standard_candidates(
        self,
        model: str,
        task: str,
        progress: dict,
        shallow_stats: dict,
        failure_constraints: dict,
    ) -> list[ExperimentTask]:
        candidates = []
        std_stats = self._get_stats(progress, model, task, PatientLevel.STANDARD)

        # Verification
        v_task = self._check_verification_needed(
            std_stats, model, task, PatientLevel.STANDARD
        )
        if v_task:
            self._log(
                f"verify_std_{model}_{task}",
                "VERIFICATION",
                f"Verifying best result for {model} (Standard Tier).",
            )
            candidates.append(v_task)

        # Low Data
        ld_task = self._check_low_data_needed(std_stats, progress, model, task)
        if ld_task:
            self._log(
                f"low_data_{model}_{task}",
                "LOW_DATA_REGIME",
                (
                    f"Scheduling Low-Data experiment"
                    f" ({ld_task.fixed_config['data_fraction']:.0%})"
                    f" for {model}."
                ),
            )
            candidates.append(ld_task)

        # Ablation
        ab_task = self._check_ablation_needed(std_stats, progress, model, task)
        if ab_task:
            self._log(
                f"ablation_{model}_{task}_{ab_task.ablation_param}",
                "ABLATION_STUDY",
                f"Scheduling ablation study for {model} to verify components.",
                {"param": ab_task.ablation_param},
            )
            candidates.append(ab_task)

        # Continual Learning
        cl_task = self._check_continual_learning_needed(
            std_stats, progress, model, task
        )
        if cl_task:
            self._log(
                f"cl_{model}_{task}_{cl_task.continual_step}",
                "CONTINUAL_LEARNING",
                (
                    f"Attempting Continual Learning Step"
                    f" {cl_task.continual_step} for {model}."
                ),
            )
            candidates.append(cl_task)

        # Transfer Learning
        tf_task = self._check_transfer_needed(std_stats, progress, model, task)
        if tf_task:
            self._log(
                f"transfer_{model}_{task}",
                "TRANSFER_LEARNING",
                f"Attempting Transfer Learning from {task} for {model}.",
            )
            candidates.append(tf_task)

        # Cross Validation
        cv_task = self._check_cv_needed(std_stats, progress, model, task)
        if cv_task:
            self._log(
                f"cv_{model}_{task}",
                "CROSS_VALIDATION",
                f"Running 5-Fold Cross-Validation for {model} to confirm stability.",
            )
            candidates.append(cv_task)

        # Main Standard Exploration
        if std_stats["count"] < 20:
            base_p = 60.0 + (
                shallow_stats["best_acc"] * 20.0
            )  # Reduced from 40.0 to prevent excessive boosting

            if std_stats["count"] == 0:
                self._log(
                    f"standard_{model}_{task}",
                    "PROMOTION",
                    (
                        f"Promoting {model} to Standard Tier"
                        f" (Passed Shallow with"
                        f" {shallow_stats['best_acc']:.2%})."
                    ),
                )

            if std_stats["count"] > 15:
                base_p -= 10.0

            refine_constraints = self._refine_search_space(
                progress, model, task, PatientLevel.SHALLOW
            )
            fail_constraints = failure_constraints.get(model, {})

            final_constraints = {}
            if refine_constraints:
                self._log(
                    f"refine_std_{model}_{task}",
                    "REFINEMENT",
                    "Refining search space for Standard Tier based on Shallow results.",
                    refine_constraints,
                )
                final_constraints.update(refine_constraints)
            if fail_constraints:
                final_constraints.update(fail_constraints)

            task_obj = self._make_task(model, task, PatientLevel.STANDARD, base_p)
            if final_constraints:
                task_obj.constraints = final_constraints

            candidates.append(task_obj)

        return candidates

    def _generate_deep_candidates(
        self,
        model: str,
        task: str,
        progress: dict,
        std_stats: dict,
        failure_constraints: dict,
    ) -> list[ExperimentTask]:
        candidates = []
        deep_stats = self._get_stats(progress, model, task, PatientLevel.DEEP)

        # Robustness
        r_task = self._check_robustness_needed(deep_stats, progress, model, task)
        if r_task:
            self._log(
                f"robust_{model}_{task}",
                "ROBUSTNESS_CHECK",
                (
                    f"Triggering Robustness Analysis for {model}"
                    f" due to high Deep Tier performance."
                ),
            )
            candidates.append(r_task)

        # Verification
        v_task = self._check_verification_needed(
            deep_stats, model, task, PatientLevel.DEEP
        )
        if v_task:
            candidates.append(v_task)

        # Main Deep Exploration
        if deep_stats["count"] < 5:
            if deep_stats["count"] == 0:
                self._log(
                    f"deep_{model}_{task}",
                    "PROMOTION",
                    (
                        f"Promoting {model} to Deep Tier"
                        f" (Passed Standard with"
                        f" {std_stats['best_acc']:.2%})."
                    ),
                )

            p = 20.0 + (
                std_stats["best_acc"] * 25.0
            )  # Reduced from 50.0 to prevent excessive boosting

            refine_constraints = self._refine_search_space(
                progress, model, task, PatientLevel.STANDARD
            )
            fail_constraints = failure_constraints.get(model, {})

            final_constraints = {}
            if refine_constraints:
                self._log(
                    f"refine_deep_{model}_{task}",
                    "REFINEMENT",
                    "Refining search space for Deep Tier based on Standard results.",
                    refine_constraints,
                )
                final_constraints.update(refine_constraints)
            if fail_constraints:
                final_constraints.update(fail_constraints)

            task_obj = self._make_task(model, task, PatientLevel.DEEP, p)
            if final_constraints:
                task_obj.constraints = final_constraints

            candidates.append(task_obj)

        return candidates

    def _apply_failure_logging(self, failure_constraints: dict) -> None:
        if failure_constraints:
            for model, constraints in failure_constraints.items():
                self._log(
                    f"fail_constraint_{model}",
                    "CONSTRAINT_APPLIED",
                    (
                        f"High failure rate detected for {model}."
                        f" Restricting search space."
                    ),
                    constraints,
                )

    def _apply_saturation_logging(self, saturated_tasks: dict) -> None:
        if saturated_tasks:
            for model, tasks in saturated_tasks.items():
                for t in tasks:
                    self._log(
                        f"saturation_{model}_{t}",
                        "SATURATION",
                        f"Task {t} saturated (solved) for {model}. Skipping.",
                    )

    def _filter_by_tier_limit(self, candidates: list[ExperimentTask]) -> None:
        if self.tier_limit:
            limit_level = -1
            for tier, level in self._tier_order().items():
                if tier.value == self.tier_limit:
                    limit_level = level
                    break

            if limit_level != -1:
                candidates[:] = [
                    c
                    for c in candidates
                    if self._tier_order().get(c.tier, 999) <= limit_level
                ]

    def _apply_prioritization(self, candidates: list[ExperimentTask]) -> None:
        from computronium.execution.task_weights import (
            TASK_WEIGHTS,
            calculate_complexity_penalty,
            calculate_future_boost,
        )

        for c in candidates:
            weight = TASK_WEIGHTS.get(c.task_name, 0.10)
            future_boost = calculate_future_boost(c.task_name, weight, self.curriculum)
            effective_weight = weight + future_boost
            c.priority *= effective_weight * 5.0

        recent_tasks = self.state.get_recent_tasks(limit=10)
        task_counts: dict[str, int] = {}
        for t in recent_tasks:
            task_counts[t] = task_counts.get(t, 0) + 1

        recent_models = self.state.get_recent_models(limit=10)
        model_counts: dict[str, int] = {}
        for m in recent_models:
            model_counts[m] = model_counts.get(m, 0) + 1

        for c in candidates:
            t_count = task_counts.get(c.task_name, 0)
            if t_count > 0:
                c.priority *= 0.9**t_count

            m_count = model_counts.get(c.model_name, 0)
            if m_count > 0:
                c.priority *= 0.8**m_count

            complexity_penalty = calculate_complexity_penalty(c.model_name)
            c.priority *= complexity_penalty

    def _refine_search_space(
        self, progress, model, task, source_tier
    ) -> dict[str, object] | None:
        """
        Analyze successful trials from source_tier to refine search space for next tier.
        """
        stats = self._get_stats(progress, model, task, source_tier)
        trials = stats.get("trials", [])

        if len(trials) < 3:
            return None

        trials.sort(key=lambda x: x.accuracy, reverse=True)
        top_n = max(3, len(trials) // 2)
        top_trials = trials[:top_n]

        if top_trials[0].accuracy < 0.2:
            return None

        constraints = {}

        lrs = [t.config["lr"] for t in top_trials if "lr" in t.config]
        if lrs:
            min_lr = min(lrs)
            max_lr = max(lrs)
            constraints["min_lr"] = min_lr * 0.5
            constraints["max_lr"] = max_lr * 2.0

        betas = [
            t.config["beta"]
            for t in top_trials
            if "beta" in t.config and t.config["beta"] is not None
        ]
        if betas:
            min_beta = min(betas)
            max_beta = max(betas)
            constraints["min_beta"] = max(0.0, min_beta - 0.1)
            constraints["max_beta"] = min(1.0, max_beta + 0.1)

        return constraints

    def _analyze_fragility(self) -> dict[str, dict[str, object]]:
        """
        Identify models that perform well but break easily, and suggest constraints.
        """
        constraints = {}
        if hasattr(self.state, "get_fragile_models"):
            fragile_models = self.state.get_fragile_models()
            for model, score in fragile_models.items():
                constraints[model] = {
                    "min_weight_decay": 1e-4,
                    "min_dropout": 0.2,
                    "use_spectral_norm": True,
                }
        return constraints

    def _analyze_failures(self, progress) -> dict[str, dict[str, object]]:
        """
        Analyze failure rates to suggest constraints.
        Returns: Dict[model_name, constraint_dict]
        """
        constraints = {}

        # 1. Query FailureTracker via State for Hard Failures
        if hasattr(self.state, "get_failure_analysis"):
            try:
                analysis = self.state.get_failure_analysis()
                recommendations = analysis.get("recommendations", [])

                for rec in recommendations:
                    if rec.get("issue") == "High NaN failure rate":
                        affected = rec.get("affected_models", [])
                        for model in affected:
                            if model not in constraints:
                                constraints[model] = {}
                            # Aggressive restriction
                            constraints[model]["max_lr"] = 0.001
                            constraints[model]["max_beta"] = 0.1

                    elif rec.get("issue") == "Out of memory errors":
                        # Constrain models to prevent OOM loop
                        # OOM often crashes system, so blame last run
                        # If affected_models is empty, apply to all active models
                        # Trust FailureTracker to have identified context if
                        # possible, else apply to all models in progress.
                        affected = rec.get("affected_models", [])
                        if not affected:
                            # Fallback: Apply to everything if systemic OOM
                            affected = list(progress.keys())

                        for model in affected:
                            if model not in constraints:
                                constraints[model] = {}
                            constraints[model]["max_batch_size"] = 64
                            constraints[model]["max_hidden_dim"] = (
                                512  # Relaxed aggressive scaling prevention
                            )

                    elif rec.get("issue") == "Frequent timeouts":
                        affected = rec.get("affected_models", [])
                        if not affected:
                            affected = list(progress.keys())

                        for model in affected:
                            if model not in constraints:
                                constraints[model] = {}
                            constraints[model]["max_hidden_dim"] = 256
                            constraints[model]["max_num_layers"] = 6

                    elif rec.get("issue") == "Early Training Instability":
                        # If we knew which models, we'd constrain them.
                        pass

            except (RuntimeError, ValueError, KeyError) as e:
                logger.warning("Failed to query failure analysis: %s", e)

        # 2. Analyze Progress for Soft Failures (Divergence/No Learning)
        for model, task_data in progress.items():
            total = 0
            failures = 0
            for task, tier_data in task_data.items():
                for tier, stats in tier_data.items():
                    trials = stats.get("trials", [])
                    for t in trials:
                        total += 1
                        if (
                            t.final_loss > 100 or t.accuracy < 0.11
                        ):  # Divergence or random chance
                            failures += 1

            if total > 5 and (failures / total) > 0.3:
                # If not already constrained more strictly
                if model not in constraints:
                    constraints[model] = {}
                    constraints[model]["max_lr"] = 0.005
                    constraints[model]["max_beta"] = 0.5

        return constraints

    def _analyze_saturation(self, progress) -> dict[str, list[str]]:
        """
        Identify tasks that are effectively "solved" (saturated) for a given model.
        Returns: Dict[model, List[task_name]]
        """
        saturation = {}

        for model, task_data in progress.items():
            solved_tasks = []

            # Check for direct saturation
            for task, tiers in task_data.items():
                best_acc = 0.0
                for tier_stats in tiers.values():
                    best_acc = max(best_acc, tier_stats.get("best_acc", 0.0))

                # Dynamic saturation thresholds
                threshold = 0.99
                if task == "digits":
                    threshold = 0.98
                elif task == "mnist":
                    threshold = 0.99
                elif task == "fashion_mnist":
                    threshold = 0.94

                if best_acc > threshold:
                    solved_tasks.append(task)

            # Implicit Saturation: If a harder task is solved, easier ones are "solved"
            if "mnist" in solved_tasks:
                if "digits" not in solved_tasks:
                    solved_tasks.append("digits")
                if "usps" not in solved_tasks:
                    solved_tasks.append("usps")

            if "fashion_mnist" in solved_tasks:
                if "mnist" not in solved_tasks:
                    solved_tasks.append("mnist")
                if "kmnist" not in solved_tasks:
                    solved_tasks.append("kmnist")

            if solved_tasks:
                saturation[model] = solved_tasks

        return saturation

    def _matches_filter(self, task: str) -> bool:
        if not self.task_filter or self.task_filter == "all":
            return True
        if self.task_filter == task:
            return True
        if self.task_filter in TASK_GROUPS:
            return task in TASK_GROUPS[self.task_filter]
        return False

    def _check_curriculum(self, progress: dict, model_name: str, task: str) -> bool:
        """
        Check if we are allowed to run this task based on curriculum.
        """
        from computronium.execution.task_weights import TASK_GROUPS

        track = None
        for t_list in self.curriculum.TRACKS.values():
            if task in t_list:
                track = t_list
                break

        if not track:
            return True

        try:
            curr_idx = track.index(task)
        except ValueError:
            return True

        if curr_idx == 0:
            return True

        prev_task = track[curr_idx - 1]

        if model_name not in progress or prev_task not in progress[model_name]:
            return False

        best_metrics = {"accuracy": 0.0, "reward": -float("inf")}
        tiers_run = False

        for tier_data in progress[model_name][prev_task].values():
            if tier_data.get("count", 0) > 0:
                tiers_run = True
                if "best_acc" in tier_data:
                    best_metrics["accuracy"] = max(
                        best_metrics["accuracy"], tier_data["best_acc"]
                    )

        if not tiers_run:
            return False

        return PromotionGate.check_promotion(prev_task, best_metrics)

    def _get_stats(self, progress, model, task, tier):
        return get_stats(progress, model, task, tier)

    def _check_continual_learning_needed(
        self, stats, progress, model, task
    ) -> ExperimentTask | None:
        return check_continual_learning_needed(stats, progress, model, task)

    def _check_transfer_needed(
        self, stats, progress, model, task
    ) -> ExperimentTask | None:
        return check_transfer_needed(stats, progress, model, task, self.curriculum)

    def _check_low_data_needed(
        self, stats, progress, model, task
    ) -> ExperimentTask | None:
        return check_low_data_needed(stats, progress, model, task)

    def _check_ablation_needed(
        self, stats, progress, model, task
    ) -> ExperimentTask | None:
        return check_ablation_needed(
            stats, progress, model, task, check_criterion
        )

    def _check_robustness_needed(
        self, deep_stats, progress, model, task
    ) -> ExperimentTask | None:
        return check_robustness_needed(
            deep_stats, progress, model, task, check_criterion
        )

    def _check_cv_needed(
        self, std_stats, progress, model, task
    ) -> ExperimentTask | None:
        return check_cv_needed(std_stats, progress, model, task)

    def _check_verification_needed(
        self, stats, model, task, tier
    ) -> ExperimentTask | None:
        return check_verification_needed(
            stats, model, task, tier, check_criterion
        )

    def _make_task(self, model, task, tier, priority):
        return ExperimentTask(
            model_name=model,
            task_name=task,
            tier=tier,
            study_name=f"{model}_{task}_{tier.value}",
            priority=priority,
        )

    def _tier_order(self):
        from computronium.execution.task_weights import TIER_ORDER
        return {PatientLevel[k.upper()]: v for k, v in TIER_ORDER.items()}


__all__ = [
    "CandidateGenerator",
    "ExecutionStrategyConfig",
]