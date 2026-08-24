"""
Counterfactual Generator for AutoScientist.

Generates "what-if" hypotheses by perturbing experimental configurations
and predicting outcomes using surrogate models and causal analysis.
"""

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from computronium.knowledge import KnowledgeBase

from computronium.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Counterfactual:
    """A counterfactual scenario: what if we changed X?"""

    name: str
    description: str
    base_config: dict[str, object]
    modifications: dict[str, object]
    predicted_outcome: dict[str, float]
    confidence: float
    reasoning: str
    category: str  # "hyperparameter", "architecture", "algorithm", "data"


@dataclass(frozen=True, slots=True)
class CounterfactualBatch:
    """A batch of related counterfactuals for systematic exploration."""

    base_experiment_id: str
    base_config: dict[str, object]
    base_outcome: dict[str, float]
    counterfactuals: list[Counterfactual]
    generation_strategy: str


class CounterfactualGenerator:
    """
    Generates counterfactual hypotheses from experiment results.

    Uses:
    1. Surrogate models from KnowledgeBase to predict outcomes
    2. Causal analysis to identify high-impact parameters
    3. Domain knowledge to propose meaningful perturbations
    """

    def __init__(self, knowledge_base: KnowledgeBase | None = None):
        self.kb = knowledge_base

    def generate_from_experiment(
        self,
        experiment_id: str,
        n_counterfactuals: int = 10,
        strategies: list[str] | None = None,
    ) -> CounterfactualBatch:
        """
        Generate counterfactuals from a completed experiment.

        Args:
            experiment_id: ID of the base experiment
            n_counterfactuals: Number of counterfactuals to generate
            strategies: Subset of strategies to use
                ("hyperparameter", "architecture", "algorithm", "data", "training")

        Returns:
            CounterfactualBatch with generated hypotheses
        """
        if not self.kb:
            raise ValueError("KnowledgeBase required for counterfactual generation")

        exp = self.kb.get_experiment(experiment_id)
        if not exp:
            raise ValueError(f"Experiment {experiment_id} not found in KB")

        base_config = json.loads(exp.get("config", "{}"))
        base_metrics = json.loads(exp.get("metrics", "{}"))
        base_outcome = {
            k: v for k, v in base_metrics.items() if isinstance(v, (int, float))
        }

        all_strategies = [
            "hyperparameter",
            "architecture",
            "algorithm",
            "data",
            "training",
        ]
        active_strategies = strategies or all_strategies

        counterfactuals = []

        if "hyperparameter" in active_strategies:
            counterfactuals.extend(
                self._hyperparameter_counterfactuals(
                    base_config,
                    base_outcome,
                    n_counterfactuals // len(active_strategies) + 1,
                )
            )

        if "architecture" in active_strategies:
            counterfactuals.extend(
                self._architecture_counterfactuals(
                    base_config,
                    base_outcome,
                    n_counterfactuals // len(active_strategies) + 1,
                )
            )

        if "algorithm" in active_strategies:
            counterfactuals.extend(
                self._algorithm_counterfactuals(
                    base_config,
                    base_outcome,
                    n_counterfactuals // len(active_strategies) + 1,
                )
            )

        if "data" in active_strategies:
            counterfactuals.extend(
                self._data_counterfactuals(
                    base_config,
                    base_outcome,
                    n_counterfactuals // len(active_strategies) + 1,
                )
            )

        if "training" in active_strategies:
            counterfactuals.extend(
                self._training_counterfactuals(
                    base_config,
                    base_outcome,
                    n_counterfactuals // len(active_strategies) + 1,
                )
            )

        # Sort by confidence and take top N
        counterfactuals.sort(key=lambda c: c.confidence, reverse=True)
        counterfactuals = counterfactuals[:n_counterfactuals]

        return CounterfactualBatch(
            base_experiment_id=experiment_id,
            base_config=base_config,
            base_outcome=base_outcome,
            counterfactuals=counterfactuals,
            generation_strategy=",".join(active_strategies),
        )

    def _hyperparameter_counterfactuals(
        self,
        base_config: dict[str, object],
        base_outcome: dict[str, float],
        n: int,
    ) -> list[Counterfactual]:
        """Generate hyperparameter perturbation counterfactuals."""
        counterfactuals = []

        # Define hyperparameter search spaces
        hp_spaces = {
            "lr": [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03],
            "batch_size": [16, 32, 64, 128, 256],
            "beta": [0.1, 0.2, 0.3, 0.5, 0.8, 1.0],
            "max_steps": [5, 10, 20, 30, 50],
            "gamma": [0.1, 0.3, 0.5, 0.7, 0.9],
            "weight_decay": [0.0, 1e-5, 1e-4, 1e-3, 1e-2],
        }

        current_lr = base_config.get("lr", 0.001)
        current_bs = base_config.get("batch_size", 64)
        current_beta = base_config.get("beta", 0.5)
        current_steps = base_config.get("max_steps", 10)
        current_gamma = base_config.get("gamma", 0.5)

        perturbations = [
            ("lr", "learning rate", hp_spaces["lr"], current_lr),
            ("batch_size", "batch size", hp_spaces["batch_size"], current_bs),
            ("beta", "nudging strength (β)", hp_spaces["beta"], current_beta),
            ("max_steps", "equilibrium steps", hp_spaces["max_steps"], current_steps),
            ("gamma", "leak rate (γ)", hp_spaces["gamma"], current_gamma),
        ]

        for param_name, param_desc, values, current in perturbations:
            for new_val in values:
                if new_val == current:
                    continue

                new_config = copy.deepcopy(base_config)
                new_config[param_name] = new_val

                predicted = self._predict_outcome(new_config, base_outcome)
                confidence = self._estimate_confidence(param_name, current, new_val)

                direction = "increase" if new_val > current else "decrease"
                counterfactuals.append(
                    Counterfactual(
                        name=f"{param_name}_{direction}_{new_val}",
                        description=f"What if we {direction} {param_desc} from {current} to {new_val}?",
                        base_config=base_config,
                        modifications={param_name: new_val},
                        predicted_outcome=predicted,
                        confidence=confidence,
                        reasoning=(
                            f"Changing {param_name} from {current} to {new_val} may "
                            f"{'improve' if confidence > 0.5 else 'affect'} convergence "
                            f"and final accuracy"
                        ),
                        category="hyperparameter",
                    )
                )

        return counterfactuals[:n]

    def _architecture_counterfactuals(
        self,
        base_config: dict[str, object],
        base_outcome: dict[str, float],
        n: int,
    ) -> list[Counterfactual]:
        """Generate architecture modification counterfactuals."""
        counterfactuals = []

        current_hidden = base_config.get("hidden_dim", 256)
        current_layers = base_config.get("num_layers", 2)
        current_model = base_config.get("model", "eqprop_mlp")

        # Hidden dimension variations
        for hidden in [128, 256, 512, 1024]:
            if hidden == current_hidden:
                continue
            new_config = copy.deepcopy(base_config)
            new_config["hidden_dim"] = hidden
            predicted = self._predict_outcome(new_config, base_outcome)
            counterfactuals.append(
                Counterfactual(
                    name=f"hidden_dim_{hidden}",
                    description=f"What if we change hidden dimension from {current_hidden} to {hidden}?",
                    base_config=base_config,
                    modifications={"hidden_dim": hidden},
                    predicted_outcome=predicted,
                    confidence=0.6 if hidden > current_hidden else 0.5,
                    reasoning="Larger hidden dim increases capacity but may overfit; smaller is more efficient",
                    category="architecture",
                )
            )

        # Layer depth variations
        for layers in [1, 2, 3, 4, 5, 6]:
            if layers == current_layers:
                continue
            new_config = copy.deepcopy(base_config)
            new_config["num_layers"] = layers
            predicted = self._predict_outcome(new_config, base_outcome)
            counterfactuals.append(
                Counterfactual(
                    name=f"num_layers_{layers}",
                    description=f"What if we change depth from {current_layers} to {layers} layers?",
                    base_config=base_config,
                    modifications={"num_layers": layers},
                    predicted_outcome=predicted,
                    confidence=0.55,
                    reasoning="Deeper networks may capture more complex patterns but harder to train with local rules",
                    category="architecture",
                )
            )

        # Spectral norm variations
        current_spectral = base_config.get("spectral_bound_gamma")
        for gamma in [0.9, 0.95, 0.99, 1.0, None]:
            if gamma == current_spectral:
                continue
            new_config = copy.deepcopy(base_config)
            if gamma is None:
                new_config.pop("spectral_bound_gamma", None)
            else:
                new_config["spectral_bound_gamma"] = gamma
            predicted = self._predict_outcome(new_config, base_outcome)
            counterfactuals.append(
                Counterfactual(
                    name=f"spectral_gamma_{gamma}",
                    description=f"What if we {'enable' if gamma else 'disable'} spectral normalization (γ={gamma})?",
                    base_config=base_config,
                    modifications={"spectral_bound_gamma": gamma},
                    predicted_outcome=predicted,
                    confidence=0.65,
                    reasoning="Spectral norm stabilizes dynamics; critical for deep local learning",
                    category="architecture",
                )
            )

        return counterfactuals[:n]

    def _algorithm_counterfactuals(
        self,
        base_config: dict[str, object],
        base_outcome: dict[str, float],
        n: int,
    ) -> list[Counterfactual]:
        """Generate algorithm substitution counterfactuals."""
        counterfactuals = []

        current_model = base_config.get("model", "eqprop_mlp")
        current_propagator = base_config.get("propagator")

        # Model family alternatives
        model_alternatives = {
            "eqprop_mlp": [
                "fa",
                "direct_fa",
                "contrastive_hebbian_learning",
                "forward_forward",
                "pepita",
            ],
            "fa": ["eqprop_mlp", "direct_fa", "adaptive_fa", "stochastic_fa"],
            "forward_forward": ["pepita", "eqprop_mlp", "contrastive_hebbian_learning"],
            "contrastive_hebbian_learning": [
                "eqprop_mlp",
                "forward_forward",
                "hebbian_chain",
            ],
        }

        alternatives = model_alternatives.get(current_model, [])
        for alt_model in alternatives[:3]:
            new_config = copy.deepcopy(base_config)
            new_config["model"] = alt_model
            predicted = self._predict_outcome(new_config, base_outcome)
            counterfactuals.append(
                Counterfactual(
                    name=f"model_{alt_model}",
                    description=f"What if we switch from {current_model} to {alt_model}?",
                    base_config=base_config,
                    modifications={"model": alt_model},
                    predicted_outcome=predicted,
                    confidence=0.45,
                    reasoning=f"{alt_model} offers different bio-plausibility/accuracy tradeoffs",
                    category="algorithm",
                )
            )

        # Propagator alternatives (if using MEP/structured)
        if current_model in ["eqprop_mlp", "tile_pc", "tile_ep"]:
            propagator_alts = ["muon_backprop", "local_ep", "natural_ep"]
            for alt_prop in propagator_alts:
                new_config = copy.deepcopy(base_config)
                new_config["propagator"] = alt_prop
                predicted = self._predict_outcome(new_config, base_outcome)
                counterfactuals.append(
                    Counterfactual(
                        name=f"propagator_{alt_prop}",
                        description=f"What if we use {alt_prop} propagator instead of default?",
                        base_config=base_config,
                        modifications={"propagator": alt_prop},
                        predicted_outcome=predicted,
                        confidence=0.5,
                        reasoning="Different propagators implement different update strategies",
                        category="algorithm",
                    )
                )

        # Tile algorithm variations (for TileNet models)
        if "tile" in current_model:
            tile_algos = ["ep", "fa", "tp", "pc", "hebbian", "snn"]
            current_algo = base_config.get("algorithm", "ep")
            for algo in tile_algos:
                if algo == current_algo:
                    continue
                new_config = copy.deepcopy(base_config)
                new_config["algorithm"] = algo
                predicted = self._predict_outcome(new_config, base_outcome)
                counterfactuals.append(
                    Counterfactual(
                        name=f"tile_algo_{algo}",
                        description=f"What if we use {algo.upper()} dynamics on the tile substrate?",
                        base_config=base_config,
                        modifications={"algorithm": algo},
                        predicted_outcome=predicted,
                        confidence=0.55,
                        reasoning=f"TileNet substrate supports 6 algorithms; {algo} may suit this task better",
                        category="algorithm",
                    )
                )

        return counterfactuals[:n]

    def _data_counterfactuals(
        self,
        base_config: dict[str, object],
        base_outcome: dict[str, float],
        n: int,
    ) -> list[Counterfactual]:
        """Generate data-related counterfactuals."""
        counterfactuals = []

        current_task = base_config.get("task", "mnist")
        current_data_frac = base_config.get("data_fraction", 1.0)

        # Data fraction variations
        for frac in [0.1, 0.25, 0.5, 0.75, 1.0]:
            if frac == current_data_frac:
                continue
            new_config = copy.deepcopy(base_config)
            new_config["data_fraction"] = frac
            predicted = self._predict_outcome(new_config, base_outcome)
            counterfactuals.append(
                Counterfactual(
                    name=f"data_fraction_{frac}",
                    description=f"What if we train on {frac * 100:.0f}% of the data?",
                    base_config=base_config,
                    modifications={"data_fraction": frac},
                    predicted_outcome=predicted,
                    confidence=0.7,
                    reasoning="Less data tests sample efficiency; local learning may degrade less than backprop",
                    category="data",
                )
            )

        # Task transfer counterfactuals
        current_model = base_config.get("model", "unknown")
        task_groups = {
            "mnist": ["fashion_mnist", "cifar10", "svhn"],
            "fashion_mnist": ["mnist", "cifar10", "svhn"],
            "cifar10": ["cifar100", "svhn", "mnist"],
            "tiny_shakespeare": ["wikitext2", "penn_treebank"],
            "cora": ["citeseer", "pubmed"],
        }

        if current_task in task_groups:
            for target_task in task_groups[current_task][:2]:
                new_config = copy.deepcopy(base_config)
                new_config["task"] = target_task
                predicted = self._predict_outcome(new_config, base_outcome)
                counterfactuals.append(
                    Counterfactual(
                        name=f"task_{target_task}",
                        description=f"What if we transfer {current_model} from {current_task} to {target_task}?",
                        base_config=base_config,
                        modifications={"task": target_task},
                        predicted_outcome=predicted,
                        confidence=0.4,
                        reasoning="Cross-domain transfer tests representation generality",
                        category="data",
                    )
                )

        return counterfactuals[:n]

    def _training_counterfactuals(
        self,
        base_config: dict[str, object],
        base_outcome: dict[str, float],
        n: int,
    ) -> list[Counterfactual]:
        """Generate training procedure counterfactuals."""
        counterfactuals = []

        current_epochs = base_config.get("epochs", 10)
        current_optimizer = base_config.get("optimizer", "adam")

        # Epoch variations
        for epochs in [5, 10, 20, 50, 100]:
            if epochs == current_epochs:
                continue
            new_config = copy.deepcopy(base_config)
            new_config["epochs"] = epochs
            predicted = self._predict_outcome(new_config, base_outcome)
            counterfactuals.append(
                Counterfactual(
                    name=f"epochs_{epochs}",
                    description=f"What if we train for {epochs} epochs instead of {current_epochs}?",
                    base_config=base_config,
                    modifications={"epochs": epochs},
                    predicted_outcome=predicted,
                    confidence=0.65,
                    reasoning="More epochs may improve convergence; local learning often needs more time",
                    category="training",
                )
            )

        # Optimizer alternatives
        for opt in ["sgd", "adamw", "smep", "smep_fast", "sdmep", "local_ep"]:
            if opt == current_optimizer:
                continue
            new_config = copy.deepcopy(base_config)
            new_config["optimizer"] = opt
            predicted = self._predict_outcome(new_config, base_outcome)
            counterfactuals.append(
                Counterfactual(
                    name=f"optimizer_{opt}",
                    description=f"What if we use {opt} optimizer instead of {current_optimizer}?",
                    base_config=base_config,
                    modifications={"optimizer": opt},
                    predicted_outcome=predicted,
                    confidence=0.5,
                    reasoning=f"{opt} may provide better optimization dynamics for local learning",
                    category="training",
                )
            )

        # Beta schedule (for EqProp)
        if "eqprop" in base_config.get("model", "") or "tile" in base_config.get(
            "model", ""
        ):
            schedules = [
                "constant",
                "linear_anneal",
                "cosine_anneal",
                "exponential_anneal",
            ]
            for sched in schedules:
                new_config = copy.deepcopy(base_config)
                new_config["beta_schedule"] = sched
                predicted = self._predict_outcome(new_config, base_outcome)
                counterfactuals.append(
                    Counterfactual(
                        name=f"beta_schedule_{sched}",
                        description=f"What if we use {sched} β schedule instead of constant?",
                        base_config=base_config,
                        modifications={"beta_schedule": sched},
                        predicted_outcome=predicted,
                        confidence=0.6,
                        reasoning="Annealing β can improve EqProp convergence and final accuracy",
                        category="training",
                    )
                )

        return counterfactuals[:n]

    def _predict_outcome(
        self, config: dict[str, object], base_outcome: dict[str, float]
    ) -> dict[str, float]:
        """Predict outcome for a modified config using KB surrogate or heuristics."""
        if self.kb:
            try:
                pred = self.kb.predict_outcome(config, "val_accuracy")
                if pred > 0:
                    return {"val_accuracy": pred}
            except Exception:
                pass

        # Heuristic fallback
        return self._heuristic_prediction(config, base_outcome)

    def _heuristic_prediction(
        self, config: dict[str, object], base_outcome: dict[str, float]
    ) -> dict[str, float]:
        """Heuristic prediction when no surrogate available."""
        base_acc = base_outcome.get("val_accuracy", base_outcome.get("accuracy", 0.5))

        # Simple heuristics based on config changes
        predicted = base_acc

        # Learning rate effects
        lr = config.get("lr", 0.001)
        if lr > 0.01:
            predicted *= 0.8  # Too high LR hurts
        elif lr < 0.0005:
            predicted *= 0.9  # Too low LR slow

        # Hidden dim effects
        hidden = config.get("hidden_dim", 256)
        if hidden > 512:
            predicted *= 1.05  # More capacity helps slightly
        elif hidden < 128:
            predicted *= 0.9  # Too small hurts

        # Depth effects
        layers = config.get("num_layers", 2)
        if layers > 4:
            predicted *= 0.85  # Deep local learning is hard
        elif layers == 1:
            predicted *= 0.95

        # Spectral norm helps deep nets
        if config.get("spectral_bound_gamma") and layers > 2:
            predicted *= 1.1

        # Data fraction
        frac = config.get("data_fraction", 1.0)
        predicted *= 0.5 + 0.5 * frac  # Rough scaling

        # Clip
        predicted = max(0.0, min(1.0, predicted))

        return {"val_accuracy": predicted}

    def _estimate_confidence(self, param: str, current: object, new: object) -> float:
        """Estimate confidence in counterfactual prediction."""
        # Higher confidence for well-understood parameters
        high_confidence_params = {
            "lr",
            "batch_size",
            "epochs",
            "data_fraction",
            "beta",
            "gamma",
        }
        if param in high_confidence_params:
            return 0.7

        # Architecture changes are less certain
        if param in {"hidden_dim", "num_layers", "spectral_bound_gamma"}:
            return 0.55

        # Algorithm changes are most uncertain
        if param in {"model", "propagator", "algorithm", "optimizer"}:
            return 0.45

        return 0.5


class BetaScheduleCounterfactuals:
    """
    Specialized counterfactuals for β (nudge) schedule in Equilibrium Propagation.

    This addresses the specific P2.10 idea: "Meta-Learned β Schedule"
    """

    SCHEDULES = {
        "constant": lambda step, total, beta: beta,
        "linear_anneal": lambda step, total, beta: beta * (1 - step / total),
        "cosine_anneal": lambda step, total, beta: (
            beta * 0.5 * (1 + np.cos(np.pi * step / total))
        ),
        "exponential_anneal": lambda step, total, beta: (
            beta * np.exp(-3 * step / total)
        ),
        "cyclic": lambda step, total, beta: (
            beta * (0.5 + 0.5 * np.sin(2 * np.pi * step / 10))
        ),
        "warmup_then_anneal": lambda step, total, beta: (
            beta * min(1.0, step / (total * 0.1)) * (1 - step / total)
        ),
    }

    @classmethod
    def generate_schedule_variants(
        cls,
        base_config: dict[str, object],
        base_outcome: dict[str, float],
    ) -> list[Counterfactual]:
        """Generate counterfactuals for different β schedules."""
        counterfactuals = []
        base_beta = base_config.get("beta", 0.5)

        for name, schedule_fn in cls.SCHEDULES.items():
            if base_config.get("beta_schedule") == name:
                continue

            new_config = copy.deepcopy(base_config)
            new_config["beta_schedule"] = name
            new_config["beta_schedule_params"] = {"base_beta": base_beta}

            predicted = {
                "val_accuracy": base_outcome.get("val_accuracy", 0.5)
                * 1.05  # Schedules typically help
            }

            counterfactuals.append(
                Counterfactual(
                    name=f"beta_schedule_{name}",
                    description=f"What if we use {name} β schedule (base β={base_beta})?",
                    base_config=base_config,
                    modifications={
                        "beta_schedule": name,
                        "beta_schedule_params": {"base_beta": base_beta},
                    },
                    predicted_outcome=predicted,
                    confidence=0.65,
                    reasoning=f"{name} schedule may improve EqProp convergence dynamics",
                    category="training",
                )
            )

        return counterfactuals

    @classmethod
    def simulate_schedule(
        cls, schedule_name: str, total_steps: int, base_beta: float = 0.5
    ) -> list[float]:
        """Simulate a β schedule over training steps."""
        schedule_fn = cls.SCHEDULES.get(schedule_name)
        if not schedule_fn:
            return [base_beta] * total_steps

        return [
            schedule_fn(step, total_steps, base_beta) for step in range(total_steps)
        ]


def generate_what_if_report(
    batch: CounterfactualBatch,
    output_path: str | Path | None = None,
) -> str:
    """Generate a human-readable report of counterfactual hypotheses."""
    lines = [
        "# Counterfactual Analysis Report",
        "",
        f"Base Experiment: {batch.base_experiment_id}",
        f"Base Config: {json.dumps(batch.base_config, indent=2, default=str)}",
        f"Base Outcome: {json.dumps(batch.base_outcome, indent=2, default=str)}",
        f"Generation Strategy: {batch.generation_strategy}",
        "",
        f"## Counterfactual Hypotheses ({len(batch.counterfactuals)})",
        "",
    ]

    for i, cf in enumerate(batch.counterfactuals, 1):
        pred_acc = cf.predicted_outcome.get("val_accuracy", "N/A")
        base_acc = batch.base_outcome.get("val_accuracy", "N/A")
        delta = (
            pred_acc - base_acc
            if isinstance(pred_acc, (int, float)) and isinstance(base_acc, (int, float))
            else "N/A"
        )

        lines.extend([
            f"### {i}. {cf.name}",
            f"**Description:** {cf.description}",
            f"**Category:** {cf.category}",
            f"**Modifications:** {json.dumps(cf.modifications, default=str)}",
            f"**Predicted Accuracy:** {pred_acc:.4f}"
            if isinstance(pred_acc, float)
            else f"**Predicted Accuracy:** {pred_acc}",
            f"**Delta vs Base:** {delta:+.4f}"
            if isinstance(delta, float)
            else f"**Delta vs Base:** {delta}",
            f"**Confidence:** {cf.confidence:.0%}",
            f"**Reasoning:** {cf.reasoning}",
            "",
        ])

    report = "\n".join(lines)

    if output_path:
        Path(output_path).write_text(report)
        logger.info("Counterfactual report saved to %s", output_path)

    return report


__all__ = [
    "BetaScheduleCounterfactuals",
    "Counterfactual",
    "CounterfactualBatch",
    "CounterfactualGenerator",
    "generate_what_if_report",
]
