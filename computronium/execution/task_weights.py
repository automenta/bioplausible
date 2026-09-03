"""
Task weights, groups, and curriculum tracks for prioritization.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from computronium.execution._lifecycle import CurriculumManager

# Task weights for prioritization (higher = more important)
TASK_WEIGHTS = {
    "digits": 0.50,  # Fastest proxy (Tiny) - Boosted for early filtering
    "usps": 0.45,  # Fast proxy (Small) - Boosted
    "kmnist": 0.35,  # Boosted
    "mnist": 0.30,
    "cartpole": 0.40,  # RL Smoke test
    "pendulum": 0.35,  # RL Intermediate
    "acrobot": 0.30,  # RL Hard
    "fashion_mnist": 0.25,
    "svhn": 0.20,
    "char_ngram": 0.30,
    "tiny_shakespeare": 0.35,
    "cifar10": 0.15,
    "cifar100": 0.10,
}

# Task groups for filtering
TASK_GROUPS = {
    "vision": [
        "digits",
        "usps",
        "kmnist",
        "mnist",
        "fashion_mnist",
        "svhn",
        "cifar10",
        "cifar100",
    ],
    "lm": ["char_ngram", "tiny_shakespeare"],
    "rl": ["cartpole", "pendulum", "acrobot"],
}

# Tier order for filtering
TIER_ORDER = {
    "smoke": 0,
    "shallow": 1,
    "standard": 2,
    "deep": 3,
    "cross_val": 4,
}


def calculate_future_boost(
    task_name: str, current_weight: float, curriculum: CurriculumManager
) -> float:
    """Calculate priority boost based on future tasks in curriculum tracks."""
    future_boost = 0.0
    for track_name, track_tasks in curriculum.TRACKS.items():
        if task_name in track_tasks:
            idx = track_tasks.index(task_name)
            for forward_idx in range(idx + 1, len(track_tasks)):
                future_task = track_tasks[forward_idx]
                future_weight = TASK_WEIGHTS.get(future_task, 0.10)

                if future_weight > current_weight:
                    distance = forward_idx - idx
                    boost = (future_weight - current_weight) * (0.9**distance)
                    future_boost = max(future_boost, boost)
    return future_boost


def calculate_complexity_penalty(model_name: str) -> float:
    """
    Calculate a penalty factor based on the computational complexity of the model.
    Models with high computational complexity get lower priority to prevent
    the scientist from getting stuck on expensive trials.
    """
    # Define complexity penalties for known computationally expensive models
    complexity_penalties = {
        "Deep Hebbian (Hundred-Layer)": 0.7,  # Reduced penalty for very deep models
        "EqProp Transformer (Full)": 0.8,  # Reduced penalty for transformers
        "EqProp Transformer (Attention Only)": 0.8,
        "EqProp Transformer (Hybrid)": 0.8,
        "EqProp Transformer (Recurrent)": 0.8,
        "EqProp Diffusion": 0.7,  # Reduced penalty for diffusion models
    }

    # Return penalty if model is in the list, otherwise 1.0 (no penalty)
    return complexity_penalties.get(model_name, 1.0)


__all__ = [
    "TASK_GROUPS",
    "TASK_WEIGHTS",
    "TIER_ORDER",
    "calculate_complexity_penalty",
    "calculate_future_boost",
]
