"""
Tier criteria and task-specific threshold logic.
"""

from computronium.hyperopt import PatientLevel

# Base criteria per tier
CRITERIA = {
    PatientLevel.SMOKE: lambda acc: acc > 0.12,  # Beat random (0.10) slightly
    PatientLevel.SHALLOW: lambda acc: acc > 0.30,  # Relaxed for early feedback
    PatientLevel.STANDARD: lambda acc: acc > 0.60,
    PatientLevel.CROSS_VAL: lambda acc: True,  # CV just needs to run 5 times
    PatientLevel.DEEP: lambda acc: acc > 0.80,  # Deep bar
}


def check_criterion(tier: PatientLevel, task: str, acc: float) -> bool:
    """
    Check if accuracy meets the success criterion for a given tier and task.
    Allows task-specific overrides (e.g., lower threshold for CIFAR-100).
    """
    # Task-specific overrides
    if task == "cifar100":
        if tier == PatientLevel.SMOKE:
            return acc > 0.05  # 5x random chance
        elif tier == PatientLevel.SHALLOW:
            return acc > 0.15
        elif tier == PatientLevel.STANDARD:
            return acc > 0.30
        elif tier == PatientLevel.DEEP:
            return acc > 0.50

    # Fast Fail for Easy Tasks
    if task in ["digits", "usps"]:
        if tier == PatientLevel.SMOKE:
            return acc > 0.50  # Must be much better than random
        elif tier == PatientLevel.SHALLOW:
            return acc > 0.80

    if task == "tiny_shakespeare":
        # LM uses perplexity mostly but acc is tracked too.
        # Character-level LM accuracy is usually lower.
        if tier == PatientLevel.SMOKE:
            return acc > 0.30
        elif tier == PatientLevel.STANDARD:
            return acc > 0.45

    return CRITERIA[tier](acc)


__all__ = [
    "CRITERIA",
    "check_criterion",
]