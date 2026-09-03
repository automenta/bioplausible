"""
Validation Utilities for EqProp Scientific Validation Framework

This module provides:
1. Basic utilities: progress bars, synthetic datasets, model training
2. Scientific rigor functions: effect sizes, statistical tests, evidence classification
3. Formatting helpers: tables, statistical comparisons, claim-evidence-limitation format

Scientific Rigor Features:
- Cohen's d effect size calculation with interpretation
- Paired and independent t-tests for significance
- Evidence level classification (smoke/directional/conclusive)
- Formatted statistical comparisons with 95% CI
- Reproducibility tracking
"""

import numpy as np
import torch
from torch import nn

from computronium.core.logging import get_logger
from computronium.core.losses import compute_accuracy
from computronium.core.trainer import dispatch_train_step
from computronium.core.utils.optimizer import OptimizerConfig, create_optimizer
from computronium.utils import seed_everything

__all__ = [
    "classify_evidence_level",
    "compute_cohens_d",
    "compute_reproducibility_hash",
    "create_synthetic_dataset",
    "determine_status",
    "evaluate_accuracy",
    "format_claim_with_evidence",
    "format_metrics_table",
    "format_statistical_comparison",
    "format_value",
    "independent_ttest",
    "interpret_effect_size",
    "interpret_pvalue",
    "logger",
    "paired_ttest",
    "progress_bar",
    "robustness_summary",
    "train_model",
]
logger = get_logger()


def progress_bar(current: int, total: int, width: int = 20) -> str:
    filled = int(width * current / total)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {current}/{total}"


def create_synthetic_dataset(
    n_samples: int, input_dim: int, n_classes: int, seed: int = 42
):
    seed_everything(seed)

    centers = torch.randn(n_classes, input_dim) * 2
    samples_per_class = n_samples // n_classes
    X, y = [], []

    for c in range(n_classes):
        class_samples = centers[c] + torch.randn(samples_per_class, input_dim) * 0.5
        X.append(class_samples)
        y.append(torch.full((samples_per_class,), c, dtype=torch.long))

    X, y = torch.cat(X), torch.cat(y)
    perm = torch.randperm(len(y))
    return X[perm], y[perm]


def train_model(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 50,
    lr: float = 0.01,
    name: str = "Model",
    verifier=None,
    track_id=0,
    seed=0,
) -> list[float]:
    """Train a model on a full batch, routing each step through the shared seam.

    Each epoch delegates to :func:`dispatch_train_step`, so models exposing a
    ``train_step`` (learning-rule) use it while plain models fall back to the
    canonical BPTT fallback.
    """
    optimizer = create_optimizer(model, OptimizerConfig(name="adam", lr=lr))
    losses = []
    perfect_streak = 0  # Track consecutive 100% accuracy epochs

    for epoch in range(epochs):
        metrics = dispatch_train_step(
            model=model,
            x=X,
            y=y,
            adapt_input=lambda x: x,
            optimizer=optimizer,
            config=None,
        )
        loss = float(metrics.get("loss", 0.0))
        losses.append(loss)

        grad_norm = sum(
            p.grad.norm().item() for p in model.parameters() if p.grad is not None
        )

        if verifier:
            verifier.record_metric(track_id, seed, epoch, f"{name}_loss", loss)
            verifier.record_metric(
                track_id, seed, epoch, f"{name}_grad_norm", grad_norm
            )

        logits = metrics.get("logits")
        if logits is None:
            logits = model(X)
        acc = compute_accuracy(logits, y, scale=100)
        logger.debug(
            "  %s: %s loss=%.3f acc=%.1f%%",
            name,
            progress_bar(epoch + 1, epochs),
            loss,
            acc,
        )

        # Early stopping: if 100% accuracy for 3 consecutive epochs, stop
        if acc >= 100.0:
            perfect_streak += 1
            if perfect_streak >= 3:
                logger.info("  [early stop] %s", name)
                break
        else:
            perfect_streak = 0

    logger.info("Training complete for %s", name)
    return losses


def evaluate_accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out = model(X)
            acc = compute_accuracy(out, y)
    finally:
        if was_training:
            model.train()
    return acc


def format_metrics_table(
    metrics: dict[str, object], headers: tuple[str, str] = ("Metric", "Value")
) -> str:
    """Format a dict of metrics as a markdown table."""
    rows = [f"| {k} | {format_value(v)} |" for k, v in metrics.items()]
    header = f"| {headers[0]} | {headers[1]} |"
    separator = (
        "|" + "-" * (len(headers[0]) + 2) + "|" + "-" * (len(headers[1]) + 2) + "|"
    )
    return "\n".join([header, separator] + rows)  # ruff: ignore[collection-literal-concatenation]


def format_value(v: object) -> str:
    """Format a value for display in a table."""
    if isinstance(v, float):
        if abs(v) < 0.01 or abs(v) > 1000:
            return f"{v:.2e}"
        return f"{v:.3f}"
    elif isinstance(v, bool):
        return "[OK]  Yes" if v else "[FAIL]  No"
    return str(v)


def determine_status(
    score: float, pass_threshold: float = 80, partial_threshold: float = 50
) -> str:
    """Determine track status based on score."""
    if score >= pass_threshold:
        return "pass"
    elif score >= partial_threshold:
        return "partial"
    else:
        return "fail"


def robustness_summary(results: dict, metric_name: str = "accuracy") -> str:
    """Format robustness test results summary."""
    if f"{metric_name}_mean" not in results.get("metrics", {}):
        return "N/A"

    mean = results["metrics"][f"{metric_name}_mean"]
    std = results["metrics"].get(f"{metric_name}_std", 0.0)

    return f"{mean * 100:.1f}% ± {std * 100:.1f}%"


# =============================================================================
# SCIENTIFIC RIGOR ENHANCEMENTS
# =============================================================================


def compute_cohens_d(group1: list[float], group2: list[float]) -> float:
    """
    Compute Cohen's d effect size between two groups.

    Interpretation:
        |d| < 0.2: negligible
        |d| 0.2-0.5: small
        |d| 0.5-0.8: medium
        |d| > 0.8: large
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0

    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std < 1e-10:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def paired_ttest(group1: list[float], group2: list[float]) -> tuple[float, float]:
    """
    Perform paired t-test between two groups.

    Returns:
        (t_statistic, p_value)
    """
    from scipy import stats

    if len(group1) != len(group2) or len(group1) < 2:
        return (0.0, 1.0)

    # Handle zero-variance case (all differences are 0)
    diffs = [a - b for a, b in zip(group1, group2)]
    if np.std(diffs) < 1e-10:
        return (0.0, 1.0)  # No difference = not significant

    t_stat, p_val = stats.ttest_rel(group1, group2)
    return (float(t_stat), float(p_val))


def independent_ttest(group1: list[float], group2: list[float]) -> tuple[float, float]:
    """
    Perform independent samples t-test between two groups.

    Returns:
        (t_statistic, p_value)
    """
    from scipy import stats

    if len(group1) < 2 or len(group2) < 2:
        return (0.0, 1.0)

    t_stat, p_val = stats.ttest_ind(group1, group2)
    return (float(t_stat), float(p_val))


def classify_evidence_level(n_samples: int, n_seeds: int, epochs: int) -> str:
    """
    Classify evidence strength based on experimental parameters.

    Returns:
        "smoke": Mechanics only, not statistically meaningful
        "directional": Indicative of trend, limited confidence
        "conclusive": Statistically robust, publication-ready
    """
    # Smoke test: minimal parameters
    if n_samples < 500 or n_seeds < 2 or epochs < 10:
        return "smoke"

    # Conclusive: substantial parameters
    if n_samples >= 5000 and n_seeds >= 3 and epochs >= 50:
        return "conclusive"

    return "directional"


def interpret_effect_size(d: float) -> str:
    """Human-readable interpretation of Cohen's d."""
    abs_d = abs(d)
    direction = "higher" if d > 0 else "lower"

    if abs_d < 0.2:
        return f"negligible ({d:+.2f})"
    elif abs_d < 0.5:
        return f"small effect, {direction} ({d:+.2f})"
    elif abs_d < 0.8:
        return f"medium effect, {direction} ({d:+.2f})"
    else:
        return f"**large effect**, {direction} ({d:+.2f})"


def interpret_pvalue(p: float) -> str:
    """Human-readable interpretation of p-value."""
    if p < 0.001:
        return "***p < 0.001*** (highly significant)"
    elif p < 0.01:
        return f"**p = {p:.3f}** (significant)"
    elif p < 0.05:
        return f"*p = {p:.3f}* (marginally significant)"
    else:
        return f"p = {p:.3f} (not significant)"


def format_statistical_comparison(
    name1: str,
    values1: list[float],
    name2: str,
    values2: list[float],
    metric_name: str = "accuracy",
) -> str:
    """
    Generate a complete statistical comparison report.

    Returns markdown-formatted statistical summary with:
    - Means with 95% CI
    - Cohen's d effect size
    - p-value from t-test
    """
    n1, n2 = len(values1), len(values2)

    mean1 = np.mean(values1)
    mean2 = np.mean(values2)

    # 95% CI  # ruff: ignore[commented-out-code]
    se1 = np.std(values1, ddof=1) / np.sqrt(n1) if n1 > 1 else 0
    se2 = np.std(values2, ddof=1) / np.sqrt(n2) if n2 > 1 else 0
    ci1 = 1.96 * se1
    ci2 = 1.96 * se2

    # Effect size and p-value
    d = compute_cohens_d(values1, values2)

    # Use independent t-test if different sizes, paired if same
    if n1 == n2:
        _, p = paired_ttest(values1, values2)
    else:
        _, p = independent_ttest(values1, values2)

    return f"""### Statistical Comparison: {name1} vs {name2}

| Metric | {name1} | {name2} |
|--------|---------|---------|
| Mean {metric_name} | {mean1:.3f} | {mean2:.3f} |
| 95% CI | ±{ci1:.3f} | ±{ci2:.3f} |
| n | {n1} | {n2} |

**Effect Size**: {interpret_effect_size(d)}
**Significance**: {interpret_pvalue(p)}
"""


def format_claim_with_evidence(
    claim: str, evidence: str, evidence_level: str, limitations: list[str] | None = None
) -> str:
    """
    Format a scientific claim with structured evidence and limitations.

    This creates undeniable, honest reporting that acknowledges constraints.
    """
    level_icons = {"smoke": "[TEST] ", "directional": "[DATA] ", "conclusive": "[OK] "}
    level_labels = {
        "smoke": "Smoke Test (mechanics verified)",
        "directional": "Directional (trend observed)",
        "conclusive": "Conclusive (statistically significant)",
    }

    icon = level_icons.get(evidence_level, "❓")
    label = level_labels.get(evidence_level, "Unknown")

    result = f"""
> **Claim**: {claim}
>
> {icon} **Evidence Level**: {label}

{evidence}
"""

    if limitations:
        result += "\n**Limitations**:\n"
        for lim in limitations:
            result += f"- {lim}\n"

    return result


def compute_reproducibility_hash(
    seed: int, n_samples: int, epochs: int, model_name: str
) -> str:
    """Generate a short hash for reproducibility tracking."""
    import hashlib

    content = f"{seed}-{n_samples}-{epochs}-{model_name}"
    return hashlib.md5(content.encode()).hexdigest()[:8]  # ruff: ignore[hashlib-insecure-hash-function]
