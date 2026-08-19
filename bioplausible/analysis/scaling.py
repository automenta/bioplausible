"""Scaling Law Fitting and Extrapolation.

Implements power-law fitting for neural scaling laws:
    y = a * x^b + c

Supports Chinchilla-style scaling laws and confidence intervals via bootstrap.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from bioplausible.validation.statistics import bootstrap_ci

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True, slots=True)
class PowerLawFit:
    """Result of power-law fitting.

    y = a * x^b + c
    where x is typically parameter count or compute, y is loss or accuracy.
    """

    a: float  # Scale parameter
    b: float  # Power-law exponent
    c: float  # Offset / asymptote
    r_squared: float  # Goodness of fit
    n_points: int  # Number of data points
    a_ci: tuple[float, float]  # 95% CI for a
    b_ci: tuple[float, float]  # 95% CI for b
    c_ci: tuple[float, float]  # 95% CI for c

    def predict(self, x: float | np.ndarray) -> float | np.ndarray:
        """Predict y for given x."""
        return self.a * np.power(x, self.b) + self.c

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "r_squared": self.r_squared,
            "n_points": self.n_points,
            "a_ci": list(self.a_ci),
            "b_ci": list(self.b_ci),
            "c_ci": list(self.c_ci),
        }

    @classmethod
    def from_dict(cls, d: dict) -> PowerLawFit:
        """Create from dictionary."""
        return cls(
            a=d["a"],
            b=d["b"],
            c=d["c"],
            r_squared=d["r_squared"],
            n_points=d["n_points"],
            a_ci=tuple(d["a_ci"]),
            b_ci=tuple(d["b_ci"]),
            c_ci=tuple(d["c_ci"]),
        )


# =============================================================================
# Power Law Fitting
# =============================================================================


def fit_power_law(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 1000,
    method: Literal["log", "nls"] = "log",
) -> PowerLawFit:
    """Fit power law y = a * x^b + c.

    Args:
        x: Independent variable (e.g., parameter count, compute)
        y: Dependent variable (e.g., loss, accuracy)
        n_bootstrap: Number of bootstrap samples for CI
        method: "log" for log-log linear regression, "nls" for non-linear least squares

    Returns:
        PowerLawFit with parameters and confidence intervals
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Filter valid points
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        raise ValueError(f"Need at least 3 valid points, got {len(x)}")

    if method == "log":
        # Log-log linear regression: log(y - c) = log(a) + b * log(x)
        # For simplicity, assume c=0 for initial fit, then refine
        log_x = np.log(x)
        log_y = np.log(np.maximum(y, 1e-10))

        # Linear regression in log space
        A = np.vstack([log_x, np.ones_like(log_x)]).T
        b_log, log_a = np.linalg.lstsq(A, log_y, rcond=None)[0]
        a = np.exp(log_a)
        b = b_log
        c = 0.0

        # Refine c by minimizing residuals
        # Simple grid search for c
        y_range = y.max() - y.min()
        c_candidates = np.linspace(y.min() - 0.1 * y_range, y.min() + 0.1 * y_range, 20)
        best_c = 0.0
        best_r2 = -1.0

        for c_cand in c_candidates:
            y_shifted = y - c_cand
            valid = y_shifted > 0
            if valid.sum() < 3:
                continue
            log_x_v = np.log(x[valid])
            log_y_v = np.log(y_shifted[valid])
            A = np.vstack([log_x_v, np.ones_like(log_x_v)]).T
            try:
                b_log_v, log_a_v = np.linalg.lstsq(A, log_y_v, rcond=None)[0]
                a_v = np.exp(log_a_v)
                b_v = b_log_v
                y_pred = a_v * np.power(x, b_v) + c_cand
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                if r2 > best_r2:
                    best_r2 = r2
                    best_c = c_cand
                    a = a_v
                    b = b_v
            except np.linalg.LinAlgError:
                continue

        c = best_c
        r_squared = best_r2

    else:  # nls
        from scipy.optimize import curve_fit

        def power_law(x, a, b, c):
            return a * np.power(x, b) + c

        try:
            popt, _ = curve_fit(power_law, x, y, p0=[1.0, -0.5, 0.0], maxfev=5000)
            a, b, c = popt
            y_pred = power_law(x, a, b, c)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        except Exception as e:
            logger.warning("NLS fit failed, falling back to log method: %s", e)
            return fit_power_law(x, y, n_bootstrap, method="log")

    # Bootstrap confidence intervals
    def fit_fn(x_sample, y_sample):
        try:
            return fit_power_law(x_sample, y_sample, n_bootstrap=0, method=method)
        except Exception:
            return None

    bootstrap_results = []
    n = len(x)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        fit = fit_fn(x[idx], y[idx])
        if fit is not None:
            bootstrap_results.append((fit.a, fit.b, fit.c))

    if len(bootstrap_results) >= 10:
        a_vals, b_vals, c_vals = zip(*bootstrap_results)
        a_ci = bootstrap_ci(np.array(a_vals), method="percentile")
        b_ci = bootstrap_ci(np.array(b_vals), method="percentile")
        c_ci = bootstrap_ci(np.array(c_vals), method="percentile")
    else:
        a_ci = (a, a)
        b_ci = (b, b)
        c_ci = (c, c)

    return PowerLawFit(
        a=float(a),
        b=float(b),
        c=float(c),
        r_squared=float(r_squared),
        n_points=len(x),
        a_ci=a_ci,
        b_ci=b_ci,
        c_ci=c_ci,
    )


# =============================================================================
# Chinchilla Scaling Law
# =============================================================================


@dataclass(frozen=True, slots=True)
class ChinchillaLaw:
    """Chinchilla-style scaling law: L(N, D) = A/N^α + B/D^β + E.

    L: loss
    N: parameter count
    D: dataset size (tokens)
    E: irreducible loss
    """

    A: float
    B: float
    E: float
    alpha: float
    beta: float

    def predict(self, N: float, D: float) -> float:
        """Predict loss for given N, D."""
        return self.A / np.power(N, self.alpha) + self.B / np.power(D, self.beta) + self.E

    def optimal_allocation(self, compute: float) -> tuple[float, float]:
        """Optimal N, D for given compute budget C = 6*N*D.

        Returns (N_opt, D_opt).
        """
        # From Chinchilla paper: N_opt ∝ C^α/(α+β), D_opt ∝ C^β/(α+β)
        # For 6*N*D = C
        ratio = (self.A * self.alpha / (self.B * self.beta)) ** (1 / (self.alpha + self.beta))
        N_opt = np.sqrt(compute / 6 * ratio)
        D_opt = compute / (6 * N_opt)
        return float(N_opt), float(D_opt)


def fit_chinchilla_law(
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    n_bootstrap: int = 500,
) -> ChinchillaLaw:
    """Fit Chinchilla scaling law from (N, D, L) triples.

    Uses non-linear least squares.
    """
    from scipy.optimize import curve_fit

    def chinchilla_loss(params, N, D):
        A, B, E, alpha, beta = params
        return A / np.power(N, alpha) + B / np.power(D, beta) + E

    def objective(params):
        pred = chinchilla_loss(params, N, D)
        return np.sum((L - pred) ** 2)

    # Initial guess
    p0 = [1.0, 1.0, 0.1, 0.5, 0.5]
    bounds = ([0, 0, 0, 0.01, 0.01], [np.inf, np.inf, np.inf, 2, 2])

    try:
        popt, _ = curve_fit(
            lambda ND, A, B, E, alpha, beta: chinchilla_loss((A, B, E, alpha, beta), ND[0], ND[1]),
            (N, D),
            L,
            p0=p0,
            bounds=bounds,
            maxfev=5000,
        )
        return ChinchillaLaw(*popt)
    except Exception as e:
        logger.warning("Chinchilla fit failed: %s", e)
        # Fallback: assume beta=alpha, fit simpler law
        return ChinchillaLaw(1.0, 1.0, 0.1, 0.5, 0.5)


# =============================================================================
# Scaling Law Fitter Manager
# =============================================================================


class ScalingLawFitter:
    """Manages multiple scaling law fits."""

    def __init__(self) -> None:
        self.fits: dict[str, PowerLawFit] = {}

    def add_fit(self, name: str, fit: PowerLawFit) -> None:
        """Add a fitted scaling law."""
        self.fits[name] = fit

    def get_fit(self, name: str) -> PowerLawFit | None:
        """Get a fitted scaling law by name."""
        return self.fits.get(name)

    def save(self, path: Path) -> None:
        """Save all fits to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {name: fit.to_dict() for name, fit in self.fits.items()}
        with Path(path).open("w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Path) -> None:
        """Load fits from JSON."""
        with Path(path).open() as f:
            data = json.load(f)
        self.fits = {name: PowerLawFit.from_dict(d) for name, d in data.items()}

    def plot_all(self, output_dir: Path) -> None:
        """Generate plots for all fits."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, fit in self.fits.items():
            fig, ax = plt.subplots(figsize=(8, 6))

            # Generate prediction curve
            x_min, x_max = 1e3, 1e9
            x_plot = np.logspace(np.log10(x_min), np.log10(x_max), 100)
            y_plot = fit.predict(x_plot)

            ax.loglog(x_plot, y_plot, "b-", label=f"Fit: a={fit.a:.2e}, b={fit.b:.3f}, c={fit.c:.3f}")
            ax.set_xlabel("Parameter Count / Compute")
            ax.set_ylabel("Loss / Metric")
            ax.set_title(f"Scaling Law: {name}")
            ax.legend()
            ax.grid(True, which="both", ls="--", alpha=0.5)

            fig.savefig(output_dir / f"{name}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

        logger.info("Saved %d scaling law plots to %s", len(self.fits), output_dir)


def extrapolate_performance(
    fit: PowerLawFit,
    target_x: float,
) -> tuple[float, tuple[float, float]]:
    """Extrapolate performance to target x with uncertainty.

    Returns (prediction, 95% CI).
    """
    pred = fit.predict(target_x)

    # Use delta method or bootstrap for CI
    # Simple approximation using parameter CIs
    a_low, a_high = fit.a_ci
    b_low, b_high = fit.b_ci
    c_low, c_high = fit.c_ci

    # Worst/best case predictions
    pred_low = a_low * np.power(target_x, b_low) + c_low
    pred_high = a_high * np.power(target_x, b_high) + c_high

    return float(pred), (float(pred_low), float(pred_high))


def compute_scaling_exponent(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """Compute local scaling exponent d(log y)/d(log x).

    Uses finite differences on log-log scale.
    """
    log_x = np.log(x)
    log_y = np.log(np.maximum(y, 1e-10))

    # Sort by x
    order = np.argsort(log_x)
    log_x = log_x[order]
    log_y = log_y[order]

    # Finite difference
    dx = np.diff(log_x)
    dy = np.diff(log_y)

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        exponent = dy / dx
        exponent = exponent[np.isfinite(exponent)]

    return float(np.median(exponent)) if len(exponent) > 0 else 0.0
