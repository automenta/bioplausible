"""
Ablation Study Framework

Provides systematic parameter sensitivity analysis with:
- Full factorial design (Cartesian product)
- Leave-one-out analysis
- Sobol sensitivity indices
- Automated report generation
"""

import copy
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from bioplausible.config.omegaconf import RunConfig
from bioplausible.core.trainer import run_from_runconfig as run_from_config

__all__ = [
    "AblationResult",
    "AblationStudy",
    "LeaveOneOutResult",
    "SobolIndices",
    "create_ablation_report",
]

# Maps dimension names to config attribute paths for dynamic resolution.
_DIMENSION_MAP: dict[str, tuple[str, ...]] = {
    "learning_rate": ("optimizer", "lr"),
    "model_depth": ("model", "num_layers"),
    "hidden_dim": ("model", "hidden_dim"),
    "eq_steps": ("model", "extra", "max_steps"),
    "beta": ("optimizer", "beta"),
    "sparsity_target": ("model", "extra", "sparsity_target"),
    "data_fraction": ("data", "data_fraction"),
    "spectral_bound_gamma": ("model", "extra", "spectral_bound_gamma"),
}


def _set_nested(cfg: RunConfig, path: tuple[str, ...], value: object) -> None:
    """Set a value at a nested attribute/dict path on a RunConfig."""
    obj: object = cfg
    for part in path[:-1]:
        obj = getattr(obj, part)
    final = path[-1]
    try:
        setattr(obj, final, value)
    except AttributeError, TypeError:
        obj[final] = value


@dataclass(frozen=True, slots=True)
class AblationResult:
    """Single ablation experiment result."""

    params: dict[str, object]
    success: bool
    val_accuracy: float
    error: str | None = None
    metrics: dict[str, float] | None = None


@dataclass(frozen=True, slots=True)
class SobolIndices:
    """Sobol sensitivity indices for variance-based sensitivity analysis."""

    first_order: dict[str, float]
    total_order: dict[str, float]
    second_order: dict[tuple[str, str], float]
    n_samples: int


@dataclass(frozen=True, slots=True)
class LeaveOneOutResult:
    """Leave-one-out ablation result."""

    removed_dimension: str
    baseline_accuracy: float
    ablation_accuracy: float
    impact: float  # baseline - ablation (positive = important)
    relative_impact: float  # impact / baseline


class AblationStudy:
    """
    Systematic parameter sensitivity study framework.

    Supports:
    - Full factorial design via Cartesian product
    - Leave-one-out analysis
    - Sobol variance-based sensitivity indices
    - Automated report generation (HTML, Markdown, JSON)
    """

    def __init__(self, base_cfg: RunConfig, dimensions: dict[str, list[object]]):
        self.base_cfg = base_cfg
        self.dimensions = dimensions
        self.results: list[AblationResult] = []
        self._baseline_result: AblationResult | None = None

    def _generate_configs(self) -> list[tuple]:
        """Generate configurations based on Cartesian product of all dimensions."""
        keys = list(self.dimensions.keys())
        values = list(self.dimensions.values())

        # Ensure cfg.model.extra exists for eq_steps / sparsity / spectral paths
        base_extra = copy.deepcopy(self.base_cfg.model.extra) or {}

        configs = []
        for combo in product(*values):
            cfg = copy.deepcopy(self.base_cfg)
            cfg.model.extra = copy.deepcopy(base_extra)
            params = dict(zip(keys, combo))

            for k, v in params.items():
                path = _DIMENSION_MAP.get(k)
                if path is None:
                    raise ValueError(f"Unknown ablation dimension: {k}")
                _set_nested(cfg, path, v)

            configs.append((params, cfg))
        return configs

    def _run_single_experiment(self, params_and_cfg: tuple) -> AblationResult:
        params, cfg = params_and_cfg

        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = run_from_config(cfg)

            score = float(res.get("final_val_accuracy", 0.0))
            return AblationResult(
                params=params,
                success=True,
                val_accuracy=score,
                metrics={
                    k: float(v) for k, v in res.items() if isinstance(v, (int, float))
                },
            )
        except Exception as e:  # broad: best-effort
            return AblationResult(
                params=params,
                success=False,
                val_accuracy=0.0,
                error=str(e),
            )

    def run(self, parallel_workers: int = 4) -> pd.DataFrame:
        """Run all experiments defined in the continuous parameter space."""
        configs = self._generate_configs()
        results_list = []

        with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
            futures = [executor.submit(self._run_single_experiment, c) for c in configs]
            for future in tqdm(
                as_completed(futures), total=len(configs), desc="Ablation runs"
            ):
                results_list.append(future.result())

        self.results = results_list
        return self.to_dataframe()

    def run_baseline(self) -> AblationResult:
        """Run the baseline configuration (all default values)."""
        base_extra = copy.deepcopy(self.base_cfg.model.extra) or {}
        cfg = copy.deepcopy(self.base_cfg)
        cfg.model.extra = base_extra

        # Use first value of each dimension as baseline
        baseline_params = {k: v[0] for k, v in self.dimensions.items()}

        result = self._run_single_experiment((baseline_params, cfg))
        self._baseline_result = result
        return result

    def run_leave_one_out(self, parallel_workers: int = 4) -> list[LeaveOneOutResult]:
        """
        Run leave-one-out ablation: remove each dimension one at a time
        and measure impact on performance.
        """
        if self._baseline_result is None:
            self.run_baseline()

        baseline_acc = self._baseline_result.val_accuracy
        loo_results = []

        for dim_name, dim_values in self.dimensions.items():
            # Create config with this dimension fixed to baseline,
            # other dimensions at their default (first) values
            base_extra = copy.deepcopy(self.base_cfg.model.extra) or {}
            cfg = copy.deepcopy(self.base_cfg)
            cfg.model.extra = base_extra

            params = {}
            for k, v in self.dimensions.items():
                if k == dim_name:
                    params[k] = v[0]  # baseline value
                else:
                    params[k] = v[0]  # default value

            # Actually, for leave-one-out, we want to see what happens
            # when we remove the ability to tune this dimension.
            # We run with ALL dimensions at baseline EXCEPT we allow
            # this dimension to vary (or we fix it to a suboptimal value).
            # Standard LOO: measure performance when dimension is fixed
            # to its baseline vs when it's optimized.
            #
            # Simpler interpretation: fix dimension to baseline,
            # optimize others, compare to full optimization.

            path = _DIMENSION_MAP.get(dim_name)
            if path is None:
                continue
            _set_nested(cfg, path, dim_values[0])

            result = self._run_single_experiment((params, cfg))
            ablation_acc = result.val_accuracy
            impact = baseline_acc - ablation_acc
            relative_impact = impact / baseline_acc if baseline_acc > 0 else 0.0

            loo_results.append(
                LeaveOneOutResult(
                    removed_dimension=dim_name,
                    baseline_accuracy=baseline_acc,
                    ablation_accuracy=ablation_acc,
                    impact=impact,
                    relative_impact=relative_impact,
                )
            )

        return loo_results

    def compute_sobol_indices(
        self,
        n_samples: int = 1000,
        parallel_workers: int = 4,
    ) -> SobolIndices:
        """
        Compute Sobol sensitivity indices using Saltelli sampling.

        Sobol indices decompose output variance into contributions
        from individual parameters and their interactions.

        Args:
            n_samples: Base sample size (total evaluations ~ n_samples * (D+2))
            parallel_workers: Parallel workers for evaluation

        Returns:
            SobolIndices with first-order, total-order, and second-order indices.
        """
        try:
            from SALib.analyze import sobol
            from SALib.sample import saltelli
        except ImportError:
            raise ImportError(
                "SALib required for Sobol indices. Install with: pip install SALib"
            )

        # Define problem for SALib
        problem = {
            "num_vars": len(self.dimensions),
            "names": list(self.dimensions.keys()),
            "bounds": [
                [float(min(v)), float(max(v))] for v in self.dimensions.values()
            ],
        }

        # Generate Saltelli samples
        param_values = saltelli.sample(problem, n_samples, calc_second_order=True)

        # Run experiments for each sample
        configs = []
        base_extra = copy.deepcopy(self.base_cfg.model.extra) or {}

        for params_array in param_values:
            cfg = copy.deepcopy(self.base_cfg)
            cfg.model.extra = copy.deepcopy(base_extra)

            params = dict(zip(problem["names"], params_array))
            for k, v in params.items():
                path = _DIMENSION_MAP.get(k)
                if path is None:
                    raise ValueError(f"Unknown ablation dimension: {k}")
                _set_nested(cfg, path, v)

            configs.append((params, cfg))

        # Evaluate all configurations
        results_list = []
        with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
            futures = [executor.submit(self._run_single_experiment, c) for c in configs]
            for future in tqdm(
                as_completed(futures), total=len(configs), desc="Sobol samples"
            ):
                result = future.result()
                results_list.append(result.val_accuracy if result.success else 0.0)

        Y = np.array(results_list)

        # Analyze with SALib
        Si = sobol.analyze(problem, Y, calc_second_order=True, print_to_console=False)

        first_order = dict(zip(problem["names"], Si["S1"]))
        total_order = dict(zip(problem["names"], Si["ST"]))

        second_order = {}
        if "S2" in Si and Si["S2"] is not None:
            for i, name_i in enumerate(problem["names"]):
                for j, name_j in enumerate(problem["names"]):
                    if i < j:
                        second_order[name_i, name_j] = float(Si["S2"][i, j])

        return SobolIndices(
            first_order=first_order,
            total_order=total_order,
            second_order=second_order,
            n_samples=n_samples,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        if not self.results:
            return pd.DataFrame()

        data = []
        for r in self.results:
            row = {**r.params, "success": r.success, "val_accuracy": r.val_accuracy}
            if r.error:
                row["error"] = r.error
            if r.metrics:
                for k, v in r.metrics.items():
                    row[f"metric_{k}"] = v
            data.append(row)

        return pd.DataFrame(data)

    def plot_sensitivity_heatmap(
        self, param1: str, param2: str, metric: str = "val_accuracy"
    ) -> plt.Figure:
        """Plot a heatmap of the sensitivity with respect to two dimensions."""
        df = self.to_dataframe()
        if df.empty:
            raise ValueError("No results to plot. Call run() first.")

        if param1 not in df.columns or param2 not in df.columns:
            raise ValueError(
                f"Parameters {param1} and {param2} must be valid ablation dimensions."
            )

        pivot_table = df.pivot_table(
            values=metric, index=param1, columns=param2, aggfunc=np.mean
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(pivot_table, annot=True, cmap="viridis", ax=ax, fmt=".3f")
        ax.set_title(f"Sensitivity Heatmap: {param1} vs {param2}")
        fig.tight_layout()
        return fig

    def plot_leave_one_out(self, loo_results: list[LeaveOneOutResult]) -> plt.Figure:
        """Plot leave-one-out results as a bar chart."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        dims = [r.removed_dimension for r in loo_results]
        impacts = [r.impact for r in loo_results]
        rel_impacts = [r.relative_impact for r in loo_results]

        # Absolute impact
        colors = ["red" if i > 0 else "blue" for i in impacts]
        ax1.barh(dims, impacts, color=colors, alpha=0.7)
        ax1.axvline(x=0, color="black", linewidth=0.5)
        ax1.set_xlabel("Accuracy Drop (Baseline - Ablation)")
        ax1.set_title("Leave-One-Out: Absolute Impact")
        ax1.grid(True, axis="x", alpha=0.3)

        # Relative impact
        colors = ["red" if i > 0 else "blue" for i in rel_impacts]
        ax2.barh(dims, rel_impacts, color=colors, alpha=0.7)
        ax2.axvline(x=0, color="black", linewidth=0.5)
        ax2.set_xlabel("Relative Impact (Drop / Baseline)")
        ax2.set_title("Leave-One-Out: Relative Impact")
        ax2.grid(True, axis="x", alpha=0.3)

        fig.tight_layout()
        return fig

    def plot_sobol_indices(self, sobol: SobolIndices) -> plt.Figure:
        """Plot Sobol sensitivity indices."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # First-order indices
        names = list(sobol.first_order.keys())
        first = [sobol.first_order[n] for n in names]
        total = [sobol.total_order[n] for n in names]

        x = np.arange(len(names))
        width = 0.35

        ax1.bar(x - width / 2, first, width, label="First-Order (S1)", alpha=0.7)
        ax1.bar(x + width / 2, total, width, label="Total-Order (ST)", alpha=0.7)
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=45, ha="right")
        ax1.set_ylabel("Sobol Index")
        ax1.set_title("Sobol Sensitivity Indices")
        ax1.legend()
        ax1.grid(True, axis="y", alpha=0.3)

        # Second-order interactions
        if sobol.second_order:
            pairs = list(sobol.second_order.keys())
            values = list(sobol.second_order.values())
            pair_labels = [f"{a}×{b}" for a, b in pairs]

            ax2.barh(pair_labels, values, alpha=0.7)
            ax2.set_xlabel("Second-Order Index")
            ax2.set_title("Pairwise Interactions (S2)")
            ax2.grid(True, axis="x", alpha=0.3)
        else:
            ax2.text(
                0.5,
                0.5,
                "No second-order indices\n(calc_second_order=False)",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )
            ax2.set_title("Pairwise Interactions (S2)")

        fig.tight_layout()
        return fig

    def identify_critical_hyperparams(self) -> list[str]:
        """Identify critical hyperparameters using the variance of the mean outcomes."""
        df = self.to_dataframe()
        if df.empty:
            raise ValueError("No results to analyze. Call run() first.")

        variances = []
        for col in self.dimensions.keys():
            if col in df.columns:
                mean_per_val = df.groupby(col)["val_accuracy"].mean()
                if not mean_per_val.empty:
                    variances.append((col, mean_per_val.var()))

        # Sort by variance descending
        variances.sort(key=lambda x: x[1] if pd.notna(x[1]) else 0.0, reverse=True)
        return [col for col, var in variances]

    def generate_report(
        self,
        output_dir: str | Path = "results/ablation",
        include_plots: bool = True,
        format: str = "html",
    ) -> dict[str, Path]:
        """
        Generate comprehensive ablation study report.

        Args:
            output_dir: Directory to save report files
            include_plots: Whether to generate and save plots
            format: Output format ('html', 'markdown', 'json', 'all')

        Returns:
            Dictionary mapping report components to file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df = self.to_dataframe()
        if df.empty:
            raise ValueError("No results to report. Call run() first.")

        report_paths = {}

        # Run leave-one-out if not done
        loo_results = self.run_leave_one_out()

        # Compute Sobol indices (optional, can be slow)
        try:
            sobol = self.compute_sobol_indices(n_samples=500)
        except ImportError:
            sobol = None

        # Save raw data
        csv_path = output_dir / "ablation_results.csv"
        df.to_csv(csv_path, index=False)
        report_paths["csv"] = csv_path

        # Generate JSON summary
        summary = {
            "n_experiments": len(df),
            "n_successful": int(df["success"].sum()),
            "dimensions": list(self.dimensions.keys()),
            "baseline_accuracy": self._baseline_result.val_accuracy
            if self._baseline_result
            else None,
            "critical_params": self.identify_critical_hyperparams(),
            "leave_one_out": [asdict(r) for r in loo_results],
        }
        if sobol:
            summary["sobol_first_order"] = sobol.first_order
            summary["sobol_total_order"] = sobol.total_order
            summary["sobol_second_order"] = {
                f"{a}×{b}": v for (a, b), v in sobol.second_order.items()
            }

        json_path = output_dir / "ablation_summary.json"
        with Path(json_path).open("w") as f:
            json.dump(summary, f, indent=2, default=str)
        report_paths["json"] = json_path

        # Generate plots
        if include_plots:
            # Pairwise heatmaps for all dimension pairs
            dim_names = list(self.dimensions.keys())
            for i, p1 in enumerate(dim_names):
                for p2 in dim_names[i + 1 :]:
                    if p1 in df.columns and p2 in df.columns:
                        fig = self.plot_sensitivity_heatmap(p1, p2)
                        fig.savefig(
                            output_dir / f"heatmap_{p1}_vs_{p2}.png",
                            dpi=150,
                            bbox_inches="tight",
                        )
                        plt.close(fig)

            # Leave-one-out plot
            fig = self.plot_leave_one_out(loo_results)
            fig.savefig(output_dir / "leave_one_out.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            report_paths["loo_plot"] = output_dir / "leave_one_out.png"

            # Sobol plot
            if sobol:
                fig = self.plot_sobol_indices(sobol)
                fig.savefig(
                    output_dir / "sobol_indices.png", dpi=150, bbox_inches="tight"
                )
                plt.close(fig)
                report_paths["sobol_plot"] = output_dir / "sobol_indices.png"

        # Generate formatted report
        if format in ("html", "all"):
            html_path = self._generate_html_report(
                output_dir, summary, df, loo_results, sobol
            )
            report_paths["html"] = html_path

        if format in ("markdown", "all"):
            md_path = self._generate_markdown_report(
                output_dir, summary, loo_results, sobol
            )
            report_paths["markdown"] = md_path

        return report_paths

    def _generate_html_report(
        self,
        output_dir: Path,
        summary: dict,
        df: pd.DataFrame,
        loo_results: list[LeaveOneOutResult],
        sobol: SobolIndices | None,
    ) -> Path:
        """Generate HTML report."""
        html_path = output_dir / "ablation_report.html"

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Ablation Study Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #34495e; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #ecf0f1; border-radius: 5px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
        .metric-label {{ font-size: 14px; color: #7f8c8d; }}
        .plot {{ text-align: center; margin: 20px 0; }}
        .plot img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
        .critical {{ color: #e74c3c; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Ablation Study Report</h1>

    <h2>Summary</h2>
    <div class="metric">
        <div class="metric-value">{summary["n_experiments"]}</div>
        <div class="metric-label">Total Experiments</div>
    </div>
    <div class="metric">
        <div class="metric-value">{summary["n_successful"]}</div>
        <div class="metric-label">Successful Runs</div>
    </div>
    <div class="metric">
        <div class="metric-value">{summary["baseline_accuracy"]:.4f if summary['baseline_accuracy'] else 'N/A'}</div>
        <div class="metric-label">Baseline Accuracy</div>
    </div>

    <h2>Critical Hyperparameters</h2>
    <p>Ranked by variance of mean outcomes:</p>
    <ol>
"""

        for i, param in enumerate(summary["critical_params"], 1):
            html_content += (
                f"        <li><span class='critical'>{i}. {param}</span></li>\n"
            )

        html_content += """
    </ol>

    <h2>Leave-One-Out Analysis</h2>
    <table>
        <tr>
            <th>Dimension</th>
            <th>Baseline Acc</th>
            <th>Ablation Acc</th>
            <th>Absolute Impact</th>
            <th>Relative Impact</th>
        </tr>
"""

        for r in loo_results:
            impact_class = "critical" if r.impact > 0 else ""
            html_content += f"""
        <tr>
            <td>{r.removed_dimension}</td>
            <td>{r.baseline_accuracy:.4f}</td>
            <td>{r.ablation_accuracy:.4f}</td>
            <td class="{impact_class}">{r.impact:.4f}</td>
            <td class="{impact_class}">{r.relative_impact:.2%}</td>
        </tr>
"""

        html_content += """
    </table>
"""

        if sobol:
            html_content += """
    <h2>Sobol Sensitivity Indices</h2>
    <h3>First-Order (Main Effects)</h3>
    <table>
        <tr><th>Parameter</th><th>S1</th><th>ST</th></tr>
"""
            for name in sobol.first_order:
                html_content += f"""
        <tr>
            <td>{name}</td>
            <td>{sobol.first_order[name]:.4f}</td>
            <td>{sobol.total_order[name]:.4f}</td>
        </tr>
"""
            html_content += """
    </table>

    <h3>Second-Order (Interactions)</h3>
    <table>
        <tr><th>Interaction</th><th>S2</th></tr>
"""
            for (a, b), val in sobol.second_order.items():
                html_content += f"""
        <tr>
            <td>{a} × {b}</td>
            <td>{val:.4f}</td>
        </tr>
"""
            html_content += """
    </table>
"""

        # Include plots
        html_content += """
    <h2>Visualizations</h2>
"""

        for plot_file in output_dir.glob("*.png"):
            html_content += f"""
    <div class="plot">
        <h3>{plot_file.stem.replace("_", " ").title()}</h3>
        <img src="{plot_file.name}" alt="{plot_file.stem}">
    </div>
"""

        html_content += """
</body>
</html>
"""

        with Path(html_path).open("w") as f:
            f.write(html_content)

        return html_path

    def _generate_markdown_report(
        self,
        output_dir: Path,
        summary: dict,
        loo_results: list[LeaveOneOutResult],
        sobol: SobolIndices | None,
    ) -> Path:
        """Generate Markdown report."""
        md_path = output_dir / "ablation_report.md"

        with Path(md_path).open("w") as f:
            f.write("# Ablation Study Report\n\n")

            f.write("## Summary\n\n")
            f.write(f"- **Total Experiments**: {summary['n_experiments']}\n")
            f.write(f"- **Successful Runs**: {summary['n_successful']}\n")
            f.write(
                f"- **Baseline Accuracy**: {summary['baseline_accuracy']:.4f}\n"
                if summary["baseline_accuracy"]
                else "- **Baseline Accuracy**: N/A\n"
            )
            f.write(f"- **Dimensions Tested**: {', '.join(summary['dimensions'])}\n\n")

            f.write("## Critical Hyperparameters\n\n")
            f.write("Ranked by variance of mean outcomes:\n\n")
            for i, param in enumerate(summary["critical_params"], 1):
                f.write(f"{i}. **{param}**\n")
            f.write("\n")

            f.write("## Leave-One-Out Analysis\n\n")
            f.write(
                "| Dimension | Baseline Acc | Ablation Acc | Absolute Impact | Relative Impact |\n"
            )
            f.write(
                "|-----------|-------------|--------------|-----------------|----------------|\n"
            )
            for r in loo_results:
                f.write(
                    f"| {r.removed_dimension} | {r.baseline_accuracy:.4f} | "
                    f"{r.ablation_accuracy:.4f} | {r.impact:.4f} | {r.relative_impact:.2%} |\n"
                )
            f.write("\n")

            if sobol:
                f.write("## Sobol Sensitivity Indices\n\n")
                f.write("### First-Order (Main Effects)\n\n")
                f.write("| Parameter | S1 (First-Order) | ST (Total-Order) |\n")
                f.write("|-----------|------------------|------------------|\n")
                for name in sobol.first_order:
                    f.write(
                        f"| {name} | {sobol.first_order[name]:.4f} | {sobol.total_order[name]:.4f} |\n"
                    )
                f.write("\n")

                if sobol.second_order:
                    f.write("### Second-Order (Interactions)\n\n")
                    f.write("| Interaction | S2 |\n")
                    f.write("|-------------|----|\n")
                    for (a, b), val in sobol.second_order.items():
                        f.write(f"| {a} × {b} | {val:.4f} |\n")
                    f.write("\n")

            f.write("## Visualizations\n\n")
            for plot_file in sorted(output_dir.glob("*.png")):
                f.write(f"### {plot_file.stem.replace('_', ' ').title()}\n\n")
                f.write(f"![{plot_file.stem}]({plot_file.name})\n\n")

        return md_path


def create_ablation_report(
    base_cfg: RunConfig,
    dimensions: dict[str, list[object]],
    output_dir: str | Path = "results/ablation",
    parallel_workers: int = 4,
    run_sobol: bool = True,
    format: str = "all",
) -> dict[str, Path]:
    """
    Convenience function to run a full ablation study and generate report.

    Args:
        base_cfg: Base RunConfig
        dimensions: Dict of dimension names to lists of values
        output_dir: Output directory
        parallel_workers: Number of parallel workers
        run_sobol: Whether to compute Sobol indices (requires SALib)
        format: Report format ('html', 'markdown', 'json', 'all')

    Returns:
        Dictionary of report file paths.
    """
    study = AblationStudy(base_cfg, dimensions)
    study.run(parallel_workers=parallel_workers)

    if run_sobol:
        try:
            study.compute_sobol_indices(
                n_samples=500, parallel_workers=parallel_workers
            )
        except ImportError:
            pass

    return study.generate_report(output_dir=output_dir, format=format)
