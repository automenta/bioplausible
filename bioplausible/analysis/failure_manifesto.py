from pathlib import Path

import pandas as pd

from bioplausible.execution._state import FailureTracker

__all__ = [
    "FailureManifestoGenerator",
    "main",
]


class FailureManifestoGenerator:
    """
    Auto-generates reports/failure_manifesto.md from experiment DB.
    """

    def __init__(self, db_path: str):
        self.tracker = FailureTracker(db_path)

    def generate(
        self,
        output_path: str = "reports/failure_manifesto.md",
        model: str | None = None,
    ):
        """
        Extracts failures from DB and groups them by algorithm and FailureCategory.
        Outputs a markdown manifesto report.
        """
        _ = self.tracker.get_failure_stats()
        recent_failures = self.tracker.get_recent_failures(limit=1000)

        # Build DataFrame for easier cross-tabulation
        fail_data = []
        for r in recent_failures:
            if model is not None and r.model_name != model:
                continue
            fail_data.append({
                "model": r.model_name,
                "task": r.task_name,
                "type": r.failure_type,
                "epoch": r.failure_epoch,
            })

        df = pd.DataFrame(fail_data)

        Path(output_path).parent.mkdir(exist_ok=True, parents=True)

        with Path(output_path).open("w") as f:
            f.write("# Failure Modes Manifesto\n\n")
            f.write(
                "This document tracks the explicit failure modes encountered "
                "across different bioplausible algorithms.\n\n"
            )
            if model is not None:
                f.write(f"### Scope: `{model}`\n\n")
            if df.empty:
                if model is not None:
                    f.write(f"No failures logged for `{model}` yet.\n")
                else:
                    f.write("No failures logged yet.\n")
                return output_path

            _write_distribution(f, df)
            _write_crosstab(f, df)
            _write_diagnostics(f, self.tracker)

        return output_path


def _write_distribution(f, df: pd.DataFrame) -> None:
    """Write the overall failure-type distribution table."""
    f.write("## Overall Failure Distribution\n\n")
    type_counts = df["type"].value_counts()
    f.write("| Failure Type | Count |\n")
    f.write("|--------------|-------|\n")
    for t, c in type_counts.items():
        f.write(f"| `{t}` | {c} |\n")
    f.write("\n")


def _write_crosstab(f, df: pd.DataFrame) -> None:
    """Write failures-by-model-and-type as a markdown table."""
    f.write("## Failures by Model and Type\n\n")
    cross_tab = pd.crosstab(df["model"], df["type"])
    cols = ["Model"] + list(cross_tab.columns)
    f.write("| " + " | ".join(cols) + " |\n")
    f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
    for index, row in cross_tab.iterrows():
        row_vals = [str(index)] + [str(v) for v in row.values]
        f.write("| " + " | ".join(row_vals) + " |\n")
    f.write("\n")


def _write_diagnostics(f, tracker: FailureTracker) -> None:
    """Write the advanced failure-pattern diagnostics section."""
    f.write("## Advanced Diagnostics\n\n")
    analysis = tracker.analyze_failure_patterns()
    if not analysis.get("recommendations"):
        f.write(
            "No critical failure patterns detected requiring immediate intervention.\n"
        )
        return
    for rec in analysis["recommendations"]:
        sev = rec.get("severity", "info")
        f.write(
            "### [Severity: {}] {}\n".format(
                sev.upper(), rec.get("issue", "Unknown Issue")
            )
        )
        f.write(f"- **Recommendation**: {rec.get('suggestion')}\n")
        if "affected_models" in rec:
            f.write(
                "- **Affected Models**: {}\n".format(", ".join(rec["affected_models"]))
            )
        if "details" in rec:
            f.write(f"- **Details**: {rec['details']}\n")
        f.write("\n")


def main(argv: list[str] | None = None) -> int:
    """``biopl-failure-manifesto`` entry point (Sprint 2.4).

    Generates a markdown failure manifesto from the experiment failure DB,
    optionally scoped to a single model.
    """
    import argparse
    import logging

    parser = argparse.ArgumentParser(
        description="Generate a markdown failure-mode manifesto from the DB."
    )
    parser.add_argument("--db", default="bioplausible.db", help="Path to the DB.")
    parser.add_argument(
        "--model",
        default=None,
        help="Only include failures for this model (e.g. eqprop_mlp).",
    )
    parser.add_argument(
        "--output",
        default="reports/failure_manifesto.md",
        help="Output markdown path.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    generator = FailureManifestoGenerator(args.db)
    out = generator.generate(args.output, model=args.model)
    logging.info("wrote failure manifesto to %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
