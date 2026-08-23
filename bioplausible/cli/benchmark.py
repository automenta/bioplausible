"""Joint Benchmark CLI (``biopl benchmark``).

Runs benchmark suites for 6-D joint architecture coordinates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="biopl benchmark",
        description="Run joint architecture benchmark suites",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Benchmark subcommand")

    # run
    run_parser = subparsers.add_parser("run", help="Run a benchmark suite")
    run_parser.add_argument("--suite", required=True,
        choices=[
            "adaptation_efficiency",
            "compute_efficiency",
            "structural_robustness",
            "algorithm_migration",
            "z3_fixed_weights",
        ],
        help="Benchmark suite to run"
    )
    run_parser.add_argument("--coordinates", nargs="+", help="Specific 6-D coordinates to test (default: all from suite)")
    run_parser.add_argument("--output-dir", default="benchmark_results", help="Output directory")
    run_parser.add_argument("--epochs", type=int, default=10, help="Epochs per evaluation")
    run_parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    run_parser.add_argument("--device", default="auto", help="Device (auto, cpu, cuda)")
    run_parser.add_argument("--seeds", type=int, default=3, help="Number of seeds per coordinate")
    run_parser.add_argument("--quick", action="store_true", help="Quick mode (3 epochs, 1 seed)")

    # list
    list_parser = subparsers.add_parser("list", help="List available benchmark suites")

    # report
    report_parser = subparsers.add_parser("report", help="Generate benchmark report")
    report_parser.add_argument("--results-dir", required=True, help="Benchmark results directory")
    report_parser.add_argument("--format", choices=["text", "json", "html"], default="text")
    report_parser.add_argument("--output", help="Output file path")

    return parser


def _get_suite_coordinates(suite: str) -> list[str]:
    """Get coordinates for a benchmark suite."""
    suites = {
        "adaptation_efficiency": [
            # Null plasticity (baseline)
            "digital/recurrent/energy_minimization/null/thermodynamic_contrast/euclidean",
            # Routing plasticity
            "digital/recurrent/energy_minimization/routing/thermodynamic_contrast/euclidean",
            # Fast weights plasticity
            "digital/recurrent/energy_minimization/fast_weights/thermodynamic_contrast/euclidean",
            # Substrate coupled
            "digital/recurrent/energy_minimization/substrate_coupled/thermodynamic_contrast/euclidean",
        ],
        "compute_efficiency": [
            # Dense baseline
            "digital/feedforward/instantaneous/null/thermodynamic_contrast/euclidean",
            # Routing with sparse activation
            "digital/feedforward/instantaneous/routing/thermodynamic_contrast/euclidean",
            # Sparse substrate
            "sparse/feedforward/instantaneous/null/thermodynamic_contrast/euclidean",
            # Ternary substrate
            "ternary/feedforward/instantaneous/null/thermodynamic_contrast/euclidean",
        ],
        "structural_robustness": [
            # Standard recurrent
            "digital/recurrent/energy_minimization/null/thermodynamic_contrast/euclidean",
            # Routing (can reroute around damage)
            "digital/recurrent/energy_minimization/routing/thermodynamic_contrast/euclidean",
            # Substrate coupled (memristive noise resilience)
            "memristive/recurrent/energy_minimization/substrate_coupled/thermodynamic_contrast/euclidean",
            # Neuromorphic (spike-based resilience)
            "neuromorphic/recurrent/spike_integration/null/thermodynamic_contrast/euclidean",
        ],
        "algorithm_migration": [
            # Task A0 -> A1 with routing
            "digital/recurrent/energy_minimization/routing/thermodynamic_contrast/euclidean",
            # Task A0 -> A1 with fast weights
            "digital/recurrent/energy_minimization/fast_weights/thermodynamic_contrast/euclidean",
            # Task A0 -> A1 with rule state
            "digital/recurrent/energy_minimization/rule_state/thermodynamic_contrast/euclidean",
        ],
        "z3_fixed_weights": [
            # Frozen theta, rule state plasticity only
            "digital/recurrent/energy_minimization/rule_state/thermodynamic_contrast/euclidean",
        ],
    }
    return suites.get(suite, [])


def _run_benchmark(args) -> int:
    """Run a benchmark suite by delegating to experiment modules."""
    import subprocess
    import sys
    from pathlib import Path

    coordinates = args.coordinates or _get_suite_coordinates(args.suite)
    if not coordinates:
        print(f"Unknown suite: {args.suite}")
        return 1

    output_dir = Path(args.output_dir) / args.suite
    output_dir.mkdir(parents=True, exist_ok=True)

    # Map suite to experiment module and argument mappings
    suite_configs = {
        "adaptation_efficiency": {
            "module": "bioplausible.experiments.joint.adaptation_efficiency",
            "epoch_arg": "--epochs",
        },
        "compute_efficiency": {
            "module": "bioplausible.experiments.joint.compute_efficiency",
            "epoch_arg": "--epochs",
        },
        "structural_robustness": {
            "module": "bioplausible.experiments.joint.structural_robustness",
            "epoch_arg": "--epochs",
        },
        "algorithm_migration": {
            "module": "bioplausible.experiments.joint.algorithm_migration",
            "epoch_arg": "--epochs-a0",
            "extra_args": ["--epochs-a1", str(args.epochs)],
        },
        "z3_fixed_weights": {
            "module": "bioplausible.experiments.joint.z3_fixed_weights",
            "epoch_arg": "--meta-train-epochs",
            "extra_args": ["--eval-epochs", str(args.epochs)],
        },
    }

    config = suite_configs.get(args.suite)
    if not config:
        print(f"Unknown suite: {args.suite}")
        return 1

    module = config["module"]
    epoch_arg = config.get("epoch_arg", "--epochs")
    extra_args = config.get("extra_args", [])

    # Build command for the experiment module
    cmd = [
        sys.executable, "-m", module,
        "--coordinates", *coordinates,
        "--output-dir", str(output_dir),
        epoch_arg, str(args.epochs),
        *extra_args,
        "--batch-size", str(args.batch_size),
        "--seeds", str(args.seeds),
        "--device", args.device,
    ]
    if args.quick:
        cmd.append("--quick")

    print(f"Running benchmark suite: {args.suite}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Coordinates: {len(coordinates)}")
    print(f"Device: {args.device}")

    # Run the experiment module
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"Benchmark suite {args.suite} failed with return code {result.returncode}")
        return result.returncode

    print(f"\nResults saved to {output_dir}")
    return 0


def _list_suites(_args) -> int:
    """List available benchmark suites."""
    suites = {
        "adaptation_efficiency": "Does plasticity adapt faster than Null? (Switching input distribution)",
        "compute_efficiency": "Does routing reduce effective ops? (Mixture-of-experts synthetic)",
        "structural_robustness": "Can system recover after damage? (Zeroed weights, removed nodes)",
        "algorithm_migration": "Can ψ switch strategy without θ update? (Cumulative sum -> Last symbol)",
        "z3_fixed_weights": "Can frozen θ solve multiple tasks via ψ? (Parity, Last-symbol, Threshold)",
    }

    print("Available Benchmark Suites:")
    print("=" * 80)
    for name, desc in suites.items():
        print(f"  {name:<25} {desc}")

    return 0


def _generate_report(args) -> int:
    """Generate benchmark report from results."""
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return 1

    result_files = list(results_dir.glob("*_results.json"))
    if not result_files:
        print(f"No result files found in {results_dir}")
        return 1

    all_results = {}
    for rf in result_files:
        with rf.open() as f:
            data = json.load(f)
        all_results[data["suite"]] = data

    if args.format == "json":
        output = json.dumps(all_results, indent=2)
    elif args.format == "html":
        output = _generate_html_report(all_results)
    else:
        output = _generate_text_report(all_results)

    if args.output:
        Path(args.output).write_text(output)
        print(f"Report written to {args.output}")
    else:
        print(output)

    return 0


def _generate_text_report(all_results: dict) -> str:
    """Generate text benchmark report."""
    lines = ["Joint Architecture Benchmark Report", "=" * 80, ""]

    for suite_name, data in all_results.items():
        lines.append(f"Suite: {suite_name}")
        lines.append("-" * 40)
        results = data.get("results", [])

        if not results:
            lines.append("  No results")
            continue

        lines.append(f"{'Coordinate':<50} {'Mean Acc':<10} {'Std':<8} {'Plasticity':<15} {'ρ(J)':<8} {'Basin':<8}")
        lines.append("-" * 100)

        for r in results:
            coord_short = r["coordinate"][:48] + ".." if len(r["coordinate"]) > 50 else r["coordinate"]
            prim = r["coordinate"].split("/")[3]
            mean_acc = r.get("mean_accuracy", 0)
            std_acc = r.get("std_accuracy", 0)
            # Get average stability metrics
            seeds = r.get("seeds", [])
            avg_rho = sum(s.get("rho_jacobian", 0) for s in seeds) / len(seeds) if seeds else 0
            avg_basin = sum(s.get("basin_stability", 0) for s in seeds) / len(seeds) if seeds else 0
            lines.append(f"{coord_short:<50} {mean_acc:<10.4f} {std_acc:<8.4f} {prim:<15} {avg_rho:<8.3f} {avg_basin:<8.3f}")

        lines.append("")

    return "\n".join(lines)


def _generate_html_report(all_results: dict) -> str:
    """Generate HTML benchmark report."""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Joint Architecture Benchmark Report</title>
    <style>
        body { font-family: monospace; margin: 20px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 30px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        h1 { color: #333; }
        h2 { color: #666; border-bottom: 1px solid #ddd; padding-bottom: 5px; }
    </style>
</head>
<body>
    <h1>Joint Architecture Benchmark Report</h1>
"""

    for suite_name, data in all_results.items():
        html += f"<h2>Suite: {suite_name}</h2>"
        results = data.get("results", [])

        if not results:
            html += "<p>No results</p>"
            continue

        html += """<table>
        <tr>
            <th>Coordinate</th>
            <th>Mean Accuracy</th>
            <th>Std Accuracy</th>
            <th>Plasticity</th>
            <th>ρ(Jacobian)</th>
            <th>Basin Stability</th>
        </tr>
"""

        for r in results:
            prim = r["coordinate"].split("/")[3]
            mean_acc = r.get("mean_accuracy", 0)
            std_acc = r.get("std_accuracy", 0)
            seeds = r.get("seeds", [])
            avg_rho = sum(s.get("rho_jacobian", 0) for s in seeds) / len(seeds) if seeds else 0
            avg_basin = sum(s.get("basin_stability", 0) for s in seeds) / len(seeds) if seeds else 0
            html += f"""        <tr>
            <td>{r['coordinate']}</td>
            <td>{mean_acc:.4f}</td>
            <td>{std_acc:.4f}</td>
            <td>{prim}</td>
            <td>{avg_rho:.3f}</td>
            <td>{avg_basin:.3f}</td>
        </tr>
"""

        html += """    </table>
"""

    html += """</body>
</html>"""

    return html


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point for ``biopl benchmark``."""
    args = _build_parser().parse_args(argv)

    if not args.subcommand:
        _build_parser().print_help()
        return 1

    if args.subcommand == "run":
        return _run_benchmark(args)
    elif args.subcommand == "list":
        return _list_suites(args)
    elif args.subcommand == "report":
        return _generate_report(args)
    else:
        print(f"Unknown subcommand: {args.subcommand}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())