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
            # Frozen theta, routing plasticity
            "digital/recurrent/energy_minimization/routing/thermodynamic_contrast/euclidean",
            # Frozen theta, fast weights
            "digital/recurrent/energy_minimization/fast_weights/thermodynamic_contrast/euclidean",
            # Frozen theta, rule state
            "digital/recurrent/energy_minimization/rule_state/thermodynamic_contrast/euclidean",
        ],
    }
    return suites.get(suite, [])


def _run_benchmark(args) -> int:
    """Run a benchmark suite."""
    import torch
    import random
    import time
    from bioplausible.core.campaign import CampaignStore
    from bioplausible.core.campaign.frontier_record import FrontierRecord
    from bioplausible.core.campaign.resource_vector import ResourceUsage

    coordinates = args.coordinates or _get_suite_coordinates(args.suite)
    if not coordinates:
        print(f"Unknown suite: {args.suite}")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device

    print(f"Running benchmark suite: {args.suite}")
    print(f"Coordinates: {len(coordinates)}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, Seeds: {args.seeds}")

    results = []

    for coord_idx, coordinate in enumerate(coordinates):
        print(f"\n[{coord_idx + 1}/{len(coordinates)}] {coordinate}")

        coord_results = {"coordinate": coordinate, "seeds": []}

        for seed in range(args.seeds):
            if args.quick and seed > 0:
                break

            torch.manual_seed(seed)
            random.seed(seed)

            start_time = time.time()

            # TODO: Actually run the joint system evaluation
            # For now, create mock results
            task_name = "mnist"  # Default task

            # Mock evaluation
            time.sleep(0.1)  # Simulate computation
            elapsed = time.time() - start_time

            # Create mock frontier record
            frontier_record = FrontierRecord(
                coordinate=coordinate,
                task_name=task_name,
                task_loss=random.uniform(0.1, 0.5),
                task_accuracy=random.uniform(0.7, 0.98),
                adaptation_time=random.randint(10, 50),
                rho_jacobian=random.uniform(0.5, 1.2),
                lyapunov_local=random.uniform(-0.3, 0.3),
                settling_time=random.uniform(5, 30),
                basin_stability=random.uniform(0.4, 0.95),
                resources=ResourceUsage(
                    compute=random.uniform(1e10, 1e12),
                    memory=random.uniform(100, 2000),
                    energy=random.uniform(1, 100),
                    latency=elapsed,
                    plastic_state_capacity=random.uniform(1e6, 1e8),
                ),
                plasticity_primitive=coordinate.split("/")[3],
                campaign_id=f"benchmark_{args.suite}",
                episode_index=coord_idx,
            )

            coord_results["seeds"].append({
                "seed": seed,
                "task_accuracy": frontier_record.task_accuracy,
                "task_loss": frontier_record.task_loss,
                "adaptation_time": frontier_record.adaptation_time,
                "rho_jacobian": frontier_record.rho_jacobian,
                "lyapunov_local": frontier_record.lyapunov_local,
                "settling_time": frontier_record.settling_time,
                "basin_stability": frontier_record.basin_stability,
                "resources": frontier_record.resources.to_dict(),
                "elapsed_time": elapsed,
            })

            print(f"  Seed {seed}: acc={frontier_record.task_accuracy:.4f}, "
                  f"rho={frontier_record.rho_jacobian:.3f}, "
                  f"time={elapsed:.1f}s")

        # Aggregate across seeds
        if coord_results["seeds"]:
            accuracies = [s["task_accuracy"] for s in coord_results["seeds"]]
            coord_results["mean_accuracy"] = sum(accuracies) / len(accuracies)
            coord_results["std_accuracy"] = (sum((a - coord_results["mean_accuracy"])**2 for a in accuracies) / len(accuracies))**0.5 if len(accuracies) > 1 else 0.0

        results.append(coord_results)

    # Save results
    results_file = output_dir / f"{args.suite}_results.json"
    with results_file.open("w") as f:
        json.dump({
            "suite": args.suite,
            "config": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "device": device,
                "seeds": args.seeds,
                "quick": args.quick,
            },
            "results": results,
        }, f, indent=2)

    print(f"\nResults saved to {results_file}")

    # Print summary
    print("\n" + "=" * 80)
    print(f"Benchmark Summary: {args.suite}")
    print("=" * 80)
    print(f"{'Coordinate':<50} {'Mean Acc':<10} {'Std Acc':<10} {'Plasticity'}")
    print("-" * 80)
    for r in results:
        coord_short = r["coordinate"][:48] + ".." if len(r["coordinate"]) > 50 else r["coordinate"]
        prim = r["coordinate"].split("/")[3]
        print(f"{coord_short:<50} {r.get('mean_accuracy', 0):<10.4f} {r.get('std_accuracy', 0):<10.4f} {prim}")

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