"""Stability Report CLI (``comp stability``).

Generates stability reports for joint architecture evaluations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="comp stability",
        description="Generate stability reports for joint architecture coordinates",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Stability subcommand")

    # report
    report_parser = subparsers.add_parser(
        "report", help="Generate stability report for a run"
    )
    report_parser.add_argument(
        "--run-id", required=True, help="Run ID (campaign ID or episode ID)"
    )
    report_parser.add_argument("--db", help="SQLite database path")
    report_parser.add_argument(
        "--format", choices=["text", "json", "html"], default="text"
    )
    report_parser.add_argument("--output", help="Output file path")

    # compare
    compare_parser = subparsers.add_parser(
        "compare", help="Compare stability across coordinates"
    )
    compare_parser.add_argument(
        "--coordinates", nargs="+", required=True, help="6-D coordinates to compare"
    )
    compare_parser.add_argument("--db", help="SQLite database path")
    compare_parser.add_argument(
        "--format", choices=["text", "json", "html"], default="text"
    )

    # summary
    summary_parser = subparsers.add_parser(
        "summary", help="Summary stability statistics"
    )
    summary_parser.add_argument("--db", required=True, help="SQLite database path")
    summary_parser.add_argument("--branch", help="Branch name to filter")
    summary_parser.add_argument("--task", help="Task name to filter")

    return parser


def _generate_report(args) -> int:
    """Generate stability report for a run."""
    from computronium.core.campaign import CampaignStore

    db_path = args.db or "campaigns/campaign.db"
    store = CampaignStore(db_path)

    # Try to find the run as campaign ID first
    campaign = store.get_campaign(args.run_id)
    if campaign:
        episodes = store.get_episodes(args.run_id)
        run_type = "campaign"
    else:
        # Try to find as episode - search across all campaigns
        print(f"Run ID {args.run_id} not found as campaign")
        return 1

    if not episodes:
        print(f"No episodes found for {run_type} {args.run_id}")
        return 1

    # Compute stability statistics
    stability_data = []
    for ep in episodes:
        fr = ep.frontier_record
        stability_data.append({
            "iteration": ep.iteration,
            "coordinate": ep.coordinate,
            "task": ep.task_name,
            "rho_jacobian": fr.get("rho_jacobian"),
            "lyapunov_local": fr.get("lyapunov_local"),
            "settling_time": fr.get("settling_time"),
            "basin_stability": fr.get("basin_stability"),
            "stability_score": fr.get("stability_score", 0),
            "task_accuracy": fr.get("task_accuracy"),
            "task_loss": fr.get("task_loss"),
        })

    # Sort by stability score
    stability_data.sort(key=lambda x: x["stability_score"], reverse=True)

    if args.format == "json":
        output = json.dumps(
            {
                "run_id": args.run_id,
                "run_type": run_type,
                "n_episodes": len(episodes),
                "stability_data": stability_data,
            },
            indent=2,
        )
    elif args.format == "html":
        output = _generate_html_report(args.run_id, stability_data)
    else:
        output = _generate_text_report(args.run_id, stability_data)

    if args.output:
        Path(args.output).write_text(output)
        print(f"Report written to {args.output}")
    else:
        print(output)

    return 0


def _generate_text_report(run_id: str, data: list[dict]) -> str:
    """Generate text stability report."""
    lines = [
        f"Stability Report for {run_id}",
        "=" * 60,
        f"Total episodes: {len(data)}",
        "",
        "Top 10 by Stability Score:",
        "-" * 60,
        f"{'Rank':<5} {'Iter':<5} {'Coordinate':<40} {'Score':<8} {'ρ(J)':<8} {'λ':<8} {'Settle':<8} {'Basin':<8} {'Acc':<8}",
        "-" * 60,
    ]

    for i, d in enumerate(data[:10]):
        coord_short = (
            d["coordinate"][:38] + ".."
            if len(d["coordinate"]) > 40
            else d["coordinate"]
        )
        lines.append(
            f"{i + 1:<5} {d['iteration']:<5} {coord_short:<40} "
            f"{d['stability_score']:<8.3f} {d['rho_jacobian']:<8.3f} "
            f"{d['lyapunov_local']:<8.3f} {d['settling_time']:<8.1f} "
            f"{d['basin_stability']:<8.3f} {d['task_accuracy']:<8.4f}"
        )

    lines.extend([
        "",
        "Statistics:",
        "-" * 60,
    ])

    if data:
        import statistics

        scores = [d["stability_score"] for d in data]
        rho_vals = [d["rho_jacobian"] for d in data if d["rho_jacobian"] is not None]
        lyap_vals = [
            d["lyapunov_local"] for d in data if d["lyapunov_local"] is not None
        ]
        settle_vals = [
            d["settling_time"] for d in data if d["settling_time"] is not None
        ]
        basin_vals = [
            d["basin_stability"] for d in data if d["basin_stability"] is not None
        ]

        lines.append(
            f"  Stability Score: mean={statistics.mean(scores):.3f}, stdev={statistics.stdev(scores) if len(scores) > 1 else 0:.3f}"
        )
        if rho_vals:
            lines.append(
                f"  ρ(Jacobian): mean={statistics.mean(rho_vals):.3f}, min={min(rho_vals):.3f}, max={max(rho_vals):.3f}"
            )
        if lyap_vals:
            lines.append(
                f"  Lyapunov: mean={statistics.mean(lyap_vals):.3f}, min={min(lyap_vals):.3f}, max={max(lyap_vals):.3f}"
            )
        if settle_vals:
            lines.append(
                f"  Settling Time: mean={statistics.mean(settle_vals):.1f}, min={min(settle_vals):.1f}, max={max(settle_vals):.1f}"
            )
        if basin_vals:
            lines.append(
                f"  Basin Stability: mean={statistics.mean(basin_vals):.3f}, min={min(basin_vals):.3f}, max={max(basin_vals):.3f}"
            )

    return "\n".join(lines)


def _generate_html_report(run_id: str, data: list[dict]) -> str:
    """Generate HTML stability report."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Stability Report - {run_id}</title>
    <style>
        body {{ font-family: monospace; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .metric {{ font-weight: bold; color: #333; }}
    </style>
</head>
<body>
    <h1>Stability Report for {run_id}</h1>
    <p>Total episodes: {len(data)}</p>

    <h2>Episodes by Stability Score</h2>
    <table>
        <tr>
            <th>Rank</th>
            <th>Iteration</th>
            <th>Coordinate</th>
            <th>Stability Score</th>
            <th>ρ(Jacobian)</th>
            <th>Lyapunov</th>
            <th>Settling Time</th>
            <th>Basin Stability</th>
            <th>Task Accuracy</th>
        </tr>
"""

    for i, d in enumerate(data):
        html += f"""        <tr>
            <td>{i + 1}</td>
            <td>{d["iteration"]}</td>
            <td>{d["coordinate"]}</td>
            <td>{d["stability_score"]:.3f}</td>
            <td>{d["rho_jacobian"]:.3f}</td>
            <td>{d["lyapunov_local"]:.3f}</td>
            <td>{d["settling_time"]:.1f}</td>
            <td>{d["basin_stability"]:.3f}</td>
            <td>{d["task_accuracy"]:.4f}</td>
        </tr>
"""

    html += """    </table>
</body>
</html>"""

    return html


def _compare_stability(args) -> int:
    """Compare stability across coordinates."""
    from computronium.core.campaign import CampaignStore

    db_path = args.db or "campaigns/campaign.db"
    store = CampaignStore(db_path)

    all_data = []
    for coord in args.coordinates:
        episodes = store.get_episodes_by_coordinate(coord)
        if not episodes:
            print(f"No episodes found for coordinate: {coord}")
            continue

        for ep in episodes:
            fr = ep.frontier_record
            all_data.append({
                "coordinate": coord,
                "iteration": ep.iteration,
                "rho_jacobian": fr.get("rho_jacobian"),
                "lyapunov_local": fr.get("lyapunov_local"),
                "settling_time": fr.get("settling_time"),
                "basin_stability": fr.get("basin_stability"),
                "stability_score": fr.get("stability_score", 0),
                "task_accuracy": fr.get("task_accuracy"),
            })

    if not all_data:
        print("No data found for any coordinates")
        return 1

    if args.format == "json":
        output = json.dumps(all_data, indent=2)
    else:
        output = "Stability Comparison\n" + "=" * 60 + "\n"
        output += f"{'Coordinate':<50} {'Score':<8} {'ρ(J)':<8} {'λ':<8} {'Settle':<8} {'Basin':<8} {'Acc':<8}\n"
        output += "-" * 100 + "\n"
        for d in all_data:
            coord_short = (
                d["coordinate"][:48] + ".."
                if len(d["coordinate"]) > 50
                else d["coordinate"]
            )
            output += f"{coord_short:<50} {d['stability_score']:<8.3f} {d['rho_jacobian']:<8.3f} {d['lyapunov_local']:<8.3f} {d['settling_time']:<8.1f} {d['basin_stability']:<8.3f} {d['task_accuracy']:<8.4f}\n"

    if args.output:
        Path(args.output).write_text(output)
        print(f"Comparison written to {args.output}")
    else:
        print(output)

    return 0


def _summary_stability(args) -> int:
    """Summary stability statistics."""
    import statistics

    from computronium.core.campaign import CampaignStore

    db_path = args.db
    store = CampaignStore(db_path)

    campaigns = store.list_campaigns(args.branch)
    if not campaigns:
        print("No campaigns found")
        return 0

    all_episodes = []
    for camp in campaigns:
        eps = store.get_episodes(camp.campaign_id)
        if args.task:
            eps = [e for e in eps if e.task_name == args.task]
        all_episodes.extend(eps)

    if not all_episodes:
        print("No episodes found")
        return 0

    # Aggregate stability metrics
    scores = []
    rho_vals = []
    lyap_vals = []
    settle_vals = []
    basin_vals = []

    for ep in all_episodes:
        fr = ep.frontier_record
        scores.append(fr.get("stability_score", 0))
        if fr.get("rho_jacobian") is not None:
            rho_vals.append(fr["rho_jacobian"])
        if fr.get("lyapunov_local") is not None:
            lyap_vals.append(fr["lyapunov_local"])
        if fr.get("settling_time") is not None:
            settle_vals.append(fr["settling_time"])
        if fr.get("basin_stability") is not None:
            basin_vals.append(fr["basin_stability"])

    print(f"Stability Summary ({len(all_episodes)} episodes)")
    print("=" * 60)
    print(
        f"Stability Score: mean={statistics.mean(scores):.3f}, stdev={statistics.stdev(scores) if len(scores) > 1 else 0:.3f}, min={min(scores):.3f}, max={max(scores):.3f}"
    )
    if rho_vals:
        print(
            f"ρ(Jacobian): mean={statistics.mean(rho_vals):.3f}, min={min(rho_vals):.3f}, max={max(rho_vals):.3f}"
        )
    if lyap_vals:
        print(
            f"Lyapunov: mean={statistics.mean(lyap_vals):.3f}, min={min(lyap_vals):.3f}, max={max(lyap_vals):.3f}"
        )
    if settle_vals:
        print(
            f"Settling Time: mean={statistics.mean(settle_vals):.1f}, min={min(settle_vals):.1f}, max={max(settle_vals):.1f}"
        )
    if basin_vals:
        print(
            f"Basin Stability: mean={statistics.mean(basin_vals):.3f}, min={min(basin_vals):.3f}, max={max(basin_vals):.3f}"
        )

    # By plasticity primitive
    print("\nBy Plasticity Primitive:")
    print("-" * 60)
    by_plasticity: dict[str, list[float]] = {}
    for ep in all_episodes:
        fr = ep.frontier_record
        prim = fr.get("plasticity_primitive", "unknown")
        by_plasticity.setdefault(prim, []).append(fr.get("stability_score", 0))

    for prim, vals in sorted(by_plasticity.items()):
        print(f"  {prim:<20} n={len(vals):<4} mean={statistics.mean(vals):.3f}")

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point for ``comp stability``."""
    args = _build_parser().parse_args(argv)

    if not args.subcommand:
        _build_parser().print_help()
        return 1

    if args.subcommand == "report":
        return _generate_report(args)
    elif args.subcommand == "compare":
        return _compare_stability(args)
    elif args.subcommand == "summary":
        return _summary_stability(args)
    else:
        print(f"Unknown subcommand: {args.subcommand}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
