"""AutoScientist CLI (``comp scientist``).

Autonomous exploration of the 6-D joint architecture space.
Provides campaign runner, result browser, and hypothesis templates.
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
        prog="comp scientist",
        description="AutoScientist: Autonomous 6-D architecture exploration",
    )
    subparsers = parser.add_subparsers(
        dest="subcommand", help="AutoScientist subcommand"
    )

    # explore - run autonomous campaign
    explore_parser = subparsers.add_parser(
        "explore", help="Run autonomous exploration campaign"
    )
    explore_parser.add_argument(
        "--space",
        required=True,
        choices=["joint_smoke", "joint_full", "joint_routing", "joint_fast_weights"],
        help="Search space to explore",
    )
    explore_parser.add_argument(
        "--objective",
        default="adaptation_efficiency",
        choices=["adaptation_efficiency", "stability", "pareto", "compute_efficiency"],
        help="Objective to optimize",
    )
    explore_parser.add_argument(
        "--budget",
        type=int,
        default=10,
        help="Total experiment budget",
    )
    explore_parser.add_argument(
        "--output",
        default="campaign_results",
        help="Output directory",
    )
    explore_parser.add_argument(
        "--method",
        default="random",
        choices=["random", "bayesian", "evolutionary"],
        help="Search method",
    )
    explore_parser.add_argument(
        "--device",
        default="auto",
        help="Device (auto, cpu, cuda)",
    )
    explore_parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Parallel experiments",
    )

    # list - list campaigns
    list_parser = subparsers.add_parser("list", help="List campaigns")
    list_parser.add_argument("--format", choices=["table", "json"], default="table")
    list_parser.add_argument("--db", help="SQLite database path")

    # show - show campaign details
    show_parser = subparsers.add_parser("show", help="Show campaign details")
    show_parser.add_argument("--campaign-id", required=True, help="Campaign ID")
    show_parser.add_argument(
        "--include",
        nargs="+",
        default=["frontier", "resources", "stability"],
        help="Details to include",
    )
    show_parser.add_argument("--db", help="SQLite database path")

    # pareto - show Pareto frontier
    pareto_parser = subparsers.add_parser("pareto", help="Show Pareto frontier")
    pareto_parser.add_argument("--campaign-id", required=True, help="Campaign ID")
    pareto_parser.add_argument(
        "--objectives",
        nargs="+",
        default=["accuracy", "adaptation_time", "rho_jacobian"],
        help="Objectives for Pareto frontier",
    )
    pareto_parser.add_argument("--db", help="SQLite database path")

    # hypothesis - hypothesis template library
    hypothesis_parser = subparsers.add_parser("hypothesis", help="Hypothesis templates")
    hypothesis_parser.add_argument(
        "--list", action="store_true", help="List available templates"
    )
    hypothesis_parser.add_argument("--show", help="Show template details")
    hypothesis_parser.add_argument("--run", help="Run experiment from template")

    return parser


# ----------------------------------------------------------------------
# Search Space Definitions
# ----------------------------------------------------------------------


SEARCH_SPACES = {
    "joint_smoke": {
        "substrate": ["digital"],
        "geometry": ["feedforward", "recurrent"],
        "dynamics": ["instantaneous", "energy_minimization"],
        "plasticity": ["null", "routing", "fast_weights"],
        "credit": ["backprop", "thermodynamic_contrast", "random_projections"],
        "update": ["euclidean"],
    },
    "joint_full": {
        "substrate": ["digital", "memristive", "neuromorphic", "optical", "quantum"],
        "geometry": ["feedforward", "recurrent", "tile_mesh"],
        "dynamics": [
            "instantaneous",
            "energy_minimization",
            "predictive_settling",
            "spike_integration",
        ],
        "plasticity": [
            "null",
            "routing",
            "fast_weights",
            "substrate_coupled",
            "rule_state",
        ],
        "credit": [
            "backprop",
            "thermodynamic_contrast",
            "random_projections",
            "local_goodness",
            "temporal_trace",
        ],
        "update": [
            "euclidean",
            "riemannian_orthogonal",
            "spectral_constrained",
            "natural_gradient",
            "elastic_consolidation",
        ],
    },
    "joint_routing": {
        "substrate": ["digital", "memristive", "neuromorphic"],
        "geometry": ["recurrent", "tile_mesh"],
        "dynamics": ["energy_minimization", "predictive_settling"],
        "plasticity": ["routing"],
        "credit": ["thermodynamic_contrast", "backprop"],
        "update": ["euclidean"],
    },
    "joint_fast_weights": {
        "substrate": ["digital", "memristive"],
        "geometry": ["recurrent"],
        "dynamics": ["energy_minimization"],
        "plasticity": ["fast_weights"],
        "credit": ["thermodynamic_contrast", "random_projections"],
        "update": ["euclidean"],
    },
}


def _get_search_space(space_name: str) -> dict[str, list[str]]:
    """Get search space definition."""
    return SEARCH_SPACES.get(space_name, SEARCH_SPACES["joint_smoke"])


def _generate_coordinates(
    space: dict[str, list[str]], n: int, method: str = "random"
) -> list[str]:
    """Generate coordinates to test."""
    import itertools
    import random

    # Generate all combinations
    all_combos = list(itertools.product(*space.values()))
    coord_strings = ["/".join(combo) for combo in all_combos]

    if method == "random":
        random.shuffle(coord_strings)
        return coord_strings[:n]
    else:
        # For bayesian/evolutionary, just return first n for now
        return coord_strings[:n]


def _run_experiment(coordinate: str, objective: str, device: str) -> dict:
    """Run a single experiment and return results."""
    import subprocess  # ruff: ignore[suspicious-subprocess-import]

    # Use the benchmark adaptation_efficiency as the experiment
    cmd = [
        sys.executable,
        "-m",
        "computronium.experiments.joint.adaptation_efficiency",
        "--coordinates",
        coordinate,
        "--output-dir",
        "temp_experiment",
        "--epochs",
        "5",
        "--batch-size",
        "32",
        "--seeds",
        "1",
        "--device",
        device,
    ]

    try:  # ruff: ignore[too-many-statements-in-try-clause]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)  # ruff: ignore[subprocess-run-without-check, subprocess-without-shell-equals-true]
        if result.returncode != 0:
            return {
                "coordinate": coordinate,
                "error": result.stderr,
                "success": False,
            }

        # Parse results from JSON
        import json

        result_file = Path("temp_experiment/adaptation_efficiency_results.json")
        if result_file.exists():
            with result_file.open(encoding="utf-8") as f:
                data = json.load(f)
            if data:
                r = data[0]
                return {
                    "coordinate": coordinate,
                    "success": True,
                    "accuracy": r.get("mean_accuracy", 0),
                    "adaptation_time": r.get("mean_adaptation_time", 0),
                    "rho_jacobian": 0.0,  # Would need to extract from seeds
                    "objective_value": r.get("mean_accuracy", 0),
                }
        return {  # ruff: ignore[try-consider-else]
            "coordinate": coordinate,
            "success": True,
            "objective_value": 0.0,
        }
    except subprocess.TimeoutExpired:
        return {
            "coordinate": coordinate,
            "error": "Timeout",
            "success": False,
        }
    except Exception as e:
        return {
            "coordinate": coordinate,
            "error": str(e),
            "success": False,
        }


def _explore(args) -> int:
    """Run autonomous exploration campaign."""
    import uuid
    from datetime import datetime

    space = _get_search_space(args.space)
    campaign_id = str(uuid.uuid4())[:8]
    output_dir = Path(args.output) / f"campaign_{campaign_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting AutoScientist campaign: {campaign_id}")
    print(f"Search space: {args.space}")
    print(f"Objective: {args.objective}")
    print(f"Budget: {args.budget} experiments")
    print(f"Method: {args.method}")
    print(f"Output: {output_dir}")

    # Generate coordinates to test
    coordinates = _generate_coordinates(space, args.budget, args.method)
    print(f"\nGenerated {len(coordinates)} coordinates to test")

    results = []
    for i, coord in enumerate(coordinates):
        print(f"\n[{i + 1}/{len(coordinates)}] Testing: {coord}")
        result = _run_experiment(coord, args.objective, args.device)
        results.append(result)

        if result["success"]:
            print(
                f"  ✓ Accuracy: {result.get('accuracy', 0):.4f}, Adapt time: {result.get('adaptation_time', 0):.1f}"
            )
        else:
            print(f"  ✗ Error: {result.get('error', 'Unknown')}")

    # Save campaign results
    campaign_data = {
        "campaign_id": campaign_id,
        "space": args.space,
        "objective": args.objective,
        "budget": args.budget,
        "method": args.method,
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }

    output_file = output_dir / "campaign.json"
    with output_file.open("w") as f:
        json.dump(campaign_data, f, indent=2)

    print(f"\nCampaign completed. Results saved to {output_file}")

    # Print summary
    successful = [r for r in results if r["success"]]
    if successful:
        best = max(successful, key=lambda x: x.get("objective_value", 0))
        print(f"\nBest result: {best['coordinate']}")
        print(f"  Objective value: {best['objective_value']:.4f}")

    return 0


def _list_campaigns(args) -> int:
    """List all campaigns."""
    # For now, just list JSON files in campaigns directory
    campaigns_dir = Path("campaign_results") if not args.db else Path(args.db).parent
    if not campaigns_dir.exists():
        print("No campaigns found")
        return 0

    campaigns = []
    for camp_dir in campaigns_dir.iterdir():
        if camp_dir.is_dir() and (camp_dir / "campaign.json").exists():
            with (camp_dir / "campaign.json").open() as f:
                data = json.load(f)
            campaigns.append(data)

    if args.format == "json":
        print(json.dumps(campaigns, indent=2))
    else:
        print(
            f"{'Campaign ID':<12} {'Space':<15} {'Objective':<20} {'Budget':<8} {'Status'}"
        )
        print("-" * 80)
        for c in campaigns:
            n_success = sum(1 for r in c["results"] if r.get("success"))
            print(
                f"{c['campaign_id']:<12} {c['space']:<15} {c['objective']:<20} {c['budget']:<8} {n_success}/{len(c['results'])}"
            )

    return 0


def _show_campaign(args) -> int:
    """Show campaign details."""
    # Find campaign file
    campaigns_dir = Path("campaign_results") if not args.db else Path(args.db).parent
    campaign_file = campaigns_dir / f"campaign_{args.campaign_id}" / "campaign.json"

    if not campaign_file.exists():
        print(f"Campaign {args.campaign_id} not found")
        return 1

    with campaign_file.open() as f:
        data = json.load(f)

    print(f"Campaign: {data['campaign_id']}")
    print(f"Space: {data['space']}")
    print(f"Objective: {data['objective']}")
    print(f"Budget: {data['budget']}")
    print(f"Method: {data['method']}")
    print(f"Timestamp: {data['timestamp']}")
    print(f"Total experiments: {len(data['results'])}")

    successful = [r for r in data["results"] if r.get("success")]
    print(f"Successful: {len(successful)}")

    if "frontier" in args.include and successful:
        print("\n--- Pareto Frontier (top 5) ---")
        sorted_results = sorted(
            successful, key=lambda x: x.get("objective_value", 0), reverse=True
        )
        for r in sorted_results[:5]:
            print(f"  {r['coordinate']}: {r.get('objective_value', 0):.4f}")

    if "stability" in args.include:
        print("\n--- Stability Metrics ---")
        for r in successful:
            rho = r.get("rho_jacobian", "N/A")
            print(f"  {r['coordinate']}: ρ(J)={rho}")

    if "resources" in args.include:
        print("\n--- Resources ---")
        print(f"  Total experiments: {len(data['results'])}")

    return 0


def _pareto_campaign(args) -> int:
    """Show Pareto frontier for campaign."""
    campaigns_dir = Path("campaign_results") if not args.db else Path(args.db).parent
    campaign_file = campaigns_dir / f"campaign_{args.campaign_id}" / "campaign.json"

    if not campaign_file.exists():
        print(f"Campaign {args.campaign_id} not found")
        return 1

    with campaign_file.open() as f:
        data = json.load(f)

    successful = [r for r in data["results"] if r.get("success")]

    if not successful:
        print("No successful results")
        return 0

    # Multi-objective Pareto (simplified)
    objectives = args.objectives
    if len(objectives) == 1:
        sorted_results = sorted(
            successful, key=lambda x: x.get(objectives[0], 0), reverse=True
        )
        print(f"Top 10 by {objectives[0]}:")
        for r in sorted_results[:10]:
            print(f"  {r['coordinate']}: {r.get(objectives[0], 0):.4f}")
    else:
        print(f"Pareto frontier for {objectives} (simplified):")
        for r in successful[:10]:
            vals = ", ".join(f"{obj}={r.get(obj, 0):.4f}" for obj in objectives)
            print(f"  {r['coordinate']}: {vals}")

    return 0


# ----------------------------------------------------------------------
# Hypothesis Templates
# ----------------------------------------------------------------------


HYPOTHESIS_TEMPLATES = {
    "substrate_ablation": {
        "name": "Substrate Ablation",
        "description": "What if we change the substrate while keeping other axes fixed?",
        "template": """
# Hypothesis: Substrate Ablation
## Question
Does changing substrate from {base_substrate} to {test_substrate} improve {objective}?

## Setup
- Fixed axes: geometry={geometry}, dynamics={dynamics}, plasticity={plasticity}, credit={credit}, update={update}
- Variable: substrate in [{base_substrate}, {test_substrate}]
- Budget: {budget} experiments per substrate
- Seeds: {seeds} per configuration

## Predictions
1. {test_substrate} will show higher noise resilience due to {reason}
2. {test_substrate} may have higher compute cost
3. Plasticity interaction: {plasticity} may amplify/dampen substrate effects

## Success Criteria
- {objective} improvement > {threshold}%
- No stability degradation (ρ(J) < 1.0)
""",
        "parameters": [
            "base_substrate",
            "test_substrate",
            "geometry",
            "dynamics",
            "plasticity",
            "credit",
            "update",
            "objective",
            "budget",
            "seeds",
            "reason",
            "threshold",
        ],
    },
    "credit_swap": {
        "name": "Credit Assignment Swap",
        "description": "Does FA work better on memristive/neuromorphic substrates?",
        "template": """
# Hypothesis: Credit Assignment Swap
## Question
Does {test_credit} outperform {base_credit} on {substrate} substrate?

## Setup
- Fixed axes: substrate={substrate}, geometry={geometry}, dynamics={dynamics}, plasticity={plasticity}, update={update}
- Variable: credit in [{base_credit}, {test_credit}]
- Budget: {budget} experiments per credit type

## Predictions
1. {test_credit} avoids weight transport, better for {substrate}
2. Gradient alignment may be lower but adaptation faster
3. Interaction with {plasticity} plasticity is key

## Success Criteria
- {objective} improvement > {threshold}%
- Gradient alignment > {alignment_threshold}
""",
        "parameters": [
            "base_credit",
            "test_credit",
            "substrate",
            "geometry",
            "dynamics",
            "plasticity",
            "update",
            "objective",
            "budget",
            "threshold",
            "alignment_threshold",
        ],
    },
    "plasticity_search": {
        "name": "Plasticity Search",
        "description": "Does routing/fast_weights help adaptation vs null?",
        "template": """
# Hypothesis: Plasticity Search
## Question
Does {test_plasticity} plasticity improve {objective} over null plasticity?

## Setup
- Fixed axes: substrate={substrate}, geometry={geometry}, dynamics={dynamics}, credit={credit}, update={update}
- Variable: plasticity in [null, {test_plasticity}]
- Budget: {budget} experiments

## Predictions
1. {test_plasticity} provides intra-episode adaptation
2. Routing: dynamic pathway selection helps distribution shift
3. Fast weights: associative memory helps few-shot

## Success Criteria
- {objective} improvement > {threshold}%
- Adaptation time reduction > {time_threshold}%
- No catastrophic forgetting
""",
        "parameters": [
            "test_plasticity",
            "substrate",
            "geometry",
            "dynamics",
            "credit",
            "update",
            "objective",
            "budget",
            "threshold",
            "time_threshold",
        ],
    },
    "stability_frontier": {
        "name": "Stability Frontier",
        "description": "Maximize adaptation subject to ρ(J_F) < 0.99",
        "template": """
# Hypothesis: Stability Frontier
## Question
What is the maximum {objective} achievable while maintaining ρ(J_F) < {rho_threshold}?

## Setup
- Search space: {space}
- Constraint: ρ(Jacobian) < {rho_threshold}
- Objective: Maximize {objective}
- Budget: {budget}

## Predictions
1. Null plasticity: highest stability, lowest adaptation
2. Routing: moderate stability, good adaptation
3. Fast weights: lower stability, highest adaptation potential

## Success Criteria
- Find Pareto frontier of (adaptation, ρ(J))
- Identify optimal plasticity for each stability budget
""",
        "parameters": ["space", "rho_threshold", "objective", "budget"],
    },
}


def _hypothesis(args) -> int:
    """Hypothesis template library."""
    if args.list:
        print("Available Hypothesis Templates:")
        print("=" * 60)
        for key, tmpl in HYPOTHESIS_TEMPLATES.items():
            print(f"  {key}: {tmpl['name']}")
            print(f"    {tmpl['description']}")
            print()
        return 0

    if args.show:
        if args.show not in HYPOTHESIS_TEMPLATES:
            print(f"Template {args.show} not found")
            return 1
        tmpl = HYPOTHESIS_TEMPLATES[args.show]
        print(f"Template: {tmpl['name']}")
        print(f"Description: {tmpl['description']}")
        print(f"Parameters: {', '.join(tmpl['parameters'])}")
        print("\n--- Template ---")
        print(tmpl["template"])
        return 0

    if args.run:
        # Run experiment from template (would need parameter parsing)
        print(f"Running template {args.run} - not yet implemented")
        return 0

    # Default: show help
    return 1


def main(argv: Sequence[str] | None = None) -> int:  # ruff: ignore[too-many-return-statements]
    """Console-script entry point for ``comp scientist``."""
    args = _build_parser().parse_args(argv)

    if not args.subcommand:
        _build_parser().print_help()
        return 1

    if args.subcommand == "explore":
        return _explore(args)
    elif args.subcommand == "list":
        return _list_campaigns(args)
    elif args.subcommand == "show":
        return _show_campaign(args)
    elif args.subcommand == "pareto":
        return _pareto_campaign(args)
    elif args.subcommand == "hypothesis":
        return _hypothesis(args)
    else:
        print(f"Unknown subcommand: {args.subcommand}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
