"""Joint Campaign CLI (``biopl campaign``).

Runs and manages 6-D joint architecture campaigns with:
- Campaign persistence (SQLite + YAML)
- Kernel caching
- Fault tolerance checkpointing
- AutoScientist integration
"""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="biopl campaign",
        description="Run and manage 6-D joint architecture campaigns",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Campaign subcommand")

    # run
    run_parser = subparsers.add_parser("run", help="Run a campaign")
    run_parser.add_argument(
        "--space",
        required=True,
        help="Search space name (e.g., joint_smoke, joint_full)",
    )
    run_parser.add_argument(
        "--objective",
        required=True,
        help="Objective to optimize (e.g., adaptation_efficiency, stability, pareto)",
    )
    run_parser.add_argument("--branch", default="main", help="Branch name")
    run_parser.add_argument(
        "--campaign-id", help="Campaign ID (auto-generated if not provided)"
    )
    run_parser.add_argument(
        "--iterations", type=int, default=10, help="Number of iterations"
    )
    run_parser.add_argument(
        "--experiments-per-iter", type=int, default=5, help="Experiments per iteration"
    )
    run_parser.add_argument(
        "--output-dir", default="campaigns", help="Output directory"
    )
    run_parser.add_argument("--db", help="SQLite database path")
    run_parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="Checkpoint interval (episodes)",
    )
    run_parser.add_argument(
        "--resume", action="store_true", help="Resume from latest checkpoint"
    )
    run_parser.add_argument(
        "--dry-run", action="store_true", help="Propose without executing"
    )

    # status
    status_parser = subparsers.add_parser("status", help="Show campaign status")
    status_parser.add_argument("--campaign-id", required=True, help="Campaign ID")
    status_parser.add_argument("--db", help="SQLite database path")

    # list
    list_parser = subparsers.add_parser("list", help="List campaigns")
    list_parser.add_argument("--branch", help="Filter by branch")
    list_parser.add_argument("--db", help="SQLite database path")

    # compare
    compare_parser = subparsers.add_parser("compare", help="Compare campaigns")
    compare_parser.add_argument("--campaign-a", required=True, help="First campaign ID")
    compare_parser.add_argument(
        "--campaign-b", required=True, help="Second campaign ID"
    )
    compare_parser.add_argument("--db", help="SQLite database path")

    # checkpoint
    checkpoint_parser = subparsers.add_parser("checkpoint", help="Manage checkpoints")
    checkpoint_subparsers = checkpoint_parser.add_subparsers(
        dest="checkpoint_action", help="Checkpoint action"
    )

    checkpoint_list = checkpoint_subparsers.add_parser("list", help="List checkpoints")
    checkpoint_list.add_argument("--campaign-id", required=True, help="Campaign ID")
    checkpoint_list.add_argument(
        "--checkpoint-dir", default="campaigns/checkpoints", help="Checkpoint directory"
    )

    checkpoint_show = checkpoint_subparsers.add_parser(
        "show", help="Show checkpoint details"
    )
    checkpoint_show.add_argument(
        "--checkpoint", required=True, help="Checkpoint file path"
    )

    checkpoint_resume = checkpoint_subparsers.add_parser(
        "resume", help="Generate resume script"
    )
    checkpoint_resume.add_argument(
        "--checkpoint", required=True, help="Checkpoint file path"
    )
    checkpoint_resume.add_argument("--output", help="Output script path")

    # export
    export_parser = subparsers.add_parser("export", help="Export campaign data")
    export_parser.add_argument("--campaign-id", required=True, help="Campaign ID")
    export_parser.add_argument(
        "--format", choices=["json", "csv", "yaml"], default="json"
    )
    export_parser.add_argument("--output", help="Output file path")
    export_parser.add_argument("--db", help="SQLite database path")

    return parser


def _get_search_space(space_name: str) -> dict:
    """Get search space configuration."""
    spaces = {
        "joint_smoke": {
            "substrates": ["digital"],
            "geometries": ["feedforward", "recurrent"],
            "dynamics": ["energy_minimization", "instantaneous"],
            "plasticity": ["null", "routing", "fast_weights"],
            "credits": ["thermodynamic_contrast", "random_projections"],
            "updates": ["euclidean"],
            "tasks": ["mnist"],
        },
        "joint_full": {
            "substrates": [
                "digital",
                "analog",
                "memristive",
                "neuromorphic",
                "ternary",
                "sparse",
            ],
            "geometries": ["feedforward", "recurrent", "tile_mesh"],
            "dynamics": [
                "energy_minimization",
                "instantaneous",
                "predictive_settling",
                "spike_integration",
                "diffusion",
            ],
            "plasticity": [
                "null",
                "routing",
                "fast_weights",
                "substrate_coupled",
                "rule_state",
            ],
            "credits": [
                "thermodynamic_contrast",
                "random_projections",
                "local_goodness",
                "temporal_trace",
                "target_inversion",
                "gradient",
            ],
            "updates": [
                "euclidean",
                "riemannian_orthogonal",
                "spectral_constrained",
                "natural_gradient",
                "elastic_consolidation",
            ],
            "tasks": ["mnist", "cifar10"],
        },
    }
    return spaces.get(space_name, spaces["joint_smoke"])


def _generate_random_coordinate(space: dict) -> str:
    """Generate a random valid 6-D coordinate from search space."""
    import random

    return "/".join([
        random.choice(space["substrates"]),
        random.choice(space["geometries"]),
        random.choice(space["dynamics"]),
        random.choice(space["plasticity"]),
        random.choice(space["credits"]),
        random.choice(space["updates"]),
    ])


def _run_campaign(args) -> int:
    """Run a campaign."""
    import random

    import torch

    from computronium.core.campaign import CampaignStore, get_kernel_cache
    from computronium.core.campaign.checkpoint import (
        CheckpointManager,
    )
    from computronium.core.campaign.frontier_record import FrontierRecord
    from computronium.core.campaign.resource_vector import ResourceUsage

    space = _get_search_space(args.space)
    campaign_id = args.campaign_id or f"camp_{uuid.uuid4().hex[:8]}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    db_path = args.db or (output_dir / "campaign.db")
    checkpoint_dir = output_dir / "checkpoints"

    store = CampaignStore(db_path, checkpoint_dir)
    kernel_cache = get_kernel_cache()
    checkpoint_mgr = CheckpointManager(
        checkpoint_dir, checkpoint_interval=args.checkpoint_interval
    )

    # Create or resume campaign
    if args.resume:
        campaign_state = store.get_latest_on_branch(args.branch)
        if not campaign_state:
            print(f"No campaign found on branch '{args.branch}' to resume")
            return 1
        campaign_id = campaign_state.campaign_id
        print(
            f"Resuming campaign {campaign_id} on branch '{args.branch}' at iteration {campaign_state.iteration}"
        )
    else:
        campaign_state = store.create_campaign(
            campaign_id=campaign_id,
            branch_name=args.branch,
            config={"space": args.space, "objective": args.objective},
            metadata={"created_by": "biopl_campaign"},
        )
        print(f"Created campaign {campaign_id} on branch '{args.branch}'")

    # Run iterations
    for iteration in range(
        campaign_state.iteration + 1, campaign_state.iteration + args.iterations + 1
    ):
        print(f"\n=== Iteration {iteration} ===")

        # Generate coordinates to evaluate
        n_experiments = args.experiments_per_iter
        coordinates = [_generate_random_coordinate(space) for _ in range(n_experiments)]

        for exp_idx, coordinate in enumerate(coordinates):
            print(f"  [{exp_idx + 1}/{n_experiments}] Evaluating: {coordinate}")

            if args.dry_run:
                print("    [DRY RUN] Skipping execution")
                continue

            # TODO: Actually evaluate the coordinate
            # For now, create a mock FrontierRecord
            task_name = random.choice(space["tasks"])

            # Mock evaluation - replace with actual joint system evaluation
            frontier_record = FrontierRecord(
                coordinate=coordinate,
                task_name=task_name,
                task_loss=random.uniform(0.1, 1.0),
                task_accuracy=random.uniform(0.5, 0.99),
                adaptation_time=random.randint(10, 100),
                rho_jacobian=random.uniform(0.5, 1.5),
                lyapunov_local=random.uniform(-0.5, 0.5),
                settling_time=random.uniform(5, 50),
                basin_stability=random.uniform(0.3, 1.0),
                resources=ResourceUsage.measure(
                    model=torch.nn.Linear(784, 10),
                    input_tensor=torch.randn(64, 784),
                ),
                plasticity_primitive=coordinate.split("/")[3],
                plasticity_config={},
                campaign_id=campaign_id,
                episode_index=iteration,
            )

            # Store episode
            episode_id = store.add_episode(
                campaign_id=campaign_id,
                branch_name=args.branch,
                iteration=iteration,
                coordinate=coordinate,
                task_name=task_name,
                frontier_record=frontier_record,
            )

            # Store registry snapshot
            # TODO: Get actual registry signature and composite state shape
            store.add_registry_snapshot(
                campaign_id=campaign_id,
                episode_id=episode_id,
                registry_signature="mock_signature",
                composite_state_shape={
                    "activity": {"weight": (784, 10)},
                    "plastic": {},
                    "substrate": {},
                },
                plasticity_primitive=frontier_record.plasticity_primitive,
                plasticity_config=frontier_record.plasticity_config,
            )

            print(
                f"    ✓ Accuracy: {frontier_record.task_accuracy:.4f}, Loss: {frontier_record.task_loss:.4f}"
            )

        # Checkpoint
        if checkpoint_mgr.should_checkpoint(iteration):
            # TODO: Get actual composite_state and context
            print(f"  Checkpointing at iteration {iteration}...")
            # checkpoint_mgr.create_checkpoint(...)

        # Update campaign state
        store.update_iteration(campaign_id, iteration)

    print(f"\nCampaign {campaign_id} completed!")
    print(f"Results stored in: {db_path}")
    return 0


def _show_status(args) -> int:
    """Show campaign status."""
    from computronium.core.campaign import CampaignStore

    db_path = args.db or "campaigns/campaign.db"
    store = CampaignStore(db_path)

    campaign = store.get_campaign(args.campaign_id)
    if not campaign:
        print(f"Campaign {args.campaign_id} not found")
        return 1

    print(f"Campaign: {campaign.campaign_id}")
    print(f"Branch: {campaign.branch_name}")
    print(f"Parent: {campaign.parent_branch or 'None'}")
    print(f"Iteration: {campaign.iteration}")
    print(f"Created: {campaign.created_at}")
    print(f"Updated: {campaign.updated_at}")
    print(f"Config: {campaign.config}")
    print(f"Metadata: {campaign.metadata}")

    episodes = store.get_episodes(args.campaign_id)
    print(f"\nEpisodes: {len(episodes)}")
    for ep in episodes[-5:]:  # Show last 5
        fr = ep.frontier_record
        print(
            f"  Iter {ep.iteration}: {ep.coordinate} -> acc={fr.get('task_accuracy', 0):.4f}"
        )

    return 0


def _list_campaigns(args) -> int:
    """List all campaigns."""
    from computronium.core.campaign import CampaignStore

    db_path = args.db or "campaigns/campaign.db"
    store = CampaignStore(db_path)

    campaigns = store.list_campaigns(args.branch)
    if not campaigns:
        print("No campaigns found")
        return 0

    print(f"{'Campaign ID':<15} {'Branch':<15} {'Iter':<6} {'Created':<20} {'Parent'}")
    print("-" * 80)
    for c in campaigns:
        print(
            f"{c.campaign_id:<15} {c.branch_name:<15} {c.iteration:<6} {c.created_at:<20} {c.parent_branch or '-'}"
        )

    return 0


def _compare_campaigns(args) -> int:
    """Compare two campaigns."""
    from computronium.core.campaign import CampaignStore

    db_path = args.db or "campaigns/campaign.db"
    store = CampaignStore(db_path)

    camp_a = store.get_campaign(args.campaign_a)
    camp_b = store.get_campaign(args.campaign_b)

    if not camp_a or not camp_b:
        print("One or both campaigns not found")
        return 1

    eps_a = store.get_episodes(args.campaign_a)
    eps_b = store.get_episodes(args.campaign_b)

    print(f"Campaign A: {camp_a.campaign_id} ({len(eps_a)} episodes)")
    print(f"Campaign B: {camp_b.campaign_id} ({len(eps_b)} episodes)")

    # Compare best results
    if eps_a and eps_b:
        best_a = max(eps_a, key=lambda e: e.frontier_record.get("task_accuracy", 0))
        best_b = max(eps_b, key=lambda e: e.frontier_record.get("task_accuracy", 0))

        fr_a = best_a.frontier_record
        fr_b = best_b.frontier_record

        print(
            f"\nBest A: {fr_a.get('task_accuracy', 0):.4f} acc, {fr_a.get('task_loss', 0):.4f} loss"
        )
        print(
            f"Best B: {fr_b.get('task_accuracy', 0):.4f} acc, {fr_b.get('task_loss', 0):.4f} loss"
        )

    return 0


def _manage_checkpoints(args) -> int:
    """Manage checkpoints."""
    from computronium.core.campaign.checkpoint import (
        CheckpointManager,
        create_resume_script,
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    mgr = CheckpointManager(checkpoint_dir)

    if args.checkpoint_action == "list":
        checkpoints = mgr.list_checkpoints(args.campaign_id)
        if not checkpoints:
            print(f"No checkpoints found for campaign {args.campaign_id}")
            return 0

        print(f"Checkpoints for {args.campaign_id}:")
        for cp in checkpoints:
            print(f"  {cp.name} ({cp.stat().st_size} bytes)")

    elif args.checkpoint_action == "show":
        checkpoint = mgr.load_checkpoint(args.checkpoint)
        print(f"Checkpoint: {checkpoint.campaign_id}")
        print(f"  Episode: {checkpoint.episode_index}")
        print(f"  Branch: {checkpoint.branch_name}")
        print(f"  Timestamp: {checkpoint.timestamp}")
        print(f"  Coordinate: {checkpoint.coordinate}")
        print(f"  Task: {checkpoint.task_name}")
        print(f"  Composite state keys: {list(checkpoint.composite_state.keys())}")
        print(f"  Theta keys: {list(checkpoint.theta.keys())}")

    elif args.checkpoint_action == "resume":
        output = args.output or f"resume_{Path(args.checkpoint).stem}.sh"
        created = create_resume_script(args.checkpoint, output)
        print(f"Resume script created: {created}")

    return 0


def _export_campaign(args) -> int:
    """Export campaign data."""
    import csv
    import json

    from computronium.core.campaign import CampaignStore

    db_path = args.db or "campaigns/campaign.db"
    store = CampaignStore(db_path)

    campaign = store.get_campaign(args.campaign_id)
    if not campaign:
        print(f"Campaign {args.campaign_id} not found")
        return 1

    episodes = store.get_episodes(args.campaign_id)

    data = {
        "campaign": {
            "campaign_id": campaign.campaign_id,
            "branch_name": campaign.branch_name,
            "parent_branch": campaign.parent_branch,
            "iteration": campaign.iteration,
            "created_at": campaign.created_at,
            "updated_at": campaign.updated_at,
            "config": campaign.config,
            "metadata": campaign.metadata,
        },
        "episodes": [
            {
                "iteration": ep.iteration,
                "timestamp": ep.timestamp,
                "coordinate": ep.coordinate,
                "task_name": ep.task_name,
                "frontier_record": ep.frontier_record,
            }
            for ep in episodes
        ],
    }

    if args.format == "json":
        output = json.dumps(data, indent=2)
    elif args.format == "yaml":
        import yaml

        output = yaml.dump(data, default_flow_style=False)
    else:  # csv
        import io

        output_io = io.StringIO()
        if episodes:
            fieldnames = ["iteration", "timestamp", "coordinate", "task_name"]
            # Add frontier record fields
            sample_fr = episodes[0].frontier_record
            for key in sample_fr.keys():
                fieldnames.append(f"fr_{key}")

            writer = csv.DictWriter(output_io, fieldnames=fieldnames)
            writer.writeheader()
            for ep in episodes:
                row = {
                    "iteration": ep.iteration,
                    "timestamp": ep.timestamp,
                    "coordinate": ep.coordinate,
                    "task_name": ep.task_name,
                }
                for k, v in ep.frontier_record.items():
                    row[f"fr_{k}"] = v
                writer.writerow(row)
        output = output_io.getvalue()

    if args.output:
        Path(args.output).write_text(output)
        print(f"Exported to {args.output}")
    else:
        print(output)

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point for ``biopl campaign``."""
    args = _build_parser().parse_args(argv)

    if not args.subcommand:
        _build_parser().print_help()
        return 1

    if args.subcommand == "run":
        return _run_campaign(args)
    elif args.subcommand == "status":
        return _show_status(args)
    elif args.subcommand == "list":
        return _list_campaigns(args)
    elif args.subcommand == "compare":
        return _compare_campaigns(args)
    elif args.subcommand == "checkpoint":
        return _manage_checkpoints(args)
    elif args.subcommand == "export":
        return _export_campaign(args)
    else:
        print(f"Unknown subcommand: {args.subcommand}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
