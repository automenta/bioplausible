"""``biopl-run`` CLI — run, validate, and dry-view campaign YAMLs (FIX2a §13).

Subcommands:

* ``validate``  — parse a campaign YAML and report schema errors.
* ``dry-run``   — print the resolved model plan without executing anything.
* ``gates``     — run the TIER 0 / 0.5 staircase gates and write a JSONL report.
* ``run``       — alias of ``gates`` (run the campaign's triage tiers).

Register the entry point as ``biopl-run = bioplausible.campaign.cli:main``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from bioplausible.campaign.logger import ExperimentLogger, GateOutcome
from bioplausible.campaign.runner import CampaignRunner, run_gates
from bioplausible.campaign.schema import load_campaign

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="biopl-run", description=(__doc__ or "").splitlines()[0] or "biopl-run"
    )
    parser.add_argument("--log-level", default="INFO")

    config_parent = argparse.ArgumentParser(add_help=False)
    config_parent.add_argument(
        "--config", required=True, help="Path to the campaign YAML"
    )

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "validate", parents=[config_parent], help="Validate the campaign YAML"
    )
    sub.add_parser(
        "dry-run",
        parents=[config_parent],
        help="Print the resolved plan without running",
    )
    gates = sub.add_parser(
        "gates", parents=[config_parent], help="Run the TIER 0 / 0.5 gates"
    )
    run = sub.add_parser(
        "run",
        parents=[config_parent],
        help="Run the campaign's gate tiers (alias of gates)",
    )
    for subparser in (gates, run):
        subparser.add_argument(
            "--tier",
            choices=("tier0", "tier0.5", "all"),
            default="all",
            help="TIER 0.5 is skipped unless 'tier0.5' or 'all' is requested",
        )
        subparser.add_argument("--device", default="cpu", help="cpu / cuda / auto")
        subparser.add_argument(
            "--seeds", type=int, default=3, help="TIER 0.5 seed count"
        )
    return parser


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def _cmd_validate(config: str) -> int:
    campaign = load_campaign(config)
    arms = ", ".join(sorted(campaign.arms))
    print(f"OK: valid campaign {campaign.meta.name!r} with arms [{arms}]")
    return 0


def _cmd_dry_run(args: argparse.Namespace) -> int:
    campaign = load_campaign(args.config)
    print(CampaignRunner(campaign).dry_run())
    return 0


def _cmd_gates(args: argparse.Namespace) -> int:
    # Model registration is a lazy import side effect (zoo is excluded from the
    # package import graph); the gates actually construct models, so ensure the
    # registry is populated before training.
    import bioplausible.zoo  # ruff: ignore[unused-import]  # triggers model registration

    campaign = load_campaign(args.config)
    report = Path(campaign.output.artifacts_dir) / "gates.jsonl"
    start = time.time()

    result = run_gates(
        campaign,
        device=args.device,
        n_seeds=args.seeds,
        run_tier05=args.tier in {"all", "tier0.5"},
    )
    with ExperimentLogger(report) as log:
        for tier in ("tier0", "tier0.5"):
            for outcome in result.tiers.get(tier, []):
                log.log(
                    GateOutcome(
                        tier=outcome.tier,
                        model=outcome.model,
                        task=outcome.task,
                        passed=outcome.passed,
                        reason=outcome.reason,
                        metrics=outcome.metrics,
                    )
                )

    logger.info("Wrote %s", report)
    for tier in ("tier0", "tier0.5"):
        outcomes = result.tiers.get(tier, [])
        if not outcomes:
            continue
        logger.info("TIER %s outcomes:", tier)
        for o in outcomes:
            verdict = (
                "PASS" if o.passed else ("digits-fail" if tier == "tier0.5" else "FAIL")
            )
            logger.info("  %-22s %s", o.model, verdict)
    logger.info("done in %.1fs", time.time() - start)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point for ``biopl-run``."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.log_level)

    try:
        match args.command:
            case "validate":
                return _cmd_validate(args.config)
            case "dry-run":
                return _cmd_dry_run(args)
            case "gates" | "run":
                return _cmd_gates(args)
            case _:
                return 2
    except (yaml.YAMLError, ValueError, FileNotFoundError) as exc:
        logger.error("campaign error: %s", exc)  # ruff: ignore[error-instead-of-exception]  # user-facing CLI: a traceback is noise
        return 1


if __name__ == "__main__":
    sys.exit(main())
