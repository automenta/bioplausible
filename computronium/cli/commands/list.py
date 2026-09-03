"""List models CLI command."""

import logging
from typing import TYPE_CHECKING

from computronium.core.registry import ComponentCategory, Registry

if TYPE_CHECKING:
    import argparse


def add_list_subparsers(subparsers: argparse._SubParsersAction) -> None:
    """Add List command to subparsers."""
    subparsers.add_parser("list", help="List available models")


def run_list(args: argparse.Namespace) -> None:
    """List available models."""
    logger = logging.getLogger(__name__)
    logger.info("Available Models (Zoo Registry):")
    for entry in Registry.query(category=ComponentCategory.MODEL):
        bio = entry.get("bio_plausibility", 0)
        logger.info("  %-30s bio=%.1f", entry["name"], bio)
