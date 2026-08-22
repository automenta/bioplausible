"""
CLI Lab for Verification and Inspection
"""

import argparse
from typing import cast

import torch
from torch import nn

from bioplausible.core.logging import get_logger
from bioplausible.core.registry import ComponentCategory, Registry
from bioplausible.core.utils.device import get_device
from bioplausible.domains import create_task
from bioplausible.utils import count_parameters

logger = get_logger()

__all__ = [
    "inspect_model",
    "logger",
    "main",
]


def inspect_model(args):
    logger.info("[LAB]  Inspecting Model: %s", args.model)

    device = str(get_device())

    # Create Task
    task = create_task(args.task, device=device)
    task.setup()
    logger.info(
        "Task: %s, Input: %s, Output: %s", args.task, task.input_dim, task.output_dim
    )

    # Create System via 5-D ontology projection
    # This uses the native 5-D composition for models that support it,
    # or falls back to ModelAdapter for legacy models.
    system = Registry.to_system(
        args.model,
        input_dim=task.input_dim or 0,
        hidden_dim=64,
        output_dim=task.output_dim,
        num_layers=2,
    )

    logger.info("System Created: %s", type(system).__name__)

    # Get parameter count from geometry
    if hasattr(system, "geometry") and hasattr(system.geometry, "params"):
        param_count = sum(p.numel() for p in system.geometry.params.values())
        logger.info("Parameters: %.2fM", param_count / 1e6)

    # Run Dummy Forward
    logger.info("Running Verification Inference...")
    x, _ = task.get_batch("val")
    with torch.no_grad():
        # LM models that expose `embed` expect integer token ids — task.get_batch
        # already returns those ids, so forward handles the embedding internally.
        # Non-LM models may receive raw features; flatten spatially for MLPs.
        if x.dim() > 2 and "Conv" not in args.model:  # ruff: ignore[magic-value-comparison]  # flatten >2D feature tensors for MLPs
            x = x.view(x.size(0), -1)

        try:
            x = x.to(device)
            # Move system geometry to device if needed
            if hasattr(system.geometry, "to"):
                system.geometry.to(device)
            out = system.forward(x)
        except RuntimeError, ValueError, TypeError:
            logger.exception("Forward pass failed for model %s", args.model)
            return
        logger.info("[OK]  Forward pass successful. Output shape: %s", out.shape)


def main():
    parser = argparse.ArgumentParser(description="Bioplausible Lab CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    inspect = subparsers.add_parser("inspect", help="Inspect a model architecture")
    inspect.add_argument("--model", required=True, help="Model name")
    inspect.add_argument("--task", default="vision", help="Task type")

    args = parser.parse_args()

    if args.command == "inspect":
        inspect_model(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
