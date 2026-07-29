"""
CLI Lab for Verification and Inspection
"""

import argparse
import logging

import torch

logger = logging.getLogger(__name__)

from bioplausible.core.registry import ComponentCategory, Registry
from bioplausible.hyperopt.tasks import create_task

__all__ = [
    "inspect_model",
    "logger",
    "main",
]


def inspect_model(args):
    logger.info("[LAB]  Inspecting Model: %s", args.model)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create Task
    task = create_task(args.task, device=device)
    task.setup()
    logger.info(
        "Task: %s, Input: %s, Output: %s", args.task, task.input_dim, task.output_dim
    )

    # Create Model
    model_cls = Registry.get(ComponentCategory.MODEL, args.model)
    model = model_cls(input_dim=task.input_dim, output_dim=task.output_dim).to(device)

    logger.info("Model Created: %s", model.__class__.__name__)
    logger.info("Parameters: %.2fM", sum(p.numel() for p in model.parameters()) / 1e6)

    # Run Dummy Forward
    logger.info("Running Verification Inference...")
    x, _ = task.get_batch("val")
    model.eval()
    with torch.no_grad():
        # LM models that expose `embed` expect integer token ids — task.get_batch
        # already returns those ids, so forward handles the embedding internally.
        # Non-LM models may receive raw features; flatten spatially for MLPs.
        if x.dim() > 2 and "Conv" not in args.model:
            x = x.view(x.size(0), -1)

        try:
            x = x.to(device)
            out = model(x)
        except Exception:
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
