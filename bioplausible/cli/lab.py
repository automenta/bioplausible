"""
CLI Lab for Verification and Inspection
"""

import argparse
from typing import cast

import torch
from torch import nn

from bioplausible.core.construction import construct_model
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

    # Create Model
    model_cls = Registry.get(ComponentCategory.MODEL, args.model)
    # Task geometry is passed straight through: vision tasks expose spatial
    # ``input_dim`` tuples that the construction layer folds (same contract as
    # ``_build_runconfig_model``). ``0`` guards the LM ``None`` case.
    model = cast(
        "nn.Module",
        construct_model(
            model_cls,
            {"hidden_dim": 64, "num_layers": 2},
            input_dim=cast("int", task.input_dim or 0),
            output_dim=task.output_dim,
            model_name=args.model,
        ),
    ).to(device)

    logger.info("Model Created: %s", model.__class__.__name__)
    logger.info(
        "Parameters: %.2fM", count_parameters(model, trainable_only=False) / 1e6
    )

    # Run Dummy Forward
    logger.info("Running Verification Inference...")
    x, _ = task.get_batch("val")
    model.eval()
    with torch.no_grad():
        # LM models that expose `embed` expect integer token ids — task.get_batch
        # already returns those ids, so forward handles the embedding internally.
        # Non-LM models may receive raw features; flatten spatially for MLPs.
        if x.dim() > 2 and "Conv" not in args.model:  # ruff: ignore[magic-value-comparison]  # flatten >2D feature tensors for MLPs
            x = x.view(x.size(0), -1)

        try:
            x = x.to(device)
            out = model(x)
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
