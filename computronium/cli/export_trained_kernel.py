"""``python -m computronium.cli.export_trained_kernel`` — train a
kernel-backed model and export its weights.

Builds a model via ``CoreTrainer(use_kernel=True)``, runs training (or loads a
checkpoint), then exports the bound backend's state dict + hardware manifest via
:func:`computronium.acceleration.export.export_kernel`.

Usage::

    uv run python -m computronium.cli.export_trained_kernel \
        --algorithm fa --target triton --epochs 20 --output ./trained_fa
    uv run python -m computronium.cli.export_trained_kernel \
        --algorithm backprop --target cpu \
        --checkpoint ./checkpoints/best.pt --output ./trained_bp
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from computronium.acceleration import (
    AlgorithmFamily,
    HardwareTarget,
    KernelRegistry,
    get_algorithm_kernels,
)
from computronium.acceleration.export import export_kernel
from computronium.core.logging import get_logger
from computronium.core.trainer import CoreTrainer, TrainerConfig

# Import zoo models to trigger registration


logger = get_logger()


def main(argv: list[str] | None = None) -> int:  # ruff: ignore[too-many-locals, too-many-statements]
    """Run the trained-kernel export CLI."""
    parser = argparse.ArgumentParser(
        description="Train a kernel-backed model and export its weights"
    )
    parser.add_argument(
        "--algorithm",
        default="backprop",
        choices=[a.value for a in AlgorithmFamily],
        help="Algorithm family to train and export",
    )
    parser.add_argument(
        "--target",
        default="triton",
        choices=[h.value for h in HardwareTarget],
        help="Hardware target for the kernel backend",
    )
    parser.add_argument(
        "--output",
        default="./exports/trained",
        help="Output directory for exported artifacts",
    )
    parser.add_argument(
        "--precision",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Computation dtype",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs (ignored if --checkpoint provided)",
    )
    parser.add_argument(
        "--dataset",
        default="digits",
        choices=["digits", "mnist", "fashion_mnist"],
        help="Training dataset",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to checkpoint to load instead of training",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden layer dimension",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of hidden layers",
    )
    parser.add_argument(
        "--include-onnx",
        action="store_true",
        default=True,
        help="Attempt ONNX export of the Linear stack",
    )
    parser.add_argument(
        "--no-onnx",
        action="store_false",
        dest="include_onnx",
        help="Skip ONNX export",
    )
    args = parser.parse_args(argv)

    get_algorithm_kernels()  # populate the registry (lazy import side effect)

    family = AlgorithmFamily(args.algorithm)
    target = HardwareTarget(args.target)

    # Map precision string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.precision]  # ruff: ignore[unused-variable]

    # Determine device from target
    device_map = {
        HardwareTarget.CPU: torch.device("cpu"),
        HardwareTarget.CUDA: torch.device("cuda"),
        HardwareTarget.TRITON: torch.device("cuda"),
        HardwareTarget.FPGA: torch.device("cpu"),
        HardwareTarget.NEUROMORPHIC: torch.device("cpu"),
        HardwareTarget.OPTICAL: torch.device("cpu"),
        HardwareTarget.CROSSBAR: torch.device("cpu"),
        HardwareTarget.QUANTUM: torch.device("cpu"),
    }
    device = device_map.get(
        target, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Skip CUDA-only targets on CPU
    if (
        target in (HardwareTarget.TRITON, HardwareTarget.CUDA)  # ruff: ignore[literal-membership]
        and not torch.cuda.is_available()
    ):
        logger.warning("%s requires CUDA; falling back to CPU", target.value)
        target = HardwareTarget.CPU
        device = torch.device("cpu")  # ruff: ignore[unused-variable]

    # Map target to kernel_backend string for CoreTrainer
    # Trainer's _wrap_with_kernel expects: "triton", "cupy", "pytorch", "contrastive"
    backend_str_map = {
        HardwareTarget.CPU: "pytorch",
        HardwareTarget.CUDA: "cupy",
        HardwareTarget.TRITON: "triton",
        HardwareTarget.FPGA: "pytorch",  # FPGA simulation on CPU
        HardwareTarget.NEUROMORPHIC: "pytorch",
        HardwareTarget.OPTICAL: "pytorch",
        HardwareTarget.CROSSBAR: "pytorch",
        HardwareTarget.QUANTUM: "pytorch",
    }
    kernel_backend_str = backend_str_map.get(target, "pytorch")

    # Get best available backend for the algorithm family
    backend = KernelRegistry.get_best(family, target)
    if backend is None:
        logger.error("No kernel backend registered for %s on %s", family, target)
        return 1

    logger.info("Using kernel backend: %s", type(backend).__name__)

    # Create trainer config
    trainer_config = TrainerConfig(
        model=_model_name_for_algorithm(family),
        model_kwargs={
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
        },
        optimizer="adam",
        optimizer_kwargs={"lr": args.learning_rate},
        task=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_kernel=True,
        kernel_backend=kernel_backend_str,
        kernel_dtype=args.precision,
        target_hardware=None,  # Let the kernel backend handle hardware mapping
        use_compile=False,
        track_energy=True,
        track_flops=True,
        track_memory=True,
        save_checkpoints=True,
        checkpoint_dir=str(Path(args.output).parent / "checkpoints"),
        seed=42,
        deterministic=False,
        device="auto",
    )

    # Create trainer
    trainer = CoreTrainer(trainer_config)
    trainer.setup()

    # The model should now have the kernel backend attached
    if (
        not hasattr(trainer.model, "_kernel_backend")
        or trainer.model._kernel_backend is None
    ):
        logger.error("Model does not have a kernel backend attached")
        return 1

    kernel_backend = trainer.model._kernel_backend
    kernel_config = trainer.model._kernel_config

    # Train or load checkpoint
    if args.checkpoint:
        logger.info("Loading checkpoint from %s", args.checkpoint)
        trainer.load_checkpoint(str(args.checkpoint))
    else:
        logger.info("Training model for %d epochs on %s", args.epochs, args.dataset)
        trainer.fit()

    # Export the BOUND backend (with trained weights)
    logger.info("Exporting trained kernel backend...")
    result = export_kernel(
        kernel_backend,
        kernel_config,
        target=target,
        output_dir=args.output,
        include_onnx=args.include_onnx,
    )

    # Write a summary JSON
    summary = {
        "algorithm": family.value,
        "hardware_target": target.value,
        "precision": args.precision,
        "epochs": args.epochs if not args.checkpoint else "loaded_from_checkpoint",
        "dataset": args.dataset,
        "model_config": {
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
        },
        "artifacts": {
            "manifest": result.manifest_path,
            "state_dict": result.state_dict_path,
            "onnx": result.onnx_path,
        },
    }
    summary_path = Path(args.output) / "export_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    logger.info("Export complete!")
    logger.info("Manifest: %s", result.manifest_path)
    logger.info("State dict: %s", result.state_dict_path)
    if result.onnx_path:
        logger.info("ONNX: %s", result.onnx_path)
    logger.info("Summary: %s", summary_path)

    return 0


def _model_name_for_algorithm(family: AlgorithmFamily) -> str:
    """Map AlgorithmFamily to registered model name."""
    mapping = {
        AlgorithmFamily.BACKPROP: "backprop_mlp",
        AlgorithmFamily.FA: "standard_fa",
        AlgorithmFamily.HEBBIAN: "standard_fa",  # Uses same model, different optimizer
        AlgorithmFamily.FF: "forward_forward",
        AlgorithmFamily.PEPITA: "pepita",
        AlgorithmFamily.TP: "diff_target_prop",
        AlgorithmFamily.PC: "predictive_coding_hybrid",
        AlgorithmFamily.SNN: "spiking_stdp",
        AlgorithmFamily.TILE: "tile_pc",
        AlgorithmFamily.MEP: "eqprop_mlp",  # MEP uses eqprop architecture
        AlgorithmFamily.O1MEMORY: "eqprop_mlp",
        AlgorithmFamily.EQPROP: "eqprop_mlp",
    }
    return mapping.get(family, "backprop_mlp")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
