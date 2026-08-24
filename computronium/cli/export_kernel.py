"""``biopl-export-kernel`` — export a kernel backend to a deployment target.

Builds a :class:`~computronium.acceleration.kernel_backend.KernelBackend` from the
registry for an algorithm family + hardware target, initializes it, and exports
its weights + hardware manifest via
:func:`computronium.acceleration.export_kernel`.

Usage::

    uv run biopl-export-kernel --algorithm backprop --target cpu --output ./exports
    uv run biopl-export-kernel --algorithm eqprop --target fpga --output ./hls
    uv run biopl-export-kernel --algorithm spiking --target neuromorphic --output ./n
"""

import argparse
from typing import TYPE_CHECKING, cast

import torch

from computronium.acceleration import (
    AlgorithmFamily,
    HardwareTarget,
    KernelConfig,
    KernelRegistry,
    get_algorithm_kernels,
)
from computronium.acceleration.export import export_kernel
from computronium.core.logging import get_logger

if TYPE_CHECKING:
    from computronium.acceleration.kernel_backend import KernelBackend

logger = get_logger()


def main(argv: list[str] | None = None) -> int:
    """Run the kernel-export CLI."""
    parser = argparse.ArgumentParser(description="Export a kernel backend")
    parser.add_argument(
        "--algorithm",
        default="backprop",
        choices=[a.value for a in AlgorithmFamily],
        help="Algorithm family to export",
    )
    parser.add_argument(
        "--target",
        default="cpu",
        choices=[h.value for h in HardwareTarget],
        help="Hardware target for the manifest descriptor",
    )
    parser.add_argument(
        "--output",
        default="./exports",
        help="Output directory for exported artifacts",
    )
    parser.add_argument(
        "--precision",
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Computation dtype",
    )
    args = parser.parse_args(argv)

    get_algorithm_kernels()  # populate the registry (lazy import side effect)
    family = AlgorithmFamily(args.algorithm)
    target = HardwareTarget(args.target)
    backend = KernelRegistry.get_best(family, target)
    if backend is None:
        logger.error("No kernel backend registered for %s on %s", family, target)
        return 1
    kernel = cast("KernelBackend", backend)

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    config = KernelConfig(
        algorithm=family,
        hardware=target,
        dtype=dtype_map[args.precision],
        use_autograd=False,
        settle_steps=30
        if family in {AlgorithmFamily.MEP, AlgorithmFamily.O1MEMORY}
        else 0,
        beta=0.5,
        gamma=1.0,
    )
    kernel.initialize(config)

    result = export_kernel(kernel, config, target=target, output_dir=args.output)
    logger.info("Wrote manifest: %s", result.manifest_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
