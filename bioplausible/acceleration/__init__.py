"""
Acceleration Module for Bioplausible

Provides multiple acceleration backends for Equilibrium Propagation and
all bio-plausible algorithm families:
EqProp, FA, Hebbian, FF, PEPITA, TP, PC, SNN, Tile, MEP, O1Memory, Backprop.

Backends (in order of priority for speed):
    1. Triton Kernels: Custom GPU kernels for fused operations (fastest)
    2. CuPy: NumPy-compatible GPU arrays via CUDA
    3. torch.compile: PyTorch 2.0+ JIT compilation
    4. Pure PyTorch: Standard autograd (fallback)
    5. Pure NumPy: CPU-only kernel (portability)

Usage:
    from bioplausible.acceleration import (
        get_optimal_backend,
        compile_model,
        HAS_CUPY,
        HAS_TRITON,
        KernelBackend,
        KernelRegistry,
        KernelConfig,
        AlgorithmFamily,
        HardwareTarget,
    )

    # Check available backends
    >>> from bioplausible.core.logging import get_logger
    >>> get_logger().info("CuPy: %s, Triton: %s", HAS_CUPY, HAS_TRITON)
"""

# Import to trigger EQPROP kernel backend registration
from bioplausible.acceleration import (
    eqprop_kernel_backend,
)
from bioplausible.acceleration.backends import (
    HAS_CUPY,
    HAS_TRITON,
    AutoDispatcher,
    BackendBenchmark,
    BackendDetector,
    BackendType,
    CupyChecker,
    KernelProfiler,
    TritonChecker,
    check_cupy_available,
    check_triton_available,
    dispatch_kernel,
    enable_tf32,
    get_dispatcher,
    get_optimal_backend,
    profile_kernel,
)
from bioplausible.acceleration.compile import compile_model, compile_settling_loop
from bioplausible.acceleration.contrastive_kernels import (
    BaseContrastiveKernel,
    ContrastiveConfig,
    ContrastiveKernel,
    FAContrastiveKernel,
    FFContrastiveKernel,
    HebbianContrastiveKernel,
    MEPContrastiveKernel,
    O1MemoryContrastiveKernel,
    PCContrastiveKernel,
    PEPITAContrastiveKernel,
    SNNContrastiveKernel,
    TileContrastiveKernel,
    TPContrastiveKernel,
    get_contrastive_kernel,
    register_contrastive_kernels,
)
from bioplausible.acceleration.contrastive_primitives import (
    batched_outer_product,
    conductance_matmul,
    contrastive_delta,
    contrastive_hebbian_update,
    forward_forward_goodness,
    lif_step,
    pepita_error_modulation,
    phase_encode,
    predictive_coding_inference_step,
    spectral_norm_power_iteration,
    stdp_update,
    target_propagation_target,
)
from bioplausible.acceleration.kernel_backend import (
    AlgorithmFamily,
    HardwareTarget,
    KernelBackend,
    KernelConfig,
    KernelRegistry,
    LocalityLevel,
    infer_algorithm_family,
)
from bioplausible.core.utils.activations import (
    cross_entropy,
    get_backend,
    softmax,
    spectral_normalize,
    to_numpy,
)


def get_kernel_classes() -> tuple[type[object], type[object]]:
    """Lazily import kernel classes to avoid circular imports."""
    from bioplausible.acceleration.kernels import EqPropKernel as _EqPropKernel
    from bioplausible.acceleration.kernels import (
        EqPropKernelBPTT as _EqPropKernelBPTT,
    )

    return _EqPropKernel, _EqPropKernelBPTT


def get_triton_ops() -> type[object] | None:
    """Lazily import Triton ops, returning None if unavailable."""
    try:
        from bioplausible.acceleration.triton_kernels import (
            TritonEqPropOps as _TritonEqPropOps,
        )
    except ImportError:
        return None
    return _TritonEqPropOps


def get_algorithm_kernels() -> dict[str, type[object]]:
    """Get all algorithm-specific kernel backends (uniform interface)."""
    kernels = {}
    kernel_specs = [
        ("bioplausible.acceleration.fa_kernels", ["FAKernelBackend"], ["fa"]),
        (
            "bioplausible.acceleration.hebbian_kernels",
            ["HebbianKernelBackend", "ThreeFactorKernelBackend"],
            ["hebbian", "three_factor"],
        ),
        (
            "bioplausible.acceleration.ff_kernels",
            ["FFKernelBackend", "PEPITAKernelBackend"],
            ["ff", "pepita"],
        ),
        ("bioplausible.acceleration.tp_kernels", ["TPKernelBackend"], ["tp"]),
        ("bioplausible.acceleration.pc_kernels", ["PCKernelBackend"], ["pc"]),
        ("bioplausible.acceleration.snn_kernels", ["SNNKernelBackend"], ["snn"]),
        ("bioplausible.acceleration.tile_kernels", ["TileKernelBackend"], ["tile"]),
        (
            "bioplausible.acceleration.mep_kernels",
            ["MEPKernelBackend", "O1MemoryEPv2KernelBackend"],
            ["mep", "o1memory"],
        ),
        (
            "bioplausible.acceleration.backprop_kernels",
            ["BackpropKernelBackend"],
            ["backprop"],
        ),
        (
            "bioplausible.acceleration.eqprop_kernel_backend",
            ["EqPropKernelBackend"],
            ["eqprop"],
        ),
    ]
    for module_name, class_names, keys in kernel_specs:
        try:
            module = __import__(module_name, fromlist=class_names)
            for cls_name, key in zip(class_names, keys):
                kernels[key] = getattr(module, cls_name)
        except ImportError:
            pass
    return kernels


def get_contrastive_kernels() -> dict[str, type[object]]:
    """Get all contrastive Hebbian kernel backends (O(1) memory interface).

    Pure getter: does NOT touch the KernelRegistry (registering would clobber
    the standard uniform-interface backends registered for the same families).
    Use ``get_contrastive_kernel(AlgorithmFamily)`` for instances.
    """
    return {
        "fa": FAContrastiveKernel,
        "hebbian": HebbianContrastiveKernel,
        "ff": FFContrastiveKernel,
        "pepita": PEPITAContrastiveKernel,
        "tp": TPContrastiveKernel,
        "pc": PCContrastiveKernel,
        "snn": SNNContrastiveKernel,
        "tile": TileContrastiveKernel,
        "mep": MEPContrastiveKernel,
        "o1memory": O1MemoryContrastiveKernel,
    }


__all__ = [
    "HAS_CUPY",
    "HAS_TRITON",
    "AlgorithmFamily",
    "AutoDispatcher",
    "BackendBenchmark",
    "BackendDetector",
    "BackendType",
    "BaseContrastiveKernel",
    "ContrastiveConfig",
    "ContrastiveKernel",
    "CupyChecker",
    "FAContrastiveKernel",
    "FFContrastiveKernel",
    "HardwareTarget",
    "HebbianContrastiveKernel",
    "KernelBackend",
    "KernelConfig",
    "KernelProfiler",
    "KernelRegistry",
    "LocalityLevel",
    "MEPContrastiveKernel",
    "O1MemoryContrastiveKernel",
    "PCContrastiveKernel",
    "PEPITAContrastiveKernel",
    "SNNContrastiveKernel",
    "TPContrastiveKernel",
    "TileContrastiveKernel",
    "TritonChecker",
    "batched_outer_product",
    "check_cupy_available",
    "check_triton_available",
    "compile_model",
    "compile_settling_loop",
    "conductance_matmul",
    "contrastive_delta",
    "contrastive_hebbian_update",
    "cross_entropy",
    "dispatch_kernel",
    "enable_tf32",
    "eqprop_kernel_backend",
    "forward_forward_goodness",
    "get_algorithm_kernels",
    "get_backend",
    "get_contrastive_kernel",
    "get_contrastive_kernels",
    "get_dispatcher",
    "get_kernel_classes",
    "get_optimal_backend",
    "get_triton_ops",
    "infer_algorithm_family",
    "lif_step",
    "pepita_error_modulation",
    "phase_encode",
    "predictive_coding_inference_step",
    "profile_kernel",
    "register_contrastive_kernels",
    "softmax",
    "spectral_norm_power_iteration",
    "spectral_normalize",
    "stdp_update",
    "target_propagation_target",
    "to_numpy",
]
