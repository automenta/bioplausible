#!/usr/bin/env python3
"""
Permutation Matrix Verification Tool

Verifies test coverage across the full permutation matrix:
(algorithm × hardware × kernel_type) × test_type

Generates a coverage report showing implemented/tested/green cells.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

# ──────────────────────────────────────────────────────────────────────────────
# Configuration: The full permutation matrix
# ──────────────────────────────────────────────────────────────────────────────

ALGORITHMS = [
    "backprop",
    "fa",
    "hebbian",
    "ff",
    "tp",
    "pc",
    "snn",
    "tile",
    "mep",
    "o1memory",
    "eqprop",
    "contrastive",
]

HARDWARE_TARGETS = [
    "cpu",
    "cuda",
    "triton",
    "cupy",
    "metal",
    "fpga",
    "neuromorphic",
    "analog",
]

KERNEL_TYPES = [
    "standard",  # Standard kernel
    "contrastive",  # Contrastive/O(1) memory kernel
]

TEST_TYPES = [
    "unit",  # Unit tests (parity, init, etc.)
    "integration",  # Integration tests (training, export)
    "benchmark",  # Performance benchmarks
    "accuracy_parity",  # FP16/BF16/INT8 accuracy parity
]

# Expected test files per algorithm/hardware/kernel_type/test_type
TEST_FILE_MAP = {
    ("backprop", "standard"): {
        "unit": "tests/unit/validation/test_backprop_parity.py",
        "integration": "tests/integration/test_kernel_backprop.py",
        "benchmark": "tools/benchmark_all_kernels.py",
        "accuracy_parity": "tests/unit/acceleration/test_mixed_precision.py",
    },
    ("fa", "standard"): {
        "unit": "tests/unit/acceleration/test_fa_kernel_init.py",
        "integration": "tests/integration/test_kernel_fa.py",
        "benchmark": "tools/benchmark_all_kernels.py",
        "accuracy_parity": "tests/unit/acceleration/test_mixed_precision.py",
    },
    ("fa", "contrastive"): {
        "unit": "tests/unit/validation/test_family_kernel_parity.py",
        "integration": "tests/integration/test_contrastive_kernel.py",
        "benchmark": "tools/benchmark_all_kernels.py",
        "accuracy_parity": "tests/unit/acceleration/test_mixed_precision.py",
    },
    ("hebbian", "standard"): {
        "unit": "tests/unit/validation/test_family_kernel_parity.py",
        "integration": "tests/integration/test_kernel_hebbian.py",
        "benchmark": "tools/benchmark_all_kernels.py",
        "accuracy_parity": "tests/unit/acceleration/test_mixed_precision.py",
    },
    ("hebbian", "contrastive"): {
        "unit": "tests/unit/validation/test_family_kernel_parity.py",
        "integration": "tests/integration/test_contrastive_kernel.py",
        "benchmark": "tools/benchmark_all_kernels.py",
        "accuracy_parity": "tests/unit/acceleration/test_mixed_precision.py",
    },
    ("ff", "standard"): {
        "unit": "tests/unit/validation/test_family_kernel_parity.py",
        "integration": "tests/integration/test_kernel_ff.py",
        "benchmark": "tools/benchmark_all_kernels.py",
        "accuracy_parity": "tests/unit/acceleration/test_mixed_precision.py",
    },
    ("tp", "standard"): {
        "unit": "tests/unit/validation/test_family_kernel_parity.py",
        "integration": "tests/integration/test_kernel_tp.py",
        "benchmark": "tools/benchmark_all_kernels.py",
        "accuracy_parity": "tests/unit/acceleration/test_mixed_precision.py",
    },
    ("pc", "standard"): {
        "unit": "tests/unit/validation/test_family_kernel_parity.py",
        "integration": "tests/integration/test_kernel_pc.py",
        "benchmark": "tools/benchmark_all_kernels.py",
        "accuracy_parity": "tests/unit/acceleration/test_mixed_precision.py",
    },
    ("snn", "standard"): {
        "unit": "tests/unit/validation/test_family_kernel_parity.py",
        "integration": "tests/integration/test_kernel_snn.py",
        "benchmark": "tools/benchmark_all_kernels.py",
        "accuracy_parity": "tests/unit/acceleration/test_mixed_precision.py",
    },
    ("tile", "standard"): {
        "unit": "tests/unit/validation/test_family_kernel_parity.py",
        "integration": "tests/integration/test_kernel_tile.py",
        "benchmark": "tools/benchmark_all_kernels.py",
        "accuracy_parity": "tests/unit/acceleration/test_mixed_precision.py",
    },
    ("mep", "standard"): {
        "unit": "tests/unit/validation/test_family_kernel_parity.py",
        "integration": "tests/integration/test_kernel_mep.py",
        "benchmark": "tools/benchmark_all_kernels.py",
        "accuracy_parity": "tests/unit/acceleration/test_mixed_precision.py",
    },
    ("o1memory", "standard"): {
        "unit": "tests/unit/validation/test_family_kernel_parity.py",
        "integration": "tests/integration/test_kernel_o1memory.py",
        "benchmark": "tools/benchmark_all_kernels.py",
        "accuracy_parity": "tests/unit/acceleration/test_mixed_precision.py",
    },
    ("eqprop", "standard"): {
        "unit": "tests/unit/acceleration/test_eqprop_kernel_backend.py",
        "integration": "tests/integration/test_kernel_eqprop.py",
        "benchmark": "tools/benchmark_all_kernels.py",
        "accuracy_parity": "tests/unit/acceleration/test_mixed_precision.py",
    },
    ("contrastive", "contrastive"): {
        "unit": "tests/unit/validation/test_family_kernel_parity.py",
        "integration": "tests/integration/test_contrastive_kernel.py",
        "benchmark": "tools/benchmark_all_kernels.py",
        "accuracy_parity": "tests/unit/acceleration/test_mixed_precision.py",
    },
}

# Hardware support matrix: which hardware targets each algorithm supports
HARDWARE_SUPPORT = {
    "backprop": ["cpu", "cuda", "triton", "metal"],
    "fa": ["cpu", "cuda", "triton", "metal"],
    "hebbian": ["cpu", "cuda", "triton", "metal"],
    "ff": ["cpu", "cuda", "triton", "metal"],
    "tp": ["cpu", "cuda", "triton"],
    "pc": ["cpu", "cuda", "triton"],
    "snn": ["cpu", "cuda", "triton", "neuromorphic"],
    "tile": ["cpu", "cuda", "triton"],
    "mep": ["cpu", "cuda", "triton"],
    "o1memory": ["cpu", "cuda"],
    "eqprop": ["cpu", "cuda", "triton"],
    "contrastive": ["cpu", "cuda", "triton"],
}

# Export targets
EXPORT_TARGETS = ["hls", "verilog", "nxsdk", "spice", "onnx"]
EXPORT_SUPPORT = {
    "fa": ["hls", "verilog", "onnx"],
    "hebbian": ["hls", "verilog", "onnx"],
    "ff": ["hls", "onnx"],
    "tp": ["hls", "onnx"],
    "pc": ["hls", "onnx"],
    "snn": ["nxsdk", "onnx"],
    "tile": ["hls", "verilog", "onnx"],
    "mep": ["hls", "onnx"],
    "eqprop": ["hls", "onnx"],
    "backprop": ["hls", "onnx"],
}

# ──────────────────────────────────────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class CellStatus:
    """Status of a single matrix cell."""

    algorithm: str
    hardware: str
    kernel_type: str
    test_type: str
    status: Literal[
        "supported",
        "not_supported",
        "implemented",
        "tested",
        "passing",
        "failing",
        "skipped",
    ]
    test_file: str | None = None
    test_result: str | None = None
    notes: str = ""


@dataclass
class CoverageReport:
    """Full coverage report."""

    timestamp: str
    total_cells: int
    supported_cells: int
    implemented_cells: int
    tested_cells: int
    passing_cells: int
    cells: list[CellStatus]


# ──────────────────────────────────────────────────────────────────────────────
# Coverage Analysis
# ──────────────────────────────────────────────────────────────────────────────


def check_file_exists(path: str) -> bool:
    """Check if a test file exists."""
    return Path(path).exists()


def run_test_file(test_file: str, test_filter: str = "") -> tuple[bool, str]:
    """Run a test file and return (passed, output)."""
    try:
        cmd = ["uv", "run", "pytest", test_file, "-x", "--tb=short", "-q"]
        if test_filter:
            cmd.extend(["-k", test_filter])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)


def analyze_cell(
    algorithm: str,
    hardware: str,
    kernel_type: str,
    test_type: str,
    run_tests: bool = False,
) -> CellStatus:
    """Analyze a single matrix cell."""

    # Check hardware support
    supported_hw = HARDWARE_SUPPORT.get(algorithm, [])
    if hardware not in supported_hw:
        return CellStatus(
            algorithm=algorithm,
            hardware=hardware,
            kernel_type=kernel_type,
            test_type=test_type,
            status="not_supported",
            notes=f"{algorithm} does not support {hardware}",
        )

    # Check test file exists
    test_file = TEST_FILE_MAP.get((algorithm, kernel_type), {}).get(test_type)
    if not test_file:
        return CellStatus(
            algorithm=algorithm,
            hardware=hardware,
            kernel_type=kernel_type,
            test_type=test_type,
            status="implemented",
            notes=f"No test file defined for {algorithm}/{kernel_type}/{test_type}",
        )

    file_exists = check_file_exists(test_file)
    if not file_exists:
        return CellStatus(
            algorithm=algorithm,
            hardware=hardware,
            kernel_type=kernel_type,
            test_type=test_type,
            status="implemented",
            test_file=test_file,
            notes=f"Test file missing: {test_file}",
        )

    # Test file exists
    if run_tests:
        passed, output = run_test_file(test_file, f"{algorithm} and {hardware}")
        if passed:
            return CellStatus(
                algorithm=algorithm,
                hardware=hardware,
                kernel_type=kernel_type,
                test_type=test_type,
                status="passing",
                test_file=test_file,
                test_result=output[:200],
            )
        else:
            return CellStatus(
                algorithm=algorithm,
                hardware=hardware,
                kernel_type=kernel_type,
                test_type=test_type,
                status="failing",
                test_file=test_file,
                test_result=output[:500],
            )
    else:
        return CellStatus(
            algorithm=algorithm,
            hardware=hardware,
            kernel_type=kernel_type,
            test_type=test_type,
            status="tested",
            test_file=test_file,
            notes="Test file exists (not executed)",
        )


def generate_report(run_tests: bool = False) -> CoverageReport:
    """Generate full coverage report."""
    from datetime import datetime

    cells = []

    print(
        f"Analyzing {len(ALGORITHMS)} algorithms × {len(HARDWARE_TARGETS)} hardware × {len(KERNEL_TYPES)} kernel types × {len(TEST_TYPES)} test types..."
    )
    print(
        f"Total theoretical cells: {len(ALGORITHMS) * len(HARDWARE_TARGETS) * len(KERNEL_TYPES) * len(TEST_TYPES)}"
    )

    for algo in ALGORITHMS:
        for hw in HARDWARE_TARGETS:
            for kt in KERNEL_TYPES:
                for tt in TEST_TYPES:
                    cell = analyze_cell(algo, hw, kt, tt, run_tests)
                    cells.append(cell)

    # Count statistics
    total = len(cells)
    supported = sum(1 for c in cells if c.status != "not_supported")
    implemented = sum(
        1 for c in cells if c.status in ("implemented", "tested", "passing", "failing")
    )
    tested = sum(1 for c in cells if c.status in ("tested", "passing", "failing"))
    passing = sum(1 for c in cells if c.status == "passing")

    return CoverageReport(
        timestamp=datetime.now().isoformat(),
        total_cells=total,
        supported_cells=supported,
        implemented_cells=implemented,
        tested_cells=tested,
        passing_cells=passing,
        cells=cells,
    )


def print_summary(report: CoverageReport) -> None:
    """Print human-readable summary."""
    print("\n" + "=" * 80)
    print("PERMUTATION MATRIX COVERAGE REPORT")
    print("=" * 80)
    print(f"Timestamp: {report.timestamp}")
    print(f"Total cells: {report.total_cells}")
    print(
        f"Supported:   {report.supported_cells:4d} ({report.supported_cells / report.total_cells * 100:.1f}%)"
    )
    print(
        f"Implemented: {report.implemented_cells:4d} ({report.implemented_cells / report.total_cells * 100:.1f}%)"
    )
    print(
        f"Tested:      {report.tested_cells:4d} ({report.tested_cells / report.total_cells * 100:.1f}%)"
    )
    print(
        f"Passing:     {report.passing_cells:4d} ({report.passing_cells / report.total_cells * 100:.1f}%)"
    )
    print("=" * 80)

    # Per-algorithm summary
    print("\nPer-Algorithm Summary:")
    print("-" * 80)
    for algo in ALGORITHMS:
        algo_cells = [c for c in report.cells if c.algorithm == algo]
        if not algo_cells:
            continue
        sup = sum(1 for c in algo_cells if c.status != "not_supported")
        imp = sum(
            1
            for c in algo_cells
            if c.status in ("implemented", "tested", "passing", "failing")
        )
        tst = sum(1 for c in algo_cells if c.status in ("tested", "passing", "failing"))
        pas = sum(1 for c in algo_cells if c.status == "passing")
        total_algo = len(algo_cells)
        print(
            f"  {algo:15s}: {sup:3d}/{total_algo:3d} supported, {imp:3d} implemented, {tst:3d} tested, {pas:3d} passing"
        )

    # Per-hardware summary
    print("\nPer-Hardware Summary:")
    print("-" * 80)
    for hw in HARDWARE_TARGETS:
        hw_cells = [c for c in report.cells if c.hardware == hw]
        if not hw_cells:
            continue
        sup = sum(1 for c in hw_cells if c.status != "not_supported")
        imp = sum(
            1
            for c in hw_cells
            if c.status in ("implemented", "tested", "passing", "failing")
        )
        tst = sum(1 for c in hw_cells if c.status in ("tested", "passing", "failing"))
        pas = sum(1 for c in hw_cells if c.status == "passing")
        total_hw = len(hw_cells)
        print(
            f"  {hw:15s}: {sup:3d}/{total_hw:3d} supported, {imp:3d} implemented, {tst:3d} tested, {pas:3d} passing"
        )


def print_failing(report: CoverageReport) -> None:
    """Print failing cells."""
    failing = [c for c in report.cells if c.status == "failing"]
    if failing:
        print("\nFailing Cells:")
        print("-" * 80)
        for c in failing:
            print(
                f"  {c.algorithm:12s} | {c.hardware:12s} | {c.kernel_type:12s} | {c.test_type:15s}"
            )
            if c.test_result:
                print(f"    Error: {c.test_result[:200]}")


def save_json(report: CoverageReport, output_path: Path) -> None:
    """Save report as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Path(output_path).open("w") as f:
        json.dump(asdict(report), f, indent=2)
    print(f"\nReport saved to {output_path}")


def save_markdown(report: CoverageReport, output_path: Path) -> None:
    """Save report as Markdown table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Path(output_path).open("w") as f:
        f.write("# Permutation Matrix Coverage Report\n\n")
        f.write(f"Generated: {report.timestamp}\n\n")
        f.write(f"**Total cells:** {report.total_cells}\n")
        f.write(
            f"**Supported:** {report.supported_cells} ({report.supported_cells / report.total_cells * 100:.1f}%)\n"
        )
        f.write(
            f"**Implemented:** {report.implemented_cells} ({report.implemented_cells / report.total_cells * 100:.1f}%)\n"
        )
        f.write(
            f"**Tested:** {report.tested_cells} ({report.tested_cells / report.total_cells * 100:.1f}%)\n"
        )
        f.write(
            f"**Passing:** {report.passing_cells} ({report.passing_cells / report.total_cells * 100:.1f}%)\n\n"
        )

        # Status legend
        f.write("## Status Legend\n\n")
        f.write("| Status | Meaning |\n")
        f.write("|--------|---------|\n")
        f.write("| `supported` | Hardware target supported by algorithm |\n")
        f.write("| `implemented` | Test file exists but not executed |\n")
        f.write("| `tested` | Test file exists, tests discovered |\n")
        f.write("| `passing` | All tests pass |\n")
        f.write("| `failing` | Some tests fail |\n")
        f.write("| `not_supported` | Algorithm doesn't support this hardware |\n\n")

        # Per-algorithm tables
        for algo in ALGORITHMS:
            algo_cells = [c for c in report.cells if c.algorithm == algo]
            if not algo_cells:
                continue

            f.write(f"## {algo.upper()}\n\n")
            f.write("| Hardware | Kernel Type | Test Type | Status | Test File |\n")
            f.write("|----------|-------------|-----------|--------|-----------|\n")

            for c in sorted(
                algo_cells, key=lambda x: (x.hardware, x.kernel_type, x.test_type)
            ):
                test_file = c.test_file or "-"
                f.write(
                    f"| {c.hardware} | {c.kernel_type} | {c.test_type} | {c.status} | {test_file} |\n"
                )
            f.write("\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify permutation matrix coverage for bioplausible kernels"
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Actually run tests (slow but accurate)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/permutation_coverage.json"),
        help="Output JSON report path",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("artifacts/permutation_coverage.md"),
        help="Output Markdown report path",
    )
    parser.add_argument("--algorithm", type=str, help="Filter to specific algorithm")
    parser.add_argument(
        "--hardware", type=str, help="Filter to specific hardware target"
    )

    args = parser.parse_args()

    # Generate report
    report = generate_report(run_tests=args.run_tests)

    # Filter if requested
    if args.algorithm:
        report.cells = [c for c in report.cells if c.algorithm == args.algorithm]
    if args.hardware:
        report.cells = [c for c in report.cells if c.hardware == args.hardware]

    # Recalculate stats after filtering
    if args.algorithm or args.hardware:
        report.total_cells = len(report.cells)
        report.supported_cells = sum(
            1 for c in report.cells if c.status != "not_supported"
        )
        report.implemented_cells = sum(
            1
            for c in report.cells
            if c.status in ("implemented", "tested", "passing", "failing")
        )
        report.tested_cells = sum(
            1 for c in report.cells if c.status in ("tested", "passing", "failing")
        )
        report.passing_cells = sum(1 for c in report.cells if c.status == "passing")

    # Print and save
    print_summary(report)
    print_failing(report)
    save_json(report, args.output)
    save_markdown(report, args.markdown)

    # Exit code based on passing rate
    if report.passing_cells / max(1, report.supported_cells) < 0.8:
        print("\n⚠️  Coverage below 80% threshold")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
