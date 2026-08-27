#!/usr/bin/env python3
"""Lock Matrix Generator - discovers test_* in test_ontology_locks.py and generates matrix."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def extract_lock_tests(filepath: Path) -> dict[str, list[str]]:
    """Extract test functions and their lock labels from test file."""
    content = filepath.read_text()

    # Pattern to match test functions and classes
    test_pattern = r'(?:^    def (test_\w+)\(.*\):|^class (Test\w+):)'
    lock_labels = {
        'test_l1': 'L1',
        'test_l2': 'L2',
        'test_l3': 'L3',
        'test_l4': 'L4',
        'test_l5': 'L5',
        'test_l6': 'L6',
        'test_l7': 'L7',
        'test_l0': 'L0',
        'test_s_': 'S',
        'test_d_': 'D',
        'test_c_': 'C',
        'test_u_': 'U',
        'test_u_': 'U',
    }

    # Also check for class-based tests
    class_lock_map = {
        'TestL2OrthogonalityLock': 'L2',
        'TestL3LocalityLock': 'L3',
        'TestL4LyapunovLock': 'L4',
        'TestL5DeterminismLock': 'L5',  # parameterized
        'TestL6': 'L6',
        'TestU_EuclideanProperties': 'U',
        'TestC_BackpropCreditProperties': 'C',
        'TestC_SurrogateLocks': 'C',
        'TestC_TemporalTraceSTDP': 'C',
        'TestU_StepProperties': 'U',
    }

    locks = {label: [] for label in ['L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'S', 'G', 'D', 'C', 'U']}

    # Find class-based tests
    for match in re.finditer(r'^class (\w+):', content, re.MULTILINE):
        class_name = match.group(1)
        if class_name in class_lock_map:
            lock = class_lock_map[class_name]
            # Find methods in this class
            class_start = match.end()
            next_class = re.search(r'^class \w+:', content[class_start:], re.MULTILINE)
            class_end = class_start + next_class.start() if next_class else len(content)
            class_content = content[class_start:class_end]
            for method_match in re.finditer(r'^    def (test_\w+)\(.*?\):', class_content, re.MULTILINE | re.DOTALL):
                locks[lock].append(f"{class_name}::{method_match.group(1)}")

    # Find standalone test functions
    for match in re.finditer(r'^def (test_\w+)\(.*?\):', content, re.MULTILINE | re.DOTALL):
        test_name = match.group(1)
        # Determine lock from name
        assigned = False
        for prefix, label in lock_labels.items():
            if test_name.startswith(prefix):
                locks[label].append(test_name)
                assigned = True
                break
        if not assigned:
            locks.setdefault('Other', []).append(test_name)

    return locks


def generate_matrix(locks: dict[str, list[str]]) -> str:
    """Generate markdown lock matrix."""
    lines = [
        "# Correctness Lock Matrix",
        "",
        "| Lock | Tests | Count | Status |",
        "|------|-------|-------|--------|",
    ]

    total_tests = 0
    for label in ['L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'S', 'G', 'D', 'C', 'U']:
        tests = locks.get(label, [])
        count = len(tests)
        total_tests += count
        status = "✅" if count > 0 else "❌"
        test_list = ", ".join(tests[:3]) + (f" +{count-3} more" if count > 3 else "")
        lines.append(f"| {label} | {test_list} | {count} | {status} |")

    lines.extend([
        "",
        f"**Total Tests: {total_tests}**",
        "",
        "## Legend",
        "- L0: Config Schema Round-trip",
        "- L1: Parity Lock (Strangler Fig)",
        "- L2: Orthogonality Lock",
        "- L3: Locality Lock",
        "- L4: Lyapunov/Energy Lock",
        "- L5: Determinism Lock",
        "- L6: Round-trip & Totality Lock",
        "- L7: Seam Lock (P2P)",
        "- S: Substrate Axis",
        "- G: Geometry Axis",
        "- D: Dynamics Axis",
        "- C: Credit Axis",
        "- U: Update Axis",
    ])

    return "\n".join(lines)


def main():
    test_file = Path("tests/property/test_ontology_locks.py")
    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        sys.exit(1)

    locks = extract_lock_tests(test_file)
    matrix = generate_matrix(locks)

    output_file = Path("docs/CORRECTNESS_LOCK_MATRIX.md")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(matrix)
    print(f"Generated lock matrix: {output_file}")
    print(f"Total tests: {sum(len(v) for v in locks.values())}")

    # Print summary
    for label in ['L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'S', 'G', 'D', 'C', 'U']:
        count = len(locks.get(label, []))
        if count > 0:
            print(f"  {label}: {count} tests")


if __name__ == "__main__":
    main()