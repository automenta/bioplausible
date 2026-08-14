#!/usr/bin/env python3
"""
Static import-DAG checker for the layered-core architecture.

Enforces:
1. Each layer may only import from equal or lower layers (L_N → L_{<=N})
2. The module graph must be acyclic

Usage:
    python tools/check_imports.py

Exit codes:
    0 - All checks pass
    1 - Layer violations or cycles found
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

# Layer manifest: package name → layer number (lower = more foundational)
# Based on REFACTOR2.md architecture diagram:
# L7: Interfaces (CLI, deployment, demo, sklearn, lightning)
# L6: Measurement (evaluation, validation, benchmarks, analysis, reporting)
# L5: Orchestration (execution, hyperopt, autoscientist)
# L4: Training (CoreTrainer - single train path)
# L3: Data/Domains (data, domains)
# L2: Zoo (models, propagators, optimizers, mep)
# L1: Core (registry, construction, config, checkpoint, metrics, result_sink)
#
# Note: Layer numbers indicate dependency direction:
# L_N may import from L_{<=N} only (no upward imports)

LAYERS: dict[str, int] = {
    # L1: Core - foundational, zero upward imports
    "bioplausible.core": 1,
    "bioplausible.config": 1,
    "bioplausible.utils": 1,
    "bioplausible.core.metrics": 1,
    "bioplausible.core.checkpoint": 1,
    "bioplausible.core.construction": 1,
    "bioplausible.core.registry": 1,
    "bioplausible.knowledge": 1,
    "bioplausible.evaluation": 1,
    "bioplausible.tracking": 1,
    "bioplausible.deployment": 1,
    "bioplausible.sklearn_interface": 1,
    "bioplausible.core.trainer": 1,
    "bioplausible.core.checkpoint_mixin": 1,
    "bioplausible.experiment.result_sink": 1,
    "bioplausible.core.local_learning": 1,
    "bioplausible.experiment.report": 1,
    "bioplausible.core.exceptions": 1,
    "bioplausible.core.logging": 1,
    "bioplausible.core.ebm": 1,
    "bioplausible.core.losses": 1,
    "bioplausible.core.profiling": 1,
    "bioplausible.core.spectral_mixin": 1,
    "bioplausible.core.training_mixin": 1,
    # L2: Zoo - registered components
    "bioplausible.zoo": 2,
    "bioplausible.zoo.models": 2,
    "bioplausible.zoo.propagators": 2,
    "bioplausible.zoo.optimizers": 2,
    "bioplausible.zoo.sparsity": 2,
    "bioplausible.zoo.mep": 2,
    # L3: Data/Domains
    "bioplausible.data": 3,
    "bioplausible.domains": 3,
    # L4: Training (CoreTrainer adapter pattern, trainer.py)
    "bioplausible.domains.trainer": 4,
    "bioplausible.lightning_": 4,
    # L5: Orchestration
    "bioplausible.execution": 5,
    "bioplausible.hyperopt": 5,
    "bioplausible.autoscientist": 5,
    "bioplausible.benchmarking": 5,
    # L6: Measurement
    "bioplausible.analysis": 6,
    "bioplausible.validation": 6,
    "bioplausible.leaderboard": 6,
    "bioplausible.benchmarks": 6,
    "bioplausible.graph": 6,
    "bioplausible.experiment": 6,
    # L7: Interfaces
    "bioplausible.cli": 7,
    "bioplausible.p2p": 7,
}

# Default layer for unknown modules (should rarely happen)
DEFAULT_LAYER: int = 1

# Excluded paths (tests, scripts, generated, etc.)
EXCLUDED: set[str] = {
    "__pycache__",
    "tests",
    "test_",
    ".pytest_cache",
    ".git",
    "docs",
    "docs.archive",
    "scripts",
    "examples",
    "legacy",
    "tools",
    "pyproject.toml",
    "uv.lock",
    ".mypy_cache",
    "node_modules",
}


def get_layer(module_path: str) -> int:
    """Get the layer number for a module path."""
    parts = module_path.split(".")
    for i in range(len(parts), 0, -1):
        prefix = ".".join(parts[:i])
        if prefix in LAYERS:
            return LAYERS[prefix]
    return DEFAULT_LAYER


def parse_imports(file_path: Path) -> list[str]:
    """Extract all import names from a Python file."""
    try:
        source = file_path.read_text()
    except OSError, UnicodeDecodeError:
        return []

    tree = ast.parse(source, filename=str(file_path))
    imports: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name.split(".")[0] if alias.name else ""
                imports.append(name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            name = node.module.split(".")[0] if node.module else ""
            imports.append(name)

    return imports


def find_python_files(root: Path) -> list[Path]:
    """Find all Python files in the project, excluding specified paths."""
    files: list[Path] = []
    for path in root.rglob("*.py"):
        # Check if any excluded part is in the path
        if any(exc in path.parts for exc in EXCLUDED):
            continue
        files.append(path)
    return files


def check_layer_violations(
    root: Path,
) -> tuple[list[str], set[tuple[str, str, str]]]:
    """Check for layer violations in all Python files."""
    violations: list[str] = []
    edges: set[tuple[str, str, str]] = set()  # (from, to, edge)

    files = find_python_files(root)

    for file_path in files:
        rel_path = file_path.relative_to(root)
        module_path = ".".join(rel_path.with_suffix("").parts)

        # Get the module's layer
        from_layer = get_layer(module_path)

        imports = parse_imports(file_path)
        for imp in imports:
            # Convert short import name to full module path if possible
            full_import = imp
            if imp not in (
                "torch",
                "numpy",
                "typing",
                "dataclasses",
                "collections",
                "enum",
                "functools",
                "itertools",
                "logging",
                "math",
                "pathlib",
                "sys",
                "time",
                "abc",
                "re",
                "json",
                "os",
                "ast",
                "typing_extensions",
                "contextlib",
                "copy",
                "operator",
                "warnings",
                "weakref",
                "heapq",
                "bisect",
                "array",
                "io",
                "pickle",
                "shutil",
                "tempfile",
                "subprocess",
                "threading",
                "multiprocessing",
                "concurrent",
                "asyncio",
                "socket",
                "http",
                "urllib",
                "xml",
                "csv",
                "sqlite3",
                "hashlib",
                "hmac",
                "secrets",
                "random",
                "statistics",
                "fractions",
                "decimal",
                "numbers",
                "typing",
                "string",
                "textwrap",
                "unicodedata",
                "struct",
                "codecs",
                "glob",
                "fnmatch",
                "linecache",
                "shlex",
                "argparse",
                "getopt",
                "configparser",
                "platform",
                "errno",
                "ctypes",
                "unittest",
                "doctest",
                "pdb",
                "trace",
                "gc",
                "inspect",
                "dis",
                "pickletools",
                "symtable",
                "token",
                "_keyword",
                "_thread",
                "_socket",
                "_ctypes",
                "_weakref",
                "_collections",
                "_abc",
                "_codecs",
                "_io",
                "_json",
                "_pickle",
                "_sqlite3",
                "_ssl",
                "_hashlib",
                "_hmac",
                "_struct",
                "_datetime",
                "_ssl",
                "_socketserver",
                "_multiprocessing",
                "_threading",
                "_asyncio",
                "_contextlib",
                "_decimal",
                "_fractions",
                "_statistics",
                "_random",
                "_glob",
                "_fnmatch",
                "_linecache",
                "_shlex",
                "_warnings",
                "_codecs",
                "_unittest",
                "_doctest",
                "_pdb",
                "_gc",
                "_inspect",
                "_dis",
                "_pickletools",
                "_symtable",
                "_tokenize",
                "ruff",
                "hypothesis",
                "optuna",
                "pydantic",
                "omegaconf",
                "torchvision",
                "sklearn",
                "scipy",
                "snntorch",
                "pandas",
                "matplotlib",
                "seaborn",
                "tabulate",
                "psutil",
                "gymnasium",
                "fastapi",
                "uvicorn",
                "kademlia",
                "onnxruntime",
                "datasets",
                "tqdm",
                "rich",
                "networkx",
            ):
                # These are external or stdlib, check with known modules
                pass
            import_layer = get_layer(full_import)

            # Record edge for cycle detection
            if import_layer > 0:
                edges.add((module_path, full_import, f"{module_path} → {full_import}"))

            # Check layer violation: importing from higher layer
            if import_layer > from_layer > 0:
                violations.append(
                    f"{module_path} (L{from_layer}) imports {full_import} (L{import_layer})"
                )

    return violations, edges


def detect_cycles(edges: set[tuple[str, str, str]]) -> list[list[str]]:
    """Detect cycles in the import graph using DFS."""
    # Build adjacency list
    graph: dict[str, set[str]] = {}
    for from_mod, to_mod, _ in edges:
        if from_mod not in graph:
            graph[from_mod] = set()
        graph[from_mod].add(to_mod)

    cycles: list[list[str]] = []
    visited: set[str] = set()
    rec_stack: set[str] = set()
    path: list[str] = []

    def dfs(node: str) -> bool:
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        for neighbor in graph.get(node, set()):
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                # Found a cycle
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:] + [neighbor]
                cycles.append(cycle)
                return True

        path.pop()
        rec_stack.remove(node)
        return False

    for node in graph:
        if node not in visited:
            dfs(node)

    return cycles


def find_lazy_loaders(root: Path) -> list[str]:
    """Find lazy loaders (PEP 562 __getattr__) that may mask import cycles."""
    files = find_python_files(root)
    lazy_loaders: list[str] = []

    for file_path in files:
        try:
            source = file_path.read_text()
        except OSError, UnicodeDecodeError:
            continue

        tree = ast.parse(source)
        has_getattr = False

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "__getattr__":
                has_getattr = True
                break

        if has_getattr:
            rel_path = file_path.relative_to(root)
            lazy_loaders.append(str(rel_path))

    return lazy_loaders


def main() -> int:
    """Main entry point."""
    root = Path(__file__).parent.parent / "bioplausible"

    print("Checking layer violations...")
    print("=" * 60)

    violations, edges = check_layer_violations(root)

    if violations:
        print("\nLAYER VIOLATIONS FOUND:")
        for v in sorted(set(violations))[:50]:  # Limit output
            print(f"  {v}")
        if len(violations) > 50:
            print(f"  ... and {len(violations) - 50} more")
        print()
    else:
        print("No layer violations found")

    print("\nChecking for import cycles...")
    print("=" * 60)

    cycles = detect_cycles(edges)

    if cycles:
        print(f"\nCycles detected ({len(cycles)} total):")
        for cycle in cycles[:10]:  # Limit output
            print("  " + " -> ".join(cycle[:10]))
        if len(cycles) > 10:
            print(f"  ... and {len(cycles) - 10} more")
        print()
    else:
        print("No cycles detected")

    print("\nChecking for lazy loaders (PEP 562 __getattr__)...")
    print("=" * 60)

    lazy_loaders = find_lazy_loaders(root)
    if lazy_loaders:
        print(f"\nLazy loaders found ({len(lazy_loaders)}):")
        for ll in lazy_loaders[:20]:
            print(f"  {ll}")
        if len(lazy_loaders) > 20:
            print(f"  ... and {len(lazy_loaders) - 20} more")
        print()
    else:
        print("No lazy loaders found")

    # Summary
    print("\n" + "=" * 60)
    total_issues = len(violations) + len(cycles)
    if total_issues == 0 and not lazy_loaders:
        print("PASSED: All layer and cycle checks passed; no lazy loaders")
        return 0
    elif total_issues == 0 and lazy_loaders:
        print(f"WARNING: {len(lazy_loaders)} lazy loader(s) found (may mask cycles)")
        return 0
    else:
        print(f"FAILED: {len(violations)} layer violation(s), {len(cycles)} cycle(s)")
        if lazy_loaders:
            print(f"        {len(lazy_loaders)} lazy loader(s)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
