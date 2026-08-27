"""Add ``__all__`` to public Python modules missing it.

Scans ``computronium/`` (excluding ``docs/``, ``tests/``, ``_``-prefixed dirs
and files) and inserts ``__all__ = ["Name1", "Name2", ...]`` after the last
top-level import.  Only adds ``__all__`` to files that don't already have one.

Safe to run multiple times.

Usage:
    python scripts/add_all_exports.py
"""

import ast
import pathlib
import re

PKG = pathlib.Path("computronium")
EXCLUDE_DIRS = {"__pycache__", "lm_demo"}
SKIP_HAS_ALL = re.compile(r"^\s*__all__\s*=", re.MULTILINE)


def _is_public(path: pathlib.Path) -> bool:
    for part in path.parts:
        if part.startswith("_") and part not in ("__init__", "__main__"):
            return False
        if part in EXCLUDE_DIRS:
            return False
    return True


def _module_public_names(source: str) -> list[str]:
    """Return public names defined at the module top level (not inside functions/classes)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            names.append(node.name)
        elif isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    names.append(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and not node.target.id.startswith("_"):
                names.append(node.target.id)
    return names


def _find_last_import_line(lines: list[str]) -> int:
    """Find the index of the last line that is part of a top-level import.

    Uses AST to find the last top-level import statement, then finds the
    actual last line of that statement (handles multiline imports properly).
    """
    source = "\n".join(lines)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return -1

    # Only consider top-level imports — direct children of Module
    last_import = None
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last_import = node

    if last_import is None:
        return -1

    # Find the last line of the last import statement (0-indexed)
    last_line = last_import.end_lineno - 1
    # Skip any blank / comment lines that follow
    while last_line + 1 < len(lines):
        stripped = lines[last_line + 1].strip()
        if stripped == "" or stripped.startswith("#"):
            last_line += 1
        else:
            break
    return last_line


def _insert_all(source: str, names: list[str]) -> str:
    """Insert ``__all__`` after the last top-level import."""
    lines = source.splitlines()
    insert_after = _find_last_import_line(lines)

    if insert_after < 0:
        # No imports — find first non-blank, non-docstring line
        insert_after = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(('"""', "'''")):
                # Skip docstring — find its end
                for j in range(i + 1, len(lines)):
                    if stripped in lines[j]:
                        insert_after = j + 1
                        break
                continue
            insert_after = i - 1
            break

    # Skip blank lines after insert point
    while insert_after + 1 < len(lines) and lines[insert_after + 1].strip() == "":
        insert_after += 1

    all_lines = ["", "__all__ = ["]
    for name in sorted(set(names)):
        all_lines.append(f'    "{name}",')
    all_lines.append("]")

    result = lines[: insert_after + 1] + all_lines + lines[insert_after + 1 :]
    return "\n".join(result)


def main() -> None:
    count = 0
    for pyfile in sorted(PKG.rglob("*.py")):
        rel = pyfile.relative_to(PKG)
        if not _is_public(rel):
            continue
        if SKIP_HAS_ALL.search(pyfile.read_text()):
            continue

        source = pyfile.read_text()
        names = _module_public_names(source)
        new_source = _insert_all(source, names)
        pyfile.write_text(new_source)
        plural = "names" if len(names) != 1 else "name"
        print(f"  {rel}  ({len(names)} {plural})")
        count += 1

    print(f"\nDone. Added __all__ to {count} files.")


if __name__ == "__main__":
    main()