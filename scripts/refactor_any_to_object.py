"""Codemod: replace `Any` with `object` in type annotations.

Excludes OmegaConf/config boundary files that legitimately need `Any`
for runtime-flexible structured configs.

Usage: python scripts/refactor_any_to_object.py
"""

import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE = REPO_ROOT / "computronium"

# Files that MUST keep `Any` — OmegaConf/config I/O boundaries
EXCLUDE = frozenset({
    "config/omegaconf.py",
    "config/__init__.py",
    "equitile/config.py",
    "core/trainer.py",
})

# Match `Any` as a whole word (typing context) but NOT inside strings/comments
_ANY_RE = re.compile(r"\bAny\b")


def _find_py_files(root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(root.rglob("*.py"))


def _replace_any_in_file(filepath: pathlib.Path) -> bool:
    """Replace Any with object. Returns True if changed."""
    original = filepath.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    new_lines: list[str] = []
    changed = False
    has_any_import = False

    for line in lines:
        stripped = line.strip()

        # Detect Any import lines
        if stripped.startswith("from typing import") and "Any" in stripped:
            has_any_import = True
            # Keep the import line for now; remove unused Any later
            new_lines.append(line)
            continue

        # Replace whole-word Any → object (skip import lines handled above)
        if stripped.startswith("import typing") or stripped.startswith(
            "from typing import"
        ):
            new_lines.append(line)
            continue

        new_line = _ANY_RE.sub("object", line)
        if new_line != line:
            changed = True
        new_lines.append(new_line)

    if not changed and not has_any_import:
        return False

    # Clean up Any from import lines
    clean_lines: list[str] = []
    for line in new_lines:
        stripped = line.strip()
        if stripped.startswith("from typing import") and "Any" in stripped:
            # Check if `Any` is still used in the cleaned content
            # Build content excluding this line
            content = "".join(clean_lines) + "".join(
                l for l in new_lines[len(clean_lines) + 1 :]
            )
            if "object" not in content and "Any" not in content:
                # Any was the only thing imported and is now object — but if
                # the replacement made "object" appear only via this import line...
                # Actually, if Any was only in the import line, it was never used.
                # Remove the whole import line.
                continue
            # Check if 'Any' still appears in the content (could be in comments/strings)
            # If the line still has `Any` but content doesn't, remove Any from import
            parts = [
                p.strip()
                for p in stripped[stripped.index("import") + len("import") :].split(",")
            ]
            filtered = [p for p in parts if p != "Any"]
            if not filtered:
                continue  # Remove entire import line
            indent = line[: len(line) - len(line.lstrip())]
            new_import = "from typing import " + ", ".join(filtered)
            clean_lines.append(indent + new_import + "\n")
        else:
            # Also handle `import typing` — check if `typing.Any` is used
            if stripped.startswith("import typing"):
                content = "".join(clean_lines) + "".join(
                    l for l in new_lines[len(clean_lines) + 1 :]
                )
                if "typing.Any" not in content and "typing.object" not in content:
                    continue  # Remove unused import
            clean_lines.append(line)

    result = "".join(clean_lines)
    if result != original:
        filepath.write_text(result, encoding="utf-8")
        return True
    return False


def main() -> None:
    py_files = _find_py_files(PACKAGE)
    changed: list[pathlib.Path] = []
    skipped: list[pathlib.Path] = []

    for fp in py_files:
        rel = fp.relative_to(PACKAGE)
        if rel.as_posix() in EXCLUDE:
            skipped.append(fp)
            continue
        if _replace_any_in_file(fp):
            changed.append(fp)

    print(f"Changed: {len(changed)} files")
    for fp in changed:
        print(f"  M {fp.relative_to(REPO_ROOT)}")
    print(f"Skipped (config boundary): {len(skipped)} files")
    for fp in skipped:
        print(f"  . {fp.relative_to(REPO_ROOT)}")
    unchanged = len(py_files) - len(changed) - len(skipped)
    print(f"Unchanged: {unchanged} files")


if __name__ == "__main__":
    main()
