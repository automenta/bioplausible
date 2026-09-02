"""README snippet drift lock (TODO10 R10.2.8).

The few designated README code blocks are checked verbatim against their
source demo tests: every nonblank line of a locked block must appear in the
source test file, in order, character-identical. A locked snippet that no
longer matches its test fails — no snippet/doc drift, by construction, with
minimal lock surface. Tables and prose are the hand-maintained index and are
deliberately not locked.

Locked blocks are declared in the sidecar map (``scripts/readme_snippets.json``)
and marked in README.md with ``<!-- lock: <block-id> -->`` above the fence.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
SNIPPET_MAP = Path(__file__).resolve().parent / "readme_snippets.json"

MARKER_PREFIX = "<!-- lock:"


def _readme_blocks(readme: str) -> dict[str, list[str]]:
    """Extract marked fenced python blocks: block id -> code lines."""
    blocks: dict[str, list[str]] = {}
    lines = readme.splitlines()
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith(MARKER_PREFIX) and stripped.endswith("-->"):
            block_id = stripped.removeprefix(MARKER_PREFIX).removesuffix("-->").strip()
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```python"):
                j += 1
            j += 1
            code: list[str] = []
            while j < len(lines) and not lines[j].startswith("```"):
                code.append(lines[j])
                j += 1
            blocks[block_id] = code
            i = j
        i += 1
    return blocks


def _is_ordered_subsequence(needle: list[str], haystack: list[str]) -> bool:
    """Every needle line must appear, stripped, in order in the haystack."""
    it = iter(line.strip() for line in haystack)
    return all(line.strip() in it for line in needle)


def check_readme_snippets(
    readme_path: Path = README, map_path: Path = SNIPPET_MAP
) -> list[str]:
    """Return one message per drift found; empty list means locked and clean."""
    errors: list[str] = []
    declared: dict[str, str] = json.loads(map_path.read_text(encoding="utf-8"))
    blocks = _readme_blocks(readme_path.read_text(encoding="utf-8"))

    marked = set(blocks)
    for block_id in marked - set(declared):
        errors.append(f"README block {block_id!r} is marked but not in the sidecar map")
    for block_id in set(declared) - marked:
        errors.append(f"sidecar map block {block_id!r} has no `<!-- lock: -->` marker")

    for block_id, code in blocks.items():
        source = declared.get(block_id)
        if source is None:
            continue
        test_lines = Path(source).read_text(encoding="utf-8").splitlines()
        code_lines = [line for line in code if line.strip()]
        if not _is_ordered_subsequence(code_lines, test_lines):
            errors.append(
                f"README block {block_id!r} drifted from its source test "
                f"{source!r}; update the test and the block together"
            )
    return errors


def main() -> int:
    errors = check_readme_snippets()
    for error in errors:
        print(f"readme-snippet-lock: {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
