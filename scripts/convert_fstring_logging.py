"""Convert f-string logging calls to ``%s``-style deferred formatting.

Replaces patterns like::

    logger.info(f"Task {name} failed: {e}")

with::

    logger.info("Task %s failed: %s", name, e)

This ensures deferred interpolation (safe, matches AGENTS.md mandate).

Usage:
    python scripts/convert_fstring_logging.py
"""

import pathlib
import re

PKG = pathlib.Path("computronium")
EXCLUDE_DIRS = {"__pycache__", "lm_demo"}

# Match a single-line logger f-string call
FSTRING_LOG_RE = re.compile(
    r"""
    (logger\.(?:info|debug|warning|error|critical|exception)\s*\(\s*)
    f(["'])
    (.*?)
    \2
    (\s*,\s*exc_info\s*=\s*True)?
    (\s*\)
    )
    """,
    re.VERBOSE,
)

# Match {expr} inside f-strings, including format specs
FSTRING_BRACE_RE = re.compile(r"\{([^}]+)\}")


def _parse_fstring_expr(expr: str) -> str:
    """Extract the Python expression from an f-string replacement field.

    Handles format specs (``{val:.2f}``) and conversions (``{val!r}``).
    """
    expr = expr.strip()
    # Remove format spec after `:`
    if ":" in expr:
        expr = expr.split(":")[0]
    # Remove conversion after `!`
    if "!" in expr:
        expr = expr.split("!")[0]
    return expr.strip()


def _convert_fstring_to_percent(template: str) -> tuple[str, list[str]]:
    """Convert an f-string body to a ``%s`` format string and arg list."""
    args: list[str] = []
    result: list[str] = []
    last_end = 0
    for match in FSTRING_BRACE_RE.finditer(template):
        result.append(template[last_end : match.start()])
        expr = _parse_fstring_expr(match.group(1))
        args.append(expr)
        result.append("%s")
        last_end = match.end()
    result.append(template[last_end:])
    # Escape literal % signs, then restore %s placeholders
    fmt = "".join(result).replace("%", "%%")
    fmt = fmt.replace("%%s", "%s")
    return fmt, args


def _process_file(path: pathlib.Path) -> bool:
    """Process a single file. Returns True if modified."""
    source = path.read_text()
    modified = False
    new_lines: list[str] = []

    for line in source.splitlines():
        match = re.search(
            r"""
            (logger\.(?:info|debug|warning|error|critical|exception)\s*\(\s*)
            f(["'])
            (.*?)
            \2
            (\s*,\s*exc_info\s*=\s*True)?
            (\s*\)
            )
            """,
            line.strip(),
            re.VERBOSE,
        )
        if not match:
            new_lines.append(line)
            continue

        prefix = match.group(1)   # logger.info(
        _ = match.group(2)        # quote character
        body = match.group(3)     # f-string body
        exc_info_str = (match.group(4) or "").strip()  # , exc_info=True
        suffix = match.group(5)   # )

        fmt, args = _parse_fstring_args(body)
        if not args and not exc_info_str:
            # No args and no exc_info — just use plain string
            new_line = re.sub(r"logger\.\w+\(\s*\)", f'logger.info("{fmt}")', line)
            indent = re.match(r"^\s*", line).group()
            new_line = f'{indent}{prefix}"{fmt}")'
            new_lines.append(new_line)
            modified = True
            continue

        # Build: logger.info("fmt %s", arg1, arg2, exc_info=True)
        arg_list = ", ".join(args) if args else ""
        exc_part = exc_info_str if exc_info_str else ""

        if arg_list and exc_part:
            inner = f'"{fmt}", {arg_list}, {exc_part.removeprefix(",").strip()}'
        elif arg_list:
            inner = f'"{fmt}", {arg_list}'
        else:
            inner = f'"{fmt}", {exc_part.removeprefix(",").strip()}'

        new_line = f'{prefix}{inner}{suffix}'
        indent = re.match(r"^\s*", line).group()
        new_lines.append(indent + new_line)
        modified = True

    if modified:
        path.write_text("\n".join(new_lines))
    return modified


def _parse_fstring_args(body: str) -> tuple[str, list[str]]:
    """Parse f-string body into format string and argument list."""
    args: list[str] = []
    result: list[str] = []
    last_end = 0
    for match in FSTRING_BRACE_RE.finditer(body):
        result.append(body[last_end : match.start()])
        expr = _parse_fstring_expr(match.group(1))
        args.append(expr)
        result.append("%s")
        last_end = match.end()
    result.append(body[last_end:])
    fmt = "".join(result).replace("%", "%%").replace("%%s", "%s")
    return fmt, args


def main() -> None:
    count = 0
    for pyfile in sorted(PKG.rglob("*.py")):
        rel = pyfile.relative_to(PKG)
        if any(part in EXCLUDE_DIRS for part in pyfile.parts):
            continue
        if _process_file(pyfile):
            count += 1
            print(f"  Modified: {rel}")

    print(f"\nDone. Modified {count} files.")


if __name__ == "__main__":
    main()