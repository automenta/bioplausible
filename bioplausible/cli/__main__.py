"""Bioplausible CLI dispatcher (``biopl``).

Single public command surface over the Pillar-K thin adapters. Every top-level
command maps to one module ``main``; the console-script table in
``pyproject.toml`` points at this entry point so the public API boundary stays
one place.

Usage::

    biopl <run|report|parity|repro|hpo|audit|frontier|rank|lab> [args]
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# command -> (module, attribute). Resolved lazily to keep the import graph
# shallow: the dispatcher itself must not drag in the zoo/execution layer.
_SUBCOMMANDS: dict[str, tuple[str, str]] = {
    "run": ("bioplausible.cli.run", "main"),
    "report": ("bioplausible.experiment.cli", "main_report"),
    "parity": ("bioplausible.cli.parity", "main"),
    "repro": ("bioplausible.cli.repro", "main"),
    "hpo": ("bioplausible.cli.hpo", "main"),
    "audit": ("bioplausible.core.audit", "main"),
    "frontier": ("bioplausible.cli.frontier", "main"),
    "rank": ("bioplausible.cli.rank", "main"),
    "lab": ("bioplausible.cli.lab", "main"),
    "validate": ("bioplausible.cli.validate", "main"),
    "joint-validate": ("bioplausible.cli.joint_validate", "main"),
}

_USAGE = "biopl <" + "|".join(_SUBCOMMANDS) + "> [args]"


def _load(command: str) -> Callable[[], int]:
    module_name, attr = _SUBCOMMANDS[command]
    module = __import__(module_name, fromlist=[attr])
    return getattr(module, attr)


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch to the sub-command's module ``main``.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``). The first element
            selects the command; the remainder are forwarded unchanged so each
            adapter parses its own flags.

    Returns:
        The adapter's exit code (``0`` when it returns ``None``).
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(_USAGE)
        return 0 if args and args[0] in {"-h", "--help"} else 1

    command, rest = args[0], args[1:]
    if command not in _SUBCOMMANDS:
        print(f"biopl: unknown command {command!r}\n{_USAGE}")
        return 2

    # Each adapter's argparse reads sys.argv[1:] when called with no explicit
    # argv, so rewrite it to look like the command was invoked directly.
    sys.argv = [f"biopl {command}", *rest]
    try:
        return int(_load(command)() or 0)
    except SystemExit as exc:
        return int(exc.code or 0)


if __name__ == "__main__":
    sys.exit(main())
