"""CLI entry point for biopl-hpo.

Thin shim that delegates to :mod:`computronium.cli.run` (the full experiment
runner). When invoked with no subcommand it defaults to ``search`` so that the
console-script behaves like a focused HPO tool.
"""

from __future__ import annotations

import sys

from computronium.cli.run import main as run_main

__all__ = [
    "main",
]


def main() -> None:
    """Entry point: default to the ``search`` subcommand when none is given."""
    if len(sys.argv) == 1:
        sys.argv.append("search")
    run_main()


if __name__ == "__main__":
    main()
