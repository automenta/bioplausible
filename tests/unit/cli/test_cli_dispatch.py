"""Tests for the unified ``biopl`` CLI dispatcher (Pillar K).

Guards the single public command surface: every documented sub-command maps to
its adapter module, unknown commands fail loudly, and ``--help``/no-args exit
cleanly without importing the heavy zoo/execution layer.
"""

import pytest

from bioplausible.cli.__main__ import _SUBCOMMANDS, main


def test_expected_subcommands_present():
    assert set(_SUBCOMMANDS) == {
        "run",
        "report",
        "parity",
        "repro",
        "hpo",
        "audit",
        "frontier",
        "rank",
        "lab",
    }


def test_unknown_command_exits_nonzero(capsys):
    assert main(["not-a-command"]) == 2
    out = capsys.readouterr().out
    assert "unknown command" in out
    assert "run" in out and "rank" in out


def test_no_command_exits_nonzero(capsys):
    assert main([]) == 1
    assert "biopl <" in capsys.readouterr().out


def test_help_is_zero(capsys):
    assert main(["--help"]) == 0
    assert "biopl <" in capsys.readouterr().out


@pytest.mark.parametrize(
    "command",
    ["run", "report", "parity", "repro", "hpo", "audit", "frontier", "rank", "lab"],
)
def test_each_subcommand_help_exits_zero(command):
    assert main([command, "--help"]) == 0


def test_adapter_sees_passthrough_args(monkeypatch):
    """The adapter's argparse reads sys.argv[1:] after the command is stripped."""
    from bioplausible.cli import parity

    captured: dict[str, object] = {}

    def fake_main(argv=None):
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(parity, "main", fake_main)
    assert main(["parity", "--task", "xor"]) == 0
    # The dispatcher forwards args via sys.argv, not the argv param.
    assert captured["argv"] is None
    assert sys_argv_after() == ["biopl parity", "--task", "xor"]


def sys_argv_after():
    import sys

    return sys.argv
