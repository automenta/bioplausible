"""Drift lock on the designated README code blocks (TODO10 R10.2.8).

Every locked snippet must remain a verbatim, order-preserving extract of
its source demo test (sidecar map: ``scripts/readme_snippets.json``). A
locked snippet that no longer matches its test fails CI.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from readme_snippet_lock import check_readme_snippets


def test_readme_locked_snippets_match_their_demo_tests() -> None:
    errors = check_readme_snippets()
    assert not errors, errors
