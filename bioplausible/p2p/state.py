"""
P2P State Persistence.

Saves user contribution points and job counts.
"""

import json
from pathlib import Path

from bioplausible.core.logging import get_logger

__all__ = [
    "STATE_FILE",
    "load_state",
    "logger",
    "save_state",
]
logger = get_logger()

STATE_FILE = Path("results/p2p_state.json")

logger = get_logger()


def load_state():
    if not STATE_FILE.exists():
        return {"points": 0, "jobs_done": 0}

    try:
        with Path(STATE_FILE).open("r") as f:
            return json.load(f)
    except OSError, ValueError, TypeError:
        logger.warning("Failed to load P2P state, returning defaults")
        return {"points": 0, "jobs_done": 0}


def save_state(points, jobs_done):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Path(STATE_FILE).open("w") as f:
            json.dump({"points": points, "jobs_done": jobs_done}, f)
    except OSError, ValueError, TypeError:
        logger.exception("Failed to save P2P state")
