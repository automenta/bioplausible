"""
P2P State Persistence.

Saves user contribution points and job counts.
"""

import json
import logging
from pathlib import Path

__all__ = [
    "STATE_FILE",
    "load_state",
    "logger",
    "save_state",
]
logger = logging.getLogger(__name__)

STATE_FILE = Path("results/p2p_state.json")

logger = logging.getLogger(__name__)


def load_state():
    if not STATE_FILE.exists():
        return {"points": 0, "jobs_done": 0}

    try:
        with Path(STATE_FILE).open("r") as f:
            return json.load(f)
    except Exception:
        logger.warning("Failed to load P2P state, returning defaults")
        return {"points": 0, "jobs_done": 0}


def save_state(points, jobs_done):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Path(STATE_FILE).open("w") as f:
            json.dump({"points": points, "jobs_done": jobs_done}, f)
    except Exception:
        logger.exception("Failed to save P2P state")
