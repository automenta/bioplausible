"""Test worker module."""

from __future__ import annotations

def worker(pipe):
    pipe.send("hello from worker")
    pipe.close()