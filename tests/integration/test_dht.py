import logging
import time
import unittest

import pytest

from bioplausible.p2p.dht import DHTNode

# Configure logging to see output during tests
logging.basicConfig(level=logging.DEBUG)

# Deadline used by poll-based waits below (deterministic unlike blind sleeps).
_POLL_DEADLINE = 5.0
_POLL_INTERVAL = 0.05


def _poll_get(node: DHTNode, key: str) -> object | None:
    """Retry `node.get(key)` until it returns non-None or the deadline elapses.

    `DHTNode.get` is synchronous (blocks until the kademlia RPC resolves) but
    cross-node propagation may lag a tick after a `set` returns, so we poll
    briefly instead of sleeping a fixed amount.
    """
    deadline = time.time() + _POLL_DEADLINE
    while time.time() < deadline:
        val = node.get(key)
        if val is not None:
            return val
        time.sleep(_POLL_INTERVAL)
    return None


def _poll_best_model(node: DHTNode, task: str) -> dict | None:
    """Same pattern for `get_best_model` (which delegates to `get`)."""
    deadline = time.time() + _POLL_DEADLINE
    while time.time() < deadline:
        val = node.get_best_model(task)
        if val is not None:
            return val
        time.sleep(_POLL_INTERVAL)
    return None


class TestDHT(unittest.TestCase):
    def setUp(self):
        # Only run if kademlia is installed
        try:
            import kademlia  # ruff: ignore[unused-import]
        except ImportError:
            self.skipTest("kademlia not installed")

    @pytest.mark.slow
    def test_dht_connectivity(self):
        # Create two nodes
        node1 = DHTNode(port=8470)
        node2 = DHTNode(port=8471, bootstrap_nodes=[("127.0.0.1", 8470)])

        try:
            # `start()` blocks until listen+bootstrap complete
            # (see DHTNode._run_loop), so no fixed sleep is needed.
            node1.start()
            node2.start()

            # Set on Node 1
            node1.set("test_key", {"data": "hello"})

            # Get on Node 2 (poll: the value may take a tick to propagate)
            val = _poll_get(node2, "test_key")

            self.assertIsNotNone(val)
            self.assertEqual(val.get("data"), "hello")

        finally:
            node2.stop()
            node1.stop()

    @pytest.mark.slow
    def test_best_model_propagation(self):
        node1 = DHTNode(port=8472)
        node2 = DHTNode(port=8473, bootstrap_nodes=[("127.0.0.1", 8472)])

        try:
            node1.start()
            node2.start()

            # Publish best model on Node 1
            config = {"model": "TestModel", "lr": 0.01}
            node1.publish_best_model("test_task", config, 0.95)

            # Retrieve on Node 2 (poll for propagation)
            best = _poll_best_model(node2, "test_task")
            self.assertIsNotNone(best)
            self.assertEqual(best["score"], 0.95)
            self.assertEqual(best["config"]["model"], "TestModel")

            # Try to publish worse model on Node 2 (should be ignored by
            # Node 1 logic if we implemented robust checks,
            # but currently DHT is simple KV, so it overwrites.
            # The implementation of publish_best_model does an optimistic check
            # *locally* before setting. Verify Node 2 checks locally before
            # overwriting.

            # Node 2 sees 0.95. Try to publish 0.90.
            # `publish_best_model` performs its own optimistic check
            # synchronously (get before set), so if publish returns the local
            # check has already completed and no further wait is required.
            node2.publish_best_model("test_task", config, 0.90)

            # Verify it is still 0.95
            # Note: The 'publish_best_model' logic:
            # current = self.get(key)
            # if current and current >= score: return

            best_after = node2.get_best_model("test_task")
            self.assertEqual(best_after["score"], 0.95)

        finally:
            node2.stop()
            node1.stop()


if __name__ == "__main__":
    unittest.main()
