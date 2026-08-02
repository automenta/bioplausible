"""Task loader tests (Sprint 3.3) — toy tasks only (no downloads)."""

from tasks import build_tasks


class TestTasks:
    def test_toy_tasks_sample(self):
        tasks = {t.name: t for t in build_tasks()}
        for name in ("xor", "spiral", "circles"):
            t = tasks[name]
            x, y = t.sample(16)
            assert x.shape[0] == 16
            assert x.shape[1] == t.input_dim
            assert y.shape[0] == 16

    def test_shapes_match_descriptor(self):
        for t in build_tasks():
            x, y = t.sample(8)
            assert x.shape[1] == t.input_dim
