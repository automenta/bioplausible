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
            if t.downloads:
                continue  # real-data samplers need network/data; covered elsewhere
            x, y = t.sample(8)
            assert x.shape[1] == t.input_dim

    def test_new_task_dims_declared(self):
        tasks = {t.name: t for t in build_tasks()}
        assert tasks["cifar10"].input_dim == 3072
        assert tasks["cifar10"].output_dim == 10
        assert tasks["tiny_shakespeare"].input_dim == 16
        assert tasks["tiny_shakespeare"].kind == "lm"

    def test_download_flags_only_on_real_data(self):
        by_kind = {}
        for t in build_tasks():
            by_kind.setdefault(t.kind, []).append(t)
        for t in by_kind.get("toy", []) + by_kind.get("digits", []):
            assert not t.downloads
        for t in (
            by_kind.get("mnist", [])
            + by_kind.get("cifar10", [])
            + by_kind.get("lm", [])
        ):
            assert t.downloads is True
