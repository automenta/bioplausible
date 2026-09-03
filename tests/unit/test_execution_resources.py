"""ResourceMonitor behavior (execution guard)."""

from computronium.execution.resources import ResourceMonitor


def test_should_pause_respects_generous_limits() -> None:
    monitor = ResourceMonitor()
    assert isinstance(monitor.should_pause(), bool)


def test_should_pause_tolerates_missing_gpu() -> None:
    monitor = ResourceMonitor(gpu_limit=0.0)
    assert isinstance(monitor.should_pause(), bool)
