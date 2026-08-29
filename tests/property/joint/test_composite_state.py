"""Property tests for CompositeState and JointTrajectoryRecorder."""

from __future__ import annotations

import torch

from computronium.core.joint import (
    CompositeState,
    StateRegistry,
    StateVariable,
)
from computronium.core.joint.state import JointTrajectoryRecorder


def test_composite_state_mutability():
    """CompositeState uses mutable dicts for activity/plastic/substrate."""
    z = CompositeState(
        activity={"x": torch.randn(4, 10)},
        plastic={"psi": torch.randn(4, 20)},
        substrate={"sigma": torch.randn(4, 5)},
    )

    # Should be able to mutate in-place
    z.activity["x"].zero_()
    z.plastic["psi"].fill_(1.0)
    z.substrate["sigma"] += 1.0

    assert torch.allclose(z.activity["x"], torch.zeros(4, 10))
    assert torch.allclose(z.plastic["psi"], torch.ones(4, 20))


def test_joint_trajectory_recorder_basic():
    """Test basic trajectory recording."""
    recorder = JointTrajectoryRecorder(max_steps=10)

    registry = StateRegistry()
    registry.register(StateVariable(name="x", persistent=True))
    registry.register(StateVariable(name="psi", fast_plastic=True))

    for i in range(5):
        z = CompositeState(
            activity={"x": torch.full((4, 10), float(i))},
            plastic={"psi": torch.full((4, 20), float(i * 2))},
            substrate={},
        )
        recorder.record(z)

    assert len(recorder) == 5

    traj = recorder.get_trajectory()
    assert len(traj) == 5
    assert torch.allclose(traj.activity[0]["x"], torch.zeros(4, 10))
    assert torch.allclose(traj.activity[4]["x"], torch.full((4, 10), 4.0))
    assert torch.allclose(traj.plastic[2]["psi"], torch.full((4, 20), 4.0))


def test_joint_trajectory_recorder_checkpointing():
    """Test gradient checkpointing interval."""
    recorder = JointTrajectoryRecorder(max_steps=100, checkpoint_interval=10)

    for i in range(25):
        z = CompositeState(
            activity={"x": torch.randn(4, 10)},
            plastic={},
            substrate={},
        )
        recorder.record(z)

    traj = recorder.get_trajectory()
    # Checkpoints at indices 9, 19 (0-indexed, every 10 steps)
    assert 9 in traj.checkpoint_indices
    assert 19 in traj.checkpoint_indices


def test_joint_trajectory_recorder_optional_plastic():
    """Test optional plastic/substrate recording."""
    recorder = JointTrajectoryRecorder(
        max_steps=10, record_plastic=False, record_substrate=False
    )

    for i in range(3):
        z = CompositeState(
            activity={"x": torch.randn(4, 10)},
            plastic={"psi": torch.randn(4, 20)},
            substrate={"sigma": torch.randn(4, 5)},
        )
        recorder.record(z)

    traj = recorder.get_trajectory()
    assert len(traj.activity) == 3
    assert len(traj.plastic) == 0
    assert len(traj.substrate) == 0


def test_joint_trajectory_reconstruct_step():
    """Test reconstructing CompositeState from trajectory."""
    recorder = JointTrajectoryRecorder(max_steps=10)

    for i in range(3):
        z = CompositeState(
            activity={"x": torch.full((4, 10), float(i))},
            plastic={"psi": torch.full((4, 20), float(i * 2))},
            substrate={"sigma": torch.full((4, 5), float(i * 3))},
        )
        recorder.record(z)

    traj = recorder.get_trajectory()

    # Reconstruct step 1
    z1 = traj.get_step(1)
    assert torch.allclose(z1.activity["x"], torch.full((4, 10), 1.0))
    assert torch.allclose(z1.plastic["psi"], torch.full((4, 20), 2.0))
    assert torch.allclose(z1.substrate["sigma"], torch.full((4, 5), 3.0))


def test_joint_trajectory_get_variable():
    """Test extracting single variable trajectory."""
    recorder = JointTrajectoryRecorder(max_steps=10)

    for i in range(4):
        z = CompositeState(
            activity={
                "x": torch.full((4, 10), float(i)),
                "y": torch.full((4, 2), float(i * 10)),
            },
            plastic={},
            substrate={},
        )
        recorder.record(z)

    traj = recorder.get_trajectory()

    x_traj = traj.get_activity("x")
    y_traj = traj.get_activity("y")

    assert len(x_traj) == 4
    assert torch.allclose(x_traj[2], torch.full((4, 10), 2.0))
    assert torch.allclose(y_traj[2], torch.full((4, 2), 20.0))


def test_joint_trajectory_max_steps():
    """Test max_steps limit."""
    recorder = JointTrajectoryRecorder(max_steps=3)

    for i in range(5):
        z = CompositeState(
            activity={"x": torch.randn(4, 10)},
            plastic={},
            substrate={},
        )
        recorder.record(z)

    assert len(recorder) == 3  # Capped at max_steps


def test_recorder_clear():
    """Test clearing recorded trajectory."""
    recorder = JointTrajectoryRecorder(max_steps=10)

    for i in range(3):
        z = CompositeState(
            activity={"x": torch.randn(4, 10)},
            plastic={},
            substrate={},
        )
        recorder.record(z)

    assert len(recorder) == 3
    recorder.clear()
    assert len(recorder) == 0
