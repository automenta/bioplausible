"""Unit tests for the cross-domain task registry.

Geometry is derived from the concrete ``DomainTask`` (via the domain factory),
so these tests assert the resolution contract and that every advertised name
resolves offline — covering all domains, not just vision.
"""

from __future__ import annotations

import pytest

from bioplausible.domains.registry import SUPPORTED_TASKS, TaskSpec, resolve_task


def test_every_supported_name_resolves_offline():
    resolved = {name: resolve_task(name) for name in SUPPORTED_TASKS}
    assert set(resolved) == set(SUPPORTED_TASKS)
    # No geometry may be degenerate.
    for name, spec in resolved.items():
        assert spec.name == name
        assert spec.input_dim > 0
        assert spec.output_dim > 0


def test_unknown_task_rejected():
    with pytest.raises(ValueError):
        resolve_task("does_not_exist")


def test_vision_geometry_matches_data():
    assert resolve_task("mnist").input_dim == 784
    assert resolve_task("mnist").output_dim == 10
    assert resolve_task("usps").input_dim == 256
    assert resolve_task("usps").output_dim == 10
    assert resolve_task("cifar10").input_dim == 3072
    assert resolve_task("cifar10").output_dim == 10
    assert resolve_task("digits").input_dim == 64
    assert resolve_task("digits").output_dim == 10


def test_toy_tasks_are_two_class_two_feature():
    assert resolve_task("xor").input_dim == 2
    assert resolve_task("xor").output_dim == 2


def test_geometry_derived_not_hardcoded_for_language():
    # Regressive: the real LMTask reports its own vocab/seq geometry — the
    # registry must reflect the task, not a hardcoded placeholder.
    spec = resolve_task("tiny_shakespeare")
    assert spec.input_dim > 16  # not the old hardcoded 16/16
    assert spec.output_dim > 16


def test_cross_domain_coverage():
    names = SUPPORTED_TASKS
    # Registry spans vision, language, RL, and tabular — not just vision.
    assert {"mnist", "tiny_shakespeare", "cartpole", "breast_cancer"} <= names


def test_task_spec_is_frozen_and_ownable():
    spec = TaskSpec(name="xor", input_dim=2, output_dim=2)
    with pytest.raises(Exception):
        spec.input_dim = 3  # frozen dataclass
