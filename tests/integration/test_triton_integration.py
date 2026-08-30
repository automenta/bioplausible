"""Triton integration tests.

These tests are skipped because they were testing the legacy EquilibriumMLP
which used Triton kernels directly. The native models use a different
architecture (5-D ontology composition) and don't expose the same internal
methods (_initialize_hidden_state, _transform_input, forward_step).

If Triton acceleration is needed for native models, it would be implemented
at the Substrate or Geometry operator level, not at the model level.
"""

import pytest

pytestmark = pytest.mark.skip(
    reason="Legacy EquilibriumMLP with Triton kernels deleted. "
    "Native models use 5-D ontology composition; Triton acceleration "
    "would be at Substrate/Geometry level, not model level."
)


def test_triton_placeholder():
    """Placeholder test to keep test file valid."""
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
