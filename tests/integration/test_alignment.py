"""Legacy LoopedMLP tests - skipped as LoopedMLP removed in Sprint 9 (zoo facade collapse)."""

import pytest

pytestmark = pytest.mark.skip(
    reason="LoopedMLP removed in Sprint 9; use EquilibriumMLP or native 5-D compositions"
)
