"""Legacy CoreTrainer tests - skipped as CoreTrainer/TrainerConfig removed in Sprint 7."""

import pytest

pytestmark = pytest.mark.skip(
    reason="CoreTrainer/TrainerConfig removed in Sprint 7; uses new SystemTrainer/ExperimentConfig"
)
