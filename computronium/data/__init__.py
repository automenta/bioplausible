"""Data package: loaders, curricula, and dataset utilities."""

from computronium.data.curricula import (
    AntiCurriculum,
    Curriculum,
    CurriculumScheduler,
    FixedCurriculum,
    ProgressiveCurriculum,
)
from computronium.data.lm import (
    CharacterTokenizer,
    LMDataset,
    create_shakespeare_dataset,
    get_lm_dataset,
)
from computronium.data.vision import (
    CharDataset,
    create_data_loaders,
    get_vision_dataset,
)

__all__ = [
    "AntiCurriculum",
    "CharDataset",
    "CharacterTokenizer",
    "Curriculum",
    "CurriculumScheduler",
    "FixedCurriculum",
    "LMDataset",
    "ProgressiveCurriculum",
    "create_data_loaders",
    "create_shakespeare_dataset",
    "get_lm_dataset",
    "get_vision_dataset",
]
