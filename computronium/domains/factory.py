"""
Task factory for the merged task hierarchy.

Moved from ``hyperopt/tasks.py`` during Phase 3.1 merge.
All concrete task classes now live in ``domains/``.
"""

import torch
from torch import nn

from computronium.core.logging import get_logger
from computronium.domains.trainer import TaskProtocol, _TaskTrainer

__all__ = [
    "CharNGramTask",
    "create_task",
    "logger",
]

logger = get_logger()


class CharNGramTask:
    """Synthetic task: Predict next character from previous N chars.

    Dataset: Deterministic repeating patterns or simple probabilistic grammar.
    Input: [B, SeqLen] (indices)
    Output: [B, VocabSize] (logits for last char)
    """

    def __init__(
        self,
        name: str = "char_ngram",
        device: str = "cpu",
        quick_mode: bool = False,
        vocab_size: int = 27,
        context_len: int = 3,
    ):
        self.name = name
        self.device = device
        self.quick_mode = quick_mode
        self.vocab_size = vocab_size
        self.context_len = context_len
        self._input_dim = context_len
        self._output_dim = vocab_size
        self.pattern = torch.arange(vocab_size)

    @property
    def task_type(self) -> str:
        return "lm"

    @property
    def input_dim(self) -> int | None:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def setup(self) -> None:
        pass

    def get_batch(
        self, split: str = "train", batch_size: int = 32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(0, self.vocab_size - self.context_len, (batch_size,))
        x_list: list[torch.Tensor] = []
        y_list: list[torch.Tensor] = []
        for s in starts:
            seq = (
                torch.arange(s.item(), s.item() + self.context_len + 1)
                % self.vocab_size
            )
            x_list.append(seq[:-1])
            y_list.append(seq[-1])
        x = torch.stack(x_list).to(self.device).float().unsqueeze(2)
        x = x.view(x.size(0), -1)
        y = torch.stack(y_list).to(self.device).long()
        return x, y

    def create_trainer(self, model: nn.Module, **kwargs) -> _TaskTrainer:
        kwargs.pop("device", None)
        return _TaskTrainer(model, self, device=self.device, **kwargs)

    def compute_metrics(
        self, logits: torch.Tensor, y: torch.Tensor, loss: float
    ) -> dict[str, float]:
        return {"loss": loss}


def _parse_split_digits(task_name: str) -> tuple[list[int] | None, str]:
    """Parse 'mnist_01' into ([0, 1], 'mnist'). Returns (None, task_name) if no digits."""
    if "_" not in task_name or not any(c.isdigit() for c in task_name):
        return None, task_name

    parts = task_name.split("_")
    digits: list[int] = []
    clean_name_parts: list[str] = []
    for p in parts:
        if p.isdigit():
            for d in p:
                digits.append(int(d))
        elif p == "split":
            continue
        else:
            clean_name_parts.append(p)

    if not digits:
        return None, task_name

    included_classes = sorted(set(digits))
    base_name = "_".join(clean_name_parts)
    if "mnist" in task_name and "mnist" not in base_name:
        base_name = "mnist"
    elif "cifar" in task_name and "cifar" not in base_name:
        base_name = "cifar10"

    return included_classes, base_name


def _normalize_vision_name(base_name: str) -> str:
    """Normalize a vision dataset name to the canonical key."""
    match base_name:
        case n if "kmnist" in n or "kuzushiji" in n:
            return "kmnist"
        case n if "cifar" in n:
            return "cifar100" if "100" in n else "cifar10"
        case n if "fashion" in n:
            return "fashion_mnist"
        case n if "digits" in n:
            return "digits"
        case n if "usps" in n:
            return "usps"
        case n if "svhn" in n:
            return "svhn"
        case n if "mnist" in n:
            return "mnist"
        case _:
            return base_name


def create_task(
    task_name: str, device: str = "cpu", quick_mode: bool = False, **kwargs
) -> TaskProtocol:
    """Factory function for tasks. Maps string names to Task classes via heuristics."""
    match task_name:
        case "char_ngram":
            return CharNGramTask(name=task_name, device=device, quick_mode=quick_mode)
        case "pendulum":
            from computronium.domains.rl import RLTask

            return RLTask(
                name=task_name, env_id="Pendulum-v1", device=str(device), **kwargs
            )
        case "acrobot":
            from computronium.domains.rl import RLTask

            return RLTask(
                name=task_name, env_id="Acrobot-v1", device=str(device), **kwargs
            )
        case "cartpole" | "rl":
            from computronium.domains.rl import RLTask

            return RLTask(
                name=task_name, env_id="CartPole-v1", device=str(device), **kwargs
            )
        case "shakespeare" | "tiny_shakespeare":
            from computronium.domains.lm import LMTask

            return LMTask(
                name=task_name,
                dataset_name="tiny_shakespeare",
                device=str(device),
                **kwargs,
            )
        case "xor" | "spiral" | "circles":
            from computronium.domains.vision import VisionTask

            return VisionTask(
                name=task_name,
                dataset_name=task_name,
                device=str(device),
                **kwargs,
            )
        case _:
            included_classes, base_name = _parse_split_digits(task_name)

    VISION_KEYWORDS = {"vision", "mnist", "cifar", "fashion", "digits", "usps", "svhn"}

    if any(kw in base_name for kw in VISION_KEYWORDS):
        from computronium.domains.vision import VisionTask

        name = _normalize_vision_name(base_name)
        return VisionTask(
            name=name,
            dataset_name=name,
            device=str(device),
            **kwargs,
        )

    match base_name:
        case "cora" | "pubmed" | "citeseer":
            from computronium.domains.graph import GraphTask

            return GraphTask(
                name=base_name, dataset_name=base_name, device=str(device), **kwargs
            )
        case "breast_cancer" | "california_housing" | "iris" | "wine":
            from computronium.domains.tabular import TabularTask

            return TabularTask(
                name=base_name, dataset_name=base_name, device=str(device), **kwargs
            )
        case _:
            logger.warning(
                "Unknown task '%s', defaulting to tiny_shakespeare LM", task_name
            )
            from computronium.domains.lm import LMTask

            return LMTask(
                name="tiny_shakespeare",
                dataset_name="tiny_shakespeare",
                device=str(device),
                **kwargs,
            )
