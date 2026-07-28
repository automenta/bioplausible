"""
Task Abstraction for Hyperopt and Experiments

Encapsulates data loading, batch generation, and evaluation logic for different tasks.
"""

import functools
import logging
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import nn

from bioplausible.data.lm import get_lm_dataset
from bioplausible.data.vision import get_vision_dataset

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=64)
def _load_vision_dataset_cached(
    name: str,
    device_str: str,
    quick_mode: bool,
    included_classes_tuple: tuple[int, ...] | None,
    fold: int | None,
    num_folds: int,
    data_fraction: float | None,
    augment: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple | int, int]:
    """Load and preprocess a vision dataset, cached to avoid reloading per trial."""
    device = torch.device(device_str)
    dataset = get_vision_dataset(
        name,
        train=True,
        flatten=False,
        included_classes=included_classes_tuple,
        augment=augment,
    )
    test_dataset = get_vision_dataset(
        name,
        train=False,
        flatten=False,
        included_classes=included_classes_tuple,
    )

    def load_to_tensor(ds):
        has_data_targets = hasattr(ds, "data") and hasattr(ds, "targets")
        has_data_labels = hasattr(ds, "data") and hasattr(ds, "labels")
        has_tensors = hasattr(ds, "tensors")

        use_bulk = included_classes_tuple is None and (
            has_data_targets or has_data_labels or has_tensors
        )

        if use_bulk:
            if has_tensors:
                raw_x, raw_y = ds.tensors
            else:
                raw_x = ds.data
                raw_y = ds.targets if has_data_targets else ds.labels

            if isinstance(raw_y, list):
                raw_y = torch.tensor(raw_y)
            if isinstance(raw_x, np.ndarray):
                raw_x = torch.from_numpy(raw_x)

            if raw_x.dtype == torch.uint8 or raw_x.dtype == np.uint8:
                raw_x = raw_x.float() / 255.0
            elif raw_x.dtype in [torch.float32, torch.float64, np.float32, np.float64]:
                if raw_x.max() > 1.0:
                    raw_x = raw_x / 255.0

            if raw_x.dim() == 3:
                raw_x = raw_x.unsqueeze(1)
                is_nhwc = False
            elif raw_x.dim() == 4:
                is_nhwc = raw_x.shape[3] in [1, 3] and raw_x.shape[1] not in [1, 3]
            else:
                is_nhwc = False

            if is_nhwc and not has_tensors:
                raw_x = raw_x.permute(0, 3, 1, 2).contiguous()

            raw_x = (raw_x - 0.5) / 0.5

            if not raw_x.is_contiguous():
                raw_x = raw_x.contiguous()

            if not isinstance(raw_y, torch.Tensor):
                raw_y = torch.tensor(raw_y)

            return raw_x.to(device), raw_y.to(device)
        else:
            loader = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=False)
            xs, ys = [], []
            for x, y in loader:
                xs.append(x)
                ys.append(y)
            return torch.cat(xs).to(device), torch.cat(ys).to(device)

    full_train_x, full_train_y = load_to_tensor(dataset)
    full_test_x, full_test_y = load_to_tensor(test_dataset)

    if fold is not None:
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        splits = list(kf.split(full_train_x))
        train_idx, val_idx = splits[fold]
        train_x = full_train_x[train_idx]
        train_y = full_train_y[train_idx]
        val_x = full_train_x[val_idx]
        val_y = full_train_y[val_idx]
    else:
        train_x = full_train_x
        train_y = full_train_y
        val_x = full_test_x
        val_y = full_test_y

    if quick_mode:
        n = min(100, len(train_x))
        train_x = train_x[:n]
        train_y = train_y[:n]
        val_x = val_x[:n]
        val_y = val_y[:n]

    if data_fraction is not None and 0.0 < data_fraction < 1.0:
        n = int(len(train_x) * data_fraction)
        train_x = train_x[:n]
        train_y = train_y[:n]
        val_x = val_x[:n]
        val_y = val_y[:n]

    if included_classes_tuple:
        output_dim = len(included_classes_tuple)
    else:
        output_dim = int(train_y.max().item() + 1)

    if train_x.dim() > 2:
        input_dim = tuple(train_x.shape[1:])
    else:
        input_dim = train_x.shape[1]

    return train_x, train_y, val_x, val_y, input_dim, output_dim


@runtime_checkable
class TaskProtocol(Protocol):
    """Structural interface for experiment tasks.

    All task classes (whether they inherit from ``BaseTask`` or not) should
    satisfy this protocol.  Type annotations should use ``TaskProtocol``
    instead of ``BaseTask`` to allow duck-typed task implementations.
    """

    name: str
    device: str
    quick_mode: bool

    @property
    def input_dim(self) -> int | None: ...

    @property
    def output_dim(self) -> int: ...

    @property
    def task_type(self) -> str: ...

    def setup(self) -> None: ...

    def get_batch(
        self, split: str = "train", batch_size: int = 32
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def create_trainer(self, model: nn.Module, **kwargs) -> object: ...

    def compute_metrics(
        self, logits: torch.Tensor, y: torch.Tensor, loss: float
    ) -> dict[str, float]: ...


def _resolve_task_loss(task: TaskProtocol) -> nn.Module:
    """Pick a torch loss module matching the task's output geometry.

    The `_TaskTrainer` codepath is the generic Protocol-based trainer used
    by ``run_from_runconfig`` for vision/lm/tabular experiment probes.
    Regression tasks (``task_type == "tabular"`` with ``output_dim == 1``
    — e.g. California Housing) emit float ``[B, 1]`` targets and must use
    MSELoss; everything else (vision/lm/discrete-tabular) treats the
    target as a class index and uses CrossEntropyLoss. Note: RL tasks
    bypass ``_TaskTrainer`` entirely via ``RLTask.create_trainer``
    returning ``RLTrainer``, so RL never flows through this resolver.
    """
    if task.task_type == "tabular" and task.output_dim == 1:
        return nn.MSELoss()
    return nn.CrossEntropyLoss()


class _TaskTrainer:
    """Lightweight task-protocol trainer for run_from_runconfig.

    Thin wrapper around ``CoreTrainer`` that delegates training to
    ``CoreTrainer.from_task()``.  The wrapper exists to preserve the
    ``train_*``-prefixed metric shape and inline validation behaviour
    expected by ``hyperopt`` callers.

    Used by ``BaseTask.create_trainer`` when called from
    ``CoreTrainer.run_from_runconfig``.
    """

    def __init__(
        self,
        model: nn.Module,
        task: TaskProtocol,
        device: str = "cpu",
        optimizer=None,
        epochs: int = 1,
        batches_per_epoch: int = 1,
        grad_clip: float | None = None,
        use_compile: bool = False,
        track_energy: bool = False,
        ablation_tags: dict | None = None,
        output_dir: str = "",
        **kwargs,
    ):
        from bioplausible.core.trainer import CoreTrainer

        self._trainer = CoreTrainer.from_task(
            model=model,
            task=task,
            device=device,
            optimizer=optimizer,
            epochs=epochs,
            batches_per_epoch=batches_per_epoch,
            grad_clip=grad_clip,
            use_compile=use_compile,
            track_energy=track_energy,
            ablation_tags=ablation_tags or {},
            output_dir=output_dir,
            batch_size=kwargs.pop("batch_size", 32),
            # NOTE: unknown **kwargs intentionally NOT forwarded — they may
            # contain non-config objects (tracker, safety_config, etc.) that
            # Crash OmegaConf. The old _TaskTrainer silently dropped them.
        )
        # Keep direct references for introspection / backward compat.
        self.model = model
        self.task = task
        self.epochs = epochs

    def train_epoch(self) -> dict[str, float]:
        """Run one epoch of training and return aggregated metrics.

        Delegates to ``CoreTrainer.train_epoch()`` then wraps the result
        with ``train_*`` metric prefixes and inline validation (matching
        the historical ``_TaskTrainer`` output shape).
        """
        import time

        epoch_t0 = time.time()

        raw = self._trainer.train_epoch()

        # Prefix train metrics to match historical _TaskTrainer shape.
        metrics: dict[str, float] = {}
        for k, v in raw.items():
            if k in ("loss", "accuracy"):
                metrics[f"train_{k}"] = v
            elif k == "samples_seen":
                continue
            else:
                metrics[k] = v
        metrics["loss"] = metrics.get("train_loss", 0.0)
        metrics["accuracy"] = metrics.get("train_accuracy", 0.0)

        # Inline validation — same contract as the original implementation:
        # NaN on failure, real values on success.
        metrics["val_loss"] = float("nan")
        metrics["val_accuracy"] = float("nan")
        try:
            val_raw = self._trainer._validate(1)
            metrics["val_loss"] = val_raw.get("val_loss", float("nan"))
            metrics["val_accuracy"] = val_raw.get("val_accuracy", float("nan"))
            if "val_perplexity" in val_raw:
                metrics["val_perplexity"] = val_raw["val_perplexity"]
        except (NotImplementedError, RuntimeError) as e:
            logger.warning("Validation skipped for %s: %s", self.task.name, e)

        metrics["time"] = time.time() - epoch_t0
        return metrics


class BaseTask(ABC):
    """Abstract base class for all tasks."""

    def __init__(self, name: str, device: str = "cpu", quick_mode: bool = False):
        self.name = name
        self.device = device
        self.quick_mode = quick_mode
        self._input_dim = None
        self._output_dim = None

    @abstractmethod
    def setup(self):
        """Load datasets and prepare for training."""

    @abstractmethod
    def get_batch(
        self, split: str = "train", batch_size: int = 32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of data."""

    @abstractmethod
    def create_trainer(self, model: nn.Module, **kwargs) -> _TaskTrainer:
        """Create a trainer specific to this task."""

    @property
    def input_dim(self) -> int | None:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    @abstractmethod
    def task_type(self) -> str:
        """Return 'lm', 'vision', or 'rl'."""

    def compute_metrics(
        self, logits: torch.Tensor, y: torch.Tensor, loss: float
    ) -> dict[str, float]:
        """Compute task-specific metrics."""
        return {"loss": loss}


class LMTask(BaseTask):
    """Language Modeling Task (Character level)."""

    def __init__(
        self,
        name: str = "tiny_shakespeare",
        device: str = "cpu",
        quick_mode: bool = False,
        seq_len: int = 64,
    ):
        super().__init__(name, device, quick_mode)
        self.seq_len = seq_len
        self.data_train = None
        self.data_val = None

    @property
    def task_type(self) -> str:
        return "lm"

    def setup(self):
        logger.info("Loading LM dataset: %s...", self.name)
        try:
            dataset = get_lm_dataset(self.name, seq_len=self.seq_len)
            data = dataset.data
            self._output_dim = dataset.vocab_size
            self._input_dim = None  # Uses embeddings

            # Split train/val
            n = int(0.9 * len(data))
            self.data_train = data[:n]
            self.data_val = data[n:]
            # Quick Mode Truncation
            if self.quick_mode:
                n_quick = min(len(self.data_train), 1000)
                self.data_train = self.data_train[:n_quick].clone()
                self.data_val = self.data_val[: min(len(self.data_val), 1000)].clone()

            logger.info(
                "Dataset ready: %d train, %d val tokens",
                len(self.data_train),
                len(self.data_val),
            )
        except Exception:
            logger.exception("Failed to load dataset %s", self.name)
            raise

    def get_batch(
        self, split: str = "train", batch_size: int = 32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.data_train is None:
            raise RuntimeError("Dataset not loaded. Call setup() first.")

        data = self.data_train if split == "train" else self.data_val
        idx = torch.randint(0, len(data) - self.seq_len - 1, (batch_size,))
        x = torch.stack([data[i : i + self.seq_len] for i in idx]).to(self.device)
        y = torch.stack([data[i + self.seq_len] for i in idx]).to(self.device)
        return x, y

    def create_trainer(self, model: nn.Module, **kwargs) -> _TaskTrainer:
        kwargs.pop("device", None)

        return _TaskTrainer(model, self, device=self.device, **kwargs)

    def compute_metrics(
        self, logits: torch.Tensor, y: torch.Tensor, loss: float
    ) -> dict[str, float]:
        if logits.dim() == 3:
            logits = logits[:, -1, :]

        acc = (logits.argmax(1) == y).float().mean().item()
        perplexity = np.exp(min(loss, 10))
        return {"loss": loss, "accuracy": acc, "perplexity": perplexity}


class VisionTask(BaseTask):
    """Vision Task (MNIST, CIFAR-10)."""

    def __init__(
        self,
        name: str = "mnist",
        device: str = "cpu",
        quick_mode: bool = False,
        included_classes: list | None = None,
        augment: bool = False,
        fold: int | None = None,
        num_folds: int = 5,
        data_fraction: float | None = None,
    ):
        super().__init__(name, device, quick_mode)
        self.train_x = None
        self.train_y = None
        self.val_x = None
        self.val_y = None
        self.included_classes = included_classes
        self.augment = augment
        self.fold = fold
        self.num_folds = num_folds
        self.data_fraction = data_fraction

    @property
    def task_type(self) -> str:
        return "vision"

    def setup(self):
        included_tuple = tuple(self.included_classes) if self.included_classes else None
        result = _load_vision_dataset_cached(
            name=self.name,
            device_str=str(self.device),
            quick_mode=self.quick_mode,
            included_classes_tuple=included_tuple,
            fold=self.fold,
            num_folds=self.num_folds,
            data_fraction=self.data_fraction,
            augment=self.augment,
        )
        (
            self.train_x,
            self.train_y,
            self.val_x,
            self.val_y,
            self._input_dim,
            self._output_dim,
        ) = result
        logger.info(
            "Loaded Vision dataset: %s (Fold=%s, Frac=%s)",
            self.name,
            self.fold,
            self.data_fraction,
        )

    def get_batch(
        self, split: str = "train", batch_size: int = 32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.train_x is None:
            raise RuntimeError("Dataset not loaded. Call setup() first.")

        if split == "train":
            dataset_x, dataset_y = self.train_x, self.train_y
        else:
            dataset_x, dataset_y = self.val_x, self.val_y

        if len(dataset_x) == 0:
            return torch.empty(0).to(self.device), torch.empty(0).to(self.device)

        idx = torch.randint(0, len(dataset_x), (batch_size,))
        x = dataset_x[idx]
        y = dataset_y[idx]
        return x, y

    def create_trainer(self, model: nn.Module, **kwargs) -> _TaskTrainer:
        kwargs.pop("device", None)

        return _TaskTrainer(model, self, device=self.device, **kwargs)

    def compute_metrics(
        self, logits: torch.Tensor, y: torch.Tensor, loss: float
    ) -> dict[str, float]:
        if logits.dim() == 3:
            logits = logits[:, -1, :]

        acc = (logits.argmax(1) == y).float().mean().item()
        return {"loss": loss, "accuracy": acc, "perplexity": 0.0}


class CharNGramTask(BaseTask):
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
        super().__init__(name, device, quick_mode)
        self.vocab_size = vocab_size
        self.context_len = context_len
        self._input_dim = context_len  # Since we flatten
        self._output_dim = vocab_size
        self.pattern = torch.arange(vocab_size)

    @property
    def task_type(self) -> str:
        return "lm"

    def setup(self):
        pass

    def get_batch(
        self, split: str = "train", batch_size: int = 32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(0, self.vocab_size - self.context_len, (batch_size,))
        x_list = []
        y_list = []
        for s in starts:
            seq = (
                torch.arange(s.item(), s.item() + self.context_len + 1)
            ) % self.vocab_size
            x_list.append(seq[:-1])
            y_list.append(seq[-1])
        x = torch.stack(x_list).to(self.device).float().unsqueeze(2)  # [B, L, 1]
        x = x.view(x.size(0), -1)  # Flatten [B, L*1] -> [B, L]

        y = torch.stack(y_list).to(self.device).long()
        return x, y

    def create_trainer(self, model: nn.Module, **kwargs) -> _TaskTrainer:
        kwargs.pop("device", None)

        return _TaskTrainer(model, self, device=self.device, **kwargs)


class RLTask(BaseTask):
    """Reinforcement Learning Task (CartPole)."""

    def __init__(
        self, name: str = "cartpole", device: str = "cpu", quick_mode: bool = False
    ):
        super().__init__(name, device, quick_mode)
        self.env_name = "CartPole-v1" if name == "cartpole" else name
        self.env = None

    @property
    def task_type(self) -> str:
        return "rl"

    def setup(self):
        import gymnasium as gym

        try:
            self.env = gym.make(self.env_name)

            # Determine Output Dim (Action Space)
            if hasattr(self.env.action_space, "n"):
                self._output_dim = self.env.action_space.n  # Discrete
            else:
                self._output_dim = self.env.action_space.shape[0]  # Continuous (Box)

            # Determine Input Dim (Observation Space)
            self._input_dim = self.env.observation_space.shape[0]

        except Exception:
            logger.exception("Failed to load env %s", self.env_name)
            raise

    def get_batch(self, split: str = "train", batch_size: int = 32):
        raise NotImplementedError(
            "RL Task does not support get_batch directly, use RLTrainer"
        )

    def create_trainer(self, model, **kwargs):
        from bioplausible.training.rl import RLTrainer

        # Filter relevant args for RLTrainer
        rl_args = {}

        # Map batches_per_epoch to episodes_per_epoch for RL
        if "batches_per_epoch" in kwargs and "episodes_per_epoch" not in kwargs:
            kwargs["episodes_per_epoch"] = kwargs["batches_per_epoch"]

        valid_keys = [
            "episodes",
            "lr",
            "gamma",
            "max_steps",
            "tracker",
            "episodes_per_epoch",
        ]
        for k in valid_keys:
            if k in kwargs:
                rl_args[k] = kwargs[k]

        return RLTrainer(model, self.env_name, device=self.device, **rl_args)


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
    # Handle special case where base name might be empty or partial
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
            return RLTask("Pendulum-v1", device, quick_mode)
        case "acrobot":
            return RLTask("Acrobot-v1", device, quick_mode)
        case "cartpole" | "rl":
            return RLTask("CartPole-v1", device, quick_mode)
        case "shakespeare" | "tiny_shakespeare":
            return LMTask(task_name, device, quick_mode)
        case _:
            included_classes, base_name = _parse_split_digits(task_name)

    VISION_KEYWORDS = {"vision", "mnist", "cifar", "fashion", "digits", "usps", "svhn"}

    if any(kw in base_name for kw in VISION_KEYWORDS):
        name = _normalize_vision_name(base_name)
        fold = kwargs.get("fold")
        data_fraction = kwargs.get("data_fraction")
        return VisionTask(
            name,
            device,
            quick_mode,
            included_classes=included_classes,
            fold=fold,
            data_fraction=data_fraction,
        )

    match base_name:
        case "cora" | "pubmed" | "citeseer":
            from bioplausible.hyperopt.graph_task import GraphTask

            return GraphTask(base_name, device, quick_mode)
        case "breast_cancer" | "california_housing":
            from bioplausible.hyperopt.tabular_task import TabularTask

            return TabularTask(base_name, device, quick_mode)
        case _:
            logger.warning(
                "Unknown task '%s', defaulting to tiny_shakespeare LM", task_name
            )
            return LMTask("tiny_shakespeare", device, quick_mode)
