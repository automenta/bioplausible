"""
Language Modeling Dataset Utilities

Functions for loading LM datasets (Shakespeare, WikiText, PTB) plus the
tokenizer/dataset machinery that powers LM demos and benchmarks
(folded from the legacy ``equitile/lm/data.py``).
"""

from __future__ import annotations

import hashlib
import json
import pickle
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from bioplausible.core.logging import get_logger
from bioplausible.data.vision import CharDataset

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "ByteLevelTokenizer",
    "CharacterTokenizer",
    "DataConfig",
    "LMDataset",
    "StreamingLMDataset",
    "Tokenizer",
    "create_custom_dataset",
    "create_dataloader",
    "create_python_dataset",
    "create_shakespeare_dataset",
    "create_tinystories_dataset",
    "get_lm_dataset",
]

logger = get_logger()


def _normalize_lm_name(name: str) -> str:
    """Normalize a friendly LM dataset name to its canonical identifier.

    Accepts display names ("Tiny Shakespeare", "WikiText-2", "PTB") and the
    canonical IDs used by :func:`get_lm_dataset`.

    Args:
        name: Dataset name to normalize.

    Returns:
        The canonical dataset identifier ("tiny_shakespeare", "wikitext-2",
        or "ptb").
    """
    lowered = name.lower().replace(" ", "_")
    if "shakespeare" in lowered or lowered == "tiny_shakespeare":
        return "tiny_shakespeare"
    if "wikitext" in lowered or "wiki_text" in lowered:
        return "wikitext-2"
    if "ptb" in lowered or "penn" in lowered:
        return "ptb"
    return lowered


def get_lm_dataset(
    name: str = "tiny_shakespeare",
    seq_len: int = 128,
    split: str = "train",
    train_frac: float = 0.9,
    val_frac: float = 0.05,
) -> CharDataset:
    """
    Load a language modeling dataset as a CharDataset.

    Accepts friendly display names ("Tiny Shakespeare", "WikiText2") or the
    canonical identifiers ("tiny_shakespeare", "wikitext-2", "ptb") — the
    name is normalized before dispatch.

    Args:
        name: Dataset name ('tiny_shakespeare', 'wikitext-2', 'ptb').
        seq_len: Sequence length for training.
        split: 'train', 'validation', or 'test'.

    Returns:
        CharDataset with vocab_size attribute.
    """
    name = _normalize_lm_name(name)
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace datasets required. Install with: pip install datasets"
        )

    if name == "tiny_shakespeare":
        try:
            dataset = load_dataset("tiny_shakespeare")
            split_data = dataset[split]
            texts = []
            for item in split_data:
                if isinstance(item["text"], str):
                    texts.append(item["text"])
            text = "\n".join(texts)
            if not text or len(text) == 0:
                raise ValueError("Empty text after loading")
        except (OSError, ValueError, RuntimeError, ImportError, KeyError) as e:
            warnings.warn(f"HuggingFace dataset failed, using fallback: {e}")
            import urllib.request

            url = (
                "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
                "tinyshakespeare/input.txt"
            )
            with urllib.request.urlopen(url) as response:
                full_text = response.read().decode("utf-8")
            if train_frac + val_frac >= 1.0:
                raise ValueError(
                    "train_frac + val_frac must be < 1.0 to leave a test split."
                )
            n = len(full_text)
            train_data = full_text[: int(n * train_frac)]
            val_split = int(n * train_frac)
            test_split = int(n * (train_frac + val_frac))
            val_data = full_text[val_split:test_split]
            test_data = full_text[test_split:]
            if split == "train":
                text = train_data
            elif split == "validation":
                text = val_data
            else:
                text = test_data

    elif name == "wikitext-2":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        text = "\n".join(dataset[split]["text"])

    elif name == "ptb":
        dataset = load_dataset("ptb_text_only")
        split_name = (
            "train"
            if split == "train"
            else "validation"
            if split == "validation"
            else "test"
        )
        text = " ".join(dataset[split_name]["sentence"])

    else:
        raise ValueError(
            f"Unknown LM dataset: {name}. Available: tiny_shakespeare, wikitext-2, ptb"
        )

    return CharDataset(text, seq_len=seq_len)


# =============================================================================
# Tokenizers
# =============================================================================


class Tokenizer:
    """Base tokenizer interface."""

    vocab_size: int
    pad_token_id: int
    eos_token_id: int

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text."""
        raise NotImplementedError

    def batch_encode(
        self,
        texts: list[str],
        max_length: int | None = None,
    ) -> Tensor:
        """Batch encode texts.

        Parameters
        ----------
        texts : list of str
            Input texts
        max_length : int, optional
            Maximum length (pads shorter sequences)

        Returns
        -------
        torch.Tensor
            Token IDs (batch, seq_len)
        """
        encoded = [self.encode(t) for t in texts]

        if max_length:
            for i, ids in enumerate(encoded):
                if len(ids) < max_length:
                    encoded[i] = ids + [self.pad_token_id] * (max_length - len(ids))
                else:
                    encoded[i] = ids[:max_length]
        else:
            max_len = max(len(ids) for ids in encoded)
            for i, ids in enumerate(encoded):
                if len(ids) < max_len:
                    encoded[i] = ids + [self.pad_token_id] * (max_len - len(ids))

        return torch.tensor(encoded, dtype=torch.long)


class CharacterTokenizer(Tokenizer):
    """Simple character-level tokenizer.

    Ideal for Shakespeare and other small datasets.

    Parameters
    ----------
    text : str, optional
        Training text to build vocabulary from
    """

    def __init__(self, text: str | None = None) -> None:
        if text:
            chars = sorted(set(text))
            self.vocab = ["<pad>", "<unk>", "<eos>"] + chars
        else:
            # Default character vocab
            self.vocab = ["<pad>", "<unk>", "<eos>"] + list(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:'\"()- \n\t"
            )

        self.char_to_idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx_to_char = {i: c for i, c in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.eos_token_id = 2

    def encode(self, text: str) -> list[int]:
        """Encode text to character IDs."""
        return [self.char_to_idx.get(c, self.unk_token_id) for c in text]

    def decode(self, ids: list[int]) -> str:
        """Decode character IDs to text."""
        return "".join(self.idx_to_char.get(i, "?") for i in ids)

    def save(self, path: str) -> None:
        """Save tokenizer to file."""
        with Path(path).open("w") as f:
            json.dump({"vocab": self.vocab}, f)

    @classmethod
    def load(cls, path: str) -> CharacterTokenizer:
        """Load tokenizer from file."""
        with Path(path).open() as f:
            data = json.load(f)
        tokenizer = cls()
        tokenizer.vocab = data["vocab"]
        tokenizer.char_to_idx = {c: i for i, c in enumerate(data["vocab"])}
        tokenizer.idx_to_char = {i: c for i, c in enumerate(data["vocab"])}
        tokenizer.vocab_size = len(data["vocab"])
        return tokenizer


class ByteLevelTokenizer(Tokenizer):
    """Byte-level tokenizer for arbitrary text.

    Uses raw byte values as tokens (256 possible tokens).
    """

    def __init__(self) -> None:
        self.vocab_size = 256
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2

        self.byte_to_idx = {i: i for i in range(256)}
        self.idx_to_byte = {i: i for i in range(256)}

    def encode(self, text: str) -> list[int]:
        """Encode text to byte IDs."""
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        """Decode byte IDs to text."""
        return bytes(ids).decode("utf-8", errors="replace")


# =============================================================================
# Datasets
# =============================================================================


class LMDataset(Dataset):
    """Language modeling dataset with sequence packing.

    Parameters
    ----------
    text : str
        Raw text data
    tokenizer : Tokenizer
        Tokenizer to use
    seq_length : int
        Sequence length for training
    cache_dir : str, optional
        Directory for caching tokenized data
    """

    def __init__(
        self,
        text: str,
        tokenizer: Tokenizer,
        seq_length: int = 256,
        cache_dir: str | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.cache_dir = cache_dir

        cache_key = self._get_cache_key(text)
        self.data = self._load_or_cache(text, cache_key)

        self.n_sequences = max(0, len(self.data) // seq_length)

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"lm_data_{text_hash}_{self.seq_length}"

    def _load_or_cache(self, text: str, cache_key: str) -> list[int]:
        """Load from cache or tokenize and cache."""
        if self.cache_dir:
            cache_path = Path(self.cache_dir) / f"{cache_key}.pkl"
            if cache_path.exists():
                with Path(cache_path).open("rb") as f:
                    return pickle.load(f)

        tokens = self.tokenizer.encode(text)

        if self.cache_dir:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with Path(cache_path).open("wb") as f:
                pickle.dump(tokens, f)

        return tokens

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Get sequence pair (input, target)."""
        start = idx * self.seq_length
        end = start + self.seq_length + 1  # +1 for target

        chunk = self.data[start:end]

        if len(chunk) < self.seq_length + 1:
            chunk = chunk + [self.tokenizer.pad_token_id] * (
                self.seq_length + 1 - len(chunk)
            )

        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        target_ids = torch.tensor(chunk[1:], dtype=torch.long)

        return input_ids, target_ids


class StreamingLMDataset(IterableDataset):
    """Streaming dataset for large corpora.

    Yields sequences on-the-fly without loading the whole corpus into memory.
    """

    def __init__(
        self,
        text_iterator: Iterator[str],
        tokenizer: Tokenizer,
        seq_length: int = 256,
    ) -> None:
        self.text_iterator = text_iterator
        self.tokenizer = tokenizer
        self.seq_length = seq_length

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        """Iterate over sequences."""
        buffer = []

        for text_chunk in self.text_iterator:
            tokens = self.tokenizer.encode(text_chunk)
            buffer.extend(tokens)

            while len(buffer) >= self.seq_length + 1:
                chunk = buffer[: self.seq_length + 1]
                buffer = buffer[self.seq_length :]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                target_ids = torch.tensor(chunk[1:], dtype=torch.long)

                yield input_ids, target_ids


# =============================================================================
# Data loading
# =============================================================================


@dataclass
class DataConfig:
    """Configuration for data loading."""

    batch_size: int = 32
    seq_length: int = 256
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True


def create_dataloader(
    dataset: Dataset,
    config: DataConfig | None = None,
    shuffle: bool = True,
    **kwargs,
) -> DataLoader:
    """Create a DataLoader with optimized settings."""
    if config is None:
        config = DataConfig()

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        persistent_workers=(
            config.persistent_workers if config.num_workers > 0 else False
        ),
        **kwargs,
    )


# =============================================================================
# Dataset factories
# =============================================================================


_SHORT_SHAKESPEARE = """
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.

Second Citizen:
Would you proceed especially against Caius Marcius?

All:
Against him first: he's a very dog to the commonalty.

Second Citizen:
Consider you what services he has done for his country?

First Citizen:
Very well; and could be content to give him good
report fort, but that he pays himself with being proud.

Second Citizen:
Nay, but speak not maliciously.

First Citizen:
I say unto you, what he hath done famously, he did
it to that end: though soft-conscienced men can be
content to say it was for his country he did it to
please his mother and to be partly proud; which he
is, even till the altitude of his virtue.

Second Citizen:
What he cannot help in his nature, you account a
vice in him. You must in no way say he is covetous.

First Citizen:
If I must not, I need not be barren of accusations;
he hath faults, with surplus, to tire in repetition.

What shouts are these? The other side o' the city
is risen: why stay we prating here? to the Capitol!

All:
Come, come, let's go; let's go.

Second Citizen:
Marry, what comes our way?

First Citizen:
Why, here comes one that is wont to challenge hotly
our party for maintaining the patricians.

Second Citizen:
Let us receive him, and know his business.

Enter MENENIUS

First Citizen:
Nay, I know you well; I know you well.

Menenius:
Are you, Caius Marcius?

First Citizen:
I know you well, sir.

Menenius:
Know me well?

First Citizen:
Do you hear, sir? do you mock at us?

Menenius:
Friends, countrymen, what do you mean?

First Citizen:
We have ever been well used by the patricians.

Menenius:
You are transported, you are transported.

First Citizen:
Transported! we are not transported.

Menenius:
Let me speak to you.

First Citizen:
Ay, sir; if you speak to us, you speak to the people.

Menenius:
I pray you, let me speak to you apart.

First Citizen:
You may do so; we will not deny you.

Menenius:
What do you think of me?

First Citizen:
I think you are a very honest man.

Menenius:
I am accounted a pretty fellow.

First Citizen:
Ay, sir, and a good one too, for aught I know.

Menenius:
I am not a common fellow.

First Citizen:
No, sir, you are not.

Menenius:
I am a man of good quality.

First Citizen:
Ay, sir, and of a good house.

Menenius:
I am one that loves the people.

First Citizen:
And we love you.

Menenius:
Then I pray you, go home.

First Citizen:
Not till we have heard what you have to say.

Menenius:
You shall hear me.

First Citizen:
Speak, speak.

Menenius:
There was a time when all the body's members
Rebell'd against the belly; thus accused it:
That only like a storehouse in the midst
O' the body it did lie, making no labour,
Receiving by the rest, feeding them first;
But that it did digest the food, and sent it
Through the rivers of the blood to the heart and brain.

First Citizen:
What then?

Menenius:
I shall tell you. The belly answered--

First Citizen:
Well, sir, the answer?

Menenius:
With a smile, not like an answer; for the proudest
of you all must know, the belly is the storehouse.

First Citizen:
Go to, sir! we know, we know.

Menenius:
I pray you, patience.

First Citizen:
You are long about it.

Menenius:
Note me this, good friend;
Your most grave belly was deliberate,
Not rash like his accusers, and thus answered:
'True is it, my incorporate friends,' quoth he,
'That I receive the general food at first,
Which you do live upon; and fit it is,
Because I am the storehouse and the shop
Of the whole body: but, if you do remember,
I send it through the rivers of your blood,
Even to the court, the heart, to the seat of the brain.'

First Citizen:
Ay, sir, well, well.

Menenius:
'Though all at once cannot see what I do deliver out
to each, yet I can make my audit up, that all
from me do back receive the flour of all, and leave
me but the bran.'

First Citizen:
And what o' this?

Menenius:
The senators of Rome are this good belly,
And you the mutinous members.

First Citizen:
We, sir! the members!

Menenius:
Ay, sir; and what do you think of this?

First Citizen:
Why, sir, we think it is a very good answer.

Menenius:
Then ought you to thank me, as the people do.

First Citizen:
We thank you, sir.

Menenius:
Go home, go home, and think upon what I have said.

All:
We will, we will.
"""


def _get_shakespeare_text(cache_dir: str | None = None) -> str:
    """Load Shakespeare text dataset (embedded excerpt unless cached)."""
    if cache_dir:
        cache_path = Path(cache_dir) / "shakespeare.txt"
        if cache_path.exists():
            return cache_path.read_text()

    text = _SHORT_SHAKESPEARE.strip()

    if cache_dir:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text)

    return text


def create_shakespeare_dataset(
    batch_size: int = 32,
    seq_length: int = 256,
    num_workers: int = 4,
    cache_dir: str | None = None,
    val_split: float = 0.1,
) -> tuple[DataLoader, DataLoader, CharacterTokenizer]:
    """Create Shakespeare character-level dataset.

    Returns
    -------
    tuple
        (train_loader, val_loader, tokenizer)
    """
    text = _get_shakespeare_text(cache_dir)

    tokenizer = CharacterTokenizer(text)

    split_idx = int(len(text) * (1 - val_split))
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    train_dataset = LMDataset(train_text, tokenizer, seq_length, cache_dir=cache_dir)
    val_dataset = LMDataset(val_text, tokenizer, seq_length, cache_dir=cache_dir)

    config = DataConfig(
        batch_size=batch_size,
        seq_length=seq_length,
        num_workers=num_workers,
    )

    train_loader = create_dataloader(train_dataset, config, shuffle=True)
    val_loader = create_dataloader(val_dataset, config, shuffle=False)

    return train_loader, val_loader, tokenizer


def create_tinystories_dataset(
    data_path: str,
    batch_size: int = 32,
    seq_length: int = 256,
    num_workers: int = 4,
    cache_dir: str | None = None,
    max_samples: int | None = None,
) -> tuple[DataLoader, DataLoader, CharacterTokenizer]:
    """Create TinyStories dataset from a local JSONL file."""
    stories = []
    with Path(data_path).open() as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            data = json.loads(line)
            stories.append(data.get("story", ""))

    text = "\n\n".join(stories)

    split_idx = int(len(text) * 0.9)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    tokenizer = CharacterTokenizer(text[:10000])

    train_dataset = LMDataset(train_text, tokenizer, seq_length, cache_dir)
    val_dataset = LMDataset(val_text, tokenizer, seq_length, cache_dir)

    config = DataConfig(batch_size=batch_size, num_workers=num_workers)
    train_loader = create_dataloader(train_dataset, config, shuffle=True)
    val_loader = create_dataloader(val_dataset, config, shuffle=False)

    return train_loader, val_loader, tokenizer


def create_python_dataset(
    data_path: str,
    batch_size: int = 32,
    seq_length: int = 512,
    num_workers: int = 4,
    cache_dir: str | None = None,
) -> tuple[DataLoader, DataLoader, ByteLevelTokenizer]:
    """Create a Python code completion dataset based on local files."""
    path = Path(data_path)
    if path.is_file():
        files = [path]
    else:
        files = list(path.glob("**/*.py"))

    code_texts = []
    for f in files:
        try:
            code_texts.append(f.read_text())
        except OSError, ValueError:
            logger.warning("Failed to read file %s", f)

    text = "\n\n# === END OF FILE ===\n\n".join(code_texts)

    split_idx = int(len(text) * 0.9)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    tokenizer = ByteLevelTokenizer()

    train_dataset = LMDataset(train_text, tokenizer, seq_length, cache_dir)
    val_dataset = LMDataset(val_text, tokenizer, seq_length, cache_dir)

    config = DataConfig(batch_size=batch_size, num_workers=num_workers)
    train_loader = create_dataloader(train_dataset, config, shuffle=True)
    val_loader = create_dataloader(val_dataset, config, shuffle=False)

    return train_loader, val_loader, tokenizer


def create_custom_dataset(
    text: str,
    tokenizer: Tokenizer | None = None,
    batch_size: int = 32,
    seq_length: int = 256,
    num_workers: int = 4,
    cache_dir: str | None = None,
    val_split: float = 0.1,
) -> tuple[DataLoader, DataLoader, Tokenizer]:
    """Create a dataset from raw text with an optional tokenizer."""
    if tokenizer is None:
        tokenizer = CharacterTokenizer(text)

    split_idx = int(len(text) * (1 - val_split))
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    train_dataset = LMDataset(train_text, tokenizer, seq_length, cache_dir)
    val_dataset = LMDataset(val_text, tokenizer, seq_length, cache_dir)

    config = DataConfig(batch_size=batch_size, num_workers=num_workers)
    train_loader = create_dataloader(train_dataset, config, shuffle=True)
    val_loader = create_dataloader(val_dataset, config, shuffle=False)

    return train_loader, val_loader, tokenizer
