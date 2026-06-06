"""Character-level tokenization and batching utilities for tiny GPT training."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from config import config


@dataclass
class CharacterTokenizer:
    """Maps characters to integer token ids and integer token ids back to characters."""

    chars: list[str]

    def __post_init__(self) -> None:
        self.stoi = {char: idx for idx, char in enumerate(self.chars)}
        self.itos = {idx: char for idx, char in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, text: str) -> list[int]:
        """Convert a string into a list of token ids."""
        return [self.stoi[char] for char in text]

    def decode(self, tokens: list[int] | torch.Tensor) -> str:
        """Convert token ids back into a string."""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return "".join(self.itos[int(token)] for token in tokens)


class ShakespeareDataset:
    """Stores encoded train/validation splits and returns random language-model batches."""

    def __init__(
        self,
        text: str,
        block_size: int = config.block_size,
        batch_size: int = config.batch_size,
        device: str = config.device,
        train_ratio: float = 0.9,
    ) -> None:
        self.tokenizer = build_vocab(text)
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

        data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        split_idx = int(train_ratio * len(data))
        self.train_data = data[:split_idx]
        self.val_data = data[split_idx:]

    @property
    def vocab_size(self) -> int:
        """Return the number of unique characters in the dataset."""
        return self.tokenizer.vocab_size

    def encode(self, text: str) -> list[int]:
        """Encode text using the dataset vocabulary."""
        return self.tokenizer.encode(text)

    def decode(self, tokens: list[int] | torch.Tensor) -> str:
        """Decode token ids using the dataset vocabulary."""
        return self.tokenizer.decode(tokens)

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return input and target tensors shaped as (batch_size, block_size)."""
        if split not in {"train", "val"}:
            raise ValueError("split must be either 'train' or 'val'")

        data = self.train_data if split == "train" else self.val_data
        if len(data) <= self.block_size:
            raise ValueError(
                f"{split} split is too small for block_size={self.block_size}; "
                "use more text or reduce block_size."
            )

        starts = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[idx : idx + self.block_size] for idx in starts])
        y = torch.stack([data[idx + 1 : idx + self.block_size + 1] for idx in starts])
        return x.to(self.device), y.to(self.device)


def build_vocab(text: str) -> CharacterTokenizer:
    """Build a deterministic character-level vocabulary from raw text."""
    chars = sorted(set(text))
    return CharacterTokenizer(chars)


def load_text(path: str = "data/tiny_shakespeare.txt") -> str:
    """Load the raw training corpus from disk."""
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


_default_dataset: ShakespeareDataset | None = None


def setup_dataset(
    text: str | None = None,
    path: str = "data/tiny_shakespeare.txt",
    block_size: int = config.block_size,
    batch_size: int = config.batch_size,
    device: str = config.device,
) -> ShakespeareDataset:
    """Create and store the default dataset used by module-level helper functions."""
    global _default_dataset

    if text is None:
        text = load_text(path)

    _default_dataset = ShakespeareDataset(
        text=text,
        block_size=block_size,
        batch_size=batch_size,
        device=device,
    )
    return _default_dataset


def encode(text: str) -> list[int]:
    """Encode text with the default dataset vocabulary."""
    if _default_dataset is None:
        setup_dataset()
    assert _default_dataset is not None
    return _default_dataset.encode(text)


def decode(tokens: list[int] | torch.Tensor) -> str:
    """Decode token ids with the default dataset vocabulary."""
    if _default_dataset is None:
        setup_dataset()
    assert _default_dataset is not None
    return _default_dataset.decode(tokens)


def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a random batch from the default train or validation split."""
    if _default_dataset is None:
        setup_dataset()
    assert _default_dataset is not None
    return _default_dataset.get_batch(split)
