"""Central configuration for training and generating with a tiny GPT model."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GPTConfig:
    """Hyperparameters shared by the tokenizer, model, training, and generation scripts."""

    batch_size: int = 64
    block_size: int = 256
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.2
    learning_rate: float = 3e-4
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 200
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


config = GPTConfig()
