"""Manual causal self-attention layers used by the GPT model."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import config


class Head(nn.Module):
    """Single causal self-attention head.

    For an input X, this layer learns query, key, and value projections:
    Q = XWq, K = XWk, V = XWv. Attention scores are computed as
    softmax((QK^T) / sqrt(head_size)) and multiplied by V. A lower-triangular
    mask forces each token to attend only to itself and previous tokens.
    """

    def __init__(
        self,
        head_size: int,
        n_embd: int = config.n_embd,
        block_size: int = config.block_size,
        dropout: float = config.dropout,
    ) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return attended values with shape (batch, time, head_size)."""
        _, time_steps, _ = x.shape
        key = self.key(x)
        query = self.query(x)

        weights = query @ key.transpose(-2, -1) * key.shape[-1] ** -0.5
        weights = weights.masked_fill(self.tril[:time_steps, :time_steps] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        self.last_attention_weights = weights.detach()
        weights = self.dropout(weights)

        value = self.value(x)
        return weights @ value


class MultiHeadAttention(nn.Module):
    """Run several causal self-attention heads in parallel.

    Each head attends to the same sequence through a different learned
    projection subspace. Their outputs are concatenated along the channel
    dimension, projected back to n_embd, and regularized with dropout.
    """

    def __init__(
        self,
        n_head: int,
        head_size: int,
        n_embd: int = config.n_embd,
        block_size: int = config.block_size,
        dropout: float = config.dropout,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_embd=n_embd, block_size=block_size, dropout=dropout) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_head * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return multi-head attention output with shape (batch, time, n_embd)."""
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        self.last_attention_weights = torch.stack(
            [head.last_attention_weights for head in self.heads],
            dim=1,
        )
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Position-wise MLP applied independently to each token.

    The original Transformer expands the embedding dimension by 4x, applies a
    non-linearity, then projects back to the model dimension. This gives each
    token extra capacity after attention has mixed information across positions.
    """

    def __init__(self, n_embd: int = config.n_embd, dropout: float = config.dropout) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return transformed token representations with shape (batch, time, n_embd)."""
        return self.net(x)


class Block(nn.Module):
    """Transformer block with pre-normalization and residual connections.

    GPT-2 style pre-norm computes x + Attention(LayerNorm(x)), followed by
    x + FeedForward(LayerNorm(x)). Residual paths preserve the token stream
    while attention mixes context and the feed-forward network refines features.
    """

    def __init__(
        self,
        n_embd: int = config.n_embd,
        n_head: int = config.n_head,
        block_size: int = config.block_size,
        dropout: float = config.dropout,
    ) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(
            n_head=n_head,
            head_size=head_size,
            n_embd=n_embd,
            block_size=block_size,
            dropout=dropout,
        )
        self.ff = FeedForward(n_embd=n_embd, dropout=dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    @property
    def attn(self) -> MultiHeadAttention:
        """Expose the self-attention module for hooks and visualization notebooks."""
        return self.sa

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the block output with shape (batch, time, n_embd)."""
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


if __name__ == "__main__":
    B, T, C = 2, config.block_size, config.n_embd
    x = torch.randn(B, T, C)

    block = Block(config.n_embd, config.n_head)
    out = block(x)

    assert out.shape == (B, T, C), f"Wrong shape: {out.shape}"
    print(f"Block OK - output shape: {out.shape}")
