"""Complete GPT language model built from manual Transformer blocks."""

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
from model.attention import Block


class GPTLanguageModel(nn.Module):
    """Character-level GPT model that predicts the next token at every position."""

    def __init__(
        self,
        vocab_size: int,
        n_embd: int = config.n_embd,
        n_head: int = config.n_head,
        n_layer: int = config.n_layer,
        block_size: int = config.block_size,
        dropout: float = config.dropout,
    ) -> None:
        """Create token/position embeddings, Transformer blocks, final norm, and tied LM head."""
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embd=n_embd,
                    n_head=n_head,
                    block_size=block_size,
                    dropout=dropout,
                )
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute logits for idx and optional cross-entropy loss against targets."""
        batch_size, time_steps = idx.shape
        if time_steps > self.block_size:
            raise ValueError(f"Sequence length {time_steps} exceeds block_size={self.block_size}")

        token_emb = self.token_embedding(idx)
        position_ids = torch.arange(time_steps, device=idx.device)
        position_emb = self.position_embedding(position_ids)
        x = token_emb + position_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        logits_flat = logits.view(batch_size * time_steps, self.vocab_size)
        targets_flat = targets.view(batch_size * time_steps)
        loss = F.cross_entropy(logits_flat, targets_flat)
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively sample new tokens from the model distribution."""
        if temperature <= 0:
            raise ValueError("temperature must be greater than 0")

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                top_k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, top_k)
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def build_model(vocab_size: int) -> GPTLanguageModel:
    """Instantiate GPTLanguageModel, initialize trainable weights, and return it on config.device."""
    model = GPTLanguageModel(vocab_size=vocab_size)
    model.apply(_init_weights)
    model.lm_head.weight = model.token_embedding.weight
    return model.to(config.device)


def _init_weights(module: nn.Module) -> None:
    """Initialize Linear and Embedding weights with the GPT-style normal distribution."""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


if __name__ == "__main__":
    vocab_size = 65
    B, T = 2, config.block_size

    model = build_model(vocab_size=vocab_size)
    idx = torch.randint(0, vocab_size, (B, T), device=config.device)
    logits, loss = model(idx, idx)

    assert logits.shape == (B, T, vocab_size), f"Wrong logits shape: {logits.shape}"
    assert loss is not None, "Loss should be computed when targets are provided"
    print(f"Forward OK - logits shape: {logits.shape}")

    generated = model.generate(idx[:, :1], max_new_tokens=10)
    assert generated.shape == (B, 11), f"Wrong generated shape: {generated.shape}"
    print(f"Generate OK - output shape: {generated.shape}")
