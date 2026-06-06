"""Generate text from a trained GPT checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import config
from model.gpt import build_model
from utils.tokenizer import CharacterTokenizer


CHECKPOINT_PATH = Path("checkpoints/best_model.pt")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for text generation."""
    parser = argparse.ArgumentParser(description="Generate text with the trained GPT model.")
    parser.add_argument("--prompt", type=str, default="\n", help="Text prompt used to start generation.")
    parser.add_argument("--max_tokens", type=int, default=200, help="Number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=40, help="Keep only the top_k logits before sampling.")
    return parser.parse_args()


def load_checkpoint(path: Path = CHECKPOINT_PATH) -> dict:
    """Load a checkpoint dictionary from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}. Run `python train.py` first.")
    return torch.load(path, map_location=config.device)


def main() -> None:
    """Load the best checkpoint, encode the prompt, generate tokens, and print decoded text."""
    args = parse_args()
    checkpoint = load_checkpoint()

    tokenizer = CharacterTokenizer(checkpoint["vocab"])
    model = build_model(checkpoint["vocab_size"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    prompt_tokens = tokenizer.encode(args.prompt)
    idx = torch.tensor([prompt_tokens], dtype=torch.long, device=config.device)
    generated = model.generate(
        idx,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
