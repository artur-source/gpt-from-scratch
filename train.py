"""Train the character-level GPT model on tiny Shakespeare."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from config import config
from model.gpt import build_model
from utils.tokenizer import ShakespeareDataset, load_text


CHECKPOINT_DIR = Path("checkpoints")
ASSETS_DIR = Path("assets")
DATA_PATH = Path("data/tiny_shakespeare.txt")


@torch.no_grad()
def estimate_loss(model: torch.nn.Module, dataset: ShakespeareDataset) -> dict[str, float]:
    """Evaluate mean train and validation loss over config.eval_iters random batches."""
    was_training = model.training
    model.eval()
    losses: dict[str, float] = {}

    for split in ("train", "val"):
        split_losses = torch.zeros(config.eval_iters)
        for idx in range(config.eval_iters):
            x, y = dataset.get_batch(split)
            _, loss = model(x, y)
            assert loss is not None
            split_losses[idx] = loss.item()
        losses[split] = split_losses.mean().item()

    if was_training:
        model.train()
    return losses


def plot_loss_curve(history: list[dict[str, float]]) -> None:
    """Save train and validation loss curves to assets/loss_curve.png."""
    if not history:
        return

    steps = [item["step"] for item in history]
    train_losses = [item["train"] for item in history]
    val_losses = [item["val"] for item in history]

    plt.figure(figsize=(8, 5))
    plt.plot(steps, train_losses, label="train loss")
    plt.plot(steps, val_losses, label="val loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("GPT Training Loss")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "loss_curve.png", dpi=150)
    plt.close()


def save_checkpoint(
    model: torch.nn.Module,
    dataset: ShakespeareDataset,
    val_loss: float,
) -> None:
    """Save the best model weights and metadata needed for generation."""
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(config),
            "val_loss": val_loss,
            "vocab": dataset.tokenizer.chars,
            "vocab_size": dataset.vocab_size,
        },
        CHECKPOINT_DIR / "best_model.pt",
    )


def main() -> None:
    """Run the full training loop, save the best checkpoint, plot losses, and sample text."""
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    ASSETS_DIR.mkdir(exist_ok=True)

    text = load_text(str(DATA_PATH))
    dataset = ShakespeareDataset(text)
    model = build_model(dataset.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.1)

    best_val_loss = float("inf")
    history: list[dict[str, float]] = []

    progress = tqdm(range(config.max_iters), desc="training")
    for step in progress:
        if step % config.eval_interval == 0 or step == config.max_iters - 1:
            losses = estimate_loss(model, dataset)
            history.append({"step": step, "train": losses["train"], "val": losses["val"]})
            print(f"step {step}: train_loss={losses['train']:.4f}, val_loss={losses['val']:.4f}")

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(model, dataset, best_val_loss)

        x, y = dataset.get_batch("train")
        _, loss = model(x, y)
        assert loss is not None

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        progress.set_postfix(loss=f"{loss.item():.4f}", best_val=f"{best_val_loss:.4f}")

    plot_loss_curve(history)

    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    generated = model.generate(context, max_new_tokens=200, temperature=0.8, top_k=40)
    print("\nGenerated sample:\n")
    print(dataset.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
