"""Training script for AudioLM model."""

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.data import ReasonAQADataset, collate_fn
from src.model import AudioLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_sample_rate_for_encoder(encoder_name: str) -> int:
    """Get the expected sample rate for an encoder."""
    if encoder_name == "htsat":
        return 48000
    elif encoder_name in ("mimi_semantic", "mimi_full"):
        return 24000
    else:
        raise ValueError(f"Unknown encoder: {encoder_name}")


def train(
    encoder: str = "htsat",
    train_json: str = "data/reasonaqa/train.json",
    val_json: str = "data/reasonaqa/val.json",
    audio_dir: str = "data",
    output_dir: str = "checkpoints",
    batch_size: int = 16,
    lr: float = 1e-4,
    max_steps: int = 20000,
    max_audio_sec: float = 10.0,
    log_every: int = 10,
    eval_every: int = 500,
    save_every: int = 1000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_workers: int = 4,
    gradient_accumulation_steps: int = 1,
):
    """Train the AudioLM model.

    Args:
        encoder: Encoder name ("htsat", "mimi_semantic", "mimi_full")
        train_json: Path to training JSON
        val_json: Path to validation JSON
        audio_dir: Directory containing audio files
        output_dir: Directory to save checkpoints
        batch_size: Batch size
        lr: Learning rate
        max_steps: Maximum training steps
        max_audio_sec: Maximum audio duration in seconds
        log_every: Log every N steps
        eval_every: Evaluate every N steps
        save_every: Save checkpoint every N steps
        device: Device to train on
        num_workers: Number of data loader workers
        gradient_accumulation_steps: Number of gradient accumulation steps
    """
    output_dir = Path(output_dir) / encoder
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get sample rate for encoder
    sample_rate = get_sample_rate_for_encoder(encoder)
    logger.info(f"Using encoder: {encoder} with sample rate: {sample_rate}")

    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = ReasonAQADataset(
        json_path=train_json,
        audio_dir=audio_dir,
        sample_rate=sample_rate,
        max_audio_sec=max_audio_sec,
    )
    logger.info(f"Train dataset: {len(train_dataset)} samples")

    val_dataset = None
    if Path(val_json).exists():
        val_dataset = ReasonAQADataset(
            json_path=val_json,
            audio_dir=audio_dir,
            sample_rate=sample_rate,
            max_audio_sec=max_audio_sec,
        )
        logger.info(f"Val dataset: {len(val_dataset)} samples")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
        )

    # Create model
    logger.info("Creating model...")
    model = AudioLM(
        encoder=encoder,
        freeze_encoder=True,
        freeze_lm=True,
    )
    model = model.to(device)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    # Create optimizer (only for trainable parameters)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
    )

    # Training loop
    logger.info("Starting training...")
    model.train()
    step = 0
    total_loss = 0.0
    best_val_loss = float("inf")

    train_iter = iter(train_loader)

    while step < max_steps:
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Move to device
        waveforms = batch["waveforms"].to(device)
        questions = batch["questions"]
        answers = batch["answers"]

        # Forward pass
        outputs = model(waveforms, questions, answers)
        loss = outputs["loss"] / gradient_accumulation_steps

        # Backward pass
        loss.backward()

        # Update weights
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps
        step += 1

        # Logging
        if step % log_every == 0:
            avg_loss = total_loss / log_every
            logger.info(f"Step {step}/{max_steps} - Loss: {avg_loss:.4f}")
            total_loss = 0.0

        # Evaluation
        if val_loader is not None and step % eval_every == 0:
            val_loss = evaluate(model, val_loader, device)
            logger.info(f"Step {step} - Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, step, val_loss,
                    output_dir / "best.pt",
                )
                logger.info(f"Saved best model with val loss: {val_loss:.4f}")

            model.train()

        # Save checkpoint
        if step % save_every == 0:
            save_checkpoint(
                model, optimizer, step, None,
                output_dir / f"step_{step}.pt",
            )

    # Save final checkpoint
    save_checkpoint(
        model, optimizer, step, None,
        output_dir / "final.pt",
    )
    logger.info("Training complete!")


def evaluate(model, val_loader, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            waveforms = batch["waveforms"].to(device)
            questions = batch["questions"]
            answers = batch["answers"]

            outputs = model(waveforms, questions, answers)
            total_loss += outputs["loss"].item()
            num_batches += 1

    return total_loss / num_batches


def save_checkpoint(model, optimizer, step, val_loss, path):
    """Save training checkpoint."""
    torch.save(
        {
            "step": step,
            "model_state_dict": model.mapper.state_dict(),  # Only save mapper
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description="Train AudioLM model")
    parser.add_argument("--encoder", type=str, default="htsat",
                        choices=["htsat", "mimi_semantic", "mimi_full"],
                        help="Encoder to use")
    parser.add_argument("--train-json", type=str, default="data/reasonaqa/train.json",
                        help="Path to training JSON")
    parser.add_argument("--val-json", type=str, default="data/reasonaqa/val.json",
                        help="Path to validation JSON")
    parser.add_argument("--audio-dir", type=str, default="data",
                        help="Directory containing audio files")
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=20000,
                        help="Maximum training steps")
    parser.add_argument("--max-audio-sec", type=float, default=10.0,
                        help="Maximum audio duration in seconds")
    parser.add_argument("--log-every", type=int, default=10,
                        help="Log every N steps")
    parser.add_argument("--eval-every", type=int, default=500,
                        help="Evaluate every N steps")
    parser.add_argument("--save-every", type=int, default=1000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to train on (default: auto)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loader workers")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="Number of gradient accumulation steps")

    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train(
        encoder=args.encoder,
        train_json=args.train_json,
        val_json=args.val_json,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        max_steps=args.max_steps,
        max_audio_sec=args.max_audio_sec,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
        device=device,
        num_workers=args.num_workers,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )


if __name__ == "__main__":
    main()
