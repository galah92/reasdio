"""Quick training script with before/after evaluation to show signal."""

import json
import random
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data import ReasonAQADataset, collate_fn
from src.model import AudioLM


def extract_answer_letter(text: str) -> str:
    """Extract answer letter (a, b, c, d) from text."""
    text = text.strip().lower()
    if len(text) >= 1 and text[0] in 'abcd':
        return text[0]
    return text[:1] if text else ''


def evaluate(model, loader, device, desc="Eval"):
    """Quick evaluation - returns loss and letter accuracy."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            waveforms = batch["waveforms"].to(device)
            questions = batch["questions"]
            answers = batch["answers"]

            # Loss
            outputs = model(waveforms, questions, answers)
            total_loss += outputs["loss"].item()

            # Generate and check accuracy
            preds = model.generate(waveforms, questions, max_new_tokens=20)

            for pred, target in zip(preds, answers):
                if extract_answer_letter(pred) == extract_answer_letter(target):
                    correct += 1
                total += 1

    return {
        "loss": total_loss / len(loader),
        "accuracy": correct / total if total > 0 else 0,
        "correct": correct,
        "total": total,
    }


def main():
    device = "cuda"
    batch_size = 64
    num_steps = 500
    eval_every = 100
    lr = 1e-4

    print("=" * 60)
    print("QUICK TRAINING - Showing signal with minimal training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Steps: {num_steps}")
    print(f"Learning rate: {lr}")
    print()

    # Load dataset
    print("Loading dataset...")
    full_dataset = ReasonAQADataset(
        json_path="data/reasonaqa/clotho_train.json",
        audio_dir="data",
        sample_rate=48000,
        max_audio_sec=10.0,
    )
    print(f"Full dataset: {len(full_dataset)} samples")

    # Create train/eval split from first 1000 samples for speed
    indices = list(range(min(5000, len(full_dataset))))
    random.seed(42)
    random.shuffle(indices)

    train_indices = indices[:4000]
    eval_indices = indices[4000:5000]

    train_dataset = Subset(full_dataset, train_indices)
    eval_dataset = Subset(full_dataset, eval_indices)

    print(f"Train subset: {len(train_dataset)} samples")
    print(f"Eval subset: {len(eval_dataset)} samples")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=32, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )

    # Create model
    print("\nCreating model...")
    model = AudioLM(encoder="htsat", freeze_encoder=True, freeze_lm=True)
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    # Optimizer
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    # BEFORE training evaluation
    print("\n" + "=" * 60)
    print("BEFORE TRAINING")
    print("=" * 60)
    before = evaluate(model, eval_loader, device, "Before")
    print(f"Loss: {before['loss']:.4f}")
    print(f"Accuracy: {before['accuracy']:.2%} ({before['correct']}/{before['total']})")

    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    model.train()
    train_iter = iter(train_loader)
    losses = []

    pbar = tqdm(range(num_steps), desc="Training")
    for step in pbar:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        waveforms = batch["waveforms"].to(device)
        outputs = model(waveforms, batch["questions"], batch["answers"])

        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Periodic evaluation
        if (step + 1) % eval_every == 0:
            avg_loss = sum(losses[-eval_every:]) / eval_every
            print(f"\nStep {step+1}: Avg train loss = {avg_loss:.4f}")

            mid = evaluate(model, eval_loader, device, f"Step {step+1}")
            print(f"  Eval loss: {mid['loss']:.4f}, Accuracy: {mid['accuracy']:.2%}")
            model.train()

    # AFTER training evaluation
    print("\n" + "=" * 60)
    print("AFTER TRAINING")
    print("=" * 60)
    after = evaluate(model, eval_loader, device, "After")
    print(f"Loss: {after['loss']:.4f}")
    print(f"Accuracy: {after['accuracy']:.2%} ({after['correct']}/{after['total']})")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Loss:     {before['loss']:.4f} -> {after['loss']:.4f} (Δ {after['loss'] - before['loss']:+.4f})")
    print(f"Accuracy: {before['accuracy']:.2%} -> {after['accuracy']:.2%} (Δ {(after['accuracy'] - before['accuracy'])*100:+.1f}%)")

    if after['loss'] < before['loss']:
        print("\n✓ Loss DECREASED - Training is working!")
    else:
        print("\n✗ Loss increased - Something may be wrong")

    if after['accuracy'] > before['accuracy']:
        print("✓ Accuracy IMPROVED - Model is learning!")
    else:
        print("✗ Accuracy did not improve")

    # Save checkpoint
    torch.save({
        "model_state_dict": model.mapper.state_dict(),
        "step": num_steps,
        "before_metrics": before,
        "after_metrics": after,
    }, "checkpoints/htsat/quick_train.pt")
    print(f"\nCheckpoint saved to checkpoints/htsat/quick_train.pt")


if __name__ == "__main__":
    main()
