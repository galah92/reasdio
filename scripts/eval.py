"""Evaluation script for AudioLM model."""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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


def compute_accuracy(predictions: list[str], targets: list[str]) -> float:
    """Compute exact match accuracy."""
    correct = sum(1 for p, t in zip(predictions, targets) if p.strip().lower() == t.strip().lower())
    return correct / len(targets) if targets else 0.0


def extract_answer_letter(text: str) -> str:
    """Extract answer letter from text (e.g., 'a', 'b', 'c', 'd')."""
    text = text.strip().lower()
    # Check if starts with letter followed by ) or .
    if len(text) >= 1 and text[0] in 'abcd':
        return text[0]
    # Check for "answer: X" pattern
    if 'answer' in text:
        for char in text.split('answer')[-1]:
            if char in 'abcd':
                return char
    return text[:1] if text else ''


def compute_letter_accuracy(predictions: list[str], targets: list[str]) -> float:
    """Compute accuracy based on answer letter extraction."""
    correct = 0
    for p, t in zip(predictions, targets):
        pred_letter = extract_answer_letter(p)
        target_letter = extract_answer_letter(t)
        if pred_letter == target_letter:
            correct += 1
    return correct / len(targets) if targets else 0.0


def evaluate(
    model: AudioLM,
    dataloader: DataLoader,
    device: str,
    max_new_tokens: int = 50,
) -> dict:
    """Evaluate model on dataset.

    Returns:
        Dict with metrics (loss, accuracy, letter_accuracy)
    """
    model.eval()

    all_predictions = []
    all_targets = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            waveforms = batch["waveforms"].to(device)
            questions = batch["questions"]
            answers = batch["answers"]

            # Compute loss
            outputs = model(waveforms, questions, answers)
            if outputs["loss"] is not None:
                total_loss += outputs["loss"].item()
            num_batches += 1

            # Generate predictions
            predictions = model.generate(
                waveforms, questions, max_new_tokens=max_new_tokens
            )

            all_predictions.extend(predictions)
            all_targets.extend(answers)

    # Compute metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    exact_accuracy = compute_accuracy(all_predictions, all_targets)
    letter_accuracy = compute_letter_accuracy(all_predictions, all_targets)

    return {
        "loss": avg_loss,
        "exact_accuracy": exact_accuracy,
        "letter_accuracy": letter_accuracy,
        "num_samples": len(all_targets),
        "predictions": all_predictions,
        "targets": all_targets,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate AudioLM model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint")
    parser.add_argument("--encoder", type=str, required=True,
                        choices=["htsat", "mimi_semantic", "mimi_full"],
                        help="Encoder type")
    parser.add_argument("--test-json", type=str, default="data/reasonaqa/test.json",
                        help="Path to test JSON")
    parser.add_argument("--audio-dir", type=str, default="data",
                        help="Directory containing audio files")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results JSON")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--max-audio-sec", type=float, default=10.0,
                        help="Maximum audio duration")
    parser.add_argument("--max-new-tokens", type=int, default=50,
                        help="Maximum tokens to generate")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to evaluate on")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loader workers")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum samples to evaluate (for debugging)")

    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get sample rate for encoder
    sample_rate = get_sample_rate_for_encoder(args.encoder)
    logger.info(f"Using encoder: {args.encoder} with sample rate: {sample_rate}")

    # Create dataset
    logger.info(f"Loading test data from {args.test_json}")
    dataset = ReasonAQADataset(
        json_path=args.test_json,
        audio_dir=args.audio_dir,
        sample_rate=sample_rate,
        max_audio_sec=args.max_audio_sec,
    )

    if args.max_samples:
        from torch.utils.data import Subset
        indices = list(range(min(args.max_samples, len(dataset))))
        dataset = Subset(dataset, indices)

    logger.info(f"Test dataset: {len(dataset)} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Create model
    logger.info("Creating model...")
    model = AudioLM(
        encoder=args.encoder,
        freeze_encoder=True,
        freeze_lm=True,
    )

    # Load checkpoint (only mapper weights)
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.mapper.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)

    # Evaluate
    logger.info("Starting evaluation...")
    results = evaluate(model, dataloader, device, args.max_new_tokens)

    # Print results
    logger.info("=" * 50)
    logger.info("RESULTS")
    logger.info("=" * 50)
    logger.info(f"Loss: {results['loss']:.4f}")
    logger.info(f"Exact Accuracy: {results['exact_accuracy']:.4f} ({results['exact_accuracy']*100:.2f}%)")
    logger.info(f"Letter Accuracy: {results['letter_accuracy']:.4f} ({results['letter_accuracy']*100:.2f}%)")
    logger.info(f"Samples evaluated: {results['num_samples']}")

    # Show some examples
    logger.info("\nExample predictions:")
    for i in range(min(5, len(results['predictions']))):
        logger.info(f"  Target: {results['targets'][i][:80]}")
        logger.info(f"  Pred:   {results['predictions'][i][:80]}")
        logger.info("")

    # Save results
    if args.output:
        output_data = {
            "loss": results["loss"],
            "exact_accuracy": results["exact_accuracy"],
            "letter_accuracy": results["letter_accuracy"],
            "num_samples": results["num_samples"],
            "predictions": results["predictions"],
            "targets": results["targets"],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
