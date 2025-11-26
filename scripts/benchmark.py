"""Benchmark training speed to estimate total training time."""

import argparse
import time

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.data import ReasonAQADataset, collate_fn
from src.model import AudioLM


def benchmark_training(
    encoder: str = "htsat",
    json_path: str = "data/reasonaqa/clotho_test_subset.json",
    audio_dir: str = "data",
    batch_size: int = 16,
    num_steps: int = 20,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Benchmark training speed."""
    # Get sample rate
    sample_rate = 48000 if encoder == "htsat" else 24000

    print(f"Device: {device}")
    print(f"Encoder: {encoder}")
    print(f"Batch size: {batch_size}")
    print(f"Sample rate: {sample_rate}")
    print()

    # Create dataset
    dataset = ReasonAQADataset(
        json_path=json_path,
        audio_dir=audio_dir,
        sample_rate=sample_rate,
        max_audio_sec=10.0,
    )
    print(f"Dataset size: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Single worker for accurate timing
    )

    # Create model
    print("Creating model...")
    model = AudioLM(encoder=encoder, freeze_encoder=True, freeze_lm=True)
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    # Optimizer
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    # Warmup
    print("\nWarmup run...")
    batch = next(iter(loader))
    waveforms = batch["waveforms"].to(device)
    outputs = model(waveforms, batch["questions"], batch["answers"])
    outputs["loss"].backward()
    optimizer.step()
    optimizer.zero_grad()

    # Benchmark
    print(f"\nBenchmarking {num_steps} steps...")
    times = []
    data_iter = iter(loader)

    for i in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        waveforms = batch["waveforms"].to(device)

        start = time.perf_counter()

        outputs = model(waveforms, batch["questions"], batch["answers"])
        outputs["loss"].backward()
        optimizer.step()
        optimizer.zero_grad()

        if device == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        times.append(elapsed)

        if (i + 1) % 5 == 0:
            print(f"Step {i+1}: {elapsed:.3f}s")

    # Statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print()
    print("=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Average step time: {avg_time:.3f}s")
    print(f"Min step time: {min_time:.3f}s")
    print(f"Max step time: {max_time:.3f}s")
    print(f"Steps per second: {1/avg_time:.2f}")
    print()

    # Estimate training time
    total_steps = 20000
    print("ESTIMATED TRAINING TIME")
    print("-" * 50)
    print(f"For {total_steps:,} steps:")
    total_seconds = total_steps * avg_time
    hours = total_seconds / 3600
    print(f"  Total: {hours:.1f} hours ({total_seconds/60:.0f} minutes)")
    print()

    # For Clotho-only training (278k samples)
    clotho_samples = 278680
    epochs_for_20k_steps = (total_steps * batch_size) / clotho_samples
    print(f"With {clotho_samples:,} Clotho samples and batch_size={batch_size}:")
    print(f"  Steps per epoch: {clotho_samples // batch_size:,}")
    print(f"  Epochs for 20k steps: {epochs_for_20k_steps:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark training speed")
    parser.add_argument("--encoder", type=str, default="htsat",
                        choices=["htsat", "mimi_semantic", "mimi_full"])
    parser.add_argument("--json-path", type=str, default="data/reasonaqa/clotho_test_subset.json")
    parser.add_argument("--audio-dir", type=str, default="data")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    benchmark_training(
        encoder=args.encoder,
        json_path=args.json_path,
        audio_dir=args.audio_dir,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        device=device,
    )


if __name__ == "__main__":
    main()
