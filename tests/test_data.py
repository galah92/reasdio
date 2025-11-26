"""Tests for data loading."""

import json
import tempfile
from pathlib import Path

import pytest
import torch
import torchaudio

from src.data import ReasonAQADataset, collate_fn, get_collate_fn


@pytest.fixture
def sample_dataset(tmp_path: Path):
    """Create a temporary dataset with synthetic audio files."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    # Create sample audio files (sine waves at different frequencies)
    sample_rate = 24000
    samples = [
        {"filepath1": "audio1.wav", "filepath2": "", "input": "What sound is this?", "answer": "A beep"},
        {"filepath1": "audio2.wav", "filepath2": "", "input": "Describe the audio", "answer": "Low tone"},
        {"filepath1": "audio3.wav", "filepath2": "", "input": "What do you hear?", "answer": "High pitch"},
    ]

    # Generate audio files with different lengths
    durations = [1.0, 2.0, 0.5]  # seconds
    for i, (item, duration) in enumerate(zip(samples, durations)):
        freq = 440 * (i + 1)  # Different frequencies
        t = torch.linspace(0, duration, int(sample_rate * duration))
        waveform = torch.sin(2 * torch.pi * freq * t).unsqueeze(0)  # (1, samples)
        torchaudio.save(audio_dir / item["filepath1"], waveform, sample_rate)

    # Write JSON file
    json_path = tmp_path / "data.json"
    with open(json_path, "w") as f:
        json.dump(samples, f)

    return json_path, audio_dir


def test_dataset_len(sample_dataset):
    """Test dataset length."""
    json_path, audio_dir = sample_dataset
    dataset = ReasonAQADataset(json_path, audio_dir)
    assert len(dataset) == 3


def test_dataset_getitem(sample_dataset):
    """Test getting individual items."""
    json_path, audio_dir = sample_dataset
    dataset = ReasonAQADataset(json_path, audio_dir, sample_rate=24000, max_audio_sec=10.0)

    item = dataset[0]
    assert "waveform" in item
    assert "question" in item
    assert "answer" in item
    assert isinstance(item["waveform"], torch.Tensor)
    assert item["waveform"].dim() == 1  # Should be 1D after squeeze
    assert item["question"] == "What sound is this?"
    assert item["answer"] == "A beep"


def test_dataset_truncation(sample_dataset):
    """Test that audio is truncated to max_audio_sec."""
    json_path, audio_dir = sample_dataset
    max_sec = 0.5
    sample_rate = 24000
    dataset = ReasonAQADataset(json_path, audio_dir, sample_rate=sample_rate, max_audio_sec=max_sec)

    # Audio 2 is 2 seconds, should be truncated
    item = dataset[1]
    expected_max_samples = int(max_sec * sample_rate)
    assert item["waveform"].shape[0] == expected_max_samples


def test_collate_fn(sample_dataset):
    """Test collate function pads correctly."""
    json_path, audio_dir = sample_dataset
    dataset = ReasonAQADataset(json_path, audio_dir, sample_rate=24000, max_audio_sec=10.0)

    batch = [dataset[i] for i in range(3)]
    collated = collate_fn(batch)

    assert "waveforms" in collated
    assert "waveform_lengths" in collated
    assert "questions" in collated
    assert "answers" in collated

    # Check shapes
    assert collated["waveforms"].dim() == 2  # (batch, samples)
    assert collated["waveforms"].shape[0] == 3
    assert collated["waveform_lengths"].shape == (3,)

    # All waveforms should be padded to the same length
    max_len = collated["waveform_lengths"].max().item()
    assert collated["waveforms"].shape[1] == max_len

    # Check that questions and answers are preserved
    assert len(collated["questions"]) == 3
    assert len(collated["answers"]) == 3


def test_get_collate_fn():
    """Test collate function factory."""
    collate = get_collate_fn(pad_value=-1.0)
    assert callable(collate)


def test_stereo_to_mono(tmp_path: Path):
    """Test that stereo audio is converted to mono."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    # Create stereo audio
    sample_rate = 24000
    duration = 1.0
    t = torch.linspace(0, duration, int(sample_rate * duration))
    left = torch.sin(2 * torch.pi * 440 * t)
    right = torch.sin(2 * torch.pi * 880 * t)
    stereo = torch.stack([left, right])  # (2, samples)
    torchaudio.save(audio_dir / "stereo.wav", stereo, sample_rate)

    # Create JSON
    json_path = tmp_path / "data.json"
    with open(json_path, "w") as f:
        json.dump([{"filepath1": "stereo.wav", "filepath2": "", "input": "Q", "answer": "A"}], f)

    dataset = ReasonAQADataset(json_path, audio_dir)
    item = dataset[0]

    # Should be mono (1D)
    assert item["waveform"].dim() == 1


def test_resampling(tmp_path: Path):
    """Test that audio is resampled to target sample rate."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    # Create audio at different sample rate
    original_sr = 16000
    target_sr = 24000
    duration = 1.0
    t = torch.linspace(0, duration, int(original_sr * duration))
    waveform = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)
    torchaudio.save(audio_dir / "audio.wav", waveform, original_sr)

    # Create JSON
    json_path = tmp_path / "data.json"
    with open(json_path, "w") as f:
        json.dump([{"filepath1": "audio.wav", "filepath2": "", "input": "Q", "answer": "A"}], f)

    dataset = ReasonAQADataset(json_path, audio_dir, sample_rate=target_sr)
    item = dataset[0]

    # Should be resampled to target rate
    expected_samples = int(duration * target_sr)
    # Allow small tolerance due to resampling
    assert abs(item["waveform"].shape[0] - expected_samples) < 100


def test_dual_audio(tmp_path: Path):
    """Test that dual audio files are concatenated with silence gap."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    sample_rate = 24000
    duration1 = 1.0
    duration2 = 0.5
    silence_gap = 0.5

    # Create two audio files
    t1 = torch.linspace(0, duration1, int(sample_rate * duration1))
    waveform1 = torch.sin(2 * torch.pi * 440 * t1).unsqueeze(0)
    torchaudio.save(audio_dir / "audio1.wav", waveform1, sample_rate)

    t2 = torch.linspace(0, duration2, int(sample_rate * duration2))
    waveform2 = torch.sin(2 * torch.pi * 880 * t2).unsqueeze(0)
    torchaudio.save(audio_dir / "audio2.wav", waveform2, sample_rate)

    # Create JSON with dual audio
    json_path = tmp_path / "data.json"
    with open(json_path, "w") as f:
        json.dump([{"filepath1": "audio1.wav", "filepath2": "audio2.wav", "input": "Q", "answer": "A"}], f)

    dataset = ReasonAQADataset(json_path, audio_dir, sample_rate=sample_rate, silence_gap_sec=silence_gap)
    item = dataset[0]

    # Expected length: duration1 + silence + duration2
    expected_samples = int((duration1 + silence_gap + duration2) * sample_rate)
    assert item["waveform"].shape[0] == expected_samples

    # Check that the silence gap is actually silent (zeros)
    silence_start = int(duration1 * sample_rate)
    silence_end = silence_start + int(silence_gap * sample_rate)
    silence_region = item["waveform"][silence_start:silence_end]
    assert torch.allclose(silence_region, torch.zeros_like(silence_region))
