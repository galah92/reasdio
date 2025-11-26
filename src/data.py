"""Dataset and collate functions for ReasonAQA."""

import json
from pathlib import Path
from typing import Callable

import torch
import torchaudio
from torch.utils.data import Dataset


class ReasonAQADataset(Dataset):
    """Dataset for ReasonAQA audio question answering.

    Expected JSON format:
    {
        "filepath1": "path/to/audio.wav",
        "filepath2": "",  # optional second audio (may be empty)
        "input": "question text",
        "answer": "expected answer",
        "taskname": "audiocaps",  # optional metadata
        "caption1": "...",  # optional
        "caption2": "...",  # optional
        "subtype": "..."  # optional
    }

    When filepath2 is provided, the two audio files are concatenated
    with a small silence gap between them.
    """

    def __init__(
        self,
        json_path: str | Path,
        audio_dir: str | Path,
        sample_rate: int = 24000,
        max_audio_sec: float = 10.0,
        silence_gap_sec: float = 0.5,
    ):
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.max_samples = int(max_audio_sec * sample_rate)
        self.silence_samples = int(silence_gap_sec * sample_rate)

        with open(json_path) as f:
            self.data = json.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def _load_audio(self, filepath: str) -> torch.Tensor:
        """Load and preprocess a single audio file."""
        audio_path = self.audio_dir / filepath
        waveform, sr = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)  # (samples,)

        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        return waveform

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]

        # Load primary audio
        waveform1 = self._load_audio(item["filepath1"])

        # Load and concatenate second audio if present
        if item.get("filepath2"):
            waveform2 = self._load_audio(item["filepath2"])
            # Add silence gap between the two audios
            silence = torch.zeros(self.silence_samples)
            waveform = torch.cat([waveform1, silence, waveform2])
        else:
            waveform = waveform1

        # Truncate to max length
        if waveform.shape[0] > self.max_samples:
            waveform = waveform[:self.max_samples]

        return {
            "waveform": waveform,
            "question": item["input"],
            "answer": item["answer"],
        }


def collate_fn(batch: list[dict], pad_value: float = 0.0) -> dict:
    """Collate function that pads waveforms to same length.

    Returns:
        dict with:
            - waveforms: (batch, max_samples) tensor
            - waveform_lengths: (batch,) tensor of original lengths
            - questions: list of question strings
            - answers: list of answer strings
    """
    waveforms = [item["waveform"] for item in batch]
    questions = [item["question"] for item in batch]
    answers = [item["answer"] for item in batch]

    # Get lengths before padding
    lengths = torch.tensor([w.shape[0] for w in waveforms])

    # Pad waveforms to max length in batch
    max_len = lengths.max().item()
    padded = torch.stack([
        torch.nn.functional.pad(w, (0, max_len - w.shape[0]), value=pad_value)
        for w in waveforms
    ])

    return {
        "waveforms": padded,
        "waveform_lengths": lengths,
        "questions": questions,
        "answers": answers,
    }


def get_collate_fn(pad_value: float = 0.0) -> Callable:
    """Returns a collate function with the specified pad value."""
    def _collate(batch: list[dict]) -> dict:
        return collate_fn(batch, pad_value=pad_value)
    return _collate
