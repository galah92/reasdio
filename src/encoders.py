"""Audio encoders with a common interface.

All encoders implement:
    def encode(self, waveform: Tensor) -> Tensor:
        '''(batch, samples) → (batch, frames, dim)'''

    @property
    def dim(self) -> int:
        '''Output embedding dimension'''

    @property
    def sample_rate(self) -> int:
        '''Expected input sample rate'''
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor


class AudioEncoder(ABC, nn.Module):
    """Base class for audio encoders."""

    @abstractmethod
    def encode(self, waveform: Tensor) -> Tensor:
        """Encode waveform to frame-level features.

        Args:
            waveform: (batch, samples) tensor at self.sample_rate

        Returns:
            (batch, frames, dim) tensor of frame-level features
        """
        pass

    @property
    @abstractmethod
    def dim(self) -> int:
        """Output embedding dimension."""
        pass

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Expected input sample rate."""
        pass


class HTSATEncoder(AudioEncoder):
    """HTSAT encoder wrapper that extracts frame-level features.

    Uses the HTSAT model from MS-CLAP to extract 768-dim frame-level features
    from audio waveforms. The model is pretrained on AudioSet.
    """

    def __init__(self, freeze: bool = True):
        """Initialize HTSAT encoder.

        Args:
            freeze: If True, freeze the encoder weights (default: True)
        """
        super().__init__()

        # Load CLAP model to get pretrained HTSAT
        from msclap import CLAP
        clap = CLAP(version='2023', use_cuda=False)

        # Extract HTSAT model
        self.htsat = clap.clap.audio_encoder.base.htsat

        # The output dimension before classification is 768
        self._dim = 768
        self._sample_rate = 48000

        if freeze:
            for param in self.htsat.parameters():
                param.requires_grad = False
            self.htsat.eval()

        self._frozen = freeze

    def train(self, mode: bool = True):
        """Override train to keep frozen encoder in eval mode."""
        super().train(mode)
        if self._frozen:
            self.htsat.eval()
        return self

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def encode(self, waveform: Tensor) -> Tensor:
        """Extract frame-level features from audio.

        Args:
            waveform: (batch, samples) at 48kHz

        Returns:
            (batch, frames, 768) tensor
        """
        # Get the intermediate features using a forward hook
        features = []

        def hook_fn(module, input, output):
            # After the transformer layers and norm, before pooling
            # output is (B, N, C) where N is num patches, C is 768
            features.append(output)

        # Register hook on the norm layer (after all transformer blocks)
        handle = self.htsat.norm.register_forward_hook(hook_fn)

        try:
            # Forward pass (we discard the output, we want the hook result)
            with torch.set_grad_enabled(not self._frozen):
                _ = self.htsat(waveform, infer_mode=False)
        finally:
            handle.remove()

        # features[0] is (B, N, 768) where N depends on audio length
        frame_features = features[0]

        return frame_features


class MimiEncoder(AudioEncoder):
    """Mimi codec encoder that extracts continuous embeddings from audio.

    Mimi is a neural audio codec that can be used in two modes:
    - semantic: Use only the first RVQ level (semantic tokens) - 512 dim
    - full: Use all 8 RVQ levels - 512 dim (summed embeddings)

    The decoded embeddings are 512-dimensional continuous vectors.
    """

    def __init__(
        self,
        mode: str = "semantic",
        freeze: bool = True,
    ):
        """Initialize Mimi encoder.

        Args:
            mode: "semantic" for first RVQ level only, "full" for all 8 levels
            freeze: If True, freeze the encoder weights (default: True)
        """
        super().__init__()

        if mode not in ("semantic", "full"):
            raise ValueError(f"mode must be 'semantic' or 'full', got {mode}")

        self.mode = mode
        self._frozen = freeze

        # Load Mimi model (None downloads from HuggingFace)
        from moshi.models import loaders
        self.mimi = loaders.get_mimi(None, device='cpu')

        # Set number of codebooks based on mode
        # semantic: 1 codebook, full: 8 codebooks
        self._num_codebooks = 1 if mode == "semantic" else 8
        self.mimi.set_num_codebooks(self._num_codebooks)

        # Mimi quantizer outputs 512-dim embeddings (sum of codebook embeddings)
        self._dim = 512
        self._sample_rate = 24000  # Mimi uses 24kHz

        if freeze:
            for param in self.mimi.parameters():
                param.requires_grad = False
            self.mimi.eval()

    def train(self, mode: bool = True):
        """Override train to keep frozen encoder in eval mode."""
        super().train(mode)
        if self._frozen:
            self.mimi.eval()
        return self

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def encode(self, waveform: Tensor) -> Tensor:
        """Extract embeddings from audio using Mimi.

        Args:
            waveform: (batch, samples) at 24kHz

        Returns:
            (batch, frames, 512) tensor
        """
        # Mimi expects (batch, channels, samples)
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)  # (B, 1, T)

        with torch.set_grad_enabled(not self._frozen):
            # Encode to codes
            codes = self.mimi.encode(waveform)  # (B, num_codebooks, frames)

            # Decode codes back to continuous embeddings
            # This gives us the sum of the codebook embeddings
            embeddings = self.mimi.quantizer.decode(codes)  # (B, 512, T)

            # Transpose to (B, T, 512) for consistency with other encoders
            frame_features = embeddings.transpose(1, 2)

        return frame_features


def get_encoder(name: str, **kwargs) -> AudioEncoder:
    """Factory function to get an encoder by name.

    Args:
        name: One of "htsat", "mimi_semantic", "mimi_full"
        **kwargs: Additional arguments passed to the encoder

    Returns:
        AudioEncoder instance
    """
    encoders = {
        "htsat": lambda: HTSATEncoder(**kwargs),
        "mimi_semantic": lambda: MimiEncoder(mode="semantic", **kwargs),
        "mimi_full": lambda: MimiEncoder(mode="full", **kwargs),
    }

    if name not in encoders:
        raise ValueError(f"Unknown encoder: {name}. Available: {list(encoders.keys())}")

    return encoders[name]()
