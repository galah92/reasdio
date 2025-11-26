"""Tests for audio encoders."""

import pytest
import torch

from src.encoders import AudioEncoder, HTSATEncoder, MimiEncoder, get_encoder


class TestHTSATEncoder:
    """Tests for HTSAT encoder."""

    @pytest.fixture
    def encoder(self):
        """Create HTSAT encoder."""
        return HTSATEncoder(freeze=True)

    def test_properties(self, encoder):
        """Test encoder properties."""
        assert encoder.dim == 768
        assert encoder.sample_rate == 48000

    def test_encode_shape(self, encoder):
        """Test encode output shape."""
        audio = torch.randn(2, 48000 * 5)  # 5 seconds, batch size 2
        features = encoder.encode(audio)

        assert features.dim() == 3
        assert features.shape[0] == 2  # batch size
        assert features.shape[2] == 768  # embedding dim
        # Frames depend on audio length, just check it's reasonable
        assert features.shape[1] > 0

    def test_encode_different_lengths(self, encoder):
        """Test encoding different length audio."""
        audio_short = torch.randn(1, 48000 * 2)  # 2 seconds
        audio_long = torch.randn(1, 48000 * 10)  # 10 seconds

        features_short = encoder.encode(audio_short)
        features_long = encoder.encode(audio_long)

        # Longer audio should produce more frames
        assert features_long.shape[1] > features_short.shape[1]

    def test_frozen_encoder(self, encoder):
        """Test that frozen encoder parameters don't require grad."""
        for param in encoder.htsat.parameters():
            assert not param.requires_grad


class TestMimiEncoder:
    """Tests for Mimi encoder."""

    @pytest.fixture
    def semantic_encoder(self):
        """Create Mimi encoder in semantic mode."""
        return MimiEncoder(mode="semantic", freeze=True)

    @pytest.fixture
    def full_encoder(self):
        """Create Mimi encoder in full mode."""
        return MimiEncoder(mode="full", freeze=True)

    def test_semantic_properties(self, semantic_encoder):
        """Test semantic encoder properties."""
        assert semantic_encoder.dim == 512
        assert semantic_encoder.sample_rate == 24000

    def test_full_properties(self, full_encoder):
        """Test full encoder properties."""
        assert full_encoder.dim == 512
        assert full_encoder.sample_rate == 24000

    def test_encode_shape(self, semantic_encoder):
        """Test encode output shape."""
        audio = torch.randn(2, 24000 * 5)  # 5 seconds, batch size 2
        features = semantic_encoder.encode(audio)

        assert features.dim() == 3
        assert features.shape[0] == 2  # batch size
        assert features.shape[2] == 512  # embedding dim
        assert features.shape[1] > 0  # frames

    def test_invalid_mode(self):
        """Test invalid mode raises error."""
        with pytest.raises(ValueError):
            MimiEncoder(mode="invalid")


class TestGetEncoder:
    """Tests for encoder factory function."""

    def test_get_htsat(self):
        """Test getting HTSAT encoder."""
        encoder = get_encoder("htsat")
        assert isinstance(encoder, HTSATEncoder)

    def test_get_mimi_semantic(self):
        """Test getting Mimi semantic encoder."""
        encoder = get_encoder("mimi_semantic")
        assert isinstance(encoder, MimiEncoder)
        assert encoder.mode == "semantic"

    def test_get_mimi_full(self):
        """Test getting Mimi full encoder."""
        encoder = get_encoder("mimi_full")
        assert isinstance(encoder, MimiEncoder)
        assert encoder.mode == "full"

    def test_unknown_encoder(self):
        """Test unknown encoder raises error."""
        with pytest.raises(ValueError):
            get_encoder("unknown")

    def test_all_encoders_implement_interface(self):
        """Test all encoders implement AudioEncoder interface."""
        for name in ["htsat", "mimi_semantic", "mimi_full"]:
            encoder = get_encoder(name)
            assert isinstance(encoder, AudioEncoder)
            assert hasattr(encoder, "encode")
            assert hasattr(encoder, "dim")
            assert hasattr(encoder, "sample_rate")
