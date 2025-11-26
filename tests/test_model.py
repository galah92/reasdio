"""Tests for AudioLM model."""

import pytest
import torch

from src.model import AudioLM, Mapper


class TestMapper:
    """Tests for the Mapper module."""

    def test_mapper_forward(self):
        """Test mapper forward pass."""
        mapper = Mapper(encoder_dim=768, lm_dim=576)

        x = torch.randn(2, 10, 768)
        out = mapper(x)

        assert out.shape == (2, 10, 576)

    def test_mapper_custom_hidden_dim(self):
        """Test mapper with custom hidden dimension."""
        mapper = Mapper(encoder_dim=512, lm_dim=576, hidden_dim=1024)

        x = torch.randn(2, 10, 512)
        out = mapper(x)

        assert out.shape == (2, 10, 576)

    def test_mapper_gradient_flow(self):
        """Test that gradients flow through mapper."""
        mapper = Mapper(encoder_dim=768, lm_dim=576)

        x = torch.randn(2, 10, 768, requires_grad=True)
        out = mapper(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestAudioLM:
    """Tests for the AudioLM model."""

    @pytest.fixture
    def model(self):
        """Create AudioLM model for testing."""
        return AudioLM(
            encoder="htsat",
            lm_name="HuggingFaceTB/SmolLM2-135M",
            freeze_encoder=True,
            freeze_lm=True,
        )

    def test_model_creation(self, model):
        """Test model can be created."""
        assert model.encoder is not None
        assert model.lm is not None
        assert model.mapper is not None
        assert model.tokenizer is not None

    def test_encode_audio(self, model):
        """Test audio encoding."""
        # 2 seconds of audio at 48kHz (HTSAT sample rate)
        waveforms = torch.randn(2, 48000 * 2)

        audio_embeds = model.encode_audio(waveforms)

        assert audio_embeds.dim() == 3
        assert audio_embeds.shape[0] == 2  # batch size
        assert audio_embeds.shape[2] == model.lm.config.hidden_size  # LM dim

    def test_prepare_inputs_inference(self, model):
        """Test input preparation for inference."""
        audio_embeds = torch.randn(2, 10, model.lm.config.hidden_size)
        questions = ["What sound is this?", "Describe the audio"]

        inputs = model.prepare_inputs(audio_embeds, questions)

        assert "inputs_embeds" in inputs
        assert "attention_mask" in inputs
        assert "labels" not in inputs  # No labels for inference

        # Check shapes
        assert inputs["inputs_embeds"].shape[0] == 2
        assert inputs["inputs_embeds"].shape[2] == model.lm.config.hidden_size
        assert inputs["attention_mask"].shape == inputs["inputs_embeds"].shape[:2]

    def test_prepare_inputs_training(self, model):
        """Test input preparation for training."""
        audio_embeds = torch.randn(2, 10, model.lm.config.hidden_size)
        questions = ["What sound is this?", "Describe the audio"]
        answers = ["A dog barking", "Birds chirping"]

        inputs = model.prepare_inputs(audio_embeds, questions, answers)

        assert "inputs_embeds" in inputs
        assert "attention_mask" in inputs
        assert "labels" in inputs  # Labels for training

        # Labels should mask audio and question tokens
        # Only answer tokens should have valid labels
        assert (inputs["labels"][:, :10] == -100).all()  # Audio frames masked

    def test_forward_training(self, model):
        """Test forward pass with loss computation."""
        waveforms = torch.randn(2, 48000 * 2)  # 2 seconds at 48kHz
        questions = ["What sound is this?", "Describe the audio"]
        answers = ["A dog barking", "Birds chirping"]

        outputs = model(waveforms, questions, answers)

        assert "loss" in outputs
        assert "logits" in outputs
        assert outputs["loss"] is not None
        assert outputs["loss"].dim() == 0  # Scalar loss

    def test_forward_inference(self, model):
        """Test forward pass without answers (inference mode)."""
        waveforms = torch.randn(2, 48000 * 2)
        questions = ["What sound is this?", "Describe the audio"]

        outputs = model(waveforms, questions)

        assert "loss" in outputs
        assert "logits" in outputs
        assert outputs["loss"] is None  # No loss without answers

    def test_generate(self, model):
        """Test answer generation."""
        waveforms = torch.randn(2, 48000 * 2)
        questions = ["What sound is this?", "Describe the audio"]

        answers = model.generate(waveforms, questions, max_new_tokens=10)

        assert len(answers) == 2
        assert all(isinstance(a, str) for a in answers)

    def test_mapper_is_trainable(self, model):
        """Test that mapper parameters are trainable."""
        trainable_params = [
            name for name, p in model.named_parameters() if p.requires_grad
        ]

        # Mapper should be trainable
        assert any("mapper" in name for name in trainable_params)

    def test_encoder_is_frozen(self, model):
        """Test that encoder is frozen."""
        for name, param in model.encoder.named_parameters():
            assert not param.requires_grad, f"Encoder param {name} is not frozen"

    def test_lm_is_frozen(self, model):
        """Test that LM is frozen."""
        for name, param in model.lm.named_parameters():
            assert not param.requires_grad, f"LM param {name} is not frozen"


class TestAudioLMWithMimi:
    """Tests for AudioLM with Mimi encoder."""

    @pytest.fixture
    def model(self):
        """Create AudioLM with Mimi encoder."""
        return AudioLM(
            encoder="mimi_semantic",
            lm_name="HuggingFaceTB/SmolLM2-135M",
            freeze_encoder=True,
            freeze_lm=True,
        )

    def test_mimi_forward(self, model):
        """Test forward pass with Mimi encoder."""
        # Mimi uses 24kHz
        waveforms = torch.randn(2, 24000 * 2)  # 2 seconds
        questions = ["What sound is this?", "Describe the audio"]
        answers = ["A dog barking", "Birds chirping"]

        outputs = model(waveforms, questions, answers)

        assert outputs["loss"] is not None
        assert outputs["logits"] is not None

    def test_mimi_generate(self, model):
        """Test generation with Mimi encoder."""
        waveforms = torch.randn(2, 24000 * 2)
        questions = ["What sound is this?"]

        answers = model.generate(waveforms[:1], questions[:1], max_new_tokens=5)

        assert len(answers) == 1
        assert isinstance(answers[0], str)
