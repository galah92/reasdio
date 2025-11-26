"""Audio Language Model for audio question answering.

Architecture:
    Audio → Encoder → Mapper → LM ← Question → Answer

The mapper projects encoder features to LM embedding space.
The LM (SmolLM2-135M) is frozen and generates answers conditioned on
audio features and the question.
"""

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from .encoders import AudioEncoder, get_encoder


class Mapper(nn.Module):
    """Projects audio encoder features to LM embedding space.

    Architecture: Linear → GELU → Linear
    """

    def __init__(self, encoder_dim: int, lm_dim: int, hidden_dim: int | None = None):
        """Initialize mapper.

        Args:
            encoder_dim: Input dimension from audio encoder
            lm_dim: Output dimension for LM embeddings
            hidden_dim: Hidden layer dimension (default: 2 * max(encoder_dim, lm_dim))
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 2 * max(encoder_dim, lm_dim)

        self.net = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, lm_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Map encoder features to LM embedding space.

        Args:
            x: (batch, frames, encoder_dim) tensor

        Returns:
            (batch, frames, lm_dim) tensor
        """
        return self.net(x)


class AudioLM(nn.Module):
    """Audio Language Model for audio question answering.

    Combines an audio encoder, mapper, and frozen LM to answer questions
    about audio. The audio features are prepended to the question tokens
    as a soft prefix.
    """

    def __init__(
        self,
        encoder: AudioEncoder | str = "htsat",
        lm_name: str = "HuggingFaceTB/SmolLM2-135M",
        freeze_encoder: bool = True,
        freeze_lm: bool = True,
        mapper_hidden_dim: int | None = None,
    ):
        """Initialize AudioLM.

        Args:
            encoder: AudioEncoder instance or encoder name ("htsat", "mimi_semantic", "mimi_full")
            lm_name: HuggingFace model name for the LM
            freeze_encoder: If True, freeze encoder weights
            freeze_lm: If True, freeze LM weights
            mapper_hidden_dim: Hidden dimension for mapper (default: auto)
        """
        super().__init__()

        # Load encoder
        if isinstance(encoder, str):
            self.encoder = get_encoder(encoder, freeze=freeze_encoder)
        else:
            self.encoder = encoder

        # Load LM and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(lm_name)
        self.lm = AutoModelForCausalLM.from_pretrained(lm_name)

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Get LM embedding dimension
        lm_dim = self.lm.config.hidden_size

        # Create mapper
        self.mapper = Mapper(
            encoder_dim=self.encoder.dim,
            lm_dim=lm_dim,
            hidden_dim=mapper_hidden_dim,
        )

        # Freeze LM if requested
        if freeze_lm:
            for param in self.lm.parameters():
                param.requires_grad = False

        self._freeze_lm = freeze_lm

    def train(self, mode: bool = True):
        """Override train to keep frozen modules in eval mode."""
        super().train(mode)
        if self._freeze_lm:
            self.lm.eval()
        return self

    @property
    def device(self) -> torch.device:
        """Get model device."""
        return next(self.parameters()).device

    def encode_audio(self, waveforms: Tensor) -> Tensor:
        """Encode audio waveforms to LM embedding space.

        Args:
            waveforms: (batch, samples) tensor at encoder's sample rate

        Returns:
            (batch, frames, lm_dim) tensor of audio embeddings
        """
        # Encode audio to frame features
        audio_features = self.encoder.encode(waveforms)  # (B, frames, encoder_dim)

        # Map to LM embedding space
        audio_embeds = self.mapper(audio_features)  # (B, frames, lm_dim)

        return audio_embeds

    def prepare_inputs(
        self,
        audio_embeds: Tensor,
        questions: list[str],
        answers: list[str] | None = None,
    ) -> dict:
        """Prepare inputs for the LM by combining audio and text.

        Args:
            audio_embeds: (batch, frames, lm_dim) audio embeddings
            questions: List of question strings
            answers: Optional list of answer strings (for training)

        Returns:
            Dict with inputs_embeds, attention_mask, labels (if training)
        """
        batch_size = audio_embeds.shape[0]
        device = audio_embeds.device

        # Format text: "Question: {question}\nAnswer: {answer}"
        if answers is not None:
            texts = [
                f"Question: {q}\nAnswer: {a}"
                for q, a in zip(questions, answers)
            ]
        else:
            texts = [f"Question: {q}\nAnswer:" for q in questions]

        # Tokenize text
        text_tokens = self.tokenizer(
            texts,
            padding=True,
            return_tensors="pt",
        ).to(device)

        # Get text embeddings from LM
        text_embeds = self.lm.get_input_embeddings()(text_tokens.input_ids)

        # Concatenate: [audio_embeds, text_embeds]
        inputs_embeds = torch.cat([audio_embeds, text_embeds], dim=1)

        # Create attention mask: 1 for all audio frames + text attention mask
        audio_attention = torch.ones(
            batch_size, audio_embeds.shape[1], device=device, dtype=torch.long
        )
        attention_mask = torch.cat([audio_attention, text_tokens.attention_mask], dim=1)

        result = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
        }

        # Create labels for training (only supervise answer tokens)
        if answers is not None:
            # Labels: -100 for audio frames and question, answer token ids for answer
            # We need to find where "Answer:" ends in each sequence

            # First, tokenize just the questions to find the answer start
            question_texts = [f"Question: {q}\nAnswer:" for q in questions]
            question_tokens = self.tokenizer(
                question_texts,
                padding=True,
                return_tensors="pt",
            ).to(device)

            # Create labels tensor
            labels = torch.full_like(
                torch.zeros(batch_size, inputs_embeds.shape[1], dtype=torch.long, device=device),
                fill_value=-100,
            )

            # For each sample, set labels for the answer portion
            for i in range(batch_size):
                # Number of audio frames
                n_audio = audio_embeds.shape[1]

                # Question token length (including "Answer:")
                q_len = question_tokens.attention_mask[i].sum().item()

                # Full text token length
                full_len = text_tokens.attention_mask[i].sum().item()

                # Answer starts after audio + question tokens
                answer_start = n_audio + q_len

                # Copy answer token ids as labels
                answer_len = full_len - q_len
                if answer_len > 0:
                    labels[i, answer_start:answer_start + answer_len] = text_tokens.input_ids[
                        i, q_len:q_len + answer_len
                    ]

            result["labels"] = labels

        return result

    def forward(
        self,
        waveforms: Tensor,
        questions: list[str],
        answers: list[str] | None = None,
    ) -> dict:
        """Forward pass.

        Args:
            waveforms: (batch, samples) tensor at encoder's sample rate
            questions: List of question strings
            answers: Optional list of answer strings (for training)

        Returns:
            Dict with loss (if training) and logits
        """
        # Encode audio
        audio_embeds = self.encode_audio(waveforms)

        # Prepare inputs
        inputs = self.prepare_inputs(audio_embeds, questions, answers)

        # Forward through LM
        outputs = self.lm(
            inputs_embeds=inputs["inputs_embeds"],
            attention_mask=inputs["attention_mask"],
            labels=inputs.get("labels"),
        )

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }

    @torch.no_grad()
    def generate(
        self,
        waveforms: Tensor,
        questions: list[str],
        max_new_tokens: int = 50,
        **generate_kwargs,
    ) -> list[str]:
        """Generate answers for audio questions.

        Args:
            waveforms: (batch, samples) tensor at encoder's sample rate
            questions: List of question strings
            max_new_tokens: Maximum tokens to generate
            **generate_kwargs: Additional args for LM.generate()

        Returns:
            List of generated answer strings
        """
        # Encode audio
        audio_embeds = self.encode_audio(waveforms)

        # Prepare inputs (no answers for generation)
        inputs = self.prepare_inputs(audio_embeds, questions)

        # Generate
        # Note: We need to handle the fact that we're using inputs_embeds
        # The LM's generate expects input_ids, so we use a workaround

        # Get position of the last token to start generation from
        batch_size = audio_embeds.shape[0]
        seq_len = inputs["inputs_embeds"].shape[1]

        # For generation with inputs_embeds, we'll do a simple autoregressive loop
        generated_ids = []

        for _ in range(max_new_tokens):
            outputs = self.lm(
                inputs_embeds=inputs["inputs_embeds"],
                attention_mask=inputs["attention_mask"],
            )

            # Get next token prediction
            next_token_logits = outputs.logits[:, -1, :]
            next_tokens = next_token_logits.argmax(dim=-1)

            generated_ids.append(next_tokens)

            # Check if all sequences have generated EOS
            if (next_tokens == self.tokenizer.eos_token_id).all():
                break

            # Append next token embedding to inputs
            next_embeds = self.lm.get_input_embeddings()(next_tokens).unsqueeze(1)
            inputs["inputs_embeds"] = torch.cat([inputs["inputs_embeds"], next_embeds], dim=1)
            inputs["attention_mask"] = torch.cat(
                [inputs["attention_mask"], torch.ones(batch_size, 1, device=self.device, dtype=torch.long)],
                dim=1,
            )

        # Decode generated tokens
        if generated_ids:
            generated_ids = torch.stack(generated_ids, dim=1)
            answers = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        else:
            answers = [""] * batch_size

        return answers
