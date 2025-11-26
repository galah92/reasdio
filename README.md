# Codec Representations for Audio Reasoning

**Research Question**: Does codec-based audio representation (Mimi) outperform classifier-based (HTSAT) for audio reasoning tasks?

## Comparison to Mellow (Baseline)

[Mellow](https://arxiv.org/abs/2503.08540) is the SoTA small audio-language model for reasoning:

| Model | Params | ReasonAQA | MMAU | Audio Encoder |
|-------|--------|-----------|------|---------------|
| Mellow | 167M | **77.0%** | 52.1 | HTSAT (frozen) |
| Qwen2-Audio | 8B | - | 52.5 | - |
| WavLLM | 3B | 59.5% | - | - |
| **Ours (Mimi)** | ~167M | ? | ? | Mimi codec |

**Our hypothesis**: Replacing HTSAT with Mimi codec preserves acoustic details (reverb, timbre, recording quality) that help with reasoning questions about environment, setting, and acoustic properties.

## Task

**Input**: Audio file(s) + text question  
**Output**: Text answer

```
Audio: [explosion.wav]
Question: "What can you infer about the environment?"
Answer: "An outdoor industrial or construction setting"
```

## Architecture

```
Audio → Encoder → Mapper → [LM] ← Question
                              ↓
                           Answer
```

| Component | Implementation |
|-----------|----------------|
| Encoder | HTSAT (baseline) or Mimi codec (experiment) |
| Mapper | Linear → GELU → Linear |
| LM | SmolLM2-135M (frozen initially) |

**Training**: Cross-entropy loss on answer tokens only.

## Data

**ReasonAQA**: https://zenodo.org/records/15036628

```json
{
    "filepath1": "path/to/audio.wav",
    "filepath2": "",
    "input": "question text",
    "answer": "expected answer"
}
```

Audio sources: [AudioCaps](https://github.com/cdjkim/audiocaps), [Clotho](https://zenodo.org/records/4783391)

## Experiments

| ID | Encoder | What we're testing |
|----|---------|-------------------|
| `htsat` | HTSAT framewise features | Baseline (Mellow's approach) |
| `mimi_semantic` | Mimi first RVQ level | Semantic-only codec |
| `mimi_full` | Mimi all 8 RVQ levels | Full codec |

Same hyperparameters across all:
```yaml
batch_size: 16
lr: 1e-4
max_steps: 20000
max_audio_sec: 10
```

## Project Structure

```
src/
├── data.py        # Dataset + collate
├── encoders.py    # HTSAT, Mimi wrappers (same interface)
├── model.py       # Mapper + full AudioLM
└── train.py       # Training loop

scripts/
├── train.py       # CLI entrypoint
└── eval.py        # Evaluation entrypoint

configs/
├── htsat.yaml
└── mimi.yaml
```

## Encoder Interface

All encoders must implement:
```python
class Encoder:
    def encode(self, waveform: Tensor) -> Tensor:
        """(batch, samples) → (batch, frames, dim)"""
    
    @property
    def dim(self) -> int:
        """Output embedding dimension"""
```

## Quick Start

```bash
# Install dependencies
uv sync

# Run quick training (500 steps, ~15 min on L4 GPU)
PYTHONPATH=. uv run python scripts/quick_train.py

# Benchmark training speed
PYTHONPATH=. uv run python scripts/benchmark.py --batch-size 64

# Evaluate a checkpoint
PYTHONPATH=. uv run python scripts/eval.py --checkpoint checkpoints/htsat/quick_train.pt --encoder htsat --test-json data/reasonaqa/clotho_train.json
```

## Current Status

- [x] Data pipeline (ReasonAQA JSONs + audio loading)
- [x] HTSAT encoder integration
- [x] Mimi encoder integration (semantic + full modes)
- [x] Mapper architecture (Linear → GELU → Linear)
- [x] Training loop with evaluation
- [x] **Quick training shows signal**: Loss 3.81 → 2.24
- [ ] Full training run
- [ ] Test set evaluation
- [ ] Encoder comparison experiments

## Validating Mimi vs HTSAT Hypothesis

**Hypothesis**: Mimi's acoustic details (reverb, timbre, recording quality) help with audio *reasoning* tasks that HTSAT's classification features miss.

### What Each Encoder Contains

| Encoder | Output Dim | Contains | Discards |
|---------|-----------|----------|----------|
| HTSAT | 768 | Sound event classification features | Acoustic details |
| Mimi semantic | 512 | Coarse content (what) | Fine acoustic details |
| Mimi full | 512 | Full reconstruction info (what + how) | Nothing |

### Validation Approaches

1. **Probing Experiment** - Train linear probes on frozen encoder outputs to predict:
   - Room type (indoor/outdoor) from reverb
   - Recording quality (professional/amateur)
   - Background noise level
   - Distance to sound source

   *Expected*: Mimi full > Mimi semantic > HTSAT

2. **Question Type Ablation** - Compare accuracy by question category:
   | Question Type | Needs Acoustics? | Expect Mimi Wins? |
   |--------------|------------------|-------------------|
   | "What sound is this?" | No | Similar |
   | "What is the environment?" | Yes | **Yes** |
   | "Is this indoor or outdoor?" | Yes | **Yes** |
   | "Describe the recording quality" | Yes | **Yes** |

3. **Reconstruction Sanity Check** - Mimi can reconstruct audio from codes; HTSAT cannot (by design)

### Relevant Data in ReasonAQA

The train.json (968k samples) includes acoustic reasoning questions:
- "What is the acoustic environment like?"
- "What can be inferred about the environment from the audio?"
- "What is the likely environment or setting for this audio clip?"
- "Is this sound typical of a specific environment?"

These questions require understanding reverb, background noise, and recording characteristics - exactly what Mimi should capture better than HTSAT.

### ReasonAQA Question Categories

| Category | Count | % | Mimi Advantage? |
|----------|-------|---|-----------------|
| MCQ (A/B/C/D) | 348k | 36% | Clean eval metric |
| Compare (dual audio) | 204k | 21% | Acoustic differences |
| Sound Event ID | 194k | 20% | Similar to HTSAT |
| **Acoustic Properties** | 122k | 12.6% | **Yes** - frequency, loudness, timbre |
| **Emotional/Mood** | 43k | 4.5% | **Yes** - atmosphere |
| **Environment/Setting** | 25k | 2.5% | **Yes** - reverb, room acoustics |
| Yes/No | 14k | 1.5% | Clean eval metric |

**Key insight**: ~190k samples (20%) ask about acoustic properties, mood, or environment - exactly where Mimi should outperform HTSAT.

### Evaluation Strategy

1. **Overall ReasonAQA accuracy** - Compare to Mellow's 77.0%
2. **Category breakdown** - Expect Mimi wins on acoustic/environment questions
3. **Per-question-type analysis** - Identify where codec helps most

## Implementation Order

1. **Data**: Load ReasonAQA JSONs + audio files, verify batching works ✓
2. **HTSAT baseline**: Encoder → Mapper → SmolLM2, train until loss decreases ✓
3. **Mimi encoder**: Swap encoder, same training, compare results
4. **Evaluate**: Accuracy on test set, breakdown by question type

## Dependencies

```
torch
transformers  # SmolLM2
torchaudio
msclap        # HTSAT
moshi         # Mimi
```

## References

- ReasonAQA data: https://zenodo.org/records/15036628
- Mellow paper: https://arxiv.org/abs/2503.08540
- Mimi codec: https://huggingface.co/kyutai/mimi
- Codec explainer: https://kyutai.org/codec-explainer
