# Codec Representations for Audio Reasoning

**Research Question**: Does codec-based audio representation (Mimi) outperform classifier-based (HTSAT) for audio reasoning tasks?

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
- [x] Mapper architecture (Linear → GELU → Linear)
- [x] Training loop with evaluation
- [x] **Quick training shows signal**: Loss 3.81 → 2.24, Accuracy 0.2% → 99.4%
- [ ] Mimi encoder integration
- [ ] Full training run
- [ ] Test set evaluation

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
