# Phase 2: Neural Conjecture Generator

## Overview

Phase 2 replaces the random conjecture generator with a **transformer-based neural model** that learns to generate provable mathematical theorems. The system uses curriculum learning to progressively increase difficulty from simple to complex conjectures.

## Key Features

### 1. Transformer-Based Generation
- **Architecture**: Decoder-only transformer (similar to GPT)
- **Vocabulary**: Custom tokenization for logical expressions
- **Autoregressive**: Generates expressions token-by-token
- **Configurable**: Adjustable model size, layers, attention heads

### 2. Curriculum Learning
- **Progressive Difficulty**: Starts with simple expressions, gradually increases complexity
- **Adaptive**: Advances stages based on proof success rate
- **Temperature Scheduling**: High exploration early, low exploitation later
- **Complexity Filtering**: Filters generated conjectures by current curriculum stage

### 3. Training Strategies
- **Supervised Pretraining**: Learn from existing proven theorems
- **Online Learning**: Continuously update from new successful proofs
- **Reinforcement Learning**: Higher reward for provable conjectures (future enhancement)

## Architecture

### Component Overview

```
Phase 2 Components:
├── src/models/
│   ├── tokenizer.py              # Expression tokenization
│   ├── transformer_generator.py  # Transformer model
│   ├── generator_trainer.py      # Training logic
│   └── curriculum.py             # Curriculum learning
├── src/generation/
│   └── neural_generator.py       # Neural generator (wraps transformer)
├── src/training/
│   └── neural_training_loop.py   # Main training loop with curriculum
└── scripts/
    └── train_neural.py           # Training script
```

### Expression Tokenization

Logical expressions are converted to token sequences:

```
Expression: ∀x.(x + 0 = x)
Tokens: [SOS, FORALL, (, var_x, EQ, (, ADD, (, VAR, var_x, ZERO, ), VAR, var_x, ), ), EOS]
```

**Vocabulary**:
- Special tokens: `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`
- Constructors: `VAR`, `ZERO`, `SUCC`, `ADD`, `MUL`, `EQ`, `FORALL`
- Variables: `var_x`, `var_y`, `var_z`, `var_w`
- Structural: `(`, `)`

### Transformer Model

**Decoder-only architecture**:
- Token embedding layer
- Positional encoding
- Multiple transformer decoder layers with self-attention
- Output projection to vocabulary

**Default Configuration**:
- Embedding dimension: 256
- Attention heads: 8
- Layers: 6
- Parameters: ~12M

### Curriculum Learning

**Stage Progression**:
1. **Stage 0**: Complexity 2-4 (simple equations like `0 = 0`)
2. **Stage 1**: Complexity 4-6 (basic arithmetic)
3. **Stage 2**: Complexity 6-8 (quantified statements)
4. **Stage 3+**: Gradually increase to complexity 15+

**Advancement Criteria**:
- Minimum 100 samples per stage
- Success rate > 30%
- Automatic progression when criteria met

**Temperature Schedule**:
- Initial: 1.5 (high exploration)
- Final: 0.8 (low exploitation)
- Linearly interpolated across stages

## Installation

Dependencies already in `requirements.txt`:
```bash
torch>=2.0.0
transformers>=4.36.0
```

Install if not already done:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

Train with default settings:
```bash
python scripts/train_neural.py
```

### Custom Configuration

Use custom config file:
```bash
python scripts/train_neural.py --config configs/phase2_neural.yaml
```

### Override Parameters

Override specific parameters:
```bash
python scripts/train_neural.py \
  --epochs 20 \
  --device cuda \
  --d-model 512 \
  --nhead 16 \
  --pretrain-epochs 10
```

### Curriculum Settings

Adjust curriculum:
```bash
python scripts/train_neural.py \
  --initial-complexity 2 \
  --final-complexity 20 \
  --conjectures-per-cycle 20
```

Disable curriculum:
```bash
python scripts/train_neural.py --no-curriculum
```

### GPU Training

If CUDA is available:
```bash
python scripts/train_neural.py --device cuda
```

## Training Process

### Phase 1: Supervised Pretraining

If knowledge base contains existing proofs:
1. Load proven theorems from knowledge base
2. Sort by complexity (simple first)
3. Train model to imitate successful proofs
4. Default: 5 epochs of pretraining

### Phase 2: Curriculum Learning

Main training loop:
1. **Generate**: Neural model generates conjectures at current complexity
2. **Filter**: Apply novelty, diversity, and complexity filters
3. **Prove**: Attempt to prove each conjecture
4. **Learn**: Update knowledge base with successful proofs
5. **Update**: Online training on recent successes
6. **Advance**: Progress to next curriculum stage when ready

### Checkpointing

Checkpoints saved every 1000 cycles (configurable):
- `data/checkpoints/neural_epoch_X_cycle_Y.json` - Knowledge base
- `data/checkpoints/generator_epoch_X_cycle_Y.pt` - Generator model
- `data/checkpoints/trainer_epoch_X_cycle_Y.pt` - Training state

## Expected Results

### Success Rate Improvements

**Phase 1 (Random)**: 2-5% success rate
**Phase 2 (Neural)**: Expected 10-30% success rate

- Initial: Similar to random (~5%)
- After pretraining: ~10-15%
- After curriculum: ~20-30%
- Depends on model size and training time

### Training Time

**CPU**:
- Pretraining: ~30 min for 1000 theorems
- Main training: ~2-4 hours for 10 epochs
- Total: ~3-5 hours

**GPU** (CUDA):
- Pretraining: ~5 min
- Main training: ~30-60 min
- Total: ~45-90 min

### Knowledge Base Growth

Expected growth with neural generation:
- **Epoch 1**: 50-100 theorems
- **Epoch 5**: 200-400 theorems
- **Epoch 10**: 500-1000 theorems
- **Epoch 20**: 1000-2000+ theorems

## Model Architecture Details

### Positional Encoding

Sinusoidal positional encoding:
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### Attention Mechanism

Multi-head self-attention with causal masking for autoregressive generation.

### Sampling Strategies

**Temperature Sampling**:
- `temperature > 1`: More random/diverse
- `temperature < 1`: More deterministic/focused
- `temperature = 1`: Standard sampling

**Top-k Sampling**:
- Keep only top k most probable tokens
- Default: k=50

**Top-p (Nucleus) Sampling**:
- Keep tokens with cumulative probability ≤ p
- Optional, disabled by default

## Configuration Reference

### Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 256 | Embedding dimension |
| `nhead` | 8 | Number of attention heads |
| `num_layers` | 6 | Number of transformer layers |
| `dropout` | 0.1 | Dropout rate |
| `max_seq_len` | 128 | Maximum sequence length |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 1e-4 | Adam optimizer learning rate |
| `batch_size` | 32 | Training batch size |
| `warmup_steps` | 500 | Learning rate warmup steps |
| `pretrain_epochs` | 5 | Supervised pretraining epochs |

### Curriculum Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_complexity` | 2 | Starting complexity |
| `final_complexity` | 15 | Maximum complexity |
| `success_threshold` | 0.3 | Advancement threshold |
| `min_samples_per_stage` | 100 | Minimum samples before advance |

## Testing

Run Phase 2 tests:
```bash
pytest tests/test_neural_components.py -v
```

Test specific components:
```bash
# Test tokenizer
pytest tests/test_neural_components.py::TestExpressionTokenizer -v

# Test transformer
pytest tests/test_neural_components.py::TestTransformerGenerator -v

# Test curriculum
pytest tests/test_neural_components.py::TestCurriculumScheduler -v
```

## Troubleshooting

### CUDA Out of Memory

Reduce model size or batch size:
```bash
python scripts/train_neural.py \
  --device cuda \
  --d-model 128 \
  --nhead 4 \
  --num-layers 4 \
  --generator-batch-size 16
```

### Low Success Rate

- Increase pretraining epochs
- Lower initial complexity
- Increase samples per curriculum stage
- Check that knowledge base has good examples

### Slow Training

**On CPU**:
- Reduce model size
- Reduce cycles per epoch
- Reduce conjectures per cycle

**On GPU**:
- Increase batch size
- Use mixed precision training (future)

### Invalid Generations

Model may generate syntactically invalid expressions:
- Normal for untrained model
- Improves with training
- Filtered out automatically
- Monitor ratio of valid/invalid

## Comparison: Phase 1 vs Phase 2

| Aspect | Phase 1 (Random) | Phase 2 (Neural) |
|--------|-----------------|------------------|
| **Generation** | Random sampling | Learned generation |
| **Success Rate** | 2-5% | 10-30% |
| **Diversity** | High | Moderate-High |
| **Efficiency** | Low | Medium-High |
| **Training Time** | None | 3-5 hours (CPU) |
| **Complexity** | Low | Medium |
| **Adaptability** | None | High (learns from success) |

## Future Enhancements (Phase 3+)

**Phase 3: RL-Based Prover**
- Policy network for tactic selection
- PPO training for proving
- Combined generator-prover training

**Phase 4: Advanced Architecture**
- Tree-structured generation
- Grammar-constrained decoding
- Proof-guided generation

**Phase 5: Scaling**
- Larger models (50M-100M+ parameters)
- Distributed training
- Multi-GPU support

## References

- **Transformer Architecture**: "Attention Is All You Need" (Vaswani et al., 2017)
- **Curriculum Learning**: "Curriculum Learning" (Bengio et al., 2009)
- **Autoregressive Generation**: GPT architecture
- **Mathematical Reasoning**: AlphaZero self-play approach

## Contributing

To extend Phase 2:

1. **New Architectures**: Modify `src/models/transformer_generator.py`
2. **Curriculum Strategies**: Extend `src/models/curriculum.py`
3. **Training Algorithms**: Update `src/models/generator_trainer.py`
4. **Sampling Methods**: Enhance generation in `TransformerGenerator.generate()`

## License

MIT License - See LICENSE file
