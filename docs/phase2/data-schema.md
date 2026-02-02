# Phase 2 Data Schema Reference

Quick reference card for Phase 2 data structures and file formats.

---

## File Types Overview

```
AutoConjecture/data/
├── checkpoints/
│   ├── *.json          → Knowledge Base Checkpoints
│   ├── generator_*.pt  → Neural Model Checkpoints
│   └── trainer_*.pt    → Training State Checkpoints
├── logs/
│   ├── *.log          → Text Logs
│   └── metrics.json   → Training Metrics
└── experiments/
    └── phase2_*/      → Experiment Data
```

---

## 1. Knowledge Base Schema (`*.json`)

### File Structure
```
neural_epoch_{epoch}_cycle_{cycle}.json
```

### JSON Schema
```json
{
  "axioms": [
    "∀x.¬(S(x) = 0)",
    "∀x.∀y.((S(x) = S(y)) → (x = y))",
    "∀x.((x + 0) = x)",
    "∀x.∀y.((x + S(y)) = S((x + y)))",
    "∀x.((x * 0) = 0)",
    "∀x.∀y.((x * S(y)) = ((x * y) + x))"
  ],
  "theorems": [
    {
      "statement": "string",           // Mathematical statement
      "proof_length": 0,               // Number of proof steps
      "proof_steps": ["string"],       // List of tactic applications
      "complexity": 0.0,               // Complexity score (2-30+)
      "timestamp": "ISO-8601",         // Discovery timestamp
      "epoch": 0,                      // Training epoch
      "cycle": 0                       // Training cycle
    }
  ],
  "metadata": {
    "num_theorems": 0,
    "saved_at": "ISO-8601"
  }
}
```

### Python Type
```python
@dataclass
class Theorem:
    statement: Expression          # The proven statement
    proof: Proof                   # Proof object
    complexity: float              # Complexity estimate
    timestamp: str                 # ISO timestamp
    epoch: int                     # Training epoch
    cycle: int                     # Training cycle
```

---

## 2. Neural Model Schema (`generator_*.pt`)

### File Structure
```
generator_epoch_{epoch}_cycle_{cycle}.pt
```

### PyTorch Checkpoint
```python
{
  'model_state_dict': OrderedDict({
    'embedding.weight': Tensor[vocab_size, d_model],
    'transformer.layers.0.self_attn.in_proj_weight': Tensor[...],
    'transformer.layers.0.self_attn.out_proj.weight': Tensor[...],
    'transformer.layers.0.linear1.weight': Tensor[...],
    'transformer.layers.0.linear2.weight': Tensor[...],
    # ... 6 layers total
    'output_projection.weight': Tensor[d_model, vocab_size],
  }),
  'tokenizer_vocab': {
    '<PAD>': 0,
    '<SOS>': 1,
    '<EOS>': 2,
    '<UNK>': 3,
    'VAR': 4,
    # ... full vocabulary
  },
  'config': {
    'd_model': 256,              # Embedding dimension
    'nhead': 8,                  # Attention heads
    'num_layers': 6,             # Transformer layers
    'dropout': 0.1,              # Dropout rate
    'vocab_size': 20,            # Vocabulary size
    'max_seq_len': 128           # Max sequence length
  }
}
```

### Model Architecture
```
TransformerGenerator (12M parameters)
├── Embedding Layer [vocab_size, d_model]
├── Positional Encoding [max_seq_len, d_model]
├── Transformer Decoder (6 layers)
│   ├── Multi-Head Self-Attention (8 heads)
│   ├── Feed-Forward Network (4 * d_model)
│   └── Layer Normalization
└── Output Projection [d_model, vocab_size]
```

---

## 3. Training State Schema (`trainer_*.pt`)

### File Structure
```
trainer_epoch_{epoch}_cycle_{cycle}.pt
```

### PyTorch Checkpoint
```python
{
  'optimizer_state_dict': {
    'state': dict,               # Per-parameter state
    'param_groups': [            # Optimizer param groups
      {
        'lr': 0.0001,            # Learning rate
        'betas': (0.9, 0.999),   # AdamW betas
        'eps': 1e-08,            # Epsilon
        'weight_decay': 0.01     # Weight decay
      }
    ]
  },
  'scheduler_state_dict': {
    'base_lrs': [0.0001],
    'last_epoch': 0,
    '_step_count': 0
  },
  'training_step': 0,            # Current step
  'best_loss': float('inf'),     # Best validation loss
  'curriculum_state': {
    'current_stage': 0,          # Current curriculum stage
    'stage_results': [           # Results per stage
      {
        'successes': 0,
        'attempts': 0,
        'complexities': []
      }
    ],
    'current_complexity': (2, 4) # (min, max) complexity
  }
}
```

---

## 4. Training Metrics Schema

### Metrics JSON
```json
{
  "epoch": 0,
  "cycle": 0,
  "total_conjectures_generated": 0,
  "total_proofs_attempted": 0,
  "total_proofs_succeeded": 0,
  "success_rate": 0.0,
  "knowledge_base_size": 0,
  "num_axioms": 6,
  "num_theorems": 0,
  "total_statements": 6,
  "avg_proof_length": 0.0,
  "avg_complexity": 0.0,
  "min_complexity": 0.0,
  "max_complexity": 0.0,
  "curriculum": {
    "current_stage": 0,
    "current_complexity": [2, 4],
    "current_temperature": 1.5,
    "stage_success_rate": 0.0
  }
}
```

---

## 5. Proof Object Schema

### Proof Structure
```python
@dataclass
class ProofStep:
    tactic: str                    # "simplify", "rewrite", etc.
    before: Expression             # Expression before step
    after: Expression              # Expression after step
    justification: str             # Reason for step

@dataclass
class Proof:
    steps: List[ProofStep]         # Sequence of proof steps
    result: ProofResult            # SUCCESS, FAILURE, TIMEOUT

    def length(self) -> int:
        return len(self.steps)
```

### Proof Result Types
```python
class ProofResult(Enum):
    SUCCESS = "success"            # Proof completed
    FAILURE = "failure"            # Cannot prove
    TIMEOUT = "timeout"            # Exceeded time limit
```

---

## 6. Expression Schema

### Expression Types
```python
# Base types
class Expression:
    def complexity(self) -> int
    def __str__(self) -> str

# Concrete types
@dataclass
class Var(Expression):
    name: str                      # "x", "y", "z", "w"

@dataclass
class Zero(Expression):
    pass                           # Constant 0

@dataclass
class Succ(Expression):
    arg: Expression                # S(x)

@dataclass
class Add(Expression):
    left: Expression               # (x + y)
    right: Expression

@dataclass
class Mul(Expression):
    left: Expression               # (x * y)
    right: Expression

@dataclass
class Eq(Expression):
    left: Expression               # (x = y)
    right: Expression

@dataclass
class Forall(Expression):
    var: str                       # ∀x.(...)
    body: Expression
```

### Example Expression Tree
```
Expression: ∀x.((x + 0) = x)

Tree:
Forall(
  var="x",
  body=Eq(
    left=Add(
      left=Var("x"),
      right=Zero()
    ),
    right=Var("x")
  )
)

String: "∀x.((x + 0) = x)"
Complexity: 10
```

---

## 7. Tokenization Schema

### Vocabulary
```python
SPECIAL_TOKENS = {
    '<PAD>': 0,    # Padding
    '<SOS>': 1,    # Start of sequence
    '<EOS>': 2,    # End of sequence
    '<UNK>': 3     # Unknown token
}

CONSTRUCTORS = {
    'VAR': 4,      # Variable reference
    'ZERO': 5,     # Zero constant
    'SUCC': 6,     # Successor function
    'ADD': 7,      # Addition operator
    'MUL': 8,      # Multiplication operator
    'EQ': 9,       # Equality predicate
    'FORALL': 10   # Universal quantifier
}

VARIABLES = {
    'var_x': 11,
    'var_y': 12,
    'var_z': 13,
    'var_w': 14
}

STRUCTURAL = {
    '(': 15,
    ')': 16
}
```

### Tokenization Example
```
Expression: 0 = (0 + 0)

Prefix notation:
EQ ZERO (ADD ZERO ZERO)

Tokens:
[<SOS>, EQ, ZERO, (, ADD, ZERO, ZERO, ), <EOS>]

Token IDs:
[1, 9, 5, 15, 7, 5, 5, 16, 2]
```

---

## 8. Curriculum Schema

### Curriculum Stages
```python
Stage 0:  complexity = (2, 4),   temp = 1.50
Stage 1:  complexity = (4, 6),   temp = 1.35
Stage 2:  complexity = (6, 8),   temp = 1.20
Stage 3:  complexity = (8, 10),  temp = 1.05
Stage 4:  complexity = (10, 12), temp = 0.90
Stage 5+: complexity = (12, 15), temp = 0.80
```

### Advancement Criteria
```python
{
  'min_samples_per_stage': 100,      # Minimum attempts
  'success_threshold': 0.3,          # 30% success rate
  'progression': 'success_rate'      # Advance on success
}
```

---

## 9. File Naming Conventions

### Checkpoint Files
```
Pattern: {type}_epoch_{epoch}_cycle_{cycle}.{ext}

Examples:
- neural_epoch_0_cycle_0.json
- neural_epoch_5_cycle_1000.json
- generator_epoch_5_cycle_1000.pt
- trainer_epoch_5_cycle_1000.pt
```

### Log Files
```
Pattern: {experiment_name}_{timestamp}.log

Examples:
- neural_training_20260201_131408.log
- phase2_gpu_exp_20260201_131258.log
```

### Experiment Directories
```
Pattern: {experiment_name}_{YYYYMMDD}_{HHMMSS}/

Examples:
- phase2_test_20260201_131408/
- phase2_gpu_exp_20260201_131258/
```

---

## 10. Data Sizes

### Typical Sizes (After 10 Epochs)

| File Type | Size | Count |
|-----------|------|-------|
| Knowledge Base JSON | 100KB - 2MB | 1 per checkpoint |
| Generator Model PT | 45-50MB | 1 per checkpoint |
| Trainer State PT | 90-100MB | 1 per checkpoint |
| Log File | 10-50MB | 1 per run |
| Total per Checkpoint | ~140-150MB | - |

### Checkpoint Frequency
- Default: Every 1000 cycles
- 10 epochs × 500 cycles = 5000 cycles
- Total checkpoints: ~5 per run
- Total storage: ~700MB per training run

---

## 11. Data Flow Diagram

```
┌─────────────────────────────────────────────────────┐
│                   Training Loop                      │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│              Neural Generator Model                  │
│  (TransformerGenerator: ~12M parameters)            │
└─────────────────────────────────────────────────────┘
                          │
                          ▼ generates
┌─────────────────────────────────────────────────────┐
│                  Conjectures                         │
│  (List of Expression objects)                       │
└─────────────────────────────────────────────────────┘
                          │
                          ▼ filters
┌─────────────────────────────────────────────────────┐
│              Filtered Conjectures                    │
│  (Novelty, Diversity, Complexity filters)           │
└─────────────────────────────────────────────────────┘
                          │
                          ▼ proves
┌─────────────────────────────────────────────────────┐
│                  Proof Engine                        │
│  (Attempts to prove each conjecture)                │
└─────────────────────────────────────────────────────┘
                          │
                          ▼ success
┌─────────────────────────────────────────────────────┐
│               Proven Theorems                        │
│  (Theorem objects with Proof objects)               │
└─────────────────────────────────────────────────────┘
                          │
                          ▼ stores
┌─────────────────────────────────────────────────────┐
│               Knowledge Base                         │
│  (Persistent storage: JSON files)                   │
└─────────────────────────────────────────────────────┘
                          │
                          ▼ trains
┌─────────────────────────────────────────────────────┐
│              Online Learning                         │
│  (Updates neural model with new proofs)             │
└─────────────────────────────────────────────────────┘
```

---

## 12. Quick Access Methods

### Load Knowledge Base
```python
from AutoConjecture.src.knowledge.knowledge_base import KnowledgeBase

kb = KnowledgeBase()
kb.load("data/checkpoints/neural_epoch_5_cycle_1000.json")
print(kb.get_statistics())
```

### Load Neural Model
```python
import torch
from AutoConjecture.src.generation.neural_generator import NeuralConjectureGenerator

generator = NeuralConjectureGenerator.load("data/checkpoints/generator_epoch_5_cycle_1000.pt")
conjectures = generator.generate(10)
```

### Access Metrics
```python
import json

with open("data/logs/metrics.json") as f:
    metrics = json.load(f)
print(f"Success rate: {metrics['success_rate']:.2%}")
```

---

## 13. Data Integrity

### Validation Checks
- ✅ All JSON files are valid JSON
- ✅ All expressions are syntactically valid
- ✅ All proofs are verified before storage
- ✅ All theorems have unique statements
- ✅ All checkpoints are resumable

### Recovery Methods
```python
# If checkpoint is corrupted, load previous checkpoint
kb = KnowledgeBase()
try:
    kb.load("data/checkpoints/neural_epoch_5_cycle_1000.json")
except:
    kb.load("data/checkpoints/neural_epoch_4_cycle_500.json")
```

---

## Summary Table

| Data Type | Format | Size | Frequency | Purpose |
|-----------|--------|------|-----------|---------|
| Knowledge Base | JSON | 100KB-2MB | Every 1000 cycles | Store theorems |
| Neural Model | PyTorch | 45-50MB | Every 1000 cycles | Store model weights |
| Training State | PyTorch | 90-100MB | Every 1000 cycles | Resume training |
| Metrics | JSON | <1MB | Continuous | Track progress |
| Logs | Text | 10-50MB | Continuous | Debugging |

---

**Last Updated**: 2026-02-02
**Version**: Phase 2 Complete ✅
