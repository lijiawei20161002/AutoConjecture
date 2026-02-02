# AutoConjecture

An AI system that discovers mathematical theorems from scratch using self-play reinforcement learning.

## Overview

AutoConjecture implements a generate-prove-learn cycle:

1. **Generate**: Produces novel mathematical conjectures about Peano arithmetic
2. **Prove**: Attempts to prove conjectures using automated theorem proving
3. **Learn**: Stores proven theorems and uses them to prove more complex statements

The system starts with only Peano axioms and progressively builds a knowledge base of proven theorems.

## Features

- **Custom Proof System**: Built from scratch with formal logic representation
- **Automated Theorem Prover**: Best-first search with configurable tactics
- **Knowledge Base**: Persistent storage of proven theorems
- **Novelty Detection**: Ensures generated conjectures are diverse and interesting
- **Comprehensive Monitoring**: Tracks success rates, complexity, and proof statistics

## Project Structure

```
AutoConjecture/
├── src/
│   ├── logic/          # Formal logic system (terms, expressions, axioms, parser)
│   ├── prover/         # Theorem prover (tactics, proof engine)
│   ├── generation/     # Conjecture generation (random, neural, novelty)
│   ├── models/         # Neural models (transformer, tokenizer, curriculum)
│   ├── knowledge/      # Knowledge base storage
│   ├── training/       # Training loops (Phase 1 & 2)
│   ├── monitoring/     # Logging and metrics
│   └── utils/          # Utilities
├── scripts/
│   ├── train.py        # Phase 1 training script
│   └── train_neural.py # Phase 2 training script
├── configs/
│   ├── default.yaml    # Phase 1 configuration
│   └── phase2_neural.yaml  # Phase 2 configuration
├── data/
│   ├── checkpoints/    # Saved knowledge bases & models
│   ├── logs/           # Training logs
│   └── proofs/         # Saved proofs
├── docs/               # Documentation (see docs/README.md)
│   ├── getting-started/    # Quick start guides
│   ├── phase2/            # Phase 2 documentation
│   ├── examples/          # Example theorems
│   └── development/       # Development notes
└── tests/              # Unit tests
    └── phase2/         # Phase 2 (Neural) tests
```

## Documentation

Comprehensive documentation is available in the [`docs/`](docs/) directory:

- **[Getting Started](docs/getting-started/)** - Quick start guides and tutorials
- **[Phase 2](docs/phase2/)** - Neural generator documentation
- **[Examples](docs/examples/)** - Real theorem examples
- **[Development](docs/development/)** - Technical notes

**Quick Links:**
- 📖 [Documentation Index](docs/README.md)
- 🚀 [Quick Start Guide](docs/getting-started/quickstart.md)
- 🧠 [Phase 2 Quick Start](docs/getting-started/quickstart-phase2.md)
- 📊 [Phase 2 Data Documentation](docs/phase2/data-documentation.md)
- 💡 [Demo Proofs](docs/examples/demo-proofs.md)

## Installation

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install package in development mode:
```bash
pip install -e .
```

## Quick Start

### Basic Training

Run training with default configuration:
```bash
python3 scripts/train.py
```

### Custom Configuration

Specify a config file:
```bash
python3 scripts/train.py --config configs/default.yaml
```

Override specific parameters:
```bash
python3 scripts/train.py --epochs 5 --cycles 500
```

Named experiment:
```bash
python3 scripts/train.py --experiment-name my_experiment
```

## Configuration

Edit `configs/default.yaml` to customize:

- **Training parameters**: epochs, cycles per epoch, random seed
- **Generation**: complexity range, conjectures per cycle
- **Prover**: max proof depth, max iterations
- **Logging**: log interval, checkpoint interval

## System Architecture

### Logic System (`src/logic/`)

Implements first-order logic over Peano arithmetic:
- **Terms**: Variables, Zero, Successor, Addition, Multiplication
- **Expressions**: Equations, Quantifiers (∀, ∃), Logical connectives (∧, ∨, ¬, →)
- **Axioms**: Peano axioms defining natural numbers

### Prover System (`src/prover/`)

Automated theorem prover with:
- **Tactics**: Rewrite, Substitute, Simplify, Reflexivity, Assumption
- **Proof Engine**: Best-first search through proof states
- **Heuristics**: Guides search toward simpler goals

### Generation System (`src/generation/`)

Generates interesting conjectures:
- **Random Generator** (Phase 1): Creates valid random expressions
- **Neural Generator** (Phase 2): Transformer-based learned generation
- **Novelty Scorer**: Measures uniqueness using edit distance
- **Complexity Estimator**: Estimates proof difficulty
- **Diversity Filter**: Maintains variety in generated statements

### Neural Models (`src/models/`)

Phase 2 neural components:
- **Tokenizer**: Converts logical expressions to token sequences
- **Transformer Generator**: Decoder-only transformer for autoregressive generation
- **Generator Trainer**: Supervised and reinforcement learning for the generator
- **Curriculum Scheduler**: Manages progressive difficulty increase

### Training Loop (`src/training/`)

Orchestrates the generate-prove-learn cycle:
1. Generate N conjectures per cycle
2. Filter for novelty, complexity, and diversity
3. Attempt to prove each conjecture
4. Add successful proofs to knowledge base
5. Update metrics and checkpoints

## Demo Proofs

See **[DEMO_PROOFS.md](DEMO_PROOFS.md)** for a comprehensive showcase of **18 theorems** automatically discovered by the system from a 200-cycle training run, including:

- Detailed proof traces
- Complexity analysis
- Pattern identification
- Training statistics

Also available:
- **[demo_theorems.json](data/demo_theorems.json)** - Structured JSON format for programmatic access
- **[scripts/display_proofs.py](scripts/display_proofs.py)** - Script to display proofs from checkpoints

### Display Proofs

View proven theorems from any checkpoint:
```bash
# Display most recent proofs
python3 scripts/display_proofs.py

# Display specific checkpoint
python3 scripts/display_proofs.py data/checkpoints/epoch_0_cycle_49.json
```

## Examples

### Simple Theorems

The system has discovered theorems like:
```
0 = (0 * S((0 + z)))                              # Zero multiplication
(0 + w) = (((0 * w) * 0) + w)                     # Addition identity
((((x + 0) + y) * ((z + 0) * (w * 0))) = (y * 0)  # Multi-variable zero product
```

### Monitoring Progress

Training logs show:
```
Epoch 1, Cycle 100: KB size = 15, Success rate = 12.3%, Epoch proofs = 12
✓ Proved: ∀x.(x + 0 = x)
  Proof length: 3 steps
```

### Knowledge Base Growth

The system tracks:
- Number of proven theorems
- Average proof length
- Complexity distribution
- Success rate over time

## Current Phase: Phase 2 (Neural Generator) ✅

**Phase 1 (MVP)** - Complete:
- ✅ Complete formal logic system
- ✅ Rule-based theorem prover
- ✅ Random conjecture generator
- ✅ Knowledge base storage
- ✅ Training loop skeleton
- ✅ Basic monitoring

**Phase 2 (Neural Generator)** - Complete:
- ✅ Transformer-based neural generator
- ✅ Expression tokenizer for logical formulas
- ✅ Curriculum learning (simple to complex)
- ✅ Supervised pretraining on proven theorems
- ✅ Online learning from new proofs
- ✅ Adaptive temperature scheduling

See **[PHASE2_NEURAL.md](PHASE2_NEURAL.md)** for complete Phase 2 documentation.

### Quick Start - Phase 2

Train with neural generator:
```bash
python3 scripts/train_neural.py --device cuda  # Use CPU if no GPU
```

## Future Phases

### Phase 3: RL-Based Prover
- Replace search-based prover with policy network
- PPO training for tactic selection
- Experience replay from successful proofs

### Phase 3: RL-Based Prover
- Replace search-based prover with policy network
- PPO training for tactic selection
- Experience replay from successful proofs

### Phase 4: Full Monitoring
- Visualization of training curves
- Proof tree visualization
- Interactive web dashboard

### Phase 5: Optimization
- GPU acceleration
- Multiprocessing for parallel proofs
- Advanced curriculum strategies

## Technical Details

### Computational Requirements

**Current (Phase 1 - CPU only)**:
- CPU: Multi-core recommended (4-8 cores)
- RAM: 8-16GB
- Storage: 10-20GB for checkpoints
- Training time: Initial results in 2-4 hours

**Future (with neural networks)**:
- GPU: NVIDIA GPU with 8GB+ VRAM recommended
- RAM: 16-32GB
- Storage: 50-100GB
- Training time: Results in hours, convergence in days

### Dependencies

Core:
- Python 3.9+
- PyYAML (configuration)
- pytest (testing)

Future:
- PyTorch (neural networks)
- Transformers (models)
- Matplotlib/Plotly (visualization)

## Development

### Running Tests

```bash
python3 -m pytest tests/ -v
```

### Adding New Tactics

See `src/prover/tactics.py` for examples. Tactics must:
1. Inherit from `Tactic` base class
2. Implement `apply(state, knowledge_base)` method
3. Return list of new proof states (or empty for QED)

### Extending the Logic System

To add new term types:
1. Add class in `src/logic/terms.py`
2. Implement `substitute()`, `free_vars()`, `complexity()`
3. Update parser in `src/logic/parser.py`

## Troubleshooting

**ImportError**: Make sure to install in development mode: `pip install -e .`

**PermissionError when saving**: Check that `data/` directories exist and are writable

**Slow training**: Reduce `cycles_per_epoch` or `max_proof_iterations` in config

**Out of memory**: Reduce `conjectures_per_cycle` or `max_proof_depth`

## Contributing

This is a research project exploring AI for mathematical reasoning. Contributions welcome!

## License

MIT License - See LICENSE file

## Citation

If you use this code in your research, please cite:

```bibtex
@software{autoconjecture2025,
  title={AutoConjecture: AI Mathematical Reasoning from Scratch},
  author={anonymous},
  year={2025},
  url={https://anonymous.4open.science/r/AutoConjecture-11C8}
}
```

## Acknowledgments

Inspired by:
- AlphaZero's self-play approach
- Automated theorem proving research
- Peano arithmetic and formal logic

## Contact

For questions or issues, please open a GitHub issue.
