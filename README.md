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
│   ├── models/         # Neural models (transformer, tokenizer, curriculum, actor-critic)
│   ├── knowledge/      # Knowledge base storage
│   ├── training/       # Training loops (Phase 1, 2, 3 & 5)
│   ├── monitoring/     # Logging, metrics, and Phase 4 visualizer/dashboard
│   └── utils/          # Utilities
├── scripts/
│   ├── train.py        # Phase 1 training script
│   ├── train_neural.py # Phase 2 training script
│   ├── train_phase3.py # Phase 3 training script
│   ├── train_phase5.py # Phase 5 training script
│   └── dashboard.py    # Phase 4 monitoring dashboard launcher
├── configs/
│   ├── default.yaml            # Phase 1 configuration
│   ├── phase2_neural.yaml      # Phase 2 configuration
│   ├── phase3_rl.yaml          # Phase 3 configuration
│   ├── phase4_monitoring.yaml  # Phase 4 dashboard configuration
│   └── phase5_optimization.yaml # Phase 5 optimization configuration
├── data/
│   ├── checkpoints/    # Saved knowledge bases & models
│   ├── logs/           # Training logs
│   └── proofs/         # Saved proofs
├── docs/               # Documentation (see docs/README.md)
│   ├── getting-started/    # Quick start guides
│   ├── phase2/            # Phase 2 documentation
│   ├── phase3/            # Phase 3 documentation & experiment results
│   ├── examples/          # Example theorems
│   └── development/       # Development notes
└── tests/              # Unit tests
    └── phase2/         # Phase 2 (Neural) tests
```

## Documentation

Comprehensive documentation is available in the [`docs/`](docs/) directory:

- **[Getting Started](docs/getting-started/)** - Quick start guides and tutorials
- **[Phase 2](docs/phase2/)** - Neural generator documentation
- **[Phase 3](docs/phase3/)** - RL-based prover documentation & experiment results
- **[Examples](docs/examples/)** - Real theorem examples
- **[Development](docs/development/)** - Technical notes

**Quick Links:**
- 📖 [Documentation Index](docs/README.md)
- 🚀 [Quick Start Guide](docs/getting-started/quickstart.md)
- 🧠 [Phase 2 Quick Start](docs/getting-started/quickstart-phase2.md)
- 📊 [Phase 2 Data Documentation](docs/phase2/data-documentation.md)
- 🤖 [Phase 3 Experiment Results](docs/phase3/experiment-results-phase3-a100-fixed.md)
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

Phase 2 & 3 neural components:
- **Tokenizer**: Converts logical expressions to token sequences
- **Transformer Generator**: Decoder-only transformer for autoregressive generation
- **Generator Trainer**: Supervised and reinforcement learning for the generator
- **Curriculum Scheduler**: Manages progressive difficulty increase
- **Actor-Critic (Phase 3)**: Transformer encoder + policy/value heads for tactic selection
- **Advanced Curriculum (Phase 5)**: Self-paced and adaptive-band strategies with per-bucket EMA tracking
- **Prioritized Experience Buffer (Phase 5)**: Weighted replay buffer targeting low-success-rate examples

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

## Current Phase: Phase 5 (Optimization) ✅

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

**Phase 3 (RL-Based Prover)** - Complete:
- ✅ Actor-critic policy network (transformer encoder + policy/value heads)
- ✅ PPO training for tactic selection
- ✅ Behavioral cloning warmup from heuristic prover
- ✅ Heuristic fallback for hard conjectures
- ✅ Experience replay from successful proofs
- ✅ Joint training with neural generator

See **[Phase 3 Experiment Results](docs/phase3/experiment-results-phase3-a100-fixed.md)** for the full A100 training run (20 epochs, ~60 min, +35 new theorems).

**Phase 4 (Full Monitoring)** - Complete:
- ✅ Interactive Streamlit web dashboard (`scripts/dashboard.py`)
- ✅ Training curve visualisation (KB growth, success rates, PPO metrics)
- ✅ Proof tree visualisation (interactive Plotly tree per theorem)
- ✅ Knowledge base browser (filterable table, complexity & proof-length histograms, discovery timeline)
- ✅ Standalone matplotlib exports for reports (`save_training_curves`, `save_kb_analysis`)
- ✅ `src/monitoring/visualizer.py` – reusable visualisation library

### Quick Start – Phase 4 Dashboard

```bash
python3 scripts/dashboard.py
# Opens http://localhost:8501
```

Options:
```bash
python3 scripts/dashboard.py --port 8888 --checkpoint-dir /path/to/checkpoints
```

The dashboard has four tabs:
| Tab | Contents |
|-----|----------|
| **Overview** | Top-level KPIs, KB growth sparkline, recent theorems |
| **Training Curves** | Interactive KB-growth, success-rate, PPO-loss charts |
| **Proof Explorer** | Select any proved theorem and view its interactive proof tree |
| **KB Browser** | Filterable theorem table, complexity / proof-length histograms, discovery timeline |

### Quick Start - Phase 2

Train with neural generator:
```bash
python3 scripts/train_neural.py --device cuda  # Use CPU if no GPU
```

### Quick Start - Phase 3

Train with RL-based prover (requires GPU recommended):
```bash
python3 scripts/train_phase3.py --device cuda
```

**Phase 5 (Optimization)** - Complete:
- ✅ Parallel heuristic proving via `ProcessPoolExecutor` (`src/training/parallel_prover.py`)
- ✅ Advanced curriculum strategies: self-paced and adaptive-band (`src/models/advanced_curriculum.py`)
- ✅ Prioritized experience buffer for targeted generator training
- ✅ Optional `torch.compile()` for neural models (PyTorch >= 2.0)
- ✅ Throughput / timing instrumentation

### Quick Start – Phase 5

```bash
python3 scripts/train_phase5.py --device cuda
```

Options:
```bash
python3 scripts/train_phase5.py --config configs/phase5_optimization.yaml --device cuda
```

Key tuning knobs in `configs/phase5_optimization.yaml`:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `parallel_prover.workers` | 4 | Parallel proof workers (set 1 to disable) |
| `curriculum.strategy` | `self_paced` | `self_paced` / `adaptive_band` / `linear` |
| `experience_buffer.use_prioritized` | `true` | Prioritized replay for generator |
| `optimization.use_torch_compile` | `false` | Enable `torch.compile` (PyTorch ≥ 2.0) |

## Technical Details

### Computational Requirements

**Phase 1 (CPU only)**:
- CPU: Multi-core recommended (4-8 cores)
- RAM: 8-16GB
- Storage: 10-20GB for checkpoints
- Training time: Initial results in 2-4 hours

**Phase 2 & 3 (Neural + RL)**:
- GPU: NVIDIA GPU with 8GB+ VRAM recommended (Phase 3 run used A100-SXM4-40GB)
- RAM: 16-32GB
- Storage: 50-100GB
- Training time: Phase 3 — 20 epochs in ~60 min on A100; longer on consumer GPUs

**Phase 5 (Optimization)**:
- GPU: NVIDIA GPU with 8GB+ VRAM recommended; A100 ideal
- RAM: 16-32GB (parallel workers increase CPU memory usage)
- Storage: 50-100GB
- Training time: Faster than Phase 3 with parallel proving + `torch.compile`; scales with `parallel_prover.workers`

### Dependencies

Core:
- Python 3.9+
- PyYAML (configuration)
- pytest (testing)
- PyTorch (neural networks — Phase 2 & 3)

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
