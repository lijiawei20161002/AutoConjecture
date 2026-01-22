# Phase 2 Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Prerequisites

```bash
cd /mnt/nw/home/j.li/AutoConjecture
pip install -r requirements.txt
```

### Run Phase 2 Training

```bash
# CPU training (works anywhere)
python scripts/train_neural.py

# GPU training (if CUDA available)
python scripts/train_neural.py --device cuda
```

That's it! The system will:
1. Initialize a transformer model
2. Train with curriculum learning
3. Generate and prove theorems
4. Save checkpoints automatically

## 📊 What to Expect

### Console Output

```
Starting Phase 2 neural training...
Device: cpu
Model parameters: 12,345,678

Phase 1: Supervised Pretraining
================================
Training on 150 proven theorems
Epoch 1/5, Loss: 3.2145

Phase 2: Curriculum Learning
=============================
Epoch 1/10
Cycle 0/500: KB size = 150, Success rate = 8.2%
✓ Proved: (0 + x) = x
  Proof length: 4 steps
  Complexity: 5

Curriculum advanced to stage 1
  Complexity range: (4, 6)
  Temperature: 1.35
```

### Performance

| Metric | Value |
|--------|-------|
| Initial success rate | ~5% (similar to random) |
| After pretraining | ~10-15% |
| After curriculum | ~20-30% |
| Training time (CPU) | 3-5 hours |
| Training time (GPU) | 45-90 minutes |

## 🎛️ Common Commands

### Basic Usage

```bash
# Default settings (good for testing)
python scripts/train_neural.py

# More epochs for better results
python scripts/train_neural.py --epochs 30

# More conjectures per cycle
python scripts/train_neural.py --conjectures-per-cycle 20

# Named experiment
python scripts/train_neural.py --experiment-name my_first_run
```

### Model Configuration

```bash
# Smaller model (faster, less accurate)
python scripts/train_neural.py --d-model 128 --nhead 4 --num-layers 4

# Larger model (slower, more accurate)
python scripts/train_neural.py --d-model 512 --nhead 16 --num-layers 8

# More pretraining
python scripts/train_neural.py --pretrain-epochs 10
```

### Curriculum Control

```bash
# Disable curriculum (flat difficulty)
python scripts/train_neural.py --no-curriculum

# Adjust complexity range
python scripts/train_neural.py --initial-complexity 3 --final-complexity 20

# Faster progression
python scripts/train_neural.py --cycles 200
```

### Resume Training

```bash
# Resume from checkpoint
python scripts/train_neural.py \
  --resume data/checkpoints/generator_epoch_5_cycle_0.pt
```

## 📈 Monitoring Progress

### Checkpoints

Saved every 1000 cycles (configurable):
```
data/checkpoints/
├── neural_epoch_0_cycle_1000.json      # Knowledge base
├── generator_epoch_0_cycle_1000.pt     # Generator model
└── trainer_epoch_0_cycle_1000.pt       # Training state
```

### Logs

```
data/logs/neural_training.log
```

## 🔧 Troubleshooting

### CUDA Out of Memory

```bash
# Reduce model size
python scripts/train_neural.py \
  --device cuda \
  --d-model 128 \
  --nhead 4 \
  --generator-batch-size 16
```

### Slow on CPU

```bash
# Reduce cycles
python scripts/train_neural.py \
  --cycles 100 \
  --conjectures-per-cycle 5
```

### Low Success Rate

```bash
# More pretraining, slower curriculum
python scripts/train_neural.py \
  --pretrain-epochs 10 \
  --initial-complexity 2 \
  --cycles 500
```

## 📝 Configuration File

Create `my_config.yaml`:

```yaml
training:
  num_epochs: 20
  cycles_per_epoch: 500
  conjectures_per_cycle: 15

model:
  d_model: 256
  nhead: 8
  num_layers: 6

curriculum:
  enabled: true
  initial_complexity: 2
  final_complexity: 15
```

Use it:
```bash
python scripts/train_neural.py --config my_config.yaml
```

## 🧪 Testing

```bash
# Run all Phase 2 tests
pytest tests/test_neural_components.py -v

# Quick smoke test
pytest tests/test_neural_components.py::test_phase2_imports -v
```

## 📚 Full Documentation

- **Complete Guide**: [PHASE2_NEURAL.md](PHASE2_NEURAL.md)
- **Implementation Details**: [PHASE2_SUMMARY.md](PHASE2_SUMMARY.md)
- **Main README**: [README.md](README.md)

## 💡 Tips

### For Best Results

1. **Start with pretraining**: Use `--pretrain-epochs 10` if you have existing proofs
2. **Use curriculum**: Leave it enabled (default)
3. **GPU if available**: Speeds up training 5-10x
4. **Monitor success rate**: Should increase from ~5% to ~20-30%
5. **Be patient**: Good results take 2-4 hours on CPU

### For Quick Testing

1. **Reduce cycles**: `--cycles 100`
2. **Fewer conjectures**: `--conjectures-per-cycle 5`
3. **Smaller model**: `--d-model 128 --nhead 4`
4. **Short run**: `--epochs 3`

### For Production

1. **More epochs**: `--epochs 30-50`
2. **More cycles**: `--cycles 1000-2000`
3. **Larger model**: `--d-model 512 --nhead 16`
4. **GPU**: `--device cuda`

## 🎯 Success Criteria

You'll know Phase 2 is working when:

✅ Model generates valid expressions (not all random noise)
✅ Success rate increases over time
✅ Curriculum advances to higher stages
✅ Knowledge base grows faster than Phase 1
✅ Proven theorems become more complex

## 🆚 Phase 1 vs Phase 2

| Feature | Phase 1 | Phase 2 |
|---------|---------|---------|
| **Script** | `train.py` | `train_neural.py` |
| **Generator** | Random | Neural (Transformer) |
| **Learning** | None | Yes (from proofs) |
| **Curriculum** | No | Yes (adaptive) |
| **Success Rate** | 2-5% | 10-30% |
| **Speed** | Fast | Medium |
| **Setup** | Simple | Medium |

## 🔜 What's Next?

After Phase 2 training:

1. **Analyze Results**: Look at proven theorems in knowledge base
2. **Compare**: Run Phase 1 and Phase 2, compare KB growth
3. **Tune**: Adjust hyperparameters based on results
4. **Scale**: Try larger models, more training
5. **Phase 3**: Implement RL-based prover

## ❓ FAQ

**Q: How long does it take?**
A: 3-5 hours on CPU, 45-90 min on GPU (for 10 epochs)

**Q: Do I need GPU?**
A: No, but it's 5-10x faster

**Q: Can I stop and resume?**
A: Yes, use `--resume` with checkpoint path

**Q: How do I know it's working?**
A: Watch for increasing success rate and curriculum progression

**Q: What if success rate doesn't improve?**
A: Increase `--pretrain-epochs` or decrease `--initial-complexity`

## 🆘 Getting Help

1. Check logs: `data/logs/neural_training.log`
2. Read full docs: `PHASE2_NEURAL.md`
3. Run tests: `pytest tests/test_neural_components.py -v`
4. Check implementation: `PHASE2_SUMMARY.md`

---

**Ready to start?**

```bash
python scripts/train_neural.py --device cuda --epochs 20
```

**Happy theorem proving! 🎓**
