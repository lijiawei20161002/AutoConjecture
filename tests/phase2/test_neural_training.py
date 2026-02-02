#!/usr/bin/env python3
"""Test neural training loop with axiom pretraining"""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.training.neural_training_loop import NeuralTrainingLoop, NeuralTrainingConfig

# Create a minimal test configuration
config = NeuralTrainingConfig(
    num_epochs=1,              # Just 1 epoch for testing
    cycles_per_epoch=10,       # Only 10 cycles
    conjectures_per_cycle=5,   # 5 conjectures per cycle
    pretrain_epochs=50,        # More epochs for small dataset
    d_model=128,               # Smaller model for faster testing
    nhead=4,
    num_layers=3,
    generator_lr=1e-3,         # Higher learning rate
    generator_batch_size=2,    # Smaller batch = more gradient steps
    generator_warmup_steps=10, # Small warmup for small dataset
    device="cuda" if torch.cuda.is_available() else "cpu",
    log_interval=1,            # Log every cycle
    checkpoint_interval=999999 # Don't checkpoint during test
)

print("="*60)
print("Testing Neural Training Loop")
print("="*60)
print(f"Device: {config.device}")
print(f"Pretrain epochs: {config.pretrain_epochs}")
print(f"Main epochs: {config.num_epochs}")
print(f"Cycles per epoch: {config.cycles_per_epoch}")
print()

# Create training loop
trainer = NeuralTrainingLoop(config=config)

print("Initial knowledge base state:")
print(f"  Axioms: {len(trainer.knowledge_base.axioms)}")
print(f"  Theorems: {trainer.knowledge_base.size()}")
print(f"  Total: {trainer.knowledge_base.total_size()}")
print()

# Check if axioms are present
if trainer.knowledge_base.axioms:
    print("Sample axioms:")
    for i, axiom in enumerate(trainer.knowledge_base.axioms[:3]):
        print(f"  {i+1}. {axiom}")
        print(f"      Complexity: {axiom.complexity()}")
    print()

# Run just the pretraining phase to test it
print("="*60)
print("Running pretraining phase...")
print("="*60)
if trainer.knowledge_base.total_size() > 0 and config.pretrain_epochs > 0:
    trainer._pretrain()
else:
    print("Skipped (no data available)")
print()

# Test generation after pretraining
print("="*60)
print("Testing generation after pretraining...")
print("="*60)
trainer.generator.eval_mode()
conjectures = [c for c in trainer.generator.generate(10) if c is not None]

print(f"Generated {len(conjectures)} valid conjectures:")
for i, conj in enumerate(conjectures[:5]):
    print(f"  {i+1}. {conj}")
print()

if len(conjectures) == 0:
    print("WARNING: No valid conjectures generated!")
    print("Checking raw token generation...")

    # Generate raw tokens to see what's happening
    with torch.no_grad():
        token_seqs = trainer.model.generate(
            batch_size=3,
            max_length=32,
            temperature=1.0,
            top_k=50,
            top_p=None,
            sos_token_id=trainer.tokenizer.sos_id,
            eos_token_id=trainer.tokenizer.eos_id,
            pad_token_id=trainer.tokenizer.pad_id,
            device=config.device
        )

    for i, seq in enumerate(token_seqs):
        tokens_list = seq.cpu().tolist()
        print(f"\nSequence {i+1}:")
        print(f"  Token IDs: {tokens_list[:15]}")
        print(f"  Tokens: {[trainer.tokenizer.id_to_token.get(t, '?') for t in tokens_list[:15]]}")
else:
    print("SUCCESS: Valid conjectures generated after pretraining!")

print("\nDone!")
