"""
Test script for full neural training loop including conjecture generation and proving cycles.
"""

import torch
from src.training.neural_training_loop import NeuralTrainingLoop, NeuralTrainingConfig

print("=" * 60)
print("Testing Full Neural Training Loop with Cycles")
print("=" * 60)

# Create config with parameters suitable for full testing
config = NeuralTrainingConfig(
    num_epochs=2,              # 2 full epochs
    cycles_per_epoch=5,        # 5 cycles per epoch
    conjectures_per_cycle=3,   # 3 conjectures per cycle
    pretrain_epochs=20,        # Quick pretraining
    d_model=128,               # Smaller model
    nhead=4,
    num_layers=3,
    generator_lr=1e-3,
    generator_batch_size=2,
    generator_warmup_steps=10,
    device="cuda" if torch.cuda.is_available() else "cpu",
    log_interval=1,            # Log every cycle
    checkpoint_interval=999999 # Don't checkpoint
)

# Create training loop
loop = NeuralTrainingLoop(config=config)

print(f"\nDevice: {config.device}")
print(f"Vocab size: {loop.tokenizer.vocab_size}")
print(f"Initial KB size: {loop.knowledge_base.size()}")

# Run training
print("\n" + "=" * 60)
print("Starting Training")
print("=" * 60)

try:
    loop.train()
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print(f"Final KB size: {loop.knowledge_base.size()}")

    # Generate some final conjectures
    print("\n" + "=" * 60)
    print("Generating final conjectures:")
    print("=" * 60)
    conjectures = loop.generator.generate(n=5)
    for i, conj in enumerate(conjectures, 1):
        print(f"{i}. {conj}")

except Exception as e:
    print(f"\nError during training: {e}")
    import traceback
    traceback.print_exc()
