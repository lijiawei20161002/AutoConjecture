"""
Test with improved settings for better generation quality.
"""

import torch
from src.training.neural_training_loop import NeuralTrainingLoop, NeuralTrainingConfig

print("=" * 70)
print("Testing Improved Neural Training")
print("=" * 70)

# Create config with parameters for better quality
config = NeuralTrainingConfig(
    num_epochs=2,
    cycles_per_epoch=5,
    conjectures_per_cycle=5,     # Generate more to get some valid ones
    pretrain_epochs=50,           # More pretraining - was 20
    initial_complexity=5,         # Realistic minimum - even simple expressions are ~7
    final_complexity=25,          # Allow medium complexity conjectures
    success_threshold=0.3,
    d_model=256,                   # Larger model - was 128
    nhead=8,                       # More heads - was 4
    num_layers=4,                  # More layers - was 3
    generator_lr=5e-4,             # Lower LR for stability
    generator_batch_size=4,        # Larger batches
    generator_warmup_steps=20,
    device="cuda" if torch.cuda.is_available() else "cpu",
    log_interval=1,
    checkpoint_interval=999999
)

print(f"\nConfiguration:")
print(f"  Device: {config.device}")
print(f"  Model: d_model={config.d_model}, layers={config.num_layers}, heads={config.nhead}")
print(f"  Pretraining: {config.pretrain_epochs} epochs")
print(f"  Complexity range: {config.initial_complexity} -> {config.final_complexity}")

# Create training loop
loop = NeuralTrainingLoop(config=config)

print(f"\nInitialized:")
print(f"  Vocab size: {loop.tokenizer.vocab_size}")
print(f"  KB theorems: {len(loop.knowledge_base.theorems)}")

# Run training
print("\n" + "=" * 70)
print("Starting Training")
print("=" * 70)

try:
    loop.train()

    print("\n" + "=" * 70)
    print("Training completed successfully!")
    print("=" * 70)

    # Try generation with different temperatures
    print("\n" + "=" * 70)
    print("Generating conjectures with different temperatures:")
    print("=" * 70)

    for temp in [0.3, 0.5, 0.8]:
        print(f"\nTemperature = {temp}:")
        loop.generator.set_temperature(temp)
        conjectures = loop.generator.generate(n=5)

        valid_count = sum(1 for c in conjectures if c is not None)
        print(f"  Valid: {valid_count}/5")

        for i, conj in enumerate(conjectures, 1):
            if conj is not None:
                print(f"  {i}. {conj}")
            else:
                print(f"  {i}. [Invalid]")

except Exception as e:
    print(f"\nError during training: {e}")
    import traceback
    traceback.print_exc()
