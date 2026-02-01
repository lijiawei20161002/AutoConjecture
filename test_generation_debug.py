"""
Debug script to understand what the generator is producing and why it's being filtered.
"""

import torch
from src.training.neural_training_loop import NeuralTrainingLoop, NeuralTrainingConfig
from src.generation.heuristics import ComplexityEstimator
from src.generation.novelty import NoveltyScorer

print("=" * 60)
print("Testing Generation and Filtering")
print("=" * 60)

# Create config
config = NeuralTrainingConfig(
    pretrain_epochs=20,
    d_model=128,
    nhead=4,
    num_layers=3,
    generator_lr=1e-3,
    generator_batch_size=2,
    generator_warmup_steps=10,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# Create training loop
loop = NeuralTrainingLoop(config=config)

# Pretrain the model
print("\nPretraining model...")
loop._pretrain()
print("Pretraining complete\n")

# Generate conjectures
print("=" * 60)
print("Generating conjectures:")
print("=" * 60)

conjectures = loop.generator.generate(n=10)
print(f"\nGenerated {len(conjectures)} conjectures (including None values):\n")

for i, conj in enumerate(conjectures, 1):
    print(f"{i}. {conj}")
    if conj is not None:
        # Check well-formed
        is_well_formed = loop.complexity_estimator.is_well_formed(conj)
        print(f"   Well-formed: {is_well_formed}")

        # Check novelty
        novelty = loop.novelty_scorer.score(conj)
        print(f"   Novelty: {novelty:.3f} {'✓' if novelty >= 0.3 else '✗ (too low)'}")

        # Check already proven
        is_proven = loop.knowledge_base.contains(conj)
        print(f"   Already proven: {is_proven}")

        # Check complexity
        complexity = loop.complexity_estimator.estimate(conj)
        print(f"   Complexity: {complexity:.2f}")

        print()

# Now check what would pass the full filter
print("=" * 60)
print("Conjectures that would pass all filters:")
print("=" * 60)

valid_count = 0
for conj in conjectures:
    if conj is None:
        continue

    if not loop.complexity_estimator.is_well_formed(conj):
        continue

    novelty = loop.novelty_scorer.score(conj)
    if novelty < 0.3:
        continue

    if not loop.diversity_filter.should_keep(conj):
        continue

    if loop.knowledge_base.contains(conj):
        continue

    valid_count += 1
    print(f"{valid_count}. {conj}")

if valid_count == 0:
    print("None! All conjectures were filtered out.")

print(f"\nTotal valid conjectures: {valid_count} out of {len([c for c in conjectures if c is not None])}")
