#!/usr/bin/env python3
"""
Display proven theorems from AutoConjecture in a human-readable format.

Usage:
    python scripts/display_proofs.py [checkpoint_file]

If no checkpoint file is specified, loads the most recent checkpoint.
"""
import json
import sys
from pathlib import Path
from datetime import datetime


def load_checkpoint(checkpoint_path=None):
    """Load a checkpoint file."""
    if checkpoint_path:
        path = Path(checkpoint_path)
    else:
        # Find most recent checkpoint
        checkpoint_dir = Path("data/checkpoints")
        if not checkpoint_dir.exists():
            print("No checkpoints found. Run training first.")
            sys.exit(1)

        checkpoints = list(checkpoint_dir.glob("*.json"))
        if not checkpoints:
            print("No checkpoint files found.")
            sys.exit(1)

        # Get most recent by modification time
        path = max(checkpoints, key=lambda p: p.stat().st_mtime)

    with open(path, 'r') as f:
        return json.load(f), path


def format_proof_steps(steps):
    """Format proof steps for display."""
    if not steps:
        return "  [No steps recorded]"
    return "\n".join(f"  {step}" for step in steps)


def display_theorem(theorem, index):
    """Display a single theorem with formatting."""
    print(f"\n{'='*80}")
    print(f"THEOREM {index}")
    print(f"{'='*80}")
    print(f"\nStatement: {theorem['statement']}")
    print(f"\nMetadata:")
    print(f"  Complexity: {theorem['complexity']:.1f}")
    print(f"  Proof Length: {theorem['proof_length']} steps")
    print(f"  Discovered: Epoch {theorem['epoch']}, Cycle {theorem['cycle']}")
    print(f"  Timestamp: {theorem['timestamp']}")
    print(f"\nProof:")
    print(format_proof_steps(theorem['proof_steps']))


def display_statistics(data):
    """Display summary statistics."""
    theorems = data['theorems']
    metadata = data['metadata']

    print(f"\n{'='*80}")
    print("KNOWLEDGE BASE STATISTICS")
    print(f"{'='*80}")
    print(f"\nTotal Theorems: {metadata['num_theorems']}")
    print(f"Axioms: {len(data['axioms'])}")
    print(f"Total Statements: {metadata['num_theorems'] + len(data['axioms'])}")
    print(f"Last Updated: {metadata['saved_at']}")

    if theorems:
        complexities = [t['complexity'] for t in theorems]
        proof_lengths = [t['proof_length'] for t in theorems]

        print(f"\nComplexity Range:")
        print(f"  Min: {min(complexities):.1f}")
        print(f"  Max: {max(complexities):.1f}")
        print(f"  Avg: {sum(complexities)/len(complexities):.2f}")

        print(f"\nProof Length Range:")
        print(f"  Min: {min(proof_lengths)}")
        print(f"  Max: {max(proof_lengths)}")
        print(f"  Avg: {sum(proof_lengths)/len(proof_lengths):.2f}")


def display_axioms(axioms):
    """Display the axioms."""
    print(f"\n{'='*80}")
    print("PEANO AXIOMS")
    print(f"{'='*80}")
    for i, axiom in enumerate(axioms, 1):
        # Parse the axiom string (format: "('name', statement)")
        parts = axiom.strip("()").split(", ", 1)
        if len(parts) == 2:
            name = parts[0].strip("'")
            statement = parts[1]
            print(f"\n{i}. {name}")
            print(f"   {statement}")
        else:
            print(f"\n{i}. {axiom}")


def main():
    """Main function."""
    print("="*80)
    print("AutoConjecture - Proven Theorems Display")
    print("="*80)

    # Load checkpoint
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else None
    data, path = load_checkpoint(checkpoint_path)

    print(f"\nLoaded checkpoint: {path.name}")

    # Display axioms
    display_axioms(data['axioms'])

    # Display statistics
    display_statistics(data)

    # Display each theorem
    theorems = data['theorems']
    if theorems:
        print(f"\n{'='*80}")
        print("DISCOVERED THEOREMS")
        print(f"{'='*80}")

        for i, theorem in enumerate(theorems, 1):
            display_theorem(theorem, i)
    else:
        print("\nNo theorems proven yet.")

    print(f"\n{'='*80}")
    print("END OF REPORT")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
